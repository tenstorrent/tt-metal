# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only (PLAN.md section 20, before any kernel of the index-key append exists): what the append's patch of
one key row in a 32-row BFP8 block rests on, checked with ttnn's own packer (host tensors, no device).

  1. A token row of a [32, 32] BFP8 tile shares nothing with another row: packing a block and packing the same
     block with one row replaced give the same stored VALUES in the other 31 rows, and the replaced row's stored
     values are those the row has when it is packed at any other tile row among any other rows (a face row of 16
     values has its own exponent). So a row packed at tile row u can be moved to tile row P % 32 of another block.
  2. The device arithmetic of a key against the reference: LayerNorm, RoPE through tt/frames.py's tables, the
     device's column order and the Hadamard basis (frames.index_to_device), from the stream's own columns.

NOT checked here: the BYTES (where a row's two exponents and 32 mantissas sit in the 1,088 bytes of a tile). ttnn
gives a host no raw view of a BFP8 tile; the layout is read from tt_metal/impl/data_format/blockfloat_common.cpp
(64 exponent bytes, one per face row, faces in order, then 4 x 256 mantissa bytes) and is confirmed on the board by
a raw dump of a block the prefill stack wrote (the first one-chip rung).

    python -m pytest models/experimental/hy4_preview/fused/layer_attn/tests/test_key_row_host.py
"""

import pytest
import torch

import ttnn
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin
from models.experimental.hy4_preview.tt import frames


def stored(block: torch.Tensor) -> torch.Tensor:
    """[32 n, 32 m] values -> what a BFP8 tile tensor of them holds (ttnn's packer and unpacker, on the host)."""
    t = ttnn.from_torch(block[None, None].float(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t)[0, 0].float()


@pytest.mark.parametrize("seed", range(4))
def test_row_is_its_own(seed):
    g = torch.Generator().manual_seed(seed)
    scale = 10.0 ** torch.randint(-3, 3, (32, 1), generator=g).float()  # rows of very different magnitudes
    block = torch.randn(32, 128, generator=g) * scale
    key = torch.randn(128, generator=g) * 3
    before = stored(block)
    for row in (0, 7, 15, 16, 31):
        patched = block.clone()
        patched[row] = key
        after = stored(patched)
        others = [r for r in range(32) if r != row]
        assert torch.equal(after[others], before[others]), f"row {row}: another row's stored values changed"
        # the row as the append packs it: at tile row u among 7 other new keys, the rest of the tile zero
        for u in (0, 3, 7):
            fresh = torch.zeros(32, 128)
            fresh[:8] = torch.randn(8, 128, generator=g) * 100
            fresh[u] = key
            assert torch.equal(stored(fresh)[u], after[row]), f"row {row} from tile row {u}: other stored values"


def test_key_arithmetic():
    """The append's steps in the device's order against Hy4Indexer.keys + frames.index_to_device."""
    cfg = IndexerConfig()
    torch.manual_seed(0)
    ix = Hy4Indexer(cfg)
    for p in ix.parameters():
        torch.nn.init.normal_(p, std=0.05)
    ix.k_norm_weight.data += 1.0
    x = torch.randn(8, cfg.hidden_size)
    positions = torch.tensor([5, 100, 2047, 2048, 6000, 40000, 131071, 1000000])
    cos, sin = rope_cos_sin(positions, AttentionConfig())
    want = frames.index_to_device(ix.keys(x, cos, sin), cfg.qk_rope_head_dim)
    # the device: the raw key in the device's column order (frames.indexer_weights: [rope, interleaved | pass]),
    # LayerNorm with the permuted weight and bias, the rotation of interleaved pairs, then the Hadamard basis
    w = frames.indexer_weights(
        {
            "wk.weight": ix.wk.data,
            "k_norm.weight": ix.k_norm_weight.data,
            "k_norm.bias": ix.k_norm_bias.data,
            "wq_b.weight": ix.wq_b.data,
        },
        cfg,
    )
    raw = x @ w["wk.weight"].t()
    mean = raw.mean(dim=1, keepdim=True)
    var = (raw - mean).pow(2).mean(dim=1, keepdim=True)
    k = (raw - mean) * torch.rsqrt(var + cfg.norm_eps) * w["k_norm.weight"] + w["k_norm.bias"]
    d_cos, d_sin = frames.rope_tables(cos, sin)
    rope = k[:, : cfg.qk_rope_head_dim]
    swapped = torch.stack((-rope[:, 1::2], rope[:, 0::2]), dim=-1).reshape(rope.shape)  # (a, b) -> (-b, a)
    k = torch.cat((rope * d_cos + swapped * d_sin, k[:, cfg.qk_rope_head_dim :]), dim=1)
    got = k @ frames.hadamard(cfg.head_dim)
    err = float((got - want).abs().max() / want.abs().max())
    assert err < 1e-5, f"the device's order of operations gives another key: relative error {err:.3e}"
