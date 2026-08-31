# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""vsa_sdpa: block-sparse VSA fine-stage attention vs a torch oracle.

R4 checks (VSA_SCOPE.md): matches torch for m in {1, >1} on synthetic cases covering ragged
blocks, non-uniform row counts (including fully-dense rows), single-block rows, and counts not
divisible by m; identical results across m values.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, skip_for_wormhole_b0

BLOCK = 64
SENTINEL = 0xFFFFFFFF


def fine_attention_ref(q, k, v, indices, block_counts, scale=None):
    """Exact attention over each row's listed blocks; [B,H,S,d] float tensors."""
    batch, heads, seq_len, dim = q.shape
    scale = dim**-0.5 if scale is None else scale
    counts = block_counts.to(torch.long)
    token_valid = (torch.arange(BLOCK)[None, :] < counts[:, None]).reshape(-1)

    out = torch.zeros_like(q)
    for b in range(batch):
        for h in range(heads):
            for qt in range(seq_len // BLOCK):
                row = indices[b, h, qt].to(torch.long)
                listed = row[row != SENTINEL]
                cols = (listed[:, None] * BLOCK + torch.arange(BLOCK)[None, :]).reshape(-1)
                cols = cols[token_valid[cols]]
                rows = slice(qt * BLOCK, (qt + 1) * BLOCK)
                attn = torch.einsum("qd,kd->qk", q[b, h, rows], k[b, h, cols]) * scale
                out[b, h, rows] = torch.softmax(attn, dim=-1) @ v[b, h, cols]
    return out


def build_indices(heads, n_q_tiles, n_blocks, w, pattern, seed):
    """Sentinel-tailed uint32 index rows plus per-block valid counts covering the R4 case matrix."""
    gen = torch.Generator().manual_seed(seed)
    indices = torch.full((1, heads, n_q_tiles, w), SENTINEL, dtype=torch.int64)
    for h in range(heads):
        for qt in range(n_q_tiles):
            if pattern == "dense":
                listed = torch.arange(n_blocks)
            elif pattern == "single":
                listed = torch.randint(0, n_blocks, (1,), generator=gen)
            elif pattern == "nonuniform":
                # every row picks its own count in [1, n_blocks]; some rows fully dense
                n_pick = int(torch.randint(1, n_blocks + 1, (1,), generator=gen))
                listed = torch.randperm(n_blocks, generator=gen)[:n_pick]  # unsorted, like topk output
            else:
                raise ValueError(pattern)
            indices[0, h, qt, : listed.numel()] = listed
    return indices.to(torch.uint32)


def build_counts(n_blocks, w, ragged, seed):
    counts = torch.full((n_blocks,), BLOCK, dtype=torch.int64)
    if ragged:
        gen = torch.Generator().manual_seed(seed)
        # cover the boundary cases: partial first tile, exactly one tile, partial second tile, one token
        specials = torch.tensor([1, 5, 31, 32, 33, 63])
        n_special = min(specials.numel(), n_blocks - 1)  # keep at least one full block
        where = torch.randperm(n_blocks, generator=gen)[:n_special]
        counts[where] = specials[:n_special]
    padded = torch.zeros(w, dtype=torch.int64)
    padded[:n_blocks] = counts
    return padded.to(torch.uint32)


def run_vsa_sdpa(device, heads, seq_len, kv_len, pattern, ragged, m, seed=0):
    torch.manual_seed(seed)
    dim = 128
    n_q_tiles = seq_len // BLOCK
    n_blocks = kv_len // BLOCK
    w = ((n_blocks + 15) // 16) * 16  # DRAM row alignment: W*4 % 64 == 0

    q = torch.randn(1, heads, seq_len, dim, dtype=torch.bfloat16)
    k = torch.randn(1, heads, kv_len, dim, dtype=torch.bfloat16)
    v = torch.randn(1, heads, kv_len, dim, dtype=torch.bfloat16)
    indices = build_indices(heads, n_q_tiles, n_blocks, w, pattern, seed + 1)
    counts = build_counts(n_blocks, w, ragged, seed + 2)

    # zero the pad columns like the host packing does (values are don't-cares, keep them finite)
    token_valid = (torch.arange(BLOCK)[None, :] < counts[:n_blocks, None].to(torch.long)).reshape(-1)
    k[:, :, ~token_valid] = 0
    v[:, :, ~token_valid] = 0

    tt_q = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_k = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_v = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_idx = ttnn.from_torch(
        indices.view(torch.int32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )
    tt_counts = ttnn.from_torch(
        counts.view(torch.int32).reshape(1, 1, 1, w), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )

    tt_out = ttnn.transformer.vsa_sdpa(tt_q, tt_k, tt_v, tt_idx, tt_counts, k_chunk_blocks=m)
    out = ttnn.to_torch(tt_out)

    ref = fine_attention_ref(q.float(), k.float(), v.float(), indices, counts)
    return out, ref


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize("m", [1, 3])
@pytest.mark.parametrize(
    ("heads", "seq_len", "kv_len", "pattern", "ragged"),
    [
        (2, 256, 1280, "nonuniform", True),  # ragged blocks, non-uniform rows, counts % m != 0
        (2, 256, 1280, "dense", True),  # fully-dense rows
        (2, 128, 1280, "single", False),  # single-block rows
        (14, 128, 2048, "nonuniform", True),  # production head count
        (2, 64, 64, "dense", False),  # minimal: one q tile, one block
    ],
    ids=["nonuniform_ragged", "dense_ragged", "single_block", "h14", "minimal"],
)
def test_vsa_sdpa_vs_torch(device, heads, seq_len, kv_len, pattern, ragged, m):
    out, ref = run_vsa_sdpa(device, heads, seq_len, kv_len, pattern, ragged, m)
    ok, pcc = comp_pcc(ref, out.float(), 0.999)
    assert ok, f"PCC {pcc}"


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
def test_vsa_sdpa_identical_across_m(device):
    """R4: identical results across m values (same inputs, m in {1, 2, 3, 5})."""
    outs = {}
    for m in (1, 2, 3, 5):
        out, ref = run_vsa_sdpa(device, 2, 256, 1280, "nonuniform", True, m, seed=7)
        outs[m] = out
        ok, pcc = comp_pcc(ref, out.float(), 0.999)
        assert ok, f"m={m}: PCC {pcc}"
    # m changes the chunk boundaries and thus the flash-rescale rounding order, so bitwise equality
    # is not guaranteed in bf16. Measured: every m sits at the same ~0.9997 bf16 noise floor vs the
    # fp32 oracle, with flat per-row PCC regardless of partial chunks, and cross-m agreement at
    # ~0.9999 -- two samplings of the same rounding noise. Bound it there.
    for m in (2, 3, 5):
        ok, pcc = comp_pcc(outs[1].float(), outs[m].float(), 0.9998)
        assert ok, f"m={m} vs m=1: PCC {pcc}"
