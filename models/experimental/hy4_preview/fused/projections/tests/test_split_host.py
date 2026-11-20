# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The split of layout.py against the references, in torch (no device): the 32 chips' shares, gathered and summed
the way the cross-chip steps do it, are reference/attention.py's latent, queries, gate and output and
reference/attention_decode.py's absorbed queries. Log lines start with HY4_MK7."""

import torch
import torch.nn.functional as F
from loguru import logger

from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections.tests.harness import (
    POSITIONS,
    check,
    random_input,
    random_reference,
    tables,
)
from models.experimental.hy4_preview.reference.attention_decode import Hy4AttentionDecode
from models.experimental.hy4_preview.tt import frames


def test_bytes_and_payloads():
    for full in (False, True):
        tiles = [sum(L.tiles_read(c, full).values()) for c in range(L.CHIPS)]
        logger.info(
            f"HY4_MK7 split {'full' if full else 'shared'} layer: tiles per chip min {min(tiles)} max {max(tiles)}; "
            f"BFP8 {max(tiles) * 1088 / 1e6:.2f} MB, BFP4 {max(tiles) * 576 / 1e6:.2f} MB, two BFP4 terms "
            f"{max(tiles) * 1152 / 1e6:.2f} MB per chip; chip 0 by projection {L.tiles_read(0, full)}"
        )
        logger.info(f"HY4_MK7 split {'full' if full else 'shared'} layer payloads (sent, landing): {L.payloads(full)}")
    # every weight is read once over the 32 chips
    shared = sum(sum(L.tiles_read(c, False).values()) for c in range(L.CHIPS))
    whole = (6144 * (2048 + 576 + 16384) + 2048 * 16384 + 64 * 448 * 512 + 16384 * 6144) // 1024
    assert shared == whole, (shared, whole)


def test_split_is_the_reference():
    ref, gamma, indexer = random_reference(full=True)
    y, scale = random_input()
    x = (y.double() * scale.double()[:, None] * gamma.double()).float()  # what the reference takes
    cos, sin, d_cos, d_sin = tables()
    lw = L.layer_weights(ref.weights(), gamma, indexer.weights())
    shares = [L.chip_weights(lw, c) for c in range(L.CHIPS)]
    pre = [L.pre_golden(y, scale, w, d_cos, d_sin) for w in shares]

    with torch.no_grad():
        r_q_a = F.linear(x, ref.q_a_proj)
        r_q_resid, r_k_pass, r_k_rope = ref.latent(x, cos, sin)
        r_gate = ref.gate(x)
        r_kv = F.linear(x, ref.kv_a_proj_with_mqa)

    # Xa: the gather of the chips' q_a tiles, lane sums and pool tiles
    q_a = torch.cat([p["q_a"] for p in pre], dim=1)
    lanes = sum(p["lanes"] for p in pre)
    check("gathered q_a", r_q_a, q_a, 0.99999)
    s = L.norm_scalar(lanes)
    check("q_a norm from the lane sums", r_q_resid, q_a * s[:, None] * ref.q_a_layernorm, 0.99999)
    kv = torch.cat([pre[c]["pool"] for c in range(18)], dim=1)
    check("gathered raw latent row (device frame)", frames.tail_to_device(r_kv, 64), kv, 0.99999)
    key = torch.cat([pre[c]["pool"] for c in range(18, 22)], dim=1)
    r_key = F.linear(x, indexer.wk)[:, frames.index_key_perm(128, 64)]
    check("gathered raw index key (device frame)", r_key, key, 0.99999)
    check("head weights", indexer.head_weights(x), pre[22]["pool"], 0.99999)
    check("gate (every chip its heads)", r_gate, torch.cat([p["gate"] for p in pre], dim=1), 0.99999)

    # the index queries from x through the fused weight: the reference's without q_a's norm scalar
    idx = torch.cat([p["idx_q"] for p in pre], dim=1).reshape(L.ROWS, 32, 128)
    r_idx = F.linear(r_q_resid, indexer.wq_b).reshape(L.ROWS, 32, 128)[:, :, frames.index_key_perm(128, 64)]
    check("index queries before RoPE, times the norm scalar", r_idx, idx * s[:, None, None], 0.99999)

    # ... and in the cache's frame (RoPE, Hadamard basis): frames.index_to_device of the reference's queries
    idx_dev = torch.cat([p["idx_q_dev"] for p in pre], dim=1).reshape(L.ROWS, 32, 128)
    with torch.no_grad():
        r_idx_dev = frames.index_to_device(indexer.queries(r_q_resid, cos, sin), 64)
    check("index queries in the cache's frame, times the norm scalar", r_idx_dev, idx_dev * s[:, None, None], 0.99999)

    # X3: the absorbed queries of every chip's heads, as the pages the sparse attention reads
    dec = Hy4AttentionDecode(ref)
    q_lat = [L.query_golden(q_a, lanes, d_cos, d_sin, w)[0] for w in shares]
    pages = L.sparse_mla_pages(q_lat)
    for u in range(L.ROWS):
        token = dec.new_token(x[u : u + 1], int(POSITIONS[u]))
        r = frames.tail_to_device(dec.latent_queries(token), 64)  # [64, 576]
        check(f"absorbed queries of row {u}", r, pages[u].reshape(64, 576), 0.99999 if u == 0 else None)
        assert torch.allclose(r, pages[u].reshape(64, 576), atol=2e-3 * r.abs().max().item())

    # X4 leaves ctx of a chip's heads on the chip; X5 sums the partials
    g = torch.Generator().manual_seed(5)
    ctx = torch.randn(L.HEADS, L.ROWS, L.KV_LORA, generator=g)
    gate = torch.cat([p["gate"] for p in pre], dim=1)
    out = sum(
        L.post_golden(ctx[2 * c : 2 * c + 2], gate[:, 512 * c : 512 * c + 512], shares[c])[0] for c in range(L.CHIPS)
    )
    _, w_v = dec.absorb_weights()
    with torch.no_grad():
        attn = torch.einsum("hvk,hrk->rhv", w_v, ctx).reshape(L.ROWS, -1)
        r_out = F.linear(attn * r_gate, ref.o_proj)
    check("sum of the 32 partials", r_out, out, 0.99999)
