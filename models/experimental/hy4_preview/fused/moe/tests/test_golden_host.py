# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the goldens of fused/moe/layout.py are reference/moe.py's functions. With a small MoE the sum of the
chips' partial outputs (tensor parallel and expert parallel) is Hy4MoE.routed, the plain slot is the shared expert,
route_golden is Hy4MoE.route, and the weight streams hold the tiles the kernels expect."""

import torch

from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.reference.moe import Hy4MoE, MoEConfig


def _ref(seed=0):
    cfg = MoEConfig(hidden_size=96, num_experts=16, top_k=4, intermediate_size=64)
    torch.manual_seed(seed)
    ref = Hy4MoE(cfg).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn(p.shape) * (0.3 if p.dim() > 1 else 0.05))
    return cfg, ref


def test_partial_golden_sums_to_reference_routed():
    cfg, ref = _ref()
    x = torch.randn(8, cfg.hidden_size) * 3  # large enough for the clamp to bite
    I = cfg.intermediate_size
    with torch.no_grad():
        scores, idx, w = L.route_golden(x, ref.router_weight, ref.router_bias, cfg.top_k, cfg.routed_scaling_factor)
        r_w, r_idx = ref.route(x)
        assert torch.equal(idx.sort(dim=-1)[0], r_idx.sort(dim=-1)[0])
        assert torch.allclose(scores, ref.scores(x))
        dense = ref.dense_weights(r_w, r_idx)
        assert torch.allclose(ref.dense_weights(w, idx), dense, atol=1e-6)
        routed = ref.routed(x, dense)
        gate, up = ref.gate_up_proj[:, :I], ref.gate_up_proj[:, I:]
        ids, uw = L.union_of(idx, w)
        # tensor parallel: 2 chips, half of the intermediate channels each
        tp = sum(
            L.partial_golden(x, ids, uw, gate[:, c], up[:, c], ref.down_proj[:, :, c])
            for c in (slice(0, I // 2), slice(I // 2, I))
        )
        assert torch.allclose(tp, routed, atol=1e-4), (tp - routed).abs().max()
        # expert parallel: 4 chips, 4 experts each
        ep = torch.zeros_like(routed)
        for chip in range(4):
            cids, cw = L.union_of(idx, w, expert_base=4 * chip, expert_count=4)
            sel = slice(4 * chip, 4 * chip + 4)
            ep += L.partial_golden(x, cids, cw, gate[sel], up[sel], ref.down_proj[sel])
        assert torch.allclose(ep, routed, atol=1e-4), (ep - routed).abs().max()
        # the plain slot is the shared expert
        sh = ref.shared
        shared = L.partial_golden(
            x, [0], torch.ones(8, 1), sh.gate_proj[None], sh.up_proj[None], sh.down_proj[None], n_plain=1
        )
        assert torch.allclose(shared, sh(x), atol=1e-4)


def test_streams_hold_the_expected_tiles():
    for share in (L.tp_share(32), L.tp_share(8), L.ep_share()):
        E = 2
        g = torch.Generator().manual_seed(1)
        gate = torch.randn(E, share.channels, L.HIDDEN, generator=g)
        up = torch.randn(E, share.channels, L.HIDDEN, generator=g)
        down = torch.randn(E, L.HIDDEN, share.channels, generator=g)
        gu = L.gate_up_stream(gate, up, share)
        dn = L.down_stream(down, share)
        tile_rows = L.TILE
        assert gu.shape == (E * share.gate_up_stride // L.BFP4_TILE_BYTES * tile_rows, 256)
        assert dn.shape == (E * share.down_stride // L.BFP4_TILE_BYTES * tile_rows, 256)
        e, bank, t, k = 1, 5, share.pb - 1, 7
        piece = t * L.BANKS + bank
        p, q = divmod(piece, share.S)
        tile0 = (e * share.pb + t) * share.piece_k * 2 + k * 2  # tile index in the bank's shard
        kk = q * share.piece_k + k
        rows = slice(tile0 * 32, tile0 * 32 + 32)
        cols = slice(bank * 32, bank * 32 + 32)
        assert torch.equal(gu[rows, cols], gate[e].T[kk * 32 : kk * 32 + 32, p * 32 : p * 32 + 32])
        rows = slice((tile0 + 1) * 32, (tile0 + 1) * 32 + 32)
        assert torch.equal(gu[rows, cols], up[e].T[kk * 32 : kk * 32 + 32, p * 32 : p * 32 + 32])
        c, kd = 13, share.P - 1
        tile0 = (e * L.COLS_PER_BANK + c) * share.P + kd
        col = bank * L.COLS_PER_BANK + c
        rows = slice(tile0 * 32, tile0 * 32 + 32)
        assert torch.equal(dn[rows, cols], down[e].T[kd * 32 : kd * 32 + 32, col * 32 : col * 32 + 32])
    x = torch.randn(L.ROWS, 96)
    assert torch.equal(L.from_row_tiles(L.to_row_tiles(x)), x)
