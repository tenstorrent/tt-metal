# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""R-row residual path for speculative verify (F.per_row): every row of an R-row call must be bit-equal to the
1-row call on that row's inputs. Covers the block attn-res (+ seal, + MoE input norm into an R-row L1 tensor), the
row norms, the loop messages and the short conv (own lags from the earlier rows, pending slots, commit)."""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tests.test_fused_twin import C, weights
from models.demos.inkling2_mini.tt import residual as R
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.attnres import FusedAttnRes, FusedRowOps
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv, commit
from models.demos.inkling2_mini.tt.fused.messages import FusedMessages
from models.demos.inkling2_mini.tt.fused.norm import FusedRowNorm


def _row(device, v, dtype=ttnn.bfloat16, l1=False):
    mem = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(
        v.reshape(1, 1, -1, C), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )


def _h(t):
    tw = F.twin(t)
    if tw is not None:  # row_out=False outputs: only the twin is written
        return F.from_dense_matrix(ttnn.to_torch(tw[0]).float(), C, tw[1]).clone()
    return ttnn.to_torch(t).float().reshape(-1, C).clone()


def _eq(a, b, what):
    assert a.shape == b.shape, (what, a.shape, b.shape)
    bad = (a != b) & ~(torch.isnan(a) & torch.isnan(b))
    assert not bad.any(), f"{what}: {int(bad.sum())} elements differ, max {float((a - b).abs().max())}"


@pytest.mark.parametrize("n_rows", [2, 3])
@pytest.mark.parametrize("is_global", [False, True])
def test_rows_residual(device, n_rows, is_global):
    g = torch.Generator().manual_seed(5 + n_rows)
    w, norms = weights(g, is_global)
    w["mlp_norm"] = (1 + torch.randn(C, generator=g) * 0.1).bfloat16().float()
    frows = FusedRowOps(device)
    nq = 3 if is_global else 1
    attn, mlp = FusedAttnRes(frows, w, "attn_res_q", nq), FusedAttnRes(frows, w, "mlp_res_q", 1)
    norm = FusedRowNorm(device, norms)
    moe_norm = FusedRowNorm(device, [w["mlp_norm"]], out_l1=True)
    depth = 3
    hs = [torch.randn(depth, C, generator=g) for _ in range(n_rows)]  # sealed states 0..depth-2 + the partial
    res = [torch.randn(C, generator=g) for _ in range(n_rows)]
    rows_h = [[_row(device, v[i]) for i in range(depth)] for v in hs]
    rows_res = [_row(device, v, ttnn.float32) for v in res]

    def app(r):
        """One row's attn-res -> norms, then mlp attn-res + MoE norm (model._app's residual part)."""
        for i in range(depth - 1):
            frows.seal(None, rows_h[r][i], i)
        x = attn(None, rows_h[r][-1], depth, split=False, row_out=False)  # R rows: one shared output
        xs = norm(x, num_rows=nq, first_row=F.row_index() * nq)
        y = mlp(None, rows_res[r], depth + 1, norm=moe_norm)
        return x, xs, y

    ref = []
    for r in range(n_rows):  # 1-row calls, one row at a time
        x, xs, y = app(r)
        ref.append((_h(x), [_h(t) for t in xs], _h(y)))
    got = F.per_row(device, n_rows, app)
    y_all = _h(got[0][2])
    assert y_all.shape[0] == n_rows
    for r, (x, xs, _) in enumerate(got):
        _eq(_h(x)[r * nq : (r + 1) * nq], ref[r][0], f"attn-res row {r}")
        for i, t in enumerate(xs):
            _eq(_h(t), ref[r][1][i], f"norm {i} row {r}")
        _eq(y_all[r : r + 1], ref[r][2], f"moe norm row {r}")


@pytest.mark.parametrize("n_rows", [2, 3])
@pytest.mark.parametrize("phase", [0, 3])
def test_rows_conv(device, n_rows, phase):
    """conv_mlp: y = an R-row bf16 L1 tensor (the R-row expert step's output), resid fp32 per row."""
    g = torch.Generator().manual_seed(11 + phase)
    w, _ = weights(g, False)
    conv = FusedShortConv(R.RowOps(device), w, "post_mlp_conv")
    hist = torch.randn(4, C, generator=g)
    ys = torch.randn(n_rows, C, generator=g)
    res = [torch.randn(C, generator=g) for _ in range(n_rows)]
    rows_res = [_row(device, v, ttnn.float32) for v in res]

    def ring_of(dev_ring):
        return [_h_dense(t) for t in dev_ring]

    def _h_dense(t):
        return F.from_dense_matrix(ttnn.to_torch(t).float(), C, 1).clone()

    # reference: n_rows single steps, the ring updated by each
    ring = conv.alloc_state(0)
    for i, t in enumerate(ring):
        ttnn.copy(F.dense_tensor(device, hist[i : i + 1]), t)
    ref, ref_rings = [], []
    for r in range(n_rows):
        y1 = _row(device, ys[r], l1=True)
        out = conv.conv_add(y1, ring, (phase + r) % 4, rows_res[r], ttnn.bfloat16)
        ref.append(_h(out))
        ref_rings.append(ring_of(ring))
    # R rows in one program from the same starting ring
    for i, t in enumerate(ring):
        ttnn.copy(F.dense_tensor(device, hist[i : i + 1]), t)
    y_all = _row(device, ys, l1=True)
    pend = [conv.pending_slot(ring, r) for r in range(n_rows)]
    outs = F.per_row(
        device,
        n_rows,
        lambda r: conv.conv_add(y_all, ring, (phase + r) % 4, rows_res[r], ttnn.bfloat16, x_row=r, slot=pend[r]),
    )
    for r in range(n_rows):
        _eq(_h(outs[r]), ref[r], f"conv row {r}")
    _eq(torch.cat(ring_of(ring)), torch.cat([h.reshape(1, C) for h in hist]), "ring untouched by the R-row step")
    for n_acc in range(1, n_rows + 1):  # commit the first n_acc rows: the ring of n_acc single steps
        for i, t in enumerate(ring):
            ttnn.copy(F.dense_tensor(device, hist[i : i + 1]), t)
        commit(device, [(pend[r], ring[(phase + r) % 4]) for r in range(n_acc)], F.pages(C))
        _eq(torch.cat(ring_of(ring)), torch.cat(ref_rings[n_acc - 1]), f"ring after commit of {n_acc}")


@pytest.mark.parametrize("n_rows", [2, 3])
def test_rows_messages(device, n_rows):
    g = torch.Generator().manual_seed(3)
    wq = torch.randn(1, 1, 12 * 32, C, generator=g)
    msg = FusedMessages(device, {"msg_q": wq})
    keys = [
        (_row(device, torch.randn(C, generator=g)), _row(device, torch.randn(C, generator=g))) for _ in range(n_rows)
    ]
    ref = []
    for a, b in keys:
        ref.append([(_h(q), _h(m)) for q, m in msg(a, b)])
    got = F.per_row(device, n_rows, lambda r: msg(*keys[r]))  # shared: qkv rows 3r.., mlp row r
    for r in range(n_rows):
        for l, (q, m) in enumerate(got[r]):
            _eq(_h(q)[3 * r : 3 * r + 3], ref[r][l][0], f"msg qkv layer {l} row {r}")
            _eq(_h(m)[r : r + 1], ref[r][l][1], f"msg mlp layer {l} row {r}")
