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


def _row(device, v, dtype=ttnn.bfloat16, l1=False, c=None):
    c = c or v.shape[-1]
    mem = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(
        v.reshape(1, 1, -1, c), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
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


@pytest.mark.parametrize("n_rows", [2, 3, 4])
def test_rows_router(device, n_rows):
    """R-row router: row r's ids (in order) and weights bit-equal to the 1-row call on row r (ties included)."""
    from models.demos.inkling2_mini.tests.test_router import Chain
    from models.demos.inkling2_mini.tt.fused.router import FusedRouter
    from models.demos.inkling2_mini.tt.moe import HIDDEN, NUM_EXPERTS, ROUTE_SCALE, TOP_K

    g = torch.Generator().manual_seed(n_rows)
    w = torch.randn(HIDDEN, NUM_EXPERTS, generator=g) * 0.02
    dup = torch.randperm(NUM_EXPERTS, generator=g)[:16]
    w[:, dup[8:]] = w[:, dup[:8]]
    bias = torch.randn(NUM_EXPERTS, generator=g) * 0.05
    bias = bias - bias.mean()
    chain = Chain(device, w, bias)
    router = FusedRouter(device, chain.w, bias, TOP_K, ROUTE_SCALE, 1e-20, first_row=2)
    xs = torch.randn(n_rows, HIDDEN, generator=g)
    xs[1] = xs[0]  # identical rows too

    def l1(v):
        return ttnn.from_torch(v.reshape(1, 1, -1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip

    def host(t):
        return ttnn.to_torch(t).reshape(-1, t.shape[-1])

    ref = []
    for r in range(n_rows):
        wt, ids = router(l1(xs[r]))
        ref.append((host(wt)[0, :TOP_K].view(torch.int16).clone(), host(ids)[0, :TOP_K].int().clone()))
    wt, ids = router(l1(xs))
    hw, hi = host(wt), host(ids)
    for r in range(n_rows):
        _eq(hw[r, :TOP_K].view(torch.int16).float(), ref[r][0].float(), f"router weights row {r}")
        _eq(hi[r, :TOP_K].int().float(), ref[r][1].float(), f"router ids row {r}")


@pytest.mark.parametrize("n_rows", [2, 3, 4])
def test_rows_mlp_gate(device, n_rows):
    """The message gate built for R rows (FusedMessageGate num_rows=R): row r bit-equal to the 1-row gate."""
    from models.demos.inkling2_mini.tt.fused.gate import FusedMessageGate

    g = torch.Generator().manual_seed(7)
    w, _ = weights(g, False)
    one = FusedMessageGate(R.RowOps(device), w, "mlp")
    many = FusedMessageGate(R.RowOps(device), w, "mlp", num_rows=n_rows)
    hs, ms = torch.randn(n_rows, C, generator=g), torch.randn(n_rows, C, generator=g) * 0.5
    ref = [_h(one(_row(device, hs[r], ttnn.float32), _row(device, ms[r], ttnn.float32))) for r in range(n_rows)]
    got = _h(many(_row(device, hs, ttnn.float32), _row(device, ms, ttnn.float32)))
    for r in range(n_rows):
        _eq(got[r : r + 1], ref[r], f"mlp gate row {r}")


@pytest.mark.parametrize("n_rows", [2, 3])
def test_rows_qkv_gate(device, n_rows):
    """The q/k/v gates for R verify rows (FusedQKVGate num_tokens=R, rows 3r + j through gate j), fed as in
    model._app: the attn-res output repeated in rows 3r..3r+2 (one shared twin) and the per-row loop messages
    (shared twins); row 3r + j bit-equal to the 1-row gate's row j. Then the attention-input norms per row."""
    from models.demos.inkling2_mini.tt.fused.gate import FusedQKVGate

    g = torch.Generator().manual_seed(9)
    w, norms = weights(g, False)
    frows = FusedRowOps(device)
    attn = FusedAttnRes(frows, w, "attn_res_q", 1)
    one = FusedQKVGate(R.RowOps(device), w)
    many = FusedQKVGate(R.RowOps(device), w, num_tokens=n_rows)
    norm = FusedRowNorm(device, norms)
    msg = FusedMessages(device, {"msg_q": torch.randn(1, 1, 12 * 32, C, generator=g)})
    hs = [torch.randn(2, C, generator=g) for _ in range(n_rows)]
    keys = [
        (_row(device, torch.randn(C, generator=g)), _row(device, torch.randn(C, generator=g))) for _ in range(n_rows)
    ]
    rows_h = [[_row(device, v[i]) for i in range(2)] for v in hs]

    def res(r):
        frows.seal(None, rows_h[r][0], 0)
        return attn(None, rows_h[r][1], 2, split=False, repeat_rows=3, row_out=False)

    ref = []
    for r in range(n_rows):
        x = res(r)
        qkv = msg(*keys[r])[4][0]  # loop layer 4's q/k/v messages
        y = one(x, qkv)
        ref.append((_h(y), [_h(t) for t in norm(y)]))
    xs = F.per_row(device, n_rows, res)
    ms = F.per_row(device, n_rows, lambda r: msg(*keys[r]))
    assert all(t is xs[0] for t in xs) and xs[0].shape[-2] == 3 * n_rows
    y = many(xs[0], ms[0][4][0])
    got = _h(y)
    ns = F.per_row(device, n_rows, lambda r: norm(y, num_rows=3, first_row=3 * r))
    for r in range(n_rows):
        _eq(got[3 * r : 3 * r + 3], ref[r][0], f"qkv gate row {r}")
        for j in range(3):
            _eq(_h(ns[r][j]), ref[r][1][j], f"norm {j} row {r}")


@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16])
def test_rows_bank_linear(device, out_dtype):
    """Bank-streamed linears (unembed, w_all, w_o): R verify rows ride in the activation tile rows; row r of the
    R-row call bit-equal to the 1-row call (L1 accumulation across K blocks, as the unembed)."""
    from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear

    torch.manual_seed(0)
    k, n, n_rows = 2304, 7 * 32 * 6, 4
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )  # fmt: skip
    bl = BankLinear(device, torch.randn(k, n) * 0.02, ttnn.bfloat8_b, n, kb=2, cores_per_bank=2,
                    compute_kernel_config=ckc, out_dtype=out_dtype)  # fmt: skip
    xs = torch.randn(n_rows, k)
    ref = [ttnn.to_torch(bl(_row(device, xs[r]))).float().reshape(-1, n)[:1].clone() for r in range(n_rows)]
    for x_rows in (0, n_rows):
        bl.x_rows = x_rows
        got = ttnn.to_torch(bl(_row(device, xs))).float().reshape(-1, n)
        for r in range(n_rows):
            _eq(got[r : r + 1], ref[r], f"linear row {r} (x_rows {x_rows})")


def test_rows_head(device):
    """Unembed (bank linear, raw row per verify row) + softcap / argmax tail per row: logits and token of row r
    bit-equal to the 1-row head on row r. A 2-chunk vocab (the model's chunk, kb and cores per bank)."""
    from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear
    from models.demos.inkling2_mini.tt.fused.head import FusedLogitsTail
    from models.demos.inkling2_mini.tt.model import SOFTCAP, UNEMBED_CHUNK

    torch.manual_seed(1)
    n_rows, vocab = 3, 2 * UNEMBED_CHUNK
    w = torch.randn(C, vocab) * 0.02
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )  # fmt: skip
    bl = BankLinear(device, w, ttnn.bfloat8_b, UNEMBED_CHUNK, kb=3, cores_per_bank=3, compute_kernel_config=ckc)
    tail = FusedLogitsTail(device, bl, vocab, vocab - 40, SOFTCAP)
    xs = torch.randn(n_rows, C)
    xs[2] = xs[0] * 300.0  # saturated softcap

    def host(lg, tok):
        return ttnn.to_torch(lg).float().reshape(1, -1).clone(), int(ttnn.to_torch(tok).reshape(-1)[0])

    ref = [host(*tail(_row(device, xs[r]))) for r in range(n_rows)]
    got = [host(*o) for o in tail.rows(_row(device, xs))]
    for r in range(n_rows):
        _eq(got[r][0], ref[r][0], f"logits row {r}")
        assert got[r][1] == ref[r][1], (r, got[r][1], ref[r][1])


def test_rows_final_norm(device):
    """model._rms (residual.RowOps.rms, ttnn rms_norm) on R rows: row r bit-equal to the 1-row call."""
    torch.manual_seed(2)
    rows = R.RowOps(device)
    w = ttnn.from_torch((1 + torch.randn(C) * 0.1).reshape(1, 1, 1, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device)  # fmt: skip
    xs = torch.randn(3, C) * 3
    ref = [_h(rows.rms(_row(device, xs[r], ttnn.float32), weight=w, dtype=ttnn.bfloat16)) for r in range(3)]
    got = _h(rows.rms(_row(device, xs, ttnn.float32), weight=w, dtype=ttnn.bfloat16))
    for r in range(3):
        _eq(got[r : r + 1], ref[r], f"final norm row {r}")
