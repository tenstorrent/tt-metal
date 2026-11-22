# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused residual-path kernels (tt/fused) vs the HF Inkling2Loop modules: T5's cases and bar (PCC >= 0.9995,
eager + traced, two replays bit-equal), plus before/after replay time against E2's tt/residual.py ops.

Reference: same file as test_residual.py (INKLING_T5_REF); timings go to INKLING_E10_RESULTS (jsonl) if set.
"""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tests.test_residual import CONV_STEPS, PCC_MIN, pcc, ref  # noqa: F401 (fixture)
from models.demos.inkling2_mini.tt import residual as R
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv

TRACE_PARAMS = [{"trace_region_size": 16 * 1024 * 1024}]
RESULTS = os.environ.get("INKLING_E10_RESULTS")


def record(name, **kw):
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(name=name, **kw)) + "\n")


@pytest.fixture
def rows(device):
    return R.RowOps(device)


def row(vecs, device, dtype=ttnn.float32):
    """[r, C] -> [1,1,r,C] device row-form tensor."""
    return ttnn.from_torch(vecs.reshape(1, 1, -1, vecs.shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def host(t):
    return ttnn.to_torch(t).float().reshape(-1, t.shape[-1])


def replay_us(device, tids, reps=200):
    """Mean wall time of one replay (cycling through tids), dispatch included."""
    t0 = time.perf_counter()
    for i in range(reps):
        ttnn.execute_trace(device, tids[i % len(tids)], cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / reps * 1e6


def capture(device, fn):
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    return tid, out


# ---------------------------------------------------------------- short conv
def conv_sequence(device, conv, state, x, xs, step):
    """Zero the ring, then feed xs[t] at phase t % 4 through step(phase). Returns [steps, C]."""
    for s in state:
        ttnn.copy(F.dense_tensor(device, torch.zeros(1, conv.c)), s)
    outs = []
    for t in range(xs.shape[0]):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(xs[t].reshape(1, 1, 1, -1), dtype=x.dtype, layout=ttnn.TILE_LAYOUT), x
        )
        outs.append(host(step(t % 4))[0])
    return torch.stack(outs)


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("channels", [1024, 2304])
def test_fused_short_conv(device, rows, ref, channels):
    case = ref[f"conv_c{channels}"]
    taps = case["weight"].flip(-1).T[None, None].contiguous()  # §4 [1,1,4,C], row r = lag r
    w = {"post_attn_conv": taps}
    conv = FusedShortConv(rows, w, "post_attn_conv")
    state = conv.alloc_state(R.cache_index(17, loop_pass=3))
    x = row(case["x"][:1], device, ttnn.bfloat16)

    eager = conv_sequence(device, conv, state, x, case["x"], lambda ph: conv(x, state, ph))
    traces = [capture(device, lambda ph=ph: conv(x, state, ph)) for ph in range(4)]

    def replay(ph):
        ttnn.execute_trace(device, traces[ph][0], cq_id=0, blocking=True)
        return traces[ph][1]

    first = conv_sequence(device, conv, state, x, case["x"], replay)
    second = conv_sequence(device, conv, state, x, case["x"], replay)
    us = replay_us(device, [t for t, _ in traces])
    for t, _ in traces:
        ttnn.release_trace(device, t)
    assert torch.equal(first, second), "trace replays differ"
    p = pcc(first, case["out"])

    # before: E2's ShortConv, same input, replay time
    old = R.ShortConv(rows, w, "post_attn_conv")
    ring = old.alloc_state(0)
    old(x, ring, 0)
    tid, _ = capture(device, lambda: old(x, ring, 0))
    us_old = replay_us(device, [tid])
    ttnn.release_trace(device, tid)
    record(f"short_conv_c{channels}", pcc=p, pcc_eager=pcc(eager, case["out"]), us=us, us_e2=us_old)
    assert min(p, pcc(eager, case["out"])) >= PCC_MIN, p


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16])
def test_fused_conv_add(device, rows, ref, out_dtype):
    """Layer glue: out = resid + bf16(conv(x)), fp32 add, out_dtype (model: post-attn fp32 / layer exit bf16)."""
    case = ref["conv_c2304"]
    taps = case["weight"].flip(-1).T[None, None].contiguous()
    w = {"post_mlp_conv": taps}
    conv = FusedShortConv(rows, w, "post_mlp_conv")
    state = conv.alloc_state(0)
    x = row(case["x"][:1], device, ttnn.bfloat16)
    resid_t = torch.randn(1, 2304) * 3
    resid = row(resid_t, device)
    got = conv_sequence(device, conv, state, x, case["x"][:5], lambda ph: conv.conv_add(x, state, ph, resid, out_dtype))
    want = (
        (resid_t + case["out"][:5].to(torch.bfloat16).float())
        .to(torch.bfloat16 if out_dtype == ttnn.bfloat16 else torch.float32)
        .float()
    )
    p = pcc(got, want)
    err = (got - want).abs().max().item()
    record(f"conv_add_{out_dtype}", pcc=p, max_abs_err=err)
    assert p >= 0.99999, (p, err)


# ---------------------------------------------------------------- block attention residual
def traced_twice(device, fn):
    """Eager run, trace capture, two replays (bit-equal asserted). fn returns one device tensor."""
    eager = host(fn())
    tid, out = capture(device, fn)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    first = host(out)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    second = host(out)
    us = replay_us(device, [tid])
    ttnn.release_trace(device, tid)
    assert torch.equal(first, second), "trace replays differ"
    return eager, first, us


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("nq", [1, 3])
def test_fused_attn_res(device, rows, ref, nq, depth):
    from models.demos.inkling2_mini.tt.fused.attnres import FusedAttnRes, FusedRowOps

    case = ref[f"block_nq{nq}_d{depth}"]
    w = {"attn_res_q": torch.nn.functional.pad(case["query"], (0, 0, 0, 32 - nq))[None, None]}
    frows = FusedRowOps(device)
    attn = FusedAttnRes(frows, w, "attn_res_q", nq)
    block = [row(case["block"][k : k + 1], device, ttnn.bfloat16) for k in range(depth - 1)]
    stack = None
    for k in range(depth - 1):  # sealed states arrive as bf16 layer inputs
        stack = frows.seal(stack, block[k], k)
    partial = row(case["partial"][None], device)
    eager, traced, us = traced_twice(device, lambda: attn(stack, partial, depth, split=False))
    per_q = [pcc(traced[j], case["out"][j]) for j in range(nq)]
    p_eager = min(pcc(eager[j], case["out"][j]) for j in range(nq))
    # seal of a fresh tensor (one program) vs E2's seal (4 ops)
    _, _, us_seal = traced_twice(
        device, lambda: (frows.fresh.clear(), frows.seal(stack, block[0] if block else partial, 0), partial)[2]
    )

    # before: E2 AttnRes (seal of the partial + attend), same inputs
    old = R.AttnRes(rows, w, "attn_res_q", nq)
    st = None
    for k in range(depth - 1):
        st = rows.seal(st, block[k], k)
    _, old_out, us_old = traced_twice(device, lambda: old(st, partial, depth, split=False))
    _, _, us_seal_old = traced_twice(
        device, lambda: rows.seal(st, partial, depth - 1) if depth > 1 else rows.seal(None, partial, 0)
    )
    record(
        f"block_attn_res_nq{nq}",
        depth=depth,
        pcc_min_query=min(per_q),
        pcc_eager_min=p_eager,
        us=us,
        us_e2=us_old,
        us_seal=us_seal,
        us_seal_e2=us_seal_old,
        max_abs_vs_e2=(traced - old_out).abs().max().item(),
    )
    assert min(per_q + [p_eager]) >= PCC_MIN, per_q


# ---------------------------------------------------------------- message gates
def gate_weights(case, names):
    w = {}
    for g in names:  # §4: down [1,1,2C,64] bf16, forget / gain [1,1,64,C] fp32 ([in, out])
        w[f"msg_gate.{g}.down"] = case["down"].T[None, None].contiguous().bfloat16()
        w[f"msg_gate.{g}.forget"] = case["forget"].T[None, None].contiguous()
        w[f"msg_gate.{g}.gain"] = case["gain"].T[None, None].contiguous()
    return w


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
def test_fused_message_gate(device, rows, ref):
    from models.demos.inkling2_mini.tt.fused.gate import FusedMessageGate

    case = ref["gate"]
    w = gate_weights(case, ["k"])
    gate = FusedMessageGate(rows, w, "k", num_rows=4)
    h, m = row(case["h"], device), row(case["m"], device)  # rows 0-3: four hidden-state scales
    eager, traced, us = traced_twice(device, lambda: gate(h, m))
    per_row = [pcc(traced[i], case["out"][i]) for i in range(4)]
    old = R.MessageGate(rows, w, "k")
    _, old_out, us_old = traced_twice(device, lambda: old(h, m))
    record(
        "message_gate",
        pcc_min_row=min(per_row),
        us=us,
        us_e2=us_old,
        max_abs_vs_e2=(traced - old_out).abs().max().item(),
    )
    assert min(per_row + [pcc(eager[i], case["out"][i]) for i in range(4)]) >= PCC_MIN, per_row


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
def test_fused_qkv_gate(device, rows, ref):
    """Rows 0/1/2 through gates q/k/v (same HF weights for all three, so row i must equal HF row i)."""
    from models.demos.inkling2_mini.tt.fused.gate import FusedQKVGate
    from models.demos.inkling2_mini.tt.model import _QKVGate

    case = ref["gate"]
    w = gate_weights(case, ["q", "k", "v"])
    gate = FusedQKVGate(rows, w)
    h, m = row(case["h"][:3], device), row(case["m"][:3], device)
    eager, traced, us = traced_twice(device, lambda: gate(h, m))
    per_row = [pcc(traced[i], case["out"][i]) for i in range(3)]
    old = _QKVGate(rows, w)
    _, old_out, us_old = traced_twice(device, lambda: old(h, m))
    record(
        "qkv_gate", pcc_min_row=min(per_row), us=us, us_e2=us_old, max_abs_vs_e2=(traced - old_out).abs().max().item()
    )
    assert min(per_row + [pcc(eager[i], case["out"][i]) for i in range(3)]) >= PCC_MIN, per_row


# ---------------------------------------------------------------- attention-input norms + row split
@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("n_weights", [1, 3])
def test_fused_row_norm(device, rows, n_weights):
    """vs E2's rows.rms (model._rms) on each row: global layers 3 weights, loop passes 1 weight for 3 rows."""
    from models.demos.inkling2_mini.tt.fused.norm import FusedRowNorm

    torch.manual_seed(3)
    x_t = torch.randn(3, 2304) * torch.tensor([[0.5], [2.0], [8.0]])
    wts = [(1 + 0.1 * torch.randn(2304)).bfloat16().float() for _ in range(n_weights)]  # model norms are bf16
    x = row(x_t, device)
    norm = FusedRowNorm(device, wts)
    eager = [host(t) for t in norm(x)]
    tid, outs = capture(device, lambda: norm(x))
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    first = [host(t) for t in outs]
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    second = [host(t) for t in outs]
    us = replay_us(device, [tid])
    ttnn.release_trace(device, tid)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    wd = [
        ttnn.from_torch(w.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for w in wts
    ]
    old = [
        host(
            rows.rms(
                ttnn.slice(x, [0, 0, i, 0], [1, 1, i + 1, 2304]), weight=wd[min(i, n_weights - 1)], dtype=ttnn.bfloat16
            )
        )
        for i in range(3)
    ]
    want = [
        (x_t[i] * torch.rsqrt(x_t[i].pow(2).mean() + 1e-6) * wts[min(i, n_weights - 1)].bfloat16().float())
        .bfloat16()
        .float()
        for i in range(3)
    ]
    p = min(pcc(first[i][0], want[i]) for i in range(3))
    p_e2 = min(pcc(old[i][0], want[i]) for i in range(3))
    record(f"row_norm_w{n_weights}", pcc_min=p, pcc_e2_min=p_e2, us=us)
    assert p >= 0.99999 and min(pcc(eager[i][0], want[i]) for i in range(3)) >= 0.99999, (p, p_e2)
