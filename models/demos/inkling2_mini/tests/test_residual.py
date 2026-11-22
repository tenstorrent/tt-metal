# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini residual ops on TT vs the HF Inkling2Loop modules, random weights, batch-1 decode.

Each op runs eagerly, then under trace twice: both replays must match bit for bit and reach PCC >= 0.9995.

The HF reference needs transformers >= 5.24, which the tt-metal venv lacks, so this file doubles as the
reference generator, run with another python (INKLING_REF_PYTHON):
    $INKLING_REF_PYTHON test_residual.py OUT.pt
The test calls it once when INKLING_T5_REF (default /tmp/inkling2_mini_t5_ref.pt) does not exist.
"""

import json
import os
import subprocess
import sys
import time

import torch

H = 2304
DEPTHS = (1, 2, 3, 4, 5)
CONV_STEPS = 12
PCC_MIN = 0.9995
REF_PATH = os.environ.get("INKLING_T5_REF", "/tmp/inkling2_mini_t5_ref.pt")
REF_PYTHON = os.environ.get("INKLING_REF_PYTHON", "/work/inkling-bringup/ref-venv/bin/python")
RESULTS = os.environ.get("INKLING_T5_RESULTS")  # optional jsonl of pcc / timings for the report


# ---------------------------------------------------------------- reference (runs in the HF venv)
def make_reference(out_path):
    from transformers.models.inkling2_loop import modeling_inkling2_loop as m
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

    gen = torch.Generator().manual_seed(1234)
    cfg = Inkling2LoopTextConfig()
    cfg._attn_implementation = "eager"
    assert cfg.hidden_size == H and cfg.loop_gate_rank == 64 and cfg.conv_kernel_size == 4

    def bf16(t):
        return t.to(torch.bfloat16).float()

    def states(depth):
        # Sealed block states are bf16 layer inputs; the partial is fp32. Per-state scales exercise the norm.
        s = torch.randn(depth, H, generator=gen) * (torch.rand(depth, 1, generator=gen) * 4 + 0.25)
        return bf16(s[:-1]), s[-1]

    out = {}
    with torch.no_grad():
        for nq in (1, 3):  # 1 query: local attn / mlp / final input; 3: global q, k, v inputs
            mod = m.Inkling2LoopBlockAttentionResidual(cfg, num_queries=nq)
            mod.query.copy_(torch.randn(mod.query.shape, generator=gen) * 2)
            for d in DEPTHS:
                bs, part = states(d)
                y = torch.stack(mod(bs[None, None], part[None, None]), 0)[:, 0, 0]  # [nq, H]
                out[f"block_nq{nq}_d{d}"] = dict(query=mod.query.view(nq, H).clone(), block=bs, partial=part, out=y)

        mod = m.Inkling2LoopMessageAttentionResidual(cfg)  # 12 loop layers x (q, k, v, mlp)
        mod.query.copy_(torch.randn(mod.query.shape, generator=gen) * 2)
        # Keys: [input of L12 (sealed block state), output of L17] of the previous pass, both bf16.
        h12, h17 = bf16(torch.randn(2, H, generator=gen) * torch.tensor([[1.0], [3.0]]))
        y = torch.stack([mod(h12[None, None, None], h17[None, None], l)[0, 0] for l in range(12)])  # [12, 4, H]
        out["msg"] = dict(query=mod.query.clone(), h12=h12, h17=h17, out=y)

        gate = m.Inkling2LoopMessageGate(cfg)
        gate.down_proj.to(torch.bfloat16)  # bf16 as in the checkpoint; the gate projections stay fp32
        for p in gate.parameters():
            p.mul_(3)  # default init barely leaves tanh's linear range
        hs = torch.randn(4, H, generator=gen) * torch.tensor([[0.5], [1.0], [3.0], [10.0]])
        msg = torch.randn(4, H, generator=gen)
        out["gate"] = dict(
            down=gate.down_proj.weight.float().clone(),
            forget=gate.forget_gate_proj.weight.clone(),
            gain=gate.message_gate_proj.weight.clone(),
            h=hs,
            m=msg,
            out=gate(hs, msg),
        )

        for c in (2304, 1024):
            conv = m.Inkling2LoopShortConvolution(c, 4, layer_idx=0, conv_idx=2).to(torch.float32)
            conv.conv1d.weight.copy_(torch.randn(conv.conv1d.weight.shape, generator=gen) * 0.5)
            x = torch.randn(1, CONV_STEPS, c, generator=gen).to(torch.bfloat16)
            y = conv(x)  # no cache: causal conv over the sequence == decode from a zero state
            out[f"conv_c{c}"] = dict(weight=conv.conv1d.weight[:, 0].clone(), x=x[0].float(), out=y[0].float())
    torch.save(out, out_path)


if __name__ == "__main__":
    make_reference(sys.argv[1])
    sys.exit(0)

# ---------------------------------------------------------------- TT test (runs in the tt-metal venv)
import pytest

import ttnn
from models.demos.inkling2_mini.tt.residual import (
    TILE,
    AttnRes,
    MessageAttnRes,
    MessageGate,
    RowOps,
    ShortConv,
    cache_index,
)

TRACE_PARAMS = [{"trace_region_size": 8 * 1024 * 1024}]


@pytest.fixture(scope="module")
def ref():
    if not os.path.exists(REF_PATH):
        subprocess.run([REF_PYTHON, "-I", os.path.abspath(__file__), REF_PATH], check=True)
    return torch.load(REF_PATH)


@pytest.fixture
def rows(device):
    return RowOps(device)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def record(name, **kw):
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(name=name, **kw)) + "\n")


def cache_tensor(t, dtype):
    """§4 cache form: host ttnn TILE tensor (what ttnn.load_tensor returns)."""
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def tok(vecs, device, dtype=ttnn.float32):
    """[r, C] -> [1,1,32,C] device tensor with rows 0..r-1."""
    t = torch.zeros(TILE, vecs.shape[-1])
    t[: vecs.shape[0]] = vecs.reshape(-1, vecs.shape[-1])
    return ttnn.from_torch(t[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def host(t):
    return ttnn.to_torch(t).float()[0, 0]


def hosts(ts):
    return torch.stack([host(t) for t in ts])


def time_trace(device, tid, reps=300):
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / reps * 1e6


def traced_twice(device, fn):
    """Eager run (compiles), trace capture, two checked replays, mean replay time in us.
    fn returns a list of device tensors."""
    eager = hosts(fn())
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    first = hosts(out)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    second = hosts(out)
    us = time_trace(device, tid)
    ttnn.release_trace(device, tid)
    assert torch.equal(first, second), "trace replays differ"
    return eager, first, us


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("nq", [1, 3])
def test_block_attn_res(device, rows, ref, nq, depth):
    case = ref[f"block_nq{nq}_d{depth}"]
    weights = {
        "attn_res_q": cache_tensor(
            torch.nn.functional.pad(case["query"], (0, 0, 0, TILE - nq))[None, None], ttnn.float32
        )
    }
    attn = AttnRes(rows, weights, "attn_res_q", nq)
    stack = None
    for k in range(depth - 1):  # sealed states arrive as bf16 layer inputs
        stack = rows.seal(stack, tok(case["block"][k : k + 1], device, ttnn.bfloat16), k)
    partial = tok(case["partial"][None], device)
    eager, traced, us = traced_twice(device, lambda: attn(stack, partial, depth))
    per_q = [pcc(traced[j, 0], case["out"][j]) for j in range(nq)]
    p = pcc(traced[:, 0], case["out"])
    record(f"block_attn_res_nq{nq}", depth=depth, pcc=p, pcc_min_query=min(per_q), trace_us=us)
    assert min(per_q + [p, pcc(eager[:, 0], case["out"])]) >= PCC_MIN, (p, per_q)


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
def test_messages_for_pass(device, rows, ref):
    case = ref["msg"]
    msg_q = torch.nn.functional.pad(case["query"], (0, 0, 0, TILE - 4))[None]  # §4 [1,12,32,C]
    mod = MessageAttnRes(rows, {"msg_q": cache_tensor(msg_q, ttnn.float32)})
    h12, h17 = tok(case["h12"][None], device, ttnn.bfloat16), tok(case["h17"][None], device, ttnn.bfloat16)
    eager, traced, us = traced_twice(device, lambda: [m for layer in mod.messages_for_pass(h12, h17) for m in layer])
    got = traced[:, 0].reshape(12, 4, -1)
    per = [pcc(got[l, j], case["out"][l, j]) for l in range(12) for j in range(4)]
    p = pcc(got, case["out"])
    record("messages_for_pass", depth=2, pcc=p, pcc_min_message=min(per), trace_us=us)
    assert min(per + [p, pcc(eager[:, 0].reshape(12, 4, -1), case["out"])]) >= PCC_MIN, (p, min(per))


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
def test_message_gate(device, rows, ref):
    case = ref["gate"]
    weights = {  # §4: down [1,1,2C,64] bf16, forget / gain [1,1,64,C] fp32 ([in, out])
        "msg_gate.k.down": cache_tensor(case["down"].T[None, None].contiguous(), ttnn.bfloat16),
        "msg_gate.k.forget": cache_tensor(case["forget"].T[None, None].contiguous(), ttnn.float32),
        "msg_gate.k.gain": cache_tensor(case["gain"].T[None, None].contiguous(), ttnn.float32),
    }
    gate = MessageGate(rows, weights, "k")
    h, m = tok(case["h"], device), tok(case["m"], device)  # rows 0-3: four hidden-state scales
    eager, traced, us = traced_twice(device, lambda: [gate(h, m)])
    per_row = [pcc(traced[0, i], case["out"][i]) for i in range(4)]
    record("message_gate", pcc=pcc(traced[0, :4], case["out"]), pcc_min_row=min(per_row), trace_us=us)
    assert min(per_row + [pcc(eager[0, :4], case["out"])]) >= PCC_MIN, per_row


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("channels", [2304, 1024])
def test_short_conv(device, rows, ref, channels):
    case = ref[f"conv_c{channels}"]
    taps = case["weight"].flip(-1).T[None, None].contiguous()  # §4 [1,1,4,C], row r = lag r
    conv = ShortConv(rows, {"post_attn_conv": cache_tensor(taps, ttnn.float32)}, "post_attn_conv")
    state = conv.alloc_state(cache_index(17, loop_pass=3))
    zero = ttnn.from_torch(torch.zeros(1, 1, TILE, channels), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    x = tok(case["x"][:1], device, ttnn.bfloat16)

    def run_sequence(step):
        ttnn.copy_host_to_device_tensor(zero, state)
        outs = []
        for t in range(CONV_STEPS):
            xt = torch.zeros(1, 1, TILE, channels)
            xt[0, 0, 0] = case["x"][t]
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x)
            outs.append(host(step())[0])
        return torch.stack(outs)

    eager = run_sequence(lambda: conv(x, state))
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = conv(x, state)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    def replay():
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        return out

    first, second = run_sequence(replay), run_sequence(replay)
    us = time_trace(device, tid)
    ttnn.release_trace(device, tid)
    assert torch.equal(first, second), "trace replays differ"
    p = pcc(first, case["out"])
    record(f"short_conv_c{channels}", pcc=p, steps=CONV_STEPS, trace_us=us)
    assert min(p, pcc(eager, case["out"])) >= PCC_MIN, p


def test_cache_index():
    assert [cache_index(6), cache_index(6, 2), cache_index(17, 2), cache_index(6, 3), cache_index(17, 3)] == [
        6,
        24,
        35,
        36,
        47,
    ]
