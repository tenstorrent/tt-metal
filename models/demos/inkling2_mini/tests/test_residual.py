# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""HF Inkling2Loop residual-path references (random weights, batch-1 decode) and the cross-pass messages on TT.

The reference cases (block attention residual, messages, message gate, short conv) also feed tests/test_fused.py,
which checks the fused kernels the model runs. Here: MessageAttnRes.all_messages eagerly, then under trace twice:
both replays must match bit for bit and reach PCC >= 0.9995.

The HF reference needs transformers >= 5.24 (see tests/hf_ref.py), so this file doubles as the reference
generator, run with that python:
    $INKLING_REF_PYTHON test_residual.py OUT.pt
The test calls it once per reference cache directory; without a reference python the tests that need it skip.
"""

import json
import os
import sys
import time

import torch

H = 2304
DEPTHS = (1, 2, 3, 4, 5)
CONV_STEPS = 12
PCC_MIN = 0.9995
RESULTS = os.environ.get("INKLING_RESIDUAL_RESULTS")  # optional jsonl of pcc / timings


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
from models.demos.inkling2_mini.tests import hf_ref
from models.demos.inkling2_mini.tt.residual import TILE, MessageAttnRes, RowOps, cache_index

TRACE_PARAMS = [{"trace_region_size": 8 * 1024 * 1024}]


@pytest.fixture(scope="module")
def ref():
    hf_ref.require()
    path = hf_ref.cache_dir("residual") / "ref.pt"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        hf_ref.run(__file__, tmp, isolated=True)
        tmp.replace(path)
    return torch.load(path)


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
    """cache form: host ttnn TILE tensor (what ttnn.load_tensor returns)."""
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def row(vec, device, dtype=ttnn.bfloat16):
    """[C] -> [1,1,1,C] row-form device tensor (the model's per-token form)."""
    return ttnn.from_torch(vec.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


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
def test_all_messages(device, rows, ref):
    """MessageAttnRes.all_messages, as the model calls it: row-form bf16 keys (input of L12, output of L17) ->
    [1,1,48,C] fp32, row 4l+j = message j (q, k, v, mlp) of loop layer l."""
    case = ref["msg"]
    msg_q = torch.nn.functional.pad(case["query"], (0, 0, 0, TILE - 4))[None]  # cache form [1,12,32,C]
    mod = MessageAttnRes(rows, {"msg_q": cache_tensor(msg_q, ttnn.float32)})
    h12, h17 = row(case["h12"], device), row(case["h17"], device)
    eager, traced, us = traced_twice(device, lambda: [mod.all_messages(h12, h17)])
    got = traced[0].reshape(12, 4, -1)
    per = [pcc(got[l, j], case["out"][l, j]) for l in range(12) for j in range(4)]
    p = pcc(got, case["out"])
    record("all_messages", pcc=p, pcc_min_message=min(per), trace_us=us)
    assert hf_ref.finite_min(per + [p, pcc(eager[0].reshape(12, 4, -1), case["out"])]) >= PCC_MIN, (p, min(per))


def test_cache_index():
    assert [cache_index(6), cache_index(6, 2), cache_index(17, 2), cache_index(6, 3), cache_index(17, 3)] == [
        6,
        24,
        35,
        36,
        47,
    ]
