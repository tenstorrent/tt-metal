# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini attention decode vs the HF reference, random weights in §4 dtypes.

The HF reference needs transformers >= 5.24 (inkling2_loop), which the ttnn venv
does not have. The test writes weights + inputs to a temp file and runs this file
as a script under `INKLING_REF_PYTHON` (default: /work/inkling-bringup/ref-venv),
which replays HF decode token by token with its own DynamicCache.

    flock /tmp/tt-card.lock pytest models/demos/inkling2_mini/tests/test_attention.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import torch

REF_PYTHON = os.environ.get("INKLING_REF_PYTHON", "/work/inkling-bringup/ref-venv/bin/python")
REF_CONFIG = os.environ.get("INKLING_CONFIG", "/work/inkling-bringup/ref-sources/config.json")
HIDDEN = 2304
LOCAL_LAYER, GLOBAL_LAYER = 0, 5


# ------------------------------------------------------------------ reference (ref venv)
def _hf_config():
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

    cfg = Inkling2LoopTextConfig(**json.load(open(REF_CONFIG))["text_config"])
    cfg._attn_implementation = "eager"
    return cfg


def _hf_module(cfg, is_global):
    from transformers.models.inkling2_loop import modeling_inkling2_loop as M

    if is_global:
        return M.Inkling2LoopFullAttention(cfg, GLOBAL_LAYER)
    return M.Inkling2LoopSlidingAttention(cfg, LOCAL_LAYER)


def run_reference(in_path, out_path):
    """HF decode, one token per call, like generate() would drive the layer."""
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
        create_sliding_window_causal_mask,
    )
    from transformers.models.inkling2_loop.modeling_inkling2_loop import get_relative_distance

    torch.set_num_threads(int(os.environ.get("INKLING_REF_THREADS", "4")))
    data = torch.load(in_path)
    is_global = data["is_global"]
    cfg = _hf_config()
    attn = _hf_module(cfg, is_global)
    attn.load_state_dict(data["state_dict"])
    attn.to(torch.bfloat16)
    if not is_global:  # `_keep_in_fp32_modules_strict`: sconv.conv1d
        attn.k_proj.sconv.conv1d.float()
        attn.v_proj.sconv.conv1d.float()
    attn.eval()
    cache = DynamicCache(config=cfg)
    xq, xk, xv = (data[n].bfloat16() for n in ("xq", "xk", "xv"))
    outs = []
    t0 = time.time()
    with torch.no_grad():
        for t in range(xq.shape[0]):
            x = xq[t].view(1, 1, HIDDEN)
            pid = torch.tensor([[t]])
            mk = dict(config=cfg, inputs_embeds=x, attention_mask=None, cache=cache, position_ids=pid)
            mask = create_causal_mask(**mk) if is_global else create_sliding_window_causal_mask(**mk)
            conv_mask = create_recurrent_attention_mask(**mk)
            rel = get_relative_distance(x, cache, is_sliding=not is_global)
            kw = dict(
                hidden_states=x,
                attention_mask=mask,
                relative_distance=rel,
                conv_mask=conv_mask,
                cache=cache,
                position_ids=pid,
            )
            if is_global:
                kw.update(key_hidden_states=xk[t].view(1, 1, HIDDEN), value_hidden_states=xv[t].view(1, 1, HIDDEN))
            outs.append(attn(**kw)[0].view(HIDDEN))
    torch.save(torch.stack(outs), out_path)
    print(f"reference: {xq.shape[0]} steps in {time.time() - t0:.1f}s")


# ------------------------------------------------------------------ weights + inputs
def random_state_dict(is_global, seed):
    """Random HF state dict; every value representable in its §4 dtype (BFP8 / BF16)."""
    import ttnn

    g = torch.Generator().manual_seed(seed)

    def normal(shape, std):
        return torch.randn(shape, generator=g) * std

    def uniform(shape, lo, hi):
        return lo + (hi - lo) * torch.rand(shape, generator=g)

    def bfp8(t):  # [out, in] linear weight, quantized in the stored [in, out] orientation
        host = ttnn.from_torch(t.T.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(host).float().T.contiguous()

    def bf16(t):
        return t.bfloat16().float()

    sd = {
        "q_proj.linear.weight": bfp8(normal((3072, HIDDEN), 0.03)),
        "q_proj.norm.scale": bf16(uniform((24, 128), 0.5, 1.5)),
        "o_proj.gate_proj.weight": bfp8(normal((3072, HIDDEN), 0.03)),
        "o_proj.linear.weight": bfp8(normal((HIDDEN, 3072), 0.02)),
        "sink_proj.weight": bfp8(normal((24, HIDDEN), 0.04)),
        "relative_position_bias.linear.weight": bfp8(normal((384, HIDDEN), 0.04)),
        "relative_position_bias.norm.scale": bf16(uniform((24, 16), 0.5, 1.5)),
        "relative_position_bias.proj": bf16(normal((16, 1024), 0.5)),
    }
    for kv in ("k", "v"):
        lin = normal((1024, HIDDEN), 0.03)
        if is_global:
            dyn = normal((32, HIDDEN), 0.01)
            both = bfp8(torch.cat([lin, dyn]))  # one stored [2304, 1056] tensor
            sd[f"{kv}_proj.linear.weight"] = both[:1024]
            sd[f"{kv}_proj.sconv.weight_proj.weight"] = both[1024:]
            sd[f"{kv}_proj.sconv.static_weight"] = bf16(uniform((4, 1024), -0.5, 0.5))
        else:
            sd[f"{kv}_proj.linear.weight"] = bfp8(lin)
            sd[f"{kv}_proj.sconv.conv1d.weight"] = bf16(uniform((1024, 1, 4), -0.5, 0.5))
    # Pieces of the packed §4 tensors start at multiples of 32 columns, so quantizing each
    # piece alone gives the same BFP8 blocks as the packed tensor.
    return sd


def reference_outputs(sd, inputs, is_global, tag):
    cache_dir = os.environ.get("INKLING_REF_CACHE", "/tmp/inkling-t3-ref")
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{tag}.pt")
    if os.path.exists(out_path):
        return torch.load(out_path)
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.pt")
        torch.save({"state_dict": sd, "is_global": is_global, **inputs}, in_path)
        subprocess.run([REF_PYTHON, os.path.abspath(__file__), "ref", in_path, out_path], check=True)
    return torch.load(out_path)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "ref":
    run_reference(sys.argv[2], sys.argv[3])
    sys.exit(0)

# ------------------------------------------------------------------ device tests
import pytest  # noqa: E402

import ttnn  # noqa: E402
from models.demos.inkling2_mini.tt.attention import (  # noqa: E402
    DecodePositions,
    Inkling2Attention,
    host_weights_from_hf,
)

TRACE_REGION = 64 * 1024 * 1024


def _x_host(x):
    return ttnn.from_torch(x.view(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


class Runner:
    """One layer + its state + positions, driven eagerly or by a captured trace."""

    def __init__(self, device, is_global, sd, max_ctx, cache_dtype):
        self.device = device
        self.is_global = is_global
        self.attn = Inkling2Attention(device, host_weights_from_hf(sd, is_global), is_global)
        self.state = self.attn.new_state(max_ctx, cache_dtype)
        self.positions = DecodePositions(device, max_ctx)
        zero = torch.zeros(HIDDEN)
        self.x = [ttnn.to_device(_x_host(zero), device) for _ in range(3 if is_global else 1)]
        self.trace = None
        self.out = None

    def _step(self):
        self.positions.prepare()
        if self.is_global:
            return self.attn.decode(self.x[0], self.state, self.positions, self.x[1], self.x[2])
        return self.attn.decode(self.x[0], self.state, self.positions)

    def set_inputs(self, pos, xs):
        self.positions.update(pos)
        for dev, x in zip(self.x, xs):
            ttnn.copy_host_to_device_tensor(_x_host(x), dev)

    def reset_state(self):
        for t in self.state.tensors():
            ttnn.copy(ttnn.zeros_like(t), t)

    def capture(self):
        self._step()  # compile
        self.trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.out = self._step()
        ttnn.end_trace_capture(self.device, self.trace, cq_id=0)
        self.reset_state()

    def run(self):
        ttnn.execute_trace(self.device, self.trace, cq_id=0, blocking=False)
        return ttnn.to_torch(self.out).float().view(-1)[:HIDDEN]

    def release(self):
        if self.trace is not None:
            ttnn.release_trace(self.device, self.trace)


def make_inputs(steps, is_global, seed):
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(steps, HIDDEN, generator=g).bfloat16().float() for _ in range(3 if is_global else 1)]
    if not is_global:
        xs = xs * 3
    return dict(zip(("xq", "xk", "xv"), xs))


def summarize(per_step, ranges):
    rows = []
    for lo, hi in ranges:
        seg = per_step[lo:hi]
        rows.append((lo, hi - 1, min(seg), sum(seg) / len(seg)))
    return rows


CASES = {
    # name: (is_global, steps, max_ctx)
    "local": (False, 1100, 2048),  # max_ctx only sizes DecodePositions here
    "global": (True, 1100, 2048),
    "global_long": (True, 8400, 8448),
}


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("cache_dtype", ["bf16", "bfp8"])
@pytest.mark.parametrize("case", list(CASES))
def test_attention_decode(device, case, cache_dtype):
    is_global, steps, max_ctx = CASES[case]
    if case == "global_long" and cache_dtype == "bfp8" and not os.environ.get("INKLING_LONG_BFP8"):
        pytest.skip("long global run only with the bf16 cache (set INKLING_LONG_BFP8=1)")
    sd = random_state_dict(is_global, seed=11 if is_global else 7)
    inputs = make_inputs(steps, is_global, seed=3)
    ref = reference_outputs(sd, inputs, is_global, tag=f"{case}-{steps}")

    dtype = ttnn.bfloat16 if cache_dtype == "bf16" else ttnn.bfloat8_b
    runner = Runner(device, is_global, sd, max_ctx, dtype)
    xs = [inputs[n] for n in ("xq", "xk", "xv")][: 3 if is_global else 1]
    runner.set_inputs(0, [x[0] for x in xs])
    runner.capture()

    per_step, times = [], []
    try:
        for t in range(steps):
            runner.set_inputs(t, [x[t] for x in xs])
            t0 = time.perf_counter()
            ttnn.execute_trace(device, runner.trace, cq_id=0, blocking=True)
            times.append(time.perf_counter() - t0)
            out = ttnn.to_torch(runner.out).float().view(-1)[:HIDDEN]
            per_step.append(pcc(out, ref[t].float()))
    finally:
        runner.release()

    marks = [0, 1, 2, 3, 1023, 1024, 1025, 1099]
    if steps > 8192:
        marks += [2047, 2048, 8191, 8192, 8193, steps - 1]
    ranges = [(0, 1024), (1024, min(steps, 1100))]
    if steps > 1100:
        ranges += [(1100, 8192), (8192, steps)]
    times.sort()
    report = {
        "case": case,
        "cache": cache_dtype,
        "ranges": summarize(per_step, ranges),
        "marks": {m: round(per_step[m], 6) for m in marks if m < steps},
        "step_ms_median": round(1e3 * times[len(times) // 2], 3),
        "min_pcc": min(per_step),
    }
    print("T3RESULT", json.dumps(report))
    assert min(per_step) >= 0.999, report


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("case", ["local", "global"])
def test_trace_replay_deterministic(device, case):
    """Same inputs + same state -> the trace gives bit-identical outputs twice."""
    is_global, _, max_ctx = CASES[case]
    sd = random_state_dict(is_global, seed=5)
    inputs = make_inputs(40, is_global, seed=9)
    runner = Runner(device, is_global, sd, max_ctx, ttnn.bfloat16)
    xs = [inputs[n] for n in ("xq", "xk", "xv")][: 3 if is_global else 1]
    # snapshot buffers must exist before capture: later allocations may alias trace scratch
    saved = [ttnn.clone(s) for s in runner.state.tensors()]
    runner.set_inputs(0, [x[0] for x in xs])
    runner.capture()
    try:
        for t in range(32):  # build some history
            runner.set_inputs(t, [x[t] for x in xs])
            runner.run()
        for src, dst in zip(runner.state.tensors(), saved):
            ttnn.copy(src, dst)
        runner.set_inputs(32, [x[32] for x in xs])
        first = runner.run()
        for src, dst in zip(saved, runner.state.tensors()):
            ttnn.copy(src, dst)
        second = runner.run()
    finally:
        runner.release()
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
