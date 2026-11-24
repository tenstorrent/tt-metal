# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""`attention_fast.py` (sdpa_decode core, local + global) vs the HF reference.

- the cases of test_attention.py (same weights, inputs and cached HF refs): local / global 0..1099,
  global_long 0..8399 (covers 0-3, 1022-1025, 8192).
- seeded cases at 9000, 32K, 120K (local and global, max_ctx 128K): P synthetic post-norm k/v are
  loaded into HF's DynamicCache and into the device state (conv history zero on both sides), then
  4 tokens are decoded. HF runs in the reference python (tests/hf_ref.py); refs are cached.
- trace replay bit-identical; per-use time vs v1 at max_ctx 4K and 128K.

    pytest -q -s models/demos/inkling2_mini/tests/test_attention_fast.py
"""

import json
import os
import sys
import tempfile
import time

import torch

HIDDEN = 2304
LOCAL_LAYER, GLOBAL_LAYER = 0, 5
MAX_CTX = 128 * 1024
SEEDED_POS = {"9000": 9000, "32k": 32768, "120k": 120 * 1024}
SEEDED_STEPS = 4


def run_seeded_reference(in_path, out_path):
    """HF decode of 4 tokens at P, P+1, ... after loading the prefix k/v into the cache."""
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
        create_sliding_window_causal_mask,
    )
    from transformers.models.inkling2_loop import modeling_inkling2_loop as M
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

    torch.set_num_threads(int(os.environ.get("INKLING_REF_THREADS", "4")))
    data = torch.load(in_path)
    is_global = data["is_global"]
    cfg = Inkling2LoopTextConfig(**json.load(open(os.environ["INKLING_CONFIG"]))["text_config"])
    cfg._attn_implementation = "eager"
    layer = GLOBAL_LAYER if is_global else LOCAL_LAYER
    attn = (M.Inkling2LoopFullAttention if is_global else M.Inkling2LoopSlidingAttention)(cfg, layer)
    attn.load_state_dict(data["state_dict"])
    attn.to(torch.bfloat16)
    if not is_global:  # `_keep_in_fp32_modules_strict`: sconv.conv1d
        attn.k_proj.sconv.conv1d.float()
        attn.v_proj.sconv.conv1d.float()
    attn.eval()

    cache = DynamicCache(config=cfg)
    pk, pv = (data[n].to(torch.bfloat16).transpose(0, 1).unsqueeze(0) for n in ("pk", "pv"))  # [1, 8, P, 128]
    cache.update(pk, pv, layer)
    p = pk.shape[2]
    xq, xk, xv = (data[n].bfloat16() for n in ("xq", "xk", "xv"))
    outs = []
    with torch.no_grad():
        for i in range(xq.shape[0]):
            x = xq[i].view(1, 1, HIDDEN)
            pid = torch.tensor([[p + i]])
            mk = dict(config=cfg, inputs_embeds=x, attention_mask=None, cache=cache, position_ids=pid)
            mask = create_causal_mask(**mk) if is_global else create_sliding_window_causal_mask(**mk)
            kw = dict(
                hidden_states=x,
                attention_mask=mask,
                relative_distance=M.get_relative_distance(x, cache, is_sliding=not is_global),
                conv_mask=create_recurrent_attention_mask(**mk),
                cache=cache,
                position_ids=pid,
            )
            if is_global:
                kw.update(key_hidden_states=xk[i].view(1, 1, HIDDEN), value_hidden_states=xv[i].view(1, 1, HIDDEN))
            outs.append(attn(**kw)[0].view(HIDDEN))
    torch.save(torch.stack(outs), out_path)


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "ref":
    run_seeded_reference(sys.argv[2], sys.argv[3])
    sys.exit(0)

# ------------------------------------------------------------------ device side
import pytest  # noqa: E402

import models.demos.inkling2_mini.tests.test_attention as ta  # noqa: E402
import ttnn  # noqa: E402
from models.demos.inkling2_mini.tests import hf_ref  # noqa: E402
from models.demos.inkling2_mini.tt.attention import (  # noqa: E402
    QK_CLIP,
    DecodePositions,
    Inkling2Attention,
    host_weights_from_hf,
)
from models.demos.inkling2_mini.tt.attention_fast import DecodePositionsFast, Inkling2AttentionFast  # noqa: E402

TRACE_REGION = 64 * 1024 * 1024
DEVICE_PARAMS = [{"trace_region_size": TRACE_REGION}]
FAST = (Inkling2AttentionFast, DecodePositionsFast)
V1 = (Inkling2Attention, DecodePositions)


class Runner(ta.Runner):
    """ta.Runner with a choice of attention / positions class."""

    def __init__(self, device, is_global, sd, max_ctx, cache_dtype, classes=FAST):
        attn_cls, pos_cls = classes
        self.device = device
        self.is_global = is_global
        self.attn = attn_cls(device, host_weights_from_hf(sd, is_global), is_global)
        self.state = self.attn.new_state(max_ctx, cache_dtype)
        self.positions = pos_cls(device, max_ctx)
        zero = torch.zeros(HIDDEN)
        self.x = [ttnn.to_device(ta._x_host(zero), device) for _ in range(3 if is_global else 1)]
        self.trace = None
        self.out = None


def _dtype(name):
    return ttnn.bfloat16 if name == "bf16" else ttnn.bfloat8_b


def _xs(inputs, is_global):
    return [inputs[n] for n in ("xq", "xk", "xv")][: 3 if is_global else 1]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("cache_dtype", ["bf16", "bfp8"])
@pytest.mark.parametrize("case", list(ta.CASES))
def test_fast_decode(device, case, cache_dtype):
    is_global, steps, max_ctx = ta.CASES[case]
    sd = ta.random_state_dict(is_global, seed=11 if is_global else 7)
    inputs = ta.make_inputs(steps, is_global, seed=3)
    ref = ta.reference_outputs(sd, inputs, is_global, tag=f"{case}-{steps}")
    runner = Runner(device, is_global, sd, max_ctx, _dtype(cache_dtype))
    xs = _xs(inputs, is_global)
    runner.set_inputs(0, [x[0] for x in xs])
    runner.capture()
    per_step = []
    try:
        for t in range(steps):
            runner.set_inputs(t, [x[t] for x in xs])
            per_step.append(ta.pcc(runner.run(), ref[t].float()))
    finally:
        runner.release()
    marks = [m for m in (0, 1, 2, 3, 1022, 1023, 1024, 1025, 8191, 8192, 8193) if m < steps]
    report = {
        "case": case,
        "cache": cache_dtype,
        "min": round(hf_ref.finite_min(per_step), 6),
        "marks": {m: round(per_step[m], 6) for m in marks},
    }
    print("ATTN-FAST-RESULT", json.dumps(report))
    assert hf_ref.finite_min(per_step) >= 0.999, report


def seeded_prefix(p, seed):
    """Synthetic post-norm prefix [P, 8, 128], bf16 values: keys L2 norm in [0.5, 1] * clip, v RMS 1."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(p, 8, 128, generator=g)
    k = k / k.norm(dim=-1, keepdim=True) * (QK_CLIP * (0.5 + 0.5 * torch.rand(p, 8, 1, generator=g)))
    v = torch.randn(p, 8, 128, generator=g)
    v = v * torch.rsqrt(v.square().mean(-1, keepdim=True))
    return k.bfloat16().float(), v.bfloat16().float()


def seeded_case(is_global, p):
    sd = ta.random_state_dict(is_global, seed=11 if is_global else 7)
    inputs = ta.make_inputs(SEEDED_STEPS, is_global, seed=p)
    pk, pv = seeded_prefix(p, seed=p + 1)
    hf_ref.require(config=True)
    cache = hf_ref.cache_dir("attention_fast")
    out_path = cache / f"{'global' if is_global else 'local'}_{p}.pt"
    if not out_path.exists():
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            in_path = os.path.join(tmp, "in.pt")
            torch.save({"state_dict": sd, "is_global": is_global, "pk": pk, "pv": pv, **inputs}, in_path)
            hf_ref.run(__file__, "ref", in_path, os.path.join(tmp, "out.pt"))
            os.replace(os.path.join(tmp, "out.pt"), out_path)
    return sd, inputs, pk, pv, torch.load(out_path)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("cache_dtype", ["bf16", "bfp8"])
@pytest.mark.parametrize("at", list(SEEDED_POS))
@pytest.mark.parametrize("is_global", [False, True], ids=["local", "global"])
def test_fast_decode_seeded(device, is_global, at, cache_dtype):
    """Positions P..P+3 after a P-token seeded prefix, max_ctx 128K."""
    p = SEEDED_POS[at]
    sd, inputs, pk, pv, ref = seeded_case(is_global, p)
    runner = Runner(device, is_global, sd, MAX_CTX, _dtype(cache_dtype))
    xs = _xs(inputs, is_global)
    runner.set_inputs(p, [x[0] for x in xs])
    runner.capture()
    runner.state.fill(pk, pv)
    per_step = {}
    try:
        for i in range(SEEDED_STEPS):
            runner.set_inputs(p + i, [x[i] for x in xs])
            per_step[p + i] = round(ta.pcc(runner.run(), ref[i].float()), 6)
    finally:
        runner.release()
    print(
        "ATTN-FAST-RESULT",
        json.dumps({"case": f"{'global' if is_global else 'local'}_{at}", "cache": cache_dtype, "pcc": per_step}),
    )
    assert hf_ref.finite_min(per_step.values()) >= 0.999, per_step


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("is_global", [False, True], ids=["local", "global"])
def test_fast_trace_replay_deterministic(device, is_global):
    """Same inputs + same state -> bit-identical outputs, at a step past the 1024 boundary."""
    sd = ta.random_state_dict(is_global, seed=5)
    steps = 1040
    inputs = ta.make_inputs(steps + 1, is_global, seed=9)
    runner = Runner(device, is_global, sd, 2048, ttnn.bfloat16)
    xs = _xs(inputs, is_global)
    saved = [ttnn.clone(s) for s in runner.state.tensors()]  # before capture (trace scratch aliasing)
    runner.set_inputs(0, [x[0] for x in xs])
    runner.capture()
    try:
        for t in range(steps):
            runner.set_inputs(t, [x[t] for x in xs])
            runner.run()
        for src, dst in zip(runner.state.tensors(), saved):
            ttnn.copy(src, dst)
        runner.set_inputs(steps, [x[steps] for x in xs])
        first = runner.run()
        for src, dst in zip(saved, runner.state.tensors()):
            ttnn.copy(src, dst)
        second = runner.run()
    finally:
        runner.release()
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_far_bfp4_rows(device, monkeypatch):
    """bfp4 far cache (FAR_DTYPE opt-in, bfp8 rings): the rows decode writes equal the host bfp4 rounding of the
    ring's bfp8 rows ([v | 1, 0 x 31 | k]); the neighbouring rows of the same tiles keep their values."""
    import models.demos.inkling2_mini.tt.attention_fast as AF

    monkeypatch.setattr(AF, "FAR_DTYPE", ttnn.bfloat4_b)
    sd = ta.random_state_dict(True, seed=11)
    p, steps = 3001, 3  # rows 3001..3003: odd / even rows of one 16 B mantissa block, mid tile
    inputs = ta.make_inputs(steps, True, seed=4)
    runner = Runner(device, True, sd, 4096, ttnn.bfloat8_b)
    xs = _xs(inputs, True)
    runner.set_inputs(p, [x[0] for x in xs])
    runner.capture()
    pk, pv = seeded_prefix(p, seed=2)
    runner.state.fill(pk, pv)
    st = runner.state
    assert st.kv_full.dtype == ttnn.bfloat4_b
    before = ttnn.to_torch(st.kv_full).float()[0]
    try:
        for i in range(steps):
            runner.set_inputs(p + i, [x[i] for x in xs])
            runner.run()
    finally:
        runner.release()
    after = ttnn.to_torch(st.kv_full).float()[0]
    bfp4 = lambda t: ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa
    k = ttnn.to_torch(st.k_ring).float()[0]
    v = ttnn.to_torch(st.v_ring).float()[0]
    rows = list(range(p, p + steps))
    slots = [r % 1024 for r in rows]
    mass = torch.zeros(8, steps, 32)
    mass[..., 0] = 1.0
    mid = [] if AF.FAR_UNIT else [mass]  # the stored mass block of a build without unit_k_tile_col
    want = bfp4(torch.cat([v[:, slots], *mid, k[:, slots]], dim=-1).unsqueeze(0))[0]
    assert torch.equal(after[:, rows], want)
    keep = [r for r in range(p - 32, p + 32) if r not in rows]
    assert torch.equal(after[:, keep], before[:, keep])


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_far_split_rows(device, monkeypatch):
    """far keys and values in two caches (FAR_SPLIT opt-in, bfp8 rings): decode writes the ring's k row to the key
    cache as it is and the host bfp4 rounding of the ring's v row to the value cache; the neighbouring rows keep
    their values. The far call on the two caches gives finite output (its values: split_probe in the L3 report)."""
    import models.demos.inkling2_mini.tt.attention_fast as AF

    if not hasattr(ttnn.SDPAProgramConfig, "unit_v_tile_col"):
        pytest.skip("this build has no sdpa unit_v_tile_col")
    monkeypatch.setattr(AF, "FAR_SPLIT", True)
    monkeypatch.setattr(AF.FA, "FAR_Q", (4, 0))
    sd = ta.random_state_dict(True, seed=11)
    p, steps = 3001, 3
    inputs = ta.make_inputs(steps, True, seed=4)
    runner = Runner(device, True, sd, 4096, ttnn.bfloat8_b)
    xs = _xs(inputs, True)
    runner.set_inputs(p, [x[0] for x in xs])
    runner.capture()
    pk, pv = seeded_prefix(p, seed=2)
    runner.state.fill(pk, pv)
    st = runner.state
    assert st.v_full.dtype == ttnn.bfloat4_b and st.kv_full.shape[-1] == 128
    before = [ttnn.to_torch(t).float()[0] for t in (st.kv_full, st.v_full)]
    try:
        for i in range(steps):
            runner.set_inputs(p + i, [x[i] for x in xs])
            out = runner.run()
    finally:
        runner.release()
    assert torch.isfinite(out).all()
    after = [ttnn.to_torch(t).float()[0] for t in (st.kv_full, st.v_full)]
    bfp4 = lambda t: ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa
    rows = list(range(p, p + steps))
    slots = [r % 1024 for r in rows]
    k = ttnn.to_torch(st.k_ring).float()[0][:, slots]
    v = ttnn.to_torch(st.v_ring).float()[0][:, slots]
    assert torch.equal(after[0][:, rows], k)
    assert torch.equal(after[1][:, rows], bfp4(v.unsqueeze(0))[0])
    keep = [r for r in range(p - 32, p + 32) if r not in rows]
    for a, b in zip(after, before):
        assert torch.equal(a[:, keep], b[:, keep])


def _median_ms(device, trace, n=15):
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
        samples.append(time.perf_counter() - t0)
    return 1e3 * sorted(samples)[n // 2]


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("impl", ["v1", "fast"])
@pytest.mark.parametrize("max_ctx", [4096, MAX_CTX], ids=["4k", "128k"])
@pytest.mark.parametrize("is_global", [False, True], ids=["local", "global"])
def test_use_time(device, is_global, max_ctx, impl):
    """Benchmark (perf marker), not a correctness test. Device time of one layer use = traced (prepare + decode) - traced prepare; bfp8 cache."""
    sd = ta.random_state_dict(is_global, seed=5)
    runner = Runner(device, is_global, sd, max_ctx, ttnn.bfloat8_b, V1 if impl == "v1" else FAST)
    xs = [x[0] for x in _xs(ta.make_inputs(1, is_global, seed=1), is_global)]
    runner.set_inputs(0, xs)
    runner.capture()
    prep = runner.positions
    prep.prepare()
    prep_trace = ttnn.begin_trace_capture(device, cq_id=0)
    prep.prepare()
    ttnn.end_trace_capture(device, prep_trace, cq_id=0)
    positions = [2048, max_ctx - 1] if max_ctx == 4096 else [0, 8192, 32768, 65536, 122880, max_ctx - 1]
    times = {}
    try:
        for pos in positions:
            runner.set_inputs(pos, xs)
            runner.run()
            times[pos] = round(_median_ms(device, runner.trace) - _median_ms(device, prep_trace), 3)
        prep_ms = round(_median_ms(device, prep_trace), 3)
    finally:
        ttnn.release_trace(device, prep_trace)
        runner.release()
    assert all(0 < t < 1e3 for t in list(times.values()) + [prep_ms]), times
    kind = "global" if is_global else "local"
    print(
        "P1BTIME", json.dumps({"kind": kind, "max_ctx": max_ctx, "impl": impl, "use_ms": times, "prepare_ms": prep_ms})
    )
