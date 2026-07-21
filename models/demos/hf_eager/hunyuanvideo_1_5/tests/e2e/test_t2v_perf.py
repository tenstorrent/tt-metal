# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Performance (profiler) test for the HunyuanVideo-1.5 't2v' TTNN pipeline.

Runs ONE bounded, in-process forward of the chained TTNN t2v pipeline EXACTLY as
``demo/demo_t2v.py`` does (``run_demo("t2v", args)`` self-opens the sharded mesh,
builds the pipeline on it, and drives the real denoise forward) — but with the
heavy diffusion axes (timesteps / frames / resolution) trimmed to a small,
representative, dispatch-dense size so tracy's marker buffer never overflows.

Everything runs in THIS process (never a subprocess) so every device op is
visible to the profiler. Perf only — no PCC / correctness assertions.
"""

from __future__ import annotations

import os
import time

import ttnn

from models.demos.hf_eager.hunyuanvideo_1_5.demo._common import build_argparser, run_demo

# ---- bounded profiling axes (SMALL representative sizes; env-overridable) --------------------
# The heavy axis for a video-diffusion model is TIMESTEPS x FRAMES x RESOLUTION — trim ALL of
# them small. These are perf sizes, NOT the demo's full-length correctness sizes.
PERF_STEPS = int(os.environ.get("TT_PERF_STEPS", os.environ.get("TT_PERF_MAX_NEW_TOKENS", "2")))
PERF_FRAMES = int(os.environ.get("TT_PERF_FRAMES", "5"))
PERF_H = int(os.environ.get("TT_PERF_HEIGHT", "256"))
PERF_W = int(os.environ.get("TT_PERF_WIDTH", "256"))
PERF_PROMPT = os.environ.get("TT_PERF_PROMPT", "a cat walking")

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))

# source mesh topology (HunyuanVideo-1.5 DiT runs tensor-parallel on a (1, 8) mesh).
_SRC_ROWS = int(os.environ.get("TT_PERF_SRC_ROWS", "1"))
_SRC_COLS = int(os.environ.get("TT_PERF_SRC_COLS", "8"))

# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    # Reserve the trace + 2-CQ budget at device-open, ONCE, for baseline and every candidate: the
    # second queue and the trace region exist before any candidate runs, so trace+2CQ is the fixed
    # measurement mode (never a per-candidate downgrade for lack of a queue). A device/config that
    # genuinely can't open 2 CQs still degrades gracefully in measure_adapter; override with TT_PERF_NUM_CQ.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))


def _dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_args():
    """Parse the t2v argparser with defaults (never sys.argv — pytest owns that)."""
    p = build_argparser("t2v")
    for _argv in ([], [PERF_PROMPT], ["--prompt", PERF_PROMPT]):
        try:
            return p.parse_args(_argv)
        except SystemExit:
            continue
    return p.parse_args([])


def _make_bounded_args():
    """Build the demo's args, then TRIM the heavy diffusion axes to small perf sizes."""
    args = _parse_args()
    for _name, _val in (
        ("num_inference_steps", PERF_STEPS),
        ("infer_steps", PERF_STEPS),
        ("inference_steps", PERF_STEPS),
        ("steps", PERF_STEPS),
        ("num_steps", PERF_STEPS),
        ("denoise_steps", PERF_STEPS),
        ("num_frames", PERF_FRAMES),
        ("video_length", PERF_FRAMES),
        ("frames", PERF_FRAMES),
        ("length", PERF_FRAMES),
        ("height", PERF_H),
        ("width", PERF_W),
        ("resolution", (PERF_H, PERF_W)),
        ("prompt", PERF_PROMPT),
    ):
        if hasattr(args, _name):
            try:
                setattr(args, _name, _val)
            except Exception:
                pass
    return args


def test_t2v_perf():
    # DEVICE OPEN — the demo's run_demo("t2v", args) self-opens the sharded mesh EXACTLY as the
    # source does, builds the pipeline on it, and drives the real t2v denoise forward. We do NOT use
    # a single `device` fixture (that would disable sharding and profile the wrong single-chip
    # config). We monkeypatch ttnn's mesh-open so we can (a) drain the profiler on the RIGHT device
    # during the bounded forward and (b) capture the EXACT MeshShape run_demo opened, then reuse that
    # same shape for the trace pass — guaranteeing the trace runs the same valid tensor-parallel
    # topology the eager forward ran (avoiding a mis-planned shape whose tp does not divide heads).
    counter = [0]
    _orig = []
    _dev_holder = []
    _shape_holder = []
    _orig_opens = []

    def _mk_capture(orig):
        def _cap(*a, **k):
            d = orig(*a, **k)
            try:
                _dev_holder.append(d)
                _shape = None
                if a:
                    _shape = a[0]
                elif "mesh_shape" in k:
                    _shape = k["mesh_shape"]
                if _shape is not None:
                    _shape_holder.append(_shape)
            except Exception:
                pass
            return d

        return _cap

    for _oname in ("open_mesh_device", "open_device"):
        _of = getattr(ttnn, _oname, None)
        if callable(_of):
            _orig_opens.append((_oname, _of))
            setattr(ttnn, _oname, _mk_capture(_of))

    # Drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
    # operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter tracks
    # TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/transpose/
    # reduction slip through) and the 12000-marker buffer overflows on some device, dropping ops ->
    # non-reproducible device_ms. Wrapping by TYPE never misses an op.
    def _draining(fn):
        def inner(*a, **k):
            r = fn(*a, **k)
            counter[0] += 1
            if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                dev = _dev_holder[-1] if _dev_holder else None
                if dev is not None:
                    try:
                        ttnn.ReadDeviceProfiler(dev)
                    except Exception:
                        pass
            return r

        return inner

    _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
    for _mod in [_m for _m in _mods if _m is not None]:
        for _n in dir(_mod):
            _op = getattr(_mod, _n, None)
            if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                _orig.append((_mod, _n, _op))
                setattr(_mod, _n, _draining(_op))

    out = None
    _fw0 = time.monotonic()
    try:
        # Build + run the t2v pipeline EXACTLY as demo/demo_t2v.py does, but BOUNDED: run_demo
        # self-opens the sharded mesh, builds the pipeline, constructs the inputs the forward needs,
        # and runs the denoise forward — with timesteps / frames / resolution trimmed small.
        args = _make_bounded_args()
        out = run_demo("t2v", args)
        dev = _dev_holder[-1] if _dev_holder else None
        if dev is not None:
            try:
                ttnn.ReadDeviceProfiler(dev)
            except Exception:
                pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
        for _oname, _of in _orig_opens:
            setattr(ttnn, _oname, _of)
    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))

    # perf only — NO PCC. A completed forward is proven by real device dispatch (or a returned output).
    assert out is not None or counter[0] > 0

    if _PERF_TRACE:
        try:
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import (
                PipelineStageAdapter,
                resolve_mesh_shape,
            )

            def _build_for_perf(dev):
                from models.demos.hf_eager.hunyuanvideo_1_5.tt.pipeline import build_pipeline

                # Return the RESIDENT, stage-exposing pipeline (carries PIPELINE_STAGES=["denoise"]
                # + per-stage trace hooks) built on the SAME sharded mesh — so the trace profiles the
                # same topology the eager forward ran.
                return build_pipeline(dev)

            _prompt_ids = PERF_PROMPT

            # Prefer the EXACT MeshShape run_demo opened (it produced a valid tensor-parallel
            # factoring for the eager forward), then the tool-planned shape, then the source shape,
            # then the head-divisible fallbacks. Try each: open + build + measure, break on the first
            # that engages a trace. This shrinks past a mis-planned shape whose tp does not divide the
            # head count instead of aborting.
            _candidates = []
            for _s in _shape_holder[::-1]:
                _candidates.append(_s)
            try:
                _rr, _cc = resolve_mesh_shape(default_rows=_SRC_ROWS, default_cols=_SRC_COLS)
            except Exception:
                _rr, _cc = _SRC_ROWS, _SRC_COLS
            for _r, _c in _dedupe(
                [(_rr, _cc), (_SRC_ROWS, _SRC_COLS), (1, 8), (1, 4), (1, 2), (1, 1)]
            ):
                _candidates.append(ttnn.MeshShape(_r, _c))

            _traced = False
            for _shape in _candidates:
                trace_dev = None
                try:
                    trace_dev = ttnn.open_mesh_device(_shape, **_DEV_PARAMS)
                    # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry
                    # gets traced (+2CQ where the stage stages its inputs). Falls back to the single
                    # decode contract for pipelines that expose only decode_step.
                    _adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1)
                    measure_adapter(_adapter, trace_dev, mode="auto")
                    _traced = True
                except Exception as _te:  # noqa: BLE001
                    print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
                finally:
                    if trace_dev is not None:
                        try:
                            ttnn.close_mesh_device(trace_dev)
                        except Exception:
                            pass
                if _traced:
                    break
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)