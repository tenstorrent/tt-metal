# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Performance (profiler) test for the ``i2v`` HunyuanVideo-1.5 TTNN pipeline.

Runs the SAME chained TTNN i2v denoise forward as ``demo/demo_i2v.py`` (via the
shared ``run_demo`` entrypoint), but BOUNDED and profiler-safe: the heavy video
axes are trimmed, the DiT depth is capped via ``TT_PERF_LAYERS``, and every
dispatched ttnn op drains the device profiler so tracy's marker buffer never
overflows. The device forward runs IN-PROCESS (never shelled out) so tracy can
see every op. No PCC / correctness assertion — perf only.
"""

from __future__ import annotations

import os
import time

import ttnn

# Lift the demo's build + run entrypoint straight from demo/demo_i2v.py.
from models.demos.hf_eager.hunyuanvideo_1_5.demo._common import build_argparser, run_demo

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))

# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# Trim the HEAVY axes for a bounded, dispatch-representative pass. This is a diffusion VIDEO model, so
# the op/dispatch count is driven by FRAMES + spatial resolution (and, if present, the denoise step
# count) — NOT by any token length. We do NOT reuse the demo's full-length production shapes: a
# full-length forward under tracy runs orders of magnitude slower and stalls in synchronize_device.
# Applied defensively (hasattr-guarded) so unknown arg names are simply ignored.
_PERF_SHRINK = {
    "num_inference_steps": int(os.environ.get("TT_PERF_STEPS", "1")),
    "steps": int(os.environ.get("TT_PERF_STEPS", "1")),
    "num_denoise_steps": int(os.environ.get("TT_PERF_STEPS", "1")),
    "num_frames": int(os.environ.get("TT_PERF_FRAMES", "5")),
    "frames": int(os.environ.get("TT_PERF_FRAMES", "5")),
    "video_length": int(os.environ.get("TT_PERF_FRAMES", "5")),
    "height": int(os.environ.get("TT_PERF_HEIGHT", "256")),
    "width": int(os.environ.get("TT_PERF_WIDTH", "256")),
    "num_videos_per_prompt": 1,
    "batch": 1,
    "batch_size": 1,
}

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    # Reserve the trace + 2-CQ budget at device-open, ONCE, so trace+2CQ is the fixed measurement mode
    # (never a per-candidate downgrade for lack of a queue). A device/config that genuinely can't open
    # 2 CQs still degrades gracefully in measure_adapter; override with TT_PERF_NUM_CQ.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))

# The DiT shards its 16 attention heads across the TP dimension, so the TP width MUST divide 16. The
# perf planner can hand back a wider mesh than this 16-head model can shard (e.g. tp=24 -> "heads_total=16
# not divisible by tp=24"); the largest head-dividing TP the source is proven at is 8 (the demo's (1,8)).
_DIT_HEADS = 16


def _resolve_i2v_mesh():
    # Honor the tool's topology (--devices/--mesh reshape via resolve_mesh_shape; env unset -> source's
    # own (1,8)), but clamp the TP (column) width to a divisor of the head count so the DiT can actually
    # shard, and keep DP=1 (the proven i2v layout). This adapts to however many chips the planner
    # provisioned without hardcoding a single fixed shape, while never asking the model to shard 16
    # heads across a non-dividing TP.
    from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape

    rows, cols = resolve_mesh_shape(default_rows=1, default_cols=8)
    total = max(1, rows * cols)
    tp = max(c for c in (8, 4, 2, 1) if _DIT_HEADS % c == 0 and total >= c)
    return 1, tp


def _build_args():
    # Same argparser the demo uses (build_argparser("i2v")); parse with NO CLI overrides ([]) so pytest's
    # own argv is ignored and we get the demo's defaults, then trim the heavy axes for a bounded pass.
    args = build_argparser("i2v").parse_args([])
    for _k, _v in _PERF_SHRINK.items():
        if hasattr(args, _k):
            try:
                setattr(args, _k, _v)
            except Exception:  # noqa: BLE001
                pass
    return args


def test_i2v_perf():
    # DEVICE OPEN — match the source topology exactly. The demo self-opens a sharded DiT mesh (default
    # (1,8)); we lift that open here and reuse ONE device for BOTH the eager forward and the trace-replay
    # measure. Opening a second mesh (as a naive trace block would) fights for the same mmio chips and
    # wedges the run — so there is exactly one open/close. The TP width is clamped to divide the 16 heads.
    rows, cols = _resolve_i2v_mesh()
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **_DEV_PARAMS)

    try:
        # --- profiler drain: wrap EVERY ttnn op (type 'FastOperation') across ttnn + its op submodules,
        # so the flush counter tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts
        # (sdpa/eltwise/transpose/reduction slip through) and the 12000-marker buffer overflows, dropping
        # ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
        counter = [0]
        _orig = []

        def _draining(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                counter[0] += 1
                if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                    try:
                        ttnn.ReadDeviceProfiler(device)  # captured mesh_device on multi-chip
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

        # --- inject OUR self-opened mesh into the demo's device-open path so run_demo runs its real
        # forward on the correct sharded topology WITHOUT opening (or closing) a second device. We patch
        # ttnn's mesh open/close (the leaf every helper funnels through) + the same names in the demo's
        # _common module in case it imported them by name.
        import models.demos.hf_eager.hunyuanvideo_1_5.demo._common as _common_mod

        _patched = []

        def _patch(_mod, _name, _val):
            if hasattr(_mod, _name):
                _patched.append((_mod, _name, getattr(_mod, _name)))
                setattr(_mod, _name, _val)

        _ret_dev = lambda *a, **k: device  # noqa: E731
        _noop = lambda *a, **k: None  # noqa: E731
        for _tgt in (ttnn, _common_mod):
            for _nm in ("open_mesh_device", "open_device", "CreateMeshDevice"):
                _patch(_tgt, _nm, _ret_dev)
            for _nm in ("close_mesh_device", "close_device", "CloseMeshDevice"):
                _patch(_tgt, _nm, _noop)

        _fw0 = time.monotonic()
        out = None
        try:
            # Run ONE bounded i2v denoise forward EXACTLY as demo/demo_i2v.py does (demo's own input
            # construction + code path), on our device, DiT depth capped via TT_PERF_LAYERS, heavy axes
            # trimmed. run_demo prints the velocity/PCC; completion is the output for this perf run.
            args = _build_args()
            _res = run_demo("i2v", args)
            out = _res if _res is not None else "ok"
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
            for _mod, _n, _f in _patched:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter
                from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

                def _build_for_perf(dev):
                    # Return the RESIDENT, stage-exposing pipeline (PIPELINE_STAGES = ["denoise"] + trace
                    # hooks) built on the passed-in device, so the trace runs the SAME sharded topology as
                    # the eager forward above (same mesh handle -> no second open).
                    from models.demos.hf_eager.hunyuanvideo_1_5.tt.pipeline import build_pipeline

                    return build_pipeline(dev)

                _prompt_ids = "a cat"
                # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
                # traced (+2CQ where the stage stages its inputs). Falls back to the single decode
                # contract for pipelines that expose only decode_step.
                _adapter = PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        try:
            ttnn.close_mesh_device(device)
        except Exception:
            pass