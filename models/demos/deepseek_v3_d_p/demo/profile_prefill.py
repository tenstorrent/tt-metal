# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Profile one warm prefill forward of Mistral Small 4, on device and on the wall clock.

The prefill-only demo re-runs a whole prefill per generated token, so a single forward IS the
per-token latency. This module measures it two ways, because the two answer different questions
and only one of them is trustworthy on a shared box:

``test_ops``       device-op breakdown from the tracy profiler. DEVICE KERNEL DURATION comes from
                   hardware counters, so it is immune to host noise, host-side program compilation,
                   and other users' CPU load. This is the number to reason about.
``test_walltime``  end-to-end wall clock, repeated, reporting min. Only meaningful compared against
                   the device total from ``test_ops``: wall_clock - device_total is the host share.

Two earlier attempts at this got wrong answers; the traps are worth stating so they are not
re-entered.

* **You cannot detect host-bound-ness by enqueuing many forwards before one synchronize.**
  ``TtPrefillTransformer.forward`` ends with ``logit_to_host()`` -- a blocking D2H read -- and then
  samples on the host, so the forward self-synchronizes and never pipelines. That measurement
  returns enqueue-time == total-time regardless of where the time goes, which reads exactly like
  "the device is saturated" and is not.
* **Process-to-process wall-clock variance is ~25% at a FIXED configuration.** Window 512 /
  actual_isl 64 / pad tail measured min 1401 ms in one process and 1760 ms in another. That is the
  single most important fact about measuring this: it swamps essentially every effect worth chasing.
  Hypotheses that died inside this noise band, each of which looked real for a while: cost scaling
  with the window, cost scaling with ``actual_isl``, a ~288 ms first-touch ``build_padding_config``
  penalty (it reproduced with the opposite sign), a ~30% penalty for filling the pad tail with
  random tokens, and a large win from suppressing the ~1539 DEBUG log lines per forward. None is
  established. Do not re-derive them from wall clock; use ``test_ops``.
* ``_make_window`` still reproduces the demo's pad-tail layout, on the principle that the harness
  should match what it claims to measure -- not because the tail was shown to matter.

Also: per the profiler's own accounting rules, read device times PER DEVICE, not summed across the
mesh -- all 32 devices run concurrently, so latency is what one device does and summing inflates
everything 32x.

Run the device-op breakdown (the signposted region only)::

    cd /data/kmabee/tt-metal && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
    export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
    export TT_MISTRAL4_PREFILL_TTNN_CACHE=/home/kmabee/mistral4_ttnn_cache
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 \
      ./python_env/bin/python -m tracy -p -r -v -m pytest \
      models/demos/deepseek_v3_d_p/demo/profile_prefill.py::test_ops -s

then summarize the CSV it leaves under ``generated/profiler/reports/``::

    ./python_env/bin/python models/demos/deepseek_v3_d_p/tests/analyze_ops_perf.py <report.csv>

Wall clock (no profiler, so timings are real)::

    PROFILE_SEQ_LEN=512 ./python_env/bin/pytest \
      models/demos/deepseek_v3_d_p/demo/profile_prefill.py::test_walltime -s
"""

import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.mistral_small4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode

try:
    from tracy import signpost
except ImportError:  # profiler build absent: keep the module runnable for the wall-clock path

    def signpost(*_a, **_k):
        pass


PROFILE_SEQ_LEN = int(os.environ.get("PROFILE_SEQ_LEN", 512))
# Layers this profile builds. 36 = the whole model (default, the single-rank numbers). 9 = 36/4,
# i.e. ONE PP=4 stage — pair with mesh-8x1 to time the stage geometry the PP proposal targets.
PROFILE_NUM_LAYERS = int(os.environ.get("PROFILE_NUM_LAYERS", 36))
PROFILE_ISL = int(os.environ.get("PROFILE_ISL", 64))  # real tokens, like a short chat prompt
WARMUP = int(os.environ.get("PROFILE_WARMUP", 3))
REPS = int(os.environ.get("PROFILE_REPS", 8))
PAD_ID = 11


def _make_window(isl_total, actual_isl):
    """The demo's exact buffer layout: real tokens then a PAD tail.

    Not random-filled. A random tail routes to all 128 experts and costs ~30% more than the pad
    tail the demo actually has, which is enough to invalidate a comparison against the live demo.
    """
    w = torch.full((1, isl_total), PAD_ID, dtype=torch.int64)
    w[0, :actual_isl] = torch.randint(0, 1000, (actual_isl,), dtype=torch.int64)
    return w


def _forward_fn(ctx):
    transformer, mesh_device = ctx["transformer"], ctx["mesh_device"]
    sp_factor, isl_per_chip = ctx["sp_factor"], ctx["isl_per_chip"]

    def run(window, actual_isl):
        tt = ttnn.from_torch(
            window.reshape(sp_factor, 1, isl_per_chip),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
        )
        transformer(
            tt,
            ctx["kvpe_cache"],
            actual_isl=actual_isl,
            return_intermediates=False,
            read_profiler=False,
            temperature=0.0,
            index_kv_cache=ctx["index_kv_cache"],
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(tt)

    return run


def ops_hook(**ctx):
    """Warm up, then bracket exactly ONE forward in signposts for the profiler to slice."""
    run = _forward_fn(ctx)
    isl_total = ctx["isl_total"]
    window = _make_window(isl_total, PROFILE_ISL)

    logger.info(f"profiling window={isl_total} actual_isl={PROFILE_ISL}; {WARMUP} warmup forwards first")
    for i in range(WARMUP):
        t0 = time.time()
        run(window, PROFILE_ISL)
        logger.info(f"  warmup {i}: {(time.time()-t0)*1000:.0f} ms")

    # Only the region between these two signposts is the measured forward. Device kernel durations
    # are hardware counters, so they stay valid even though the profiler inflates wall clock here.
    logger.info("=== signposted measured forward ===")
    signpost("start")
    t0 = time.time()
    run(window, PROFILE_ISL)
    dt = time.time() - t0
    signpost("stop")
    logger.info(f"measured forward wall clock UNDER PROFILER (inflated, do not quote): {dt*1000:.0f} ms")
    logger.info("now summarize the CSV with models/demos/deepseek_v3_d_p/tests/analyze_ops_perf.py")


def logging_ab_hook(**ctx):
    """Is the ~1539 DEBUG lines/forward a real cost? A/B/A/B *inside one process*.

    The device profiler settled that this forward is ~95% host-bound (80 ms of device kernel time
    against a 1760 ms wall clock, 2316 op invocations per forward), which makes host-side Python work
    the thing worth attacking -- and there are ~0.66 DEBUG lines per op.

    Interleaved A/B/A/B in a SINGLE process on purpose: process-to-process variance is ~25% here, so
    two separate runs cannot resolve a 20% effect, which is how an earlier attempt "measured" a
    nonsensical -68%. Within one process the conditions alternate, so slow drift cancels.
    """
    import sys as _sys

    run = _forward_fn(ctx)
    isl_total = ctx["isl_total"]
    window = _make_window(isl_total, PROFILE_ISL)

    def block(n):
        out = []
        for _ in range(n):
            t0 = time.time()
            run(window, PROFILE_ISL)
            out.append(time.time() - t0)
        return out

    def set_level(level):
        logger.remove()
        return logger.add(_sys.stderr, level=level)

    for _ in range(WARMUP):
        run(window, PROFILE_ISL)

    reps = max(3, REPS // 2)
    debug_s, quiet_s = [], []
    for cycle in range(2):
        set_level("DEBUG")
        debug_s += block(reps)
        set_level("WARNING")
        quiet_s += block(reps)
    set_level("INFO")  # restore so the summary below is visible

    d_lo, q_lo = min(debug_s), min(quiet_s)
    d_mean, q_mean = statistics.mean(debug_s), statistics.mean(quiet_s)
    logger.info("=== loguru DEBUG vs WARNING, interleaved A/B/A/B in one process ===")
    logger.info(f"  DEBUG   : min {d_lo*1000:8.1f} ms  mean {d_mean*1000:8.1f} ms  n={len(debug_s)}")
    logger.info(f"  WARNING : min {q_lo*1000:8.1f} ms  mean {q_mean*1000:8.1f} ms  n={len(quiet_s)}")
    logger.info(
        f"  saved   : {(d_lo-q_lo)*1000:8.1f} ms on min ({(d_lo-q_lo)/d_lo*100:+.1f}%), "
        f"{(d_mean-q_mean)*1000:.1f} ms on mean ({(d_mean-q_mean)/d_mean*100:+.1f}%)"
    )
    logger.info("  (device kernel time is ~80 ms/forward, so anything saved here is pure host time)")
    out = os.environ.get("PROFILE_OUT", f"/data/kmabee/mistral4_repro_logs/34_logging_ab_{isl_total}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "window": isl_total,
                "actual_isl": PROFILE_ISL,
                "debug": debug_s,
                "quiet": quiet_s,
                "debug_min": d_lo,
                "quiet_min": q_lo,
            },
            f,
            indent=1,
        )
    logger.info(f"wrote {out}")


def walltime_hook(**ctx):
    """Wall clock, no profiler. min over REPS; the mean absorbs other users' interference."""
    run = _forward_fn(ctx)
    isl_total = ctx["isl_total"]
    window = _make_window(isl_total, PROFILE_ISL)

    for _ in range(WARMUP):
        run(window, PROFILE_ISL)

    samples = []
    for _ in range(REPS):
        t0 = time.time()
        run(window, PROFILE_ISL)
        samples.append(time.time() - t0)
    lo, mean = min(samples), statistics.mean(samples)
    logger.info(
        f"window={isl_total} actual_isl={PROFILE_ISL} (pad tail): "
        f"min {lo*1000:.1f} ms  mean {mean*1000:.1f} ms  n={REPS}"
    )
    logger.info("compare min against the device total from test_ops; the difference is the host share")
    out = os.environ.get("PROFILE_OUT", f"/data/kmabee/mistral4_repro_logs/32_walltime_{isl_total}_{PROFILE_ISL}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "window": isl_total,
                "actual_isl": PROFILE_ISL,
                "min_s": lo,
                "mean_s": mean,
                "reps": REPS,
                "samples": [round(s, 4) for s in samples],
            },
            f,
            indent=1,
        )
    logger.info(f"wrote {out}")


# --- shared parametrization: the configuration the demo actually serves ---

_marks = [
    pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 bring-up targets Blackhole"),
    pytest.mark.parametrize("tokenizer", ["right"], indirect=True, ids=["right_pad"]),
    pytest.mark.parametrize(
        "mesh_device, device_params, num_links, topology",
        [
            pytest.param(
                (8, 4),
                {
                    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                    "fabric_router_config": create_fabric_router_config(
                        max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE
                    ),
                },
                2,
                ttnn.Topology.Linear,
                id="mesh-8x4",
            ),
            # The PP=4 stage geometry: SP=8 x TP=1 on an 8-chip carve (TT_VISIBLE_DEVICES=0..7).
            # Pair with PROFILE_NUM_LAYERS=9 to time one stage. Per-chip token count matches mesh-8x4
            # at the same window (both isl/8), so the traced times are directly comparable, and
            # PP=4 steady-state throughput ~= 1 / T(one stage).
            pytest.param(
                (8, 1),
                {
                    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                    "fabric_router_config": create_fabric_router_config(
                        max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE
                    ),
                },
                2,
                ttnn.Topology.Linear,
                id="mesh-8x1",
            ),
        ],
        indirect=["mesh_device", "device_params"],
    ),
    pytest.mark.parametrize("variant", ["mistral_small4"], indirect=True, ids=["mistral4"]),
    pytest.mark.timeout(0),
]


def _apply(marks, fn):
    for m in reversed(marks):
        fn = m(fn)
    return fn


def _build_and_run(
    hook,
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
    request,
):
    from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer import run_model

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        False,  # is_balanced
        PROFILE_SEQ_LEN,
        8,  # dispatch_buffer_capacity_factor
        PROFILE_NUM_LAYERS,  # num_layers (9 = one PP=4 stage; 36 = whole model)
        MistralSmall4Config.NUM_ROUTED_EXPERTS,
        GateComputeMode.GPT_DEVICE,
        num_links,
        topology,
        False,  # pcc_validation
        False,  # determinism_check
        1,
        "json_prompts",
        True,  # use_pretrained -- real weights, from the warm TTNN cache
        False,
        0.0,
        weight_cache_path,
        is_ci_env,
        is_ci_v2_env,
        tokenizer,
        request,
        serve_hook=hook,
    )


def _test_ops(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
    request,
):
    """Device-op breakdown of one warm forward. Run under `python -m tracy`."""
    _build_and_run(
        ops_hook,
        variant,
        config_only,
        mesh_device,
        device_params,
        num_links,
        topology,
        weight_cache_path,
        is_ci_env,
        is_ci_v2_env,
        tokenizer,
        request,
    )


def _test_walltime(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
    request,
):
    """End-to-end wall clock of one warm forward, min over REPS. Run WITHOUT the profiler."""
    _build_and_run(
        walltime_hook,
        variant,
        config_only,
        mesh_device,
        device_params,
        num_links,
        topology,
        weight_cache_path,
        is_ci_env,
        is_ci_v2_env,
        tokenizer,
        request,
    )


test_ops = _apply(_marks, _test_ops)
test_walltime = _apply(_marks, _test_walltime)


def _test_logging_ab(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
    request,
):
    """Interleaved A/B/A/B of loguru DEBUG vs WARNING in one process. Run WITHOUT the profiler."""
    _build_and_run(
        logging_ab_hook,
        variant,
        config_only,
        mesh_device,
        device_params,
        num_links,
        topology,
        weight_cache_path,
        is_ci_env,
        is_ci_v2_env,
        tokenizer,
        request,
    )


test_logging_ab = _apply(_marks, _test_logging_ab)


def _trace_body(ctx, controller, trace_input, window, const_isl, eager_min, eager_tok, eager_s):
    """Capture the block stack, replay it, and check the traced token matches eager.

    Split out of ``trace_hook`` so the caller can wrap it in try/finally: if anything in here raises,
    the trace buffers and the MoE-created sub-device managers must still be released, or
    ``close_mesh_device`` SEGFAULTS in pytest teardown and strands the 32-chip galaxy.
    """
    transformer, mesh_device = ctx["transformer"], ctx["mesh_device"]
    kvpe_cache, index_kv_cache = ctx["kvpe_cache"], ctx["index_kv_cache"]
    isl_total, sp_factor, isl_per_chip = ctx["isl_total"], ctx["sp_factor"], ctx["isl_per_chip"]

    def fwd_blocks():
        return transformer(
            trace_input,
            kvpe_cache,
            actual_isl=const_isl,
            return_intermediates=False,
            read_profiler=False,
            temperature=0.0,
            index_kv_cache=index_kv_cache,
            stop_after_blocks=True,
        )

    # Warm-compile the stop_after_blocks program before capturing: a capture records dispatch, not
    # compilation, and an uncompiled program would be compiled *inside* the capture.
    logger.info("=== warm-compiling the stop_after_blocks program, then capturing ===")
    fwd_blocks()
    ttnn.synchronize_device(mesh_device)

    controller.begin_capture()
    hidden = fwd_blocks()
    controller.end_capture()
    # num_segments is a @property; trace_bytes() is a method. (Calling the property is a TypeError
    # that fires right after capture and skips cleanup -- which is how the first attempt segfaulted.)
    logger.success(
        f"captured {controller.num_segments} segments, "
        f"{controller.trace_bytes() / (1024 * 1024):.1f} MB of trace buffers"
    )

    def traced_token(win, isl):
        """Replay the block stack, then run norm/LM-head/sample eagerly on the captured output."""
        host = ttnn.from_torch(
            win.reshape(sp_factor, 1, isl_per_chip),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
        )
        # Into the SAME tensor the capture recorded -- a fresh from_torch would allocate elsewhere
        # and the replay would keep reading the old address.
        ttnn.copy_host_to_device_tensor(host, trace_input)
        controller.replay()
        h = transformer.norm(hidden)
        _logits_host, first_token_logits = transformer._lm_head_and_extract(h, isl)
        tok, _prob, _sweep = transformer._sample(first_token_logits, isl, 0.0)
        return int(tok)

    logger.info("=== traced replay ===")
    for _ in range(2):
        traced_token(window, const_isl)
    traced_s, traced_tok = [], None
    for _ in range(REPS):
        t0 = time.time()
        traced_tok = traced_token(window, const_isl)
        traced_s.append(time.time() - t0)
    t_lo = min(traced_s)
    logger.info(f"  traced min {t_lo*1000:.1f} ms  mean {statistics.mean(traced_s)*1000:.1f} ms  token={traced_tok}")

    ok = traced_tok == eager_tok
    logger.info("=== RESULT ===")
    logger.info(f"  correctness: traced token {traced_tok} vs eager {eager_tok} -> {'MATCH' if ok else 'MISMATCH'}")
    logger.info(f"  eager  min {eager_min*1000:8.1f} ms")
    logger.info(f"  traced min {t_lo*1000:8.1f} ms")
    logger.info(f"  speedup {eager_min/t_lo:.2f}x  (device-kernel floor ~0.11 s => ceiling ~{eager_min/0.112:.1f}x)")
    if not ok:
        logger.error("  traced output DISAGREES with eager -- the speedup is meaningless until fixed")

    out = os.environ.get("PROFILE_OUT", f"/data/kmabee/mistral4_repro_logs/36_trace_{isl_total}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "window": isl_total,
                "const_isl": const_isl,
                "eager_min": eager_min,
                "traced_min": t_lo,
                "speedup": eager_min / t_lo,
                "token_match": ok,
                "eager_token": eager_tok,
                "traced_token": traced_tok,
                "segments": controller.num_segments,
                "trace_mb": controller.trace_bytes() / (1024 * 1024),
                "eager": eager_s,
                "traced": traced_s,
            },
            f,
            indent=1,
        )
    logger.info(f"wrote {out}")


def trace_hook(**ctx):
    """Prototype: capture the block stack in a ttnn trace, replay per token, run the tail eagerly.

    The forward is ~95% host-bound -- 2316 op dispatches at ~0.55 ms each against ~80 ms of actual
    device kernel time -- so replacing per-op dispatch with a trace replay is the fix that matches the
    diagnosis. Three things make it possible without new tracing machinery:

    * ``SubDeviceTraceController`` already splits a capture at the MoE's sub-device swaps, which a
      naive single capture cannot survive, and ``TtPrefillTransformer.set_trace_controller`` accepts one.
    * ``actual_isl`` is held CONSTANT for the whole generation so the captured op sequence -- and the
      MoE padding config it memoizes -- is invariant. Causality makes that safe: the LM head reads row
      ``n-1`` and nothing attends past it, so marking not-yet-generated positions "real" cannot change
      the logits we read. The runtime captures with ``actual_isl=chunk_size`` for the same reason.
    * The tail (norm/LM-head/sample) ends in a blocking D2H that cannot be traced, so it is excluded
      via ``stop_after_blocks`` and run eagerly -- ~36 of 2316 ops, so ~98% of dispatch is still captured.
    """
    from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

    transformer, mesh_device = ctx["transformer"], ctx["mesh_device"]
    kvpe_cache, index_kv_cache = ctx["kvpe_cache"], ctx["index_kv_cache"]
    isl_total, sp_factor, isl_per_chip = ctx["isl_total"], ctx["sp_factor"], ctx["isl_per_chip"]

    const_isl = min(isl_total, PROFILE_ISL + int(os.environ.get("TRACE_MAX_NEW", 32)))
    window = _make_window(isl_total, PROFILE_ISL)

    def host_to_tt(win):
        return ttnn.from_torch(
            win.reshape(sp_factor, 1, isl_per_chip),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
        )

    def eager_token(win, isl):
        tt = host_to_tt(win)
        tok, _, _ = transformer(
            tt,
            kvpe_cache,
            actual_isl=isl,
            return_intermediates=False,
            read_profiler=False,
            temperature=0.0,
            index_kv_cache=index_kv_cache,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(tt)
        return int(tok)

    logger.info(f"=== eager baseline (constant actual_isl={const_isl}) ===")
    for _ in range(WARMUP):
        eager_token(window, const_isl)
    eager_s, eager_tok = [], None
    for _ in range(REPS):
        t0 = time.time()
        eager_tok = eager_token(window, const_isl)
        eager_s.append(time.time() - t0)
    e_lo = min(eager_s)
    logger.info(f"  eager min {e_lo*1000:.1f} ms  mean {statistics.mean(eager_s)*1000:.1f} ms  token={eager_tok}")

    # Persistent input buffer: the capture records THIS address, so each replay must find the step's
    # tokens written into it.
    trace_input = host_to_tt(window)
    controller = SubDeviceTraceController(mesh_device)
    transformer.set_trace_controller(controller)
    try:
        _trace_body(ctx, controller, trace_input, window, const_isl, e_lo, eager_tok, eager_s)
    finally:
        try:
            controller.release()
        finally:
            transformer.set_trace_controller(None)
            transformer.release_sub_device_managers()
        logger.info("trace buffers + sub-device managers released")


def _test_trace(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
    request,
):
    """Capture the block stack in a trace and compare against eager. Run WITHOUT the profiler."""
    _build_and_run(
        trace_hook,
        variant,
        config_only,
        mesh_device,
        device_params,
        num_links,
        topology,
        weight_cache_path,
        is_ci_env,
        is_ci_v2_env,
        tokenizer,
        request,
    )


test_trace = _apply(_marks, _test_trace)
