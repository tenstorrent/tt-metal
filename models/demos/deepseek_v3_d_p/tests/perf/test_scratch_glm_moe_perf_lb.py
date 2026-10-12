# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""SCRATCH: GLM 5.3 MoE real-time-profiler device time on a 2x4 LoudBox.

Mirrors test_kimi_moe_perf.py's K3 case (warm-up pass, then sum of per-program critical paths),
but with the routed SiTU-GLU activation the pcc test uses, and logs a per-program breakdown.
"""

import os

import pytest

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    fabric_1d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

_ITERS = int(os.environ.get("K3_PERF_ITERS", "1"))


_MESH = tuple(int(x) for x in os.environ.get("K3_MESH", "2x4").split("x"))
_FABRIC = {
    "mesh": fabric2d_device_params,
    "1d": fabric_1d_device_params,
    "torus_x": torus_x_device_params,
    "torus_y": torus_y_device_params,
    "torus_xy": torus_xy_device_params,
}[os.environ.get("K3_FABRIC", "mesh")]


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            _MESH,
            _FABRIC(
                fabric_payload_size=int(os.environ.get("K3_FABRIC_PAYLOAD", GLM53Config.FABRIC_PAYLOAD_SIZE)),
                trace_region_size=200_000_000,
            ),
            2,
            id="lb",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["ds-ref"])
def test_glm_moe_perf_lb(variant, config_only, mesh_device, device_params, num_links, request):
    tracy_mode = os.environ.get("K3_TRACY") == "1" or os.environ.get("K3_TRACE") == "1"
    if not tracy_mode:
        require_realtime_profiler("K3 MoE LB perf")
    topology = per_axis_topology(device_params["fabric_config"])
    print(f"K3_FABRIC_INFO mesh={tuple(mesh_device.shape)} fabric={ttnn.get_fabric_config()} topology={topology}")
    runs = []

    def measure(forward):
        warm = forward()
        ttnn.synchronize_device(mesh_device)
        del warm
        if os.environ.get("K3_TRACE") == "1":
            import time

            from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

            moe = next(c.cell_contents for c in forward.__closure__ if type(c.cell_contents).__name__ == "TtMoe")
            iters = int(os.environ.get("K3_TRACE_ITERS", "10"))

            def wall(fn):
                ts = []
                for _ in range(iters):
                    ttnn.synchronize_device(mesh_device)
                    t = time.perf_counter()
                    o = fn()
                    ttnn.synchronize_device(mesh_device)
                    ts.append((time.perf_counter() - t) * 1e3)
                    del o
                ts.sort()
                return ts

            eager = wall(forward)
            ctrl = SubDeviceTraceController(mesh_device)
            moe.set_trace_controller(ctrl)
            ctrl.begin_capture()
            out = forward()
            ctrl.end_capture()
            traced = wall(ctrl.replay)
            print(
                f"K3_TRACE eager  ms: min {eager[0]:.3f} p50 {eager[len(eager) // 2]:.3f} all {[round(x, 3) for x in eager]}"
            )
            print(
                f"K3_TRACE traced ms: min {traced[0]:.3f} p50 {traced[len(traced) // 2]:.3f} all {[round(x, 3) for x in traced]}"
            )
            ctrl.release()
            moe.set_trace_controller(None)
            runs.append({})
            return out
        if tracy_mode:
            from tracy import signpost

            ttnn.ReadDeviceProfiler(mesh_device)  # flush the warm-up's device records
            signpost(header="K3_MEASURE_START")
            out = forward()
            ttnn.synchronize_device(mesh_device)
            signpost(header="K3_MEASURE_END")
            runs.append({})
            return out
        if os.environ.get("K3_OVERLAP") == "1":
            import time

            recs = []

            def cb(batch):
                for r in batch.records:
                    if int(r.end_timestamp) > int(r.start_timestamp) and float(r.frequency) > 0:
                        recs.append(
                            (
                                int(r.chip_id),
                                int(r.runtime_id),
                                int(r.start_timestamp),
                                int(r.end_timestamp),
                                float(r.frequency),
                                os.path.basename(str(list(r.kernel_sources)[0])) if len(r.kernel_sources) else "?",
                            )
                        )

            h = ttnn.device.RegisterProgramRealtimeProfilerCallback(cb)
            out = forward()
            ttnn.synchronize_device(mesh_device)
            time.sleep(3)
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(h)
            by_chip = {}
            for r in recs:
                by_chip.setdefault(r[0], []).append(r)
            for chip, rs in sorted(by_chip.items()):
                f = rs[0][4]
                ivs = sorted((a, b) for _, _, a, b, _, _ in rs)
                total = sum(b - a for a, b in ivs)
                union, cur_a, cur_b = 0, None, None
                for a, b in ivs:
                    if cur_b is None or a > cur_b:
                        if cur_b is not None:
                            union += cur_b - cur_a
                        cur_a, cur_b = a, b
                    else:
                        cur_b = max(cur_b, b)
                union += cur_b - cur_a
                span = max(b for _, b in ivs) - ivs[0][0]  # latest END, not the end of the last-starting program
                print(
                    f"K3_OVL chip {chip}: progs {len(rs)} sum {total / f:,.0f} ns  union {union / f:,.0f} ns  "
                    f"overlap {(total - union) / f:,.0f} ns  span {span / f:,.0f} ns  gaps {(span - union) / f:,.0f} ns"
                )
                if chip == min(by_chip):
                    rs = sorted(rs, key=lambda r: r[2])
                    t0 = rs[0][2]
                    for c, rid, a, b, f, k in rs:
                        conc = [x[5] for x in rs if x[1] != rid and x[2] < b and x[3] > a]
                        print(f"K3_OVL  t={(a - t0) / f:>10,.0f} dur={(b - a) / f:>10,.0f}  {rid:>4} {k:<45} || {conc}")
            del out
        result = None
        for _ in range(_ITERS):
            if result is not None:
                del result
            result, records = profile_realtime_program_merged(mesh_device, forward, record_timeout_seconds=5.0)
            runs.append(records)
        return result

    loop_n = int(os.environ.get("K3_HOST_LOOP", "0"))
    if loop_n:
        import time

        dp = ttnn.experimental.deepseek_prefill
        orig = dp.unified_routed_expert_moe
        calls = [0]

        def looped(*a, **k):
            r = orig(*a, **k)
            calls[0] += 1
            if calls[0] == 2:  # the measured forward: replay the cache-hit launch with zero counts
                a = list(a)
                zero_counts = ttnn.zeros_like(a[2])  # expert_token_counts -> every expert skipped on device
                a[2] = zero_counts
                ttnn.synchronize_device(mesh_device)
                ts = []
                try:
                    from tracy import signpost

                    signpost(header="K3_HOST_LOOP_START")
                except ImportError:
                    signpost = None
                print(f"K3_LOOP_BEGIN pid={os.getpid()}", flush=True)
                burst = int(os.environ.get("K3_HOST_BURST", "4"))
                interleave = os.environ.get("K3_INTERLEAVE") == "1"
                alt_x = [a[0], ttnn.clone(a[0])] if os.environ.get("K3_ALT_X") == "1" else None
                base_x = a[0]
                for i in range(loop_n):
                    if alt_x:
                        a[0] = alt_x[i % 2]  # input address changes on every launch
                    fresh = os.environ.get("K3_FRESH_X") == "1"
                    if fresh:
                        if i:
                            ttnn.deallocate(a[0])
                        a[0] = ttnn.clone(base_x)  # a never-launched tensor object every call
                    if interleave:
                        ttnn.deallocate(ttnn.add(zero_counts, zero_counts))  # a different program in between
                    t = time.perf_counter()
                    out = orig(*a, **k)
                    ts.append(time.perf_counter() - t)
                    ttnn.deallocate(out)
                    if burst and i % burst == burst - 1:
                        ttnn.synchronize_device(mesh_device)  # keep the queue shallow: time host work, not backpressure
                ttnn.synchronize_device(mesh_device)
                print("K3_LOOP_DONE", flush=True)
                if signpost:
                    signpost(header="K3_HOST_LOOP_END")
                if burst:
                    for j in range(burst):
                        b = sorted(ts[j::burst])
                        print(f"K3_BURSTPOS {j}: p50 {b[len(b) // 2] * 1e6:,.1f} us  min {b[0] * 1e6:,.1f}")
                ts_us = sorted(x * 1e6 for x in ts)
                print(
                    f"K3_HOSTLOOP n={loop_n} min {ts_us[0]:,.1f} p50 {ts_us[len(ts_us) // 2]:,.1f} "
                    f"p90 {ts_us[int(len(ts_us) * 0.9)]:,.1f} max {ts_us[-1]:,.1f} us  first5 {[round(x * 1e6) for x in ts[:5]]}"
                )
                ttnn.deallocate(zero_counts)
            return r

        dp.unified_routed_expert_moe = looped
    if os.environ.get("K3_PROBE") == "1":
        import time

        dp = ttnn.experimental.deepseek_prefill
        orig_p = dp.unified_routed_expert_moe
        pc = [0]

        def probe(*a, **k):
            pc[0] += 1
            if pc[0] != 2:
                return orig_p(*a, **k)

            def timed_call(args, label):
                ttnn.synchronize_device(mesh_device)
                t = time.perf_counter()
                r = orig_p(*args, **k)
                dt = (time.perf_counter() - t) * 1e6
                print(f"K3_PROBE {label}: {dt:,.0f} us", flush=True)
                return r

            x0 = a[0]
            for j in range(3):
                ttnn.synchronize_device(mesh_device)
                t = time.perf_counter()
                e = ttnn.empty(
                    x0.shape,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                print(
                    f"K3_PROBE ttnn.empty like the op's output #{j}: {(time.perf_counter() - t) * 1e6:,.0f} us  shape={tuple(x0.shape)} layout={x0.layout}",
                    flush=True,
                )
                ttnn.deallocate(e)
            r = timed_call(a, "real call (fresh x, real counts)")
            ttnn.deallocate(timed_call(a, "repeat, same args (real counts)"))
            ttnn.deallocate(timed_call(a, "repeat, same args again"))
            x2 = ttnn.clone(a[0])
            a2 = list(a)
            a2[0] = x2
            ttnn.deallocate(timed_call(a2, "fresh x copy (new address)"))
            ttnn.deallocate(timed_call(a2, "repeat with that copy"))
            ttnn.deallocate(x2)
            return r

        dp.unified_routed_expert_moe = probe
    if os.environ.get("K3_HOST_TIMING") == "1":
        import time

        import models.demos.deepseek_v3_d_p.tt.moe.tt_moe as tmoe
        import models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert as tre

        def timed(name, fn):
            def w(*a, **k):
                t = time.perf_counter()
                r = fn(*a, **k)
                print(f"K3_HOST {name}: {(time.perf_counter() - t) * 1e6:,.0f} us")
                return r

            return w

        dp = ttnn.experimental.deepseek_prefill
        dp.unified_routed_expert_moe = timed("unified_routed_expert_moe", dp.unified_routed_expert_moe)
        tre.TtRoutedExpert.forward = timed("TtRoutedExpert.forward", tre.TtRoutedExpert.forward)
        tmoe.TtMoe.forward = timed("TtMoe.forward", tmoe.TtMoe.forward)
        for n in ("load_sub_device_manager", "clear_loaded_sub_device_manager"):
            cls = type(mesh_device)
            setattr(cls, n, timed(n, getattr(cls, n)))
    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        PREFILL_CHUNK_TOKENS_PER_CHIP,
        GLM53Config.EMB_SIZE,
        GLM53Config.MOE_INTERMEDIATE_SIZE,
        # 64 experts on the 2x4 LB = 8 per chip, the 8x4 Galaxy's per-chip load with all 256.
        int(os.environ.get("GLM_NUM_EXPERTS", GLM53Config.NUM_ROUTED_EXPERTS)),
        GLM53Config.NUM_EXPERTS_PER_TOKEN,
        8,
        os.environ.get("GLM_PCC") == "1",
        num_links,
        topology,
        GateComputeMode.DEVICE_FP32,
        request,
        is_balanced=os.environ.get("GLM_BALANCED") == "1",
        measure=measure,
    )

    assert runs
    if tracy_mode:
        return
    totals = []
    for i, per_program in enumerate(runs):
        total = sum(e["duration_ns"] for e in per_program.values())
        totals.append(total)
        print(f"K3_PERF run {i}: total {total:,.0f} ns over {len(per_program)} programs")
    # Per-program breakdown of the last run, identified by compute kernel source.
    for rid, e in runs[-1].items():
        srcs = [os.path.basename(s) for s in e["kernel_sources"]]
        print(f"K3_PERF prog {rid}: {e['duration_ns']:>12,.0f} ns  {srcs}")
    totals.sort()
    print(f"K3_PERF median total: {totals[len(totals) // 2]:,.0f} ns  all={[f'{t:,.0f}' for t in totals]}")
