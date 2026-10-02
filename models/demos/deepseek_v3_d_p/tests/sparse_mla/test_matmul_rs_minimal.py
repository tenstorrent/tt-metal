# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Existing fused MMRS versus production linear + RS, SP2/TP4.

Build: ./build_metal.sh --release
Run: scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/sparse_mla/test_matmul_rs_minimal.py -m perf -s
MMRS_REPORT_DIR selects reports; MMRS_RUN=2 reverses measurement order.
MMRS_TUNE accepts a JSON object of MM geometry and CCL overrides.
Single and sustained trace spans include program/dispatch gaps on each chip;
per-program timings remain separate diagnostics, not end-to-end latency.
"""

import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    assert_requested_tp_wrap_was_realized,
    fabric2d_device_params,
    torus_x_device_params,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_plugin import is_marker_explicitly_selected
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import get_matmul_config
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler

# K, N, scatter dimension, MM grid width, per-core N, block N, subblock H/W, CCL workers.
CASES = {
    "o_proj": (4096, 6144, 3, 10, 20, 10, 1, 5, 4),
    "q_a_proj": (1536, 2048, 3, 8, 8, 8, 1, 8, 4),
    "indexer.wk": (1536, 128, 3, 4, 1, 1, 2, 1, 1),
    "indexer.weights_proj": (1536, 32, 2, 1, 1, 1, 2, 1, 1),
}


def check(actual, expected):
    a, b = actual.flatten().double(), expected.flatten().double()
    assert torch.isfinite(a).all()
    pcc = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
    nrmse = ((a - b).square().mean() / b.square().mean()).sqrt().item()
    if pcc < 0.999 or nrmse > 0.03:
        extreme = actual[actual.abs() > 1e20].to(torch.bfloat16).view(torch.int16)
        print(
            "Extreme BF16 bit patterns:",
            [(hex(int(v) & 65535), int(c)) for v, c in zip(*torch.unique(extreme, return_counts=True))],
        )
    assert pcc >= 0.999 and nrmse <= 0.03, (pcc, nrmse)
    return dict(pcc=pcc, nrmse=nrmse)


@pytest.mark.perf
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params,topology",
    [
        pytest.param(fabric2d_device_params(trace_region_size=32 << 20), ttnn.Topology.Linear, id="linear"),
        pytest.param(torus_x_device_params(trace_region_size=32 << 20), ttnn.Topology.Ring, id="ring"),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("case", CASES)
def test_matmul_rs_minimal(mesh_device, topology, case, request, expect_error):
    if not is_marker_explicitly_selected(request.config, "perf"):
        pytest.skip("Select explicitly with -m perf")
    require_realtime_profiler("Fused MMRS comparison")
    if topology == ttnn.Topology.Ring:
        assert_requested_tp_wrap_was_realized(mesh_device)
    k, n, dim, gx, pn, bn, sh, sw, workers = CASES[case]
    # Use MM and CCL geometry independently of their existing synchronization.
    pm, gy = 2, 10
    ccl_offset = ttnn.CoreCoord(gx, 0)
    if os.getenv("MMRS_PLACEMENT", "rows") == "rows" and n > 128:
        pm, gy = 3, 7
        gx, pn, bn, sh, sw = (12, 16, 16, 1, 8) if case == "o_proj" else (8, 8, 8, 1, 8)
        ccl_offset = ttnn.CoreCoord(0, gy)
    generator = torch.Generator().manual_seed(1713)
    x = torch.randn((1, 1, 1280, 4 * k), generator=generator)
    w = torch.randn((4 * k, n), generator=generator) / math.sqrt(4 * k)
    golden = x @ w
    tx = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat8_b if case == "o_proj" else ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(2, 3)),
    )
    tw = ttnn.from_torch(
        w,
        device=mesh_device,
        dtype=ttnn.bfloat16 if dim == 2 else ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(None, 0)),
    )
    mm_compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    rs_compute = (
        ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        if dim == 2
        else None
    )
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)]
    barrier = ttnn.create_global_semaphore(mesh_device, cores, 0)
    mm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=8,
        per_core_M=pm,
        per_core_N=pn,
        out_block_h=pm,
        out_block_w=bn,
        out_subblock_h=sh,
        out_subblock_w=sw,
        transpose_mcast=False,
        fuse_batch=False,
        allowed_worker_cores=mm_cores,
    )
    baseline = get_matmul_config(case, 640)
    if isinstance(baseline, list):
        glm = glm_5_3_hf_config()
        baseline = next(
            c
            for c in baseline
            if c.get("num_heads") == glm.num_attention_heads
            and c.get("q_lora_rank") == glm.q_lora_rank
            and c.get("chunked_only") is True
        )
    if os.getenv("MMRS_CONFIG") == "production":
        pc = baseline["program_config"]
        gx, gy = math.ceil(n / 32 / pc.per_core_N), math.ceil(20 / pc.per_core_M)
        mm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=pc.in0_block_w,
            per_core_M=pc.per_core_M,
            per_core_N=pc.per_core_N,
            out_block_h=pc.out_block_h,
            out_block_w=pc.out_block_w,
            out_subblock_h=pc.out_subblock_h,
            out_subblock_w=pc.out_subblock_w,
            transpose_mcast=False,
            fuse_batch=False,
            allowed_worker_cores=mm_cores,
        )
        workers = 1
        ccl_offset = ttnn.CoreCoord(gx, 0)
    tuning = json.loads(os.getenv("MMRS_TUNE", "{}"))
    if tuning:
        pn = tuning.get("per_core_N", config.per_core_N)
        gx = math.ceil(n / 32 / pn)
        gy = math.ceil(20 / config.per_core_M)
        workers = tuning.get("workers", workers)
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=tuning.get("in0_block_w", config.in0_block_w),
            per_core_M=config.per_core_M,
            per_core_N=pn,
            out_block_h=config.out_block_h,
            out_block_w=tuning.get("out_block_w", config.out_block_w),
            out_subblock_h=tuning.get("out_subblock_h", config.out_subblock_h),
            out_subblock_w=tuning.get("out_subblock_w", config.out_subblock_w),
            transpose_mcast=False,
            fuse_batch=False,
            allowed_worker_cores=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}
            ),
        )
        ccl_offset = ttnn.CoreCoord(gx, tuning.get("ccl_y", 0))
    intermediate_memory = ttnn.L1_MEMORY_CONFIG if tuning.get("intermediate_l1") else ttnn.DRAM_MEMORY_CONFIG
    output_shape = (1, 1, 160, n) if dim == 2 else (1, 1, 640, n // 4)
    intermediate = ttnn.empty(
        (2 if topology == ttnn.Topology.Linear else 1, 1, 640, n),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=intermediate_memory,
    )
    output = ttnn.empty(
        output_shape,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def fused():
        return ttnn.experimental.matmul_reduce_scatter_async(
            tx,
            tw,
            intermediate,
            output,
            dim,
            semaphores,
            ccl_offset,
            barrier_semaphore=barrier,
            num_links=2,
            cluster_axis=1,
            num_workers_per_link=workers,
            topology=topology,
            program_config=config,
            memory_config_mm=ttnn.L1_MEMORY_CONFIG,
            memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config_rs=intermediate_memory,
            dtype=ttnn.bfloat16,
            compute_kernel_config=mm_compute,
            rs_compute_kernel_config=rs_compute,
            chunks_per_sync=tuning.get("chunks_per_sync"),
            num_buffers_per_channel=tuning.get("num_buffers_per_channel"),
        )

    def composed():
        mm = ttnn.linear(
            tx,
            tw,
            program_config=baseline["program_config"],
            memory_config=baseline["out_mem_config"],
            dtype=baseline["out_dtype"],
            compute_kernel_config=mm_compute,
        )
        rs = ttnn.experimental.reduce_scatter_minimal_async(
            mm,
            dim=dim,
            cluster_axis=1,
            num_links=2,
            topology=topology,
            multi_device_global_semaphore=semaphores,
            barrier_semaphore=barrier,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=rs_compute,
        )
        return mm, rs

    def read(tensor):
        return (
            ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, mesh_shape=(2, 4), dims=(0, 2) if dim == 2 else (2, 3)
                ),
            )
            .float()
            .reshape(golden.shape)
        )

    def free(values):
        for tensor in values:
            if tensor.buffer_address() != output.buffer_address():
                ttnn.deallocate(tensor)

    order = [("composed", composed, 2), ("fused", fused, 1)]
    if int(os.getenv("MMRS_RUN", "1")) % 2 == 0:
        order.reverse()
    report = dict(case=case, topology=str(topology), config=str(config), tuning=tuning, measurements={})
    for name, fn, programs in order:
        print("CHECK", name, flush=True)
        values = fn()
        accuracy = check(read(values[-1]), golden)
        free(values)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        values = fn()
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        try:

            def replay(count):
                for _ in range(count):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)

            profile_realtime_program(mesh_device, lambda: replay(3), collect_all=True)
            samples, single_samples = [], []
            trace_spans = defaultdict(list)
            program_samples, program_sources = defaultdict(list), {}
            # Single invocation latency and sustained replay throughput are different workloads.
            # The model executes each projection once per forward, not twenty times back-to-back.
            for count in (1, 20):
                for _ in range(5):
                    _, records = profile_realtime_program(mesh_device, lambda: replay(count), collect_all=True)
                    grouped = defaultdict(lambda: defaultdict(list))
                    for r in records:
                        grouped[r["runtime_id"]][r["chip_id"]].append(r["duration_ns"])
                        program_sources[r["runtime_id"]] = r["kernel_sources"]
                    assert len(grouped) == programs
                    # Use each chip's own clock. Include inter-program and inter-replay gaps;
                    # summing program durations alone hides rank phase drift in sustained CCLs.
                    per_chip = defaultdict(list)
                    for record in records:
                        per_chip[record["chip_id"]].append(record)
                    trace_spans[count].append(
                        max(
                            (
                                max(r["end_timestamp"] for r in chip_records)
                                - min(r["start_timestamp"] for r in chip_records)
                            )
                            / chip_records[0]["frequency"]
                            / 1000
                            / count
                            for chip_records in per_chip.values()
                        )
                    )
                    for runtime_id, chips in grouped.items():
                        assert set(chips) == set(mesh_device.get_device_ids())
                        assert all(len(v) == count for v in chips.values())
                        if count == 20:
                            program_samples[runtime_id].extend(
                                max(chips[c][i] for c in chips) / 1000 for i in range(count)
                            )
                    measured = [
                        sum(max(chips[c][i] for c in chips) for chips in grouped.values()) / 1000 for i in range(count)
                    ]
                    (single_samples if count == 1 else samples).extend(measured)
            check(read(values[-1]), golden)
            # Change inputs while replaying the same cached program; reject stale publication/counter state.
            for scale in (2, -1, 1):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        x * scale,
                        dtype=tx.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(2, 3)),
                    ),
                    tx,
                )
                replay(1)
                check(read(values[-1]), golden * scale)
        finally:
            ttnn.release_trace(mesh_device, trace)
            free(values)
        report["measurements"][name] = dict(
            device_median_us=statistics.median(samples),
            device_us=samples,
            single_replay_median_us=statistics.median(single_samples),
            single_replay_us=single_samples,
            single_trace_span_median_us=statistics.median(trace_spans[1]),
            sustained_trace_span_per_replay_median_us=statistics.median(trace_spans[20]),
            trace_span_samples_us=dict(trace_spans),
            accuracy=accuracy,
            programs=[
                dict(sources=program_sources[key], median_us=statistics.median(times))
                for key, times in program_samples.items()
            ],
        )
    if os.getenv("MMRS_COMPONENTS") == "1":

        def time_single(fn, persistent=False):
            warm = fn()
            ttnn.synchronize_device(mesh_device)
            if not persistent:
                ttnn.deallocate(warm)
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            captured = fn()
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            try:

                def replay(count):
                    for _ in range(count):
                        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)

                profile_realtime_program(mesh_device, lambda: replay(3), collect_all=True)
                times = []
                for _ in range(5):
                    _, records = profile_realtime_program(mesh_device, lambda: replay(20), collect_all=True)
                    assert len({r["runtime_id"] for r in records}) == 1
                    chips = defaultdict(list)
                    for r in records:
                        chips[r["chip_id"]].append(r["duration_ns"])
                    assert set(chips) == set(mesh_device.get_device_ids())
                    assert all(len(v) == 20 for v in chips.values())
                    times.extend(max(chips[c][i] for c in chips) / 1000 for i in range(20))
                return statistics.median(times)
            finally:
                ttnn.release_trace(mesh_device, trace)
                if not persistent:
                    ttnn.deallocate(captured)

        def standalone_mm():
            return ttnn.linear(
                tx,
                tw,
                program_config=config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=mm_compute,
            )

        same_mm_us = time_single(standalone_mm)
        mm = standalone_mm()

        def standalone_rs():
            return ttnn.experimental.reduce_scatter_minimal_async(
                mm,
                persistent_output_buffers=(intermediate, output),
                dim=dim,
                cluster_axis=1,
                num_links=2,
                topology=topology,
                multi_device_global_semaphore=semaphores,
                barrier_semaphore=barrier,
                num_workers_per_link=workers,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=rs_compute,
            )

        same_rs_us = time_single(standalone_rs, persistent=True)
        check(read(output), golden)
        ttnn.deallocate(mm)
        report["components"] = dict(
            matmul_same_config_us=same_mm_us,
            rs_tiled_same_workers_us=same_rs_us,
            residual_us=report["measurements"]["fused"]["device_median_us"] - same_mm_us - same_rs_us,
            residual_scope="Includes fusion synchronization and different CCL core placement; not pure barrier time",
        )
    # Rebind every cached address category, including caller-owned scratch/output and semaphores.
    saved = tx, tw, intermediate, output, semaphores, barrier
    tx, tw = ttnn.clone(tx), ttnn.clone(tw)
    intermediate = ttnn.empty(
        saved[2].shape,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=intermediate_memory,
    )
    output = ttnn.empty(
        output_shape,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)]
    barrier = ttnn.create_global_semaphore(mesh_device, cores, 0)
    cache_before = mesh_device.num_program_cache_entries()
    values = fused()
    check(read(values[-1]), golden)
    assert mesh_device.num_program_cache_entries() == cache_before
    free(values)
    replacement_tensors = tx, tw, intermediate, output
    if topology == ttnn.Topology.Linear:
        valid_intermediate = intermediate
        intermediate = ttnn.empty(
            (1, 1, 640, n),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=intermediate_memory,
        )
        with expect_error(RuntimeError, "MMRS intermediate must be input-shaped"):
            fused()
        ttnn.deallocate(intermediate)
        intermediate = valid_intermediate
    valid_semaphores = semaphores
    semaphores = []
    with expect_error(RuntimeError, "Insufficient CCL semaphores"):
        fused()
    semaphores = valid_semaphores
    for knob in ("chunks_per_sync", "num_buffers_per_channel"):
        previous = tuning.get(knob)
        tuning[knob] = 0
        try:
            with expect_error(RuntimeError, f"{knob} must be positive"):
                fused()
        finally:
            if previous is None:
                tuning.pop(knob)
            else:
                tuning[knob] = previous
    tx, tw, intermediate, output, semaphores, barrier = saved
    for tensor in replacement_tensors:
        ttnn.deallocate(tensor)
    host = {"fused": [], "composed": []}
    for trial in range(int(os.getenv("MMRS_HOST_TRIALS", "5"))):
        for name, fn, _ in order if trial % 2 == 0 else list(reversed(order)):
            timings = []
            for i in range(110):
                start = time.perf_counter_ns()
                values = fn()
                elapsed = (time.perf_counter_ns() - start) / 1000
                ttnn.synchronize_device(mesh_device)
                free(values)
                if i >= 10:
                    timings.append(elapsed)
            host[name].append(statistics.median(timings))
    for name in host:
        if not host[name]:
            continue
        report["measurements"][name]["host_median_us"] = statistics.median(host[name])
        report["measurements"][name]["host_trials_us"] = host[name]
    report[
        "host_scope"
    ] = "Warm API calls; sync/free outside timer; fused caller-owned persistent buffers allocated outside timer"
    folder = Path(os.getenv("MMRS_REPORT_DIR", "generated/profiler/fresh_mmrs/results"))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{case}-{str(topology).split('.')[-1].lower()}.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                name: {k: v for k, v in data.items() if k.endswith("median_us")}
                for name, data in report["measurements"].items()
            }
        )
    )
