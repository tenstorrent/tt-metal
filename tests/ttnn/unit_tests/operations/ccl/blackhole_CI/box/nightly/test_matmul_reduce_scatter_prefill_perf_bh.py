# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time perf gates for ttnn.experimental.matmul_reduce_scatter_prefill on a Blackhole QuietBox 2.

A representative set of per-device prefill shapes (TP=4 row-parallel linears of DeepSeek-V3, Kimi K2.7 / K3, GLM 5.x,
MiniMax-M3 and MiMo-V2, two output-chunked shapes that miss L1, Llama-70B WO and Flux2 FF2 chunked by the cost model,
and Kimi K2.7 o_proj at decode on a thin 2D grid) runs traced on the whole box as a 1x4 ring, and one shape as a 1x4
line. Each case: an eager call (compile), one traced call, one discarded replay, then REPLAYS back-to-back replays in
one real-time profiler window; the gate is the median device time of the op's program (the slowest chip per replay)
within +/- MARGIN of the case's expected time. A case faster than the band fails too: update its number, so the gate
stays tight.

A and W are replicated on every device, so the group sum is G * (A @ W) and every device's output is checked against it
(PCC), at the cost of one host matmul per case.
"""

import json
import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.ci_failure import assert_perf_gate
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_matmul_reduce_scatter_prefill_bh import (
    PCC,
    assert_thin_wide_plan,
    line_device_params,
    quantize_like_device,
    ring_device_params,
)
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher

REPLAYS = 10
MARGIN = 0.05
TRACE_REGION_BYTES = 1 << 20
# the op's program on every chip (the realtime profiler reports each program's kernel sources)
KERNEL = "matmul_reduce_scatter_prefill/device/kernels/matmul_reduce_scatter_prefill_reader.cpp"

# name: (a_shape, w_shape, scatter_dim, activation dtype, weight dtype); production precision (HiFi2, bf8b weights)
PERF_CASES = {
    "kimi_k27_o_proj": ((1, 1, 640, 2048), (2048, 7168), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "deepseek_v3_o_proj": ((1, 1, 640, 4096), (4096, 7168), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "minimax_m3_o_proj_bf8b_act": ((1, 1, 640, 2048), (2048, 6144), -1, ttnn.bfloat8_b, ttnn.bfloat8_b),
    "mimo_v2_o_proj": ((1, 1, 2048, 2048), (2048, 4096), -2, ttnn.bfloat16, ttnn.bfloat8_b),
    "kimi_k3_dense_ffn": ((1, 1, 640, 8448), (8448, 7168), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "glm5_q_a_proj": ((1, 1, 640, 1536), (1536, 2048), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    # output chunks (the block misses L1): 1 x 6 and 3 x 2 chunks on a 1x4 ring
    "chunked_n": ((1, 1, 2048, 1024), (1024, 16384), -1, ttnn.bfloat16, ttnn.bfloat16),
    "chunked_mn": ((1, 1, 4096, 1024), (1024, 8192), -1, ttnn.bfloat16, ttnn.bfloat16),
    # chunked although the block fits L1 unchunked (with a 1-2 tile K-block): 3 x 1 and 3 x 2 chunks
    "llama_70b_wo_prefill": ((1, 1, 8192, 1024), (1024, 2048), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "flux2_ff2": ((1, 1, 4224, 2304), (2304, 6144), -1, ttnn.bfloat16, ttnn.bfloat16),
    # decode (M = 32) on the default 2D grid: a 1 x 10 thin-wide plan, W read on NoC0 (asserted, THIN_WIDE_PERF_CASES)
    "kimi_k27_o_proj_decode_2d": ((1, 1, 32, 2048), (2048, 7168), -1, ttnn.bfloat16, ttnn.bfloat8_b),
}
# the cases that must take the thin-wide plan (test_matmul_reduce_scatter_prefill_bh.assert_thin_wide_plan)
THIN_WIDE_PERF_CASES = {"kimi_k27_o_proj_decode_2d"}

# Expected device time [ns] per (topology, case): QuietBox 2 (1x4, 2 links), medians of back-to-back traced replays.
EXPECTED_NS = {
    ("ring", "kimi_k27_o_proj"): 117_500,
    ("ring", "deepseek_v3_o_proj"): 211_000,
    ("ring", "minimax_m3_o_proj_bf8b_act"): 102_000,
    ("ring", "mimo_v2_o_proj"): 206_000,
    ("ring", "kimi_k3_dense_ffn"): 506_000,
    ("ring", "glm5_q_a_proj"): 53_500,
    ("ring", "chunked_n"): 549_500,
    ("ring", "chunked_mn"): 594_000,
    ("ring", "llama_70b_wo_prefill"): 297_000,
    ("ring", "flux2_ff2"): 641_000,
    ("ring", "kimi_k27_o_proj_decode_2d"): 62_000,
    ("line", "kimi_k27_o_proj"): 157_000,
}


def perf_device_params(params):
    return {**params, "trace_region_size": TRACE_REGION_BYTES}


def replicated(mesh_device, t, dtype):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def check_replicated(out, expected, mesh_device, cluster_axis, scatter_dim, weight_dtype):
    """Device (r, c) holds chunk (r if cluster_axis == 0 else c) of the group sum."""
    rows, cols = tuple(mesh_device.shape)
    group = rows if cluster_axis == 0 else cols
    chunks = torch.chunk(expected, group, dim=scatter_dim)
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        ref = chunks[r if cluster_axis == 0 else c]
        got = ttnn.to_torch(t).float()
        assert tuple(got.shape) == tuple(ref.shape), f"device {idx}: shape {tuple(got.shape)} vs {tuple(ref.shape)}"
        passed, msg = comp_pcc(ref, got, PCC[weight_dtype])
        assert passed, f"device {idx}: {msg}"


def run_perf_case(
    mesh_device, case_name, *, cluster_axis, topology, expected_ns, label, cases=PERF_CASES, ceiling_ns=None
):
    """Traced device time of one case, checked for accuracy and gated against expected_ns (+/- MARGIN), or, with
    ceiling_ns, against that ceiling alone (a faster op passes)."""
    require_realtime_profiler("matmul_reduce_scatter_prefill perf checks")
    a_shape, w_shape, scatter_dim, dtype, weight_dtype = cases[case_name]
    group = tuple(mesh_device.shape)[cluster_axis]

    torch.manual_seed(0)
    a = torch.randn(a_shape, dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn(w_shape, dtype=torch.float32) * w_shape[0] ** -0.5).to(torch.bfloat16)
    w = quantize_like_device(w, weight_dtype)
    expected = group * (a.float() @ w.float())
    ta = replicated(mesh_device, a, dtype)
    tw = replicated(mesh_device, w, weight_dtype)

    def run_once():
        return ttnn.experimental.matmul_reduce_scatter_prefill(
            ta,
            tw,
            cluster_axis=cluster_axis,
            scatter_dim=scatter_dim,
            topology=topology,
            num_links=2,
            compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2),
        )

    eager = run_once()  # compile + program cache
    ttnn.synchronize_device(mesh_device)
    if case_name in THIN_WIDE_PERF_CASES:
        assert_thin_wide_plan(scatter_dim)
    check_replicated(eager, expected, mesh_device, cluster_axis, scatter_dim, weight_dtype)
    ttnn.deallocate(eager)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = run_once()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:

        def replay():
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

        replay()  # first-replay jitter
        ttnn.synchronize_device(mesh_device)

        def replays():
            for _ in range(REPLAYS):
                replay()

        # Back-to-back replays in one profiler window (a replay after the window's idle settle runs measurably slower
        # on small shapes). Every replay of a trace reports the same program id, so the replays are told apart per
        # chip by arrival order: replay i is the i-th of each chip's last REPLAYS records (records arrive asynchronously,
        # so the discarded warm-up replay's can land in the window first), its time the slowest chip's.
        _, records = profile_realtime_program(mesh_device, replays, collect_all=True)
        per_chip = {}
        for record in records:
            if any(KERNEL in source.replace("\\", "/") for source in record["kernel_sources"]):
                per_chip.setdefault(record["chip_id"], []).append(record["duration_ns"])
        num_chips = mesh_device.get_num_devices()
        assert len(per_chip) == num_chips and all(REPLAYS <= len(d) <= REPLAYS + 1 for d in per_chip.values()), (
            f"expected {REPLAYS} matmul_reduce_scatter_prefill records on each of {num_chips} chips, got "
            f"{ {chip: len(d) for chip, d in per_chip.items()} }"
        )
        durations = [max(d[len(d) - REPLAYS + i] for d in per_chip.values()) for i in range(REPLAYS)]
        check_replicated(traced, expected, mesh_device, cluster_axis, scatter_dim, weight_dtype)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(traced)

    median_ns = statistics.median(durations)
    logger.info(
        f"MRS-PERF {label}: median {median_ns / 1e3:.2f} us over {REPLAYS} replays "
        f"(min {min(durations) / 1e3:.2f}, max {max(durations) / 1e3:.2f}), expected "
        f"{'none' if expected_ns is None else f'{expected_ns / 1e3:.2f} us'}, ceiling "
        f"{'none' if ceiling_ns is None else f'{ceiling_ns / 1e3:.2f} us'} samples_us={[round(d / 1e3, 2) for d in durations]}"
    )
    if ceiling_ns is not None:
        assert_perf_gate(
            median_ns <= ceiling_ns,
            f"{label}: median device time {median_ns:.0f} ns over its ceiling {ceiling_ns:.0f} ns",
        )
        return
    assert expected_ns is not None, f"{label}: no expected device time yet; measured {median_ns:.0f} ns"
    lower, upper = expected_ns * (1 - MARGIN), expected_ns * (1 + MARGIN)
    assert lower <= median_ns <= upper, (
        f"{label}: device time {median_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(expected {expected_ns} ns +/- {MARGIN:.0%}); a faster op needs its expected time updated"
    )


@run_for_blackhole("matmul_reduce_scatter_prefill perf gates are Blackhole numbers")
@pytest.mark.requires_host_iommu
@skip_with_llk_assert("LLK asserts perturb kernel timing; perf checks are not meaningful with them.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("device_params", [perf_device_params(ring_device_params())], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("case_name", PERF_CASES.keys())
def test_ring_perf(mesh_device, case_name):
    run_perf_case(
        mesh_device,
        case_name,
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        expected_ns=EXPECTED_NS[("ring", case_name)],
        label=f"qb2 ring {case_name}",
    )


@run_for_blackhole("matmul_reduce_scatter_prefill perf gates are Blackhole numbers")
@pytest.mark.requires_host_iommu
@skip_with_llk_assert("LLK asserts perturb kernel timing; perf checks are not meaningful with them.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("device_params", [perf_device_params(line_device_params())], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_line_perf(mesh_device):
    run_perf_case(
        mesh_device,
        "kimi_k27_o_proj",
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        expected_ns=EXPECTED_NS[("line", "kimi_k27_o_proj")],
        label="qb2 line kimi_k27_o_proj",
    )


# GLM-5.3's matmul_reduce_scatter_prefill calls on a [RING, RING] 4x4 stage (SP 4 on mesh axis 1, TP 4 on axis 0, a
# 2,560-token chunk: 640 rows a chip, the hidden width / 4 per TP chip): MLA's q_a (mla.py _q_a_latent) and o_proj, the
# indexer's key and weights projections, and the dense FFN's down projection (tt_ffn.py).
GLM_4X4_PERF_CASES = {
    "glm53_q_a_proj": ((1, 1, 640, 1536), (1536, 2048), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "glm53_o_proj": ((1, 1, 640, 4096), (4096, 6144), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "glm53_indexer_wk": ((1, 1, 640, 1536), (1536, 128), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "glm53_indexer_weights_proj": ((1, 1, 640, 1536), (1536, 32), -2, ttnn.bfloat16, ttnn.bfloat8_b),
    "glm53_dense_ffn_down": ((1, 1, 640, 3072), (3072, 6144), -1, ttnn.bfloat16, ttnn.bfloat8_b),
}

# Ceilings [ns] per case: the slowest median over three runs on three 14 kW Galaxies (the GLM chunked perf leg's runners; the
# medians agree within 2%) plus 10%. Override per case with
# TT_METAL_MRS_RING_RING_4X4_CEILING_US='{"glm53_o_proj": 300.0}' (us).
RING_RING_4X4_CEILING_NS = {
    "glm53_q_a_proj": 62000,
    "glm53_o_proj": 195000,
    "glm53_indexer_wk": 27000,
    "glm53_indexer_weights_proj": 18000,
    "glm53_dense_ffn_down": 154000,
}


def ring_ring_4x4_ceiling_ns(case_name):
    override = json.loads(os.environ.get("TT_METAL_MRS_RING_RING_4X4_CEILING_US", "{}"))
    if case_name in override:
        return override[case_name] * 1e3
    return RING_RING_4X4_CEILING_NS[case_name]


@run_for_blackhole("matmul_reduce_scatter_prefill perf gates are Blackhole numbers")
@pytest.mark.skipif(
    ttnn.get_num_devices() != 16,
    reason="needs a 16-chip stage mesh (TT_VISIBLE_DEVICES + a 4x4 descriptor: run_subtorus_4x4_tests.sh)",
)
@skip_with_llk_assert("LLK asserts perturb kernel timing; perf checks are not meaningful with them.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("device_params", [perf_device_params(ring_device_params())], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 4)], indirect=True)
@pytest.mark.parametrize("case_name", GLM_4X4_PERF_CASES.keys())
def test_ring_ring_4x4_perf(mesh_device, case_name):
    """GLM-5.3's row-parallel projections on the 16-chip [RING, RING] carve, TP ring on mesh axis 0, each gated by its
    ceiling."""
    require_high_power("[RING, RING] 4x4 matmul_reduce_scatter_prefill perf")
    assert tuple(mesh_device.shape) == (4, 4)
    run_perf_case(
        mesh_device,
        case_name,
        cluster_axis=0,
        topology=ttnn.Topology.Ring,
        expected_ns=None,
        ceiling_ns=ring_ring_4x4_ceiling_ns(case_name),
        label=f"ring-ring 4x4 {case_name}",
        cases=GLM_4X4_PERF_CASES,
    )
