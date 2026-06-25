# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Device-perf measurement for the TP-axis all-gathers introduced by the DeepSeek sparse-MLA
head->sequence reshard (GLM-5.1, 64 q-heads, tp=4). See models/demos/deepseek_v3_d_p/tt/mla.

The reshard gathers run over the TP axis (N=4 devices), 2 links, bf16, DRAM. This is the same
4-device gather whether the box is an 8-chip LoudBox (opened (4,2)) or a 32-chip 8x4 (opened
(4,8)) — only the TP axis (size 4) participates; the SP axis just replicates the op.

Two topology variants are parametrized (id "linear" / "ring"):
  - linear : FABRIC_1D + Topology.Linear  — what MLA runs today (mla.py); works on any box.
  - ring   : FABRIC_1D_RING + Topology.Ring — needs the 4-axis physically wired as a ring/torus
             (target 8x4 box). The 8-chip LoudBox is NOT ring-wired on that axis (FABRIC_1D_RING
             errors "no forwarding direction D0->D3"), so run "ring" only on target HW.

ISL = 5120 with sp=8 -> 640 rows/device (under reshard the attention op further splits seq
across tp so the per-op seq is 160; the *gather* input shapes below already encode that).

Each case = the FULL (gathered) output shape + gather dim; per-device input = output/N along dim.
  q_heads     attn  _sparse_mla   q-heads gather (dim1):  [1,16,640,576] -> [1,64,640,576]
  out_seq     attn  _sparse_mla   out-seq gather (dim2):  [1,64,160,512] -> [1,64,640,512]
  qdev_heads  idx   forward       q_dev heads gather(dim1):[1, 8,640,128] -> [1,32,640,128]
  wts         idx   _tp_rs_ag     weights all-gather(dim3):[1, 1,640, 8] -> [1, 1,640, 32]

The runner test (test_reshard_ag_run) executes one gather under trace with signposts so the
profiler can time it. The perf test (test_reshard_ag_perf) re-invokes the runner under the
device profiler, reads the measured DEVICE KERNEL duration and the op's own PM IDEAL [ns]
column, and logs the utilization (PM IDEAL / measured).
"""

import math

import pytest
import ttnn
from loguru import logger

# NOTE: run_all_gather_impl / validate_test live in TEST modules whose module-level @skip_for_*_dev
# decorators call ttnn.get_num_devices() at IMPORT time (opening the cluster + holding CHIP_IN_USE).
# They are therefore imported lazily inside the runner body so that merely collecting/importing this
# file in the outer perf process never opens the device. See the runner for the full rationale.

# case_id -> (full gathered output shape, gather dim)
RESHARD_AG_CASES = {
    "q_heads": ([1, 64, 640, 576], 1),
    "out_seq": ([1, 64, 640, 512], 2),
    "qdev_heads": ([1, 32, 640, 128], 1),
    "wts": ([1, 1, 640, 32], 3),
}
CASE_IDS = list(RESHARD_AG_CASES.keys())

N_TP = 4  # devices along the gather (TP) axis
NUM_LINKS = 2

# Mesh shapes to open via the generic `mesh_device` fixture (opens EXACTLY this shape + handles fabric).
# The TP gather is over whichever axis is size N_TP: (8,4) -> axis1 (target 8x4 GLM layout, sp=8/tp=4);
# (4,2) -> axis0 (the 8-chip LoudBox). The fixture auto-skips a shape that exceeds available devices.
MESH_SHAPES = [(8, 4), (4, 2)]
MESH_IDS = ["8x4", "4x2"]

# (device_params, ttnn.Topology) coupled per topology variant; ids drive `-k` selection + analytic branch.
TRACE_REGION = 90112
TOPOLOGIES = [
    ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}, ttnn.Topology.Linear),
    ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": TRACE_REGION}, ttnn.Topology.Ring),
]
TOPOLOGY_IDS = ["linear", "ring"]

# Blackhole fabric constants, mirrored from
# ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp (lookup_fabric_link_bw / lookup_fabric_hop_latency_ns).
BH_LINK_BW_GBPS = 50.0  # 400 Gbps per link == 50 B/ns
BH_HOP_LAT_NS_1D = 515.0
BH_PEAK_DRAM_GBPS = 512.0


def _tile_pad_bytes(shape, elem_size=2):
    """Physical (tile-padded) byte size the perf model sees: last two dims rounded up to 32."""
    *lead, h, w = shape
    h = math.ceil(h / 32) * 32
    w = math.ceil(w / 32) * 32
    vol = w * h
    for d in lead:
        vol *= d
    return vol * elem_size


def analytic_pm_ideal_ns(out_shape, dim, is_ring, n=N_TP, num_links=NUM_LINKS):
    """First-principles all-gather roofline (max of the competing floors), matching
    AllGatherAsyncDeviceOperation::create_op_performance_model for Blackhole."""
    in_shape = list(out_shape)
    in_shape[dim] //= n
    s = _tile_pad_bytes(in_shape)  # bytes each device contributes
    if is_ring:
        bottleneck = math.ceil((n - 1) * s / 2)  # Ring bisection cuts 2 links
        hops = math.ceil((n - 1) / 2)
    else:
        bottleneck = (n - 1) * s  # Line: edge link carries all (N-1) slices
        hops = n - 1
    fabric_bw_ns = bottleneck / (BH_LINK_BW_GBPS * num_links)
    fabric_fill_ns = BH_HOP_LAT_NS_1D * hops
    dram_ns = _tile_pad_bytes(out_shape) / BH_PEAK_DRAM_GBPS  # coarse output-bandwidth floor
    return max(fabric_bw_ns, fabric_fill_ns, dram_ns)


# --------------------------------------------------------------------------------------------------
# Runner: executes ONE gather under trace + signposts so the profiler can time it.
# --------------------------------------------------------------------------------------------------
# Uses the generic `mesh_device` fixture (NOT bh_2d_mesh_device, which hardcodes the mesh shape and
# would force a 32-chip box to (4,8)); the shape is parametrized explicitly so the target 8x4 layout
# is exact. Device gating is IN-BODY, NOT via @skip_for_*_dev decorators: those call get_num_devices()
# at module-import time, opening the cluster + holding CHIP_IN_USE in whatever process imports this
# file — including the outer perf test that only spawns a subprocess. In-body checks run only here
# (the subprocess), the legitimate device owner.
@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("device_params, topology", TOPOLOGIES, indirect=["device_params"], ids=TOPOLOGY_IDS)
@pytest.mark.parametrize("mesh_device", MESH_SHAPES, indirect=True, ids=MESH_IDS)
@pytest.mark.timeout(900)  # cold JIT kernel compile can take ~5 min; override the 300s default
def test_reshard_ag_run(mesh_device, case_id, num_iters, topology):
    from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
    from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test

    if ttnn.get_arch_name() == "wormhole_b0":
        pytest.skip("Reshard AG perf targets Blackhole")

    ag_output_shape, dim = RESHARD_AG_CASES[case_id]

    # The TP gather spans N_TP devices: pick whichever mesh axis has that size.
    # (8,4) -> axis1 (sp=8/tp=4 GLM layout); (4,2) -> axis0 (LoudBox).
    shape = list(mesh_device.shape)
    cluster_axis = 0 if shape[0] == N_TP else 1
    if shape[cluster_axis] != N_TP:
        pytest.skip(f"No mesh axis of size {N_TP} in shape {tuple(shape)}")
    submesh_shape = (N_TP, 1) if cluster_axis == 0 else (1, N_TP)

    validate_test(N_TP, topology, mesh_device.shape, cluster_axis)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape))

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_all_gather_impl(
        submesh_device,
        N_TP,
        ag_output_shape,
        dim,
        NUM_LINKS,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        dram,
        dram,
        all_gather_topology=topology,
        enable_trace=True,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=20,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
        allowed_pcc=0.9999,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


# --------------------------------------------------------------------------------------------------
# Perf: re-invoke the runner under the profiler, read measured DEVICE KERNEL + PM IDEAL, log util.
# --------------------------------------------------------------------------------------------------
# NOTE: this outer test MUST NOT touch the device — no ttnn.get_num_devices(), no skip_for_*_dev
# decorators (they call get_num_devices at collection time, which opens the UMD cluster and holds
# the CHIP_IN_USE lock for the process lifetime). Its only job is to spawn the profiler subprocess,
# which is the sole legitimate device owner (test_reshard_ag_run). A held lock here deadlocks that
# child. The 8-device / arch gating lives entirely in the runner.
@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("topo_id, is_ring", [("linear", False), ("ring", True)], ids=["linear", "ring"])
@pytest.mark.parametrize("mesh_id", MESH_IDS)
@pytest.mark.parametrize("warmup_iters", [5])
@pytest.mark.timeout(1200)  # the profiler subprocess re-opens the device + JIT-compiles kernels (~5 min cold)
@pytest.mark.models_device_performance_bare_metal
def test_reshard_ag_perf(case_id, topo_id, is_ring, mesh_id, warmup_iters):
    from models.perf.device_perf_utils import run_device_perf_detailed
    from tracy.process_model_log import post_process_ops_log

    if case_id == "wts":
        # Per-device shard is sub-tile (dim3 = 32/4 = 8 < 32), so this gather does NOT lower to a
        # single AllGatherAsync — it becomes AllBroadcast + Tilize/Untilize/Permute/Concat, with no
        # AllGatherAsyncDeviceOperation row to read PM IDEAL from. It is fill-bound (~1.5 us ideal)
        # and negligible vs the head/seq gathers, so it is not measured as an all-gather here.
        pytest.skip("wts gather is sub-tile -> lowers to AllBroadcast, not AllGatherAsync; negligible")

    ag_output_shape, dim = RESHARD_AG_CASES[case_id]
    subdir = "reshard_ag_perf"
    op_name = "AllGatherAsyncDeviceOperation"
    cols = ["DEVICE KERNEL"]
    # Select the matching runner variant: `-k '<case_id> and <topo_id> and <mesh_id>'` — these are the
    # case_id / topology / mesh parametrize ids on test_reshard_ag_run.
    command = f"pytest {__file__}::test_reshard_ag_run -k '{case_id} and {topo_id} and {mesh_id}'"

    # The runner self-skips combos invalid on this box (mesh too big, or ring on a non-ring-wired axis →
    # the gather errors). Then no AllGatherAsync is timed; surface that as a skip, not a confusing error.
    try:
        results = run_device_perf_detailed(
            command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters
        )
        pm_ideal_arr = post_process_ops_log(
            subdir, ["PM IDEAL [ns]"], sum_vals=False, op_name=op_name, has_signposts=True
        )["PM IDEAL [ns]"]
    except (ValueError, IndexError, KeyError) as e:
        pytest.skip(
            f"No AllGatherAsync timed for {case_id}/{topo_id}/{mesh_id} on this box (runner skipped/invalid combo): {e}"
        )

    measured_avg_ns = results[cols[0]]["AVG"]
    measured_min_ns = results[cols[0]]["MIN"]

    # The op's own roofline straight from the profiler CSV column the user referenced.
    pm_ideal_csv_ns = float(sorted(pm_ideal_arr)[len(pm_ideal_arr) // 2])  # median
    pm_ideal_match_ns = analytic_pm_ideal_ns(ag_output_shape, dim, is_ring=is_ring)  # this topology
    pm_ideal_other_ns = analytic_pm_ideal_ns(ag_output_shape, dim, is_ring=not is_ring)  # the other, ref

    util_csv = 100.0 * pm_ideal_csv_ns / measured_avg_ns
    util_match = 100.0 * pm_ideal_match_ns / measured_avg_ns
    topo_name = "Ring" if is_ring else "Linear"
    other_name = "Linear" if is_ring else "Ring"

    logger.info(
        f"\n[reshard AG: {case_id} / {topo_name}] out={ag_output_shape} dim={dim} N={N_TP} links={NUM_LINKS}\n"
        f"  measured DEVICE KERNEL    : avg {measured_avg_ns/1000:.2f} us | min {measured_min_ns/1000:.2f} us\n"
        f"  PM IDEAL (csv column)     : {pm_ideal_csv_ns/1000:.2f} us  -> util {util_csv:.1f}%\n"
        f"  PM IDEAL (analytic {topo_name:6}): {pm_ideal_match_ns/1000:.2f} us  -> util {util_match:.1f}%\n"
        f"  PM IDEAL (analytic {other_name:6}): {pm_ideal_other_ns/1000:.2f} us  (reference)"
    )

    assert measured_avg_ns > 0, "No AllGatherAsync op timed — check signposts / op_name"
