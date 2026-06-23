# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Trace-mode profiling harness for the MLA q_ag all-gather (mla.py:729), isolated.

Goal: separate the per-device kernel-duration spread seen in the untraced unit test into its
components. The untraced all_gather shows a ~4-5x spread across devices because each device's
start-barrier wait absorbs dispatch/launch skew. Under metal trace the whole captured region
replays from a single synchronized go-signal, so arrival skew -> ~0 and the spread should collapse
toward the real exchange floor.

Three configs, one per tracy run (profile each separately so each gets its own CSV):

    cfg id           | trace | persistent out buf | barrier sem | what it isolates
    -----------------|-------|--------------------|-------------|----------------------------------
    untraced_bar     |  no   |  no (op allocs)    |   yes       | anchor: the spread on this 1x8 box
    trace_bar        |  yes  |  no (baked by cap) |   yes       | "+ trace": does sync launch help
    trace_persist    |  yes  |  yes (caller owns) |   no        | "+ persistent buffers, drop barrier"

q_ag shape: per-device [1,1,640,384] -> gather dim=3 over the tp ring -> [1,1,640,384*tp].
On the 8-chip Blackhole box this runs as a 1x8 ring (tp=8).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config as Cfg
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

try:
    from tracy import signpost
except ImportError:  # tracy not always importable outside the profiler

    def signpost(*_a, **_k):
        pass


Q_LORA_RANK = Cfg.Q_LORA_RANK  # 1536
TP_FACTOR = 4
SEQ_LOCAL = 640
FEAT = Q_LORA_RANK // TP_FACTOR  # 384, the per-device q_ag input width
DIM = 3

# Ring fabric + a trace region big enough for the captured all-gathers.
RING_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
    "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    "trace_region_size": 90000000,
}


def _make_sems(mesh_device, cores, n):
    return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(n)]


def _run_q_ag_trace(mesh_device, *, trace_mode, use_persistent, use_barrier, warmup_iters, num_iters):
    tp_axis = 1
    sp, tp = list(mesh_device.shape)
    num_links = 2 if is_blackhole() else 1
    topology = ttnn.Topology.Ring

    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    # All semaphores allocated ONCE (trace requires stable addresses). 2 ccl sems per iter.
    ccl_sems = _make_sems(mesh_device, ccl_crs, num_iters * 2)
    barrier_sems = _make_sems(mesh_device, ccl_crs, 2) if use_barrier else None

    torch_in = torch.randn(1, tp, SEQ_LOCAL, FEAT, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, 1]),
    )

    # Caller-owned persistent output buffer: per-device [1,1,640,FEAT*tp], replicated over the mesh.
    persistent_out = None
    if use_persistent:
        persistent_out = ttnn.from_torch(
            torch.zeros(1, 1, SEQ_LOCAL, FEAT * tp, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, None]),
        )

    def one_call(i):
        sems = [ccl_sems[2 * i], ccl_sems[2 * i + 1]]
        bsem = barrier_sems[i % 2] if use_barrier else None
        if use_persistent:
            return ttnn.experimental.all_gather_async(
                tt_in,
                persistent_out,
                dim=DIM,
                multi_device_global_semaphore=sems,
                num_links=num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                cluster_axis=tp_axis,
                barrier_semaphore=bsem,
                subdevice_id=worker_sub_device_id,
            )
        return ttnn.experimental.all_gather_async(
            tt_in,
            dim=DIM,
            multi_device_global_semaphore=sems,
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            cluster_axis=tp_axis,
            barrier_semaphore=bsem,
            subdevice_id=worker_sub_device_id,
        )

    try:
        if trace_mode:
            # Compile/allocate once outside capture.
            tt_out = one_call(0)
            ttnn.synchronize_device(mesh_device)

            if warmup_iters > 0:
                trace_warm = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                for i in range(warmup_iters):
                    tt_out = one_call(i % num_iters)
                ttnn.end_trace_capture(mesh_device, trace_warm, cq_id=0)

            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for i in range(num_iters):
                tt_out = one_call(i)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)

            if warmup_iters > 0:
                ttnn.execute_trace(mesh_device, trace_warm, blocking=False)
            signpost("start")
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
            signpost("stop")
        else:
            signpost("start")
            for i in range(num_iters):
                tt_out = one_call(i)
            ttnn.synchronize_device(mesh_device)
            signpost("stop")

        # Correctness sanity check on the final output (each device holds the full concat over tp).
        out_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, DIM)),
        )[0:1]
        golden = torch.cat([torch_in[:, d : d + 1] for d in range(tp)], dim=3)
        out_torch = out_torch[:, :, :, : FEAT * tp]
        passed, msg = comp_pcc(out_torch, golden, 0.999)
        logger.info(f"q_ag trace={trace_mode} persist={use_persistent} barrier={use_barrier} PCC: {msg}")
        assert passed, f"q_ag FAILED: {msg}"
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


# (cfg_id, trace_mode, use_persistent, use_barrier)
CONFIGS = [
    ("untraced_bar", False, False, True),
    ("trace_bar", True, False, True),
    ("trace_persist", True, True, False),
]


@pytest.mark.parametrize("device_params", [RING_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8), (8, 4)], ids=["1x8", "8x4"], indirect=True)
@pytest.mark.parametrize("warmup_iters, num_iters", [(10, 20)], ids=["w10n20"])
@pytest.mark.parametrize("cfg_id, trace_mode, use_persistent, use_barrier", CONFIGS, ids=[c[0] for c in CONFIGS])
@pytest.mark.timeout(0)
def test_q_ag_trace_profile(
    mesh_device, device_params, warmup_iters, num_iters, cfg_id, trace_mode, use_persistent, use_barrier
):
    """Profile the q_ag all-gather on a 1x8 ring under three configs (see module docstring).
    Run one config at a time under tracy so each produces its own ops_perf CSV."""
    _run_q_ag_trace(
        mesh_device,
        trace_mode=trace_mode,
        use_persistent=use_persistent,
        use_barrier=use_barrier,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
    )
