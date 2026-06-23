# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Trace-mode profiling harness for the OTHER three MLA CCLs, isolated -- the q_ag all-gather has its
own dedicated harness in test_q_ag_trace_profile.py; this file covers the remaining three CCLs the
chunked-prefill MLA forward runs along the TP axis (see test_mla_ccl.py for the op map):

    | id          | op             | dim | per-device in -> out          | sems | mla.py |
    |-------------|----------------|-----|-------------------------------|------|--------|
    | q_a_proj_rs | reduce_scatter |  3  | [1,1,640,1536] -> [1,1,640,384]| 3   | :718   |
    | kv_ag       | all_gather     |  1  | [1,1,640,576]  -> [1,4,640,576]| 2   | :797   |
    | o_proj_rs   | reduce_scatter |  3  | [1,1,640,7168] -> [1,1,640,1792]|3   | :912   |

Same goal and method as test_q_ag_trace_profile.py: the untraced unit test shows a large per-device
kernel-duration spread because each device's start-barrier wait absorbs dispatch/launch skew. Under
metal trace the captured region replays from a single synchronized go-signal, so arrival skew -> ~0
and the spread should collapse toward the real exchange floor.

Three configs, one per tracy run (profile each separately so each gets its own CSV):

    cfg id           | trace | persistent out buf | barrier sem | what it isolates
    -----------------|-------|--------------------|-------------|----------------------------------
    untraced_bar     |  no   |  no (op allocs)    |   yes       | anchor: the spread on this box
    trace_bar        |  yes  |  no (baked by cap) |   yes       | "+ trace": does sync launch help
    trace_persist    |  yes  |  yes (caller owns) |   no        | "+ persistent buffers, drop barrier"

Analyze each CSV with analyze_trace_csv.py, passing the matching OP CODE:
    reduce_scatter -> ReduceScatterMinimalAsyncDeviceOperation
    all_gather     -> AllGatherAsyncDeviceOperation   (the analyzer default)
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
KVPE_DIM = Cfg.KV_LORA_RANK + Cfg.QK_ROPE_HEAD_DIM  # 512 + 64 = 576
HIDDEN_SIZE = Cfg.EMB_SIZE  # 7168
TP_FACTOR = 4
SEQ_LOCAL = 640  # per-device seq: chunk_size_global(5120) / sp(8) on the 8x4 Galaxy

# (ccl_id, kind, dim, feat)  -- feat is the per-device input width on the gathered/scattered dim.
#   rs:        out width = feat // tp   (3 ccl sems)
#   ag dim=3:  out width = feat * tp    (2 ccl sems)   [q_ag lives in its own file]
#   ag dim=1:  gather over the head dim, out heads = tp (2 ccl sems)
MLA_CCL_OPS = [
    ("q_a_proj_rs", "rs", 3, Q_LORA_RANK),  # mla.py:718
    ("kv_ag", "ag", 1, KVPE_DIM),  # mla.py:797
    ("o_proj_rs", "rs", 3, HIDDEN_SIZE),  # mla.py:912
]

# Ring fabric + a trace region big enough for the captured CCLs (matches test_q_ag_trace_profile).
RING_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
    "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    "trace_region_size": 90000000,
}


def _make_sems(mesh_device, cores, n):
    return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(n)]


def _run_mla_ccl_trace(
    mesh_device, *, kind, dim, feat, trace_mode, use_persistent, use_barrier, warmup_iters, num_iters
):
    tp_axis = 1
    sp, tp = list(mesh_device.shape)
    num_links = 2 if is_blackhole() else 1
    topology = ttnn.Topology.Ring
    if kind == "rs":
        assert (feat // tp) % 32 == 0, f"rs scatter width {feat}//{tp} must be tile-aligned"

    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    # All semaphores allocated ONCE (trace requires stable addresses). rs uses 3 ccl sems/iter, ag 2.
    sems_per_iter = 3 if kind == "rs" else 2
    ccl_sems = _make_sems(mesh_device, ccl_crs, num_iters * sems_per_iter)
    barrier_sems = _make_sems(mesh_device, ccl_crs, 2) if use_barrier else None

    # Each of the tp devices holds an independent [1,1,S,feat] slice (dim1 shard); SP replicates.
    torch_in = torch.randn(1, tp, SEQ_LOCAL, feat, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, 1]),
    )

    # Caller-owned persistent output buffer(s), replicated over the mesh.
    #   ag: single output tensor (its full gathered shape)
    #   rs: [intermediate (= per-device input shape), output (= scattered shape)]
    persist_ag_out = None
    persist_rs_bufs = None
    if use_persistent:

        def _replicated(shape):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, None]),
            )

        if kind == "ag":
            ag_shape = (1, 1, SEQ_LOCAL, feat * tp) if dim == 3 else (1, tp, SEQ_LOCAL, feat)
            persist_ag_out = _replicated(ag_shape)
        else:  # rs
            persist_rs_bufs = [_replicated((1, 1, SEQ_LOCAL, feat)), _replicated((1, 1, SEQ_LOCAL, feat // tp))]

    def one_call(i):
        bsem = barrier_sems[i % 2] if use_barrier else None
        if kind == "ag":
            sems = ccl_sems[2 * i : 2 * i + 2]
            args = [tt_in] + ([persist_ag_out] if use_persistent else [])
            return ttnn.experimental.all_gather_async(
                *args,
                dim=dim,
                multi_device_global_semaphore=sems,
                num_links=num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                cluster_axis=tp_axis,
                barrier_semaphore=bsem,
                subdevice_id=worker_sub_device_id,
            )
        # rs
        sems = ccl_sems[3 * i : 3 * i + 3]
        return ttnn.experimental.reduce_scatter_minimal_async(
            tt_in,
            persistent_output_buffers=persist_rs_bufs if use_persistent else None,
            dim=dim,
            multi_device_global_semaphore=sems,
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

        # --- correctness sanity check on the final output ---
        out_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, dim)),
        )[0:1]
        if kind == "rs":
            golden = torch_in.to(torch.float32).sum(dim=1, keepdim=True)
            out_torch = out_torch.to(torch.float32)
        elif dim == 3:
            golden = torch.cat([torch_in[:, d : d + 1] for d in range(tp)], dim=3)
            out_torch = out_torch[:, :, :, : feat * tp]
        else:  # ag dim == 1 (head gather)
            golden = torch_in
            out_torch = out_torch[:, :tp]
        passed, msg = comp_pcc(out_torch, golden, 0.999)
        logger.info(
            f"{kind} dim={dim} feat={feat} trace={trace_mode} persist={use_persistent} bar={use_barrier} PCC: {msg}"
        )
        assert passed, f"{kind} dim={dim} feat={feat} FAILED: {msg}"
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
@pytest.mark.parametrize("ccl_id, kind, dim, feat", MLA_CCL_OPS, ids=[c[0] for c in MLA_CCL_OPS])
@pytest.mark.parametrize("cfg_id, trace_mode, use_persistent, use_barrier", CONFIGS, ids=[c[0] for c in CONFIGS])
@pytest.mark.timeout(0)
def test_mla_ccl_trace_profile(
    mesh_device,
    device_params,
    warmup_iters,
    num_iters,
    ccl_id,
    kind,
    dim,
    feat,
    cfg_id,
    trace_mode,
    use_persistent,
    use_barrier,
):
    """Profile one of the three other MLA CCLs (q_a_proj_rs / kv_ag / o_proj_rs) under the three
    trace configs (see module docstring). Run one (op, config) at a time under tracy so each
    produces its own ops_perf CSV."""
    _run_mla_ccl_trace(
        mesh_device,
        kind=kind,
        dim=dim,
        feat=feat,
        trace_mode=trace_mode,
        use_persistent=use_persistent,
        use_barrier=use_barrier,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
    )
