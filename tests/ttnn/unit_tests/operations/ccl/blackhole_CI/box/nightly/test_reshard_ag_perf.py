# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
By-hand trace-mode profiling of the TP-axis all-gathers added by the DeepSeek sparse-MLA
head->sequence reshard (GLM-5.1, 64 q-heads, tp=4). See models/demos/deepseek_v3_d_p/tt/mla.

Mirrors the CCL-by-hand harness on branch ipotkonjak/ccl_ring_debug
(models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_mla_ccl_trace_profile.py): run the op
directly on the full mesh over the TP cluster axis (NO submesh) with the model's sub-device /
semaphore scaffolding, capture under metal trace, bracket with tracy signposts, and read the
op's PM IDEAL [ns] / DEVICE KERNEL duration out of the ops_perf CSV.

The reshard runs three TP all-gathers (tp_axis=1; the gather is over whichever tensor dim the
reshard transposes). feat/seq are the PER-DEVICE shape on the 8x4 Galaxy with a 5120-token chunk
(sp=8 -> seq_local = 5120/8 = 640). Running on a 2x4 (or 1x4) mesh reproduces the exact per-device
all-gather shape (the gather is on the TP axis = 4 on all of these; SP just replicates the op).

    | id          | dim | per-device in -> out (gathered)        | mla.py            |
    |-------------|-----|----------------------------------------|-------------------|
    | q_heads     |  1  | [1,16,640,576] -> [1,64,640,576]       | _sparse_mla       |
    | out_seq     |  2  | [1,64,160,512] -> [1,64,640,512]       | _sparse_mla (inv) |
    | qdev_heads  |  1  | [1, 8,640,128] -> [1,32,640,128]       | TtIndexer.forward |

(The indexer weights all-gather is sub-tile (per-device width 8) -> lowers to AllBroadcast, not
AllGatherAsync, and is fill-bound/negligible, so it is not profiled here.)

A/B axis `ag_impl` selects the op: "old" = ttnn.experimental.all_gather_async (OP CODE
AllGatherAsyncDeviceOperation), "new" = ttnn.all_gather (OP CODE AllGatherDeviceOperation, links/
topology/semaphores managed internally).

Run ONE (impl, op, config, mesh, topology, dtype) at a time under tracy so each gets its own CSV, e.g.:
    pytest .../test_reshard_ag_perf.py -k "new and q_heads and trace_bar and 2x4 and ring and bf16"
Then read PM IDEAL [ns] vs DEVICE KERNEL DURATION [ns] for the impl's OP CODE
from generated/profiler/.../ops_perf_results_*.csv.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

try:
    from tracy import signpost
except ImportError:  # tracy not always importable outside the profiler

    def signpost(*_a, **_k):
        pass


TP_FACTOR = 4  # production TP; tp_axis size on 1x4 / 2x4 / 8x4
SEQ_LOCAL = 640  # per-device seq: chunk_size_global(5120) / sp(8) on the 8x4 Galaxy

# (id, gather_dim, FULL gathered shape). Per-device input = full shape with gather_dim // tp.
# gather_dim=1 -> head gather (heads/tp per device); gather_dim=2 -> sequence gather (seq/tp per device).
RESHARD_AG_OPS = [
    ("q_heads", 1, [1, 64, SEQ_LOCAL, 576]),  # attn absorbed-MQA q heads
    ("out_seq", 2, [1, 64, SEQ_LOCAL, 512]),  # attn output (inverse seq gather)
    ("qdev_heads", 1, [1, 32, SEQ_LOCAL, 128]),  # indexer q_dev heads
]

# (device_params, ttnn.Topology); device_params is consumed by the mesh_device fixture (indirect).
# fabric_router_config + worker_l1_size mirror the working MLA tests (test_ds_mla / chunked) so the
# *_erisc fabric kernels build identically to the proven model path.
DEVICE_PARAMS_TOPOLOGY = [
    (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
            "trace_region_size": 90000000,
        },
        ttnn.Topology.Linear,
    ),
    (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
            "trace_region_size": 90000000,
        },
        ttnn.Topology.Ring,
    ),
]
DEVICE_PARAMS_TOPOLOGY_IDS = ["line", "ring"]

# (cfg_id, trace_mode, use_persistent, use_barrier) — same isolation configs as the ccl_ring_debug harness.
CONFIGS = [
    ("untraced_bar", False, False, True),
    ("trace_bar", True, False, True),
    ("trace_persist", True, True, False),
]

# (dtype, pcc_threshold). bfp8 is the deployment ("end game") dtype: ~half the bytes per tile vs bf16
# (1B mantissa/elem + shared exponent per 16-elem block), so PM IDEAL and time roughly halve. all_gather
# is pure data movement, so the only precision loss is the bf16->bfp8 quantization at from_torch -> a
# slightly looser PCC bar.
AG_DTYPES = [
    (ttnn.bfloat16, 0.999),
    (ttnn.bfloat8_b, 0.99),
]
AG_DTYPE_IDS = ["bf16", "bfp8"]


def _make_sems(mesh_device, cores, n):
    return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(n)]


def _run_reshard_ag(
    mesh_device,
    *,
    gather_dim,
    full_shape,
    topology,
    trace_mode,
    use_persistent,
    use_barrier,
    warmup_iters,
    num_iters,
    ag_dtype=ttnn.bfloat16,
    pcc_threshold=0.999,
    ag_impl="old",
):
    tp_axis = 1
    sp, tp = list(mesh_device.shape)
    num_links = 2 if is_blackhole() else 1
    assert full_shape[gather_dim] % tp == 0, f"gather dim {full_shape[gather_dim]} not divisible by tp={tp}"

    # A/B: "old" = ttnn.experimental.all_gather_async (caller-owned ccl/barrier sems, explicit num_links +
    # topology; CSV OP CODE AllGatherAsyncDeviceOperation). "new" = ttnn.all_gather (semaphores, links and
    # topology managed internally; persistent buffer via output_tensor=; OP CODE AllGatherDeviceOperation).
    is_new = ag_impl == "new"

    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    # All semaphores allocated ONCE (trace needs stable addresses). all_gather_async uses 2 ccl sems/iter;
    # the new ttnn.all_gather manages its own semaphores, so none are allocated for it.
    ccl_sems = None if is_new else _make_sems(mesh_device, ccl_crs, num_iters * 2)
    barrier_sems = _make_sems(mesh_device, ccl_crs, 2) if (use_barrier and not is_new) else None

    # Build the FULL gathered tensor, shard gather_dim across the TP axis (each chip owns its 1/tp
    # slice), replicate across SP. all_gather over TP then reassembles the full tensor on every chip.
    torch_full = torch.randn(*full_shape, dtype=torch.bfloat16)
    in_dims = [None, None]
    in_dims[tp_axis] = gather_dim
    tt_in = ttnn.from_torch(
        torch_full,
        dtype=ag_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=in_dims),
    )

    persist_out = None
    if use_persistent:
        persist_out = ttnn.from_torch(
            torch.zeros(*full_shape, dtype=torch.bfloat16),
            dtype=ag_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, None]),
        )

    def one_call(i):
        if is_new:
            # New op: links/topology/semaphores are internal; persistent buffer via output_tensor=.
            return ttnn.all_gather(
                tt_in,
                dim=gather_dim,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tensor=(persist_out if use_persistent else None),
                cluster_axis=tp_axis,
                subdevice_id=worker_sub_device_id,
            )
        bsem = barrier_sems[i % 2] if use_barrier else None
        sems = ccl_sems[2 * i : 2 * i + 2]
        args = [tt_in] + ([persist_out] if use_persistent else [])
        return ttnn.experimental.all_gather_async(
            *args,
            dim=gather_dim,
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
            tt_out = one_call(0)  # compile/allocate once outside capture
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

        # Flush the device-side profiler buffers to host, else tracy reports "No device logs found"
        # and the ops_perf CSV has no DEVICE KERNEL DURATION / PM IDEAL rows.
        ttnn.ReadDeviceProfiler(mesh_device)

        # --- correctness: every chip holds the full gather; the tp concat stacks identical copies
        #     along gather_dim, so narrow back to the first copy and compare to the full tensor. ---
        out_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, gather_dim)),
        )[0:1]
        out_torch = out_torch.narrow(gather_dim, 0, full_shape[gather_dim])
        passed, msg = comp_pcc(out_torch, torch_full, pcc_threshold)
        op_code = "AllGatherDeviceOperation" if is_new else "AllGatherAsyncDeviceOperation"
        logger.info(
            f"reshard AG impl={ag_impl} OP_CODE={op_code} dim={gather_dim} full={full_shape} dtype={ag_dtype} "
            f"trace={trace_mode} persist={use_persistent} bar={use_barrier} mesh={sp}x{tp} {topology} PCC: {msg}"
        )
        assert passed, f"reshard AG dim={gather_dim} full={full_shape} FAILED: {msg}"
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize("ag_impl", ["old", "new"], ids=["old", "new"])
@pytest.mark.parametrize("ag_id, gather_dim, full_shape", RESHARD_AG_OPS, ids=[c[0] for c in RESHARD_AG_OPS])
@pytest.mark.parametrize("ag_dtype, pcc_threshold", AG_DTYPES, ids=AG_DTYPE_IDS)
@pytest.mark.parametrize(
    "device_params, topology", DEVICE_PARAMS_TOPOLOGY, indirect=["device_params"], ids=DEVICE_PARAMS_TOPOLOGY_IDS
)
@pytest.mark.parametrize("mesh_device", [(8, 4), (2, 4), (1, 4)], ids=["8x4", "2x4", "1x4"], indirect=True)
@pytest.mark.parametrize("warmup_iters, num_iters", [(10, 20)], ids=["w10n20"])
@pytest.mark.parametrize("cfg_id, trace_mode, use_persistent, use_barrier", CONFIGS, ids=[c[0] for c in CONFIGS])
@pytest.mark.timeout(0)
def test_reshard_ag(
    mesh_device,
    device_params,
    topology,
    ag_id,
    gather_dim,
    full_shape,
    ag_dtype,
    pcc_threshold,
    warmup_iters,
    num_iters,
    cfg_id,
    trace_mode,
    use_persistent,
    use_barrier,
    ag_impl,
):
    """One reshard TP all-gather, run by hand on the full mesh over the TP cluster axis. Run one
    (impl, op, dtype, config, mesh, topology) at a time under tracy so each produces its own ops_perf CSV;
    read PM IDEAL [ns] vs DEVICE KERNEL DURATION [ns] for the impl's OP CODE (old=AllGatherAsyncDeviceOperation,
    new=AllGatherDeviceOperation)."""
    if not is_blackhole():
        pytest.skip("Reshard AG perf targets Blackhole")
    _run_reshard_ag(
        mesh_device,
        gather_dim=gather_dim,
        full_shape=full_shape,
        topology=topology,
        trace_mode=trace_mode,
        use_persistent=use_persistent,
        use_barrier=use_barrier,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
        ag_dtype=ag_dtype,
        pcc_threshold=pcc_threshold,
        ag_impl=ag_impl,
    )
