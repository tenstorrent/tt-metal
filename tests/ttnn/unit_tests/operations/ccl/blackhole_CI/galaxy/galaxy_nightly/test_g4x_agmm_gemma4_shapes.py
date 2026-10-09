# LOCAL EXPERIMENT (kmabee, not for review): fused vs separate AG+matmul at Gemma4 8k-step shapes on BH Galaxy 8x4.
# Hidden-sharded TP residual: rows split 8 ways (CP), K gathered over the 4-way TP axis, weight N split over TP.
import os

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_strided_all_gather_minimal_matmul_async import (
    run_strided_all_gather_minimal_matmul_impl,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.galaxy.galaxy_nightly.test_strided_all_gather_minimal_matmul_async_bh import (
    create_fabric_router_config,
)

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


_CASES = {  # name: (rows per device, N per device, N block, fused activation, separate AG + matmul)
    "gate/fused": (1024, 5376, 448, "gelu_tanh", False),  # MLP gate at chunk 8192 (fused GELU)
    "up/fused": (1024, 5376, 448, None, False),  # MLP up
    "up/separate": (1024, 5376, 448, None, True),
    "qkv/fused": (1024, 4608, 384, None, False),  # global QKV
    "qkv/separate": (1024, 4608, 384, None, True),
}
# G4X_AGMM_CASE selects one case at collection time, so one tracy capture holds one case.
_SELECTED = [c for c in _CASES if os.environ.get("G4X_AGMM_CASE", c) == c]


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("case", _SELECTED)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(8192),
                "trace_region_size": 1171456,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_g4x_agmm(mesh_device, case, all_gather_topology):
    rows, n_local, block_n, activation, use_non_fused = _CASES[case]
    run_strided_all_gather_minimal_matmul_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        8 * rows,
        5376,
        4 * n_local,
        3,
        2,
        2,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        DRAM,
        DRAM,
        DRAM,
        all_gather_topology=all_gather_topology,
        enable_trace=True,
        num_iters=10,
        num_workers_per_link=3,
        num_buffers_per_channel=8,
        mm_block_m=128,
        mm_block_k=256,
        mm_block_n=block_n,
        subblock_h=2,
        subblock_w=2,
        mm_core_grid=ttnn.CoreCoord(12, 8),
        use_non_fused=use_non_fused,
        activation=activation,
        shard_weights=True,
        ag_core_grid_offset=(0, 8),
        read_local_slice_from_input=True,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_acc=True,
        allowed_pcc=0.99,
    )


# Direct harness: fused AG+MM vs matmul alone on a pre-gathered input, with the weight dtype as a parameter.
# Inputs are random bf16; no accuracy check (timing probe). Select with G4X_AGMM_DIRECT="<mode>/<wdtype>/<shape>".
_DIRECT = {}
for _mode in ("fused", "mm_only"):
    for _wd in ("bf16", "bfp8"):
        for _shape, (_n, _bn, _act) in {"up": (5376, 448, None), "gate": (5376, 448, "gelu_tanh"), "qkv": (4608, 384, None)}.items():
            _DIRECT[f"{_mode}/{_wd}/{_shape}"] = (_mode, _wd, _n, _bn, _act)
_DIRECT_SELECTED = [c for c in _DIRECT if os.environ.get("G4X_AGMM_DIRECT", "none") == c]


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("case", _DIRECT_SELECTED)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(8192),
            "trace_region_size": 1171456,
        }
    ],
    indirect=True,
    ids=["fabric_ring"],
)
def test_g4x_agmm_direct(mesh_device, case):
    import torch
    from tracy import signpost
    from tests.nightly.t3000.ccl.test_strided_all_gather_minimal_matmul_async import create_global_semaphores

    mode, wd, n_local, block_n, act = _DIRECT[case]
    rows, K, iters = 1024, 5376, 10
    wdtype = ttnn.bfloat16 if wd == "bf16" else ttnn.bfloat8_b
    shape = tuple(mesh_device.shape)
    grid = ttnn.CoreCoord(12, 8)
    w = ttnn.from_torch(
        torch.randn(1, 1, K, 4 * n_local),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=wdtype,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[None, 3], mesh_shape=shape),
    )
    mb, kb, nb, sh, sw = (int(v) for v in os.environ.get("G4X_AGMM_BLK", f"4,8,{block_n // 32},2,2").split(","))
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw, compute_with_storage_grid_size=grid
    )
    cc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH, 1.0) if act else None
    if mode == "mm_only":
        x = ttnn.from_torch(
            torch.randn(1, 1, 8 * rows, K),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[2, None], mesh_shape=shape),
        )

        def run(i):
            return ttnn.experimental.minimal_matmul(x, w, fused_activation=activation, compute_kernel_config=cc, config=cfg)

    else:
        x = ttnn.from_torch(
            torch.randn(1, 1, 8 * rows, K),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[2, 3], mesh_shape=shape),
        )
        g = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        sems = [create_global_semaphores(mesh_device, 32, cores, 0, num_extra=2 * 2 * 3) for _ in range(iters)]
        bufs = [
            ttnn.from_torch(
                torch.zeros(1, 1, rows, K),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in range(iters)
        ]

        def run(i):
            return ttnn.experimental.strided_all_gather_minimal_matmul_async(
                x,
                w,
                persistent_output_buffer=bufs[i],
                dim=3,
                multi_device_global_semaphore=sems[i],
                strided_all_gather_core_grid_offset=(0, 8),
                num_links=2,
                memory_config_ag=DRAM,
                topology=ttnn.Topology.Ring,
                cluster_axis=1,
                fused_activation=activation,
                config=cfg,
                memory_config_mm=DRAM,
                compute_kernel_config=cc,
                num_workers_per_link=3,
                num_buffers_per_channel=8,
                read_local_slice_from_input=True,
                mm_signal_aggregator_mode=ttnn.MMSignalAggregatorMode.On,
            )[1]

    run(0)  # compile
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [run(i) for i in range(iters)]
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    signpost("start")
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    ttnn.release_trace(mesh_device, tid)
