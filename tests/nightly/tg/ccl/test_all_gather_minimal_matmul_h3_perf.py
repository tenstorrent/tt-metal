# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Op timing of all_gather_minimal_matmul_async at the MiniMax-H3 per-chip shapes on a 4x8 Galaxy.

The three H3 AGMM layers at the 5 s (4736 rows/chip) and 15 s (13664 rows/chip) points, with the model's
own config resolution (the legacy `force_transpose=False` rule) and, when AGMM_PERF_PIN=1, the pinned
orientation with one frozen blocking per layer. 20 warm iterations, host sync before and after each,
min and median printed as `AGMM_TIMING ...` lines (run with -s). Not a correctness test.
"""

import math
import os
import statistics
import time

import pytest
import torch
import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.tt_dit.models.transformers.minimax_h3.agmm_config import agmm_block_size
from models.tt_dit.utils.matmul import agmm_worker_grid, get_agmm_config

from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    create_global_semaphores,
)

TP, SP, NUM_LINKS, ITERS = 4, 8, 2, 20
ROWS = [4736, 13664]
# name -> (K, per-device N, fused addcmul, fused swiglu)
LAYERS = {"to_qkv": (5376, 5376, False, False), "to_out": (7168, 1344, True, False), "ff1": (5376, 7168, False, True)}
# Frozen blocking + pinned orientation per layer (K_block divides K_tiles_per_device).
PINNED = {"to_qkv": ((4, 6, 16), False), "to_out": ((6, 8, 6), True), "ff1": ((6, 3, 16), False)}
DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": create_fabric_router_config(4096),
    "trace_region_size": 90112,
}


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True, ids=["fabric_ring"])
@pytest.mark.parametrize("layer", list(LAYERS))
@pytest.mark.parametrize("rows", ROWS, ids=[f"rows{r}" for r in ROWS])
@pytest.mark.parametrize("pin", [False, True], ids=["legacy", "pinned"])
def test_h3_agmm_timing(mesh_device, layer, rows, pin):
    if is_wormhole_b0():
        pytest.skip("blackhole-only core grids")
    if pin and not os.environ.get("AGMM_PERF_PIN"):
        pytest.skip("pinned orientation needs the transpose_core_grid build; set AGMM_PERF_PIN=1")
    K, N, addcmul, swiglu = LAYERS[layer]
    weight_N = 2 * N if swiglu else N
    full_grid = mesh_device.compute_with_storage_grid_size()
    extra = {}
    if pin:
        (mb, kb, nb), transposed = PINNED[layer]
        core_grid = agmm_worker_grid(full_grid, transposed, NUM_LINKS)
        workers = math.ceil((core_grid.x if transposed else core_grid.y) / NUM_LINKS)
        config = ttnn.MinimalMatmulConfig(
            M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=2, subblock_w=2,
            compute_with_storage_grid_size=core_grid,
        )
        extra["transpose_core_grid"] = transposed
    else:
        core_grid, config, workers = get_agmm_config(
            rows, K, weight_N, full_grid=full_grid, cluster_size=TP, num_links=NUM_LINKS,
            default_block_size=agmm_block_size(K, N, rows), force_transpose=False,
        )
        transposed = rows > weight_N
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    # Same sharding as the model: M over the SP axis (1), K over the TP ring (axis 0).
    mesh_shape = tuple(mesh_device.shape)
    x = ttnn.from_torch(
        torch.randn(rows * SP, K, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=[1, 0]),
    )
    w = ttnn.from_torch(torch.randn(K, weight_N) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    bias = ttnn.from_torch(torch.randn(1, weight_N) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    a = b = None
    if addcmul:
        a = ttnn.from_torch(
            torch.randn(rows * SP, N, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=[None, 0]),
        )
        b = ttnn.from_torch(torch.randn(1, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))})
    sems = [create_global_semaphores(mesh_device, mesh_device.get_num_devices(), cores, 0) for _ in range(2)]
    barriers = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]

    def call(i):
        # The 4x8 model path: no persistent gather buffer, host sync before the op, barrier semaphore.
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = ttnn.experimental.all_gather_minimal_matmul_async(
            x, w, bias_tensor=bias, config=config, compute_kernel_config=compute_config,
            multi_device_global_semaphore=sems[i % 2], num_links=NUM_LINKS, topology=ttnn.Topology.Ring,
            cluster_axis=0, barrier_semaphore=barriers[i % 2], force_transpose=False,
            num_workers_per_link=workers, num_buffers_per_channel=24, scalar=1.0 if addcmul else None,
            addcmul_input_tensor1=a, addcmul_input_tensor2=b, fuse_swiglu=swiglu, **extra,
        )
        ttnn.synchronize_device(mesh_device)
        dt = time.perf_counter() - t0
        for o in out:
            ttnn.deallocate(o)
        return dt

    call(0)
    call(1)
    times = [call(i) for i in range(ITERS)]
    print(
        f"AGMM_TIMING {layer} rows={rows} mode={'pinned' if pin else 'legacy'} grid={core_grid.x}x{core_grid.y} "
        f"blocks={config.M_block_size},{config.K_block_size},{config.N_block_size} workers={workers} "
        f"transposed={int(transposed)} min_ms={min(times) * 1e3:.3f} median_ms={statistics.median(times) * 1e3:.3f}",
        flush=True,
    )
