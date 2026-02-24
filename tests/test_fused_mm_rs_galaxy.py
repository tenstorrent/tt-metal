# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone test for fused MatMul + Strided ReduceScatter on Galaxy.
# Run: pytest tests/test_fused_mm_rs_galaxy.py -svv --timeout=300
#
# Edit the TEST_CONFIGS list below to try different grid/block combinations.

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.common.utility_functions import skip_for_blackhole


TILE = 32


def create_global_semaphores(mesh_device, cores, initial_value):
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(3)]


def run_fused_mm_rs(
    mesh_device,
    M,
    K,
    N,
    dim,
    mm_core_grid,
    mm_block_m,
    mm_block_k,
    mm_block_n,
    subblock_h,
    subblock_w,
    chunk_width_in_mm_blocks,
    num_links=1,
    num_workers_per_link=None,
    cluster_axis=1,
    topology=ttnn.Topology.Ring,
    rs_mode="fused",
    allowed_pcc=0.99,
):
    num_devices = mesh_device.shape[cluster_axis]
    logger.info(
        f"=== Config: grid=({mm_core_grid.x},{mm_core_grid.y}), "
        f"blocks=({mm_block_m},{mm_block_k},{mm_block_n}), "
        f"sub=({subblock_h},{subblock_w}), cwimb={chunk_width_in_mm_blocks}, "
        f"rs_mode={rs_mode} ==="
    )
    logger.info(f"M={M}, K={K}, N={N}, devices={num_devices}, topology={topology}")

    N_tiles = N // TILE
    mm_N_block_wt = N_tiles // mm_core_grid.x
    slice_Wt = N_tiles // (num_devices // 2)
    logger.info(f"N_tiles={N_tiles}, mm_N_block_wt={mm_N_block_wt}, slice_Wt={slice_Wt}")
    logger.info(f"slice_Wt % mm_N_block_wt = {slice_Wt % mm_N_block_wt}")
    if slice_Wt % mm_N_block_wt != 0:
        logger.error(f"WILL FAIL: slice_Wt ({slice_Wt}) not divisible by mm_N_block_wt ({mm_N_block_wt})")

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    logger.info(f"Device compute grid: {compute_grid_size.x} x {compute_grid_size.y}")
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    ccl_sems = create_global_semaphores(mesh_device, all_cores, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    rs_core_grid_offset = ttnn.CoreCoord(0, mm_core_grid.y)
    logger.info(
        f"RS core grid offset: (0, {mm_core_grid.y}), " f"RS rows available: {compute_grid_size.y - mm_core_grid.y}"
    )

    mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    torch.manual_seed(0)
    input_shape = [1, 1, M, K]
    weight_shape = [num_devices, 1, K, N]

    torch_input = torch.randn(input_shape, dtype=torch.float32)
    torch_weight = torch.randn(weight_shape, dtype=torch.float32)

    weight_chunks = torch.chunk(torch_weight, num_devices, dim=0)
    mm_outputs = [torch.matmul(torch_input, weight_chunks[d]) for d in range(num_devices)]
    rs_reduced = torch.sum(torch.stack(mm_outputs), dim=0)
    rs_scattered = torch.chunk(rs_reduced, num_devices, dim=dim)

    input_tt = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_cfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    weight_tt = ttnn.from_torch(
        torch_weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_cfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[None, 0], mesh_shape=tuple(mesh_device.shape)),
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=mm_block_m // TILE,
        K_block_size=mm_block_k // TILE,
        N_block_size=mm_block_n // TILE,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=mm_core_grid,
    )

    logger.info("Running op...")
    ttnn.synchronize_device(mesh_device)

    if rs_mode == "fused":
        mm_out, rs_inter, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            input_tt,
            weight_tt,
            dim,
            ccl_sems,
            rs_core_grid_offset,
            num_links=num_links,
            memory_config_mm=mem_cfg,
            rs_output_mem_config=mem_cfg,
            topology=topology,
            cluster_axis=cluster_axis,
            config=matmul_config,
            compute_kernel_config=compute_config,
            barrier_semaphore=barrier_sem,
            num_workers_per_link=num_workers_per_link,
            chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
        )
    elif rs_mode == "separate":
        mm_out = ttnn.experimental.minimal_matmul(
            input_tt,
            weight_tt,
            compute_kernel_config=compute_config,
            config=matmul_config,
        )
        rs_out = ttnn.experimental.reduce_scatter_minimal_async(
            mm_out,
            None,
            dim,
            ccl_sems,
            barrier_semaphore=barrier_sem,
            num_links=num_links,
            memory_config=mem_cfg,
            topology=topology,
            cluster_axis=cluster_axis,
            num_workers_per_link=num_workers_per_link,
        )
        rs_inter = None

    ttnn.synchronize_device(mesh_device)
    logger.info("Op completed, checking results...")

    if rs_mode == "fused":
        tt_mm_torch = ttnn.to_torch(
            ttnn.from_device(mm_out),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        for d in range(num_devices):
            eq, output = comp_pcc(tt_mm_torch[d : d + 1], mm_outputs[d], allowed_pcc)
            logger.info(f"MM device {d}: {output}")
            assert eq, f"MM device {d} FAILED: {output}"

    tt_rs_torch = ttnn.to_torch(
        ttnn.from_device(rs_out),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim),
    )
    rs_chunks = torch.chunk(tt_rs_torch, num_devices, dim=dim)
    for d in range(num_devices):
        eq, output = comp_pcc(rs_chunks[d], rs_scattered[d], allowed_pcc)
        logger.info(f"RS device {d}: {output}")
        assert eq, f"RS device {d} FAILED: {output}"

    logger.info("ALL CHECKS PASSED!")


# ============================================================================
# TEST CONFIGS - Edit these to experiment!
# ============================================================================
# Galaxy W1/FF1 dimensions: M=8192(8k), K=2048, N=3584
# Galaxy compute grid: 7 columns x 10 rows (harvested)
# Ring 8 devices: slice_Wt = N_tiles/4 = 28
# Constraint: slice_Wt(28) % (N_tiles/grid_x) == 0 → grid_x must be multiple of 4
#
# grid_x=4: mm_N_block_wt=28, 28%28=0 ✓
# grid_x=7: mm_N_block_wt=16, 28%16≠0 ✗
# grid_x=8: mm_N_block_wt=14, 28%14=0 ✓ (but >7 cols, impossible on harvested)

TEST_CONFIGS = [
    # --- 8-column devices (unharvested, 8x9 grid): grid_x=8, grid_y=8 ---
    # ring_size=8 → slice_Wt=112/8=14, mm_N_block_wt=112/8=14, 14%14=0 ✓
    # M_tiles=256, slice_Ht=256, 256%8=0 ✓ (grid_y=7 fails: 256%7≠0)
    # 64 MM cores + RS on row 8 (1 row available on 8x9 device)
    # N_per_core=112/8=14 tiles. N_block must divide 14 → use 7 tiles (224 elements)
    # subblock_w must divide 7 → subblock_w=7, subblock_h=1 (h*w<=8)
    pytest.param(
        dict(
            M=8192,
            K=2048,
            N=3584,
            dim=3,
            mm_core_grid=ttnn.CoreCoord(8, 8),
            mm_block_m=256,
            mm_block_k=256,
            mm_block_n=224,
            subblock_h=1,
            subblock_w=7,
            chunk_width_in_mm_blocks=2,
        ),
        id="ff1_8k_8x8_fused",
    ),
    pytest.param(
        dict(
            M=8192,
            K=2048,
            N=3584,
            dim=3,
            mm_core_grid=ttnn.CoreCoord(8, 8),
            mm_block_m=256,
            mm_block_k=256,
            mm_block_n=224,
            subblock_h=1,
            subblock_w=7,
            chunk_width_in_mm_blocks=2,
            rs_mode="separate",
        ),
        id="ff1_8k_8x8_separate",
    ),
    # --- 4-column devices (harvested Galaxy model): grid_x=4 ---
    # ring_size=4 → slice_Wt=112/4=28, mm_N_block_wt=112/4=28, 28%28=0 ✓
    # NOTE: only works when ring_size=4 (model topology). Fails if ring_size=8.
    pytest.param(
        dict(
            M=8192,
            K=2048,
            N=3584,
            dim=3,
            mm_core_grid=ttnn.CoreCoord(4, 8),
            mm_block_m=256,
            mm_block_k=256,
            mm_block_n=256,
            subblock_h=4,
            subblock_w=2,
            chunk_width_in_mm_blocks=2,
        ),
        id="ff1_8k_4x8_fused",
    ),
    pytest.param(
        dict(
            M=8192,
            K=2048,
            N=3584,
            dim=3,
            mm_core_grid=ttnn.CoreCoord(4, 8),
            mm_block_m=256,
            mm_block_k=256,
            mm_block_n=256,
            subblock_h=4,
            subblock_w=2,
            chunk_width_in_mm_blocks=2,
            rs_mode="separate",
        ),
        id="ff1_8k_4x8_separate",
    ),
    # --- Smaller shapes for quick testing ---
    # M=4096 → M_tiles=128, 128%8=0 ✓. N_block=224 (7 tiles), 14/7=2 ✓
    pytest.param(
        dict(
            M=4096,
            K=2048,
            N=3584,
            dim=3,
            mm_core_grid=ttnn.CoreCoord(8, 8),
            mm_block_m=256,
            mm_block_k=256,
            mm_block_n=224,
            subblock_h=1,
            subblock_w=7,
            chunk_width_in_mm_blocks=2,
        ),
        id="ff1_4k_8x8_fused",
    ),
]


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}],
    indirect=True,
)
@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_fused_mm_rs_galaxy(mesh_device, config):
    defaults = dict(
        num_links=1,
        num_workers_per_link=None,
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        rs_mode="fused",
        allowed_pcc=0.99,
    )
    defaults.update(config)
    grid = defaults.pop("mm_core_grid")
    run_fused_mm_rs(mesh_device, mm_core_grid=grid, **defaults)
