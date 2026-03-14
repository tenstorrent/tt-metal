# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test to compare fused AllGather+MatMul vs separate operations.

Run individual tests with profile_this.py:
    python tools/tracy/profile_this.py -c "pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py::test_ag_mm[wan2_4k4k4k-8x8_4links-fused] -v"
    python tools/tracy/profile_this.py -c "pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py::test_ag_mm[wan2_4k4k4k-8x8_4links-separate] -v"

    python tools/tracy/profile_this.py -c "pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py::test_ag_mm[llama_8k_ff2-7x8_1link-fused] -v"
    python tools/tracy/profile_this.py -c "pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py::test_ag_mm[llama_8k_ff2-7x8_1link-separate] -v"
"""

import pytest
import torch
from loguru import logger

import ttnn


def comp_pcc(golden, calculated, pcc_threshold=0.99):
    """Compare PCC between golden and calculated tensors."""
    if golden.dtype != calculated.dtype:
        calculated = calculated.to(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are all NaN")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all NaN")
        return False, 0.0

    # Flatten tensors
    golden_flat = golden.flatten().float()
    calculated_flat = calculated.flatten().float()

    # Remove NaN values
    mask = ~(torch.isnan(golden_flat) | torch.isnan(calculated_flat))
    golden_flat = golden_flat[mask]
    calculated_flat = calculated_flat[mask]

    if len(golden_flat) == 0:
        logger.error("No valid values to compare")
        return False, 0.0

    # Compute PCC
    pcc = torch.corrcoef(torch.stack([golden_flat, calculated_flat]))[0, 1].item()

    if torch.isnan(torch.tensor(pcc)):
        pcc = 0.0

    logger.info(f"PCC: {pcc:.6f}")
    return pcc >= pcc_threshold, pcc


def create_global_semaphores(mesh_device, cores, initial_value):
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]


def run_ag_mm_test(
    mesh_device,
    M,
    K,
    N,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    core_grid,
    num_links,
    use_fused,
    num_iters=5,
    validate=True,
    force_transpose=True,
):
    """Run AllGather + MatMul test (either fused or separate)."""

    logger.info(
        f"{'FUSED' if use_fused else 'SEPARATE'}: M={M}, K={K}, N={N}, grid=({core_grid.x},{core_grid.y}), links={num_links}, force_transpose={force_transpose}"
    )

    # Use the matmul core_grid for semaphores (not full compute grid)
    ccl_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )

    # Create semaphores for each iteration
    semaphore_list = [create_global_semaphores(mesh_device, ccl_cores, 0) for _ in range(num_iters)]

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=core_grid,
    )

    # Adjust num_links based on grid divisibility (fused op: grid_x % num_links when force_transpose, else grid_y % num_links)
    effective_num_links = num_links
    div_axis = core_grid.x if force_transpose else core_grid.y
    if div_axis % num_links != 0:
        for nl in [4, 3, 2, 1]:
            if div_axis % nl == 0:
                effective_num_links = nl
                break
        logger.warning(
            f"Adjusted num_links: {num_links} -> {effective_num_links} (div_axis={'x' if force_transpose else 'y'}={div_axis})"
        )

    # Per-device K (ring_size=4 for Galaxy cluster_axis=1)
    ring_size = mesh_device.shape[1]  # cluster_axis=1
    K_per_device = K // ring_size

    # Fused op expects full K (a_logical[-1]*ring_size); use full (M,K) input sharded, full (K,N) weight, golden = full@full.
    # Use BFP8 for better performance (matches Llama model dtype)
    torch.manual_seed(42)
    torch_input_f32 = torch.randn((1, 1, M, K), dtype=torch.float32)
    torch_weight_f32 = torch.randn((K, N), dtype=torch.float32)
    torch_input = torch_input_f32.to(torch.bfloat16)
    torch_weight = torch_weight_f32.to(torch.bfloat16)
    torch_golden = torch_input_f32.squeeze() @ torch_weight_f32  # (M, K) @ (K, N) = (M, N)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[None, 3], mesh_shape=tuple(mesh_device.shape)),
    )

    tt_weight = ttnn.from_torch(
        torch_weight,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Create persistent output buffers for all_gather (separate path)
    # all_gather_async output shape: dim 3 is full K (input K_per_device * ring_size). Each device holds full (1,1,M,K).
    if not use_fused:
        ag_output_shape = (1, 1, M, K)
        persistent_ag_buffers = [
            ttnn.from_torch(
                torch.zeros(ag_output_shape, dtype=torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in range(num_iters)
        ]

    # Calculate num_workers_per_link: total workers must fit in in0 axis (grid.x when force_transpose, else grid.y).
    # Otherwise the last link gets fewer cores and barrier/sync can hang (e.g. 6x8 grid + 2 links
    # with 4 workers/link => 8 workers but only 6 cores => link1 has 2 cores, sync waits forever).
    max_workers_total = core_grid.x if force_transpose else core_grid.y
    num_workers_per_link = max(1, min(8 // effective_num_links, max_workers_total // effective_num_links))

    # Run iterations and validate first iteration
    for i in range(num_iters):
        if use_fused:
            output = ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                config=matmul_config,
                compute_kernel_config=compute_config,
                multi_device_global_semaphore=semaphore_list[i],
                num_links=effective_num_links,
                topology=ttnn.Topology.Ring,
                cluster_axis=1,
                force_transpose=force_transpose,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=8,
            )
            ttnn.synchronize_device(mesh_device)

            # Validate first iteration
            if validate and i == 0:
                tt_output = output[0]
                output_torch = ttnn.to_torch(
                    tt_output,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=[0, 3], mesh_shape=tuple(mesh_device.shape)
                    ),
                )
                output_single = output_torch[0, :, :, :N].squeeze()
                passed, pcc = comp_pcc(torch_golden.to(torch.bfloat16), output_single, pcc_threshold=0.95)
                logger.info(f"Iteration {i}: PCC = {pcc:.6f}, Passed = {passed}")

            ttnn.deallocate(output[0])
        else:
            ag_output = ttnn.experimental.all_gather_async(
                tt_input,
                persistent_output_buffer=persistent_ag_buffers[i],
                dim=3,
                multi_device_global_semaphore=semaphore_list[i],
                num_links=effective_num_links,
                topology=ttnn.Topology.Ring,
                cluster_axis=1,
                num_workers_per_link=3,
                num_buffers_per_channel=2,
            )
            mm_output = ttnn.experimental.minimal_matmul(
                ag_output,
                tt_weight,
                config=matmul_config,
                compute_kernel_config=compute_config,
            )
            ttnn.synchronize_device(mesh_device)

            # Validate first iteration
            if validate and i == 0:
                output_torch = ttnn.to_torch(
                    mm_output,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=[0, 3], mesh_shape=tuple(mesh_device.shape)
                    ),
                )
                output_single = output_torch[0, :, :, :N].squeeze()
                passed, pcc = comp_pcc(torch_golden.to(torch.bfloat16), output_single, pcc_threshold=0.95)
                logger.info(f"Iteration {i}: PCC = {pcc:.6f}, Passed = {passed}")

            ttnn.deallocate(mm_output)

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_weight)
    if not use_fused:
        for buf in persistent_ag_buffers:
            ttnn.deallocate(buf)

    logger.info("Done")


# =============================================================================
# Pytest Parametrized Tests
# =============================================================================


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((8, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="galaxy"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K, N, M_block, K_block, N_block, subblock_h, subblock_w",
    [
        (4096, 4096, 4096, 8, 8, 8, 2, 2),  # WAN 2.2 default
        (8192, 3584, 2048, 8, 8, 8, 2, 2),  # Llama 8k ISL FF2 (subblock 2x2: max_dest_volume constraint)
        (131072, 3584, 2048, 8, 8, 8, 2, 2),  # Llama 128k ISL FF2 (subblock 2x2: max_dest_volume constraint)
        (8192, 4096, 2048, 8, 8, 8, 2, 2),  # Llama 8k with K padded to 4096 (test PCC / non-Po2 K hypothesis)
        (131072, 4096, 2048, 8, 8, 8, 2, 2),  # Llama 128k with K padded to 4096
    ],
    ids=["wan2_4k4k4k", "llama_8k_ff2", "llama_128k_ff2", "llama_8k_ff2_K4096", "llama_128k_ff2_K4096"],
)
@pytest.mark.parametrize(
    "grid_x, grid_y, num_links",
    [
        (8, 8, 4),  # Should show improvement
        (8, 8, 2),  # Should show improvement
        (6, 8, 3),  # Should show improvement
        (6, 8, 2),  # Should show improvement
        (4, 8, 2),  # 4×8 4 links removed (CoreRangeSet overlap)
        (7, 8, 1),  # Llama grid (7×7 removed; 7×8 works better)
        (8, 7, 4),  # Option 2: Transposed grid - 8 cols divisible by 4
        # 7x9 excluded: triggers TT_FATAL "Illegal NOC usage" on core (6,8) - both DM kernels use same NOC
    ],
    ids=["8x8_4links", "8x8_2links", "6x8_3links", "6x8_2links", "4x8_2links", "7x8_1link", "8x7_4links"],
)
@pytest.mark.parametrize(
    "use_fused",
    [False, True],
    ids=["separate", "fused"],
)
def test_ag_mm(
    mesh_device,
    device_params,
    M,
    K,
    N,
    M_block,
    K_block,
    N_block,
    subblock_h,
    subblock_w,
    grid_x,
    grid_y,
    num_links,
    use_fused,
):
    run_ag_mm_test(
        mesh_device=mesh_device,
        M=M,
        K=K,
        N=N,
        M_block_size=M_block,
        K_block_size=K_block,
        N_block_size=N_block,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        core_grid=ttnn.CoreCoord(grid_x, grid_y),
        num_links=num_links,
        use_fused=use_fused,
        num_iters=5,
        force_transpose=True,
    )


# 7×8 grid with force_transpose=False: divisibility uses grid_y (8), so num_links=2 is valid.
# Only for 4k4k4k (M=N) so C++ transpose_core_grid = (M>N) = false and check is on grid_y.
# Known: fused path segfaults (op bug with force_transpose=False on 7×8); skipped until op is fixed.
@pytest.mark.skip(reason="Fused op segfaults with force_transpose=False on 7×8 grid; skip until op/kernel fixed")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((8, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="galaxy"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_fused",
    [False, True],
    ids=["separate", "fused"],
)
def test_ag_mm_7x8_force_transpose_false(
    mesh_device,
    device_params,
    use_fused,
):
    """7×8 grid, 2 links, force_transpose=False — one shape (4k4k4k) so grid_y % num_links and we get 2-link bandwidth."""
    run_ag_mm_test(
        mesh_device=mesh_device,
        M=4096,
        K=4096,
        N=4096,
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=2,
        core_grid=ttnn.CoreCoord(7, 8),
        num_links=2,
        use_fused=use_fused,
        num_iters=5,
        force_transpose=False,
    )
