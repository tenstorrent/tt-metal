# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_reduce_scatter_variance_test(
    mesh_device,
    variance_values,
    dim=3,
    cluster_axis=0,
    num_iters=1,
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    topology=ttnn.Topology.Linear,
    num_links=1,
):
    """
    Test reduce_scatter operation with specific variance values.

    Args:
        mesh_device: The mesh device (e.g., 2x4)
        variance_values: List of 128 variance values from RMS norm
        dim: Dimension to scatter along (default: 3)
        cluster_axis: Axis to reduce across (default: 0)
        num_iters: Number of iterations to run
        dtype: Data type for tensors
        layout: Memory layout
        topology: Fabric topology
        num_links: Number of fabric links to use

    Returns:
        bool: True if test passed
    """

    # Get mesh dimensions
    num_devices_rows = mesh_device.shape[0]
    num_devices_cols = mesh_device.shape[1]

    logger.info(f"=" * 80)
    logger.info(f"Running reduce_scatter variance test")
    logger.info(f"Mesh shape: {num_devices_rows}x{num_devices_cols}")
    logger.info(f"Cluster axis: {cluster_axis}")
    logger.info(f"Scatter dim: {dim}")
    logger.info(f"Num iterations: {num_iters}")
    logger.info(f"=" * 80)

    # Create input tensor [1, 1, 1, 128]
    torch_input = torch.tensor(variance_values, dtype=torch.float32).reshape(1, 1, 1, 128)

    # Memory config
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Prepare test data for all iterations
    tt_input_list = []
    expected_output_list = []

    for iter_idx in range(num_iters):
        logger.info(f"\nPreparing iteration {iter_idx + 1}/{num_iters}")

        # For each iteration, we can add noise or use same values
        if iter_idx == 0:
            current_input = torch_input
        else:
            # Add small noise for subsequent iterations (optional)
            noise = torch.randn_like(torch_input) * 0.01
            current_input = torch_input + noise

        # Replicate input across all mesh devices
        # Shape: [num_devices_rows, num_devices_cols, 1, 128]
        torch_input_mesh = current_input.repeat(num_devices_rows, num_devices_cols, 1, 1)

        # Convert to ttnn tensor
        tt_input = ttnn.from_torch(
            torch_input_mesh,
            device=mesh_device,
            layout=layout,
            dtype=dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, dims=(0, 1), mesh_shape=(num_devices_rows, num_devices_cols)
            ),
        )
        tt_input_list.append(tt_input)

        # Calculate expected output
        # Step 1: Reduce (sum) across cluster_axis
        if cluster_axis == 0:
            # Sum across rows (num_devices_rows copies)
            reduced = current_input * num_devices_rows
        else:
            # Sum across columns (num_devices_cols copies)
            reduced = current_input * num_devices_cols

        # Step 2: Scatter along dim
        # After reduce_scatter, output shape changes:
        # output_shape[dim] = input_shape[dim] / num_devices_along_cluster_axis
        if cluster_axis == 0:
            num_devices_scatter = num_devices_rows
        else:
            num_devices_scatter = num_devices_cols

        # Split reduced tensor into chunks for each device
        scatter_chunks = torch.chunk(reduced, num_devices_scatter, dim=dim)

        # Reconstruct expected mesh output
        # Each device along cluster_axis gets one chunk
        expected_mesh = []
        for row_idx in range(num_devices_rows):
            row_chunks = []
            for col_idx in range(num_devices_cols):
                if cluster_axis == 0:
                    # Scatter along rows
                    chunk_idx = row_idx
                else:
                    # Scatter along columns
                    chunk_idx = col_idx
                row_chunks.append(scatter_chunks[chunk_idx])
            expected_mesh.append(torch.cat(row_chunks, dim=1))
        expected_output = torch.cat(expected_mesh, dim=0)
        expected_output_list.append(expected_output)

    # Run reduce_scatter operations
    logger.info(f"\n{'='*80}")
    logger.info("Executing reduce_scatter operations...")
    logger.info(f"{'='*80}")

    tt_output_list = []
    for iter_idx in range(num_iters):
        logger.info(f"Iteration {iter_idx + 1}/{num_iters}")

        tt_output = ttnn.reduce_scatter(
            tt_input_list[iter_idx],
            dim=dim,
            cluster_axis=cluster_axis,
            memory_config=mem_config,
            num_links=num_links,
            topology=topology,
        )

        ttnn.synchronize_device(mesh_device)
        tt_output_list.append(tt_output)

    logger.info("All operations completed!")

    # Verify results
    logger.info(f"\n{'='*80}")
    logger.info("Verifying results...")
    logger.info(f"{'='*80}")

    passed = True
    for iter_idx in range(num_iters):
        logger.info(f"\nChecking iteration {iter_idx + 1}/{num_iters}")

        # Convert output to torch
        tt_output_torch = ttnn.to_torch(
            tt_output_list[iter_idx],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, mesh_shape=(num_devices_rows, num_devices_cols), dims=(0, 1)
            ),
        )

        expected = expected_output_list[iter_idx]

        logger.info(f"  Output shape: {tt_output_torch.shape}")
        logger.info(f"  Expected shape: {expected.shape}")
        logger.info(f"  Output range: [{tt_output_torch.min():.4f}, {tt_output_torch.max():.4f}]")
        logger.info(f"  Expected range: [{expected.min():.4f}, {expected.max():.4f}]")

        # Compare with PCC
        eq, pcc_msg = comp_pcc(tt_output_torch, expected, pcc=0.99)
        logger.info(f"  PCC result: {pcc_msg}")

        if not eq:
            logger.error(f"  ✗ Iteration {iter_idx + 1} FAILED!")
            passed = False

            # Print sample values for debugging
            logger.error(f"  First 5 expected values: {expected.flatten()[:5].tolist()}")
            logger.error(f"  First 5 actual values: {tt_output_torch.flatten()[:5].tolist()}")
            break
        else:
            logger.info(f"  ✓ Iteration {iter_idx + 1} passed!")

    logger.info(f"\n{'='*80}")
    if passed:
        logger.info("✓ ALL TESTS PASSED!")
    else:
        logger.error("✗ TEST FAILED!")
    logger.info(f"{'='*80}\n")

    return passed


# ============================================================================
# Test cases using the custom run_reduce_scatter_variance_test
# ============================================================================


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (2, 4),  # 2x4 mesh matching your MLIR
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iters", [1, 5])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_reduce_scatter_rms_norm_2x4(mesh_device, num_iters, topology):
    """
    Test reduce_scatter with RMS norm variance values on 2x4 mesh.
    Reduces across cluster_axis=0 (rows), scatters along dim=3.
    """

    # Variance values from RMS norm
    variance_values = [
        34738.7695,
        249.1941,
        260.4489,
        177.5385,
        199.2874,
        275.0461,
        299.0211,
        188.0558,
        182.3192,
        300.6374,
        200.0179,
        290.1855,
        224.7958,
        204.7478,
        149.1924,
        296.7355,
        123.5096,
        250.9123,
        355.0552,
        127.8251,
        112.3397,
        107.7042,
        220.0517,
        141.4263,
        223.3916,
        138.8412,
        174.3175,
        264.4438,
        163.6414,
        123.2869,
        118.0919,
        223.2983,
        140.9461,
        210.5886,
        127.4317,
        179.6107,
        129.5005,
        184.3295,
        304.0739,
        134.4110,
        175.1834,
        117.0457,
        122.8643,
        214.7888,
        197.2131,
        163.7758,
        225.3279,
        166.7933,
        237.3999,
        111.1024,
        198.6654,
        109.5154,
        212.0784,
        165.9720,
        107.2304,
        175.8852,
        151.7798,
        172.3650,
        237.9021,
        92.4863,
        177.0984,
        199.4871,
        243.9700,
        194.4257,
        182.3212,
        186.8201,
        172.7106,
        152.0067,
        170.8373,
        174.7033,
        386.6372,
        2472.2349,
        2578.8806,
        2621.1753,
        2709.7971,
        2762.3381,
        2781.8025,
        2809.7917,
        2815.4802,
        2805.5400,
        2819.7412,
        2797.5308,
        2731.0989,
        2751.9348,
        2761.3269,
        2766.0684,
        2747.1982,
        2724.2375,
        2714.1360,
        2716.3562,
        2725.0154,
        2747.7791,
        2793.5955,
        2849.8733,
        2879.8489,
        2883.3757,
        2902.0820,
        2860.8955,
        2850.4705,
        2854.8511,
        2859.5005,
        2843.9390,
        2839.6272,
        2885.4998,
        2915.4143,
        2912.8018,
        2909.9365,
        2891.3801,
        2891.9956,
        2865.3618,
        2927.4949,
        2985.7021,
        2971.0090,
        2984.9619,
        2980.3198,
        2984.3562,
        3012.5742,
        3023.6077,
        3056.9958,
        3105.6138,
        3088.5920,
        3101.9368,
        3119.3777,
        3076.7812,
        3048.8435,
        3059.4365,
        3093.8174,
        3135.1506,
    ]

    passed = run_reduce_scatter_variance_test(
        mesh_device=mesh_device,
        variance_values=variance_values,
        dim=3,
        cluster_axis=0,
        num_iters=num_iters,
        dtype=ttnn.float32,
        topology=topology,
        num_links=1,
    )

    assert passed, "Test failed!"


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 8),  # 1x8 mesh - single row
    ],
    indirect=True,
)
def test_reduce_scatter_rms_norm_1x8(mesh_device):
    """
    Test reduce_scatter with RMS norm variance values on 1x8 mesh.
    Reduces across cluster_axis=1 (columns), scatters along dim=3.
    """

    variance_values = [
        34738.7695,
        249.1941,
        260.4489,
        177.5385,
        199.2874,
        275.0461,
        299.0211,
        188.0558,
        182.3192,
        300.6374,
        200.0179,
        290.1855,
        224.7958,
        204.7478,
        149.1924,
        296.7355,
        123.5096,
        250.9123,
        355.0552,
        127.8251,
        112.3397,
        107.7042,
        220.0517,
        141.4263,
        223.3916,
        138.8412,
        174.3175,
        264.4438,
        163.6414,
        123.2869,
        118.0919,
        223.2983,
        140.9461,
        210.5886,
        127.4317,
        179.6107,
        129.5005,
        184.3295,
        304.0739,
        134.4110,
        175.1834,
        117.0457,
        122.8643,
        214.7888,
        197.2131,
        163.7758,
        225.3279,
        166.7933,
        237.3999,
        111.1024,
        198.6654,
        109.5154,
        212.0784,
        165.9720,
        107.2304,
        175.8852,
        151.7798,
        172.3650,
        237.9021,
        92.4863,
        177.0984,
        199.4871,
        243.9700,
        194.4257,
        182.3212,
        186.8201,
        172.7106,
        152.0067,
        170.8373,
        174.7033,
        386.6372,
        2472.2349,
        2578.8806,
        2621.1753,
        2709.7971,
        2762.3381,
        2781.8025,
        2809.7917,
        2815.4802,
        2805.5400,
        2819.7412,
        2797.5308,
        2731.0989,
        2751.9348,
        2761.3269,
        2766.0684,
        2747.1982,
        2724.2375,
        2714.1360,
        2716.3562,
        2725.0154,
        2747.7791,
        2793.5955,
        2849.8733,
        2879.8489,
        2883.3757,
        2902.0820,
        2860.8955,
        2850.4705,
        2854.8511,
        2859.5005,
        2843.9390,
        2839.6272,
        2885.4998,
        2915.4143,
        2912.8018,
        2909.9365,
        2891.3801,
        2891.9956,
        2865.3618,
        2927.4949,
        2985.7021,
        2971.0090,
        2984.9619,
        2980.3198,
        2984.3562,
        3012.5742,
        3023.6077,
        3056.9958,
        3105.6138,
        3088.5920,
        3101.9368,
        3119.3777,
        3076.7812,
        3048.8435,
        3059.4365,
        3093.8174,
        3135.1506,
    ]

    passed = run_reduce_scatter_variance_test(
        mesh_device=mesh_device,
        variance_values=variance_values,
        dim=3,
        cluster_axis=1,  # For 1x8 mesh, reduce across columns
        num_iters=1,
        dtype=ttnn.float32,
        topology=ttnn.Topology.Linear,
        num_links=1,
    )

    assert passed, "Test failed!"
