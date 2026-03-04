# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test for batch_view operation with multi-device sharding.

This test:
1. Creates a (2,1) linear topology mesh with 2 chips
2. Shards input tensor [2, 8, 4K, 7K] across 2 chips (each gets [8, 4K, 7K])
3. Runs the same matmul [7K, 2K] 8 times using batch_view on each chip
4. Collects results back to host to get [2, 8, 4K, 2K]
5. Validates against torch reference

Usage:
    pytest models/demos/deepseek_v3_d_p/tests/pcc/test_batch_view_e2e.py -vvv
"""

import pytest
import torch
import ttnn
from loguru import logger
from tracy import signpost

from tests.ttnn.utils_for_testing import comp_pcc


# @pytest.mark.parametrize(
#     "device_params",
#     [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
#     indirect=["device_params"],
# )
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch, M,K,N", [(8, 4 * 1024, 7 * 1024, 2 * 1024)])
def test_batch_view_e2e_matmul(mesh_device, device_params, batch, M, K, N):
    """
    End-to-end test: shard [2, 8, 4K, 7K] across 2 chips, run matmul per batch using batch_view.
    """
    # Dimensions
    num_chips = mesh_device.shape[0]  # 2 chips

    signpost(f"batch_view: {mesh_device.shape=}, {batch=}, {M=}, {K=}, {N=}")
    logger.info(f"batch_view: {mesh_device.shape=}, {batch=}, {M=}, {K=}, {N=}")

    # Create input tensors on host (float32 for reference)
    torch_input = torch.randn((num_chips, batch, M, K), dtype=torch.float32)
    torch_weights = torch.randn((K, N), dtype=torch.float32)

    # Compute reference output on torch
    # For each chip c and batch b: output[c, b] = input[c, b] @ weights
    torch_output = torch.zeros((num_chips, batch, M, N), dtype=torch.float32)
    for c in range(num_chips):
        for b in range(batch):
            torch_output[c, b] = torch_input[c, b] @ torch_weights

    logger.info(f"Torch reference computed: output shape = {torch_output.shape}")

    input_mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),  # Shard first dim across mesh rows
    )

    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    ttnn_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        mesh_mapper=input_mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights.to(torch.bfloat16),
        mesh_mapper=weights_mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.visualize_tensor(ttnn_input)
    ttnn.visualize_tensor(ttnn_weights)
    logger.info(f"Input tensor on device: shape={ttnn_input.shape}")
    logger.info(f"Weights tensor on device: shape={ttnn_weights.shape}")

    # Each device has [8, 4K, 7K] after sharding
    # We'll run matmul for each batch using batch_view
    output_tensors = []

    for batch_idx in range(batch):
        # Create view of batch [4K, 7K] from the [8, 4K, 7K] tensor on each device
        batch_view = ttnn.experimental.deepseek.batch_view(ttnn_input, batch_idx)

        # Run matmul with HiFi4: [4K, 7K] @ [7K, 2K] -> [4K, 2K]
        batch_output = ttnn.matmul(
            batch_view,
            ttnn_weights,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
        )

        output_tensors.append(batch_output)
        logger.info(f"Batch {batch_idx}/{batch}: matmul done {batch_output.shape=}")

    logger.info(f"{output_tensors[0].shape=}")
    ttnn.visualize_tensor(output_tensors[0])

    output_mesh_composer = ttnn.ConcatMesh2dToTensor(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, 1),
    )

    all_pcc = []

    for batch_idx in range(batch):
        logger.info(f"Batch {batch_idx}/{batch} output tensor shape = {output_tensors[batch_idx].shape}")
        ttnn_output_torch = ttnn.to_torch(output_tensors[batch_idx], mesh_composer=output_mesh_composer).to(
            torch.float32
        )
        logger.info(f"Batch {batch_idx}/{batch}: {ttnn_output_torch.shape=} {torch_output[:, batch_idx].shape=}")

        for c in range(num_chips):
            # Handle both 2D (single chip) and 3D (multi-chip) output shapes
            if ttnn_output_torch.dim() == 2:
                ttnn_slice = ttnn_output_torch  # [M, N]
            else:
                ttnn_slice = ttnn_output_torch[c, :, :]  # [M, N] from [num_chips, M, N]

            pcc_passed, pcc_message = comp_pcc(torch_output[c, batch_idx, :, :], ttnn_slice, 0.997)

            logger.info(f"{batch_idx}/{batch} chip {c} {pcc_passed} PCC = {pcc_message}. Threshold = {0.997}")
            all_pcc.append(pcc_passed)
    assert (
        any(not pcc_good for pcc_good in all_pcc) == False
    ), f"At least one batch output did not match reference {all_pcc=}"

    logger.info("Validation PASSED: ttnn output matches torch reference")
