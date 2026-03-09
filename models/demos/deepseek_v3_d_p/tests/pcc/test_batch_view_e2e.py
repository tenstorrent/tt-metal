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
from loguru import logger
from tracy import signpost

import ttnn
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


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch, M,K,N", [(8, 4 * 1024, 7 * 1024, 2 * 1024)])
def test_batch_view_e2e_matmul_output(mesh_device, device_params, batch, M, K, N):
    """
    End-to-end test using batch_view for BOTH input and output.

    This test:
    1. Pre-allocates output buffer [num_chips, batch, M, N] sharded across chips
    2. Uses batch_view on input to read each batch
    3. Uses batch_view on output to write matmul results directly to output buffer
    4. Reads back full output buffer (no separate concat needed - results already contiguous)
    """
    num_chips = mesh_device.shape[0]

    signpost(f"batch_view_output: {mesh_device.shape=}, {batch=}, {M=}, {K=}, {N=}")
    logger.info(f"batch_view_output: {mesh_device.shape=}, {batch=}, {M=}, {K=}, {N=}")

    # Create input tensors on host (float32 for reference)
    torch_input = torch.randn((num_chips, batch, M, K), dtype=torch.float32)
    torch_weights = torch.randn((K, N), dtype=torch.float32)

    # Compute reference output on torch
    torch_output = torch.zeros((num_chips, batch, M, N), dtype=torch.float32)
    for c in range(num_chips):
        for b in range(batch):
            torch_output[c, b] = torch_input[c, b] @ torch_weights

    logger.info(f"Torch reference computed: output shape = {torch_output.shape}")

    # Mesh mappers
    input_mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),  # Shard first dim across mesh rows
    )
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # Create input tensor on device
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

    # Pre-allocate output buffer [num_chips, batch, M, N] sharded across chips
    # Each chip will have [batch, M, N]
    torch_output_init = torch.zeros((num_chips, batch, M, N), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_init,
        mesh_mapper=input_mesh_mapper,  # Same sharding as input
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"Input tensor: shape={ttnn_input.shape}")
    logger.info(f"Output buffer: shape={ttnn_output.shape}")
    ttnn.visualize_tensor(ttnn_input)
    ttnn.visualize_tensor(ttnn_output)

    # Run matmul for each batch, writing directly to output buffer views
    for batch_idx in range(batch):
        # Create view of batch [M, K] from input
        input_view = ttnn.experimental.deepseek.batch_view(ttnn_input, batch_idx)
        # Create view of batch [M, N] from output buffer
        output_view = ttnn.experimental.deepseek.batch_view(ttnn_output, batch_idx)

        logger.info(f"Batch {batch_idx}/{batch}: input_view={input_view.shape}, output_view={output_view.shape}")

        # Run matmul writing directly to output view
        ttnn.matmul(
            input_view,
            ttnn_weights,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
            optional_output_tensor=output_view,
        )
        logger.info(f"Batch {batch_idx}/{batch}: matmul done")

    # Read back full output buffer - no concat needed, results already contiguous!
    logger.info(f"Reading back full output buffer: shape={ttnn_output.shape}")
    ttnn.visualize_tensor(ttnn_output)

    output_mesh_composer = ttnn.ConcatMesh2dToTensor(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, 1),
    )

    ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=output_mesh_composer).to(torch.float32)
    logger.info(f"Output tensor readback: {ttnn_output_torch.shape=}, expected: {torch_output.shape=}")

    # Validate all batches
    all_pcc = []
    for c in range(num_chips):
        for batch_idx in range(batch):
            # Handle shape differences
            if ttnn_output_torch.dim() == 4:
                ttnn_slice = ttnn_output_torch[c, batch_idx]  # [M, N]
            elif ttnn_output_torch.dim() == 3:
                ttnn_slice = ttnn_output_torch[batch_idx]  # [M, N] for single chip
            else:
                ttnn_slice = ttnn_output_torch  # [M, N]

            pcc_passed, pcc_message = comp_pcc(torch_output[c, batch_idx], ttnn_slice, 0.997)
            logger.info(f"chip {c} batch {batch_idx}: {pcc_passed} PCC = {pcc_message}. Threshold = 0.997")
            all_pcc.append(pcc_passed)

    assert all(all_pcc), f"At least one batch output did not match reference {all_pcc=}"
    logger.info("Validation PASSED: batch_view output test matches torch reference")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch, M,K,N", [(8, 4 * 1024, 7 * 1024, 2 * 1024)])
def test_ttnn_slice_concat_baseline(mesh_device, device_params, batch, M, K, N):
    """
    Baseline test using pure ttnn.slice and ttnn.concat (no batch_view).

    This test:
    1. Uses ttnn.slice to extract each batch from input
    2. Runs matmul for each batch
    3. Uses ttnn.concat to assemble final output

    Compare performance with batch_view tests to see the overhead of slice/concat.
    """
    num_chips = mesh_device.shape[0]

    signpost(f"ttnn_slice_concat: {mesh_device.shape=}, {batch=}, {M=}, {K=}, {N=}")
    logger.info(f"ttnn_slice_concat: {mesh_device.shape=}, {batch=}, {M=}, {K=}, {N=}")

    # Create input tensors on host (float32 for reference)
    torch_input = torch.randn((num_chips, batch, M, K), dtype=torch.float32)
    torch_weights = torch.randn((K, N), dtype=torch.float32)

    # Compute reference output on torch
    torch_output = torch.zeros((num_chips, batch, M, N), dtype=torch.float32)
    for c in range(num_chips):
        for b in range(batch):
            torch_output[c, b] = torch_input[c, b] @ torch_weights

    logger.info(f"Torch reference computed: output shape = {torch_output.shape}")

    # Mesh mappers
    input_mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),
    )
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # Create input tensor on device
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

    logger.info(f"Input tensor: shape={ttnn_input.shape}")
    ttnn.visualize_tensor(ttnn_input)

    # Run matmul for each batch using ttnn.slice (creates copies, not views)
    output_tensors = []
    for batch_idx in range(batch):
        # Use ttnn.slice to extract batch - this creates a COPY
        # Input shape on device: [batch, M, K] -> slice to [1, M, K] -> reshape to [M, K]
        batch_slice = ttnn_input[:, batch_idx : batch_idx + 1, :, :]  # [1, 1, M, K]
        batch_slice = ttnn.reshape(batch_slice, [M, K])  # [M, K]

        logger.info(f"Batch {batch_idx}/{batch}: slice shape={batch_slice.shape}")

        # Run matmul
        batch_output = ttnn.matmul(
            batch_slice,
            ttnn_weights,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
        )

        # Reshape back to [1, M, N] for concat
        batch_output = ttnn.reshape(batch_output, [1, M, N])
        output_tensors.append(batch_output)
        logger.info(f"Batch {batch_idx}/{batch}: matmul done, output shape={batch_output.shape}")

    # Concat all outputs along batch dimension
    logger.info(f"Concatenating {len(output_tensors)} tensors...")
    ttnn_output = ttnn.concat(output_tensors, dim=0)  # [batch, M, N]
    logger.info(f"Concat done: shape={ttnn_output.shape}")
    ttnn.visualize_tensor(ttnn_output)

    # Read back and validate
    output_mesh_composer = ttnn.ConcatMesh2dToTensor(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, 1),
    )

    ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=output_mesh_composer).to(torch.float32)
    logger.info(f"Output tensor readback: {ttnn_output_torch.shape=}, expected: {torch_output.shape=}")

    # Validate all batches
    all_pcc = []
    for c in range(num_chips):
        for batch_idx in range(batch):
            if ttnn_output_torch.dim() == 4:
                ttnn_slice = ttnn_output_torch[c, batch_idx]
            elif ttnn_output_torch.dim() == 3:
                ttnn_slice = ttnn_output_torch[batch_idx]
            else:
                ttnn_slice = ttnn_output_torch

            pcc_passed, pcc_message = comp_pcc(torch_output[c, batch_idx], ttnn_slice, 0.997)
            logger.info(f"chip {c} batch {batch_idx}: {pcc_passed} PCC = {pcc_message}. Threshold = 0.997")
            all_pcc.append(pcc_passed)

    assert all(all_pcc), f"At least one batch output did not match reference {all_pcc=}"
    logger.info("Validation PASSED: ttnn slice/concat baseline matches torch reference")
