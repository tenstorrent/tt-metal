# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
import os
from models.common.utility_functions import (
    comp_pcc,
    comp_allclose,
)


def compute_atol_stats(torch_ref, ttnn_result):
    """Compute ATOL statistics between torch reference and ttnn result."""
    diff = torch.abs(torch_ref.float() - ttnn_result.float())
    atol_max = diff.max().item()
    atol_mean = diff.mean().item()
    atol_std = diff.std().item()
    return atol_max, atol_mean, atol_std


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "layer_num",
    [6, 7],
)
@pytest.mark.parametrize(
    "use_fp32",
    [True, False],
    ids=["fp32", "bfp8"],
)
def test_binary_mult_from_saved_tensors(mesh_device, layer_num, use_fp32, reset_seeds):
    """
    Test binary mult (silu(w1) * w3) operation by loading saved tensors from MLP.
    Compares ttnn implementation against torch reference.
    """
    # Path to saved tensors - adjust as needed
    tensor_dir = "/localdev/aling/tt-metal"
    w1_path = os.path.join(tensor_dir, f"w1_out_reduced_layer{layer_num}.pt")
    w3_path = os.path.join(tensor_dir, f"w3_out_reduced_layer{layer_num}.pt")

    # Check if files exist
    if not os.path.exists(w1_path):
        pytest.skip(f"Tensor file not found: {w1_path}")
    if not os.path.exists(w3_path):
        pytest.skip(f"Tensor file not found: {w3_path}")

    # Load saved tensors
    logger.info(f"Loading tensors for layer {layer_num}")
    w1_torch = torch.load(w1_path)
    w3_torch = torch.load(w3_path)

    logger.info(f"w1_out_reduced shape: {w1_torch.shape}, dtype: {w1_torch.dtype}")
    logger.info(f"w3_out_reduced shape: {w3_torch.shape}, dtype: {w3_torch.dtype}")

    # Compute torch reference: silu(w1) * w3
    logger.info("Computing torch reference: silu(w1) * w3")
    torch_silu_w1 = torch.nn.functional.silu(w1_torch.float())
    torch_ff1ff3 = torch_silu_w1 * w3_torch.float()

    logger.info(f"Torch output shape: {torch_ff1ff3.shape}")

    # Convert to ttnn tensors
    # The saved tensors have shape [1, 1, 32, hidden_dim] after slicing from mesh
    # We need to shard them back to the mesh
    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 3), mesh_shape=list(mesh_device.shape))

    w1_ttnn = ttnn.from_torch(
        w1_torch,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    w3_ttnn = ttnn.from_torch(
        w3_torch,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"TTNN w1 shape: {w1_ttnn.shape}")
    logger.info(f"TTNN w3 shape: {w3_ttnn.shape}")

    # Run ttnn binary mult
    if use_fp32:
        logger.info("Running TTNN silu + mul in FP32 precision")
        # Cast to fp32
        w1_fp32 = ttnn.typecast(w1_ttnn, ttnn.float32)
        w1_silu_fp32 = ttnn.silu(w1_fp32)
        ttnn.deallocate(w1_fp32)

        w3_fp32 = ttnn.typecast(w3_ttnn, ttnn.float32)

        ff1ff3_fp32 = ttnn.mul(
            w1_silu_fp32,
            w3_fp32,
            dtype=ttnn.float32,
        )

        # Cast back to bfp8
        ff1ff3_ttnn = ttnn.typecast(ff1ff3_fp32, ttnn.bfloat8_b)
        ttnn.deallocate(ff1ff3_fp32)
        ttnn.deallocate(w1_silu_fp32)
        ttnn.deallocate(w3_fp32)
    else:
        logger.info("Running TTNN silu + mul with fused activation")
        ff1ff3_ttnn = ttnn.mul(
            w1_ttnn,
            w3_ttnn,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
        )

    # Convert ttnn output back to torch
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=[8, 4])
    ff1ff3_ttnn_torch = ttnn.to_torch(ff1ff3_ttnn, mesh_composer=mesh_composer)[:1, :, :, :]

    logger.info(f"TTNN output shape: {ff1ff3_ttnn_torch.shape}")

    # Compute PCC
    passing, pcc_message = comp_pcc(torch_ff1ff3, ff1ff3_ttnn_torch)
    allclose_result = comp_allclose(torch_ff1ff3, ff1ff3_ttnn_torch)

    # Compute ATOL statistics
    atol_max, atol_mean, atol_std = compute_atol_stats(torch_ff1ff3, ff1ff3_ttnn_torch)

    # Print results
    print(f"\n{'='*80}")
    print(f"Binary Mult Test Results - Layer {layer_num} ({'FP32' if use_fp32 else 'BFP8'})")
    print(f"{'='*80}")
    print(f"PCC: {pcc_message}")
    print(f"Allclose: {allclose_result}")
    print(f"ATOL Max: {atol_max:.6e}")
    print(f"ATOL Mean: {atol_mean:.6e}")
    print(f"ATOL Std: {atol_std:.6e}")
    print(f"{'='*80}\n")

    logger.info(f"PCC: {pcc_message}")
    logger.info(f"Allclose: {allclose_result}")
    logger.info(f"ATOL - Max: {atol_max:.6e}, Mean: {atol_mean:.6e}, Std: {atol_std:.6e}")

    # Cleanup
    ttnn.deallocate(w1_ttnn)
    ttnn.deallocate(w3_ttnn)
    ttnn.deallocate(ff1ff3_ttnn)

    assert passing, f"PCC check failed: {pcc_message}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "layer_num",
    [7],
)
def test_binary_mult_detailed_comparison(mesh_device, layer_num, reset_seeds):
    """
    Detailed comparison of binary mult with per-device analysis.
    Tests both FP32 and BFP8 paths and compares them.
    """
    # Path to saved tensors
    tensor_dir = "/localdev/aling/tt-metal"
    w1_path = os.path.join(tensor_dir, f"w1_out_reduced_layer{layer_num}.pt")
    w3_path = os.path.join(tensor_dir, f"w3_out_reduced_layer{layer_num}.pt")

    if not os.path.exists(w1_path) or not os.path.exists(w3_path):
        pytest.skip(f"Tensor files not found for layer {layer_num}")

    # Load saved tensors
    w1_torch = torch.load(w1_path)
    w3_torch = torch.load(w3_path)

    # Compute torch reference
    torch_silu_w1 = torch.nn.functional.silu(w1_torch.float())
    torch_ff1ff3 = torch_silu_w1 * w3_torch.float()

    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 3), mesh_shape=list(mesh_device.shape))
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=[8, 4])

    results = {}

    for method_name, use_fp32 in [("BFP8 Fused", False), ("FP32 Separate", True)]:
        # Create fresh ttnn tensors for each test
        w1_ttnn = ttnn.from_torch(
            w1_torch,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        w3_ttnn = ttnn.from_torch(
            w3_torch,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if use_fp32:
            w1_fp32 = ttnn.typecast(w1_ttnn, ttnn.float32)
            w1_silu_fp32 = ttnn.silu(w1_fp32)
            ttnn.deallocate(w1_fp32)

            w3_fp32 = ttnn.typecast(w3_ttnn, ttnn.float32)

            ff1ff3_fp32 = ttnn.mul(w1_silu_fp32, w3_fp32, dtype=ttnn.float32)
            ff1ff3_ttnn = ttnn.typecast(ff1ff3_fp32, ttnn.bfloat8_b)

            ttnn.deallocate(ff1ff3_fp32)
            ttnn.deallocate(w1_silu_fp32)
            ttnn.deallocate(w3_fp32)
        else:
            ff1ff3_ttnn = ttnn.mul(
                w1_ttnn,
                w3_ttnn,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat8_b,
            )

        ff1ff3_ttnn_torch = ttnn.to_torch(ff1ff3_ttnn, mesh_composer=mesh_composer)[:1, :, :, :]

        passing, pcc_message = comp_pcc(torch_ff1ff3, ff1ff3_ttnn_torch)
        allclose_result = comp_allclose(torch_ff1ff3, ff1ff3_ttnn_torch)
        atol_max, atol_mean, atol_std = compute_atol_stats(torch_ff1ff3, ff1ff3_ttnn_torch)

        results[method_name] = {
            "pcc": pcc_message,
            "allclose": allclose_result,
            "atol_max": atol_max,
            "atol_mean": atol_mean,
            "atol_std": atol_std,
            "passing": passing,
        }

        ttnn.deallocate(w1_ttnn)
        ttnn.deallocate(w3_ttnn)
        ttnn.deallocate(ff1ff3_ttnn)

    # Print comparison
    print(f"\n{'='*100}")
    print(f"Binary Mult Comparison - Layer {layer_num}")
    print(f"{'='*100}")
    print(f"{'Method':<20} {'PCC':<30} {'ATOL Max':<15} {'ATOL Mean':<15} {'ATOL Std':<15}")
    print(f"{'-'*100}")
    for method_name, res in results.items():
        print(
            f"{method_name:<20} {res['pcc']:<30} {res['atol_max']:<15.6e} {res['atol_mean']:<15.6e} {res['atol_std']:<15.6e}"
        )
    print(f"{'='*100}\n")

    # Assert at least one method passes
    assert any(r["passing"] for r in results.values()), "All methods failed PCC check"
