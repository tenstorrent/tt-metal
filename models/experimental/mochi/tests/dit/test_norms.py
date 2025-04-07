import os
import torch
import pytest
import ttnn
from loguru import logger
from models.utility_functions import skip_for_grayskull, comp_allclose
from models.experimental.mochi.tt.dit.norms import modulated_rmsnorm
from models.experimental.mochi.tt.dit.norms import residual_tanh_gated_rmsnorm
from models.experimental.mochi.tt.common import compute_metrics

from genmo.mochi_preview.dit.joint_model.residual_tanh_gated_rmsnorm import (
    residual_tanh_gated_rmsnorm as ref_residual_tanh_gated_rmsnorm,
)


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "S, D",
    [
        (44520, 3072),
        # (118, 1536),
    ],
)
def test_modulated_rmsnorm(mesh_device, use_program_cache, reset_seeds, S, D):
    dtype = ttnn.bfloat16

    mesh_device.enable_async(True)

    # Create random input and scale tensors
    torch_input = torch.randn(1, 1, S, D)
    torch_scale = torch.randn(1, 1, 1, D) * 0.1  # Small scale modulation

    # Convert to TT tensors
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_scale = ttnn.from_torch(
        torch_scale,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TT implementation
    tt_output = modulated_rmsnorm(tt_input, tt_scale, high_fidelity=True)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).float()

    # Reference implementation in PyTorch
    def reference_modulated_rmsnorm(x, scale, eps=1e-6):
        weight = 1.0 + scale
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps) * weight
        return x

    reference_output = reference_modulated_rmsnorm(torch_input, torch_scale)

    # Compute metrics
    pcc, mse, mae = compute_metrics(reference_output, tt_output_torch)
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")
    print(comp_allclose(reference_output, tt_output_torch.view(reference_output.shape)))

    # Check if model meets requirements
    pcc_required = 0.99
    passing = pcc >= pcc_required

    if passing:
        logger.info("Modulated RMSNorm Passed!")
    else:
        logger.warning("Modulated RMSNorm Failed!")

    assert (
        passing
    ), f"Modulated RMSNorm output does not meet PCC requirement {pcc_required}: {pcc}, MSE: {mse}, MAE: {mae}."


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "S, D",
    [
        (44520, 3072),
        (256, 1536),
    ],
)
def test_residual_tanh_gated_rmsnorm(mesh_device, use_program_cache, reset_seeds, S, D):
    dtype = ttnn.bfloat16
    mesh_device.enable_async(True)

    # Create random input tensors
    torch_x = torch.randn(1, 1, S, D)
    torch_x_res = torch.randn(1, 1, S, D)
    torch_gate = torch.randn(1, D)  # Gate is per sequence position

    # Convert to TT tensors
    tt_x = ttnn.from_torch(
        torch_x,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_x_res = ttnn.from_torch(
        torch_x_res,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_gate = ttnn.from_torch(
        torch_gate,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TT implementation
    tt_output = residual_tanh_gated_rmsnorm(tt_x, tt_x_res, tt_gate)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    # Run reference implementation
    reference_output = ref_residual_tanh_gated_rmsnorm(torch_x, torch_x_res, torch_gate)

    # Compute metrics
    pcc, mse, mae = compute_metrics(reference_output, tt_output_torch)
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")
    print(comp_allclose(reference_output, tt_output_torch.view(reference_output.shape)))

    # Check if model meets requirements
    pcc_required = 0.99
    passing = pcc >= pcc_required

    if passing:
        logger.info("Residual Tanh Gated RMSNorm Passed!")
    else:
        logger.warning("Residual Tanh Gated RMSNorm Failed!")

    assert passing, f"Output does not meet PCC requirement {pcc_required}: {pcc}, MSE: {mse}, MAE: {mae}."
