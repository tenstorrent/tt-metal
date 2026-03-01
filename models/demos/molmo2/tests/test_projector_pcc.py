# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Compare ImageProjector output between TTNN and PyTorch.
"""

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn


def calculate_pcc(ref, out):
    """Calculate Pearson Correlation Coefficient."""
    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()
    ref_mean = ref_flat.mean()
    out_mean = out_flat.mean()
    numerator = ((ref_flat - ref_mean) * (out_flat - out_mean)).sum()
    denominator = torch.sqrt(((ref_flat - ref_mean) ** 2).sum() * ((out_flat - out_mean) ** 2).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return (numerator / denominator).item()


def main():
    """Compare projector output."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    logger.info("Loading weights...")
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")

    # Get projector weights
    prefix = "model.vision_backbone.image_projector"
    w1 = state_dict[f"{prefix}.w1.weight"]
    w2 = state_dict[f"{prefix}.w2.weight"]
    w3 = state_dict[f"{prefix}.w3.weight"]

    logger.info(f"w1 shape: {w1.shape}, std: {w1.std():.6f}")
    logger.info(f"w2 shape: {w2.shape}, std: {w2.std():.6f}")
    logger.info(f"w3 shape: {w3.shape}, std: {w3.std():.6f}")

    # Create sample input with same stats as pooled_features
    sample_input = torch.randn(1, 64, 1152) * 0.5  # std=0.5
    logger.info(f"\nSample input: shape={sample_input.shape}, std={sample_input.std():.4f}")

    # ========= PyTorch Reference =========
    logger.info("\n=== PyTorch Reference ===")
    gate = F.silu(F.linear(sample_input, w1))
    up = F.linear(sample_input, w3)
    hidden = gate * up
    pytorch_output = F.linear(hidden, w2)
    logger.info(
        f"PyTorch output: std={pytorch_output.std():.4f}, range=[{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]"
    )

    # ========= TTNN =========
    logger.info("\n=== TTNN ===")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, 8)
    device = ttnn.open_mesh_device(mesh_shape)

    try:
        from models.demos.molmo2.tt.image_projector import ImageProjector

        # Create TTNN projector
        projector = ImageProjector(
            mesh_device=device,
            state_dict=state_dict,
            input_dim=1152,
            intermediate_dim=12288,
            output_dim=4096,
            dtype=ttnn.bfloat16,  # Use bfloat16 for precision
        )

        # Convert input to TTNN
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)

        x_ttnn = ttnn.from_torch(
            sample_input.unsqueeze(0),  # [1, 1, 64, 1152]
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Run TTNN projector
        ttnn_output = projector(x_ttnn)
        ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=mesh_composer)[0].squeeze()

        logger.info(
            f"TTNN output: std={ttnn_output_torch.std():.4f}, range=[{ttnn_output_torch.min():.4f}, {ttnn_output_torch.max():.4f}]"
        )

        # Compare
        pytorch_squeezed = pytorch_output.squeeze()
        pcc = calculate_pcc(pytorch_squeezed, ttnn_output_torch)
        diff = (pytorch_squeezed - ttnn_output_torch).abs()
        logger.info(f"\nPCC: {pcc:.6f}")
        logger.info(f"Max diff: {diff.max():.4f}")
        logger.info(f"Mean diff: {diff.mean():.6f}")

        # Check scale ratio
        ratio = ttnn_output_torch.std() / pytorch_output.std()
        logger.info(f"Scale ratio (TTNN/PyTorch): {ratio:.4f}x")

        ttnn.deallocate(x_ttnn)
        ttnn.deallocate(ttnn_output)

    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
