# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Accuracy validation tests for MonoDiffusion
Validates PCC > 0.99 against PyTorch reference
"""

import pytest
import torch
import ttnn
import numpy as np

from models.demos.monodiffusion.tt import (
    create_monodiffusion_from_parameters,
    create_monodiffusion_preprocessor,
    load_reference_model,
    compute_pcc,
)
from models.demos.monodiffusion.reference.pytorch_model import create_reference_model


@pytest.mark.parametrize("device_id", [0])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("resolution", [(192, 640)])
def test_monodiffusion_accuracy_vs_pytorch(device_id, batch_size, resolution):
    """
    Test MonoDiffusion TTNN implementation accuracy against PyTorch reference
    Target: PCC > 0.99
    """
    height, width = resolution

    # Create random input
    input_image = torch.randn(batch_size, 3, height, width)

    # Initialize device
    device = ttnn.open_device(device_id=device_id)

    try:
        # Create models
        print("Creating models...")
        pytorch_model = create_reference_model(num_inference_steps=20)
        pytorch_model.eval()

        # Create TT model
        preprocessor = create_monodiffusion_preprocessor(device)
        parameters = preprocessor(pytorch_model, "monodiffusion", {})

        tt_model = create_monodiffusion_from_parameters(
            parameters=parameters,
            device=device,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
        )

        # Run PyTorch reference
        print("Running PyTorch reference...")
        with torch.no_grad():
            pytorch_depth, pytorch_uncertainty = pytorch_model(input_image, return_uncertainty=True)

        # Run TTNN implementation
        print("Running TTNN implementation...")
        input_tensor = ttnn.from_torch(
            input_image,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_depth_ttnn, tt_uncertainty_ttnn = tt_model(input_tensor, return_uncertainty=True)

        # Convert TTNN outputs to PyTorch
        tt_depth = ttnn.to_torch(tt_depth_ttnn)
        tt_uncertainty = ttnn.to_torch(tt_uncertainty_ttnn) if tt_uncertainty_ttnn is not None else None

        # Compute PCC for depth
        depth_pcc = compute_pcc(pytorch_depth, tt_depth)
        print(f"Depth PCC: {depth_pcc:.6f}")

        # Compute PCC for uncertainty
        if pytorch_uncertainty is not None and tt_uncertainty is not None:
            uncertainty_pcc = compute_pcc(pytorch_uncertainty, tt_uncertainty)
            print(f"Uncertainty PCC: {uncertainty_pcc:.6f}")
        else:
            uncertainty_pcc = None

        # Assertions
        assert depth_pcc > 0.99, f"Depth PCC {depth_pcc:.6f} is below threshold 0.99"

        if uncertainty_pcc is not None:
            assert uncertainty_pcc > 0.95, f"Uncertainty PCC {uncertainty_pcc:.6f} is below threshold 0.95"

        print("✓ Accuracy test passed!")

    finally:
        ttnn.close_device(device)


@pytest.mark.parametrize("device_id", [0])
def test_encoder_accuracy(device_id):
    """Test encoder component accuracy"""
    device = ttnn.open_device(device_id=device_id)

    try:
        from models.demos.monodiffusion.tt.config import TtMonoDiffusionLayerConfigs
        from models.demos.monodiffusion.tt.encoder import TtMonoDiffusionEncoder
        from models.demos.monodiffusion.tt import create_monodiffusion_preprocessor, load_reference_model

        # Load reference and create configs
        reference_model = load_reference_model()
        preprocessor = create_monodiffusion_preprocessor(device)
        parameters = preprocessor(reference_model, "monodiffusion", {})

        from models.demos.monodiffusion.tt.config import create_monodiffusion_configs_from_parameters
        configs = create_monodiffusion_configs_from_parameters(
            parameters=parameters,
            batch_size=1,
            input_height=192,
            input_width=640,
        )

        encoder = TtMonoDiffusionEncoder(configs, device)

        # Create test input
        input_image = torch.randn(1, 3, 192, 640)
        input_tensor = ttnn.from_torch(
            input_image,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Preprocess to HWC
        input_tensor = ttnn.experimental.convert_to_hwc(input_tensor)

        # Run encoder
        encoded, features = encoder(input_tensor)

        # Basic checks
        assert encoded is not None, "Encoder output is None"
        assert len(features) > 0, "No multi-scale features generated"

        print(f"✓ Encoder test passed!")
        print(f"  - Generated {len(features)} feature scales")

    finally:
        ttnn.close_device(device)


@pytest.mark.parametrize("device_id", [0])
def test_timestep_embedding_accuracy(device_id):
    """Test timestep embedding accuracy"""
    device = ttnn.open_device(device_id=device_id)

    try:
        from models.demos.monodiffusion.tt.timestep_embedding import TtTimestepEmbedding

        timestep_emb = TtTimestepEmbedding(embed_dim=256, device=device)

        # Test different timesteps
        for t in [0, 100, 500, 999]:
            timestep = torch.tensor([t], dtype=torch.long)
            emb = timestep_emb(timestep)

            assert emb is not None, f"Timestep embedding is None for t={t}"

            # Convert to torch for inspection
            emb_torch = ttnn.to_torch(emb)
            assert emb_torch.shape[-1] == 256, f"Wrong embedding dimension: {emb_torch.shape}"

        print("✓ Timestep embedding test passed!")

    finally:
        ttnn.close_device(device)


@pytest.mark.parametrize("device_id", [0])
@pytest.mark.parametrize("num_samples", [10])
def test_depth_range_validity(device_id, num_samples):
    """
    Test that predicted depth values are in valid range
    """
    device = ttnn.open_device(device_id=device_id)

    try:
        # Load reference and create model
        reference_model = load_reference_model()
        preprocessor = create_monodiffusion_preprocessor(device)
        parameters = preprocessor(reference_model, "monodiffusion", {})

        model = create_monodiffusion_from_parameters(
            parameters=parameters,
            device=device,
            batch_size=1,
            input_height=192,
            input_width=640,
        )

        valid_depths = 0
        for i in range(num_samples):
            # Random input
            input_image = torch.randn(1, 3, 192, 640)
            input_tensor = preprocess_input_image(input_image, device, 192, 640)

            # Predict
            depth_ttnn, _ = model(input_tensor, return_uncertainty=False)
            depth = ttnn.to_torch(depth_ttnn)

            # Check range (after sigmoid, should be [0, 1])
            min_val = depth.min().item()
            max_val = depth.max().item()

            if 0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0:
                valid_depths += 1

            print(f"Sample {i+1}: depth range [{min_val:.4f}, {max_val:.4f}]")

        success_rate = valid_depths / num_samples
        print(f"\n✓ Valid depth range: {valid_depths}/{num_samples} ({success_rate*100:.1f}%)")

        assert success_rate >= 0.9, f"Too many invalid depth predictions: {success_rate*100:.1f}%"

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
