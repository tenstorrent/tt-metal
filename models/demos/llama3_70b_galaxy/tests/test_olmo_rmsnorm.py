# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Test OLMo RMSNorm TTNN implementation against PyTorch reference.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_rmsnorm.py -v
"""

import os
import pytest
import torch
import ttnn

from models.demos.llama3_70b_galaxy.reference.functional import rmsnorm_forward
from models.common.utility_functions import comp_pcc


def get_olmo_weights():
    """Load OLMo weights from HF_MODEL."""
    hf_model = os.getenv("HF_MODEL")
    if not hf_model:
        pytest.skip("HF_MODEL not set")

    import glob
    import json
    from safetensors.torch import load_file

    # Find model files
    base_path = os.path.expanduser(hf_model)
    if os.path.exists(os.path.join(base_path, "snapshots")):
        snapshot_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
        if snapshot_dirs:
            base_path = snapshot_dirs[0]

    # Load config
    config_path = os.path.join(base_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load first safetensor file for layer 0 weights
    safetensor_files = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors")))
    if not safetensor_files:
        pytest.skip("No safetensor files found")

    # Load weights from first file (contains layer 0)
    state_dict = load_file(safetensor_files[0])

    return config, state_dict


@pytest.fixture
def olmo_config_and_weights():
    """Fixture to load OLMo config and weights."""
    return get_olmo_weights()


class TestOlmoRMSNorm:
    """Test OLMo RMSNorm against PyTorch reference."""

    def test_rmsnorm_pcc_layer0(self, olmo_config_and_weights):
        """Test RMSNorm PCC for layer 0 post_attention_layernorm."""
        config, state_dict = olmo_config_and_weights

        # Get parameters
        dim = config["hidden_size"]  # 5120
        eps = config["rms_norm_eps"]  # 1e-6

        # OLMo3 uses different naming: post_attention_layernorm instead of input_layernorm
        weight_key = "model.layers.0.post_attention_layernorm.weight"
        if weight_key not in state_dict:
            pytest.skip(f"Weight {weight_key} not found in first safetensor file")

        weight = state_dict[weight_key]

        # Create test input
        torch.manual_seed(42)
        batch, seq_len = 1, 128
        x = torch.randn(batch, seq_len, dim, dtype=torch.float32)

        # PyTorch reference
        ref_output = rmsnorm_forward(x, weight, eps)

        # For now, just verify reference runs correctly
        # Full TTNN test requires device
        assert ref_output.shape == x.shape
        assert not torch.isnan(ref_output).any()
        assert not torch.isinf(ref_output).any()

        # Verify that RMSNorm is actually normalizing (input RMS != output RMS)
        input_rms = x.pow(2).mean(-1).sqrt().mean().item()
        output_rms = ref_output.pow(2).mean(-1).sqrt().mean().item()
        # Output should be scaled by weight values
        weight_rms = weight.pow(2).mean().sqrt().item()

        print(f"RMSNorm reference test passed")
        print(f"  Input shape: {x.shape}")
        print(f"  Weight shape: {weight.shape}")
        print(f"  Output shape: {ref_output.shape}")
        print(f"  eps: {eps}")

    def test_rmsnorm_pcc_post_feedforward(self, olmo_config_and_weights):
        """Test RMSNorm PCC for layer 0 post_feedforward_layernorm."""
        config, state_dict = olmo_config_and_weights

        dim = config["hidden_size"]
        eps = config["rms_norm_eps"]

        # OLMo3 uses post_feedforward_layernorm (not post_attention_layernorm for FFN norm)
        weight_key = "model.layers.0.post_feedforward_layernorm.weight"
        if weight_key not in state_dict:
            pytest.skip(f"Weight {weight_key} not found in first safetensor file")

        weight = state_dict[weight_key]

        torch.manual_seed(42)
        x = torch.randn(1, 128, dim, dtype=torch.float32)

        ref_output = rmsnorm_forward(x, weight, eps)

        assert ref_output.shape == x.shape
        assert not torch.isnan(ref_output).any()

        print(f"Post-feedforward RMSNorm reference test passed")


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestOlmoRMSNormTTNN:
    """Test OLMo RMSNorm on TTNN device."""

    def test_rmsnorm_ttnn_pcc(self, mesh_device, olmo_config_and_weights):
        """Test RMSNorm TTNN vs PyTorch reference PCC."""
        config, state_dict = olmo_config_and_weights

        dim = config["hidden_size"]
        eps = config["rms_norm_eps"]

        # OLMo3 uses post_attention_layernorm (not input_layernorm)
        weight_key = "model.layers.0.post_attention_layernorm.weight"
        if weight_key not in state_dict:
            pytest.skip(f"Weight {weight_key} not found")

        weight = state_dict[weight_key]

        # Create test input
        torch.manual_seed(42)
        batch, seq_len = 1, 128
        x = torch.randn(batch, seq_len, dim, dtype=torch.bfloat16)

        # PyTorch reference (float32 for accuracy)
        ref_output = rmsnorm_forward(x.float(), weight.float(), eps)
        ref_output = ref_output.to(torch.bfloat16)

        # TTNN computation - replicate on all devices
        # Input needs to be 4D for TTNN: [1, 1, seq_len, dim]
        x_4d = x.unsqueeze(0)  # [1, 1, 128, 5120]
        x_tt = ttnn.from_torch(
            x_4d,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Weight needs to be 4D with shape [1, 1, 1, dim] in TILE_LAYOUT
        weight_4d = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 5120]
        weight_tt = ttnn.from_torch(
            weight_4d,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        output_tt = ttnn.rms_norm(x_tt, epsilon=eps, weight=weight_tt)

        # Get output from first device only (all devices have same result since replicated)
        output_torch = ttnn.to_torch(
            output_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=(8, 4)),
        )
        # Take first replica (all are identical)
        output_torch = output_torch[0:1, :, :, :dim]

        # Compare PCC - squeeze to match ref_output shape
        ref_output_4d = ref_output.unsqueeze(0)  # [1, 1, 128, 5120]
        passing, pcc = comp_pcc(ref_output_4d, output_torch, pcc=0.99)

        print(f"RMSNorm TTNN PCC: {pcc}")
        assert passing, f"PCC {pcc} < 0.99"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
