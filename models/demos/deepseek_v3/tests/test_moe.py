# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict

import pytest
import torch
from loguru import logger

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tests.test_moe_experts import generate_synthetic_moe_expert_weights

# Import synthetic weight generators from component tests
from models.demos.deepseek_v3.tests.test_moe_gate import ExpertDistribution
from models.demos.deepseek_v3.tests.test_moe_gate import generate_synthetic_moe_weights as generate_gate_weights
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    dequantize_state_dict,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)


def generate_synthetic_moe_module_weights(
    hf_config,
    gate_distribution: ExpertDistribution = ExpertDistribution.UNIFORM,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic weights for the complete MoE module.

    This combines weights from both the gate and experts components.

    Args:
        hf_config: HuggingFace model configuration
        gate_distribution: Distribution type for gate weights (UNIFORM typically gives best PCC)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing weight tensors for complete MoE module
    """
    torch.manual_seed(seed)

    # Initialize empty weights dict
    weights = {}

    # Generate gate weights using specified distribution
    gate_weights = generate_gate_weights(hf_config, distribution=gate_distribution, seed=seed)

    # Gate weights as float32
    weights["gate.weight"] = gate_weights["weight"].to(torch.float32)
    weights["gate.e_score_correction_bias"] = gate_weights["e_score_correction_bias"]

    # Generate expert weights (they already have inv_scale tensors in FP8 format)
    expert_weights = generate_synthetic_moe_expert_weights(hf_config, seed=seed + 1)

    # Keep the original FP8 quantized expert weights and inv_scale tensors
    # These will be used by both reference model (after dequantization) and TTNN
    weights.update(expert_weights)

    # Add quantization for gate weights (needed for TTNN)
    gate_weight_dict = {"gate.weight": weights["gate.weight"]}
    gate_with_scale = add_inv_scale_to_state_dict(
        gate_weight_dict,
        block_shape=hf_config.quantization_config["weight_block_size"],
        weight_names=["gate"],  # Match weights ending with "gate.weight"
    )
    # Add the inv_scale tensor for gate weight
    for key, value in gate_with_scale.items():
        if key != "gate.weight":  # Don't overwrite the original weight
            weights[key] = value

    return weights


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek MLP model using local implementation."""
    torch.use_deterministic_algorithms(True)
    # Note : Running Reference MoE without shared experts
    hf_config.n_shared_experts = None
    return DeepseekV3MoE(hf_config).eval()


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "use_synthetic_weights, gate_distribution",
    [
        (False, None),  # Real weights, no distribution needed
        (True, ExpertDistribution.UNIFORM),  # Best PCC based on test_moe_gate
        (True, ExpertDistribution.SPARSE),
        (True, ExpertDistribution.CLUSTERED),
        (True, ExpertDistribution.POWER_LAW),
    ],
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 128),
    ]
    + [
        ("prefill", seq_len)
        if seq_len == 128
        else pytest.param(
            "prefill",
            seq_len,
            marks=pytest.mark.skip(
                f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
            ),
        )
        for seq_len in PREFILL_SEQ_LENS
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    set_deterministic_env,
    reference_model,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
    use_synthetic_weights,
    gate_distribution,
):
    """Test forward pass against reference model."""

    batch_size = 1

    if use_synthetic_weights:
        assert gate_distribution is not None, "gate_distribution must be specified when using synthetic weights"
        logger.info(f"Using synthetic weights for MoE module with {gate_distribution.value} gate distribution")

        # Generate synthetic weights (FP8 format with inv_scale tensors)
        synthetic_weights = generate_synthetic_moe_module_weights(
            hf_config, gate_distribution=gate_distribution, seed=42
        )

        # Load synthetic weights into reference model for golden results
        # Dequantize the weights for the reference model
        dequantized_weights = dequantize_state_dict(synthetic_weights, hf_config)

        # Filter out any inv_scale tensors that might be present
        reference_weights = {}
        for key, value in dequantized_weights.items():
            # Skip scale tensors for reference model - check both possible suffixes
            if not (key.endswith("_inv_scale") or key.endswith("_scale_inv") or key.endswith("weight_scale_inv")):
                reference_weights[key] = value

        # Debug: Log what weights we're loading
        logger.info(f"Loading {len(reference_weights)} synthetic weights into reference model")
        logger.info(f"Synthetic weight keys: {sorted(reference_weights.keys())[:10]}...")  # Show first 10 keys

        # Load weights and capture missing/unexpected keys
        load_result = reference_model.load_state_dict(reference_weights, strict=False)
        if load_result.missing_keys:
            logger.warning(f"Missing keys in reference model: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.warning(f"Unexpected keys for reference model: {load_result.unexpected_keys}")

        # For TTNN, use the same synthetic weights (which have FP8 weights and inv_scale tensors)
        # TTNN's convert_weights will handle dequantization

        # Debug: Check what synthetic weights we have for experts
        expert_weight_keys = [
            k for k in synthetic_weights.keys() if "experts." in k and ".weight" in k and not "scale" in k
        ]
        expert_scale_keys = [k for k in synthetic_weights.keys() if "experts." in k and "scale" in k]
        logger.info(f"Expert weight keys (first 5): {sorted(expert_weight_keys)[:5]}")
        logger.info(f"Expert scale keys (first 5): {sorted(expert_scale_keys)[:5]}")

        # Check dtype and shape of first expert weight
        if expert_weight_keys:
            first_weight_key = sorted(expert_weight_keys)[0]
            first_weight = synthetic_weights[first_weight_key]
            logger.info(
                f"First expert weight {first_weight_key}: dtype={first_weight.dtype}, shape={first_weight.shape}"
            )

            # Check if corresponding scale exists
            scale_key = first_weight_key.replace(".weight", ".weight_scale_inv")
            if scale_key in synthetic_weights:
                scale_tensor = synthetic_weights[scale_key]
                logger.info(f"Corresponding scale tensor: dtype={scale_tensor.dtype}, shape={scale_tensor.shape}")

        state_dict = synthetic_weights
    else:
        logger.info("Using real weights for MoE module")
        # Get state dict from actual model - pass directly to convert_weights
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )

    # Create input tensor with deterministic seed
    torch.manual_seed(42)  # Ensure deterministic input
    torch_input = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_output = reference_model(torch_input)

    weight_config = get_test_weight_config(
        MoE, hf_config, (state_dict,), cache_path, mesh_device, force_recalculate=use_synthetic_weights
    )

    # Generate appropriate config using utility function
    model_config = get_model_config(MoE, mode, hf_config, mesh_device, topk_fallback=topk_fallback)

    # Create a new model state with CCL
    model_state = MoE.create_state(hf_config, mesh_device, ccl)

    # Create a new model shared state
    model_shared_state = MoE.create_shared_state(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Convert input to TTNN, DP=4 and Replicated
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass using utility function
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(MoE, mode, tt_input, run_config)

    # Verify output memory config matches expected
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"MoE output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Compare outputs using utility function
    logger.info(f"Mode: {mode}, Num tokens: {seq_len}")
    logger.info(f"Reference output - mean: {reference_output.mean():.6f}, std: {reference_output.std():.6f}")
    logger.info(f"TT output - mean: {tt_output_torch.mean():.6f}, std: {tt_output_torch.std():.6f}")
    assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
