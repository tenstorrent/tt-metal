# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP
from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN
from models.demos.deepseek_v3.tt.mlp.mlp import MLP
from models.demos.deepseek_v3.tt.mlp.mlp_dequant import MLPDequant
from models.demos.deepseek_v3.tt.mlp.non_expert import NonExpert
from models.demos.deepseek_v3.tt.mlp.shared_expert import SharedExpert
from models.demos.deepseek_v3.utils.config_helpers import dequantize, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config, load_weight
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    load_reference_io_tensors_for_module,
    run_module_forward,
)


def generate_synthetic_mlp_weights(
    hf_config,
    mlp_type: str = "shared_expert",  # "shared_expert", "non_expert", or "regular"
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic weights for MLP layers that resemble real trained weights.

    This function generates weights with distributions similar to real DeepSeek V3 MLP weights
    based on empirical analysis of the actual model weights from HuggingFace.

    Args:
        hf_config: HuggingFace model configuration
        mlp_type: Type of MLP ("shared_expert", "non_expert", or "regular")
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing weight tensors for all MLP components
    """
    torch.manual_seed(seed)

    # Extract dimensions based on MLP type
    hidden_size = hf_config.hidden_size

    if mlp_type == "shared_expert":
        intermediate_size = hf_config.moe_intermediate_size
    elif mlp_type == "non_expert":
        intermediate_size = hf_config.intermediate_size
    else:  # regular MLP
        intermediate_size = hf_config.intermediate_size

    # Block size for quantization
    block_size = 128

    def create_quantized_weight(shape, target_std_after_dequant):
        """Create FP8 weight and scale that produces target std after dequantization."""
        # Create weights in FP8 range with reasonable distribution
        fp8_std = 30.0  # Use a good portion of FP8 range
        weight_fp8 = (torch.randn(shape) * fp8_std).to(torch.float8_e4m3fn)

        # Calculate inv_scale to achieve target std after dequantization
        # After dequant: weight_float = weight_fp8 * inv_scale
        # We want: std(weight_float) = target_std_after_dequant
        # So: inv_scale ≈ target_std_after_dequant / fp8_std
        inv_scale = target_std_after_dequant / fp8_std
        return weight_fp8, inv_scale

    def create_scale_tensor_from_base(weight_shape, base_inv_scale):
        """Create scale tensor matching the blocked dimensions of the weight."""
        # Calculate number of blocks in each dimension
        num_blocks_0 = (weight_shape[0] + block_size - 1) // block_size
        num_blocks_1 = (weight_shape[1] + block_size - 1) // block_size
        # Create scale tensor with small variation around the base value
        scale = torch.ones(num_blocks_0, num_blocks_1) * base_inv_scale
        # Add small variation (±10%) to simulate block-wise quantization
        scale = scale * (1.0 + torch.randn(num_blocks_0, num_blocks_1) * 0.1)
        # Ensure positive values
        scale = torch.clamp(scale, min=1e-6)
        return scale

    # Generate weights
    weights = {}

    # Based on real DeepSeek V3 weight analysis for shared expert and MoE experts:
    # Shared expert weights have similar distribution to MoE experts
    # gate_proj: std ≈ 0.0024-0.0049
    # up_proj: std ≈ 0.0023-0.0048
    # down_proj: std ≈ 0.0038-0.0074

    # Target standard deviations based on real weight analysis
    if mlp_type == "shared_expert":
        # Shared expert weights - similar to MoE experts but slightly different
        gate_std = 0.0040  # Slightly higher for shared experts
        up_std = 0.0038
        down_std = 0.0060  # Down projections tend to have larger std
    elif mlp_type == "non_expert":
        # Non-expert (first 3 layers) weights
        gate_std = 0.0035
        up_std = 0.0035
        down_std = 0.0055
    else:
        # Regular MLP weights (for non-quantized case)
        # Use Xavier initialization approximation
        gate_std = (2.0 / (hidden_size + intermediate_size)) ** 0.5
        up_std = (2.0 / (hidden_size + intermediate_size)) ** 0.5
        down_std = (2.0 / (intermediate_size + hidden_size)) ** 0.5

    # For regular (non-quantized) MLP, create normal weights
    if mlp_type == "regular":
        weights["gate_proj.weight"] = torch.randn(intermediate_size, hidden_size) * gate_std
        weights["up_proj.weight"] = torch.randn(intermediate_size, hidden_size) * up_std
        weights["down_proj.weight"] = torch.randn(hidden_size, intermediate_size) * down_std

        # Convert to bfloat16 for regular MLP
        for key in weights:
            weights[key] = weights[key].to(torch.bfloat16)
    else:
        # Create quantized weights with appropriate scales
        gate_weight, gate_scale_base = create_quantized_weight((intermediate_size, hidden_size), gate_std)
        up_weight, up_scale_base = create_quantized_weight((intermediate_size, hidden_size), up_std)
        down_weight, down_scale_base = create_quantized_weight((hidden_size, intermediate_size), down_std)

        weights["gate_proj.weight"] = gate_weight
        weights["up_proj.weight"] = up_weight
        weights["down_proj.weight"] = down_weight

        # Generate scale tensors with proper blocked dimensions
        weights["gate_proj.weight_scale_inv"] = create_scale_tensor_from_base(
            (intermediate_size, hidden_size), gate_scale_base
        )
        weights["up_proj.weight_scale_inv"] = create_scale_tensor_from_base(
            (intermediate_size, hidden_size), up_scale_base
        )
        weights["down_proj.weight_scale_inv"] = create_scale_tensor_from_base(
            (hidden_size, intermediate_size), down_scale_base
        )

        # Ensure proper dtypes for scale tensors
        for key in weights:
            if "scale_inv" in key:
                weights[key] = weights[key].to(torch.float32)

    return weights


# TODO: Doesn't work on multi-host - we should figure out why
@pytest.mark.requires_device(["TG"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_convert_weights_for_non_dequantized_mlp(hf_config, tmp_path, mesh_device):
    # Add a skip for mesh device shape 8x8 due to known issue https://github.com/tenstorrent/tt-metal/issues/35375
    if tuple(mesh_device.shape) == (8, 8):
        pytest.skip(
            "Skipping test for mesh device shape 8x8 due to known issue https://github.com/tenstorrent/tt-metal/issues/35375"
        )
    reference_model = DeepseekV3MLP(hf_config).eval()
    reference_state_dict = reference_model.to(torch.bfloat16).state_dict()
    run_weight_conversion_test(
        MLPClass=MLP,
        hf_config=hf_config,
        state_dict=reference_model.state_dict(),
        tmp_path=tmp_path
        / "mesh_8x8",  # TODO: dummy mesh shape required until convert_weights no longer relies on this for parsing the absolutem filepaths
        mesh_device=mesh_device,
        reference_w1=reference_state_dict["gate_proj.weight"],
    )


# TODO: Doesn't work on multi-host - we should figure out why
@pytest.mark.requires_device(["TG"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "MLPClass,module_path",
    [(NonExpert, "model.layers.0.mlp"), (SharedExpert, "model.layers.3.mlp.shared_experts")],
)
def test_convert_weights_for_dequantized_mlps(MLPClass, module_path, hf_config, tmp_path, mesh_device, state_dict):
    if tuple(mesh_device.shape) == (8, 8):
        pytest.skip(
            "Skipping test for mesh device shape 8x8 due to known issue https://github.com/tenstorrent/tt-metal/issues/35375"
        )
    state_dict = sub_state_dict(state_dict, module_path + ".")
    run_weight_conversion_test(
        MLPClass=MLPClass,
        hf_config=hf_config,
        state_dict=state_dict,
        tmp_path=tmp_path
        / "mesh_8x8",  # TODO: dummy mesh shape required until convert_weights no longer relies on this for parsing the absolutem filepaths
        mesh_device=mesh_device,
        reference_w1=dequantize(
            state_dict["gate_proj.weight"],
            state_dict["gate_proj.weight_scale_inv"],
            block_shape=hf_config.quantization_config["weight_block_size"],
        ),
    )


def run_weight_conversion_test(MLPClass, hf_config, state_dict, tmp_path, reference_w1, mesh_device):
    if tuple(mesh_device.shape) == (8, 8):
        pytest.skip(
            "Skipping test for mesh device shape 8x8 due to known issue https://github.com/tenstorrent/tt-metal/issues/35375"
        )
    num_module_layers, _ = mesh_device.shape

    # Convert the weights
    weight_config = MLPClass.convert_weights(
        hf_config, [state_dict] + [None] * (num_module_layers - 1), tmp_path, mesh_device
    )

    # Verify weight_config structure
    assert "w1" in weight_config
    assert "w2" in weight_config
    assert "w3" in weight_config
    assert "input_tensor_b" in weight_config["w1"]
    assert "input_tensor_b" in weight_config["w2"]
    assert "input_tensor_b" in weight_config["w3"]

    # # Verify files exist # TODO: bring regular tensor saving back once Issue #26763 is resolved
    # assert Path(weight_config["w1"]["input_tensor_b"]).exists()
    # assert Path(weight_config["w2"]["input_tensor_b"]).exists()
    # assert Path(weight_config["w3"]["input_tensor_b"]).exists()

    # Make the path absolute - this is required since load_weight expects an absolute path
    weight_config["w1"]["input_tensor_b"].path = tmp_path / weight_config["w1"]["input_tensor_b"].path

    # Load and verify a weight
    w1_ttnn = load_weight(weight_config["w1"]["input_tensor_b"], device=mesh_device)
    w1_ttnn = ttnn.unsqueeze(w1_ttnn, 0)  # Unsqueeze to collect shards on a separate dim
    w1_torch = ttnn.to_torch(
        w1_ttnn,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
    )

    # Weight should be transposed from PyTorch format
    assert w1_torch.shape == (
        num_module_layers,
        *[1 for _ in range(w1_torch.ndim - 3)],
        reference_w1.shape[1],
        reference_w1.shape[0],
    )

    # Verify the values match (accounting for transpose and bfloat8 conversion)
    passing, pcc = comp_pcc(reference_w1.T, w1_torch[0], 0.99)
    logger.info(f"PCC: {pcc}")
    assert passing, f"Weight conversion PCC failed: {pcc}"

    # Cleanup
    ttnn.deallocate(w1_ttnn)


_max_seq_len_env = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
_prefill_seq_len = int(_max_seq_len_env) if _max_seq_len_env is not None else DEFAULT_PREFILL_SEQ_LEN


@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 32),
        ("prefill", _prefill_seq_len),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "MLPClass,module_path",
    [
        (MLP, None),
        (NonExpert, "model.layers.0.mlp"),
        (SharedExpert, "model.layers.3.mlp.shared_experts"),
    ],
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 32),
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
@pytest.mark.parametrize(
    "use_synthetic_weights",
    [True, False],  # Test both synthetic and real weights
)
def test_forward_pass(
    MLPClass,
    module_path,
    mode,
    seq_len,
    use_synthetic_weights,
    hf_config,
    mesh_device,
    ccl,
    model_path,
    tmp_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    num_module_layers, _ = mesh_device.shape

    # Skip loading state_dict from file if using synthetic weights
    if use_synthetic_weights:
        logger.info(f"Using synthetic weights for {MLPClass.__name__}")
        # Pass None as state_dict when using synthetic weights for dequantized MLPs
        if issubclass(MLPClass, MLPDequant):
            state_dict = None
    else:
        logger.info(f"Using real weights for {MLPClass.__name__}")

    # Get the reference IO
    if not issubclass(MLPClass, MLPDequant):
        reference_model = DeepseekV3MLP(hf_config).eval()

        if use_synthetic_weights:
            # Generate synthetic weights for regular MLP
            synthetic_weights = generate_synthetic_mlp_weights(hf_config, mlp_type="regular")
            reference_model.load_state_dict(synthetic_weights, strict=False)
            state_dict = synthetic_weights
        else:
            state_dict = reference_model.to(torch.bfloat16).state_dict()

        # Use deterministic input for reproducibility with synthetic weights
        torch.manual_seed(42)
        torch_input = torch.randn(num_module_layers, 1, seq_len, hf_config.hidden_size)

        reference_model = reference_model.to(torch.float32)
        reference_output = reference_model(torch_input)
    else:
        if use_synthetic_weights:
            # Generate synthetic weights for dequantized MLPs
            if MLPClass == SharedExpert:
                mlp_type = "shared_expert"
            elif MLPClass == NonExpert:
                mlp_type = "non_expert"
            else:
                mlp_type = "regular"

            synthetic_weights = generate_synthetic_mlp_weights(hf_config, mlp_type=mlp_type)

            # For dequantized MLPs, we need to dequantize the weights for the reference model
            dequantized_weights = {}
            block_shape = hf_config.quantization_config["weight_block_size"]

            for name, tensor in synthetic_weights.items():
                if name.endswith("_scale_inv"):
                    continue  # Skip scale tensors
                elif tensor.dtype == torch.float8_e4m3fn:
                    # Dequantize FP8 weights using their scale tensors
                    scale_name = name + "_scale_inv"
                    if scale_name in synthetic_weights:
                        scale_tensor = synthetic_weights[scale_name]
                        # Dequantize using the scale
                        dequantized_tensor = dequantize(tensor, scale_tensor, block_shape)
                        dequantized_weights[name] = dequantized_tensor.to(torch.bfloat16)
                    else:
                        dequantized_weights[name] = tensor.to(torch.bfloat16)
                else:
                    # Keep non-quantized weights as-is
                    dequantized_weights[name] = tensor

            # Create reference model with correct intermediate size
            # For SharedExpert, we need to temporarily modify the config to use moe_intermediate_size
            if MLPClass == SharedExpert:
                # Create a modified config for SharedExpert
                import copy

                shared_config = copy.deepcopy(hf_config)
                shared_config.intermediate_size = hf_config.moe_intermediate_size
                reference_model = DeepseekV3MLP(shared_config).eval()
            else:
                reference_model = DeepseekV3MLP(hf_config).eval()

            reference_model.load_state_dict(dequantized_weights, strict=False)

            # Use deterministic input for reproducibility
            torch.manual_seed(42)
            torch_input = torch.randn(num_module_layers, 1, seq_len, hf_config.hidden_size)

            reference_model = reference_model.to(torch.float32)
            reference_output = reference_model(torch_input)

            # Use synthetic weights (with quantization) for TTNN weight conversion
            state_dict = synthetic_weights
        else:
            state_dict = sub_state_dict(state_dict, module_path + ".")
            torch_input, reference_output = load_reference_io_tensors_for_module(
                mode, module_path, seq_len, num_module_layers
            )

    # Generate module configs and state
    # Force recalculation when using synthetic weights
    weight_config = get_test_weight_config(
        MLPClass,
        hf_config,
        (state_dict,) * num_module_layers,
        cache_path,
        mesh_device,
        use_synthetic_weights or force_recalculate_weight_config,
    )
    model_config = get_model_config(MLPClass, mode, hf_config, mesh_device)
    model_state = MLPClass.create_state(hf_config, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (0, -1)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass
    tt_output = run_module_forward(MLPClass, mode, tt_input, run_config)

    # Verify output memory config matches expected
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
    )

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Check PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.975)


if __name__ == "__main__":
    pytest.main([__file__])
