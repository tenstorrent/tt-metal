# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP as ReferenceExpert
from models.demos.deepseek_v3.tt.experts import Experts as TTExperts
from models.demos.deepseek_v3.utils.config_helpers import dequantize, even_int_div, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    dequantize_state_dict,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)


def generate_synthetic_moe_expert_weights(
    hf_config,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic weights for MoE experts that resemble real trained weights.

    This function generates weights with distributions similar to real DeepSeek V3 MoE expert weights
    based on empirical analysis of the actual model weights from HuggingFace.

    Args:
        hf_config: HuggingFace model configuration
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing weight tensors for all MoE expert components
    """
    torch.manual_seed(seed)

    # Extract dimensions
    hidden_size = hf_config.hidden_size
    moe_intermediate_size = hf_config.moe_intermediate_size
    n_routed_experts = hf_config.n_routed_experts

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

    # Generate weights for all experts
    weights = {}

    # Based on real DeepSeek V3 weight analysis:
    # Expert weights have varying std, but typical ranges are:
    # gate_proj: std ≈ 0.0024-0.0049
    # up_proj: std ≈ 0.0023-0.0048
    # down_proj: std ≈ 0.0038-0.0074

    for expert_idx in range(n_routed_experts):
        prefix = f"experts.{expert_idx}."

        # Add some variation between experts (±30% around mean)
        variation = 1.0 + (torch.randn(1).item() * 0.3)

        # Target standard deviations based on real weight analysis
        gate_std = 0.0035 * variation  # Average of observed range
        up_std = 0.0035 * variation
        down_std = 0.0055 * variation  # Down projections tend to have larger std

        # Create quantized weights with appropriate scales
        gate_weight, gate_scale_base = create_quantized_weight((moe_intermediate_size, hidden_size), gate_std)
        up_weight, up_scale_base = create_quantized_weight((moe_intermediate_size, hidden_size), up_std)
        down_weight, down_scale_base = create_quantized_weight((hidden_size, moe_intermediate_size), down_std)

        weights[f"{prefix}gate_proj.weight"] = gate_weight
        weights[f"{prefix}up_proj.weight"] = up_weight
        weights[f"{prefix}down_proj.weight"] = down_weight

        # Generate scale tensors with proper blocked dimensions
        weights[f"{prefix}gate_proj.weight_scale_inv"] = create_scale_tensor_from_base(
            (moe_intermediate_size, hidden_size), gate_scale_base
        )
        weights[f"{prefix}up_proj.weight_scale_inv"] = create_scale_tensor_from_base(
            (moe_intermediate_size, hidden_size), up_scale_base
        )
        weights[f"{prefix}down_proj.weight_scale_inv"] = create_scale_tensor_from_base(
            (hidden_size, moe_intermediate_size), down_scale_base
        )

    # Ensure proper dtypes for scale tensors
    for key in weights:
        if "scale_inv" in key:
            weights[key] = weights[key].to(torch.float32)

    return weights


class DeepseekV3MoEExperts(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.experts = nn.ModuleList(
            [
                ReferenceExpert(config, intermediate_size=config.moe_intermediate_size).eval()
                for i in range(config.n_routed_experts)
            ]
        )

    def forward(self, hidden_states):
        outputs = []
        for expert in self.experts:
            outputs.append(expert(hidden_states))

        return torch.cat(outputs, dim=0)


def create_combined_state_dict(module_path: str, model_path: Path, state_dict: dict[str, torch.Tensor]) -> dict:
    """
    Create a combined state_dict from multiple experts state_dicts.
    """
    parts = module_path.split(".")
    base_path = ".".join(parts[:-1])
    s, e = module_path.split(".")[-1].split("-")
    s, e = int(s), int(e)
    out_state_dict = {}
    for i in range(s, e + 1):
        module_path_i = f"{base_path}.{i}"
        state_dict_i = sub_state_dict(state_dict, module_path_i + ".")
        for k, v in state_dict_i.items():
            k_ = f"{base_path.split('.')[-1]}.{i}.{k}"
            out_state_dict[k_] = v

    return out_state_dict


@pytest.mark.parametrize(
    "mode, seq_len",
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
@pytest.mark.parametrize(
    "use_synthetic_weights",
    [True, False],  # Test both synthetic and real weights
    ids=["synthetic", "real"],
)
@pytest.mark.parametrize(
    "module_path",
    ["model.layers.3.mlp.experts.0-255"],
)
def test_forward_pass(
    mode: str,
    seq_len: int,
    hf_config: Any,
    cache_path: Path,
    mesh_device: Any,
    use_synthetic_weights: bool,
    module_path: str,
    model_path: Path,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict: dict[str, torch.Tensor],
):
    batch_size = 1
    num_experts_per_device = even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())

    reference_model = DeepseekV3MoEExperts(hf_config).eval()

    # Use deterministic input for reproducibility, especially important for synthetic weights
    torch.manual_seed(42)  # Fixed seed for input generation
    torch_input = torch.randn(batch_size, 1, seq_len, hf_config.hidden_size)

    if use_synthetic_weights:
        # Generate synthetic weights
        synthetic_weights = generate_synthetic_moe_expert_weights(hf_config)

        # Dequantize synthetic weights for the reference model
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

        # Load dequantized weights into reference model
        reference_model.load_state_dict(dequantized_weights)

        # Use synthetic weights (with quantization) for TTNN weight conversion
        state_dict = synthetic_weights
    else:
        # Use real weights
        state_dict = create_combined_state_dict(module_path, model_path, state_dict)
        reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))

    # Force recalculation when using synthetic weights to avoid cache issues
    weight_config = get_test_weight_config(
        TTExperts,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        use_synthetic_weights or force_recalculate_weight_config,
    )
    model_config = get_model_config(TTExperts, mode, hf_config, mesh_device)
    model_state = TTExperts.create_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state)

    tt_input = ttnn.from_torch(
        torch_input.repeat(1, run_config["num_experts_per_device"], 1, 1),  # repeat activations per expert
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(TTExperts, mode, tt_input, run_config)
    expected_output_memory_config = run_config["output_memory_config"]

    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    TARGET_CHUNK_SIZE = 2048
    num_chunks = (seq_len + TARGET_CHUNK_SIZE - 1) // TARGET_CHUNK_SIZE

    from models.common.utility_functions import comp_pcc

    min_pcc = 0.98
    passed = True

    for chunk_idx in range(num_chunks):
        start_seq = chunk_idx * TARGET_CHUNK_SIZE
        end_seq = min(start_seq + TARGET_CHUNK_SIZE, seq_len)
        chunk_seq_len = end_seq - start_seq

        chunk_input = torch_input[:, :, start_seq:end_seq, :]
        chunk_ref_output = reference_model(chunk_input)

        tt_output_chunk = ttnn.slice(
            tt_output,
            [0, 0, start_seq, 0],
            [1, num_experts_per_device, end_seq, hf_config.hidden_size],
        )

        tt_output_chunk_torch = ttnn.to_torch(
            tt_output_chunk,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
        )

        ttnn.deallocate(tt_output_chunk)

        tt_output_chunk_torch = tt_output_chunk_torch.reshape(1, -1, chunk_seq_len, hf_config.hidden_size)
        tt_output_chunk_torch = tt_output_chunk_torch[0].unsqueeze(1)

        if chunk_ref_output.shape != tt_output_chunk_torch.shape:
            chunk_ref_output = chunk_ref_output.unsqueeze(0)

        chunk_passed, chunk_pcc = comp_pcc(tt_output_chunk_torch, chunk_ref_output, pcc=0.98)

        print(f"Chunk {chunk_idx} PCC: {chunk_pcc:.6f}")
        min_pcc = min(min_pcc, chunk_pcc)
        if not chunk_passed:
            passed = False

        del chunk_ref_output
        del tt_output_chunk_torch
        del chunk_input

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    assert passed, f"PCC check failed! Min PCC: {min_pcc:.6f} < 0.98"


if __name__ == "__main__":
    pytest.main([__file__])
