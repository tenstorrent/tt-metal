# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP as ReferenceExpert
from models.demos.deepseek_v3.tt.experts import Experts as TTExperts
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    dequantize_state_dict,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)


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

        # returns a tensor of shape (topK_experts, batch_size, seq_len, hidden_size)
        return torch.cat(outputs, dim=0)


def create_combined_state_dict(module_path: str, model_path: Path, state_dict: dict[str, torch.Tensor]) -> dict:
    """
    Create a combined state_dict from multiple experts state_dicts.
    """
    # Load individual expert state_dicts and combine them
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
        ("prefill", 2048),
    ],
)
@pytest.mark.parametrize(
    "weight_type",
    ["random", "real"],
)
@pytest.mark.parametrize(
    "module_path",
    ["model.layers.3.mlp.experts.0-255"],
)
def test_forward_pass(
    mode: str,
    seq_len: int,
    hf_config: Any,
    tmp_path: Path,
    cache_path: Path,
    mesh_device: Any,
    weight_type: str,
    module_path: str,
    model_path: Path,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict: dict[str, torch.Tensor],
):
    batch_size = 1

    reference_model = DeepseekV3MoEExperts(hf_config).eval()
    torch_input = torch.randn(batch_size, 1, seq_len, hf_config.hidden_size)
    if weight_type == "random":
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(), block_shape=hf_config.quantization_config["weight_block_size"]
        )

        # Do not cache random weights
        cache_path = tmp_path
        force_recalculate_weight_config = True
    else:
        assert weight_type == "real"
        state_dict = create_combined_state_dict(module_path, model_path, state_dict)
        reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))
    reference_output = reference_model(torch_input)

    # Generate module configs and state
    weight_config = get_test_weight_config(
        TTExperts, hf_config, (state_dict,), cache_path, mesh_device, force_recalculate_weight_config
    )
    model_config = get_model_config(TTExperts, mode, hf_config, mesh_device)
    model_state = TTExperts.create_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input.repeat(1, run_config["num_experts_per_device"], 1, 1),  # repeating activations for each expert
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(TTExperts, mode, tt_input, run_config)
    expected_output_memory_config = run_config["output_memory_config"]

    # Verify output memory config matches expected
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # output shape per device  = [1, experts_per_device, seq_len, hidden_size]
    # There are 32 groups of unique experts output in case of TG
    # We first concate rows and then columns to get the final output
    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )
    # example shape (4, experts_per_device*8, seq_len, hidden_size) for TG
    tt_output_torch = tt_output_torch.reshape(1, -1, seq_len, hf_config.hidden_size)
    # example shape (1, experts_per_device*8*4, seq_len, hidden_size) for TG
    tt_output_torch = tt_output_torch[0].unsqueeze(1)

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Check PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
