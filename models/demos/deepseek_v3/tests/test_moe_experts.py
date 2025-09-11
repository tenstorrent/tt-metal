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
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    load_state_dict,
    pad_or_trim_seq_len,
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


def create_combined_state_dict(module_path: str, model_path: Path) -> dict:
    """
    Create a combined state_dict from multiple experts state_dicts.
    """
    # Load individual expert state_dicts and combine them
    parts = module_path.split(".")
    base_path = ".".join(parts[:-1])
    s, e = module_path.split(".")[-1].split("-")
    s, e = int(s), int(e)
    state_dict = {}
    for i in range(s, e + 1):
        module_path_i = f"{base_path}.{i}"
        state_dict_i = load_state_dict(model_path, module_path_i)
        for k, v in state_dict_i.items():
            k_ = f"{base_path.split('.')[-1]}.{i}.{k}"
            state_dict[k_] = v

    # Remove weight_scale_inv keys from state_dict as reference model does not have them
    keys_to_remove = [k for k in state_dict.keys() if "weight_scale_inv" in k]
    for k in keys_to_remove:
        del state_dict[k]

    return state_dict


def get_reference_model(weight_type, hf_config, module_path, model_path: Path) -> DeepseekV3MoEExperts:
    reference_model = DeepseekV3MoEExperts(hf_config)
    if weight_type == "real":
        # Load the state_dict from the specified module path and model path
        state_dict = create_combined_state_dict(module_path, model_path)
        reference_model.load_state_dict(state_dict)
    return reference_model


def get_reference_input(batch_size, seq_len, hf_config):
    return torch.randn(batch_size, 1, seq_len, hf_config.hidden_size)


def get_reference_output(torch_input, reference_model):
    return reference_model(torch_input)


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
    mesh_device: Any,
    weight_type: str,
    module_path: str,
    model_path: Path,
    reset_seeds: Any,
):
    batch_size = 1

    reference_model = get_reference_model(weight_type, hf_config, module_path, model_path)
    torch_input = get_reference_input(batch_size, seq_len, hf_config)
    reference_output = get_reference_output(torch_input, reference_model)

    torch_input = pad_or_trim_seq_len(torch_input, mode, seq_len)
    # Generate module configs and state
    weight_config = TTExperts.convert_weights(hf_config, reference_model.state_dict(), tmp_path, mesh_device)
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
