# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Any, Literal

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP as ReferenceExpert
from models.demos.deepseek_v3.tt.expert import Expert as TTExpert
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import load_reference_io, load_state_dict, pad_tensor
from models.utility_functions import comp_pcc


class DeepseekV3MoE_Experts(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.experts = nn.ModuleList(
            [
                ReferenceExpert(config, intermediate_size=config.moe_intermediate_size)
                for i in range(config.n_routed_experts)
            ]
        )

    def forward(self, hidden_states):
        outputs = []
        for expert in self.experts:
            outputs.append(expert(hidden_states))

        # returns a tensor of shape (topK_experts, batch_size, seq_len, hidden_size)
        return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 128),
        # ("prefill", 2048),
    ],
)
@pytest.mark.parametrize(
    "weight_type",
    # ["random", "real"],
    ["real"],
)
@pytest.mark.parametrize(
    "model_path",
    [Path("/proj_sw/user_dev/deepseek-ai/")],
)
@pytest.mark.parametrize(
    "module_path",
    ["model.layers.3.mlp.experts.0-255"],
)
def test_forward_pass(
    mode: Literal["decode"] | Literal["prefill"],
    seq_len: Literal[128] | Literal[256] | Literal[2048],
    hf_config: Any,
    temp_dir: Path,
    mesh_device: Any,
    weight_type: Literal["random"] | Literal["real"],
    module_path: str,
    model_path: Path,
):
    """Test forward pass against reference model."""

    torch.manual_seed(0)
    batch_size = 1

    galaxy_or_t3k_mesh = mesh_device

    # Create input tensor
    if weight_type == "random":
        reference_model = DeepseekV3MoE_Experts(hf_config)
        hf_state_dict = reference_model.state_dict()
        torch_input = torch.randn(batch_size, 1, seq_len, hf_config.hidden_size)
        reference_output = reference_model(torch_input)
    elif weight_type == "real":
        hf_state_dict = load_state_dict(model_path, module_path)
        out_ = load_reference_io(mode, module_path)
        breakpoint()
        (io_module_path, (torch_input,), _, reference_output) = out_[0]
        assert module_path == io_module_path
        torch_input.unsqueeze_(0)
        reference_output.unsqueeze_(0)
    else:
        raise ValueError(f"Unsupported weight type: {weight_type}. Supported types are 'random' and 'real'.")

    # Pad input to SEQ_LEN_CHUNK_SIZE if necessasry
    print(torch_input.shape, mode, seq_len)
    torch_input = pad_tensor(torch_input, mode, seq_len)
    print(torch_input.shape)

    # Setup: Convert weights and get weight_config
    weight_config = TTExpert.convert_weights_moe(hf_config, hf_state_dict, temp_dir, galaxy_or_t3k_mesh)
    # Generate appropriate config
    if mode == "prefill":
        model_config = TTExpert.prefill_model_config(hf_config, galaxy_or_t3k_mesh)
    else:
        model_config = TTExpert.decode_model_config(hf_config, galaxy_or_t3k_mesh)

    # Create a new model state
    model_state = TTExpert.create_state(hf_config, mesh_device=galaxy_or_t3k_mesh)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input.repeat(1, run_config["num_experts_per_device"], 1, 1),  # repeating activations for each expert
        device=galaxy_or_t3k_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(galaxy_or_t3k_mesh),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass
    if mode == "prefill":
        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_output = TTExpert.forward_prefill(tt_input, run_config)
        expected_output_memory_config = run_config["output_memory_config"]
    else:
        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_output = TTExpert.forward_decode(tt_input, run_config)
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
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            galaxy_or_t3k_mesh, dims=(0, 1), mesh_shape=tuple(galaxy_or_t3k_mesh.shape)
        ),
    )
    # example shape (4, experts_per_device*8, seq_len, hidden_size) for TG
    tt_output_torch = tt_output_torch.reshape(batch_size, -1, seq_len, hf_config.hidden_size)
    # example shape (1, experts_per_device*8*4, seq_len, hidden_size) for TG
    tt_output_torch = tt_output_torch[0].unsqueeze(1)

    # Compare outputs
    pcc_required = 0.98  # Slightly lower due to bfloat conversions
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"TTExpert output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)


if __name__ == "__main__":
    pytest.main([__file__])
