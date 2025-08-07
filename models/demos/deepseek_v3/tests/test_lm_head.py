# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import ttnn
from models.demos.deepseek_v3.tt.lm_head import LMHead as TTLMHead
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    pad_tensor,
    run_module_forward,
)


class DeepseekV3LMHead(nn.Module):
    """
    Language model head for Deepseek V3.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 32),
        ("prefill", 1024),
        ("prefill", 2048),
    ],
)
def test_forward_pass(
    mode: str,
    seq_len: int,
    hf_config: Any,
    tmp_path: Path,
    mesh_device: Any,
):
    assert mesh_device.get_num_devices() == 32, "Mesh device must have 32 devices for this test."
    batch_size = 1
    torch.manual_seed(0)

    reference_model = DeepseekV3LMHead(hf_config).eval()
    hf_state_dict = reference_model.state_dict()
    torch_input = torch.randn(batch_size, 1, seq_len, hf_config.hidden_size)
    reference_output = reference_model(torch_input)

    # Pad input to SEQ_LEN_CHUNK_SIZE if necessary
    print(f"torch_input before pad_tensor: shape = {torch_input.shape}, mode = {mode}, seq_len = {seq_len}")
    torch_input = pad_tensor(torch_input, mode, seq_len)
    print(f"torch_input after pad_tensor: shape = {torch_input.shape}")

    # Setup: Convert weights and get weight_config
    weight_config = TTLMHead.convert_weights(hf_config, hf_state_dict, tmp_path, mesh_device)
    model_config = get_model_config(TTLMHead, mode, hf_config, mesh_device)
    model_state = TTLMHead.create_state(hf_config, mesh_device=mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(TTLMHead, mode, tt_input, run_config)

    expected_output_memory_config = run_config["output_memory_config"]

    # Verify output memory config matches expected
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Check PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
