# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.lm_head import LMHead
from models.demos.deepseek_v3.utils.config_helpers import _check_weights_exist_and_convert, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    pad_or_trim_seq_len,
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
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
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
    mesh_device: ttnn.Device,
    ccl: CCL,
    cache_path: Path,
    repeat_batches,
    set_deterministic_env: Any,
):
    assert mesh_device.get_num_devices() == 32, "Mesh device must have 32 devices for this test."

    reference_model = DeepseekV3LMHead(hf_config).eval()
    state_dict = sub_state_dict(reference_model.state_dict(), "lm_head.")
    torch_input = torch.randn(1, 1, seq_len, hf_config.hidden_size)
    reference_output = reference_model(torch_input)

    # Pad input to SEQ_LEN_CHUNK_SIZE if necessary
    torch_input = pad_or_trim_seq_len(torch_input, mode, seq_len)

    weight_cache_path = (
        cache_path
        / "tests_cache"
        / os.environ.get("PYTEST_CURRENT_TEST")
        / f"{hf_config.num_hidden_layers}_layers"
        / f"mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    )

    # Setup: Convert weights and get weight_config
    weight_config = LMHead.convert_weights(hf_config, (state_dict,), weight_cache_path, mesh_device)
    _check_weights_exist_and_convert(weight_cache_path, weight_config)
    model_config = get_model_config(LMHead, mode, hf_config, mesh_device, 3)
    model_state = LMHead.create_state(hf_config, mesh_device, ccl)
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

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    for iteration in range(repeat_batches):
        tt_output = run_module_forward(LMHead, mode, tt_input, run_config)

        if iteration == 0:
            expected_output_memory_config = run_config["output_memory_config"]

            # Verify output memory config matches expected
            actual_output_memory_config = tt_output.memory_config()
            assert (
                actual_output_memory_config == expected_output_memory_config
            ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

            logger.info("running ttnn.to_torch")
            tt_output_torch = ttnn.to_torch(
                tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)
            )
            logger.info("finished ttnn.to_torch")

            # Check PCC
            assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
        else:
            ttnn.synchronize_device(mesh_device)

        ttnn.deallocate(tt_output)

    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
