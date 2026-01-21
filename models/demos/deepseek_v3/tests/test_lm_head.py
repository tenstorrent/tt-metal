# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.lm_head import LMHead
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
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
    ]
    + [("prefill", seq_len) for seq_len in PREFILL_SEQ_LENS],
)
def test_forward_pass(
    mode: str,
    seq_len: int,
    hf_config: Any,
    mesh_device: ttnn.Device,
    ccl: CCL,
    cache_path: Path,
    set_deterministic_env: Any,
):
    # Skip all prefill seq lengths except 128 to avoid exceeding CI workload time
    if mode == "prefill" and seq_len != 128:
        pytest.skip(
            f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
        )
    assert mesh_device.get_num_devices() == 32, "Mesh device must have 32 devices for this test."

    reference_model = DeepseekV3LMHead(hf_config).eval()
    state_dict = sub_state_dict(reference_model.state_dict(), "lm_head.")
    torch_input = torch.randn(1, 1, seq_len, hf_config.hidden_size)
    reference_output = reference_model(torch_input)

    # Pad input to SEQ_LEN_CHUNK_SIZE if necessary
    torch_input = pad_or_trim_seq_len(torch_input, mode, seq_len)

    weight_config = get_test_weight_config(
        LMHead, hf_config, (state_dict,), cache_path, mesh_device, force_recalculate=False
    )
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

    # TTNN forward pass
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(LMHead, mode, tt_input, run_config)

    expected_output_memory_config = run_config["output_memory_config"]

    # Verify output memory config matches expected
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    logger.info("running ttnn.to_torch")
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    logger.info("finished ttnn.to_torch")

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Check PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
