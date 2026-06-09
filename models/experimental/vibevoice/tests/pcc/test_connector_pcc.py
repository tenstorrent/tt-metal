# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1a — SpeechConnector PCC test.

Loads real acoustic_connector and semantic_connector weights,
runs reference PyTorch forward and TT forward, asserts PCC >= 0.99.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import MODEL_PATH
from models.experimental.vibevoice.tt.load_weights import (
    load_vibevoice_state_dict,
    split_submodule_weights,
)
from models.experimental.vibevoice.tt.ttnn_speech_connector import (
    preprocess_connector_parameters,
    TTSpeechConnector,
)
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config


def _reference_connector_forward(state: dict, x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference SpeechConnector: fc1 → LlamaRMSNorm → fc2."""
    import torch.nn.functional as F

    fc1_w = state["fc1.weight"].to(torch.float32)
    fc2_w = state["fc2.weight"].to(torch.float32)
    norm_w = state["norm.weight"].to(torch.float32)
    fc1_b = state["fc1.bias"].to(torch.float32) if "fc1.bias" in state else None
    fc2_b = state["fc2.bias"].to(torch.float32) if "fc2.bias" in state else None

    x = x.to(torch.float32)
    # fc1
    x = F.linear(x, fc1_w, fc1_b)
    # LlamaRMSNorm (eps=1e-6)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + 1e-6)
    x = x * norm_w
    # fc2
    x = F.linear(x, fc2_w, fc2_b)
    return x


@pytest.fixture(scope="module")
def loaded_weights():
    state_dict = load_vibevoice_state_dict(MODEL_PATH)
    return split_submodule_weights(state_dict)


@pytest.fixture(scope="module")
def vv_config():
    return load_vibevoice_model_config(MODEL_PATH)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("connector_name", ["acoustic_connector", "semantic_connector"])
def test_connector_pcc(mesh_device, loaded_weights, vv_config, connector_name):
    torch.manual_seed(0)

    state = loaded_weights[connector_name]
    if connector_name == "acoustic_connector":
        input_dim = vv_config.acoustic_connector_input_dim
    else:
        input_dim = vv_config.semantic_connector_input_dim

    # Fixed input: [1, 8, input_dim]
    x_torch = torch.randn(1, 8, input_dim, dtype=torch.bfloat16)

    # 1) Reference PyTorch forward
    ref_out = _reference_connector_forward(state, x_torch)  # float32

    # 2) Preprocess weights → device
    params = preprocess_connector_parameters(state, mesh_device, eps=1e-6)
    connector = TTSpeechConnector(params)

    # Pad sequence length to TILE_SIZE (32) if needed
    B, T, C = x_torch.shape
    pad_T = ((T + 31) // 32) * 32
    x_padded = torch.zeros(B, pad_T, C, dtype=torch.bfloat16)
    x_padded[:, :T, :] = x_torch

    x_tt = ttnn.as_tensor(
        x_padded.unsqueeze(0),  # [1, 1, pad_T, C] or use 3D with reshape
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Reshape to [1, 1, pad_T, C] for ttnn.linear (needs 4D)
    # Actually pass as [B*T, C] flattened shape the way ttnn.linear works
    # ttnn.linear: input [*, in], weight [out, in]
    # We need to ensure correct shape

    tt_out = connector(x_tt)
    tt_out_torch = ttnn.to_torch(tt_out).to(torch.float32)
    # Unpad to original T
    tt_out_torch = tt_out_torch.squeeze(0)[:, :T, :]  # [1, T, hidden]

    ref_out_3d = ref_out[:, :T, :]

    passed, pcc_val = comp_pcc(
        ref_out_3d, tt_out_torch.squeeze(0) if tt_out_torch.dim() == 3 else tt_out_torch, pcc=0.99
    )
    assert passed, f"[{connector_name}] PCC {pcc_val:.6f} < 0.99"
