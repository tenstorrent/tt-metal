# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the forecasting head (TinyTimeMixerForPredictionHead).

The head applies:
  1. Flatten: [B, C, num_patches, decoder_d_model] → [B, C, num_patches*decoder_d_model]
  2. Dropout (no-op at eval)
  3. Linear(num_patches*decoder_d_model → forecast_length)
  4. (permute to [B, forecast_length, C] is done externally by the full model)

Input shape:  [B, C, num_patches, decoder_d_model] = [1, 1, 8, 128]
Output shape: [B, forecast_length, C]              = [1, 96, 1]

Note: the PyTorch head.forward requires ``past_values`` as a positional argument;
the TTNN head ignores it.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.granite_ttm_r1.common import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MODEL_NAME,
    load_granite_ttm_config,
    load_granite_ttm_reference_model,
)
from models.demos.granite_ttm_r1.reference.eval import pcc
from models.demos.granite_ttm_r1.tt.common import preprocess_parameters, to_torch_tensor, to_ttnn_tensor
from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_head import TtnnGraniteTTMHead

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize("batch_size", [1])
def test_pcc_head(device, batch_size):
    hf_config = load_granite_ttm_config(DEFAULT_MODEL_NAME)
    model_config = GraniteTTMModelConfig.from_hf_config(hf_config, num_channels=1)
    hf_model = load_granite_ttm_reference_model(DEFAULT_MODEL_NAME, dtype=torch.float32)

    # Parameters
    parameters = preprocess_parameters(hf_model, device, model_name=DEFAULT_MODEL_NAME)
    head_params = parameters.head

    # Input: [B, C, num_patches, decoder_d_model]
    x = torch.randn(
        batch_size,
        model_config.num_channels,
        model_config.num_patches,
        model_config.decoder_d_model,
        dtype=torch.float32,
    )
    # past_values required by head.forward signature but not used for inference output.
    past_values = torch.randn(batch_size, DEFAULT_CONTEXT_LENGTH, model_config.num_channels, dtype=torch.float32)

    # PyTorch reference
    torch_module = hf_model.head
    with torch.no_grad():
        torch_output = torch_module(x, past_values=past_values)  # [B, forecast_len, C]

    # TTNN path
    ttnn_input = to_ttnn_tensor(x, device=device)
    ttnn_module = TtnnGraniteTTMHead(parameters=head_params, config=model_config)
    ttnn_output = ttnn_module(ttnn_input, device=device)  # [B, forecast_len, C]

    result = float(pcc(to_torch_tensor(ttnn_output).float(), torch_output.float()))
    assert result >= PCC_THRESHOLD, f"Head PCC {result:.4f} < {PCC_THRESHOLD}"
