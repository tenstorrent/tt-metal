# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for the full TtnnGraniteTTMModel.

Runs a complete forward pass on synthetic input and compares against the
PyTorch reference.  The encoder block uses an adaptive patching architecture
(TorchModuleFallback), so the only TTNN-introduced precision loss comes from
the patcher linear, decoder adapter linear, and head linear.  Expected PCC
is >= 0.99.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.granite_ttm_r1.common import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_FORECAST_LENGTH,
    DEFAULT_MODEL_NAME,
    create_synthetic_example,
    infer_num_channels,
    load_granite_ttm_config,
    load_granite_ttm_reference_model,
)
from models.demos.granite_ttm_r1.reference.eval import pcc
from models.demos.granite_ttm_r1.reference.model import extract_prediction_tensor
from models.demos.granite_ttm_r1.reference.preprocess import build_reference_inputs
from models.demos.granite_ttm_r1.tt.common import preprocess_inputs, preprocess_parameters, to_torch_tensor
from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_model import TtnnGraniteTTMModel

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize("batch_size", [1])
def test_pcc_full_model(device, batch_size):
    hf_config = load_granite_ttm_config(DEFAULT_MODEL_NAME)
    num_channels = infer_num_channels(hf_config)
    model_config = GraniteTTMModelConfig.from_hf_config(hf_config, num_channels=num_channels)

    hf_model = load_granite_ttm_reference_model(DEFAULT_MODEL_NAME, torch_dtype=torch.float32)

    example = create_synthetic_example(
        batch_size=batch_size,
        context_length=DEFAULT_CONTEXT_LENGTH,
        forecast_length=DEFAULT_FORECAST_LENGTH,
        num_channels=num_channels,
    )

    # PyTorch reference (float32)
    ref_inputs = build_reference_inputs(
        hf_model,
        example.history,
        future_values=example.future_values,
        observed_mask=example.observed_mask,
    )
    with torch.no_grad():
        ref_outputs = hf_model(**ref_inputs)
    torch_prediction = extract_prediction_tensor(ref_outputs)

    # TTNN path
    parameters = preprocess_parameters(hf_model, device)
    ttnn_model = TtnnGraniteTTMModel(
        parameters=parameters,
        config=model_config,
        reference_model=hf_model,
    )

    ttnn_history, ttnn_mask = preprocess_inputs(example.history, example.observed_mask, device=device)
    ttnn_prediction = ttnn_model(ttnn_history, observed_mask=ttnn_mask, device=device)

    ttnn_torch = to_torch_tensor(ttnn_prediction).float()
    result = float(pcc(ttnn_torch, torch_prediction.float()))
    assert result >= PCC_THRESHOLD, f"Full model PCC {result:.4f} < {PCC_THRESHOLD}"
