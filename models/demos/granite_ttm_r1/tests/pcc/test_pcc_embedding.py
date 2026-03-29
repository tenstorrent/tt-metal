# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the patch embedding (backbone.encoder.patcher) component.

Linear projection from patch_length (64) to d_model (192).
Input shape:  [B, C, num_patches, patch_length] = [1, 1, 8, 64]
Output shape: [B, C, num_patches, d_model]       = [1, 1, 8, 192]
"""

from __future__ import annotations

import pytest
import torch

from models.demos.granite_ttm_r1.common import (
    DEFAULT_MODEL_NAME,
    load_granite_ttm_config,
    load_granite_ttm_reference_model,
)
from models.demos.granite_ttm_r1.reference.eval import pcc
from models.demos.granite_ttm_r1.tt.common import preprocess_parameters, to_torch_tensor, to_ttnn_tensor
from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_embedding import TtnnGraniteTTMEmbedding

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize("batch_size", [1])
def test_pcc_embedding(device, batch_size):
    hf_config = load_granite_ttm_config(DEFAULT_MODEL_NAME)
    model_config = GraniteTTMModelConfig.from_hf_config(hf_config, num_channels=1)
    hf_model = load_granite_ttm_reference_model(DEFAULT_MODEL_NAME, torch_dtype=torch.float32)

    # Parameters for backbone.encoder.patcher
    parameters = preprocess_parameters(hf_model, device)

    # Input: [B, C, num_patches, patch_length]
    x = torch.randn(
        batch_size,
        model_config.num_channels,
        model_config.num_patches,
        model_config.patch_length,
        dtype=torch.float32,
    )

    # PyTorch reference
    torch_module = hf_model.backbone.encoder.patcher
    with torch.no_grad():
        torch_output = torch_module(x)

    # TTNN path
    ttnn_input = to_ttnn_tensor(x, device=device)
    ttnn_module = TtnnGraniteTTMEmbedding(
        parameters=parameters.backbone.encoder.patcher,
        config=model_config,
    )
    ttnn_output = ttnn_module(ttnn_input, device=device)

    result = float(pcc(to_torch_tensor(ttnn_output).float(), torch_output.float()))
    assert result >= PCC_THRESHOLD, f"Embedding PCC {result:.4f} < {PCC_THRESHOLD}"
