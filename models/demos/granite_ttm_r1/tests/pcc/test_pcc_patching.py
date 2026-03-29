# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the patching (TinyTimeMixerPatchify) component.

The patching layer uses torch.Tensor.unfold which has no direct TTNN
equivalent, so it is executed via TorchModuleFallback.  The only precision
loss comes from the bfloat16 round-trip, so the expected PCC is effectively
1.0 (we assert >= 0.99 for consistency with the rest of the suite).
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
from models.demos.granite_ttm_r1.tt.common import to_torch_tensor, to_ttnn_tensor
from models.demos.granite_ttm_r1.tt.config import GraniteTTMModelConfig
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_patching import TtnnGraniteTTMPatching

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize("batch_size", [1])
def test_pcc_patching(device, batch_size):
    hf_config = load_granite_ttm_config(DEFAULT_MODEL_NAME)
    model_config = GraniteTTMModelConfig.from_hf_config(hf_config, num_channels=1)
    hf_model = load_granite_ttm_reference_model(DEFAULT_MODEL_NAME, torch_dtype=torch.float32)

    # Input: [B, context_length, num_channels]
    x = torch.randn(batch_size, DEFAULT_CONTEXT_LENGTH, model_config.num_channels, dtype=torch.float32)

    # PyTorch reference
    torch_module = hf_model.backbone.patching
    with torch.no_grad():
        torch_output = torch_module(x)
    # torch_output shape: [B, C, num_patches, patch_length]

    # TTNN path (TorchModuleFallback)
    ttnn_input = to_ttnn_tensor(x, device=device)
    ttnn_module = TtnnGraniteTTMPatching(torch_module=torch_module)
    ttnn_output = ttnn_module(ttnn_input, device=device)

    result = float(pcc(to_torch_tensor(ttnn_output).float(), torch_output.float()))
    assert result >= PCC_THRESHOLD, f"Patching PCC {result:.4f} < {PCC_THRESHOLD}"
