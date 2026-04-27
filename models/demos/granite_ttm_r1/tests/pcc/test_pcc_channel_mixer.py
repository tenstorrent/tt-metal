# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the channel (feature) mixer (FeatureMixerBlock) component.

We exercise the feature_mixer from the first decoder block.
Input/output shape: [B, C, num_patches, decoder_d_model] = [1, 1, 8, 128].

The FeatureMixerBlock applies:
  1. LayerNorm on last dim (d_model)
  2. fc1 → GELU → fc2  (mixes across d_model dimension directly)
  3. GatedAttention: attn_linear → softmax → element-wise multiply
  4. Residual add
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
from models.demos.granite_ttm_r1.tt.ttnn_granite_ttm_channel_mixer import TtnnGraniteTTMChannelMixer

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize("batch_size", [1])
def test_pcc_channel_mixer(device, batch_size):
    hf_config = load_granite_ttm_config(DEFAULT_MODEL_NAME)
    model_config = GraniteTTMModelConfig.from_hf_config(hf_config, num_channels=1)
    hf_model = load_granite_ttm_reference_model(DEFAULT_MODEL_NAME, dtype=torch.float32)

    # Preprocess all parameters; extract the decoder block's feature_mixer subtree.
    parameters = preprocess_parameters(hf_model, device, model_name=DEFAULT_MODEL_NAME)
    feature_mixer_params = parameters.decoder.decoder_block.mixers[0].feature_mixer

    # Input: [B, C, num_patches, decoder_d_model]
    x = torch.randn(
        batch_size,
        model_config.num_channels,
        model_config.num_patches,
        model_config.decoder_d_model,
        dtype=torch.float32,
    )

    # PyTorch reference
    torch_module = hf_model.decoder.decoder_block.mixers[0].feature_mixer
    with torch.no_grad():
        torch_output = torch_module(x)

    # TTNN path
    ttnn_input = to_ttnn_tensor(x, device=device)
    ttnn_module = TtnnGraniteTTMChannelMixer(parameters=feature_mixer_params, config=model_config)
    ttnn_output = ttnn_module(ttnn_input, device=device)

    result = float(pcc(to_torch_tensor(ttnn_output).float(), torch_output.float()))
    assert result >= PCC_THRESHOLD, f"ChannelMixer PCC {result:.4f} < {PCC_THRESHOLD}"
