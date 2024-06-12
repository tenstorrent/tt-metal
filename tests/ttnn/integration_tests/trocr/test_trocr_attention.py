# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from transformers import VisionEncoderDecoderModel

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_trocr.tt.ttnn_trocr_attention import trocr_attention
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_trocr_attention(device, reset_seeds):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model = model.decoder.model.decoder.layers[0].self_attn

        embed_dim = 1024
        num_heads = 16
        input = torch.rand(1, 3, 1024)

        torch_output = model(input)[0]

        ttnn_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
        )

        ttnn_output = trocr_attention(
            hidden_states=ttnn_input,
            embed_dim=embed_dim,
            num_heads=num_heads,
            parameters=parameters,
            device=device,
        )

        ttnn_output = ttnn.to_torch(ttnn_output[0])

        assert_with_pcc(torch_output, ttnn_output, 0.99)
