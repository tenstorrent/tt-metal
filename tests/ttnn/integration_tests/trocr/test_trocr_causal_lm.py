# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from transformers import VisionEncoderDecoderModel
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_trocr.ttnn_trocr_causal_lm import trocr_causal_lm
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_trocr_causal_lm(device, reset_seeds):
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    config = model.decoder.config

    model = model.decoder

    input = torch.rand(1, 3)

    torch_output = model(input.long()).logits

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    ttnn_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ttnn_output = trocr_causal_lm(
        input_ids=ttnn_input,
        config=config,
        device=device,
        parameters=parameters,
    )
    ttnn_output = ttnn.to_torch(ttnn_output[0])

    assert_with_pcc(torch_output, ttnn_output, 0.98)
