# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from transformers import VisionEncoderDecoderModel

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_trocr.tt.ttnn_trocr_decoder import trocr_decoder
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_trocr_decoder(device, reset_seeds):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model.eval()

        config = model.decoder.config

        model = model.decoder.model.decoder

        # run torch model
        input = torch.randint(2, 100, (1, 5), dtype=torch.int32)

        torch_output = model(input.long()).last_hidden_state
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
        )

        ttnn_input = ttnn.from_torch(input, device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)

        ttnn_output = trocr_decoder(
            input_ids=ttnn_input,
            device=device,
            parameters=parameters,
            config=config,
        )
        ttnn_output = ttnn.to_torch(ttnn_output[0])

        assert_with_pcc(torch_output, ttnn_output, 0.99)
