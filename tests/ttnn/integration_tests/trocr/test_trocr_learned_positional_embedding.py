# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from transformers import VisionEncoderDecoderModel

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_trocr.tt.ttnn_trocr_learned_positional_embedding import (
    TtTrOCRPositionalEmbedding,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_trocr_learned_positional_embeddings(device, reset_seeds):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        model = model.decoder.model.decoder.embed_positions

        input = torch.rand(1, 3).long()

        torch_output = model(input)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
        )
        ttnn_output = TtTrOCRPositionalEmbedding(
            input_ids=input,
            device=device,
            parameters=parameters,
        )
        ttnn_output = ttnn.to_torch(ttnn_output)
        assert_with_pcc(torch_output, ttnn_output, 0.99)
