# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from transformers import VisionEncoderDecoderModel

from ttnn.model_preprocessing import preprocess_model_parameters


from models.experimental.functional_trocr.ttnn_trocr_learned_positional_embedding import (
    trocr_learned_positional_embeddings,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_trocr_learned_positional_embeddings(device, reset_seeds):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        config = model.decoder.config

        model = model.decoder.model.decoder

        num_heads = 512
        embed_dim = 1024

        input = torch.rand(1, 3).long()

        torch_output = model(input)[0]

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device,
            convert_to_ttnn=lambda *_: False,
        )

        ttnn_output = trocr_learned_positional_embeddings(
            input_ids=input,
            num_embeddings=num_heads,
            embedding_dim=embed_dim,
            device=device,
            parameters=parameters,
        )
        assert_with_pcc(torch_output, ttnn_output, 1)
