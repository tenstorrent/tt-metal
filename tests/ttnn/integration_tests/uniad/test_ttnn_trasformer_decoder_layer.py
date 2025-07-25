# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import ttnn
import pytest

from models.experimental.uniad.tt.ttnn_transformer_decoder_layer import TtTransformerDecoderLayer

from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_transformer_decoder_layer(device, reset_seeds):
    torch_model = nn.TransformerDecoderLayer(
        d_model=256,
        nhead=8,
        dropout=0.1,
        dim_feedforward=512,
        batch_first=True,
    )

    torch_model.eval()

    print("torch_model", torch_model)

    query = torch.randn(1, 6, 256)
    mem = torch.randn(1, 300, 256)

    torch_output = torch_model(query, mem)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
    )

    print("parameters", parameters)

    ttnn_model = TtTransformerDecoderLayer(
        parameters=parameters,
        device=device,
        d_model=256,
        nhead=8,
        dropout=0.1,
        dim_feedforward=512,
        batch_first=True,
    )

    ttnn_query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_mem = ttnn.from_torch(mem, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ttnn_output = ttnn_model(ttnn_query, ttnn_mem)
