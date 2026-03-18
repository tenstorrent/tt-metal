# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.vc.hubert import (
    TransformerSentenceEncoderLayer as TorchTransformerSentenceEncoderLayer,
)
from models.demos.rvc.tt_impl.vc.hubert import TransformerSentenceEncoderLayer as TTTransformerSentenceEncoderLayer
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("layer_norm_first", [False, True])
def test_hubert_transformer_sentence_encoder_layer(device, layer_norm_first):
    torch.manual_seed(0)

    embed_dim = 128
    ffn_embed_dim = 128
    attention_heads = 4
    activation_fn = "relu"
    t = 24
    b = 2

    torch_layer = TorchTransformerSentenceEncoderLayer(
        embed_dim=embed_dim,
        ffn_embed_dim=ffn_embed_dim,
        attention_heads=attention_heads,
        activation_fn=activation_fn,
        layer_norm_first=layer_norm_first,
    ).eval()
    tt_layer = TTTransformerSentenceEncoderLayer(
        device=device,
        embed_dim=embed_dim,
        ffn_embed_dim=ffn_embed_dim,
        attention_heads=attention_heads,
        activation_fn=activation_fn,
        layer_norm_first=layer_norm_first,
    )

    parameters = {f"layer.{k}": v for k, v in torch_layer.state_dict().items()}
    tt_layer.load_state_dict(parameters=parameters, module_prefix="layer.")

    torch_x = torch.randn(t, b, embed_dim, dtype=torch.float32)
    torch_output = torch_layer(torch_x)

    tt_x = ttnn.from_torch(
        torch_x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_output = tt_layer(tt_x)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.96)
