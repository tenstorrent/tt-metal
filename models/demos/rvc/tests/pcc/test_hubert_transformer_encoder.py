# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.vc.hubert import TransformerEncoder as TorchTransformerEncoder
from models.demos.rvc.tt_impl.vc.hubert import TransformerEncoder as TTTransformerEncoder
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("layer_norm_first", [False, True])
def test_hubert_transformer_encoder(device, layer_norm_first):
    torch.manual_seed(0)

    args = {
        "encoder_embed_dim": 64,
        "required_seq_len_multiple": 2,
        "conv_pos": 8,
        "conv_pos_groups": 8,
        "encoder_layers": 2,
        "encoder_ffn_embed_dim": 128,
        "encoder_attention_heads": 4,
        "activation_fn": "relu",
        "layer_norm_first": layer_norm_first,
        "layer_type": "transformer",
    }
    batch_size = 2
    seq_len = 23
    tgt_layer = 1

    torch_encoder = TorchTransformerEncoder(args).eval()
    tt_encoder = TTTransformerEncoder(device=device, args=args)

    parameters = {f"encoder.{k}": v for k, v in torch_encoder.state_dict().items()}
    tt_encoder.load_parameters(parameters=parameters, prefix="encoder.")

    torch_x = torch.randn(batch_size, seq_len, args["encoder_embed_dim"], dtype=torch.float32)
    torch_output = torch_encoder(torch_x, tgt_layer=tgt_layer)

    tt_x = ttnn.from_torch(
        torch_x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_output = tt_encoder(tt_x, tgt_layer=tgt_layer)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.95)
