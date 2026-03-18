# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.synthesizer.attentions import MultiHeadAttention as TorchMultiHeadAttention
from models.demos.rvc.tt_impl.synthesizer.attentions import MultiHeadAttention as TTMultiHeadAttention
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("window_size", [None, 4])
def test_multiheadattention(device, window_size):
    torch.manual_seed(0)

    batch_size = 1
    in_features = 128
    out_features = 16
    num_heads = 4
    input_length = 64
    torch_mha = TorchMultiHeadAttention(
        in_features=in_features,
        out_features=out_features,
        num_heads=num_heads,
        window_size=window_size,
    ).eval()

    torch_input = torch.randn(batch_size, in_features, input_length, dtype=torch.float32)
    torch_context = (
        torch_input.clone()
        if window_size is not None
        else torch.randn(batch_size, in_features, input_length, dtype=torch.float32)
    )
    torch_output = torch_mha(torch_input, torch_context)

    tt_mha = TTMultiHeadAttention(
        device=device,
        in_features=in_features,
        out_features=out_features,
        num_heads=num_heads,
        window_size=window_size,
    )

    state_dict = {
        "encoder.attn.linear_q.weight": torch_mha.linear_q.weight,
        "encoder.attn.linear_q.bias": torch_mha.linear_q.bias,
        "encoder.attn.linear_k.weight": torch_mha.linear_k.weight,
        "encoder.attn.linear_k.bias": torch_mha.linear_k.bias,
        "encoder.attn.linear_v.weight": torch_mha.linear_v.weight,
        "encoder.attn.linear_v.bias": torch_mha.linear_v.bias,
        "encoder.attn.linear_o.weight": torch_mha.linear_o.weight,
        "encoder.attn.linear_o.bias": torch_mha.linear_o.bias,
    }
    if window_size is not None:
        state_dict["encoder.attn.emb_rel_k"] = torch_mha.emb_rel_k
        state_dict["encoder.attn.emb_rel_v"] = torch_mha.emb_rel_v
    tt_mha.load_state_dict(state_dict=state_dict, module_prefix="encoder.attn.")

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16).permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_context = ttnn.from_torch(
        torch_context.to(torch.bfloat16).permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_output = tt_mha(tt_input, tt_context)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)
    tt_output_torch = tt_output_torch.reshape(batch_size, -1, out_features)
    tt_output_torch = tt_output_torch.permute(0, 2, 1)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
