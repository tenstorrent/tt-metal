# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.vc.hubert import MultiheadSelfAttention as TorchMultiheadSelfAttention
from models.demos.rvc.tt_impl.vc.hubert import MultiheadSelfAttention as TTMultiheadSelfAttention
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_hubert_multihead_attention(device):
    torch.manual_seed(0)

    embed_dim = 128
    num_heads = 4
    t = 24
    b = 2

    torch_attn = TorchMultiheadSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        self_attention=True,
    ).eval()
    tt_attn = TTMultiheadSelfAttention(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        self_attention=True,
    )

    parameters = {f"attn.{k}": v for k, v in torch_attn.state_dict().items()}
    tt_attn.load_parameters(parameters=parameters, prefix="attn.")

    torch_query = torch.randn(t, b, embed_dim, dtype=torch.float32)

    torch_output = torch_attn(torch_query)

    tt_query = ttnn.from_torch(
        torch_query.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_output = tt_attn(tt_query)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.97)
