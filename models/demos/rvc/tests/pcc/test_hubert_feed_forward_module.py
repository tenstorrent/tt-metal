# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.vc.hubert import FeedForwardModule as TorchFeedForwardModule
from models.demos.rvc.tt_impl.vc.hubert import FeedForwardModule as TTFeedForwardModule
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_hubert_feed_forward_module(device):
    torch.manual_seed(0)

    input_feat = 64
    hidden_units = 128
    activation_fn = "swish"
    t = 24
    b = 2

    torch_ffn = TorchFeedForwardModule(
        input_feat=input_feat,
        hidden_units=hidden_units,
        activation_fn=activation_fn,
        bias=True,
    ).eval()
    tt_ffn = TTFeedForwardModule(
        device=device,
        input_feat=input_feat,
        hidden_units=hidden_units,
        activation_fn=activation_fn,
        bias=True,
    )

    parameters = {f"ffn.{k}": v for k, v in torch_ffn.state_dict().items()}
    tt_ffn.load_parameters(parameters=parameters, prefix="ffn.")

    torch_x = torch.randn(t, b, input_feat, dtype=torch.float32)
    torch_output = torch_ffn(torch_x)

    tt_x = ttnn.from_torch(
        torch_x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_output = tt_ffn(tt_x)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
