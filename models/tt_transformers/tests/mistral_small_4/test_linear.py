# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.linear import linear_bf16_no_bias, linear_bf16_no_bias_reference_torch


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_mistral_small_4_linear_matches_torch_and_hf(mesh_device, reset_seeds):
    """Bias-free linear: ttnn vs torch ``F.linear`` / ``nn.Linear`` (Mistral4-style)."""
    torch.manual_seed(0)
    in_f, out_f = 128, 256
    b, seq = 2, 8

    hf_linear = nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16)
    x = torch.randn(b, seq, in_f, dtype=torch.bfloat16)

    with torch.no_grad():
        expected_hf = hf_linear(x)
        expected_ref = linear_bf16_no_bias_reference_torch(x, hf_linear.weight)

    assert torch.allclose(expected_hf, expected_ref)

    out = linear_bf16_no_bias(mesh_device, x, hf_linear.weight.data)

    ok, msg = comp_pcc(expected_ref, out, pcc=0.99)
    assert ok, msg
    close, amsg = comp_allclose(expected_ref, out, rtol=0.08, atol=0.08)
    assert close, amsg
