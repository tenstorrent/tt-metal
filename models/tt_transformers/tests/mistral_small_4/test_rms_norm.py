# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.rms_norm import rms_norm_bf16


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_mistral_small_4_rms_norm_matches_hf(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    torch.manual_seed(0)
    hidden_size = 128
    eps = 1e-6
    b, s = 1, 16

    ref = Mistral4RMSNorm(hidden_size, eps=eps).to(torch.bfloat16).eval()
    x = torch.randn(b, s, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        expected = ref(x)

    out = rms_norm_bf16(mesh_device, x, ref.weight.data, epsilon=eps)

    ok, msg = comp_pcc(expected, out, pcc=0.999)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.05, atol=0.05)
    assert close, amsg
