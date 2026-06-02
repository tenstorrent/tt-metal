# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1 PCC: TtConformerConvModule vs HF SeamlessM4Tv2ConformerConvolutionModule.

Uses a randomly-initialized reference module so the test is self-contained (no
9GB checkpoint required). Validates the on-device LN/GLU/SiLU/matmul path plus the
causal depthwise-conv host fallback at several real audio sequence lengths.
"""

import pytest
import torch
from transformers import SeamlessM4Tv2Config
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import (
    SeamlessM4Tv2ConformerConvolutionModule,
)

import ttnn
from models.demos.seamless_m4t_v2.tt.conformer_conv import TtConformerConvModule
from models.demos.seamless_m4t_v2.tt.model_config import SeamlessS2TTConfig
from tests.ttnn.utils_for_testing import assert_with_pcc


def _randomized_reference(hf_config):
    ref = SeamlessM4Tv2ConformerConvolutionModule(hf_config).eval()
    with torch.no_grad():
        # give the LayerNorms non-trivial affine params so the test exercises them
        for name, p in ref.named_parameters():
            if name.endswith("layer_norm.weight"):
                p.copy_(1.0 + 0.05 * torch.randn_like(p))
            elif name.endswith("layer_norm.bias"):
                p.copy_(0.05 * torch.randn_like(p))
    return ref


# ttnn.conv1d needs an L1_SMALL scratch region reserved at device open.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("conv_cpu_fallback", [True, False], ids=["host", "device"])
@pytest.mark.parametrize("seq_len", [128, 512, 1500])
def test_conformer_conv_pcc(device, seq_len, conv_cpu_fallback):
    torch.manual_seed(0)
    hf_config = SeamlessM4Tv2Config()
    s2tt = SeamlessS2TTConfig(conv_cpu_fallback=conv_cpu_fallback)
    C = s2tt.hidden_size

    ref = _randomized_reference(hf_config)
    x = torch.randn(1, seq_len, C)

    with torch.no_grad():
        golden = ref(x, attention_mask=None)  # (1, T, C)

    tt_mod = TtConformerConvModule(ref.state_dict(), s2tt, device, dtype=ttnn.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = tt_mod(x_tt)
    out = ttnn.to_torch(out_tt)

    assert out.shape == golden.shape
    assert_with_pcc(golden, out, pcc=0.99)
