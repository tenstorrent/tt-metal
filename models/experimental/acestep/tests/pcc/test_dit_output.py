# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 DiT output head (norm_out AdaLN + proj_out de-patchify) vs custom DiTOutput.

Reference = tail of genuine HF AceStepDiTModel.forward: 2-value AdaLN on norm_out then
ConvTranspose1d de-patchify. batch=1 for bring-up; threshold 0.99.
"""

import pytest
import torch

import ttnn

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.dit_output import DiTOutput, DiTOutputConfig
from models.experimental.acestep.tests.test_utils import (
    HIDDEN_SIZE,
    RMS_NORM_EPS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

OUT_CHANNELS = 64
PATCH = 2
TPRIME_LENS = [128, 256]  # patched sequence length T'


@pytest.mark.parametrize("tprime", TPRIME_LENS, ids=[f"Tp{t}" for t in TPRIME_LENS])
def test_dit_output_vs_hf(device, tprime):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    dit = m.AceStepDiTModel(cfg).eval()
    convT = dit.proj_out[1]  # ConvTranspose1d [in, out, k]
    with torch.no_grad():
        dit.norm_out.weight.copy_(1.0 + 0.02 * torch.randn_like(dit.norm_out.weight))
        dit.scale_shift_table.copy_(0.05 * torch.randn_like(dit.scale_shift_table))
        convT.weight.copy_(0.02 * torch.randn_like(convT.weight))
        convT.bias.copy_(0.01 * torch.randn_like(convT.bias))

    x = torch.randn(1, tprime, HIDDEN_SIZE, dtype=torch.float32)
    temb = torch.randn(1, HIDDEN_SIZE, dtype=torch.float32)  # [B, dim]

    # Reference tail: AdaLN + proj_out.
    with torch.no_grad():
        shift, scale = (dit.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        h = (dit.norm_out(x) * (1 + scale) + shift).type_as(x)
        ref = dit.proj_out(h)  # [1, tprime*p, out]

    # Fold ConvTranspose1d [in,out,k] -> Linear [in, out*k] in (k,out) order.
    inp, outp, k = convT.weight.shape
    w_lin = convT.weight.permute(2, 1, 0).reshape(k * outp, inp).transpose(0, 1).contiguous()  # [in, k*out]

    tt = DiTOutput(
        DiTOutputConfig(
            scale_shift_table=make_lazy_weight(
                dit.scale_shift_table.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            norm_out_weight=make_lazy_weight(
                dit.norm_out.weight.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            proj_out_weight=make_lazy_weight(w_lin, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            proj_out_bias=make_lazy_weight(
                convT.bias.detach().clone().reshape(1, -1), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            dim=HIDDEN_SIZE,
            out_channels=OUT_CHANNELS,
            patch_size=PATCH,
            eps=RMS_NORM_EPS,
        )
    )

    x_tt = to_ttnn_tensor(x.reshape(1, 1, tprime, HIDDEN_SIZE), device)
    temb_tt = ttnn.from_torch(
        temb.reshape(1, 1, 1, HIDDEN_SIZE), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    out_tt = tt.forward(x_tt, temb_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, tprime * PATCH, OUT_CHANNELS)).reshape(1, tprime * PATCH, OUT_CHANNELS)

    assert_pcc(ref, out, 0.99)
