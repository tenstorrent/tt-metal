# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 DiT proj_in (patch embed) vs custom PatchEmbed.

Reference = genuine HF AceStepDiTModel.proj_in (Conv1d in=192,out=2048,k=stride=2).
We fold the Conv1d into a Linear via the patchify equivalence and validate PCC.
"""

import pytest
import torch

import ttnn

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.patch_embed import PatchEmbed, PatchEmbedConfig
from models.experimental.acestep.tests.test_utils import (
    HIDDEN_SIZE,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

IN_CHANNELS = 192
PATCH = 2
SEQ_LENS = [256, 512]


@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"T{t}" for t in SEQ_LENS])
def test_patch_embed_vs_hf(device, seq_len):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    dit = m.AceStepDiTModel(cfg).eval()
    conv = dit.proj_in[1]  # nn.Conv1d [out, in, k]
    with torch.no_grad():
        conv.weight.copy_(0.02 * torch.randn_like(conv.weight))
        conv.bias.copy_(0.01 * torch.randn_like(conv.bias))

    x = torch.randn(1, seq_len, IN_CHANNELS, dtype=torch.float32)

    with torch.no_grad():
        ref = dit.proj_in(x)  # [1, seq/p, hidden]

    # Fold Conv1d [out, in, k] -> Linear weight [in*k, out] in (in, k) channel-major order,
    # matching PatchEmbed's reshape [.., C, p] -> flatten.
    out_ch, in_ch, k = conv.weight.shape
    w_lin = conv.weight.reshape(out_ch, in_ch * k).transpose(0, 1).contiguous()  # [in*k, out]

    tt = PatchEmbed(
        PatchEmbedConfig(
            weight=make_lazy_weight(w_lin, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            bias=make_lazy_weight(
                conv.bias.detach().clone().reshape(1, -1), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            in_channels=IN_CHANNELS,
            out_channels=HIDDEN_SIZE,
            patch_size=PATCH,
        )
    )

    x_tt = to_ttnn_tensor(x.reshape(1, 1, seq_len, IN_CHANNELS), device)
    out_tt = tt.forward(x_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len // PATCH, HIDDEN_SIZE)).reshape(
        1, seq_len // PATCH, HIDDEN_SIZE
    )

    assert_pcc(ref, out, 0.999)
