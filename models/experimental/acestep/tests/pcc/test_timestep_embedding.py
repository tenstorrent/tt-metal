# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 TimestepEmbedding vs custom TT module.

Reference = genuine HF TimestepEmbedding (in_channels=256, time_embed_dim=hidden_size).
Outputs temb [B, dim] and timestep_proj [B, 6, dim] used for AdaLN modulation.
"""

import pytest
import torch

import ttnn

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.tt.timestep_embedding import (
    TimestepEmbedding,
    TimestepEmbeddingConfig,
)
from models.experimental.acestep.tests.test_utils import (
    HIDDEN_SIZE,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
)

IN_CHANNELS = 256


def _w_T(weight, device):
    return make_lazy_weight(
        weight.detach().clone().transpose(-1, -2).contiguous(),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _bias(bias, device):
    return make_lazy_weight(
        bias.detach().clone().reshape(1, -1).contiguous(),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


@pytest.mark.parametrize("batch", [1, 2, 8], ids=["B1", "B2", "B8"])
def test_timestep_embedding_vs_hf(device, batch):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    load_config()
    ref = m.TimestepEmbedding(in_channels=IN_CHANNELS, time_embed_dim=HIDDEN_SIZE).eval()
    with torch.no_grad():
        for lin in (ref.linear_1, ref.linear_2, ref.time_proj):
            lin.weight.copy_(0.02 * torch.randn_like(lin.weight))
            if lin.bias is not None:
                lin.bias.copy_(0.01 * torch.randn_like(lin.bias))

    t = torch.rand(batch, dtype=torch.float32)

    with torch.no_grad():
        ref_temb, ref_proj = ref(t)  # [B,dim], [B,6,dim]

    tt = TimestepEmbedding(
        TimestepEmbeddingConfig(
            linear_1_weight=_w_T(ref.linear_1.weight, device),
            linear_1_bias=_bias(ref.linear_1.bias, device),
            linear_2_weight=_w_T(ref.linear_2.weight, device),
            linear_2_bias=_bias(ref.linear_2.bias, device),
            time_proj_weight=_w_T(ref.time_proj.weight, device),
            time_proj_bias=_bias(ref.time_proj.bias, device),
            in_channels=IN_CHANNELS,
            time_embed_dim=HIDDEN_SIZE,
        )
    )

    tt_temb, tt_proj = tt.forward(t)
    tt_temb = to_torch(tt_temb, expected_shape=(1, 1, batch, HIDDEN_SIZE)).reshape(batch, HIDDEN_SIZE)
    tt_proj = (
        to_torch(tt_proj, expected_shape=(1, 6, batch, HIDDEN_SIZE)).permute(0, 2, 1, 3).reshape(batch, 6, HIDDEN_SIZE)
    )

    assert_pcc(ref_temb, tt_temb, 0.999)
    assert_pcc(ref_proj, tt_proj, 0.999)
