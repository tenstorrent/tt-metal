# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC for ACE DiT scale-shift AdaLN modulation (``TtAceStepDiTLayer`` / demo DiT path)."""

from __future__ import annotations

import pytest
import torch

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import (
    assert_pcc_print,
    modulation_chunks_torch,
    tiny_dit_decoder_fixture,
)
from models.experimental.ace_step_v1_5.torch_ref.dit_decoder_core import adaln_modulate_torch


@pytest.mark.parametrize("which", ["self_attn", "mlp"])
def test_dit_scale_shift_adaln_modulation_matches_torch(mesh_device, which: str):
    """RMSNorm(x) * (1 + scale) + shift — same ops as ``TtAceStepDiTLayer`` self-attn and MLP branches."""
    import ttnn

    cfg, sd, d_model, seq_len, _enc_len = tiny_dit_decoder_fixture()
    layer_idx = 0
    b = 1

    torch.manual_seed(3)
    x = torch.randn(b, 1, seq_len, d_model, dtype=torch.bfloat16)
    timestep_proj = torch.randn(b, 6, d_model, dtype=torch.bfloat16)
    shift_msa, scale_msa, _gate_msa, c_shift, c_scale, _c_gate = modulation_chunks_torch(
        sd, layer_idx=layer_idx, timestep_proj_b6d=timestep_proj
    )

    if which == "self_attn":
        norm_w = torch.from_numpy(sd[f"layers.{layer_idx}.self_attn_norm.weight"]).to(torch.bfloat16)
        shift, scale = shift_msa, scale_msa
    else:
        norm_w = torch.from_numpy(sd[f"layers.{layer_idx}.mlp_norm.weight"]).to(torch.bfloat16)
        shift, scale = c_shift, c_scale

    y_ref = adaln_modulate_torch(x, norm_w, float(cfg.rms_norm_eps), shift, scale)

    x_tt = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    norm_w_tt = ttnn.from_torch(norm_w, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    shift_tt = ttnn.from_torch(shift, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    scale_tt = ttnn.from_torch(scale, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    x_norm = ttnn.rms_norm(
        x_tt,
        weight=norm_w_tt,
        epsilon=float(cfg.rms_norm_eps),
        memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
    )
    one_plus = ttnn.add(scale_tt, ttnn.ones_like(scale_tt))
    y_tt = ttnn.add(ttnn.multiply(x_norm, one_plus), shift_tt)
    y = ttnn.to_torch(y_tt).to(torch.bfloat16)

    assert_pcc_print(f"dit_adaln_{which}", y_ref, y)
