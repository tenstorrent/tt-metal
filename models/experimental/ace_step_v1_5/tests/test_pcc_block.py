# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC for ``TtAceStepDiTLayer`` / ``TtQwen3MLP`` — full transformer block on the demo DiT path."""

from __future__ import annotations

import torch

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print, tiny_dit_decoder_fixture
from models.experimental.ace_step_v1_5.torch_ref.dit_decoder_core import (
    TorchAceStepDiTCoreRef,
    TorchAceStepDiTLayerRef,
    qwen3_mlp,
)
from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import (
    TtAceStepDiTLayer,
    TtHfRotaryEmbedding,
    TtQwen3MLP,
)


def test_qwen3_mlp_matches_torch(mesh_device):
    """Gated MLP inside each ``TtAceStepDiTLayer``."""
    import ttnn

    cfg, sd, d_model, seq_len, _enc_len = tiny_dit_decoder_fixture()
    layer_idx = 0
    intermediate = int(sd[f"layers.{layer_idx}.mlp.gate_proj.weight"].shape[0])

    torch.manual_seed(6)
    x = torch.randn(1, 1, seq_len, d_model, dtype=torch.bfloat16)
    device = x.device
    dtype = torch.bfloat16
    core = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)
    y_ref = qwen3_mlp(
        x,
        w_gate=core._w(f"layers.{layer_idx}.mlp.gate_proj.weight", device=device, dtype=dtype),
        w_up=core._w(f"layers.{layer_idx}.mlp.up_proj.weight", device=device, dtype=dtype),
        w_down=core._w(f"layers.{layer_idx}.mlp.down_proj.weight", device=device, dtype=dtype),
    )

    mlp = TtQwen3MLP(
        state_dict=sd,
        base_address=f"layers.{layer_idx}.mlp",
        mesh_device=mesh_device,
        hidden_size=d_model,
        intermediate_size=intermediate,
        dtype=ttnn.bfloat16,
    )
    x_tt = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    y_tt = mlp(x_tt)
    y = ttnn.to_torch(y_tt).to(torch.bfloat16)
    assert_pcc_print("dit_qwen3_mlp", y_ref, y)


def test_dit_decoder_layer_matches_torch(mesh_device):
    """One ``layers.{i}`` block: self-attn + cross-attn + modulated MLP (``TtAceStepDiTLayer``)."""
    import ttnn

    cfg, sd, d_model, seq_len, enc_len = tiny_dit_decoder_fixture()
    layer_idx = 0
    core = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)
    layer_ref = TorchAceStepDiTLayerRef(cfg=cfg, state_dict=sd, layer_idx=layer_idx)

    torch.manual_seed(7)
    x = torch.randn(1, 1, seq_len, d_model, dtype=torch.bfloat16)
    timestep_proj = torch.randn(1, 6, d_model, dtype=torch.bfloat16)
    enc_raw = torch.randn(1, enc_len, 32, dtype=torch.bfloat16)
    enc = core.condition_encoder_hidden_states(enc_raw)

    y_ref = layer_ref(x, timestep_proj, enc)

    rotary = TtHfRotaryEmbedding(
        mesh_device=mesh_device,
        head_dim=int(cfg.head_dim),
        max_seq_len=int(cfg.max_position_embeddings),
        rope_theta=float(cfg.rope_theta),
        hidden_size=int(cfg.hidden_size),
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        dtype=ttnn.bfloat16,
    )
    tt_layer = TtAceStepDiTLayer(
        cfg=cfg,
        state_dict=sd,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
        rotary_embedding=rotary,
    )
    x_tt = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tp_tt = ttnn.from_torch(timestep_proj, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    enc_tt = ttnn.from_torch(enc, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    y_tt = tt_layer(x_tt, tp_tt, enc_tt)
    y = ttnn.to_torch(y_tt).to(torch.bfloat16)
    assert_pcc_print("dit_decoder_layer", y_ref, y)
