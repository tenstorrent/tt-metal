# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC for ``TtAceStepAttentionSDPA`` (self + cross) — demo ``dit_decoder_core`` path."""

from __future__ import annotations

import pytest
import torch

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print, tiny_dit_decoder_fixture
from models.experimental.ace_step_v1_5.torch_ref.dit_decoder_core import TorchAceStepDiTCoreRef, _attention_sdpa
from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import TtAceStepAttentionSDPA, TtHfRotaryEmbedding


def _torch_self_attn_ref(
    core: TorchAceStepDiTCoreRef,
    *,
    layer_idx: int,
    x: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    device = x.device
    dtype = torch.bfloat16
    h = int(core.cfg.num_attention_heads)
    dh = int(core.cfg.head_dim)
    eps = float(core.cfg.rms_norm_eps)
    base = f"layers.{layer_idx}.self_attn"
    return _attention_sdpa(
        x,
        wq=core._w(f"{base}.q_proj.weight", device=device, dtype=dtype),
        wk=core._w(f"{base}.k_proj.weight", device=device, dtype=dtype),
        wv=core._w(f"{base}.v_proj.weight", device=device, dtype=dtype),
        wo=core._w(f"{base}.o_proj.weight", device=device, dtype=dtype),
        q_norm_w=core._w(f"{base}.q_norm.weight", device=device, dtype=dtype),
        k_norm_w=core._w(f"{base}.k_norm.weight", device=device, dtype=dtype),
        n_heads=h,
        n_kv_heads=int(core.cfg.num_key_value_heads),
        head_dim=dh,
        eps=eps,
        encoder_hidden_states=None,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )


def _torch_cross_attn_ref(
    core: TorchAceStepDiTCoreRef,
    *,
    layer_idx: int,
    x: torch.Tensor,
    enc: torch.Tensor,
) -> torch.Tensor:
    device = x.device
    dtype = torch.bfloat16
    h = int(core.cfg.num_attention_heads)
    dh = int(core.cfg.head_dim)
    eps = float(core.cfg.rms_norm_eps)
    base = f"layers.{layer_idx}.cross_attn"
    return _attention_sdpa(
        x,
        wq=core._w(f"{base}.q_proj.weight", device=device, dtype=dtype),
        wk=core._w(f"{base}.k_proj.weight", device=device, dtype=dtype),
        wv=core._w(f"{base}.v_proj.weight", device=device, dtype=dtype),
        wo=core._w(f"{base}.o_proj.weight", device=device, dtype=dtype),
        q_norm_w=core._w(f"{base}.q_norm.weight", device=device, dtype=dtype),
        k_norm_w=core._w(f"{base}.k_norm.weight", device=device, dtype=dtype),
        n_heads=h,
        n_kv_heads=int(core.cfg.num_key_value_heads),
        head_dim=dh,
        eps=eps,
        encoder_hidden_states=enc,
        rope_cos=None,
        rope_sin=None,
    )


@pytest.mark.parametrize("attn_kind", ["self", "cross"])
def test_dit_attention_sdpa_matches_torch(mesh_device, attn_kind: str):
    import ttnn

    cfg, sd, d_model, seq_len, enc_len = tiny_dit_decoder_fixture()
    layer_idx = 0
    core = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)

    torch.manual_seed(4)
    x = torch.randn(1, 1, seq_len, d_model, dtype=torch.bfloat16)
    cond_dim = 32
    enc_raw = torch.randn(1, enc_len, cond_dim, dtype=torch.bfloat16)
    enc = core.condition_encoder_hidden_states(enc_raw)

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

    if attn_kind == "self":
        rope = core._rope(
            d_model=d_model,
            n_heads=int(cfg.num_attention_heads),
            head_dim=int(cfg.head_dim),
            max_seq_len=max(seq_len, 512),
            device=x.device,
        )
        dummy = torch.zeros(1, seq_len, d_model, dtype=torch.float32, device=x.device)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        with torch.no_grad():
            rope_cos, rope_sin = rope(dummy, pos)
        y_ref = _torch_self_attn_ref(core, layer_idx=layer_idx, x=x, rope_cos=rope_cos, rope_sin=rope_sin)
        tt_attn = TtAceStepAttentionSDPA(
            cfg=cfg,
            state_dict=sd,
            base_address=f"layers.{layer_idx}.self_attn",
            mesh_device=mesh_device,
            dtype=ttnn.bfloat16,
            rotary_embedding=rotary,
        )
        x_tt = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        y_tt = tt_attn(x_tt, encoder_hidden_states=None, is_causal=False)
    else:
        y_ref = _torch_cross_attn_ref(core, layer_idx=layer_idx, x=x, enc=enc)
        tt_attn = TtAceStepAttentionSDPA(
            cfg=cfg,
            state_dict=sd,
            base_address=f"layers.{layer_idx}.cross_attn",
            mesh_device=mesh_device,
            dtype=ttnn.bfloat16,
            rotary_embedding=None,
        )
        x_tt = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        enc_tt = ttnn.from_torch(enc, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        y_tt = tt_attn(x_tt, encoder_hidden_states=enc_tt, is_causal=False)

    y = ttnn.to_torch(y_tt).to(torch.bfloat16)
    assert_pcc_print(f"dit_attention_{attn_kind}", y_ref, y)


def test_dit_self_attention_gqa_matches_torch(mesh_device):
    """GQA path used by turbo/base DiT when ``num_key_value_heads < num_attention_heads``."""
    import ttnn

    cfg, sd, d_model, seq_len, enc_len = tiny_dit_decoder_fixture(n_kv_heads=2)
    layer_idx = 0
    core = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)

    torch.manual_seed(5)
    x = torch.randn(1, 1, seq_len, d_model, dtype=torch.bfloat16)
    rope = core._rope(
        d_model=d_model,
        n_heads=int(cfg.num_attention_heads),
        head_dim=int(cfg.head_dim),
        max_seq_len=max(seq_len, 512),
        device=x.device,
    )
    dummy = torch.zeros(1, seq_len, d_model, dtype=torch.float32, device=x.device)
    pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
    with torch.no_grad():
        rope_cos, rope_sin = rope(dummy, pos)

    y_ref = _torch_self_attn_ref(core, layer_idx=layer_idx, x=x, rope_cos=rope_cos, rope_sin=rope_sin)
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
    tt_attn = TtAceStepAttentionSDPA(
        cfg=cfg,
        state_dict=sd,
        base_address=f"layers.{layer_idx}.self_attn",
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
        rotary_embedding=rotary,
    )
    x_tt = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    y_tt = tt_attn(x_tt, encoder_hidden_states=None, is_causal=False)
    y = ttnn.to_torch(y_tt).to(torch.bfloat16)
    assert_pcc_print("dit_attention_self_gqa", y_ref, y)
