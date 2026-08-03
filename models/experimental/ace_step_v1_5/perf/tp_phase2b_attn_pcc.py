# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TP Phase 2b gate: ``TtAceStepAttentionSDPA`` tensor-parallel (Q/K/V column-parallel by head,
o_proj row-parallel + all-reduce) must match the replicate baseline AND the torch reference on
the real BH_QB 2×2 mesh, for both self- and cross-attention.

Run (device free):
    python models/experimental/ace_step_v1_5/perf/tp_phase2b_attn_pcc.py
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import tiny_dit_decoder_fixture
from models.experimental.ace_step_v1_5.torch_ref.dit_decoder_core import TorchAceStepDiTCoreRef, _attention_sdpa
from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import TtAceStepAttentionSDPA, TtHfRotaryEmbedding
from models.experimental.ace_step_v1_5.utils.tt_device import (
    ace_step_replicate_mesh_mapper,
    close_ace_step_device,
    open_dit_device,
)

_GATE = 0.99


def _self_ref(core, layer_idx, x, rope_cos, rope_sin):
    b = f"layers.{layer_idx}.self_attn"
    dt = torch.bfloat16
    return _attention_sdpa(
        x,
        wq=core._w(f"{b}.q_proj.weight", device=x.device, dtype=dt),
        wk=core._w(f"{b}.k_proj.weight", device=x.device, dtype=dt),
        wv=core._w(f"{b}.v_proj.weight", device=x.device, dtype=dt),
        wo=core._w(f"{b}.o_proj.weight", device=x.device, dtype=dt),
        q_norm_w=core._w(f"{b}.q_norm.weight", device=x.device, dtype=dt),
        k_norm_w=core._w(f"{b}.k_norm.weight", device=x.device, dtype=dt),
        n_heads=int(core.cfg.num_attention_heads),
        n_kv_heads=int(core.cfg.num_key_value_heads),
        head_dim=int(core.cfg.head_dim),
        eps=float(core.cfg.rms_norm_eps),
        encoder_hidden_states=None,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )


def _cross_ref(core, layer_idx, x, enc):
    b = f"layers.{layer_idx}.cross_attn"
    dt = torch.bfloat16
    return _attention_sdpa(
        x,
        wq=core._w(f"{b}.q_proj.weight", device=x.device, dtype=dt),
        wk=core._w(f"{b}.k_proj.weight", device=x.device, dtype=dt),
        wv=core._w(f"{b}.v_proj.weight", device=x.device, dtype=dt),
        wo=core._w(f"{b}.o_proj.weight", device=x.device, dtype=dt),
        q_norm_w=core._w(f"{b}.q_norm.weight", device=x.device, dtype=dt),
        k_norm_w=core._w(f"{b}.k_norm.weight", device=x.device, dtype=dt),
        n_heads=int(core.cfg.num_attention_heads),
        n_kv_heads=int(core.cfg.num_key_value_heads),
        head_dim=int(core.cfg.head_dim),
        eps=float(core.cfg.rms_norm_eps),
        encoder_hidden_states=enc,
        rope_cos=None,
        rope_sin=None,
    )


def _read0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).to(torch.float32)


def _run(mesh, cfg, sd, layer_idx, kind, x, enc, rotary):
    base = f"layers.{layer_idx}.{'self_attn' if kind == 'self' else 'cross_attn'}"
    attn = TtAceStepAttentionSDPA(
        cfg=cfg,
        state_dict=sd,
        base_address=base,
        mesh_device=mesh,
        dtype=ttnn.bfloat16,
        rotary_embedding=(rotary if kind == "self" else None),
    )
    rep = ace_step_replicate_mesh_mapper(mesh)
    x_tt = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
    if kind == "self":
        y = attn(x_tt, encoder_hidden_states=None, is_causal=False)
    else:
        enc_tt = ttnn.from_torch(enc, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
        y = attn(x_tt, encoder_hidden_states=enc_tt, is_causal=False)
    return _read0(y)


def main() -> int:
    cfg, sd, d_model, seq_len, enc_len = tiny_dit_decoder_fixture()
    layer_idx = 0
    core = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)
    torch.manual_seed(4)
    x = torch.randn(1, 1, seq_len, d_model, dtype=torch.bfloat16)
    enc = core.condition_encoder_hidden_states(torch.randn(1, enc_len, 32, dtype=torch.bfloat16))

    rope = core._rope(
        d_model=d_model,
        n_heads=int(cfg.num_attention_heads),
        head_dim=int(cfg.head_dim),
        max_seq_len=max(seq_len, 512),
        device=x.device,
    )
    pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        rope_cos, rope_sin = rope(torch.zeros(1, seq_len, d_model), pos)
    ref = {
        "self": _self_ref(core, layer_idx, x, rope_cos, rope_sin).to(torch.float32),
        "cross": _cross_ref(core, layer_idx, x, enc).to(torch.float32),
    }

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = open_dit_device(ttnn, mesh_sku="BH_QB", num_command_queues=1)
    ok = True
    try:
        rotary = TtHfRotaryEmbedding(
            mesh_device=mesh,
            head_dim=int(cfg.head_dim),
            max_seq_len=int(cfg.max_position_embeddings),
            rope_theta=float(cfg.rope_theta),
            hidden_size=int(cfg.hidden_size),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            dtype=ttnn.bfloat16,
        )
        for kind in ("self", "cross"):
            os.environ["ACE_STEP_TP"] = "off"
            y_off = _run(mesh, cfg, sd, layer_idx, kind, x, enc, rotary)
            os.environ["ACE_STEP_TP"] = "on"
            y_on = _run(mesh, cfg, sd, layer_idx, kind, x, enc, rotary)
            _, p_off = comp_pcc(ref[kind], y_off, pcc=_GATE)
            _, p_on = comp_pcc(ref[kind], y_on, pcc=_GATE)
            _, p_oo = comp_pcc(y_off, y_on, pcc=_GATE)
            print(
                f"[phase2b][{kind}] off-vs-torch={float(p_off):.6f}  on-vs-torch={float(p_on):.6f}  "
                f"on-vs-off={float(p_oo):.6f}",
                flush=True,
            )
            ok = ok and float(p_on) >= _GATE and float(p_oo) >= _GATE
        print(f"[phase2b] GATE {'PASS' if ok else 'FAIL'} (threshold {_GATE})", flush=True)
        return 0 if ok else 1
    finally:
        try:
            close_ace_step_device(ttnn, mesh)
        except Exception:
            pass
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
