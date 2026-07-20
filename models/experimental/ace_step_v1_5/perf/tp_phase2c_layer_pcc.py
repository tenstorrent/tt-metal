# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TP Phase 2c gate: the full ``TtAceStepDiTLayer`` (self-attn + cross-attn + modulated MLP) must
match the replicate baseline (and torch) on the BH_QB 2×2 mesh with TP on. No new sharding code —
this validates that the TP-aware attention + MLP compose correctly, with modulation/norms/residuals
operating on the always-replicated hidden state.

Run (device free):
    python models/experimental/ace_step_v1_5/perf/tp_phase2c_layer_pcc.py
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import tiny_dit_decoder_fixture
from models.experimental.ace_step_v1_5.torch_ref.dit_decoder_core import TorchAceStepDiTCoreRef, TorchAceStepDiTLayerRef
from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import TtAceStepDiTLayer, TtHfRotaryEmbedding
from models.experimental.ace_step_v1_5.utils.tt_device import (
    ace_step_replicate_mesh_mapper,
    close_ace_step_device,
    open_dit_device,
)

_GATE_EQUIV = 0.99  # TP-on vs replicate (same numerics)
_GATE_TORCH = 0.98  # TP-on vs torch (bf16 drift over the full block)


def _read0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).to(torch.float32)


def _run(mesh, cfg, sd, layer_idx, x, tsp, enc, rotary):
    layer = TtAceStepDiTLayer(
        cfg=cfg, state_dict=sd, layer_idx=layer_idx, mesh_device=mesh, dtype=ttnn.bfloat16, rotary_embedding=rotary
    )
    rep = ace_step_replicate_mesh_mapper(mesh)
    x_tt = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
    tp_tt = ttnn.from_torch(tsp, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
    enc_tt = ttnn.from_torch(enc, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
    return _read0(layer(x_tt, tp_tt, enc_tt))


def main() -> int:
    cfg, sd, d_model, seq_len, enc_len = tiny_dit_decoder_fixture()
    layer_idx = 0
    core = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)
    layer_ref = TorchAceStepDiTLayerRef(cfg=cfg, state_dict=sd, layer_idx=layer_idx)
    torch.manual_seed(7)
    x = torch.randn(1, 1, seq_len, d_model, dtype=torch.bfloat16)
    tsp = torch.randn(1, 6, d_model, dtype=torch.bfloat16)
    enc = core.condition_encoder_hidden_states(torch.randn(1, enc_len, 32, dtype=torch.bfloat16))
    ref = layer_ref(x, tsp, enc).to(torch.float32)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = open_dit_device(ttnn, mesh_sku="BH_QB", num_command_queues=1)
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
        os.environ["ACE_STEP_TP"] = "off"
        y_off = _run(mesh, cfg, sd, layer_idx, x, tsp, enc, rotary)
        os.environ["ACE_STEP_TP"] = "on"
        y_on = _run(mesh, cfg, sd, layer_idx, x, tsp, enc, rotary)

        _, p_off = comp_pcc(ref, y_off, pcc=_GATE_TORCH)
        _, p_on = comp_pcc(ref, y_on, pcc=_GATE_TORCH)
        _, p_oo = comp_pcc(y_off, y_on, pcc=_GATE_EQUIV)
        print(
            f"[phase2c] off-vs-torch={float(p_off):.6f}  on-vs-torch={float(p_on):.6f}  "
            f"on-vs-off={float(p_oo):.6f}",
            flush=True,
        )
        ok = float(p_on) >= _GATE_TORCH and float(p_oo) >= _GATE_EQUIV
        print(f"[phase2c] GATE {'PASS' if ok else 'FAIL'} (equiv≥{_GATE_EQUIV}, torch≥{_GATE_TORCH})", flush=True)
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
