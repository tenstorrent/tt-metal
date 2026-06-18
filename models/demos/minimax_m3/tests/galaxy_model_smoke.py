# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Step 1d — FULL-model assembly smoke at TP=8 (random weights, no HF compare).

Builds the whole Model (embed -> N decoder layers -> norm -> lm_head) and runs one
ttnn_prefill_forward -> logits -> first token. Purpose: catch OOM / hang / wiring
bugs in the full 62-layer assembly + lm_head BEFORE paying the ~460GB real-weight
load (Step 1e). Random bf16 weights; experts reduced by default to bound HOST RAM
(weights are generated on host). Use --layers 62 for full depth.

Pure TP=8 / SP=1 / EP=1 on (1,8); Linear topology (plain-MESH Galaxy).

Run:
  cd /data/vmelnykov/tt-metal
  export TT_METAL_HOME=/data/vmelnykov/tt-metal PYTHONPATH=/data/vmelnykov/tt-metal
  source python_env/bin/activate
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_1x8_mesh_graph_descriptor.textproto
  python3 models/demos/minimax_m3/tests/galaxy_model_smoke.py --layers 62 --experts 8 --seq 128
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

import ttnn


def load_cfg(num_layers, num_experts):
    p = os.path.join(os.path.dirname(__file__), "..", "configs", "MiniMax-M3", "config.json")
    with open(p) as f:
        c = json.load(f)
    c["num_hidden_layers"] = num_layers
    c["num_local_experts"] = num_experts
    c["use_qk_norm"] = False  # skip partial-rotary q/k-norm swizzle for the smoke
    c["attn_type_list"] = [1] * num_layers
    return SimpleNamespace(**c)


def make_state(cfg):
    H, I, E, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts, cfg.num_hidden_layers
    V = cfg.vocab_size
    qd = cfg.num_attention_heads * cfg.head_dim
    kvd = cfg.num_key_value_heads * cfg.head_dim
    g = torch.Generator().manual_seed(0)
    rn = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(torch.bfloat16)
    sd = {
        "model.embed_tokens.weight": rn(V, H),
        "model.norm.weight": torch.ones(H, dtype=torch.bfloat16),
        "lm_head.weight": rn(V, H),
    }
    for i in range(L):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "self_attn.q_proj.weight"] = rn(qd, H)
        sd[p + "self_attn.k_proj.weight"] = rn(kvd, H)
        sd[p + "self_attn.v_proj.weight"] = rn(kvd, H)
        sd[p + "self_attn.o_proj.weight"] = rn(H, qd)
        sd[p + "block_sparse_moe.gate.weight"] = rn(E, H)
        sd[p + "block_sparse_moe.e_score_correction_bias"] = torch.zeros(E, dtype=torch.bfloat16)
        for e in range(E):
            ep = p + f"block_sparse_moe.experts.{e}."
            sd[ep + "w1.weight"] = rn(I, H)
            sd[ep + "w3.weight"] = rn(I, H)
            sd[ep + "w2.weight"] = rn(H, I)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--layers", type=int, default=62)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    args = ap.parse_args()

    from models.demos.minimax_m3.config import MeshConfig, ModeConfig
    from models.demos.minimax_m3.tt.ccl import CCLManager
    from models.demos.minimax_m3.tt.model import Model
    from models.demos.minimax_m3.utils.general_utils import get_default_num_links

    cfg = load_cfg(args.layers, args.experts)
    H, V = cfg.hidden_size, cfg.vocab_size
    shape = (args.rows, args.cols)
    print(
        f"[model-smoke] mesh={shape} TP={args.cols} layers={args.layers} experts={args.experts} seq={args.seq}",
        flush=True,
    )

    print("[model-smoke] generating random state dict on host ...", flush=True)
    sd = make_state(cfg)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    print(f"[model-smoke] mesh opened: {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)

    try:
        mesh_config = MeshConfig(shape, decode=ModeConfig(tp=shape[1], ep=shape[0]))
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)

        model = Model(
            mesh_device=mesh,
            hf_config=cfg,
            state_dict=sd,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            create_kv_cache=True,
            max_local_batch_size=1,
        )
        print("[model-smoke] full Model assembled", flush=True)

        x = torch.randn(1, 1, args.seq, H) * 0.1
        x_tt = ttnn.from_torch(
            x,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        last = ((args.seq - 1) // 32) * 32
        logits = model.ttnn_prefill_forward(x_tt, get_last_token=last)
        print("[model-smoke] forward complete", flush=True)

        # logits are TP-sharded over vocab (cols); gather to full padded vocab
        full = ttnn.to_torch(
            logits, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=mesh.shape, dims=(-2, -1))
        ).float()
        row = full.reshape(-1, full.shape[-1])[args.seq - 1 - last]  # last real token within the 32-tile
        tok = int(row[:V].argmax())
        finite = torch.isfinite(full).all().item()
        print(
            f"[model-smoke] logits_shape={tuple(full.shape)} finite={finite} first_token_id={tok} (vocab={V})",
            flush=True,
        )
        assert finite, "non-finite logits"
        assert 0 <= tok < V, f"token id {tok} out of range"
        print("[model-smoke] PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
