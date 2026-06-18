# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
De-risk smoke: full model on (4,8) with DP-attention (row-sharded, 4 prompts) + EP=32 MoE.

Each mesh ROW = one prompt (full attention per row, TP=8 across cols, NO SP). The MoE is
a single shared EP pool (experts spread 8/chip across 32); dispatch_group_size = 4 rows
routes all 4 prompts' tokens to the shared experts. This is the "DP-attention + shared-EP"
pattern. Random weights, reduced layers/experts — checks the (4,8)+row-sharded+EP forward
RUNS (finite, no hang/OOM) before paying the real-weights run.

Run:
  cd /data/vmelnykov/tt-metal
  source python_env/bin/activate
  export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x8_mesh_graph_descriptor.textproto
  python3 models/demos/minimax_m3/tests/galaxy_ep_forward_smoke.py --layers 4 --experts 64 --seq 128
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
    c["use_qk_norm"] = False
    c["attn_type_list"] = [1] * num_layers
    return SimpleNamespace(**c)


def make_state(cfg):
    H, I, E, L, V = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts, cfg.num_hidden_layers, cfg.vocab_size
    qd, kvd = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
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
            sd[ep + "w1.weight"], sd[ep + "w3.weight"], sd[ep + "w2.weight"] = rn(I, H), rn(I, H), rn(H, I)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    args = ap.parse_args()

    from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
    from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import open_mesh_device
    from models.demos.minimax_m3.config import MeshConfig, ModeConfig
    from models.demos.minimax_m3.tt.ccl import CCLManager
    from models.demos.minimax_m3.tt.model import Model
    from models.demos.minimax_m3.utils.general_utils import get_default_num_links

    cfg = load_cfg(args.layers, args.experts)
    H, V = cfg.hidden_size, cfg.vocab_size
    shape = (4, 8)
    print(
        f"[ep-fwd] mesh={shape} DP=4 rows (4 prompts) TP=8 cols, EP MoE, layers={args.layers} experts={args.experts} seq={args.seq}",
        flush=True,
    )

    sd = make_state(cfg)
    mesh = open_mesh_device(shape, MiniMaxM27Config)
    print(f"[ep-fwd] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        mesh_config = MeshConfig(shape, decode=ModeConfig(tp=8, ep=4))
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
        model = Model(
            mesh_device=mesh,
            hf_config=cfg,
            state_dict=sd,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            create_kv_cache=False,
            max_local_batch_size=1,
            users_row_sharded=True,
            use_ep_moe=True,
            ep_seq_len_per_chip=args.seq,
        )
        print("[ep-fwd] model built (DP-attn + EP)", flush=True)

        # 4 prompts (one per row), random token ids
        tokens = torch.randint(0, V, (4, args.seq), dtype=torch.int32)
        host_out = model.prepare_inputs_prefill(tokens, page_table=None, batched_prefill=True)
        last = ((args.seq - 1) // 32) * 32
        tt_logits = model.ttnn_prefill_forward(
            host_out[0],
            rot_mats_global=host_out[1],
            rot_mats_local=host_out[2],
            page_table=host_out[3],
            kv_cache=None,
            batch_size=1,
            get_last_token=last,
        )
        ttnn.synchronize_device(mesh)
        print("[ep-fwd] forward complete", flush=True)

        # per-row logits: gather TP cols for each row
        dts = ttnn.get_device_tensors(tt_logits)
        nc = shape[1]
        for r in range(shape[0]):
            row = torch.cat([ttnn.to_torch(dts[r * nc + c]) for c in range(nc)], dim=-1).float()
            vec = row.reshape(-1, row.shape[-1])[args.seq - 1 - last][:V]
            print(
                f"[ep-fwd] row{r} (prompt {r}): finite={torch.isfinite(vec).all().item()} argmax={int(vec.argmax())}",
                flush=True,
            )
        print("[ep-fwd] PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
