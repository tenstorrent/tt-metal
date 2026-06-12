# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Step 1 — multi-device DecoderLayer smoke at PURE TP=8 (no SP, no HF compare).

First execution of minimax_m2's TP collective paths (attention reduce-scatter /
all-gather, expert TP all-reduce) on real multi-chip hardware. Random weights,
qk-norm disabled, reduced experts — this checks "does the TP=8 path run on 8
chips and produce a finite, correctly-shaped output", NOT correctness (that's
Step 2, vs HF). Topology = Linear because this Galaxy is a plain MESH (no torus).

Run:
  cd /data/vmelnykov/tt-metal
  export TT_METAL_HOME=/data/vmelnykov/tt-metal PYTHONPATH=/data/vmelnykov/tt-metal
  source python_env/bin/activate
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_1x8_mesh_graph_descriptor.textproto
  python3 models/demos/minimax_m2/tests/galaxy_layer_smoke.py --experts 32 --seq 128
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

import ttnn


def load_cfg(num_experts):
    p = os.path.join(os.path.dirname(__file__), "..", "configs", "MiniMax-M2", "config.json")
    with open(p) as f:
        c = json.load(f)
    c["num_local_experts"] = num_experts
    c["use_qk_norm"] = False  # skip the partial-rotary q/k-norm swizzle for the smoke
    return SimpleNamespace(**c)


def make_layer_state(cfg, E):
    H = cfg.hidden_size
    qd = cfg.num_attention_heads * cfg.head_dim
    kvd = cfg.num_key_value_heads * cfg.head_dim
    I = cfg.intermediate_size
    g = torch.Generator().manual_seed(0)
    rn = lambda *s: torch.randn(*s, generator=g) * 0.02
    sd = {
        "input_layernorm.weight": torch.ones(H),
        "post_attention_layernorm.weight": torch.ones(H),
        "self_attn.q_proj.weight": rn(qd, H),
        "self_attn.k_proj.weight": rn(kvd, H),
        "self_attn.v_proj.weight": rn(kvd, H),
        "self_attn.o_proj.weight": rn(H, qd),
        "block_sparse_moe.gate.weight": rn(E, H),
        "block_sparse_moe.e_score_correction_bias": torch.zeros(E),
    }
    for e in range(E):
        sd[f"block_sparse_moe.experts.{e}.w1.weight"] = rn(I, H)
        sd[f"block_sparse_moe.experts.{e}.w3.weight"] = rn(I, H)
        sd[f"block_sparse_moe.experts.{e}.w2.weight"] = rn(H, I)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    args = ap.parse_args()

    from models.demos.minimax_m2.config import MeshConfig, ModeConfig
    from models.demos.minimax_m2.tt.ccl import CCLManager
    from models.demos.minimax_m2.tt.layer import DecoderLayer
    from models.demos.minimax_m2.tt.model import create_rope_setup
    from models.demos.minimax_m2.utils.general_utils import get_default_num_links

    cfg = load_cfg(args.experts)
    H = cfg.hidden_size
    shape = (args.rows, args.cols)
    print(f"[layer-smoke] mesh={shape} TP={args.cols} experts={args.experts} seq={args.seq}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    print(f"[layer-smoke] mesh opened: {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)

    try:
        # pure TP=8: tp = cols, ep = rows (=1) -> prefill auto sp=rows(=1), ep=1
        mesh_config = MeshConfig(shape, decode=ModeConfig(tp=shape[1], ep=shape[0]))
        # Linear topology — this Galaxy is a plain MESH (no torus).
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
        rope = create_rope_setup(mesh_device=mesh, hf_config=cfg, datatype=ttnn.bfloat16)

        sd = make_layer_state(cfg, args.experts)
        layer = DecoderLayer(
            mesh,
            cfg,
            sd,
            layer_idx=0,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            transformation_mats=rope.get_both_trans_mats(),
            max_seq_len=max(args.seq, 128),
            max_local_batch_size=1,
            use_throughput_experts=False,
            expert_weight_dtype=ttnn.bfloat8_b,
        )
        print("[layer-smoke] DecoderLayer built", flush=True)

        rope_mats = [rope.cos_matrix_prefill[:, :, : args.seq, :], rope.sin_matrix_prefill[:, :, : args.seq, :]]
        x = torch.randn(1, 1, args.seq, H) * 0.1
        x_tt = ttnn.from_torch(
            x,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        out = layer(x_tt, position_embeddings=rope_mats)
        out_t = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(1, args.seq, H).float()
        finite = torch.isfinite(out_t).all().item()
        print(
            f"[layer-smoke] forward OK  out_shape={tuple(out_t.shape)} finite={finite} "
            f"mean={out_t.mean():.4f} std={out_t.std():.4f}",
            flush=True,
        )
        assert finite, "non-finite output"
        print("[layer-smoke] PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
