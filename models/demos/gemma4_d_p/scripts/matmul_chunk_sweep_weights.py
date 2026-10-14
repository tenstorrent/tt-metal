# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real TP-rank-0 projection weights of Gemma4-31B for matmul_chunk_sweep.py --weights.

    python models/demos/gemma4_d_p/scripts/matmul_chunk_sweep_weights.py <out.pt> [--model google/gemma-4-31B-it]

Takes sliding layer 0 and global layer 5 from the local Hugging Face snapshot (HF_HOME; no download) and writes
{op: [K, N] bf16 tensor} for the ops in matmul_chunk_sweep.OPS. Shards follow tt/attention/weights.py and tt/mlp.py:
column-parallel QKV/QK, gate and up; row-parallel o_proj and down. The head reorders the model applies under CP only
permute rows or columns, which does not change the timing, so they are skipped.
"""

import argparse
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

TP = 4
SLIDING_LAYER, GLOBAL_LAYER = 0, 5


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--model", default=os.environ.get("HF_MODEL", "google/gemma-4-31B-it"))
    args = p.parse_args()

    snap = snapshot_download(args.model, local_files_only=True, allow_patterns=["*.safetensors", "*.json"])
    weight_map = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]

    def get(layer, name):
        key = f"model.language_model.layers.{layer}.{name}"
        with safe_open(f"{snap}/{weight_map[key]}", framework="pt") as f:
            return f.get_tensor(key).float()

    def rank0_rows(t):  # torch.chunk(weight, tp, dim=0)[0], as the weight builders shard
        return torch.chunk(t, TP, dim=0)[0]

    q, k, v = (get(SLIDING_LAYER, f"self_attn.{n}_proj.weight") for n in "qkv")
    gq, gk = (get(GLOBAL_LAYER, f"self_attn.{n}_proj.weight") for n in "qk")
    weights = {
        "sliding_qkv": torch.cat([rank0_rows(q).T, rank0_rows(k).T, rank0_rows(v).T], dim=-1),
        "global_qk": torch.cat([rank0_rows(gq).T, rank0_rows(gk).T], dim=-1),
        "sliding_o": rank0_rows(get(SLIDING_LAYER, "self_attn.o_proj.weight").T),
        "global_o": rank0_rows(get(GLOBAL_LAYER, "self_attn.o_proj.weight").T),
        "mlp_gate": torch.chunk(get(SLIDING_LAYER, "mlp.gate_proj.weight").T, TP, dim=-1)[0],
        "mlp_up": torch.chunk(get(SLIDING_LAYER, "mlp.up_proj.weight").T, TP, dim=-1)[0],
        "mlp_down": rank0_rows(get(SLIDING_LAYER, "mlp.down_proj.weight").T),
    }
    weights = {name: t.contiguous().bfloat16() for name, t in weights.items()}
    for name, t in weights.items():
        print(f"{name:12s} {tuple(t.shape)}")
    torch.save(weights, args.out)


if __name__ == "__main__":
    main()
