"""TP rank 0 projection weights of Gemma4-31B (sliding layer 0, global layer 5), as [K, N] for matmul_chunk_sweep --weights.
Shards follow tt/attention/weights.py and tt/mlp.py (column-parallel QKV/gate/up, row-parallel o/down); the head
reorders the model applies under CP only permute columns/rows and are skipped."""
import json, sys
import torch
from safetensors import safe_open

snap, out = sys.argv[1], sys.argv[2]
wm = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]


def get(name):
    key = f"model.language_model.layers.{name}"
    with safe_open(f"{snap}/{wm[key]}", framework="pt") as f:
        return f.get_tensor(key).float()


TP = 4


def rows(t, i=0):  # torch.chunk(weight, tp, dim=0)[0]
    return torch.chunk(t, TP, dim=0)[i]


q, k, v = get("0.self_attn.q_proj.weight"), get("0.self_attn.k_proj.weight"), get("0.self_attn.v_proj.weight")
gq, gk = get("5.self_attn.q_proj.weight"), get("5.self_attn.k_proj.weight")
w = {
    "sliding_qkv": torch.cat([rows(q).T, rows(k).T, rows(v).T], dim=-1),
    "global_qk": torch.cat([rows(gq).T, rows(gk).T], dim=-1),
    "sliding_o": rows(get("0.self_attn.o_proj.weight").T),
    "global_o": rows(get("5.self_attn.o_proj.weight").T),
    "mlp_gate": torch.chunk(get("0.mlp.gate_proj.weight").T, TP, dim=-1)[0],
    "mlp_up": torch.chunk(get("0.mlp.up_proj.weight").T, TP, dim=-1)[0],
    "mlp_down": rows(get("0.mlp.down_proj.weight").T),
}
w = {n: t.contiguous().bfloat16() for n, t in w.items()}
for n, t in w.items():
    print(n, tuple(t.shape))
torch.save(w, out)
