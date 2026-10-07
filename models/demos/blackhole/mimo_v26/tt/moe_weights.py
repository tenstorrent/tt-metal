# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal reader of MiMo routed-expert weights (fp4 e2m1 low-nibble-first + e8m0 per 32 along K) -> bf16 in DEVICE layout (gate/up [K,N], down [N,K]).
Same decode as track 0's reference/loader.py (float8_e8m0fnu view) and my bfp8 exactness test. Expert e of layer L: model.layers.L.mlp.experts.E.{gate,up,down}_proj."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch

CKPT = os.environ.get("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_idx = None


def _index():
    global _idx
    if _idx is None:
        _idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    return _idx


def get_tensor(name):
    from safetensors import safe_open

    with safe_open(os.path.join(CKPT, _index()[name]), "pt") as f:
        return f.get_tensor(name)


def dequant_fp4(w_u8, s_u8, block=32):
    b = w_u8.view(torch.uint8)
    pair = torch.stack([FP4_TABLE[(b & 0x0F).long()], FP4_TABLE[(b >> 4).long()]], dim=-1)
    n = b.size(0)
    scale = torch.pow(2.0, s_u8.view(torch.uint8).float() - 127.0)
    return (pair.reshape(n, -1).view(n, -1, block) * scale.unsqueeze(-1)).reshape(n, -1)


def load_expert(layer, e):
    out = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
        out.append(
            dequant_fp4(get_tensor(base + ".weight"), get_tensor(base + ".weight_scale")).t().contiguous().bfloat16()
        )
    return tuple(out)  # gate [K,N], up [K,N], down [N,K]


def load_layer_experts(layer, n_experts=256, threads=24):
    with ThreadPoolExecutor(max_workers=threads) as ex:
        return dict(zip(range(n_experts), ex.map(lambda e: load_expert(layer, e), range(n_experts))))
