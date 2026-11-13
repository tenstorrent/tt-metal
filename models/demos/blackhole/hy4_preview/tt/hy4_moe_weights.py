# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side preparation of one Hy4-Preview MoE layer for TTMoEDecode (host tensors in the DSV4 ``load_moe_layer`` dict format).

Checkpoint (bf16, per layer ``model.layers.<L>.mlp.*``):
    gate.weight [256, 6144], gate.e_score_correction_bias [256] fp32
    experts.gate_up_proj [256, 4096, 6144]  per expert [out, in]; rows [0, 2048) = GATE, rows [2048, 4096) = UP   (RESOLVED, see layer-moe/RESULT.md:
                                            the first half has the per-expert mean offset / skew of a gate, the second half is zero-mean symmetric like the shared up_proj)
    experts.down_proj    [256, 6144, 2048]  per expert [out, in]
    shared_experts.{gate,up,down}_proj       [2048, 6144] / [2048, 6144] / [6144, 2048]
TTMoEDecode wants w0 (gate), w1 (up): [1, E, hidden, inter] and w2: [1, E, inter, hidden], i.e. [in, out]: every expert is TRANSPOSED.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from models.demos.blackhole.hy4_preview.weights import Hy4Weights

HIDDEN, INTER, NUM_ROUTED = 6144, 2048, 256
WEIGHT_CACHE = os.environ.get("HY4_WEIGHT_CACHE", "/home/ttuser/ssinghal/tt-metal-cache/hy4_weight_cache")


def ring_suffix():
    """MOE_COMPUTE_RING=N (moe-perf lever, default off): the packed expert tensors are laid out for a matmul ring of N cores -> own cache directory."""
    r = os.environ.get("MOE_COMPUTE_RING", "")
    return f"_r{r}" if r else ""


def expert_cache_dir(layer_id: int, tag: str = ""):
    return (
        os.path.join(WEIGHT_CACHE, f"layer_{layer_id}{ring_suffix()}{tag}")
        if WEIGHT_CACHE and WEIGHT_CACHE != "0"
        else None
    )


def _cache_warm(layer_id: int, bfp8: bool = False):
    d = expert_cache_dir(layer_id)
    t = "bfp8" if bfp8 else "bfp4"
    return bool(d) and all(
        os.path.exists(os.path.join(d, f))
        for f in (f"moe_w0_w1_{t}.tensorbin", f"moe_w2_{t}.tensorbin", f"moe_meta_{t}.json")
    )


def load_moe_layer(layer_id: int, dtype=torch.bfloat16, experts=None, load_experts=True, ckpt_dir=None, W=None):
    """Dict of host tensors for MoE layer ``layer_id``. experts: iterable of expert ids to load (others stay zero), for single-expert tests."""
    W = W or Hy4Weights(ckpt_dir)
    warm = (experts is None and _cache_warm(layer_id)) or not load_experts
    ids = [] if warm else (list(range(NUM_ROUTED)) if experts is None else list(experts))
    w0 = w1 = w2 = None
    if not warm:
        w0 = torch.zeros(1, NUM_ROUTED, HIDDEN, INTER, dtype=dtype)
        w1 = torch.zeros(1, NUM_ROUTED, HIDDEN, INTER, dtype=dtype)
        w2 = torch.zeros(1, NUM_ROUTED, INTER, HIDDEN, dtype=dtype)

    def one(e):
        g, u, d = W.load_expert(
            layer_id, e
        )  # [2048, 6144] x2, [6144, 2048]; split order from Hy4Weights.gate_up_order (default gate_first)
        w0[0, e] = g.T.to(dtype)
        w1[0, e] = u.T.to(dtype)
        w2[0, e] = d.T.to(dtype)

    with ThreadPoolExecutor(max_workers=int(os.environ.get("HY4_LOAD_THREADS", "16"))) as ex:
        list(ex.map(one, ids))
    p = f"layers.{layer_id}."
    sh = {k: W.load(p + f"moe.shared.{k}_proj") for k in ("gate", "up", "down")}
    t = lambda m: m.T.contiguous().to(dtype)[None, None]  # [in, out]
    return {
        "expert_mapping": None,
        "gate_weight": W.load(p + "moe.router.weight").to(
            torch.float32
        ),  # [256, 6144] checkpoint layout (Hy4Gate transposes)
        "gate_bias": W.load(p + "moe.router.bias").to(torch.float32),
        "w0": w0,
        "w1": w1,
        "w2": w2,
        "cache_dir": expert_cache_dir(layer_id) if (experts is None and load_experts) else None,
        "shared_w0": {NUM_ROUTED: t(sh["gate"])},
        "shared_w1": {NUM_ROUTED: t(sh["up"])},
        "shared_w2": {NUM_ROUTED: t(sh["down"])},
    }
