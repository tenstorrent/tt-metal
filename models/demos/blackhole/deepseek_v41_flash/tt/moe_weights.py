# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side preparation of one DeepSeek-V4.1-Flash MoE layer for ``TTMoEGate`` / ``TTMoEDecode``.

Checkpoint (per layer, ``layers.{L}.ffn.*``):
    gate.weight [384, 5120] bf16, gate.bias [384] fp32 (selection-only correction bias)
    experts.{e}.w1/w3 [2304, 5120] fp4 (packed int8 [2304, 2560]) + e8m0 scales [2304, 160]  -> gate / up
    experts.{e}.w2    [5120, 2304] fp4 (packed int8 [5120, 1152]) + e8m0 scales [5120, 72]   -> down
    shared_experts.w1/w3/w2: fp8 e4m3 + e8m0 scales on 32x32 blocks

TTMoEDecode computes ``silu(x @ w0) * (x @ w1) @ w2`` with ``w0, w1: [L, E, hidden, inter]`` and
``w2: [L, E, inter, hidden]``, so w0 = gate (checkpoint w1), w1 = up (checkpoint w3), all transposed to
``[in, out]``. The checkpoint's swiglu clamp (|up|<=10, gate<=10) is not expressible in the SILU path of
``moe_compute``; ``clamp_effect`` in ``tests`` measures what dropping it costs.
"""

import json
import os

import torch
from safetensors import safe_open

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels

CKPT_DIR = os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash")
NUM_ROUTED = 384
HIDDEN = 5120
INTER = 2304


class _Shards:
    def __init__(self, ckpt_dir=CKPT_DIR):
        self.dir = ckpt_dir
        self.index = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))["weight_map"]
        self._h = {}
        import threading

        self._lock = threading.Lock()

    def get(self, key):
        path = os.path.join(self.dir, self.index[key])
        with self._lock:
            if path not in self._h:
                self._h[path] = safe_open(path, "pt")
            h = self._h[path]
        return h.get_tensor(key)


def _fp4_expert(sh, prefix, dtype):
    w = ref_kernels.dequant_fp4_weight(sh.get(prefix + ".weight"), sh.get(prefix + ".scale"))
    return w.T.contiguous().to(dtype)  # [in, out]


def _fp8_linear(sh, prefix, dtype):
    w = ref_kernels.dequant_fp8_weight(sh.get(prefix + ".weight"), sh.get(prefix + ".scale"), 32)
    return w.T.contiguous().to(dtype)  # [in, out]


WEIGHT_CACHE = os.environ.get("DSV41_WEIGHT_CACHE", "/mnt/tt-data/ssinghal/dsv4-weight-cache")


def expert_cache_dir(layer_id: int):
    """Directory of the packed + quantized routed-expert tensors (see TTMoEDecode ``weight_cache_dir``); None = off."""
    return os.path.join(WEIGHT_CACHE, f"layer_{layer_id}") if WEIGHT_CACHE and WEIGHT_CACHE != "0" else None


def _expert_cache_warm(layer_id: int):
    d = expert_cache_dir(layer_id)
    tag = "bfp8" if os.environ.get("MOE_COMPUTE_BFP8_WEIGHTS", "0") != "0" else "bfp4"
    return bool(d) and all(
        os.path.exists(os.path.join(d, f))
        for f in (f"moe_w0_w1_{tag}.tensorbin", f"moe_w2_{tag}.tensorbin", f"moe_meta_{tag}.json")
    )


def load_moe_layer(layer_id: int, dtype=torch.bfloat16, ckpt_dir=CKPT_DIR, experts=None):
    """Return a dict of host tensors for one MoE layer.

    experts: optional iterable of routed expert ids to load (default: all 384; the rest stay zero) —
    handy for single-expert unit tests. Shapes follow TTMoEDecode with L=1.
    """
    sh = _Shards(ckpt_dir)
    p = f"layers.{layer_id}.ffn."
    warm = experts is None and _expert_cache_warm(layer_id)  # packed experts are cached: skip the 384-expert dequant
    ids = [] if warm else (list(range(NUM_ROUTED)) if experts is None else list(experts))
    w0 = w1 = w2 = None
    if not warm:
        w0 = torch.zeros(1, NUM_ROUTED, HIDDEN, INTER, dtype=dtype)
        w1 = torch.zeros(1, NUM_ROUTED, HIDDEN, INTER, dtype=dtype)
        w2 = torch.zeros(1, NUM_ROUTED, INTER, HIDDEN, dtype=dtype)

    def one(e):  # torch ops release the GIL, so experts dequantise in parallel
        ep = f"{p}experts.{e}."
        w0[0, e] = _fp4_expert(sh, ep + "w1", dtype)
        w1[0, e] = _fp4_expert(sh, ep + "w3", dtype)
        w2[0, e] = _fp4_expert(sh, ep + "w2", dtype)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=int(os.environ.get("DSV41_LOAD_THREADS", "24"))) as ex:
        list(ex.map(one, ids))
    sp = p + "shared_experts."
    shared = {
        NUM_ROUTED: (
            _fp8_linear(sh, sp + "w1", dtype)[None, None],
            _fp8_linear(sh, sp + "w3", dtype)[None, None],
            _fp8_linear(sh, sp + "w2", dtype)[None, None],
        )
    }
    return {
        "gate_weight": sh.get(p + "gate.weight").T.contiguous().to(torch.float32),  # [hidden, 384]
        "gate_bias": sh.get(p + "gate.bias").to(torch.float32),  # [384]
        "w0": w0,
        "w1": w1,
        "w2": w2,
        "cache_dir": expert_cache_dir(layer_id) if experts is None else None,
        "shared_w0": {k: v[0] for k, v in shared.items()},
        "shared_w1": {k: v[1] for k, v in shared.items()},
        "shared_w2": {k: v[2] for k, v in shared.items()},
    }
