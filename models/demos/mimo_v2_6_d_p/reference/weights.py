# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access and dequantization for MiMo-V2.6-Flash-RL (shared by the CPU reference and the HF oracle hook).

Storage (checkpoint metadata ``save_format: mxfp4, tp_size: 4``):
    qkv_proj.weight      fp8 e4m3 [q+k+v, 4096], weight_scale_inv fp32 per 128x128 block. The fused projection is
                         stored per tensor-parallel rank: rows are [q_r, k_r, v_r] for r = 0..3 (rank r holds q heads
                         [16r, 16r+16) and its share of the KV heads), and the scale blocks are counted per rank slab
                         (layer 0: 3392 rows per rank -> 27 blocks, 108 in total, not ceil(13568/128) = 106). Found
                         from the scale count and the per-row norms (k and v rows repeat every 3392 / 3712 rows).
                         ``qkv_weight`` returns the global [q; k; v] layout the HF module splits.
    mlp.{gate,up,down}_proj (dense layer 0)   fp8 e4m3 + fp32 weight_scale_inv per 128x128 block (contiguous).
    experts.{e}.{gate,up,down}_proj.weight    mxfp4: uint8 [out, in/2], two e2m1 codes per byte, low nibble = even
                         input column (checked: the low-first per-column |W| profile correlates 0.61 with the layer's
                         post_attention_layernorm weight, high-first 0.03); weight_scale uint8 e8m0 [out, in/32],
                         value = 2^(s - 127).
    o_proj, norms, router (gate.weight bf16, e_score_correction_bias fp32), sink bias, embed, lm_head: stored dense.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

FP8_BLOCK = 128
MX_BLOCK = 32
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
# byte -> (value of low nibble, value of high nibble)
_BYTE_LUT = torch.stack([E2M1[torch.arange(256) & 15], E2M1[torch.arange(256) >> 4]], dim=-1)  # [256, 2]


class WeightLoader:
    """Lazy safetensors accessor keyed by checkpoint tensor name (never touches model.mtp.* / visual / audio)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            index = json.load(f)
        self.weight_map: dict[str, str] = index["weight_map"]
        self.tp_size = int((index.get("metadata") or {}).get("tp_size", 1))
        self._handles = {}

    def get(self, name: str) -> torch.Tensor:
        fname = self.weight_map[name]
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.model_path, fname), framework="pt")
        return self._handles[fname].get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map


def fp8_block_dequant(w: torch.Tensor, scale_inv: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """W = fp8(w) * scale_inv[row // 128, col // 128], computed in fp32 (exact), then cast to dtype."""
    rows, cols = w.shape
    s = scale_inv.float().repeat_interleave(FP8_BLOCK, 0)[:rows].repeat_interleave(FP8_BLOCK, 1)[:, :cols]
    return (w.float() * s).to(dtype)


def fp8_weight(loader: WeightLoader, name: str, dtype=torch.float32) -> torch.Tensor:
    return fp8_block_dequant(loader.get(name), loader.get(name + "_scale_inv"), dtype)


def qkv_weight(loader: WeightLoader, prefix: str, sizes: tuple[int, int, int], dtype=torch.float32) -> torch.Tensor:
    """Dequantized fused qkv_proj [q+k+v, H] in the global [q; k; v] row order.

    The checkpoint stores it per TP rank ([q_r; k_r; v_r] per rank, each rank slab quantized on its own 128-row
    blocks); a scale count of ceil(rows / 128) with tp_size 1 means a plain global layout."""
    w = loader.get(prefix + "qkv_proj.weight")
    s = loader.get(prefix + "qkv_proj.weight_scale_inv")
    tp = loader.tp_size
    q, k, v = sizes
    assert w.shape[0] == q + k + v, (w.shape, sizes)
    if tp == 1:
        return fp8_block_dequant(w, s, dtype)
    assert q % tp == 0 and k % tp == 0 and v % tp == 0, (sizes, tp)
    r_rows = (q + k + v) // tp
    nb = -(-r_rows // FP8_BLOCK)
    assert s.shape[0] == tp * nb, f"{prefix}qkv scale rows {s.shape[0]} != tp {tp} x {nb} blocks per rank"
    parts = []
    for r in range(tp):
        parts.append(fp8_block_dequant(w[r * r_rows : (r + 1) * r_rows], s[r * nb : (r + 1) * nb], dtype))
    qs, ks, vs = q // tp, k // tp, v // tp
    return torch.cat(
        [p[:qs] for p in parts] + [p[qs : qs + ks] for p in parts] + [p[qs + ks :] for p in parts], dim=0
    ).contiguous()


def mxfp4_dequant(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """packed uint8 [out, in/2] (low nibble first) + e8m0 scale uint8 [out, in/32] -> [out, in].

    e2m1 x 2^k is exact in fp32 and bf16 (1 mantissa bit), so the result is exact in either dtype."""
    out, half = packed.shape
    vals = F.embedding(packed.int(), _BYTE_LUT).view(out, -1, MX_BLOCK)  # [out, in/32, 32] (a fast LUT gather)
    sc = torch.exp2(scale.float() - 127.0)  # [out, in/32]
    return vals.mul_(sc[:, :, None]).view(out, half * 2).to(dtype)


class PackedExpert:
    """One routed expert kept in its checkpoint form (mxfp4 + e8m0), dequantized on use."""

    __slots__ = ("gate", "gate_s", "up", "up_s", "down", "down_s")

    def __init__(self, loader: WeightLoader, prefix: str):
        g = loader.get
        self.gate, self.gate_s = g(prefix + "gate_proj.weight"), g(prefix + "gate_proj.weight_scale")
        self.up, self.up_s = g(prefix + "up_proj.weight"), g(prefix + "up_proj.weight_scale")
        self.down, self.down_s = g(prefix + "down_proj.weight"), g(prefix + "down_proj.weight_scale")

    def weights(self, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(gate [I, H], up [I, H], down [H, I]) dequantized."""
        return (
            mxfp4_dequant(self.gate, self.gate_s, dtype),
            mxfp4_dequant(self.up, self.up_s, dtype),
            mxfp4_dequant(self.down, self.down_s, dtype),
        )
