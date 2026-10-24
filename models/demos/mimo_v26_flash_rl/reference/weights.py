# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint reader for MiMo-V2.6-Flash: lazy safetensors access and the two dequantizers.

The checkpoint mixes three storage formats:
  * bf16: embeddings, lm_head, norms, o_proj, router weight, attention sink bias (the router's correction bias is fp32);
  * fp8 e4m3 with 128x128 block scales (``weight_scale_inv``): the fused qkv_proj and the layer-0 dense MLP;
  * mxfp4 (e2m1 nibbles, low nibble first, one e8m0 scale per 32 inputs): the routed experts.

The fused qkv_proj is stored per TP rank (index metadata ``tp_size``): rows are [q_0; k_0; v_0; q_1; k_1; v_1; ...]
and each rank slab is block-quantized on its own (layer 0: 13568 rows, 4 slabs of 3392, ceil(3392 / 128) = 27 scale
rows each = 108). ``qkv`` dequantizes slab by slab and returns q, k, v in global head order.

Every returned tensor is in the caller's weight dtype (bf16 holds mxfp4 exactly; fp8 * scale is rounded once).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

# e2m1: sign bit 3, exponent bits 2..1, mantissa bit 0
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)
BLOCK = 128  # fp8 block size
MX_BLOCK = 32  # mxfp4 block size


class Checkpoint:
    def __init__(self, path, wdtype: torch.dtype):
        self.path = Path(path)
        self.wdtype = wdtype
        idx = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = idx["weight_map"]
        self.tp_size = int((idx.get("metadata") or {}).get("tp_size", 1))
        self.config = json.loads((self.path / "config.json").read_text())
        self._files: dict[str, object] = {}

    def _handle(self, fname: str):
        h = self._files.get(fname)
        if h is None:
            h = self._files[fname] = safe_open(str(self.path / fname), "pt")
        return h

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def raw(self, name: str) -> torch.Tensor:
        return self._handle(self.weight_map[name]).get_tensor(name)

    def get(self, name: str) -> torch.Tensor:
        """A plain (bf16 / fp32) tensor in the weight dtype."""
        return self.raw(name).to(self.wdtype)

    # ---- fp8 block-scaled
    def fp8(self, prefix: str) -> torch.Tensor:
        w, s = self.raw(f"{prefix}.weight"), self.raw(f"{prefix}.weight_scale_inv")
        return dequant_fp8(w, s).to(self.wdtype)

    def qkv(self, prefix: str, q_size: int, k_size: int, v_size: int):
        """The fused, per-TP-rank fp8 qkv_proj as (q [q_size, H], k [k_size, H], v [v_size, H])."""
        w, s = self.raw(f"{prefix}.weight"), self.raw(f"{prefix}.weight_scale_inv")
        tp = self.tp_size
        rows = q_size + k_size + v_size
        assert w.shape[0] == rows and rows % tp == 0, (w.shape, rows, tp)
        slab = rows // tp
        sb = math.ceil(slab / BLOCK)
        assert s.shape[0] == sb * tp, f"{prefix}: {s.shape[0]} scale rows, expected {sb} per rank x {tp}"
        qr, kr, vr = q_size // tp, k_size // tp, v_size // tp
        qs, ks, vs = [], [], []
        for r in range(tp):
            d = dequant_fp8(w[r * slab : (r + 1) * slab], s[r * sb : (r + 1) * sb]).to(self.wdtype)
            qs.append(d[:qr])
            ks.append(d[qr : qr + kr])
            vs.append(d[qr + kr : qr + kr + vr])
        return torch.cat(qs), torch.cat(ks), torch.cat(vs)

    # ---- mxfp4
    def mxfp4(self, prefix: str) -> torch.Tensor:
        return dequant_mxfp4(self.raw(f"{prefix}.weight"), self.raw(f"{prefix}.weight_scale"), self.wdtype)


def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """fp8 [O, K] with [ceil(O/128), ceil(K/128)] block scales -> fp32 [O, K]."""
    o, k = w.shape
    s = scale_inv.float().repeat_interleave(BLOCK, 0)[:o].repeat_interleave(BLOCK, 1)[:, :k]
    return w.float() * s


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """uint8 [O, K/2] (two e2m1 per byte, low nibble = even column) with e8m0 [O, K/32] -> dtype [O, K]."""
    o, kh = packed.shape
    k = kh * 2
    idx = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(o, k).long()
    vals = FP4_VALUES[idx].view(o, k // MX_BLOCK, MX_BLOCK)
    exp = scale.to(torch.int32) - 127
    vals = torch.ldexp(vals, exp.unsqueeze(-1).to(torch.float32))
    return vals.reshape(o, k).to(dtype)
