# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 latent KV cache rows (576 = 512 latent + 64 rope) per layer: row formats and their byte geometry (CPU), plus the device cache object.

Formats (kv-design/DESIGN.md s0, s2): ``bf16`` 1152 B/row; ``scaled_fp8`` = 512 e4m3 + 4 fp32 scales (one per 128) + 64 bf16 rope = 656 B packed,
stored 704 B/row (Blackhole DRAM alignment 64 B: the stock op strides rows by the aligned page size).
"""
from dataclasses import dataclass

import torch

LATENT, ROPE = 512, 64
ROW = LATENT + ROPE
SCALE_BLOCK = 128
DRAM_ALIGN = 64


def row_bytes(fmt: str, aligned: bool = True) -> int:
    n = {"bf16": ROW * 2, "scaled_fp8": LATENT + (LATENT // SCALE_BLOCK) * 4 + ROPE * 2}[fmt]
    return -(-n // DRAM_ALIGN) * DRAM_ALIGN if aligned else n


def pack_scaled_fp8_host(kv):
    """host reference of the stock packed row: kv [T,576] (bf16/fp32) -> (packed uint8 [T,656], dequantised bf16 [T,576]); one scale per 128 latent values = 2^ceil(log2(amax/448))."""
    T = kv.shape[0]
    lat = kv[:, :LATENT].float().reshape(T, LATENT // SCALE_BLOCK, SCALE_BLOCK)
    amax = lat.abs().amax(dim=-1).clamp(min=1e-12)
    sc = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q8 = (lat / sc[..., None]).to(torch.float8_e4m3fn)
    deq = (q8.float() * sc[..., None]).reshape(T, LATENT)
    packed = torch.empty(T, 656, dtype=torch.uint8)
    packed[:, :LATENT] = q8.reshape(T, LATENT).view(torch.uint8)
    packed[:, LATENT : LATENT + 16] = sc.contiguous().view(torch.uint8).reshape(T, 16)
    packed[:, LATENT + 16 :] = kv[:, LATENT:].to(torch.bfloat16).contiguous().view(torch.uint8).reshape(T, 128)
    return packed, torch.cat([deq.to(torch.bfloat16), kv[:, LATENT:].to(torch.bfloat16)], dim=-1)


@dataclass
class KVCache:
    """Device cache of ONE layer for ONE user per mesh row: ``rows`` [1,1,T,W] ROW_MAJOR (bf16 W=576, scaled_fp8 W=656 fp8_e4m3), replicated over the 8 columns
    (v0: every column holds the whole context; the kv-design seq-shard over columns is a later change behind this class). ``length`` = valid rows.
    """

    fmt: str
    rows: object = None
    length: int = 0
