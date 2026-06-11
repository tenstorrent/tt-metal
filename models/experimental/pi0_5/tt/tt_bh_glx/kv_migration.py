# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer-paired prefill -> denoise KV migration via host bounce.

For 18 prefill layers and 6 denoise chips, layers 3*i, 3*i+1, 3*i+2 from
prefill end up grouped on denoise chip i — the same 3 layers whose
cross-attention K/V that denoise chip's expert blocks will read.
"""

from __future__ import annotations

from typing import List, Tuple

import ttnn

from . import stages
from .transport import send_via_host


def migrate_layer_paired(
    per_layer_kv,  # List[(K, V)] of length VLM_TOTAL_LAYERS, each on prefill_per_chip[i]
    denoise_per_chip,  # List of 6 denoise submeshes
) -> List[List[Tuple]]:
    """Returns prefix_kv_per_denoise_chip[chip_idx] = [(K_lo, V_lo), (K_lo+1, V_lo+1), (K_lo+2, V_lo+2)].

    Each list has EXPERT_LAYERS_PER_CHIP entries (=3). lo = chip_idx * 3.
    """
    if len(per_layer_kv) != stages.VLM_TOTAL_LAYERS:
        raise RuntimeError(f"expected {stages.VLM_TOTAL_LAYERS} per-layer KV tuples, got {len(per_layer_kv)}")
    if len(denoise_per_chip) != stages.DENOISE_NUM_CHIPS:
        raise RuntimeError(f"expected {stages.DENOISE_NUM_CHIPS} denoise chips, got {len(denoise_per_chip)}")

    out: List[List[Tuple]] = []
    n_per = stages.EXPERT_LAYERS_PER_CHIP
    for chip_idx, dst in enumerate(denoise_per_chip):
        chip_kv: List[Tuple] = []
        lo = chip_idx * n_per
        for j in range(n_per):
            k_src, v_src = per_layer_kv[lo + j]
            k_dst = send_via_host(k_src, dst)
            v_dst = send_via_host(v_src, dst)
            # The expert's per-step k_rope is bfloat8_b (xqkv linear emits bf8_b);
            # cross-attention concat([past_k, k_rope]) requires matching dtype on
            # MeshDevice paths (the single-Device path is lenient). VLM prefill
            # emits bf16 KV — convert here so prefix and current K match.
            if k_dst.dtype != ttnn.bfloat8_b:
                k_dst = ttnn.typecast(k_dst, ttnn.bfloat8_b)
            if v_dst.dtype != ttnn.bfloat8_b:
                v_dst = ttnn.typecast(v_dst, ttnn.bfloat8_b)
            chip_kv.append((k_dst, v_dst))
        out.append(chip_kv)
    return out
