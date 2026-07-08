# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer-paired prefill → denoise KV migration via fabric sockets.

For 18 prefill layers and 6 denoise chips, layers 3*i, 3*i+1, 3*i+2 from
prefill end up grouped on denoise chip i — the same 3 layers whose
cross-attention K/V the denoise chip's expert blocks read.

v2 uses ``SocketTransport.send`` (direct-write via fabric) — zero host
bounce, no ``ttnn.synchronize_device``. The bf16→bf8_b typecast (from v1
host-bounce era) is kept for now: the expert's per-step k_rope is bf8_b
and ``ttnn.concat([past_k, k_rope])`` on MeshDevice still requires
matching dtype. Recovering the ~3 PCC points the typecast costs is a
follow-up (typecast k_rope UP to bf16 inside attention instead).
"""

from __future__ import annotations

from typing import List, Tuple

import ttnn

from . import stages


def migrate_layer_paired(
    per_layer_kv,  # List[(K, V)] of length VLM_TOTAL_LAYERS, each on prefill_per_chip[i]
    denoise_per_chip,  # List of 6 denoise submeshes
    transport=None,
    to_l1=False,  # place the migrated KV in L1 (denoise reads it every step)
) -> List[List[Tuple]]:
    """Returns prefix_kv_per_denoise_chip[chip_idx] = [(K_lo, V_lo), (K_lo+1, V_lo+1), (K_lo+2, V_lo+2)].

    Each list has EXPERT_LAYERS_PER_CHIP entries (=3). lo = chip_idx * 3.
    """
    if len(per_layer_kv) != stages.VLM_TOTAL_LAYERS:
        raise RuntimeError(f"expected {stages.VLM_TOTAL_LAYERS} per-layer KV tuples, got {len(per_layer_kv)}")
    if len(denoise_per_chip) != stages.DENOISE_NUM_CHIPS:
        raise RuntimeError(f"expected {stages.DENOISE_NUM_CHIPS} denoise chips, got {len(denoise_per_chip)}")
    if transport is None:
        from .transport import SocketTransport

        transport = SocketTransport()

    out: List[List[Tuple]] = []
    n_per = stages.EXPERT_LAYERS_PER_CHIP
    for chip_idx, dst in enumerate(denoise_per_chip):
        chip_kv: List[Tuple] = []
        lo = chip_idx * n_per
        for j in range(n_per):
            k_src, v_src = per_layer_kv[lo + j]
            # Tag K vs V so they get separate cached receiver buffers (else
            # the V send would overwrite the K buffer between the same chip pair).
            # Tag also includes the local-layer index since each denoise chip
            # holds 3 KV-pairs and they each need their own bufs.
            k_dst = transport.send(k_src, dst, tag=f"kv:{j}:K")
            v_dst = transport.send(v_src, dst, tag=f"kv:{j}:V")
            # Match expert's per-step k_rope dtype (bf8_b) so the cross-attn
            # concat([past_k, k_rope]) doesn't trigger MeshDevice's strict
            # dtype check. See module docstring. When to_l1, land the KV in L1
            # so the expert reads it on-chip every step (the typecast already
            # produces a fresh tensor; the rare already-bf8 case takes a copy
            # so the cached DRAM recv buffer is left intact for reuse).
            kv_mc = ttnn.L1_MEMORY_CONFIG if to_l1 else ttnn.DRAM_MEMORY_CONFIG
            if k_dst.dtype != ttnn.bfloat8_b:
                k_dst = ttnn.typecast(k_dst, ttnn.bfloat8_b, memory_config=kv_mc)
            elif to_l1:
                k_dst = ttnn.to_memory_config(k_dst, ttnn.L1_MEMORY_CONFIG)
            if v_dst.dtype != ttnn.bfloat8_b:
                v_dst = ttnn.typecast(v_dst, ttnn.bfloat8_b, memory_config=kv_mc)
            elif to_l1:
                v_dst = ttnn.to_memory_config(v_dst, ttnn.L1_MEMORY_CONFIG)
            chip_kv.append((k_dst, v_dst))
        out.append(chip_kv)
    return out
