# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Pixtral vision sequence chunk size for L1-bound matmuls (attention + MLP).

from __future__ import annotations

import os

import ttnn

from models.common.utility_functions import nearest_32

VISION_L1_MEMCFG = ttnn.L1_MEMORY_CONFIG


def vision_activation_memcfg(num_elements: int) -> ttnn.MemoryConfig:
    """L1 for vision ops when activation fits L1; else DRAM (avoids SDPA/CB clashes on large images)."""
    cap_raw = os.environ.get("PIXTRAL_VISION_L1_SEQ_CAP", "4096")
    cap = max(32, nearest_32(int(cap_raw)))
    if int(num_elements) <= cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def vision_slice_memcfg(num_patches: int) -> ttnn.MemoryConfig:
    """L1 for patch-grid slice/reshape only on small grids (agent full-res images stay DRAM)."""
    cap_raw = os.environ.get("PIXTRAL_VISION_L1_SLICE_CAP", "1024")
    cap = max(32, nearest_32(int(cap_raw)))
    if int(num_patches) <= cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def vision_seq_memcfg(seq_len: int, feature_dim: int = 1) -> ttnn.MemoryConfig:
    """L1 for seq×feature activations (norm/MLP/QKV) when seq and element volume fit L1."""
    seq_cap_raw = os.environ.get("PIXTRAL_VISION_L1_SEQ_CAP", "4096")
    seq_cap = max(32, nearest_32(int(seq_cap_raw)))
    elem_cap_raw = os.environ.get("PIXTRAL_VISION_L1_ELEM_CAP", "1048576")
    elem_cap = max(seq_cap, int(elem_cap_raw))
    elements = int(seq_len) * max(1, int(feature_dim))
    if int(seq_len) <= seq_cap and elements <= elem_cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def vision_rms_norm_memcfg(seq_len: int, feature_dim: int = 1) -> ttnn.MemoryConfig:
    """Always DRAM: RMSNorm static CBs overlap L1 RoPE / slice buffers on WH (program ~225 clash)."""
    _ = seq_len, feature_dim
    return ttnn.DRAM_MEMORY_CONFIG


def vision_rope_memcfg(seq_len: int, head_dim: int = 1) -> ttnn.MemoryConfig:
    """L1 for RoPE embed + rotary_embedding when seq×head fits (RMSNorm stays DRAM)."""
    rope_cap_raw = os.environ.get("PIXTRAL_VISION_L1_ROPE_SEQ_CAP", os.environ.get("PIXTRAL_VISION_L1_SEQ_CAP", "4096"))
    rope_cap = max(32, nearest_32(int(rope_cap_raw)))
    elem_cap_raw = os.environ.get(
        "PIXTRAL_VISION_L1_ROPE_ELEM_CAP",
        os.environ.get("PIXTRAL_VISION_L1_ELEM_CAP", "1048576"),
    )
    elem_cap = max(rope_cap, int(elem_cap_raw))
    elements = int(seq_len) * max(1, int(head_dim))
    if int(seq_len) <= rope_cap and elements <= elem_cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def vision_collective_memcfg(seq_len: int, feature_dim: int = 1) -> ttnn.MemoryConfig:
    """L1 for all_gather/fast_reduce when seq×shard fits (norms/RoPE stay DRAM to avoid CB clash)."""
    ag_cap_raw = os.environ.get("PIXTRAL_VISION_L1_AG_SEQ_CAP", os.environ.get("PIXTRAL_VISION_L1_SEQ_CAP", "4096"))
    ag_cap = max(32, nearest_32(int(ag_cap_raw)))
    elem_cap_raw = os.environ.get(
        "PIXTRAL_VISION_L1_AG_ELEM_CAP",
        os.environ.get("PIXTRAL_VISION_L1_ELEM_CAP", "1048576"),
    )
    elem_cap = max(ag_cap, int(elem_cap_raw))
    elements = int(seq_len) * max(1, int(feature_dim))
    if int(seq_len) <= ag_cap and elements <= elem_cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def pixtral_vision_seq_chunk_len(configuration) -> int:
    """L1 matmul seq chunk (env PIXTRAL_VISION_MM_SEQ_CHUNK or min(cfg, cap))."""
    force = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK")
    if force is not None and str(force).strip() != "":
        chunk = max(32, nearest_32(int(force)))
    else:
        cap_raw = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK_CAP", "1024")
        cap = max(32, nearest_32(int(cap_raw)))
        cfg_chunk = getattr(configuration, "VISION_MAX_MM_SEQ", cap)
        if cfg_chunk is None:
            cfg_chunk = cap
        chunk = max(32, min(int(cfg_chunk), int(cap)))

    return chunk


def pixtral_effective_mm_seq_len(configuration, seq_len: int) -> int:
    """Matmul M: one kernel over full ``seq_len`` when it fits L1, else ``pixtral_vision_seq_chunk_len``."""
    chunk = pixtral_vision_seq_chunk_len(configuration)
    force = os.environ.get("PIXTRAL_VISION_MM_FULL_SEQ_CAP")
    if force is not None and str(force).strip() != "":
        full_cap = max(32, nearest_32(int(force)))
    else:
        full_cap = max(chunk, 1024)
    if seq_len <= full_cap:
        return seq_len
    return chunk


def pad_seq_to_chunk_multiple(x: ttnn.Tensor, seq_len: int, chunk: int) -> tuple[ttnn.Tensor, int, int]:
    """Pad dim=2 to a multiple of ``chunk`` so batched matmul avoids ``ttnn.concat``."""
    original = seq_len
    if seq_len <= chunk or seq_len % chunk == 0:
        return x, seq_len, original
    padded = ((seq_len + chunk - 1) // chunk) * chunk
    pad_len = padded - seq_len
    x = ttnn.pad(x, padding=[(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)
    return x, padded, original


def trim_seq_dim2(x: ttnn.Tensor, original_seq_len: int) -> ttnn.Tensor:
    if int(x.shape[2]) == original_seq_len:
        return x
    return x[:, :, :original_seq_len, :]
