# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Sequence-parallel (SP) reshard helpers for the HunyuanImage-3.0 backbone.
#
# SP splits the (long, ~22.8k) token sequence across a mesh axis so per-token work
# (norms, projections, MoE, residuals) and the per-query attention output are
# divided across the row devices. Hunyuan attention uses a dense mixed mask, so
# the proven DiT ring-SDPA does NOT apply — instead we keep Q sequence-sharded and
# all-gather K/V (see attention.py). These helpers bridge the replicated <-> shard
# representations:
#
#   sp_shard:  replicated [.., D, ..] -> per-device [.., D/n, ..]  (scatter on mesh_axis)
#   sp_gather: per-device [.., D/n, ..] -> replicated [.., D, ..]   (all-gather on mesh_axis)
#
# sp_shard is built from reduce_scatter: feeding a tensor that is IDENTICAL across
# the axis makes reduce_scatter return n * (this device's chunk), so scaling by 1/n
# recovers the plain scatter. (No native "scatter replicated" collective is exposed
# by CCLManager; this is the standard construction.)
#
# NOTE (on-box bring-up): the sharded extent D/n must be tile-aligned (multiple of
# 32). The image-gen sequence is padded to a tile multiple upstream; confirm S/n is
# also tile-aligned on the box, else pad the sequence to n*TILE before sharding.

import ttnn

TILE = 32


def sp_shard(ccl, t: ttnn.Tensor, *, dim: int, mesh_axis: int, n: int, out_memory_config=None) -> ttnn.Tensor:
    """Replicated -> sequence-sharded along `dim` over `mesh_axis` (n = axis size).

    out_memory_config: placement for the rescaled output. Defaults to DRAM (the SDPA
    mask MUST stay DRAM). Pass L1 only for the `hidden` reshard, whose consumer is the
    input_layernorm — landing it L1-resident there avoids a separate DRAM->L1 copy.
    """
    if n == 1:
        return t
    scattered = ccl.reduce_scatter(t, dim=dim, mesh_axis=mesh_axis)  # [.., D/n, ..], = n * chunk
    out = ttnn.multiply(
        scattered, 1.0 / n, memory_config=out_memory_config or ttnn.DRAM_MEMORY_CONFIG
    )  # undo the n-fold sum from replicated input
    ttnn.deallocate(scattered)
    return out


def sp_gather(ccl, t, *, dim: int, mesh_axis: int, n: int, out_memory_config=None):
    """Sequence-sharded -> replicated (full) along `dim` over `mesh_axis`.

    out_memory_config: placement for the gathered output (default follows the
    all_gather default, i.e. the input's placement). Pass L1 to land the result
    L1-resident for its consumer.
    """
    if n == 1:
        return t
    gathered = ccl.all_gather(t, dim=dim, mesh_axis=mesh_axis, use_hyperparams=False)
    if out_memory_config is not None:
        gathered = ttnn.to_memory_config(gathered, out_memory_config)
    return gathered


# --- L1-residency seq gate for full-sequence MoE-internal ops -----------------
# The MoE expert body (gate_up/SwiGLU/down/combine) runs on the FULL gathered
# sequence (tokens are all-gathered before EP), so its L1 output buffers scale
# with the global sequence length. Past a sequence bound those buffers collide
# with the ops' own static circular-buffer region — a low-address CB clash
# (tt_metal program.cpp validate_circular_buffer_region), NOT an interleaved-bank
# OOM: at the clash L1 is only ~15% full. The bound was MEASURED on hardware
# (2x2 mesh, sp=2/tp=2, bf8 experts, 2 layers): the fused SwiGLU multiply
# (moe/moe_parallel.py) is the first op to clash — the last L1-resident forward
# is S=2560, the first clash S=2816. Keep the full-S MoE ops L1-resident at or
# below the bound (fast, no DRAM round-trip) and fall back to DRAM above it so
# long sequences run instead of crashing.
MOE_L1_MAX_SEQ = 2560


def moe_full_seq_mem_config(seq_len: int) -> ttnn.MemoryConfig:
    """L1 for full-sequence MoE-internal activations up to the measured CB-clash
    bound, else DRAM. `seq_len` is the FULL (gathered, global) sequence — MoE runs
    unsharded, so this is the global ISL, not the per-device shard."""
    return ttnn.L1_MEMORY_CONFIG if seq_len <= MOE_L1_MAX_SEQ else ttnn.DRAM_MEMORY_CONFIG


# --- L1-residency seq gate for per-device (sp-sharded) residual-stream / attention
# ops (input/post-attn RMSNorm, residual adds, QKV/create-heads/RoPE/SDPA/concat/
# o_proj). These run on the PER-DEVICE sequence Sd = S/sp_factor. Above a sequence
# bound their (or a co-resident live tensor's) L1 buffers collide with an op's
# static CB region — the same CB clash as MoE, but the resident tensor here is the
# LIVE residual stream, so ALL these ops must fall to DRAM together above the bound
# (moving one leaves another in the CBs' path). Bound measured on hardware: the
# residual RMSNorm first clashes at global ISL 4096 (Sd=2048); gate at the last
# jointly-validated L1 point (Sd=1280, i.e. global 2560, matching MOE_L1_MAX_SEQ).
RESID_L1_MAX_SEQ = 1280


def resid_mem_config(seq_len: int) -> ttnn.MemoryConfig:
    """L1 for a per-device (sp-sharded) residual-stream/attention activation up to
    the measured CB-clash bound, else DRAM. `seq_len` is the PER-DEVICE sequence
    Sd = x.shape[1] (NOT the global ISL)."""
    return ttnn.L1_MEMORY_CONFIG if seq_len <= RESID_L1_MAX_SEQ else ttnn.DRAM_MEMORY_CONFIG
