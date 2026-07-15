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
# OOM: at the clash L1 is only ~15% full. Bound MEASURED on hardware (2x2 mesh,
# sp=2/tp=2, bf8 experts): full-S MoE ops clash near S~2816. Gate a 256-step below
# so the exact-boundary value is itself safe — a coarse-sweep "last-good" is NOT
# safe on every path: the 32-layer sp=2 PCC forward clashed at exactly S=2560.
MOE_L1_MAX_SEQ = 2048


def moe_full_seq_mem_config(seq_len: int) -> ttnn.MemoryConfig:
    """L1 for full-sequence MoE-internal activations up to the measured CB-clash
    bound, else DRAM. `seq_len` is the FULL (gathered, global) sequence — MoE runs
    unsharded, so this is the global ISL, not the per-device shard."""
    return ttnn.L1_MEMORY_CONFIG if seq_len <= MOE_L1_MAX_SEQ else ttnn.DRAM_MEMORY_CONFIG


def _largest_divisor_leq(x: int, cap: int) -> int:
    """Largest d in [1, cap] with x % d == 0."""
    for d in range(min(x, cap), 0, -1):
        if x % d == 0:
            return d
    return 1


def wide_mm_program_config(device, M: int, K: int, N: int):
    """2D-multicast matmul config for a large-M (prefill) projection.

    ttnn's auto-heuristic mis-schedules some large-M bf16 x bf8 shapes onto all 110
    worker cores at ~3%% FLOP / ~9 GB/s (measured: the QKV/o_proj/shared-MLP linears
    ran 900-1350 us each, vs the routed-expert matmul of the SAME 512x3072x4096 shape
    at 147 us / 49%% FLOP on a rectangular 64-core grid). This returns an explicit 2D
    MultiCast config on a rectangular grid where M and N tiles divide evenly across
    the grid rows/cols — the schedule that the fast expert matmuls already use — so
    the projection is deterministic regardless of the input tensor's memory state.

    Returns None (=> auto) unless every dim is tile-aligned and M is more than one
    tile (the single-M-tile decode path uses the 1D config instead), so it is always
    safe to pass at any tp_factor / seq.
    """
    if M % TILE or K % TILE or N % TILE:
        return None
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    if Mt <= 1:  # single-M-tile (decode) shape: use the 1D config, not this
        return None

    grid = device.compute_with_storage_grid_size()
    # 2D mcast tiles M across grid rows (y) and N across grid cols (x). Pick the
    # largest grid dims that divide Mt/Nt evenly so no core is left idle or ragged.
    gy = _largest_divisor_leq(Mt, grid.y)
    gx = _largest_divisor_leq(Nt, grid.x)
    per_core_M = Mt // gy
    per_core_N = Nt // gx

    # out_subblock_h * out_subblock_w <= 4 (dest register tiles). Prefer wide subblocks.
    out_subblock_w = next((w for w in (4, 3, 2, 1) if per_core_N % w == 0), 1)
    out_subblock_h = next((hh for hh in range(4 // out_subblock_w, 0, -1) if per_core_M % hh == 0), 1)

    # K reduction chunk: a divisor of Kt, capped so the in0/in1 blocks stay L1-sized.
    in0_block_w = _largest_divisor_leq(Kt, 4)

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def decode_mm_program_config(device, M: int, K: int, N: int):
    """1D-multicast config for a single-M-tile (decode) projection: broadcasts the
    one M-tile row of activations and splits N across the widest core count that
    divides Nt. Complements wide_mm_program_config (which handles Mt > 1). Measured
    ~1.3-1.55x vs auto on the single-token expert matmuls
    (tests/perf/test_expert_down_sweep.py).

    Returns None (=> auto) unless every dim is tile-aligned and Mt == 1, so callers
    can pass it for any M and only the decode shape is affected.
    """
    if M % TILE or K % TILE or N % TILE:
        return None
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    if Mt != 1:
        return None
    grid = device.compute_with_storage_grid_size()
    ncols = _largest_divisor_leq(Nt, grid.x * grid.y)
    per_core_n = Nt // ncols
    osw = next((w for w in (4, 3, 2, 1) if per_core_n % w == 0), 1)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=_largest_divisor_leq(Kt, 8),
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=Mt,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


# --- L1-residency seq gate for per-device (sp-sharded) residual-stream / attention
# ops (input/post-attn RMSNorm, residual adds, QKV/create-heads/RoPE/SDPA/concat/
# o_proj). These run on the PER-DEVICE sequence Sd = S/sp_factor. Above a sequence
# bound their (or a co-resident live tensor's) L1 buffers collide with an op's
# static CB region — the same CB clash as MoE, but the resident tensor here is the
# LIVE residual stream, so ALL these ops must fall to DRAM together above the bound
# (moving one leaves another in the CBs' path). Bound measured on hardware: the
# residual RMSNorm clashed at Sd=1280 (global 2560) on the 32-layer sp=2 forward,
# so gate below that at Sd=1024 (global 2048, matching MOE_L1_MAX_SEQ).
RESID_L1_MAX_SEQ = 1024


def resid_mem_config(seq_len: int) -> ttnn.MemoryConfig:
    """L1 for a per-device (sp-sharded) residual-stream/attention activation up to
    the measured CB-clash bound, else DRAM. `seq_len` is the PER-DEVICE sequence
    Sd = x.shape[1] (NOT the global ISL)."""
    return ttnn.L1_MEMORY_CONFIG if seq_len <= RESID_L1_MAX_SEQ else ttnn.DRAM_MEMORY_CONFIG
