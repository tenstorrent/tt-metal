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

    L1 guard: this 2D config has no out_block sub-tiling — each core materializes its
    full per_core_M x per_core_N output block plus the double-buffered in0/in1 blocks
    in L1. At large Mt (e.g. the T2I denoise QKV proj at seq~10.5k -> Mt~328) the
    per-core block exceeds the 1.5 MB L1 (a hard TT_THROW in program.cpp CB validation),
    so if no L1-fitting block is found we fall back to None (=> auto, which DOES
    out_block-iterate and handles arbitrary M, just slower on the mid-large range).
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

    # L1 guard (per_core cap): the CBs (in0/in1/out + fp32 partials + mcast buffers)
    # scale with the per-core work per_core_M x per_core_N, and this config has no
    # out_block sub-tiling. Exact byte modeling is unreliable (subblock/fp32-acc/mcast
    # internals) AND the real ceiling is lower in-graph than in isolation because the
    # resident expert weights / KV / residual stream already occupy L1 on every core
    # (measured: the T2I denoise o_proj at per_core_M=13 x per_core_N=16 = 208 tiles
    # fits in isolation but TT_THROWs at ~1.86 MB in the full forward). So cap the
    # per-core work conservatively — comfortably above the small mid-M expert/proj
    # blocks this is meant to accelerate (M<=2048 / Mt<=64 => <=128 tiles) but below
    # the in-graph-crashing 208 — and fall back to auto (which out_block-iterates and
    # is fine at large M) above it.
    #
    # Also cap per_core_M alone: skinny-N shapes (MoE gate N=64 => Nt=2, gx<=2) can
    # pass the product cap with a huge M-block. At ISL=22800 (SP-padded Mt=714) that
    # yields per_core_M=102 / product=102 and TT_THROWs with ~2.4 MB CBs > 1.5 MB L1.
    # Linear scale of that crash puts the safe per_core_M ceiling near ~66.
    PER_CORE_TILE_CAP = 160
    PER_CORE_M_CAP = 64
    if per_core_M * per_core_N > PER_CORE_TILE_CAP or per_core_M > PER_CORE_M_CAP:
        return None

    # K reduction chunk: a divisor of Kt, capped so the in0/in1 blocks stay small.
    in0_block_w = _largest_divisor_leq(Kt, 4)

    # out_subblock_h * out_subblock_w <= 4 (dest register tiles). Prefer wide subblocks.
    out_subblock_w = next((w for w in (4, 3, 2, 1) if per_core_N % w == 0), 1)
    out_subblock_h = next((hh for hh in range(4 // out_subblock_w, 0, -1) if per_core_M % hh == 0), 1)

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
    """1D-multicast config for a small-M (decode / mid-M) projection.

    Complements wide_mm_program_config (LARGE-M / Mt>=8). Two regimes, measured on BH:

      * Wide N (Nt > 4): mcast activations, split N across cores (mcast_in0=True).
        Mt 1-7: 1.3-1.83x vs auto (tests/perf/test_expert_down_sweep.py;
        test_matmul_shard_sweep.py — L1-interleaved beats width/height/block-sharded acts).
      * Skinny N (Nt <= 4, e.g. MoE router 64x4096x64): split-N starves the grid
        (Nt=2 → 2 cores). Parallelize the K reduction instead (mcast_in0=False /
        gather-K). BH: split-N 25.4us → gather-K nc=8 17.7us (1.43x).

    Returns None (=> auto) unless every dim is tile-aligned and Mt <= 7.
    """
    if M % TILE or K % TILE or N % TILE:
        return None
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    if Mt < 1 or Mt >= 8:
        return None
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y

    # Skinny-N: gather-K. Prefer nc=8 (BH MoE-gate sweet spot: 17.7us vs
    # split-N 25.4us); larger nc (32) regresses toward the split-N floor.
    if Nt <= 4:
        for nc in (8, 16, 4, 32, 2):
            if nc > max_cores or Kt % nc != 0:
                continue
            gy = _largest_divisor_leq(nc, grid.y)
            while gy > 0 and nc % gy != 0:
                gy -= 1
            if gy < 1:
                continue
            gx = nc // gy
            if gx > grid.x:
                continue
            osw = next((w for w in (4, 3, 2, 1) if Nt % w == 0), 1)
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=Kt // nc,
                out_subblock_h=1,
                out_subblock_w=osw,
                per_core_M=Mt,
                per_core_N=Nt,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )

    ncols = _largest_divisor_leq(Nt, max_cores)
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
# residual RMSNorm clashed at Sd=1280 (global 2560) on the 32-layer sp=2 forward.
# Gate strictly below 1024: AR KV-prefill uses sp=1 with chunk_size=1024, so the
# per-device seq IS 1024 — and at exactly that bound the interleaved rms_norm CBs
# clash with the L1 residual (program.cpp validate_circular_buffer_region). Exclusive
# upper bound keeps decode (Sd=32) and SP=2 prefill (Sd=512) in L1.
RESID_L1_MAX_SEQ = 1024


def resid_mem_config(seq_len: int) -> ttnn.MemoryConfig:
    """L1 for a per-device (sp-sharded) residual-stream/attention activation up to
    the measured CB-clash bound, else DRAM. `seq_len` is the PER-DEVICE sequence
    Sd = x.shape[1] (NOT the global ISL)."""
    return ttnn.L1_MEMORY_CONFIG if seq_len < RESID_L1_MAX_SEQ else ttnn.DRAM_MEMORY_CONFIG


# --- Block-sharded RMSNorm config for the small-M (decode-ish) residual-stream norm.
# The interleaved ttnn.rms_norm parallelizes over ROW-tiles (M/32), so at small M it
# lands on few cores and serializes the H-wide reduction: the Hunyuan input/post-attn/
# ln_f norms are 62us on a SINGLE core at Sd=32 (1 row-tile) in the recaption AR report.
# We block-shard: the H=4096 (128-tile) hidden across gx=8 cores (parallelizes the
# reduction) AND the M row-tiles across gy=grid.y cores (so each core holds few rows).
# The 2-D split is what fixes the naive 1xgx shard's L1 OOM at Mt>=8: piling all M rows
# onto 8 cores makes each shard [M, H/8] blow past the 1.46MB L1 bank as Sd grows, so
# instead we keep block_h (=Mt/gy) at ~1 by using the gy grid dim for rows.
#
# Gate is MEASURED (tests/perf/test_rmsnorm_shard_sweep.py, Blackhole 2x2, HiFi2 +
# fp32_dest_acc_en=True, incl. the I2S-in/S2I-out reshards the caller pays):
#   * per-core tile budget block_h*block_w <= NORM_SHARD_MAX_TILES_PER_CORE keeps the
#     norm's (many, fp32) circular buffers inside L1 ([1,16]=16 tiles fits; [8,16]=128
#     OOMs even in isolation). Large-Mt norms (prefill/denoise) exceed it => interleaved,
#     which already spreads row-tiles across cores there (flat ~63us to Mt=64) — no loss.
#   * QK-norm (H=head_dim=128 => Ht=4) can't host gx>=8 => None (left interleaved, ~4us).
# fp32_dest_acc_en=True (mandatory for norm precision) caps subblock_w<=4.
NORM_SHARD_MAX_TILES_PER_CORE = 32  # block_h*block_w ceiling; MEASURED-safe (see sweep).
# Decode-only: I2S+sharded CBs fight live L1 residuals once M grows into prefill
# chunks (even when the per-core tile budget still fits, e.g. Sd=512 at SP=2).
NORM_SHARD_MAX_M = 64


def rmsnorm_shard_config(device, M: int, H: int):
    """Return (program_config, shard_memory_config) to run a 2-D block-sharded rms_norm
    for a small-M / wide-H norm, or None => keep the interleaved kernel. Safe to call for
    any norm shape: returns None outside the measured-safe regime (large M, narrow QK)."""
    if M > NORM_SHARD_MAX_M or M % TILE or H % TILE:
        return None
    Mt, Ht = M // TILE, H // TILE
    grid = device.compute_with_storage_grid_size()
    # Width: largest divisor of Ht in [8, grid.x] — need real width parallelism; <8 (e.g.
    # the 4-tile QK-norm) isn't worth resharding and 8 is the validated width-core count.
    gx = next((g for g in range(min(grid.x, Ht), 7, -1) if Ht % g == 0), None)
    if gx is None:
        return None
    block_w = Ht // gx
    # Rows: most grid.y cores that evenly divide Mt while keeping per-core tiles in budget.
    gy = next(
        (
            g
            for g in range(min(grid.y, Mt), 0, -1)
            if Mt % g == 0 and (Mt // g) * block_w <= NORM_SHARD_MAX_TILES_PER_CORE
        ),
        None,
    )
    if gy is None:
        return None  # even the fewest rows/core exceed the L1 budget => interleaved
    block_h = Mt // gy
    subblock_w = next((w for w in (4, 3, 2, 1) if block_w % w == 0), 1)  # fp32 caps <=4
    shard_mc = ttnn.create_sharded_memory_config(
        shape=(1, 1, M, H),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )
    return program_config, shard_mc
