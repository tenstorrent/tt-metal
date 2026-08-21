# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP helpers for Qwen3.5/3.6 on Blackhole (9B single-device + 27B TP=4 / TP=8).

Used only when num_devices > 1. DRAM-sharded matmul cfgs, prefill progcfgs,
mesh shard/replicate, FP8 dequant, HF weight reorder for per-device sharding.
"""

import math
import os

import torch

import ttnn
from models.common.utility_functions import is_blackhole

# Hardware constants
TILE_SIZE = 32
DRAM_CORES = 8
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})


# Output-subblock ceilings on out_subblock_h * out_subblock_w, set by the DST register budget: an fp32
# destination tile takes two half-DST slots, so enabling fp32_dest_acc_en halves how many fit.
DST_TILES = 8
DST_TILES_FP32_ACC = 4


# Compute kernel configs
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# Same fidelity, fp32 destination accumulation OFF. Used ONLY by the GDN prefill in-projection (see
# create_prefill_kpass1_matmul_program_config). Turning fp32 dest acc off does two things for that
# matmul: it halves the intermediate CB, and it raises the output-subblock ceiling from 4 to 8. Both
# are what make a ONE-K-PASS blocking (out_block_w == per_core_N) fit L1 at all -- with fp32 dest acc
# on, every one-pass variant is rejected with "circular buffers grow to 1524288/1615808/1798848 B
# beyond max L1 size of 1499136 B".
#
# MEASURED (N150, M=2048 K=4096, DEVICE KERNEL DURATION; tests/perf/test_gdn_inproj_sweep.py):
#     N=6912 cols=8 sub_w=3 blk_w=9  fp32_acc ON   3 K-passes  1493us   <- previous config
#     N=6176 cols=8 sub_w=5 blk_w=25 fp32_acc OFF  1 K-pass    1255us   -16.0%
# The accuracy cost is small but real: PCC against an fp32 torch reference goes 0.99997 -> 0.99992.
# Deliberately a SEPARATE constant rather than a change to COMPUTE_HIFI2: that config is shared by the
# MLP down-proj and both attention projections, which were tuned with fp32 dest accumulation on and
# are not covered by the sweep above.
COMPUTE_HIFI2_NO_FP32_ACC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


# LoFi + no fp32 dest acc. For the two ATTENTION prefill matmuls (QKV in-proj, wo out-proj) on
# Wormhole, paired with create_prefill_kpass1_matmul_program_config exactly like
# COMPUTE_HIFI2_NO_FP32_ACC is (the one-K-pass blocking still requires fp32_dest_acc_en=False).
#
# WHY LoFi IS FREE HERE. HiFi2 costs ~2x LoFi's math passes per tile, and it was buying almost
# nothing on these two shapes because both take a BFLOAT8_B *weight*, whose 8-bit mantissa already
# dominates the product's error -- HiFi2 was paying for precision the operands cannot represent.
# MEASURED (device kernel duration + PCC vs an fp32 torch reference, N300, M=2048, one K pass,
# tests/perf/test_all_matmuls_sweep.py):
#     qkv 2048x4096x5120  HiFi2 1007.8us pcc=0.99992  ->  LoFi 887.3us pcc=0.99985   -12.0%
#     wo  2048x2048x4096  HiFi2  402.6us pcc=0.99987  ->  LoFi 328.9us pcc=0.99981   -18.3%
# i.e. ~7e-5 of PCC for 12-18% of the op. The same sweep confirms 8 columns is already the best grid
# width for both (fewer columns loses more in cores than it gains in subblock), so grid is not a
# further lever here.
#
# Kept SEPARATE from COMPUTE_HIFI2_NO_FP32_ACC rather than changing it: that constant is also the
# GDN in-projection's, whose accuracy budget was tuned at HiFi2 and is NOT covered by this sweep.
COMPUTE_LOFI_NO_FP32_ACC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def sdpa_bf8_enabled(args):
    """QWEN_SDPA_BF8: bf8 Q/K/V + bf8 paged KV cache for chunked-prefill SDPA (faster, slightly
    lower precision). Single source of truth for both attention/tp.py's TPAttention._sdpa_bf8 and
    model.py's allocate_kv_caches — they must agree, since the cache dtype the KV is filled into has
    to match what forward_prefill_paged casts K/V to before the fill.

    Default ON for Wormhole N300 only. MEASURED (device kernel duration, N300, single full-attention
    decoder layer, S=2048, Qwen3.5-9B, on top of the kpass1 matmul + fused-qk-norm changes above):
        SDPA                    770us -> 494us   -36%
        wo_proj matmul         515us -> 404us   -21.6%  (cascades: attn output narrows to bf8 too)
        PagedFillCache x2      19+19us -> 11+10us
        NLPConcatHeads           64us ->   41us
        new Q/K/V typecast x3        -> +70us   (the cost of casting into this path)
    Net effect on that layer's total device time: -600us or so, stacking with the other WH-only
    fixes to a combined 12,457us -> 10,909us (-12.4%) for this profile.

    Off by default everywhere else (unvalidated on N150/T3K/P150x4/Blackhole from this host — the
    model_config.py comment this replaces called it out explicitly: "validate PCC at long ctx").

    The env var still overrides in EITHER direction if set: QWEN_SDPA_BF8=0 forces it off on N300,
    QWEN_SDPA_BF8=1 forces it on anywhere else (at the user's own risk/validation).

    ATTEMPTED AND REVERTED for the 27B at T3K TP=8 (2026-08-19). Prefill validated fine and the wins
    were large (below), but a bf8 cache BREAKS EVERY DECODE PATH:
        TT_FATAL: Input and cache tensors must have same dtype!
    5 of 20 test_model_tp cases failed -- contract, decode_batched[B8], decode_batched[B32],
    long_prefill, prefill_paged_slots[B8] -- i.e. every case that decodes after a prefill.
    ROOT CAUSE: the comment at attention/tp.py's paged_update_cache call claims the op "takes
    bf16/fp32 and casts to bf8 cache". It does NOT -- paged_update_cache enforces input.dtype ==
    cache.dtype, unlike paged_fill_cache (which accepts a bf8 cache with any input dtype, and is why
    PREFILL worked). Enabling this needs the decode path to cast K/V to bf8 before the update, which
    costs 2 typecasts per layer per token in the decode hot path and reduces decode precision -- so it
    needs its own measurement and gate, not just this flag.
    NOTE the 64k demo PASSED with this on, because it exercises prefill + traced decode through a path
    that did not hit the failing update; module tests passed too. Only test_model_tp caught it.

    WHAT IT WAS WORTH, once the decode blocker is fixed (all MEASURED at T3K TP=8, 27B).
    The N300 clause is left exactly as it was so the 9B's measured behaviour is byte-identical.
    MEASURED (T3K TP=8, 27B, layer3_fullattn, seq 2048):
        SDPA                    455 -> 413us
        PagedFillCache x2      11+11 -> 6+6us
        new K/V/Q typecasts          -> +27us
        block                  3,476 -> 3,409us
    That understates it badly, because at seq 2048 the cache is nearly empty. The real wins are
    (a) MEMORY -- 256k KV goes ~4.3GB -> ~2.15GB per device, and (b) DECODE, which reads the whole
    cache every token: MEASURED at 64k, 16.41 -> 17.48 tok/s (+6.5%) with TTFT 22.72 -> 20.94s.
    (c) and it UNBLOCKS the bf8 attention_norm gather -- paged_fill_cache asserts
        (input==FP32 || input==BF16 || cache==BFP8 || cache==BFP4), so a bf8 cache satisfies it
        whatever the K/V input dtype is. That gather is worth a further -443us plus -149us on the
        qkv matmul that inherits the bf8 in0 (see layer.py). It is the single largest win of the day
        and it is downstream of this flag.
    Long-context gate: test_demo_text[traced_64k] passes with content checks (needs
    --timeout=1800; pytest.ini's 300s default is too short for it).
    """
    env = os.environ.get("QWEN_SDPA_BF8")
    if env is not None:
        return env == "1"
    return not is_blackhole() and getattr(args, "device_name", None) == "N300"


def wh_9b_n300(args):
    """True only for Qwen3.5-9B on a Wormhole N300 -- the exact configuration the DECODE
    optimizations below were measured and PCC-validated on.

    Single source of truth for the scope of decode changes that were all measured on this one
    config and are unvalidated anywhere else:
      * ``TPAttention._make_heads_decode``'s ``skip_v_reshard`` (v goes straight from the head split
        into the paged-cache write)
      * ``model_config.kv_cache_write_{k,v}_shard_cfg`` + ``kv_cache_write_fused_enabled``
        (paged_fused_update_cache on disjoint K/V grids)
      * the fused sigmoid-multiply attention gate in ``TPAttention.forward_decode``
      * ``mlp_w1/w3_decode_1d_progcfg``'s num_cores=56 + fp32_dest_acc_en=False, paired with
        ``Qwen36MLP.compute_kernel_config_gateup_decode``
      * ``rope_permuted_enabled`` below (permuted-head_dim full-width RoPE)
      * ``emb_decode_memcfg`` (width-sharded L1 token embedding on a tile of indices)

    Three conditions, each load-bearing:
      * ``not is_blackhole()`` -- Blackhole has 1.84x the L1 and a taller grid, takes a different
        pad-first KV path (_WH_KV_PAD_NOTE) and a fused AGMM prefill path; every core count and
        blocking here was tuned against WH's 8x8 grid.
      * ``dim <= 4096`` -- the 9B. The 27B (dim 5120) has different per-device widths (its
        hidden_dim/tp is 2176, not 6144), so the swept core counts do not transfer, and it runs at
        TP=8 on T3K where the CCL/grid arithmetic differs. Gate on dim rather than model_name because
        HF_MODEL is often a hashed snapshot directory -- same reason ``_decode_tile_opt`` and
        ``_ab_gap_scoped`` do it this way.
      * ``device_name == "N300"`` -- 2-device Wormhole. The KV-write grid split needs 2*B/cols rows of
        worker grid, and the batch-derived shard grids were only checked against this mesh; N150
        (TP=1) never reaches the TP path at all and T3K (TP=8) re-shapes every one of these grids.

    Outside this scope every one of these falls back to the previously shipped behavior.
    """
    return not is_blackhole() and getattr(args, "dim", 0) <= 4096 and getattr(args, "device_name", None) == "N300"


def rope_permuted_enabled(args):
    """Permuted-head_dim full-width RoPE (attention/rope_tp.py's rope_channel_perm has the
    derivation). Reorders head_dim so ``rotary_embedding_hf``'s native full-width rotate-half
    pairing coincides with HF's partial one, collapsing the partial-rope slice/transpose/concat
    chain into one call. The permutation is folded into q_proj/k_proj/q_norm/k_norm at load time
    (attention/tp.py's load_attention_weights_tp), so it changes the WEIGHTS, not just an op
    sequence -- the ".rp" cache tag there keeps the two variants from ever aliasing on disk.

    ON for Wormhole 9B N300 (wh_9b_n300). No env var: this is the shipping path on that config,
    plus the geometric precondition that rope_head_dim < head_dim (Qwen3.5's partial rotary --
    with no unrotated tail to skip, the "partial" chain is already one op and there is nothing
    to collapse). Off everywhere else (unvalidated on N150/T3K/P150x4/Blackhole and on the 27B).

    MEASURED on a whole decode layer (tests/perf/test_attn_rope_permuted_sweep.py, N300, device
    profiler): 46 -> 38 programs and 704.5 -> 687.4 us at B=1; 49 -> 37 programs and 907.2 ->
    778.1 us (-14.2%) at B=32. Takes the decode RoPE section from 15 device ops to 7 per
    full-attention layer per token; prefill drops slice/slice/concat per Q and K.
    """
    return wh_9b_n300(args) and getattr(args, "rope_head_dim", 0) < getattr(args, "head_dim", 0)


# Grid helpers
def prefill_grid_default():
    """BH P150: (8,10); WH: (8,8). y capped at 10 on BH (grid_x=10 breaks matmul)."""
    return (8, 10) if is_blackhole() else (8, 8)


# Max grid COLUMNS a tuned prefill config may use. A Blackhole galaxy reports a 12-wide worker
# grid, but harvested P150s expose only 11, so tuning to 12 would not port. 11 x 10 = 110 cores.
PREFILL_MAX_COLS_PORTABLE = 11

# Why TP=8 wants different values (measured at S=2048, 27B, 1x8 Ring):
#   * widest_cols -- `_best_prefill_cols` ranks candidate widths by (out_subblock_w, cols), i.e.
#     subblock first. At TP=8 the halved N makes wide grids yield a small per_core_N and hence a
#     narrow subblock, so that ranking retreats to fewer columns and leaves cores idle. Measured
#     device time is monotonically decreasing in column count instead: attn_wo went 1944us @ 60
#     cores -> 700us @ 110, and mlp_gate 2943us @ 60 -> 1935us @ 110. So take the width.
#   * in0_block_w_divisor -- `min(cap, k_tiles // grid_x)` is a function of the per-device K, which
#     halves. attn_wo/gdn_out go k_tiles 48 -> 24 and `24 // 11 = 2`, but in0_block_w only has to
#     DIVIDE k_tiles, so a larger block is legal and much faster (attn_wo @ 11 cols, from the sweep:
#     bw2 786us, bw4 719us, bw6 700us, bw8 705us).
#
# in0_block_w_cap is L1-BOUND, NOT just a legality bound. in0_block_w sizes the in0 circular
# buffer, and `_wo_proj` / the MLP prefill arm write their OUTPUT to L1 (attention/tp.py:246,
# mlp.py:284) -- so the CBs and a resident L1 output tensor compete for the same 1536 KB. Measured
# on the real model: cap=8 overflows and test_model_tp_long_prefill dies with
#   "Statically allocated circular buffers in program N clash with L1 buffers on core range
#    [0-0 - 10-8]. L1 buffer allocated at 1314560 and static circular buffer region ends at 1372032"
# from attention/tp.py:241. A standalone per-op sweep CANNOT see this: in isolation the only L1
# tenant is the op under test, so it reports a win that the full model has no room for. Any future
# raise of this cap must be validated by test_model_tp_long_prefill, not by the sweep alone.
_PREFILL_TUNING = {
    4: dict(widest_cols=False, in0_block_w_divisor=False, in0_block_w_cap=4),
    8: dict(widest_cols=True, in0_block_w_divisor=True, in0_block_w_cap=4),
}


def prefill_tuning(num_devices):
    """Prefill matmul tuning for this TP; unknown TP falls back to the frozen TP=4 values."""
    return _PREFILL_TUNING.get(num_devices, _PREFILL_TUNING[4])


def _roundup(a, b):
    return b * math.ceil(a / b)


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def _find_grid(n_tiles, target=32):
    max_r, max_c = 8, 8
    possible = [k for k in range(1, max_r * max_c + 1) if n_tiles % k == 0]
    possible.sort(key=lambda x: abs(x - target))
    for cores in possible:
        for rows in range(1, max_r + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_c:
                    return rows, cols
    raise ValueError(f"Cannot find grid for {n_tiles} tiles")


# DRAM-sharded config builders
def create_dram_sharded_mem_config(k, n):
    """WIDTH_SHARDED DRAM memory config for a weight matrix [k, n]."""
    padded_n = _roundup(n, TILE_SIZE * DRAM_CORES)
    shard_spec = ttnn.ShardSpec(
        DRAM_GRID,
        (k, padded_n // DRAM_CORES),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def create_dram_sharded_matmul_program_config(m, k, n, num_cores=None):
    """DRAM-sharded matmul program config (decode, small M)."""
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE

    if num_cores is None:
        rows, cols = _find_grid(k_tiles)
        num_cores = rows * cols

    k_tiles_per_core = k_tiles // num_cores
    if k_tiles_per_core == 0:
        k_tiles_per_core = k_tiles
        num_cores = 1
    in0_block_w = _find_largest_divisor(k_tiles_per_core)
    per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fused_activation=None,
    )


def create_matmul_1d_decode_progcfg(m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8):
    """Explicit-grid 1D (mcast_in0) decode matmul progcfg on ~`num_cores` cores — small grids beat
    the ~80-core DRAM-sharded grid on the bandwidth-bound skinny decode matmuls. Weight must be interleaved.

    Grid is shaped WIDE-first (cols up to `grid_w`, the device worker-grid width — 11 on BH P150, 8 on
    WH): for a fixed core budget a wide-short grid shortens the in0 multicast column and beats a
    tall-narrow one (~2% on this matmul; see test_mlp_matmul_sweep wide1d_* vs forced1d_*). Default
    grid_w=8 preserves the legacy shaping for callers that don't pass the device width."""
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    # mcast_in0: every core streams the full K, so in0_block_w must divide the full k_tiles.
    per_core_k = _find_largest_divisor(k_tiles)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    cap = 4 if fp32_acc else 8  # fp32_dest_acc caps subblock area at 4
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=per_core_k,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def matmul_1d_decode(x, weight, decode_1d_progcfg, compute_cfg, out_memory_config=ttnn.L1_MEMORY_CONFIG):
    """Small-grid 1D (mcast_in0) decode matmul on an interleaved weight; interleaves the K-sharded
    activation first since mcast_in0 needs the full K per core. See test_mlp_matmul_sweep.

    WORMHOLE ONLY: compares memory_config BY VALUE (matches sharded_decode_matmul's already_sharded
    check) instead of by object identity. ttnn.to_memory_config can return a tensor that aliases the
    same underlying buffer as `x` even when it is a distinct Python object, so the original `is not`
    check can pass and then deallocate() the caller's live input out from under it -- reproduced as
    a hard segfault on a real WH device when `x` was already ttnn.L1_MEMORY_CONFIG. No current
    caller passes an already-interleaved x here (they all pass sharded activations, making this a
    real copy either way), so this is a no-op in practice on both architectures; kept WH-only rather
    than touching the BH code path, which is left exactly as measured/shipped there."""
    if is_blackhole():
        x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.linear(
            x_il,
            weight,
            compute_kernel_config=compute_cfg,
            program_config=decode_1d_progcfg,
            memory_config=out_memory_config,
        )
        if x_il is not x:
            ttnn.deallocate(x_il)
        return out
    already_il = x.memory_config() == ttnn.L1_MEMORY_CONFIG
    x_il = x if already_il else ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    out = ttnn.linear(
        x_il,
        weight,
        compute_kernel_config=compute_cfg,
        program_config=decode_1d_progcfg,
        memory_config=out_memory_config,
    )
    if not already_il:
        ttnn.deallocate(x_il)
    return out


def create_activation_shard_config(k):
    """WIDTH_SHARDED L1 activation config for a [*, k] activation."""
    k_tiles = k // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    width_per_core = k // num_cores
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width_per_core),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def decode_ids_for_embed(token_ids):
    """Host flatten of decode ids to ``[1, B]``.

    ``ttnn.embedding`` fused-tilize keys off last dim (must be a multiple of 32). Decode tensors
    arrive as ``[B, 1]``; leaving them that way keeps last dim 1 and the 1-core RM factory.
    Flatten on the host (logical numel), never via on-device reshape of a padded RM ``[B,1]``.
    """
    return token_ids.reshape(1, token_ids.numel())


def decode_embed(emb, tok, args):
    """Token embedding with the decode width-shard when the indices are a full tile.

    ``EmbeddingsDeviceOperation`` with interleaved output parallelizes over token tiles, so
    decode B=32 is 1 core / ~21us of serial DRAM gathers. Width-sharded L1 output splits
    across dim instead; measured 3.0us on N300 (test_embedding_decode_sweep.py) with PCC=1.0,
    and the pre-norm all-gather accepts that layout with no extra reshard.

    Only used when ``args.emb_decode_memcfg`` is set (wh_9b_n300) AND the token tensor already
    has a full tile on the last dim (``shape[-1] % 32 == 0`` and ``<= 32``). That is the fused-
    tilize precondition. Decode call sites flatten host tokens ``[B,1]`` -> ``[1,B]`` before
    ``from_torch`` so this fires at serving batch 32. Do not reshape ``[B,1]`` on device: RM
    page padding makes that view the first padded row (1 real id + zeros), not B ids.
    B=1 (last dim 1) stays DRAM interleaved.
    """
    mc = getattr(args, "emb_decode_memcfg", None)
    if mc is None:
        return emb(tok)
    last = int(tok.shape[-1])
    if last == 0 or last > TILE_SIZE or last % TILE_SIZE != 0:
        return emb(tok)
    return emb(tok, memory_config=mc)


# 2D prefill matmul config
def _get_out_subblock_w(per_core_n, out_subblock_h, max_hw=DST_TILES_FP32_ACC):
    """Widest out_subblock_w that divides per_core_n within the DST budget.

    max_hw is the out_subblock_h*out_subblock_w ceiling, which depends on the compute kernel config:
    DST_TILES_FP32_ACC (4) when fp32_dest_acc_en is on, DST_TILES (8) when it is off. The default is
    the conservative 4 because tp_common.COMPUTE_HIFI2 (the model's shared config) enables fp32 dest
    accumulation; only callers that pass a non-fp32-acc compute config may raise it.
    """
    for w in range(min(per_core_n, max_hw // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _full_grid_crs(grid):
    """Full-grid allowed_worker_cores for CCL-fused matmuls, which bypass ttnn::prim::matmul()'s normalize_program_config()."""
    gx, gy = grid
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def _safe_half_out_block_w(per_core_N, out_subblock_w):
    """Largest divisor of per_core_N, itself a multiple of out_subblock_w, that is <= per_core_N // 2.

    Halves (at least) the output/intermediate CB footprint vs. the default out_block_w=per_core_N.
    Falls back to out_subblock_w (always a valid divisor of per_core_N by construction) if no smaller
    multiple divides evenly."""
    half = max(1, per_core_N // 2)
    best = out_subblock_w
    for w in range(out_subblock_w, half + 1, out_subblock_w):
        if per_core_N % w == 0:
            best = w
    return best


def prefill_kpass_width(n, grid_size=None, max_waste=0.20):
    """Smallest width >= n whose TILE count minimises the prefill matmul's K-pass count.

    A 2D prefill matmul walks its per-core output in ceil(per_core_N / out_block_w) N-blocks and
    re-traverses K once per block -- re-reading a DRAM-resident in0 each time. MEASURED on N300
    (M=2048, K=4096, HiFi2 BF16 x BFP8, 8x8 grid) the cost tracks that pass count almost exactly and
    is NOT explained by the output subblock:

        N=6176 (193 tiles, prime) per_core_N=25 out_block_w=5   5 passes  2631us
        N=6528 (204 tiles)        per_core_N=26 out_block_w=2  13 passes  4194us  +59%
        N=6912 (216 tiles)        per_core_N=27 out_block_w=9   3 passes  1906us  -28%
        N=7168 (224 tiles)        per_core_N=28 out_block_w=4   7 passes  2938us  +12%

    The trap is that out_block_w must be a MULTIPLE of out_subblock_w, so _get_out_subblock_w's greedy
    "widest subblock" can force a narrow block and MORE passes -- N=7168 gets the ideal 1x4 subblock
    and is still slower than N=6912's 1x3. A prime tile count (193) is worst of all: its only divisors
    are 1/5/25, and 25 (one pass) overflows L1.

    So: pad the width until the tile count factors well. Candidates are multiples of grid_size[0] (so
    per_core_N divides exactly and every column of the grid is used), within max_waste extra tiles.
    Scored by (passes, waste) -- fewest K passes, then least padding. Returns n unchanged when nothing
    beats it, so callers can use it unconditionally.

    NOT CURRENTLY CALLED BY THE MODEL. Its one user was the GDN in-proj width, which now reaches ONE K
    pass unpadded via create_prefill_kpass1_matmul_program_config -- padding to improve the tile count's
    factorization only helps when out_block_w must divide per_core_N into several blocks. Kept because it
    is the right tool for any shape stuck on the multi-pass path (and see the gdn_qkvzab_pad_tiles note
    in model_config.py for how to restore the pad); tests/perf/test_gdn_inproj_sweep.py still sweeps it."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    cols = grid_size[0]
    n_tiles = math.ceil(n / TILE_SIZE)

    def passes_for(tiles):
        per_core_N = math.ceil(tiles / cols)
        sub_w = _get_out_subblock_w(per_core_N, 1)
        blk_w = _safe_half_out_block_w(per_core_N, sub_w)
        return math.ceil(per_core_N / blk_w)

    best = (passes_for(n_tiles), 0, n_tiles)
    for tiles in range(_roundup(n_tiles, cols), int(n_tiles * (1 + max_waste)) + 1, cols):
        cand = (passes_for(tiles), tiles - n_tiles, tiles)
        if cand < best:
            best = cand
    return best[2] * TILE_SIZE


def _largest_divisor_le(n, cap):
    """Largest divisor of n that is <= cap (1 if none, since 1 always divides)."""
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


# Longest prefill M the full-grid MLP config below is used at. 2048 is the production chunk-outer
# chunk size (demo/text_demo.py PREFILL_CHUNK, model.capture_prefill_trace_chunked), so the MLP never
# sees more than this in production -- longer prompts arrive as multiple 2048 chunks. Above it, both
# per_core_M and therefore every CB scale linearly and the one-K-pass + in0_block_w=8 blocking
# overflows L1: MEASURED at m=4096 (per_core_M 8 -> 16) the down-proj asks for 2,398,912 B against
# Wormhole's 1,499,136 B limit. Rather than de-tune the shape production actually runs, fall back to
# the shared factory above this length -- the same "gate it at the only size it was measured safe at"
# call gdn/tp.py makes for its L1 placement. The seq4096 case in
# tests/perf/test_profile_single_layer_prefill.py is a scaling probe, not a served shape.
PREFILL_FULL_GRID_MAX_M = 2048


def create_prefill_mlp_matmul_program_config_full_grid(
    m, k, n, grid_size=None, fused_activation=None, out_subblock_h=1
):
    """One-K-pass prefill MLP progcfg on the FULL grid width. 27B on Wormhole; see tt/mlp.py.

    Deliberately a separate factory rather than flags on create_prefill_mlp_matmul_program_config:
    the 9B's tuned prefill configs come out of that one and must stay byte-identical, and nothing
    but the 27B MLP's prefill arm calls this.

    Two measured departures from create_prefill_mlp_matmul_program_config (both at the 27B's TP=8
    shapes with a bf8 in0 -- tests/perf/test_mlp_prefill_matmul_sweep.py, numbers in tt/mlp.py):

      * The grid width is the full device width instead of _best_prefill_cols'. That heuristic
        maximises the output subblock, and at hidden_dim/tp = 68 tiles (17408/8/32, whose only
        divisors <= 8 are 1/2/4) it settles on 6 columns = 48 of 64 cores. More cores wins by a wide
        margin at this shape -- the subblock-first premise does not survive here. The 9B's
        hidden_dim/tp is a friendlier tile count and already lands on the full width, which is why
        this never showed up there.
      * in0_block_w is the largest divisor of K (in tiles) up to DST_TILES, not min(4, ...). Once
        ff_norm hands the MLP a bf8 activation the in0 CB is half its former size, so the deeper K
        block fits -- and it is worth -10% on gate/up. Kept a divisor of K because a partial final
        K block is not a legal blocking.

    out_subblock_h is NOT derived, it is passed in. At these two shapes the measured winners
    contradict every simple rule: gate (per_core_N=9) wants 1x3 while down (per_core_N=20) wants
    2x4, so neither "widest w" nor "largest h*w" predicts both. Callers pass what the sweep measured.

    out_subblock_h is a PREFERENCE, not a demand: it has to divide per_core_M, and per_core_M scales
    with the prefill length. The tuned values were measured at the production chunk (m=2048 ->
    per_core_M=8), but the same code runs short prompts and chunk tails: m=128 gives per_core_M=1,
    which no height above 1 divides. So clamp to the largest divisor of per_core_M at or below the
    request instead of asserting -- falling back to 1 is exactly the height the shared factory would
    have picked anyway, so a short prefill loses only this one (unmeasured at that length) tweak.
    """
    if grid_size is None:
        grid_size = prefill_grid_default()
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))
    out_subblock_h = _largest_divisor_le(per_core_M, out_subblock_h)
    k_tiles = math.ceil(k / TILE_SIZE)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=_largest_divisor_le(k_tiles, DST_TILES),
        out_subblock_h=out_subblock_h,
        # DST_TILES (8) not DST_TILES_FP32_ACC (4): this path is only reached with _CKC_MLP_KPASS1,
        # whose fp32_dest_acc_en=False is what raises the ceiling. See _get_out_subblock_w.
        out_subblock_w=_get_out_subblock_w(per_core_N, out_subblock_h, max_hw=DST_TILES),
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=per_core_N,  # one K pass
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


def create_prefill_matmul_program_config(
    m,
    k,
    n,
    grid_size=None,
    fused_activation=None,
    tuning=None,
    out_block_w=None,
    halve_out_block=False,
    max_subblock_hw=None,
):
    """2D prefill matmul progcfg (DRAM-interleaved).

    fused_activation in packer; sharded kernel rejects ttnn.linear(activation=...) with progcfg.
    tuning: a `_PREFILL_TUNING` entry (see `prefill_tuning`); None = the frozen TP=4 behavior.
    out_block_w: when set (< per_core_N), the output/intermediate CB only needs to hold one
    out_block_w-wide slice of the per-core output at a time instead of the full per_core_N width —
    same lever already used by build_mmrs_decode_state/matmul_reduce_scatter_prefill below.
    halve_out_block: auto-derive a safe halved out_block_w (see _safe_half_out_block_w) instead of
    passing one explicitly. Needed on grids that are already at their physical max (WH tops out at
    8x8=64 cores vs BH's 8x10=80, with a smaller per-core L1 budget besides), where per_core_M/N can't
    be shrunk further by adding more cores.
    max_subblock_hw: out_subblock_h*out_subblock_w ceiling. Defaults to the conservative
    DST_TILES_FP32_ACC (4), which is correct for the fp32-dest-acc COMPUTE_HIFI2 most callers pass.
    Callers using a compute config with fp32_dest_acc_en=False may pass DST_TILES (8) to get the
    wider output subblock their DST budget actually allows — see _get_out_subblock_w."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(
        per_core_N, out_subblock_h, max_hw=max_subblock_hw if max_subblock_hw is not None else DST_TILES_FP32_ACC
    )

    k_tiles = math.ceil(k / TILE_SIZE)
    cap = tuning["in0_block_w_cap"]
    if tuning["in0_block_w_divisor"]:
        # in0_block_w only has to divide k_tiles (no K tail in the 2D mcast kernel), so take the
        # largest legal block rather than scaling with grid width -- see _PREFILL_TUNING.
        in0_block_w = _find_largest_divisor(k_tiles, cap)
    else:
        in0_block_w = min(cap, max(1, k_tiles // grid_size[0]))

    if halve_out_block and out_block_w is None:
        out_block_w = _safe_half_out_block_w(per_core_N, out_subblock_w)

    kwargs = dict(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )
    if out_block_w is not None:
        kwargs["out_block_w"] = out_block_w
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**kwargs)


def create_prefill_kpass1_matmul_program_config(m, k, n, grid_size=None, fused_activation=None):
    """2D prefill progcfg that walks K exactly ONCE: out_block_w == per_core_N.

    A 2D prefill matmul re-traverses K once per N-block (ceil(per_core_N / out_block_w) blocks),
    re-reading a DRAM-resident in0 every time, and prefill_kpass_width's measurements show that pass
    count is the dominant cost term for the GDN in-proj shape. One pass is the floor.

    REQUIRES a compute kernel config with fp32_dest_acc_en=False (COMPUTE_HIFI2_NO_FP32_ACC). Two
    reasons, both load-bearing:
      * the full per_core_N-wide output/intermediate CB only fits L1 at half the element size, and
      * out_subblock_w is picked against the DST_TILES (8) ceiling, not DST_TILES_FP32_ACC (4) --
        e.g. per_core_N=25 gives sub_w=5 at the 8 ceiling but collapses to 1 at the 4 ceiling, and
        out_block_w must be a multiple of out_subblock_w, so a sub_w of 1 cannot reach 25 anyway.
    Passing an fp32-dest-acc config here will fail at program creation with a CB overflow, not
    silently degrade -- which is the intended behaviour.

    Unlike create_prefill_matmul_program_config there is no halve_out_block escape hatch: halving the
    block IS the multi-pass behaviour this exists to avoid. Callers whose (m, k, n) does not fit must
    use the general factory instead.
    """
    if grid_size is None:
        grid_size = prefill_grid_default()
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))
    k_tiles = math.ceil(k / TILE_SIZE)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=min(4, max(1, k_tiles // grid_size[0])),
        out_subblock_h=1,
        out_subblock_w=_get_out_subblock_w(per_core_N, 1, max_hw=DST_TILES),
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=per_core_N,  # one K pass
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


def prefill_out_memory_config(seq_len, out_width, elem_bytes=2, budget=8 << 20):
    """L1 for prefill matmul outputs that fit; DRAM for the big ones.

    Several prefill projections (MLP down-proj, attention wo, ...) were tuned to emit into L1 — a
    real win measured on Blackhole, whose L1 is larger. Those outputs are [seq_len, out_width] and
    grow with the prefill chunk (16MB+ at seq_len=2048, out_width=4096, bf16). On Wormhole that
    leaves so little L1 free that the very matmul producing them can no longer place its own
    statically-allocated circular buffers, and the op dies with "clash with L1 buffers" — CBs are
    L1-only, so the output is what has to move. Blackhole keeps the tuned L1 path unchanged."""
    if is_blackhole():
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG if seq_len * out_width * elem_bytes > budget else ttnn.L1_MEMORY_CONFIG


def _widest_prefill_cols(n, max_cols, subblock_slack=1):
    """Widest grid whose output subblock stays within `subblock_slack` of the best achievable.

    The TP=8 counterpart to `_best_prefill_cols`. More columns is usually a win at TP=8 (the halved
    per-device N leaves cores idle), but NOT when the extra width collapses the subblock: measured
    at S=2048, mlp_gate (N=2176 -> 68 tiles) goes cols 9 -> 11, per_core_N 8 -> 7, and 7 is prime so
    out_subblock_w drops 4 -> 1 -- a 2058us -> 2118us REGRESSION, i.e. the subblock-first ranking
    was right for that shape. Guarding on the subblock keeps the wide grid exactly where it pays:

        matmul     default        this rule       measured
        attn_wo    c10_bw2_sw4    c11_bw4_sw3     803.5 -> 718.7us
        gdn_out    c10_bw2_sw4    c11_bw4_sw3     802.3 -> 719.9us
        mlp_down   c10_bw4_sw4    c11_bw4_sw3    1787.4 -> 1724.9us
        mlp_gate   c9_bw4_sw4     c9_bw4_sw4     2058.1us (unchanged -- already optimal)
    """
    n_tiles = math.ceil(n / TILE_SIZE)
    sw = {cols: _get_out_subblock_w(math.ceil(n_tiles / cols), 1) for cols in range(1, max_cols + 1)}
    floor = max(sw.values()) - subblock_slack
    return max((cols for cols, w in sw.items() if w >= floor), default=1)


def _best_prefill_cols(n, max_cols):
    """Grid width (<=max_cols) maximizing the output subblock, tie-broken to more cores — avoids the
    1x1-subblock stall (e.g. gate/up N=4352 -> 7-wide -> 1x4) the default full width can force."""
    n_tiles = math.ceil(n / TILE_SIZE)
    best_cols, best_key = 1, None
    for cols in range(1, max_cols + 1):
        sw = _get_out_subblock_w(math.ceil(n_tiles / cols), 1)
        key = (sw, cols)  # prefer wider subblock, then more columns (more compute cores)
        if best_key is None or key > best_key:
            best_key, best_cols = key, cols
    return best_cols


def create_prefill_mlp_matmul_program_config(
    m, k, n, fused_activation=None, max_cols=None, tuning=None, halve_out_block=False, max_subblock_hw=None
):
    """FPU-tuned 2D prefill progcfg for MLP matmuls: picks the grid width that maximizes the output
    subblock (drives prefill FPU) instead of the default full width.

    max_cols caps the grid width. Default = prefill_grid_default()[0] (8). Pass the device worker-grid
    width (11 on BH P150) to let the subblock heuristic go wide -> the measured prefill winners
    (gate 9-wide, down/wo 10-wide, gdn_qkvz 11-wide; test_mlp_matmul_sweep_prefill). Fused AG/RS paths
    pin 8-wide separately and are unaffected.

    tuning: a `_PREFILL_TUNING` entry. With `widest_cols` (TP=8) the subblock-first width heuristic
    is replaced by "take the width, clamped to PREFILL_MAX_COLS_PORTABLE" -- measured device time at
    TP=8 falls monotonically with column count, so trading cores for a wider subblock loses.
    halve_out_block: see create_prefill_matmul_program_config — pass on grids already at their
    physical core-count max (WH) where the full per_core_N-wide output/intermediate CB overflows L1.
    max_subblock_hw: see create_prefill_matmul_program_config. NOTE this is deliberately NOT fed into
    _best_prefill_cols below: the grid-width choice stays on the conservative cap so raising the
    subblock ceiling cannot silently move the grid too (an unswept axis). It only widens the subblock
    at the column count production already uses."""
    grid = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    limit = max_cols or grid[0]
    if tuning["widest_cols"]:
        # Cap the width at PREFILL_MAX_COLS_PORTABLE (harvested parts expose 11, not 12) and never
        # exceed the output tile count -- columns beyond it get per_core_N=1 with nothing to compute,
        # paying mcast cost for no work.
        cols = _widest_prefill_cols(n, max(1, min(limit, PREFILL_MAX_COLS_PORTABLE, math.ceil(n / TILE_SIZE))))
    else:
        cols = _best_prefill_cols(n, limit)
    return create_prefill_matmul_program_config(
        m,
        k,
        n,
        grid_size=(cols, grid[1]),
        fused_activation=fused_activation,
        tuning=tuning,
        halve_out_block=halve_out_block,
        max_subblock_hw=max_subblock_hw,
    )


# Mesh tensor helpers
def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
    """Torch weight [out,in] -> sharded mesh tensor. Transpose to [in,out]; dim=-1 column, dim=0 row."""
    w = torch_tensor.to(torch.bfloat16).T.contiguous()
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        cache_file_name=cache_path,
    )


def agmm_k_block_size(k_local, default=8):
    """Largest power-of-2 K_block_size <= `default` that divides K_tiles/device (AGMM Ring has no tail).

    TP=4: 1280->40 tiles->8; TP=8: 640->20 tiles->4. Odd divisors (e.g. 5|20) are unsafe on Ring.
    """
    k_tiles = k_local // TILE_SIZE
    b = 1 << (min(default, max(1, k_tiles)).bit_length() - 1)
    while b > 1 and k_tiles % b:
        b //= 2
    return b


def all_gather_matmul_prefill(
    x,
    weight,
    tt_ccl,
    compute_cfg,
    topology,
    grid=(7, 9),
    cluster_axis=1,
    fused_activation=None,
    out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """Fused all-gather(dim=3) + column-parallel matmul for prefill (all_gather_minimal_matmul_async).

    x: K-sharded activation [.,S,K/tp]; weight: [K,N] col-sharded (K full). Gathers x to full K and
    matmuls in one op, replacing a separate all_gather + linear. fused_activation applied per tile
    before pack (non-parametrized op, e.g. ttnn.UnaryOpType.SILU). out_memory_config places the result
    (default DRAM; L1 keeps it resident for downstream slices)."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    # AG-bound: 2 ethernet links parallelize the gather (P150x4 max; traced_8k TTFT win). grid.x must
    # = num_links*workers, and the 7-wide default (prime) forces 1 link -> widen to 8 (2 links, 4 workers).
    num_links = 2
    grid = (8, grid[1])
    workers = grid[0] // num_links
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=8,
        subblock_h=1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    out = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        fused_activation=fused_activation,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=True,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
    )[0]

    return out


def prefill_ccl_tuning():
    """(chunks_per_sync, num_workers_per_link) for the PREFILL collectives.

    tt_all_reduce() takes these as arguments defaulting to 10 / 2, and qwen36 never passed them;
    DistributedNorm hardcodes the same 10 / 2 for every non-decode call. So unlike their decode
    counterparts (which get per-op configs from model_config) the prefill all-gathers and
    reduce-scatters have never been tuned at all. All four run on 5 cores at 6.8-8.1 GB/s against a
    ~12.5 GB/s Wormhole link -- 55-65% efficiency, i.e. real headroom.

    Used by BOTH prefill collectives: the reduce-scatters (tt_all_reduce call sites in gdn/tp.py and
    mlp.py) and the all-gathers (via tt/prefill_norm_tuned.py).

    MEASURED on N300 at seq 2048, device times straight out of tt-perf-report:

      ALL-GATHER -- the reliable win. Per-op, 2 runs each:
        wpl=2 (upstream)  1,245 / 1,242 us
        wpl=4             1,012 / 1,012 us      <- used
        wpl=8             1,015 / 1,016 us
        => ~-460us per layer across the two gathers, and repeatable to ~10us.

      REDUCE-SCATTER -- smaller and noisy. Layer RS total, 3 runs each:
        wpl=2 (upstream)  2,234 / 2,478 / 2,076   mean 2,263, spread 402
        wpl=4             1,995 / 2,063 / 2,037   mean 2,032, spread  68
        => ~-230us on the mean, and it tightens the spread ~6x, which matters more for tail latency
           than the mean does. Individual RS ops still range 963-1,277us run to run, so do not read
           much into any single profile.

      wpl=8 is indistinguishable from 4 once both collectives are tuned (CCL mean 4,313 vs 4,278 us
      over 2-3 runs each), so 4 stays -- it is the smaller departure from the upstream default.
      chunks_per_sync made no measurable difference anywhere and stays at 10.

    QWEN35_PREFILL_CCL="cps,wpl" overrides, for re-sweeping.

    WORMHOLE-ONLY: every measurement above is Wormhole (N300 for the 9B pre-norm gather, T3K for the
    27B post-norm one) -- there is no Blackhole number here at all. Blackhole keeps upstream's
    untuned literals (10, 2), same as before this function existed, unless QWEN35_PREFILL_CCL forces
    an override for re-sweeping there too.
    """
    _v = os.environ.get("QWEN35_PREFILL_CCL")
    if _v:
        _c, _w = (int(t) for t in _v.split(","))
        return _c, _w
    if is_blackhole():
        return 10, 2
    return 10, 4


def mlp_gateup_agmm_enabled(num_devices):
    """Fuse the ff_norm all-gather into the MLP gate/up matmul (prefill). TP-only (needs the gather).

    BH-only: all_gather_swiglu_prefill's grid assumes BH's taller (9-10 row) compute grid; WH tops
    out at 8 rows, so this fusion is unvalidated there. Falls back to the unfused AG + matmul path on WH.

    MEASURED on N300 (2026-08): the row count is NOT the only blocker, so do not just clamp the grid.
    With grid height forced to 8, all_gather_minimal_matmul_async's program factory builds its in0/in1
    sender+receiver core ranges from grid_size.y-1/-2/-3; at y=8 those overlap so two data-movement
    kernels land on one core needing both NOCs, and program creation dies with
    "TT_FATAL ... local_noc0_in_use and local_noc1_in_use" (tt_metal.cpp:152). Reproduced at num_links
    1 AND 2, so it is not a link-count artifact — enabling WH needs a C++ change to that op's core/NOC
    assignment for 8-row grids. Worth doing: the two all-gathers this would hide are 1,239us + 1,242us
    of a 21,669us single-layer GDN prefill at seq 2048 (5 cores each, fully exposed)."""
    return num_devices > 1 and is_blackhole()


def all_gather_swiglu_prefill(
    x, weight, tt_ccl, compute_cfg, topology, grid=(7, 9), cluster_axis=1, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """Fused all-gather + col-parallel gate/up matmul + SwiGLU for prefill (packing gate+up lets ff_norm's AG fuse in).

    x: K-sharded [.,S,K/tp]; weight: tile-pair-interleaved [gate|up] [K, 2N/tp]. Emits silu(gate)*up of width N/tp."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    num_links = 2
    grid = (8, grid[1])
    workers = grid[0] // num_links
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=16,
        subblock_h=1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    return ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=True,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
        fuse_swiglu=True,
    )[0]


def build_mmrs_decode_state(mesh_device, M, K_local, N, nd, dtype=ttnn.bfloat16):
    """Build (progcfg, intermediate_buffer, output_buffer) for a decode matmul_reduce_scatter out-proj.

    M = LOGICAL decode batch (max_batch_size) — the op returns the persistent buffer with its logical
    shape, so an oversized (tile-padded) M leaks into the residual stream. TILE layout pads M<32.
    dtype MUST match the out-proj input activation (bf16 for MLP/attn; FLOAT32 for GDN, which keeps
    fp32 for stability) — the op's default output dtype is the input's, and writing it into a
    mismatched buffer corrupts the output. Matmul on reduced grid (8,6); RS workers at offset (0,6).
    interm [1,1,M,N], out [1,1,M,N/nd]."""
    cg = (8, 6)
    per_core_N = max(1, math.ceil(N / TILE_SIZE / cg[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=cg,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // cg[0])),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / cg[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(cg),
    )
    mk = lambda w: ttnn.from_torch(
        torch.zeros(1, 1, M, w),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return pc, mk(N), mk(N // nd)


def matmul_reduce_scatter_decode(
    x, weight, tt_ccl, interm_buf, out_buf, progcfg, compute_cfg, topology, rs_offset=(0, 6)
):
    """Fused row-parallel matmul + reduce-scatter(dim=3) for decode (matmul_reduce_scatter_async).

    x: K-sharded [.,M,K_local]; weight: [K_local,N] K-sharded. Matmul runs on progcfg's (reduced)
    grid; RS workers land at rs_offset (disjoint rows) to avoid the collision that deadlocks a
    full-grid fused CCL. Persistent buffers are caller-owned. Returns [.,M,N/nd] (fractured, DRAM)."""
    _, rs_out = ttnn.experimental.matmul_reduce_scatter_async(
        x,
        weight,
        persistent_intermediate_buffer=interm_buf,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=1,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=progcfg,
        compute_kernel_config=compute_cfg,
    )
    # rs_out IS the persistent output buffer; clone so the caller can deallocate its copy while the
    # persistent buffer survives for the next token (else layer.py's deallocate frees it -> corruption).
    return ttnn.clone(rs_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype):
    """Lazily allocate (and cache on tt_ccl) shared persistent buffers for the prefill fused out-proj.

    Prefill M (=chunk seq, e.g. 2048) makes per-layer buffers huge (fp32 [1,1,2048,5120]≈42MB × 64
    layers = infeasible). Prefill runs layers sequentially and each op's output is cloned before the
    next layer reuses the buffer, so ONE shared set per (M,N,nd,dtype) is safe. Allocated during the
    pre-capture warmup forward (eager), reused inside the trace. Keyed so variable M/dtype coexist."""
    cache = getattr(tt_ccl, "_qwen36_mmrs_prefill_bufs", None)
    if cache is None:
        cache = {}
        tt_ccl._qwen36_mmrs_prefill_bufs = cache
    key = (M, N, nd, str(dtype))
    if key not in cache:
        mesh = tt_ccl.mesh_device
        mk = lambda w: ttnn.from_torch(
            torch.zeros(1, 1, M, w),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cache[key] = (mk(N), mk(N // nd))
    return cache[key]


def matmul_reduce_scatter_prefill(x, weight, tt_ccl, compute_cfg, topology, nd, dtype, grid=(8, 8), rs_offset=(0, 8)):
    """Fused row-parallel out-proj matmul + reduce-scatter for PREFILL (matmul_reduce_scatter_async).

    Unlike decode (M=1, where the 2D matmul collapses to ~8 cores and this loses), at prefill M>>1 the
    2D matmul fills the grid, so overlapping the RS with the matmul is a WIN (biggest for the fp32
    GDN-out with its large RS). grid=(8,8): matmul rows 0-7, RS workers rows 8-9. x: K-sharded
    [.,M,K_local]; weight [K_local,N]. Returns [1,1,M,N/nd] (cloned; shared buffer survives).

    BH-ONLY. Porting this to Wormhole was tried and reverted (N300, 2026-08). It looks like the one CCL
    fusion that should port, because this op takes an explicit reduce_scatter_core_grid_offset, so WH's
    8-row grid is expressible as matmul (8,6) on rows 0-5 + RS workers at (0,6) on rows 6-7 — the exact
    disjoint split build_mmrs_decode_state already uses for the DECODE out-proj. It HANGS anyway: the
    op never returns and the hang wedges the ethernet cores, after which the next device open fails
    with "Timed out waiting for ETH heartbeat ... Stuck at 0xaabb0001". Recovery is
    `tt-topology -l mesh` — note this host's layout is MESH; the tool's default is linear and flashing
    that breaks device discovery entirely.

    Reproduced twice, so it is not the link count: first at num_links=2, then at num_links=1 after
    finding that an N300 (1,2) submesh reports get_num_links() == 1 on every axis while the value below
    is hardcoded to 2 (a P150x4 number). Same symptom both times. Likely the same class of defect as
    the all-gather fusion (see mlp_gateup_agmm_enabled) — the fused-CCL program factories assume a
    taller grid than WH has, and a decode config that works at M=1 does not carry to a prefill M that
    fills the matmul grid. Needs C++ investigation, not a config change. Cost of leaving it off: the
    two reduce-scatters are ~1,060us + ~995us of an 18,644us single-layer GDN prefill at seq 2048."""
    M, K_local = x.shape[-2], x.shape[-1]
    N = weight.shape[-1]
    interm, out_buf = _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype)
    x4 = ttnn.reshape(x, (1, 1, M, K_local))
    # RS-bound: 2 ethernet links parallelize the fp32 cross-device reduce (P150x4 max; traced_8k win).
    # grid (8,8) leaves rows 8-9 for the 2 RS worker rows.
    num_links = 2
    per_core_N = max(1, math.ceil(N / TILE_SIZE / grid[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // grid[0])),
        out_subblock_h=1,
        # Keep 1x1: op242 is RS-bound and this op is pipelined to overlap the matmul with the RS.
        # Widening the subblock desyncs that overlap and measured net-negative on traced_8k TTFT.
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / grid[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(grid),
    )
    _, rs = ttnn.experimental.matmul_reduce_scatter_async(
        x4,
        weight,
        persistent_intermediate_buffer=interm,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=num_links,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        compute_kernel_config=compute_cfg,
    )
    return ttnn.clone(rs, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def sharded_decode_matmul(
    x,
    weight,
    compute_cfg,
    decode_progcfg,
    act_shard_cfg,
    prefill_progcfg_fn,
    prefill_k,
    decode_out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    prefill_compute_cfg=None,
    prefill_out_dtype=None,
):
    """DRAM-WIDTH_SHARDED weight matmul; branches on M (decode vs prefill).

    Decode (M<=32): L1-sharded act + DRAM-sharded kernel. Prefill: 2D matmul.
    Gate on x.shape[-2] (seq/M), not x.shape[1] (Z=1 in both modes). Decode result placement is
    `decode_out_memory_config` (default DRAM-interleaved; pass L1 to keep the small decode
    activation resident). Prefill result is always DRAM-interleaved.

    prefill_compute_cfg: compute kernel config for the PREFILL branch only (defaults to compute_cfg).
    Exists because a prefill progcfg and its compute config are coupled — create_prefill_kpass1_
    matmul_program_config's blocking is only legal with fp32_dest_acc_en off — while decode keeps the
    shared COMPUTE_HIFI2. Pass both together or neither.

    prefill_out_dtype: output dtype for the PREFILL branch only. ttnn.linear defaults the output
    dtype to in0's (matmul.cpp: ``dtype.value_or(input_tensor_a.dtype())``), so a caller that narrows
    its ACTIVATION to bf8 silently narrows the matmul RESULT too. Pass this to pin the result where
    that is not wanted -- the in0 saving is kept either way, since it is a read-side win."""
    seq = x.shape[-2]
    if seq <= TILE_SIZE:
        # Reshard act to L1 if needed; skip dealloc when x already sharded (GDN reuses x).
        already_sharded = x.memory_config() == act_shard_cfg
        x_sh = x if already_sharded else ttnn.to_memory_config(x, act_shard_cfg)
        out = ttnn.linear(
            x_sh,
            weight,
            compute_kernel_config=compute_cfg,
            program_config=decode_progcfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        if not already_sharded:
            ttnn.deallocate(x_sh)
        return ttnn.to_memory_config(out, decode_out_memory_config)
    pc = prefill_progcfg_fn(seq, prefill_k, weight.shape[-1])
    return ttnn.linear(
        x,
        weight,
        compute_kernel_config=prefill_compute_cfg or compute_cfg,
        program_config=pc,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **({"dtype": prefill_out_dtype} if prefill_out_dtype is not None else {}),
    )


def replicate(torch_tensor, mesh, cache_path, dtype=ttnn.bfloat16):
    """Small tensor (norm/bias) -> replicated on every device."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def shard_small(torch_tensor, mesh, cache_path, dim=-1, dtype=ttnn.bfloat16):
    """Small per-head tensor (conv taps, A_log, dt_bias) -> sharded."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def replicate_kv_weight(weight, n_kv_heads, tp, head_dim):
    """Replicate KV weight so each device gets >=1 head. No-op when tp <= n_kv_heads."""
    if tp <= n_kv_heads:
        return weight
    chunks = weight.reshape(n_kv_heads, head_dim, -1)
    parts = []
    for d in range(tp):
        kv_idx = (d * n_kv_heads) // tp
        parts.append(chunks[kv_idx])
    return torch.cat(parts, dim=0).reshape(tp * head_dim, -1)


# FP8 dequantization
def dequant_fp8_block(weight_fp8, scale_inv, block_size=128):
    """Dequantize a block-wise FP8 weight tensor to bfloat16."""
    out_f, in_f = weight_fp8.shape
    weight_bf16 = weight_fp8.to(torch.bfloat16).reshape(out_f // block_size, block_size, in_f // block_size, block_size)
    weight_bf16 = weight_bf16 * scale_inv[:, None, :, None].to(torch.bfloat16)
    return weight_bf16.reshape(out_f, in_f)


# Weight-prep (reorder HF weights for per-device sharding)
def prepare_attn_qkv(q_w, k_w, v_w, qg_per, kv_per, tp):
    """Fuse attn q+gate/k/v for column-parallel shard: each device gets [qg_d|k_d|v_d].

    q_w: [n_heads*head_dim*2, in]; k_w/v_w: [n_kv_heads*head_dim, in].
    qg_per/kv_per: per-device out block sizes."""
    parts = []
    for d in range(tp):
        parts.append(q_w[d * qg_per : (d + 1) * qg_per, :])
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
    return torch.cat(parts, dim=0)


def prepare_attn_qkv_deint(q_w, k_w, v_w, nh_local, hd, kv_per, tp):
    """Like prepare_attn_qkv but de-interleaves [q,g] per head -> [all_q|all_gate|k|v] per device.

    Avoids prefill relayout in _make_heads (column perm only; numerically identical).
    q_w: [nh_total*hd*2, in]; nh_local/kv_per: per-device block sizes."""
    hd2 = hd * 2
    parts = []
    for d in range(tp):
        base = d * nh_local * hd2
        q_rows = [q_w[base + h * hd2 : base + h * hd2 + hd, :] for h in range(nh_local)]
        g_rows = [q_w[base + h * hd2 + hd : base + h * hd2 + hd2, :] for h in range(nh_local)]
        # Per-device layout [all_q | k | v | all_gate]: q/k/v contiguous so _make_heads* can hand
        # the fused q|k|v block straight to nlp_create_qkv_heads (no re-concat); gate trails, applied
        # post-SDPA. (Column perm only; numerically identical to [q|gate|k|v].)
        parts.append(torch.cat(q_rows, dim=0))  # all_q
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(torch.cat(g_rows, dim=0))  # all_gate (last)
    return torch.cat(parts, dim=0)


def prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp):
    """Interleave GDN Q/K/V heads for row-parallel shard (contiguous q/k/v block per device).

    qkv_w: [key_dim*2 + value_dim, hidden]."""
    q_part = qkv_w[:key_dim, :]
    k_part = qkv_w[key_dim : 2 * key_dim, :]
    v_part = qkv_w[2 * key_dim :, :]

    q_per = nk // tp
    v_per = nv // tp
    shards = []
    for s in range(tp):
        q_s = q_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        k_s = k_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        v_s = v_part[s * v_per * dv : (s + 1) * v_per * dv, :]
        shards.append(torch.cat([q_s, k_s, v_s], dim=0))
    return torch.cat(shards, dim=0)


def prepare_conv_taps(conv_w, key_dim, nk, dk, nv, dv, kernel_size, tp):
    """Split fused conv1d into kernel taps, reordered to match prepare_gdn_qkv grouping."""
    cw = conv_w.float()
    q_per = nk // tp
    v_per = nv // tp
    taps = []
    for j in range(kernel_size):
        tap = cw[:, 0, j]
        q_tap = tap[:key_dim]
        k_tap = tap[key_dim : 2 * key_dim]
        v_tap = tap[2 * key_dim :]
        shards = []
        for s in range(tp):
            q_s = q_tap[s * q_per * dk : (s + 1) * q_per * dk]
            k_s = k_tap[s * q_per * dk : (s + 1) * q_per * dk]
            v_s = v_tap[s * v_per * dv : (s + 1) * v_per * dv]
            shards.append(torch.cat([q_s, k_s, v_s]))
        taps.append(torch.cat(shards))
    return taps
