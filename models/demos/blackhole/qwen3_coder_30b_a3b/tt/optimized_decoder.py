# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized TTNN decoder layer for Qwen3-Coder-30B-A3B-Instruct.

Same semantics as ``functional_decoder`` -- prefill/decode contract, paged KV
cache, non-aligned sequence lengths, determinism -- with the measured path
retuned. Every number below is from real checkpoint weights on a Blackhole
p300c, 1x1 mesh; the sweeps behind them are in ``doc/optimized_decoder/``.

    prefill  536.54 -> 69.12 us/token           at S=512   (7.76x)
    decode     1.5655 -> 0.5634 ms/token traced at ctx128   (2.78x)

Both lines are cells of ``doc/{functional,optimized}_decoder/perf_prefill.csv``
and ``perf_decode.csv``, which every run of ``tests/test_perf.py`` rewrites;
the third significant figure moves between runs.

What changed, in order of how much it mattered
----------------------------------------------
**1. ``in0_block_w`` (3.0x on prefill).** Stage 01 inherited ``in0_block_w=1``
from the exemplar's config helper. With K = 2048 (64 tiles) that feeds the
kernel one tile of the inner dimension at a time, which is what held the expert
matmuls at ~5.4% of peak FLOPs -- not the core count, and not precision.

**2. bfloat4_b expert weights (2.2x on prefill).** Only visible *after* the
block-width fix: at ``in0_block_w=1`` the kernel is latency-bound, so weight
dtype cannot matter. The two knobs interact and sweeping either alone finds the
wrong optimum -- see ``EXPERT_IN0_BLOCK_W_GATE_UP``.

**3. DRAM-sharded decode attention projections (1.11x on decode).** Once the
experts were fast, ``o_proj`` and ``qkv`` were 21% of decode device time and
the stage-01 "attention is 0.08% of prefill, no action" call went stale.
``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` with the weight
width-sharded across the 8 DRAM banks takes qkv 68.3 -> 46.8 us and wo
96.0 -> 41.7 us at the op level, and the whole traced layer 0.6508 -> 0.5863 ms
at ctx128 measured like for like -- both legs otherwise at the configuration
shipped at the time, on the same bfloat8_b weights, and both before lever 7
below, which is why the fast leg reads 0.5863 rather than today's 0.5634. A
``0.697 -> 0.587`` pair (1.19x)
that this file and the docs used to carry is withdrawn; see
``attention_decode_optimized`` and ``doc/optimized_decoder/work_log.md`` §5.

**4. Packing gate and up (1.09x).** ``_sparse_matmul_config`` parallelises only
over N, so N tiles cap the usable cores: 768 -> 24, 1536 -> 48, 2048 -> 64.
Packing doubles gate/up's cores. Measured against a *properly tuned separate*
candidate it is only 1.09x (2 x 1.476 = 2.952 ms -> 2.699 ms); against the
untuned stage-01 candidate it looks like 1.66x, but most of that belongs to the
block-width fix. Re-confirmed at the end of the stage on the whole layer:
separate 0.735 ms vs packed 0.673 ms traced decode.

**5. LoFi on the expert matmuls (1.05x on prefill).** bfp4 weights carry 4
mantissa bits, so HiFi4's extra passes have nothing to work on. Prefill S=512
72.46 -> 69.13 us/token, decode 0.6746 -> 0.6638 ms, and layer PCC is 0.99910 at
LoFi vs 0.99909 at HiFi4 -- i.e. very slightly *better*.

**6. bfloat8_b attention projections (1.02x on decode).** 0.6726 -> 0.6605 ms at
PCC 0.99906 vs 0.99909. bfloat4_b is 0.6595 ms but drops layer PCC to 0.9928,
below the 0.995 bar, so it is rejected.

**7. The router's two keepdim reductions (1.045x on decode).** The router and
its routing prep were 111.6 us of decode device time -- 20.9% of it, more than
either matmul family -- and had no audit finding of their own until the fourth
review. Two thirds of the removable part was not arithmetic at all: ``ttnn.max``
and ``ttnn.sum`` each pull a ``FillPad`` behind them on a tensor whose last two
dims are not tile-aligned, 10.42 and 10.41 us against 1.43 and 1.41 us of actual
reduction. ``router_forward_optimized`` deletes both -- the max is column 0 of
the sorted top-k, and the sum moves after the scatter where the reduction length
is a whole number of tiles -- for 0.5866 -> 0.5615 ms traced at ctx128 (both
legs in one process), with the routing itself unchanged
(``test_optimized_router_matches_functional`` asserts identical expert
selection). Moving the sum past the scatter puts the divide over whole tiles,
whose row-padding then divides 0 by 0; the divisor is clamped so that padding
stays exactly zero, which costs +1.6 us and is why ``perf_decode.csv`` reads
0.5634 rather than 0.5615. See that function and ``work_log.md`` §7.


Rejected, with measurements
---------------------------
**Per-token sparsity in prefill.** Prefill hands ``sparse_matmul`` a sparsity
tensor of shape ``[1, 1, group_size, E]`` whose rows are 32-token *tiles*, so an
expert counts as active if any of the tile's 32 tokens picked it. With 256
selections landing across 128 slots essentially every expert is hit -- hence
``active=128/128``. Making it per-token requires tokens to be *batch* indices
(``sparse_matmul`` indexes sparsity by batch dims, not by M), i.e. a
``[1, T, 1, H]`` layout. Measured rather than assumed: it runs, cuts nnz 16x
from 4096 to 256, and is **2.1x slower** (14.35 ms vs 6.70 ms), because M
collapses to 1 and the op pads M to a full 32-row tile. Decode keeps real
per-token sparsity, which is free there because M is genuinely 1.

**1x32 output tiles on the decode sparse matmuls.** The M padding above is the
single largest remaining inefficiency: at decode M=1 the gate/up matmul writes
12 MB and ``down`` writes 16 MB where 0.4/0.5 MB is real, and the reshapes that
compact it away cost 31 + 33 + 46 us. ``output_tile=ttnn.Tile([1, 32])``
removes the padding at the source and is 1.07x faster end to end -- but no
downstream op consumes the result correctly. Measured, in this order:
``slice`` rejects it (``slice_device_operation.cpp:165`` hardcodes
``TILE_HEIGHT``), ``ttnn.sum`` and ``ttnn.reshape`` raise
``MeshBuffer must be large enough``, ``untilize`` returns wrong data without
erroring, and ``fast_reduce_nc`` returns all zeros. Only eltwise ops read it
correctly, and they immediately re-pad to 32 rows. Blocked on TTNN support for
non-32 tile heights outside matmul, not on this model.

**Folding the routing weight in before ``down``.** ``down`` is linear, so
scaling its *input* by the routing probability is equivalent to scaling its
output, and the input is the compact ``[B, E, I]`` tensor rather than the
32x-row-padded ``[B, E, 1, H]`` one. It looked like it should collapse the
whole tail into one reduce. Measured at ctx128 in one run, before the §7 router
change (so its shipped-tail leg is the 0.5862 ms configuration of the time, not
today's 0.5634): shipped tail 0.5862 ms, folded with a compact sum 0.5852 ms,
folded with ``fast_reduce_nc`` straight off the padded tensor 0.6316 ms. A
second run of the same three legs read 0.5870 / 0.5853 / 0.6326 and was quoted
in parallel with this one; this triple is the one whose shipped leg matches the
``perf_decode.csv`` ctx128 cell of that day, and it is now the only one quoted
anywhere. The first two are a tie and the third is 8% *worse* --
``fast_reduce_nc`` also promotes ``down``'s tile padding into the logical shape,
so recovering ``[1,1,B,H]`` needs a permute plus a slice that together cost more
than the ops they replaced. The shipped tail stays. (The intermediate version of
this that used a plain reshape instead of the permute was faster still, 0.550 ms
-- and silently wrong for every user but the first, which is how the permute
came to be needed. ``test_optimized_decode_batch`` caught it.)

**Keeping the expert path rank-6.** The obvious reading of the profile is that
the rank-changing reshapes are pure overhead. They are not: they compact the 32x
M padding away, so the elementwise ops that follow touch 192 tiles instead of
6144. Staying rank-6 and dropping all three reshapes measured **0.713 ms vs
0.673** -- 6% slower.

**Everything else tt-perf-report suggested**, each measured on the traced layer
against the tuned baseline: in0 in L1 on the sparse rows 1.001x, on the
attention rows 0.998x, ``out_subblock_w=2`` 1.001x, ``=4`` 1.001x. All noise;
none adopted. HiFi2 on the sparse rows is covered by lever 5 -- LoFi is both
faster and no less accurate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn

from .functional_decoder import (  # noqa: F401  (re-exported for callers)
    AttentionConfig,
    AttentionWeights,
    DecoderLayerConfig,
    DecoderLayerWeights,
    KVCache,
    MoEConfig,
    _apply_rope,
    _concat_heads_decode,
    _per_head_rms_norm,
    _sparse_matmul_config,
    attention_decode,
    attention_prefill,
    build_expert_sparsity,
    build_rope_cache,
    create_kv_cache,
    upload_layer_weights,
    upload_router_weight,
)
from .precision import DEFAULT_PRECISION, PrecisionConfig  # noqa: F401  (re-exported)

# Tokens per expert-path chunk. Kept at one tile: sparse_matmul folds the group
# dimension into M, so a larger chunk grows num_blocks_y and can overflow the
# core grid. Chunking at 32 keeps all blocking in N.
EXPERT_CHUNK_SIZE = 32

# Expert matmul precision and inner-block width.
#
# These two knobs INTERACT, and sweeping either alone finds the wrong optimum.
# At in0_block_w=1 the kernel is latency-bound, so weight dtype makes no
# measurable difference -- which is exactly the null result stage 02 first
# recorded, and wrongly concluded from. Widening the block makes the matmul
# bandwidth-bound, at which point precision becomes the dominant lever.
#
# Joint sweep, real weights, M=32, ms (grid 8x6 for gate/up, 8x8 for down):
#
#   packed gate+up (K=2048, 64 tiles)      down (K=768, 24 tiles)
#   blk   bf16   bfp8   bfp4  bfp4/LoFi    blk   bf16   bfp8   bfp4
#     4  2.697  2.371  2.367      2.366      4  1.489  1.062  1.058
#     8  2.726  1.669  1.445      1.431      6  1.507  1.002  0.785
#    16  2.932  1.806  1.259      1.149      8  1.507  0.993  0.712
#    32  2.910  1.734  1.158      1.153     12  1.513  0.941  0.654
#    64  3.180  1.789  1.372      1.204     24  1.622  0.982  0.733
#
# bf16's best is 2.697 + 1.489 = 4.186 ms; bfp4's is 1.259 + 0.654 = 1.913 ms.
# Block width must divide K in tiles, and the two matmuls have different K, so
# the widths are per-role rather than one shared constant.
#
# The table above is a matmul microbenchmark; the widths were re-confirmed on
# the whole layer, where the interaction with fidelity reverses the gate/up
# choice (prefill S=512 us/token, real weights):
#
#   blk       8      16      32      64
#   HiFi4  78.27   72.46   69.96   75.56
#   LoFi   78.12   69.13   69.27   70.88
#
# 16 at LoFi is the minimum, so 16 stays.
#
# **These five names are now aliases, not the source of truth.** The values
# themselves live in ``precision.PrecisionConfig``, whose defaults are exactly
# what was written here before stage 07; the names survive because probes under
# ``doc/`` and the stage-02/04 tests import them, and because a reader arriving
# at the sweep comments above should find the value they describe next to them.
# Anything that needs to *vary* the policy must take a ``PrecisionConfig``
# instead -- these are bound at import and cannot follow a non-default model.
EXPERT_WEIGHT_DTYPE = DEFAULT_PRECISION.experts_gate_up_dtype
EXPERT_IN0_BLOCK_W_GATE_UP = DEFAULT_PRECISION.experts_gate_up_in0_block_w  # divides 2048/32 = 64
EXPERT_IN0_BLOCK_W_DOWN = DEFAULT_PRECISION.experts_down_in0_block_w  # divides 768/32 = 24

# bfp4 weights carry 4 mantissa bits, so HiFi4's extra passes have nothing left
# to resolve. LoFi is 4.6% faster on prefill and 1.6% on decode at PCC 0.99910
# vs HiFi4's 0.99909. This also answers tt-perf-report's "HiFi2 may also work"
# on the sparse rows: HiFi2 measured 69.78 us/token, between the two.
EXPERT_MATH_FIDELITY = DEFAULT_PRECISION.experts_fidelity

# Attention projections. bf16 -> bfloat8_b costs 0.00003 PCC and buys 1.8% of
# decode; bfloat4_b buys another 0.1% but drops layer PCC to 0.9928, under the
# 0.995 bar, so it is rejected. q_norm/k_norm stay bf16 -- they are norms, not
# projections, and weigh 4 KB.
ATTENTION_WEIGHT_DTYPE = DEFAULT_PRECISION.attention_qkv_dtype

# Blackhole p300c has 8 DRAM banks. The DRAM-sharded matmul wants the weight
# width-sharded one shard per bank, and both the activation and the output
# width-sharded in L1 over the matching core row.
_DRAM_BANKS = 8


def _expert_compute_kernel_config(device, precision: PrecisionConfig = DEFAULT_PRECISION):
    """LoFi, and ``fp32_dest_acc_en`` deliberately OFF.

    ``fp32_dest_acc_en`` looks like the natural next lever but must not be used
    here: it halves the matmul dest from 8 tiles to 4, which corrupts expert
    output on Blackhole (tt-metal #49068, hit on BH-QB-2). It is therefore
    **not** a ``PrecisionConfig`` field -- a sweep must not be able to turn it
    on.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=precision.experts_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _attention_compute_kernel_config(device, precision: PrecisionConfig = DEFAULT_PRECISION):
    """``None`` at the default, which is what the projections have always passed.

    ``attention_fidelity=None`` means "leave the op at its own default", so this
    returns ``None`` and the ``compute_kernel_config=`` argument is a no-op. Any
    named fidelity builds a real config; the remaining flags mirror
    ``_expert_compute_kernel_config``'s, which is the closest measured
    neighbour.
    """
    if precision.attention_fidelity is None:
        return None
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=precision.attention_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _tuned_sparse_matmul_config(m: int, n: int, k: int, target_blk: int):
    """``_sparse_matmul_config`` with a tuned inner block width.

    ``k`` is the inner dimension in elements; the block width must divide it in
    tiles, so this falls back to the largest legal divisor at or below the
    target rather than failing.
    """
    k_tiles = max(1, k // 32)
    blk = min(target_blk, k_tiles)
    while blk > 1 and k_tiles % blk:
        blk -= 1
    return _sparse_matmul_config(m, n, in0_block_w=blk)


# Column of ones used to sum the dense routing row (see ``router_forward_optimized``).
# Cached per (device, length) because it is a constant, and because allocating a
# tensor inside a trace capture is illegal -- every caller runs the layer eagerly
# once to compile before capturing, which is what populates this.
#
# The key carries ``id(device)`` but the *value* carries the device object itself.
# ``mesh_device`` is function-scoped in ``conftest.py`` and is closed and deleted
# after each test, and CPython reuses freed addresses, so a later device could be
# handed the same id and collide with an entry bound to a destroyed one. Holding
# the object in the value makes the pin explicit: the address cannot be recycled
# while the entry lives, so equal ids imply the same live device. The identity
# check below is then a belt-and-braces assertion, not a hope.
_ONES_COLUMN: dict[tuple[int, int], tuple[object, ttnn.Tensor]] = {}


def _ones_column(device, n: int) -> ttnn.Tensor:
    key = (id(device), n)
    entry = _ONES_COLUMN.get(key)
    if entry is not None and entry[0] is device:
        return entry[1]
    cached = ttnn.from_torch(
        torch.ones(1, 1, n, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _ONES_COLUMN[key] = (device, cached)
    return cached


def router_forward_optimized(x: ttnn.Tensor, w_router: ttnn.Tensor, config: MoEConfig) -> ttnn.Tensor:
    """``router_forward`` with both keepdim reductions removed. Same result.

    The routing maths is unchanged from ``functional_decoder.router_forward``,
    including the part that is load-bearing for correctness: selection happens
    on the **raw fp32 logits** and the softmax is taken over the 8 survivors
    only. A 128-wide bf16 softmax misroutes 83/128 tokens and is not an option
    here; see that function's docstring for the algebra.

    What changes is how the two reductions are spelled. ``ttnn.max``/``ttnn.sum``
    call ``fill_implicit_tile_padding`` whenever *either* of the last two dims is
    unaligned (``fill_pad.cpp:17-24``); the top-k tensor is 8 wide, and decode's
    is 1 row tall, so each keepdim reduction dragged a ``FillPad`` behind it --
    **10.421 µs and 10.413 µs** in the archived stage-01 decode profile
    (``doc/functional_decoder/ops_perf_decode_paged32.csv`` rows 73 and 77),
    against **1.432 and 1.407 µs** for the reductions themselves (rows 74 and
    78). Both are avoided rather than tuned:

    * the **max** is column 0 of the top-k output, which ``sorted=True``
      guarantees is the largest, so one 0.87 µs ``slice`` replaces
      ``FillPad + Reduce``;
    * the **sum** moves *after* the scatter and becomes a matmul against a
      column of ones. Over the dense row the reduction length is
      ``num_experts`` = 128 — a whole number of tiles — so no padding lane can
      enter the sum. That is also why this is preferred to the same matmul over
      the 8-wide tensor, whose K padding would carry whatever ``topk`` left
      behind. Normalising after the scatter is legal because the scatter is a
      permutation of the 8 survivors into a field of exact zeros, so the sum
      over 128 *is* the sum over the 8.

    **The padding cost of moving the sum.** Dividing after the scatter divides
    over whole tiles, and the tile *row* padding -- rows S..ceil(S/32)*32 -- has
    ``dense`` = 0 and therefore ``total`` = 0 too. Unguarded, ``ttnn.div``
    returns **+inf** there (not NaN; measured at S = 33 and 100, where every one
    of the 31 and 28 padding rows came back +inf), where the functional router,
    which divided before the scatter, returned exact zeros. Nothing observable
    leaked -- ``ttnn.to_torch`` returns the logical shape, the sparsity path
    drops the padding in ``to_layout(ROW_MAJOR)``, and the scale multiply,
    ``rms_norm`` and ``fast_reduce_nc`` all reduce along axes that are either
    tile-aligned or not the padded one -- but it is a hazard the functional path
    did not have, so the divisor is clamped (``ttnn.maximum(total, 1e-30)``).
    Decode is the one case that was already clean: at S = 1 the padding came
    back exact zero unguarded. The clamp is one extra op -- 1.12 µs in the
    decode profile, +1.6 µs on the traced layer, 0.28% -- and
    ``test_optimized_router_padding_is_zero`` stops it being optimized back out.
    See ``work_log.md`` §7 for the padding table and the rejected free version.

    Measured on the whole traced layer at ctx128, real weights, median of 100:
    **0.5866 -> 0.5615 ms** (``perf_decode.csv`` reads 0.5634, which is that
    configuration in its own run), and the router block **111.6 -> 87.8 µs** of
    decode device time -- rows 68-88 of
    ``doc/functional_decoder/ops_perf_decode_paged32.csv``, which still holds the
    pre-fix block, against the same block in the pre-guard optimized profile.
    With the divisor guard the block is 88.9 µs, rows 69-88 of
    ``doc/optimized_decoder/ops_perf_optimized_decode.csv``. Layer PCC is
    0.99901 either way (prefill S=128 vs HF: 0.9990057 after, 0.9990050
    before). ``doc/optimized_decoder/work_log.md`` §7
    carries the rejected variants, including ``ttnn.softmax`` over the 8
    survivors — faster still, and wrong: it reduces over the whole 32-wide tile,
    so the weights sum to 0.9736 instead of 1.
    """
    assert config.norm_topk_prob, (
        "router selects on raw logits, which relies on the softmax denominator "
        "cancelling during top-k renormalisation; that only holds when "
        "norm_topk_prob is True"
    )

    logits = ttnn.linear(x, w_router, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    top_logits, top_indices = ttnn.topk(logits, k=config.num_experts_per_tok, dim=-1, largest=True, sorted=True)

    # Subtracting the max is for exp() range only; any shared shift cancels in
    # the division. sorted=True means column 0 already is that max.
    top_max = ttnn.slice(top_logits, [0, 0, 0, 0], [1, 1, top_logits.shape[2], 1])
    exp_logits = ttnn.exp(ttnn.sub(top_logits, top_max))

    zeros = ttnn.typecast(ttnn.zeros_like(logits), ttnn.bfloat16)
    dense = ttnn.scatter(
        zeros,
        dim=-1,
        index=top_indices,
        src=ttnn.typecast(exp_logits, ttnn.bfloat16),
    )
    # Sum over the dense row == sum over the 8 survivors; see the docstring.
    total = ttnn.matmul(dense, _ones_column(x.device(), config.num_experts), dtype=ttnn.bfloat16)
    # Guard the divisor's tile row-padding, which is 0 where ``dense`` is also 0
    # (see the padding note in the docstring). Every real row's denominator is
    # >= 1, because sorted=True makes column 0 of ``exp_logits`` exactly
    # exp(0) = 1, so clamping at 1e-30 cannot touch one: measured bit-identical
    # on the real rows at S = 1, 33, 100. In the padding it turns 0/0 into
    # 0/1e-30 = 0, which is what the functional router returned there.
    guarded = ttnn.maximum(total, 1e-30)
    normalised = ttnn.div(dense, guarded)
    for t in (logits, top_logits, top_indices, top_max, exp_logits, dense, total, guarded):
        ttnn.deallocate(t)
    return normalised


def _bank_row(n: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})


# The L1 shard height below, and ``per_core_M=1`` in the program config, are
# both decode's padded M of one 32-row tile. That is what caps this path at
# batch 32; see ``_dram_sharded_usable``.
_DRAM_SHARDED_MAX_BATCH = 32


def _width_sharded_l1(width: int) -> ttnn.MemoryConfig:
    """L1 width-sharded over one core per DRAM bank, 32 rows (decode's padded M)."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_bank_row(_DRAM_BANKS), [32, width // _DRAM_BANKS], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _dram_sharded_ok(k: int, n: int) -> bool:
    """Both dims must split evenly into whole tiles across the banks."""
    return k % (_DRAM_BANKS * 32) == 0 and n % (_DRAM_BANKS * 32) == 0


# Decode's two expert intermediates are 97% M padding -- ``sparse_matmul`` pads
# M=1 back to a 32-row tile -- so they are large in absolute terms:
#
#     batch * 128 experts * 32 rows * (1536 + 2048) cols * 2 B = batch * 29.4 MB
#
# Blackhole offers ~160 MB of allocatable L1 (110 banks x 1.46 MB, as the
# allocator reports it on this p300c). Holding both in L1 is therefore a
# batch-1 affordance, not a general one: at batch 8 the allocator rejects
# ``down``'s 134 MB output outright. Past the budget the pair goes to DRAM,
# which is what prefill already does at every length.
#
# The 40 MB threshold itself is **asserted, not measured**: it is one comfortable
# step above batch 1's 29.4 MB and below batch 2's 58.8 MB, so it separates the
# only two cases that exist here, and no sweep was run to find where L1 actually
# stops paying. What is measured is the pair of endpoints -- B=1 in L1 is the
# shipped, profiled configuration, and B=8 in L1 does not allocate at all.
_DECODE_EXPERT_L1_BUDGET_BYTES = 40 * 1024 * 1024


def _decode_expert_memory_config(batch: int, config: MoEConfig) -> ttnn.MemoryConfig:
    """L1 for the intermediates while they fit the budget above, else DRAM."""
    padded_rows = batch * config.num_experts * 32
    nbytes = padded_rows * (2 * config.moe_intermediate_size + config.hidden_size) * 2
    return ttnn.L1_MEMORY_CONFIG if nbytes <= _DECODE_EXPERT_L1_BUDGET_BYTES else ttnn.DRAM_MEMORY_CONFIG


def _dram_sharded_usable(weights: "OptimizedWeights", batch: int) -> bool:
    """Whether decode may take the DRAM-sharded projections at this batch.

    Two independent conditions:

    * the weight dims divided evenly across the banks at upload time, so a
      sharded copy exists at all (``_dram_sharded_ok``);
    * the batch still fits decode's single 32-row M tile. ``_width_sharded_l1``
      hardcodes a 32-row shard and ``_dram_sharded_program_config`` sets
      ``per_core_M=1``, so at ``batch > 32`` the activation no longer matches
      its shard spec. Without this check that surfaces as a shard-shape
      mismatch deep in the matmul rather than as a fallback.
    """
    if weights.wqkv_decode is None or weights.wo_decode is None:
        return False
    return batch <= _DRAM_SHARDED_MAX_BATCH


def _dram_sharded_program_config(k: int, n: int):
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=k // _DRAM_BANKS // 32,
        per_core_M=1,
        per_core_N=n // _DRAM_BANKS // 32,
        fused_activation=None,
    )


@dataclass
class OptimizedWeights:
    """Device weights for the optimized layer.

    Two things live here that ``upload_layer_weights`` does not provide:

    * experts with gate and up kept as one weight. ``weight_mapping`` already
      produces the checkpoint's fused ``[E, 2I, H]`` tensor; stage 01 split it
      apart at upload time to mirror the exemplars, so packing is *undoing*
      that split rather than inventing a layout.
    * two copies of the attention projections. Decode uses a DRAM
      width-sharded copy for ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``;
      prefill cannot -- a plain ``ttnn.linear`` on a DRAM-sharded weight throws
      ``Only L1 buffers can have an associated circular buffer`` -- so an
      interleaved copy is kept for it. At bfloat8_b (1.0625 B/elem, because
      each 16-element block carries its own exponent byte) wqkv is 11.14 MB and
      wo 8.91 MB, so the duplicate copy costs **20.05 MB** and the pair 40.11 MB,
      against 24 GB available. An 18.9 MB figure that this file used to carry
      came from rounding bfloat8_b to 1 B/elem and is withdrawn.
      ``doc/context_contract.json`` now carries 20.05 MB too; an earlier
      revision of it made the same rounding error and called the pair a wash
      against stage 01's single bf16 copy, which its ``optimized_note`` records.
    """

    gate_up_proj: ttnn.Tensor  # [1, num_experts, hidden, 2 * intermediate]
    down_proj: ttnn.Tensor  # [1, num_experts, intermediate, hidden]
    attention: AttentionWeights  # interleaved, for prefill
    wqkv_decode: ttnn.Tensor | None  # DRAM width-sharded, for decode
    wo_decode: ttnn.Tensor | None


# Stage-01 name, kept so existing callers and docs still resolve.
PackedExpertWeights = OptimizedWeights


def upload_optimized_weights(
    torch_weights,
    device,
    config: MoEConfig,
    dtype=None,
    *,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> OptimizedWeights:
    """Upload experts packed along the output dim, plus both attention copies.

    ``precision`` supplies every weight dtype. ``dtype``, the stage-02 spelling,
    still overrides **both** expert dtypes when given -- several stage-02 tests
    sweep it directly -- but new callers should pass a ``PrecisionConfig``,
    which can also move gate/up and down apart.

    The expert dtype is a parameter at all because it and ``in0_block_w`` are
    **not** independent: precision only pays once the block width is wide enough
    for the matmul to become bandwidth-bound. Sweeping either alone finds the
    wrong optimum, which is why both live in the same config object.
    """
    fused = torch_weights["experts_gate_up"]  # [E, 2I, H], gate first
    gate_up_dtype = dtype if dtype is not None else precision.experts_gate_up_dtype
    down_dtype = dtype if dtype is not None else precision.experts_down_dtype

    def up(t: torch.Tensor, tensor_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.contiguous().float(),
            dtype=tensor_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

    def as_4d(t: torch.Tensor, pad_to_4d: bool = False) -> torch.Tensor:
        if pad_to_4d:
            t = t.reshape(1, 1, 1, -1)
        while t.dim() < 4:
            t = t.unsqueeze(0)
        return t

    wqkv, wo = as_4d(torch_weights["wqkv"]), as_4d(torch_weights["wo"])

    def dram_sharded(t: torch.Tensor, tensor_dtype) -> ttnn.Tensor | None:
        k, n = int(t.shape[-2]), int(t.shape[-1])
        if not _dram_sharded_ok(k, n):
            return None
        return up(
            t,
            tensor_dtype,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(_bank_row(_DRAM_BANKS), [k, n // _DRAM_BANKS], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )

    return OptimizedWeights(
        gate_up_proj=up(fused.transpose(-2, -1).unsqueeze(0), gate_up_dtype),
        down_proj=up(torch_weights["experts_down"].transpose(-2, -1).unsqueeze(0), down_dtype),
        attention=AttentionWeights(
            wqkv=up(wqkv, precision.attention_qkv_dtype),
            wo=up(wo, precision.attention_wo_dtype),
            q_norm=up(as_4d(torch_weights["q_norm"], pad_to_4d=True), precision.norm_weight_dtype),
            k_norm=up(as_4d(torch_weights["k_norm"], pad_to_4d=True), precision.norm_weight_dtype),
        ),
        wqkv_decode=dram_sharded(wqkv, precision.attention_qkv_dtype),
        wo_decode=dram_sharded(wo, precision.attention_wo_dtype),
    )


# Stage-01 name, kept so existing callers still resolve.
upload_packed_expert_weights = upload_optimized_weights


def attention_decode_optimized(
    x: ttnn.Tensor,
    weights: OptimizedWeights,
    config: AttentionConfig,
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    kv_cache: KVCache,
    current_pos: ttnn.Tensor,
    token_index: int,
    sdpa_program_config=None,
    rope=None,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """``attention_decode`` with the two projections run DRAM-sharded.

    ``sdpa_program_config`` is passed straight through to the SDPA-decode op and
    defaults to ``None``, which is what every single-chip caller uses and what
    every number in this file was measured at. It exists for the multichip path:
    at one KV head the op's default core assignment tries to put all 110 worker
    cores on the single head and its tree reduction refuses past 64
    (``sdpa_decode_program_factory.cpp:245``). See
    ``multichip_decoder._sdpa_program_config``.

    At decode M=1 both projections are pure weight reads, so what limits them is
    how well the read spreads over the DRAM banks. The interleaved layout gave
    383 GB/s on qkv and 235 GB/s on wo; sharding the weight one shard per bank
    and keeping the activation and output width-sharded in L1 measures

        qkv (K=2048, N=5120)   68.3 -> 46.8 us   1.46x
        wo  (K=4096, N=2048)   96.0 -> 41.7 us   2.30x

    at the op level, and 0.6508 -> 0.5863 ms (1.11x) on the whole traced layer
    (both legs measured before the §7 router change, hence 0.5863 rather than
    today's 0.5634)
    at ctx128 -- both legs otherwise at the shipped configuration and on the
    same bfloat8_b weights, so only the program config and shard layout differ.
    Core count was swept: 8 (one per bank) beats 16, 32 and 64 on both matmuls,
    because past one shard per bank the extra cores only add mcast traffic. The
    8 is the tuned quantity; the profiler reports ``CORE COUNT`` 80 for these
    rows and ``tt-perf-report`` prints 12, and neither of those was chosen.

    In the archived profiles the two projections go 57.06 -> 27.33 us (qkv) and
    72.80 -> 21.91 us (wo). Only ``wo`` was SLOW-classified interleaved; qkv was
    already DRAM-classified, so its gain is duration, not a change of class.

    Batch is capped at 32 here -- see ``_dram_sharded_usable`` -- which is where
    ``nlp_create_qkv_heads_decode`` caps it anyway, on either path.

    Everything between the two projections is identical to ``attention_decode``,
    including the Blackhole staging workarounds, so the two stay diffable.
    """
    if not _dram_sharded_usable(weights, int(x.shape[-2])):
        return attention_decode(x, weights.attention, config, cos_cache, sin_cache, kv_cache, current_pos, token_index)

    k_cache, v_cache, page_table = kv_cache.k, kv_cache.v, kv_cache.page_table
    k_qkv, n_qkv = int(weights.wqkv_decode.shape[-2]), int(weights.wqkv_decode.shape[-1])
    k_o, n_o = int(weights.wo_decode.shape[-2]), int(weights.wo_decode.shape[-1])

    attn_compute_config = _attention_compute_kernel_config(x.device(), precision)
    x_sharded = ttnn.to_memory_config(x, _width_sharded_l1(k_qkv))
    xqkv = ttnn.linear(
        x_sharded,
        weights.wqkv_decode,
        program_config=_dram_sharded_program_config(k_qkv, n_qkv),
        memory_config=_width_sharded_l1(n_qkv),
        dtype=precision.activation_dtype,
        compute_kernel_config=attn_compute_config,
    )
    ttnn.deallocate(x_sharded)

    # nlp_create_qkv_heads_decode wants interleaved L1. (It also must not be
    # handed a DRAM tensor at all on Blackhole -- tt-metal #16667 zeroes
    # odd-indexed Q rows via a NoC DRAM-read alignment violation.)
    xqkv = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(xqkv)

    # rms_norm wants interleaved DRAM while paged_update_cache requires a
    # *sharded* update tensor, so remember the split's layout and restore it.
    kv_sharded_mem = k.memory_config()
    q = _per_head_rms_norm(
        ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), weights.attention.q_norm, config.rms_norm_eps
    )
    k = _per_head_rms_norm(
        ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG), weights.attention.k_norm, config.rms_norm_eps
    )
    # ``rope`` defaults to ``None`` and therefore to ``_apply_rope``, which is
    # what every caller uses -- including the shipped multichip decode path --
    # and what every number in this file was measured at. It is a seam, not a
    # switch: stage 04 used it to build and measure a Meta-ordered
    # ``rotary_embedding_llama`` alternative (3.05x faster standalone and
    # bit-identical) without disturbing the 1x1 baseline the multichip documents
    # compare against. That alternative is **rejected** -- the KV cache carries
    # the rotary's channel convention and prefill writes HF-ordered keys, so it
    # is not a decode-local change. See ``multichip_decoder._meta_rope`` and
    # ``doc/optimized_multichip_decoder/README.md`` limitation 4.
    _rope = _apply_rope if rope is None else rope
    q = _rope(q, cos_cache, sin_cache, token_index)
    k = ttnn.to_memory_config(_rope(k, cos_cache, sin_cache, token_index), kv_sharded_mem)

    # Deliberately NOT cast to the cache dtype, unlike the prefill fill writers.
    # ``paged_update_cache`` requires a FLOAT32/BFLOAT16 update and converts into
    # the cache itself (measured: bfp8 cache + bf16 update round-trips at PCC
    # 0.999969, bfp8 update is rejected at
    # ``paged_update_cache_device_operation.cpp:296``). See
    # ``functional_decoder.match_cache_dtype`` for the full table.
    ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=current_pos, page_table=page_table)
    ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=current_pos, page_table=page_table)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    if kv_cache.is_paged:
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=config.head_dim**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
        )
    else:
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=current_pos,
            scale=config.head_dim**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
        )
    ttnn.deallocate(q)

    attn = ttnn.to_memory_config(_concat_heads_decode(attn, config), _width_sharded_l1(k_o))
    out = ttnn.linear(
        attn,
        weights.wo_decode,
        program_config=_dram_sharded_program_config(k_o, n_o),
        memory_config=_width_sharded_l1(n_o),
        dtype=precision.activation_dtype,
        compute_kernel_config=attn_compute_config,
    )
    ttnn.deallocate(attn)
    return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)


def _experts_chunk_packed(
    hidden: ttnn.Tensor,
    routing: ttnn.Tensor,
    weights: OptimizedWeights,
    config: MoEConfig,
    sparsity_base: ttnn.Tensor,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """One 32-token chunk with gate and up computed in a single matmul.

    The win is core occupancy, not the saved kernel launch.
    ``_sparse_matmul_config`` parallelises only over N, so the usable core count
    is capped by the number of N tiles:

        gate or up alone   N = 768  -> 24 tiles -> 24 cores
        gate+up packed     N = 1536 -> 48 tiles -> 48 cores
        down               N = 2048 -> 64 tiles -> 64 cores

    which is also why the stage-01 profile showed down running at 127 GB/s
    while gate/up sat at 64 GB/s. Worth 1.09x against a *tuned* separate
    candidate (2 x 1.476 = 2.952 ms -> 2.699 ms); the larger figure it shows
    against an untuned one belongs to the block-width fix, not to packing.
    """
    chunk_len = hidden.shape[2]
    n_experts = config.num_experts
    hidden_size = config.hidden_size
    inter = config.moe_intermediate_size
    group_size = chunk_len // EXPERT_CHUNK_SIZE

    device = hidden.device()
    compute_config = _expert_compute_kernel_config(device, precision)
    output_tile = ttnn.Tile([32, 32])
    gate_up_config = _tuned_sparse_matmul_config(
        EXPERT_CHUNK_SIZE, 2 * inter, hidden_size, precision.experts_gate_up_in0_block_w
    )
    down_config = _tuned_sparse_matmul_config(EXPERT_CHUNK_SIZE, hidden_size, inter, precision.experts_down_in0_block_w)

    hidden_grouped = ttnn.reshape(hidden, (1, group_size, EXPERT_CHUNK_SIZE, hidden_size))
    sparsity = ttnn.repeat(sparsity_base, (1, 1, group_size, 1))
    nnz = n_experts * group_size

    fused = ttnn.sparse_matmul(
        hidden_grouped,
        weights.gate_up_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_config,
        compute_kernel_config=compute_config,
        dtype=precision.activation_dtype,
    )
    ttnn.deallocate(hidden_grouped)
    packed_width = fused.shape[-1]
    fused = ttnn.reshape(ttnn.transpose(fused, 1, 3), (1, n_experts, chunk_len, packed_width))

    # gate is the first half -- matches Qwen3MoeExperts.forward's chunk(2, dim=-1)
    half = packed_width // 2
    gate = ttnn.slice(fused, [0, 0, 0, 0], [1, n_experts, chunk_len, half])
    up = ttnn.slice(fused, [0, 0, 0, half], [1, n_experts, chunk_len, packed_width])
    ttnn.deallocate(fused)

    down_input = ttnn.reshape(ttnn.mul(ttnn.silu(gate), up), (1, n_experts, chunk_len, half))
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity_base,
        nnz=n_experts,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        compute_kernel_config=compute_config,
        dtype=precision.activation_dtype,
    )
    ttnn.deallocate(down_input)

    states = ttnn.reshape(down, (1, n_experts, chunk_len, hidden_size))
    states = ttnn.mul(states, ttnn.permute(routing, (0, 3, 2, 1)))
    states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(states, dims=[1]))
    return ttnn.reshape(states, (1, 1, chunk_len, hidden_size))


def moe_prefill_optimized(
    x: ttnn.Tensor,
    routing: ttnn.Tensor,
    weights: OptimizedWeights,
    config: MoEConfig,
    sparsity_base: ttnn.Tensor,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """Expert pass over a sequence. ``x`` ``[1, 1, S, H]``, any S.

    Non-aligned lengths are zero-padded to a chunk boundary and sliced back;
    padded rows carry an all-zero routing vector so they contribute nothing.
    """
    seq_len = x.shape[2]
    padded_len = math.ceil(seq_len / EXPERT_CHUNK_SIZE) * EXPERT_CHUNK_SIZE

    if padded_len != seq_len:
        pad = [(0, 0), (0, 0), (0, padded_len - seq_len), (0, 0)]
        x = ttnn.pad(x, pad, value=0.0)
        routing = ttnn.pad(routing, pad, value=0.0)

    outputs = []
    for start in range(0, padded_len, EXPERT_CHUNK_SIZE):
        end = start + EXPERT_CHUNK_SIZE
        outputs.append(
            _experts_chunk_packed(
                ttnn.slice(x, [0, 0, start, 0], [1, 1, end, config.hidden_size]),
                ttnn.slice(routing, [0, 0, start, 0], [1, 1, end, config.num_experts]),
                weights,
                config,
                sparsity_base,
                precision,
            )
        )
    out = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
    if padded_len != seq_len:
        out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, seq_len, config.hidden_size])
    return out


def moe_decode_optimized(
    x: ttnn.Tensor,
    routing: ttnn.Tensor,
    weights: OptimizedWeights,
    config: MoEConfig,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """Decode MoE with gate/up packed and per-token sparsity. ``x`` ``[1, 1, batch, H]``.

    Tokens are carried as *batch* indices (``[1, B, 1, H]``) rather than along M.
    ``sparse_matmul`` indexes its sparsity tensor by batch dims, so this is what
    makes the pattern per-token -- and unlike prefill it costs nothing here,
    because decode's M is genuinely 1.

    The two matmuls need different sparsity flags, which is not obvious and is
    what previously limited this path to a single user. From
    ``sparse_matmul_device_operation.cpp``::

        a_sparse && b_sparse -> batch_length = batch_length_B
        a_sparse             -> batch_length = batch_length_A
        neither              -> batch_length = batch_length_A * batch_length_B

    ``is_input_b_sparse`` defaults to true. gate/up takes the third branch and
    gets ``B * E``, which matches a ``[1, 1, B, E]`` sparsity tensor. The down
    projection has a sparse activation, so it would take the *first* branch and
    get just ``E`` -- ignoring the batch entirely and rejecting any B > 1.
    Passing ``is_input_b_sparse=False`` selects the second branch instead, so
    down sees ``batch_length_A = B * E`` and matches.

    The rank juggling is deliberate. ``sparse_matmul`` returns
    ``[1, 1, B, E, 1, N]`` and pads that M=1 to a full 32-row tile, so the
    result is 97% padding; reshaping to a compact ``[B, E, N]`` before the
    elementwise work makes those ops touch 192 tiles instead of 6144. Dropping
    the reshapes and staying rank-6 measured 6% *slower*.

    Applying the routing weight to ``down``'s *input* instead -- equivalent,
    since ``down`` is linear, and the input is the compact tensor -- was
    measured and is a tie (0.5852 vs 0.5862 ms, one run, at the pre-§7
    configuration); see the module docstring for why the variant that looked
    much better than that was not.
    """
    batch = x.shape[2]
    n_experts = config.num_experts
    hidden_size = config.hidden_size
    inter = config.moe_intermediate_size
    nnz = config.num_experts_per_tok * batch

    sparsity = ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT)
    expert_memory_config = _decode_expert_memory_config(batch, config)
    output_tile = ttnn.Tile([32, 32])
    compute_config = _expert_compute_kernel_config(x.device(), precision)
    gate_up_config = _tuned_sparse_matmul_config(1, 2 * inter, hidden_size, precision.experts_gate_up_in0_block_w)
    down_config = _tuned_sparse_matmul_config(1, hidden_size, inter, precision.experts_down_in0_block_w)

    x_batched = ttnn.reshape(x, (1, batch, 1, hidden_size))
    fused = ttnn.sparse_matmul(
        x_batched,
        weights.gate_up_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=expert_memory_config,
        output_tile=output_tile,
        program_config=gate_up_config,
        compute_kernel_config=compute_config,
        dtype=precision.activation_dtype,
    )
    packed_width = fused.shape[-1]
    fused = ttnn.reshape(fused, (batch, n_experts, packed_width))

    # gate is the first half -- matches Qwen3MoeExperts.forward's chunk(2, dim=-1)
    half = packed_width // 2
    gate = ttnn.slice(fused, [0, 0, 0], [batch, n_experts, half])
    up = ttnn.slice(fused, [0, 0, half], [batch, n_experts, packed_width])
    ttnn.deallocate(fused)

    down_input = ttnn.reshape(ttnn.mul(ttnn.silu(gate), up), (batch, n_experts, 1, half))
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=nnz,
        memory_config=expert_memory_config,
        output_tile=output_tile,
        program_config=down_config,
        is_input_a_sparse=True,
        is_input_b_sparse=False,  # see docstring: selects batch_length_A = B * E
        compute_kernel_config=compute_config,
        dtype=precision.activation_dtype,
    )
    ttnn.deallocate(down_input)

    states = ttnn.reshape(down, (batch, n_experts, hidden_size))
    states = ttnn.mul(states, ttnn.reshape(routing, (batch, n_experts, 1)))
    states = ttnn.unsqueeze_to_4D(ttnn.sum(states, dim=1))
    return ttnn.reshape(states, (1, 1, batch, hidden_size), (1, 1, max(32, batch), hidden_size))


def decoder_layer_prefill_optimized(
    x: ttnn.Tensor,
    weights: DecoderLayerWeights,
    config: DecoderLayerConfig,
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    sparsity: ttnn.Tensor,
    packed_experts: OptimizedWeights,
    kv_cache: KVCache | None = None,
    user_id: int = 0,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """Optimized prefill. Same contract as ``decoder_layer_prefill``.

    Attention runs the interleaved bfloat8_b copy: the DRAM-sharded program
    config is decode-only (``per_core_M=1``), and a plain ``ttnn.linear`` cannot
    read a DRAM-sharded weight at all.
    """
    eps = config.rms_norm_eps

    normed = ttnn.rms_norm(x, weight=weights.input_layernorm, epsilon=eps)
    attn_out = attention_prefill(
        normed, packed_experts.attention, config.attention, cos_cache, sin_cache, kv_cache, user_id
    )
    ttnn.deallocate(normed)
    hidden = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)

    normed = ttnn.rms_norm(hidden, weight=weights.post_attention_layernorm, epsilon=eps)
    routing = router_forward_optimized(normed, weights.router, config.moe)
    moe_out = moe_prefill_optimized(normed, routing, packed_experts, config.moe, sparsity, precision)
    ttnn.deallocate(normed)
    ttnn.deallocate(routing)

    out = ttnn.add(hidden, moe_out)
    ttnn.deallocate(hidden)
    ttnn.deallocate(moe_out)
    return out


def decoder_layer_decode_optimized(
    x: ttnn.Tensor,
    weights: DecoderLayerWeights,
    config: DecoderLayerConfig,
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    kv_cache: KVCache,
    current_pos: ttnn.Tensor,
    token_index: int,
    *,
    packed_experts: OptimizedWeights,
    precision: PrecisionConfig = DEFAULT_PRECISION,
) -> ttnn.Tensor:
    """Optimized decode. Decode already used per-token sparsity in stage 01."""
    eps = config.rms_norm_eps

    normed = ttnn.rms_norm(x, weight=weights.input_layernorm, epsilon=eps)
    attn_out = attention_decode_optimized(
        normed,
        packed_experts,
        config.attention,
        cos_cache,
        sin_cache,
        kv_cache,
        current_pos,
        token_index,
        precision=precision,
    )
    ttnn.deallocate(normed)
    hidden = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)

    normed = ttnn.rms_norm(hidden, weight=weights.post_attention_layernorm, epsilon=eps)
    routing = router_forward_optimized(normed, weights.router, config.moe)
    moe_out = moe_decode_optimized(normed, routing, packed_experts, config.moe, precision)
    ttnn.deallocate(normed)
    ttnn.deallocate(routing)

    out = ttnn.add(hidden, moe_out)
    ttnn.deallocate(hidden)
    ttnn.deallocate(moe_out)
    return out


__all__ = [
    "EXPERT_CHUNK_SIZE",
    "router_forward_optimized",
    "EXPERT_WEIGHT_DTYPE",
    "EXPERT_MATH_FIDELITY",
    "ATTENTION_WEIGHT_DTYPE",
    "PrecisionConfig",
    "DEFAULT_PRECISION",
    "moe_prefill_optimized",
    "moe_decode_optimized",
    "attention_decode_optimized",
    "OptimizedWeights",
    "PackedExpertWeights",
    "upload_optimized_weights",
    "upload_packed_expert_weights",
    "decoder_layer_prefill_optimized",
    "decoder_layer_decode_optimized",
    "build_rope_cache",
    "build_expert_sparsity",
    "create_kv_cache",
    "upload_layer_weights",
    "upload_router_weight",
    "DecoderLayerConfig",
    "DecoderLayerWeights",
    "KVCache",
    "MoEConfig",
    "AttentionConfig",
    "AttentionWeights",
]
