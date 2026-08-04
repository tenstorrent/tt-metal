# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-TTS Block 1: the 3.4B autoregressive backbone, on TTNN.

Ours. It replaced a `models/tt_transformers` wrapper, which is gone -- see git history if you
need it (`tt/ttnn_voxtral_backbone.py`, removed once this beat it on every metric). The rationale
for owning this is in STATUS.md's Block 1 section.

Measured against the fp32 CPU reference on REAL prompts (never random ones -- STATUS.md trap #12),
with `tt_transformers` on the same metric for comparison:

                             ours            ours all-BFP8   tt_transformers
    prefill, last position     0.999883        0.999881        0.999564
    decode step                0.99985+        0.99986         0.981
    decode ms/step             25.7            33.6            48

There is deliberately no end-to-end WER row: the same code at three seeds spans 0.88-2.06% on that
metric, so it cannot separate two builds. The gate is long-form WER (0.00% over 298 words) plus the
teacher-forced numbers above. STATUS.md 6.7 has the seed table; read it before quoting any WER.

W2 IS THE ONE WEIGHT STILL IN bf16, and w2 is the hang -- not BFP8 in general. wqkv, wo, FF1 and
FF3 are all BFP8 and were each measured on their own; only w2 brings back the card-wedging hang in
multi-utterance runs. Diagnosis in ttnn_voxtral_pipeline, repro at WEIGHT_DTYPE below.

The decode gap against tt_transformers (0.99985+ vs 0.981) is real and reproduces at every weight
dtype we tried, so it is not ours to explain -- but note it is NOT caused by sdpa_decode, which is
nearly free at these shapes.

Mirrors `voxtral_backbone_ref._layer` op-for-op. Structurally this is Block 2's `_block` plus
three things -- RoPE, a causal mask in PREFILL, and a KV cache -- so what Block 2 already proved
carries over unchanged: the row fold, q/k/v fused into one weight, HiFi4 with fp32_dest_acc_en,
bf16 activations. And so does its central lesson, which decode needed twice: a
BATCHED matmul costs per batch element, so fold whatever you can into ROWS (see `_layer_step`).

DECODE IS BOUND BY WEIGHT BYTES, so the weight dtype is the speed. At the shipped precision the
six linears stream ~3.9 GB per step and measure ~20.3 ms, i.e. 194 GB/s -- the ceiling a plain
interleaved matmul reaches, and hand-tuned matmul program configs measured SLOWER (169 vs 193 GB/s
on wq). There is no layout trick left in them: only fewer BYTES help, and the only weight still in
bf16 is w2, which is the pinned hang trigger (see WEIGHT_DTYPE). So this line is finished.

WHERE THE TIME GOES, per decode frame, steady state on one N150 (measure with
the scratch probe_perf.py harness; the numbers below are case 2, 448 frames):

    six linears, weight streaming     ~20.7 ms   AT THE CEILING -- 194 GB/s, and a plain
                                                 interleaved matmul cannot do better here.
                                                 Tuned matmul program configs are SLOWER
                                                 (169 vs 193 GB/s on wq). Only fewer BYTES help.
    rms_norm x2                         6.0 ms   CLOSED, twice. Dropping fp32 accumulation is
                                                 2.4x and takes model PCC to 0.992; width-sharding
                                                 it (which KEEPS fp32 acc) is 1.46x on the
                                                 norm+linear pair and still doubles the worst
                                                 sample, 1.06% -> 1.95%. See _norm.
    sdpa_decode                         1.8 ms   68 us/layer at pos~200; grows with cache length
    everything else                    ~2.9 ms   qkv heads, rope, 2x cache write, reshapes,
                                                 residual adds -- all of it under 4% now, which
                                                 is the decode-native layout's payoff
    ------------------------------------------
    Block 1 total                      25.7 ms   (Block 2 is now ~28, so Block 1 is still
                                                 the larger half -- see ttnn_voxtral_flow)

So the only remaining lever on Block 1 is WEIGHT BYTES, and that is capped by the hang: BFP8 on
FF1_FF3 is safe, adding FF2 reintroduces it (see WEIGHT_DTYPE). Everything else is small change --
and the one thing that looked like an exception, a width-sharded RMSNorm worth ~5 ms, failed the
WER gate; read _norm before trying it again.

That idea is now spent: wqkv and wo ARE in BFP8 as of the second sweep, worth 3.32 ms/frame with
no accuracy cost on mean or p90 worst-sample and no hang. w2 is the only weight left in bf16, and
it is the pinned hang trigger, so the byte lever is finished unless that changes.

TWO THINGS THAT ARE NOT LIKE BLOCK 2:

1. ROPE, AND WE PERMUTE THE WEIGHTS TO AVOID THE HARD VERSION. The checkpoint is Mistral-native, so
   q/k are stored for INTERLEAVED-pair rotation (r1,i1,r2,i2,...). Applying that on device means
   shuffling even/odd lanes inside a tile, which is awkward. Instead we permute wq/wk ONCE at load
   time into the half-split layout (r1,r2,...,i1,i2,...) and then apply the easy `rotate_half` form.
   The two are equivalent, and the permute is the same one `scripts/export_backbone_hf` uses and
   asserts bit-exact. Getting this wrong does NOT raise -- it produces fluent nonsense.

2. `n_heads * head_dim` (4096) != `dim` (3072), so wq and wo are NOT square. Anything assuming
   dim throughout will be silently wrong on shapes that happen to broadcast.
"""


import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_backbone_ref import load_backbone_state
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    DEFAULT_CKPT,
    DIM,
    HEAD_DIM,
    HIDDEN_DIM,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    NORM_EPS,
    ROPE_THETA,
)

SCALE = HEAD_DIM**-0.5
Q_WIDTH = N_HEADS * HEAD_DIM          # 4096, deliberately != DIM
TILE = 32

# Prefill pads its sequence to this. Correctness needs only TILE -- nothing here shards the
# sequence across the core grid, which is what forced the tt_transformers path to 256.
#
# WHY PAD AT ALL, given that ttnn handles a sub-tile remainder itself: our prefill builds an
# EXPLICIT [1,1,Sp,Sp] causal mask, and a tile-aligned length keeps mask, scores and softmax from
# disagreeing at a ragged edge. Implementations that use sdpa(is_causal=True) materialise no mask
# and so need no padding at all -- that is what the XTTS-v2 GPT and the ign/voxtral_p150_qb2
# branch both do. Measured here, sdpa prefill costs 0.99988 -> 0.99977 PCC at the LAST position,
# the one value that seeds every decode step, and saves ~30 ms of a ~100 ms prefill. Not worth it;
# see below for why 30 ms is noise.
#
# WHY 128 SPECIFICALLY. Every op in prefill carries the sequence dimension -- the norms, all five
# linears, the head split, RoPE, fill_cache, and the [1,32,Sp,Sp] score tensor -- so each distinct
# Sp is its own set of compiled kernels. The 15 fixture prompts span P=74..357: unpadded that is
# 15 shape-sets, at 128 it is three (128/256/384). Padding also caps the QUADRATIC term at three
# known sizes; the worst case is P=357 -> 384, costing (384/357)^2 = 1.16x on attention.
#
# The repo's shared helper, tt_transformers' get_padded_prefill_len, uses a coarser ladder --
# 128, then 1024, then powers of two. Do not adopt it here: it would send our 357 to 1024, i.e.
# 7x the quadratic work, because it is tuned for LLM serving where prompts vary by orders of
# magnitude. Ours are bounded by a voice preset plus a sentence.
#
# PREFILL COST HAS THREE TIERS, which is why "prefill is slow" and "prefill is free" are both
# quoted in this repo's history:
#     first ever at a shape, cold disk kernel cache      ~6 s
#     first in a process, warm disk / empty program cache ~1.5 s
#     subsequent in the same process                     ~100-146 ms
# A long-lived server pays tier 2 once per shape then tier 3 forever, so prefill settles at ~0.4%
# of a 36 s utterance -- less than Block 3's codec pass. Fewer shapes means fewer tier-1 and
# tier-2 hits, which is the second reason to keep the padding.
PREFILL_MULTIPLE = 128
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
DTYPE = ttnn.bfloat16

# DECODE'S INTERMEDIATES LIVE IN L1, not DRAM -- the same finding as Block 2's `_L1`
# (ttnn_voxtral_flow), and it transfers: 26.43 -> 25.53 ms/step for NO accuracy change at all (min
# PCC 0.999850, mean worst-sample 0.85%, p90 1.09% -- byte-identical before and after over 44
# teacher-forced frames). Decode's values are 6-24 KB and each is consumed within an op or two, so
# a DRAM round trip per intermediate is pure latency.
#
#     shipped                          26.43 ms
#     + wo output and residual L1       26.19    1.009x
#     + MLP intermediates (g, u) L1     25.53    1.035x   <- shipped
#
# TWO THINGS THAT DO NOT PAY, so they are not done. sdpa_decode's output stays forced to DRAM:
# routing it to L1 instead measures 0.999x, i.e. nothing (that `to_memory_config` looks like an
# obvious round trip to remove, and is not). And in Block 2 the norm output was likewise neutral.
# The pattern is narrower than "L1 is faster": it is values with a consumer close behind.
_L1 = ttnn.L1_MEMORY_CONFIG

# RMSNORM, WIDTH-SHARDED, for the DECODE shape. The interleaved op costs 115 us on a [1,1,3072] row
# -- latency, not arithmetic: one core reduces the whole row with a DRAM round trip either side.
# Sharded it is 44 us including BOTH memory_config moves, i.e. ~4.9 ms/frame over 52 calls.
#
# fp32 accumulation is UNCHANGED, so this is NOT the rejected "drop the compute config" trade.
#
# THE CORE COUNT BARELY MATTERS and the reason is worth knowing: 2/4/8 cores measure 42.4/40.5/44.1
# us, flat, because the norm COMPUTE is ~16 us at any of them and the two to_memory_config calls are
# the other ~28. The reshard is the tax, not the reduction. 8 is used because it is marginally the
# fastest end to end (26.54 vs 27.50 ms/step at 2 cores).
#
# THE SECOND RESHARD CANNOT BE DODGED, and both ways of trying are now measured.
#
# Feeding the sharded result straight to the DEFAULT matmul is slower -- 8.94 vs 5.32 ms per 26
# norm+linear pairs -- and the reason is the AXIS. Width-sharding splits the matmul's CONTRACTION
# dimension, so each core can only form a partial sum and the cross-core reduce is full-output-sized
# ([32,6144] x 8). Interleaved, ttnn splits by OUTPUT COLUMNS instead: each core owns its columns
# outright, reads the whole 6 KB activation, and there is nothing to reduce. The same axis that makes
# the NORM fast (it reduces over width, so the cross-core step is 8 scalars) makes the matmul slow.
#
# AND YOU CANNOT SHARD ONLY THE WEIGHT INSTEAD. Width-sharding the WEIGHT splits the OUTPUT columns,
# which is the axis that needs no reduction, so it ought to be the free version -- but ttnn couples
# the two. A width-sharded in1 is accepted by ONLY the DRAM-sharded config, which also requires a
# width-sharded in0; every other config asserts `in1.memory_config().memory_layout() == INTERLEAVED`
# (matmul_device_operation.cpp:1188ff, and the comment at :1199 states the pairing outright). So
# "sharded weight, interleaved activation" is not an expressible combination, and the paired form is
# the one measured below. An L1-RESIDENT weight fails the same assertion, and would be capped anyway:
# Block 1 streams ~3.9 GB/frame against 96 MB of total L1, i.e. 2.4% of the model. The model not
# fitting in L1 is exactly WHY 194 GB/s is the wall.
#
# `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` REQUIRES a width-sharded activation, so it
# is the one config that wants what the norm already produces. It BUILDS here -- Block 1's wqkv is
# per_core_N=24 tiles where Block 2's N=9216 overflowed L1 at 36 (STATUS.md) -- and it is still
# slower: 125.4 us against 100.9, or 128.5 charging the output unshard, i.e. 0.72 ms/frame WORSE,
# and not bit-exact (4.9e-04). No mystery: the default path already runs this matmul at ~198 GB/s,
# so there is no bandwidth left for the DRAM-sharded machinery to win back. Closed both ways.
#
# THIS WAS REVERTED ONCE AND THE REVERT WAS WRONG. It was measured while wqkv and wo were still
# bf16, where it read mean worst-sample 0.86% -> 0.92%. On the current weights it reads 0.86% ->
# 0.84% mean and 1.10% -> 1.06% p90, i.e. no cost, reproduced twice. The lesson is that a precision
# change here is not separable from the others -- re-measure against the CURRENT config, never
# against a recorded number.
_NORM_GRID_X = 8
_NORM_SHARD = ttnn.create_sharded_memory_config(
    shape=(1, 1, TILE, DIM), core_grid=ttnn.CoreGrid(y=1, x=_NORM_GRID_X),
    strategy=ttnn.ShardStrategy.WIDTH)
_NORM_PRG = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(_NORM_GRID_X, 1), subblock_w=4, block_h=1,
    block_w=DIM // _NORM_GRID_X // TILE, inplace=False)

# Decode runs in ttnn's DECODE-NATIVE head layout, [1, batch, heads, head_dim], and these are the
# memory configs those ops demand. At batch 1 the layout lines everything up for free:
# nlp_create_qkv_heads_decode emits q exactly as sdpa_decode wants it and k/v exactly as
# paged_update_cache wants them, already sharded -- so there is no permute, no hand-built shard,
# no cache slice and no attention mask (sdpa_decode bounds the cache with cur_pos instead).
# Worth 6.6 ms/frame over a hand-rolled interior, at the same decode PCC.
#
# THE 8 IS NOT THE CORE COUNT and it does not matter -- measured, so nobody re-asks. The device has
# 64 Tensix cores (8x8); this uses one row. Swept over every head-aligned option (6144 = 48 heads x
# 128, so the core count must divide 48: 8/12/16/24/48 qualify, 32 does not) and the whole decode
# step reads 31.36-31.46 ms across all of them, identical PCC. The shard fill plus the head split
# is a fraction of a millisecond out of a step that is ~20 ms of pure weight streaming, so how the
# 393 KB is spread is invisible.
#
# Contrast the NORM's grid (_NORM_SHARD in ttnn_voxtral_flow), where the count DOES matter: 16.2 us
# on 8 cores against 21.2 on 32. That op is a REDUCTION across the row, so more cores means more
# partial sums to combine and more cores to wait on -- and there the op IS the thing being timed.
_QKV_WIDTH = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM      # 6144, one fused projection
_QKV_SHARD = ttnn.create_sharded_memory_config(
    (TILE, _QKV_WIDTH // 8), core_grid=ttnn.CoreGrid(y=1, x=8),
    strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True)
# rotary_embedding_hf's decode mode requires cos/sin sharded as well as the input
# ("Cos must be sharded in decode mode"), one tile row on one core at batch 1.
_ROPE_SHARD = ttnn.create_sharded_memory_config(
    (TILE, HEAD_DIM), core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)

# WEIGHT PRECISION -- load-bearing for CORRECTNESS, not just speed.
#
# BFP8 EVERYWHERE EXCEPT w2. Decode is bandwidth-limited on weight bytes and the six linears run at
# the 194 GB/s DRAM ceiling, so the weight dtype IS the speed; every matrix that can be halved has
# been. Each was measured on its own:
#
#     weights                       decode PCC   ms/step   mean worst-sample   hang?
#     bf16 (all)                      0.99991     45.8            -             no
#     + BFP8 on FF1, FF3              0.999884    34.7          0.86%           no
#     + BFP8 on wqkv, wo   <- this    0.999852    31.4          0.86%           no
#     + BFP8 on w2 (i.e. all)         0.99977+    28.9            -             HANGS
#
# W2 IS THE HANG, not BFP8 in general -- a distinction that cost a long investigation and is worth
# keeping straight, because "all-BFP8 hangs" was true but over-attributed and it froze this line of
# work for a while. wqkv and wo were simply never tried alone; when they were, they cost 3.32
# ms/frame with mean and p90 worst-sample unchanged, and no hang.
#
# W2 WAS RETRIED, AFTER wqkv AND wo TURNED OUT FINE, AND IT STILL HANGS -- so the pin is real and
# specific to w2, not an artefact of the old all-BFP8 test. It is worth 2.5 ms/step (31.4 -> 28.9)
# and PCC holds at 0.99977-0.99985, so the only thing stopping it is the hang.
#
# What the retry added: it now hangs EARLIER and HARDER than documented. The old repro died on the
# third utterance inside Block 3; this died during the FIRST case, right after the first compute
# op, with no pipeline output at all. So the trigger is no longer the five-condition sequence
# below -- today's op mix (row fold, sharded norm, qkv fusion, device semantic head) reaches it
# sooner. Do not expect the documented repro to be the minimal one any more.
#
# Recovery is a board reset. `tt-smi` is not on PATH here; the Wormhole build lives at
# /home/software/syseng/wh/tt-smi and the command is `-wr 0` (this vintage has no plain `-r`).
# open_device simply hangs until you do it.
#
# The old five-condition repro, kept because it is still the cheapest way to test a CHANGE that is
# meant to fix this: short gen + 128-bucket codec decode, then two long gens that both land in the
# 512 bucket, the second a pure cache HIT. The shipped config clears it and the full 15-case set. Full diagnosis in ttnn_voxtral_pipeline.
#
# Judge any change here on MEAN and P90 worst-sample, never on max: max is an order statistic and
# moved 1.28-4.28% across configs non-monotonically (STATUS.md 6.8).
WEIGHT_DTYPE = ttnn.bfloat16          # w2 only
FF_WEIGHT_DTYPE = ttnn.bfloat8_b      # FF1 and FF3
ATTN_WEIGHT_DTYPE = ttnn.bfloat8_b    # wqkv and wo


def interleaved_to_halfsplit(t, n_heads):
    """Mistral-native (interleaved-pair) q/k weight -> half-split layout, so `rotate_half` applies.

    Identical to `scripts/export_backbone_hf.meta_to_hf_permute`, which asserts the round trip
    bit-exact. `t` is torch [out, in]; rows are grouped per head.
    """
    d1, d2 = t.shape
    return t.view(n_heads, d1 // n_heads // 2, 2, d2).transpose(1, 2).reshape(d1, d2)


def rope_tables(seq_len, offset=0, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """-> (cos, sin) torch [seq_len, head_dim], each half duplicated for the half-split form.

    `rope_cis` in the reference returns complex [S, d/2] for interleaved pairs; the half-split form
    wants the same angles laid out as [cos(theta), cos(theta)] so one elementwise multiply covers
    both halves.
    """
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    ang = torch.outer(torch.arange(offset, offset + seq_len, dtype=torch.float64), inv)
    return (torch.cat([ang.cos(), ang.cos()], dim=-1).float(),
            torch.cat([ang.sin(), ang.sin()], dim=-1).float())


class TtVoxtralGPT:
    """Block 1 on device. prefill(embeds) -> hidden; step(embed) -> hidden, sharing a KV cache."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, n_layers=N_LAYERS, state=None,
                 max_seq_len=2048):
        """`state` takes an already-loaded `load_backbone_state` dict. Pass it when the caller
        also needs the fp32 reference weights: that dict is ~13 GB, and loading it twice is the
        difference between comfortable and swapping.

        `max_seq_len` sizes the KV cache: 2 x n_layers x 8 x max_seq x 128 bf16, i.e. 218 MB at
        2048. Pass 0 to skip it (prefill-only harnesses; `step` then raises).
        """
        self.device = device
        self.dtype = DTYPE
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pos = 0
        wd = WEIGHT_DTYPE
        attnd = ATTN_WEIGHT_DTYPE
        w = state if state is not None else load_backbone_state(ckpt_path)

        up = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT,
                                         device=device)
        ffd = FF_WEIGHT_DTYPE or wd                         # FF1_FF3 may differ; see WEIGHT_DTYPE
        vec = lambda t: up(t.reshape(1, 1, -1), DTYPE)      # norm gammas: no bandwidth, keep bf16
        lin = lambda t, d=None: up(t.t(), d or wd)          # torch [out,in] -> ttnn wants [in,out]

        self.norm = vec(w["norm"])
        self.layers = []
        for i in range(n_layers):
            p = f"layers.{i}."
            wq = interleaved_to_halfsplit(w[p + "attention.wq"], N_HEADS)
            wk = interleaved_to_halfsplit(w[p + "attention.wk"], N_KV_HEADS)  # n_kv, not n_heads
            self.layers.append({
                "an": vec(w[p + "attention_norm"]),
                "fn": vec(w[p + "ffn_norm"]),
                # q, k and v fused into ONE weight: one matmul and one weight stream instead of
                # two, and it is what the decode-native head op expects. Same bytes as the old
                # wq + wkv pair, so this costs no memory.
                "wqkv": lin(torch.cat([wq, wk, w[p + "attention.wv"]], dim=0), attnd),
                "wo": lin(w[p + "attention.wo"], attnd),
                "w1": lin(w[p + "feed_forward.w1"], ffd),
                "w2": lin(w[p + "feed_forward.w2"]),
                "w3": lin(w[p + "feed_forward.w3"], ffd),
            })
        self._assert_shapes()
        # Allocated once and written in place, so a generation never reallocates. Zero-init is not
        # relied on for correctness -- `step` masks everything above self.pos.
        z = torch.zeros(1, N_KV_HEADS, max_seq_len, HEAD_DIM)
        self.caches = [(up(z, DTYPE), up(z, DTYPE)) for _ in range(n_layers)] if max_seq_len else []

    def reset(self):
        """Start a new utterance. The cache needs no clearing: every position is written before it
        is read, and `step`'s mask covers the rounded-up tail."""
        self.pos = 0

    def _assert_shapes(self):
        """Cheap guard against a silently wrong load: non-square wq/wo are what bite here."""
        exp = {"wqkv": (DIM, _QKV_WIDTH), "wo": (Q_WIDTH, DIM),
               "w1": (DIM, HIDDEN_DIM), "w3": (DIM, HIDDEN_DIM), "w2": (HIDDEN_DIM, DIM)}
        for i, L in enumerate(self.layers):
            for k, e in exp.items():
                got = tuple(L[k].shape)[-2:]
                assert got == e, f"layer {i} {k}: expected {e}, got {got}"

    # ----------------------------------------------------------------------------
    # SHARED PRIMITIVES -- used by both prefill and decode
    # ----------------------------------------------------------------------------
    def _rope(self, x, cos, sin):
        """Half-split RoPE on [1, heads, S, head_dim]: x*cos + rotate_half(x)*sin.

        `rotary_embedding_hf` is the HALF-SPLIT (HF) form, which is the right one HERE only
        because wq/wk were permuted at load; on the unpermuted Mistral-native weights it would be
        silently wrong. `is_decode_mode=False` for both prefill and decode: our decode q is
        [1, heads, 1, d], i.e. the prefill layout at S=1, not the op's batch-major decode layout.

        Measured against an fp32 host rotation this scores the same as the hand-rolled
        slice/neg/concat/mul/mul/add it replaces (0.999996 both, differing by bf16 rounding), and
        it is 7 dispatches fewer per call -- which is what decode is actually bound by.
        """
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=False,
                                                     compute_kernel_config=COMPUTE_CONFIG)

    def _norm(self, x, gamma):
        """RMSNorm.

        The compute config is NOT optional. Dropping it makes this op 2.4x faster (48 us against
        115 on a 6 KB tensor, ~3.5 ms/frame over 26 layers) and takes the MODEL from decode PCC
        0.99991 to 0.992, worst sample 1.7% -> 18.9%. Per-op PCC barely moves (0.999993 vs
        0.999996) -- a 3e-6 difference amplified ~100x through 26 layers, because every norm feeds
        the next residual. The fp32 accumulation in the mean-of-squares is load-bearing.

        NOR IS THE SHARDED FORM FREE, which is the less obvious version of the same trap. This op
        costs ~115 us on a 6 KB tensor -- latency, not arithmetic, one core reducing the row with a
        DRAM round trip either side. Width-sharding it over 8 cores WITH fp32 accumulation intact
        makes the norm+linear pair 1.46x (5.32 vs 7.78 ms per 26), which is ~5 ms/frame over 52
        calls, and it looked free: 0.9999973 against this op, and the decode gate barely moved.
        It is not free. Over 24 REAL teacher-forced frames the WORST SAMPLE went 1.06% -> 1.95%
        while PCC stayed flat at 0.99991. Same amplification as above, same reason, and per-op PCC
        hid it -- gate norm changes here on worst-sample against the fp32 reference, never on PCC.

        Teacher-forced is the load-bearing word: both builds see IDENTICAL inputs at every step, so
        no trajectory is involved and the comparison is deterministic. An earlier version of this
        note also cited natural WER moving 0.88% -> 2.06%. THAT PART WAS WORTHLESS -- the same code
        at seeds 0/1/2 spans 0.88-2.06% all by itself (score_quality_set.py, LONGFORM_MIN_WORDS).
        The worst-sample number is the evidence; the WER number was a coin flip.

        Block 2 DOES use the sharded form (ttnn_voxtral_flow._norm) and is fine: 3 layers instead
        of 26, so there is no 100x to amplify into.
        """
        return ttnn.rms_norm(x, weight=gamma, epsilon=NORM_EPS,
                             compute_kernel_config=COMPUTE_CONFIG)

    def _norm_dec(self, x, gamma):
        """The same RMSNorm at the DECODE shape, width-sharded -- see _NORM_SHARD. Decode only,
        because the program config pins the row count and prefill's S varies per prompt (prefill is
        ~3% of wall time, so it keeps the interleaved op)."""
        h = ttnn.rms_norm(ttnn.to_memory_config(x, _NORM_SHARD), weight=gamma, epsilon=NORM_EPS,
                          compute_kernel_config=COMPUTE_CONFIG, program_config=_NORM_PRG,
                          memory_config=_NORM_SHARD)
        return ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)


    # ----------------------------------------------------------------------------
    # PREFILL PATH -- whole prompt at once, fills the KV cache
    # Runs once per utterance (~1 s), so it is not where the frame budget goes.
    # ----------------------------------------------------------------------------
    def _qkv(self, x, w, S, cos, sin):
        """Pre-norm + fused QKV + RoPE. -> (q,k,v) as [1, heads, S, head_dim], v un-rotated."""
        h = self._norm(x, w["an"])
        qkv = ttnn.linear(h, w["wqkv"], compute_kernel_config=COMPUTE_CONFIG)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(qkv, [1, 1, S, _QKV_WIDTH]), num_heads=N_HEADS,
            num_kv_heads=N_KV_HEADS, transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._rope(qh, cos, sin), self._rope(kh, cos, sin), vh   # v carries no RoPE

    def _attend(self, qh, kh, vh, S, mask):
        """PREFILL attention: [1,32,S,128] x [1,8,S,128] -> merged [1,S,4096], `mask` additive.

        Decode does NOT come through here -- it has its own interior in `_layer_step`, because the
        two want opposite things: prefill has S rows and a triangular mask, decode has one row and
        wins by folding the head batch down (see there).

        The scale/mask/softmax is deliberately three ops. `ttnn.scale_mask_softmax_in_place` fuses
        them but READS ONLY ROW 0 OF THE MASK unless `is_causal_mask=True` -- undocumented, and it
        does not raise. On our [1,1,S,S] triangular mask it silently applied row 0 to every row and
        the 1-layer gate fell to 0.517, which reads exactly like a RoPE convention error. Decode
        uses it safely because its mask genuinely has one row. Here it would save ~1.4 ms of a
        prefill that happens once per utterance, which is not worth depending on `is_causal_mask`
        continuing to mean what our mask assumes.
        """
        rep = N_HEADS // N_KV_HEADS
        kr, vr = ttnn.repeat_interleave(kh, rep, dim=1), ttnn.repeat_interleave(vh, rep, dim=1)
        s = ttnn.matmul(qh, ttnn.transpose(kr, -2, -1), compute_kernel_config=COMPUTE_CONFIG)
        s = ttnn.add(ttnn.multiply(s, SCALE), mask)
        a = ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        a = ttnn.matmul(a, vr, compute_kernel_config=COMPUTE_CONFIG)
        # bit-identical to permute(0,2,1,3) + reshape, in one dispatch
        return ttnn.reshape(ttnn.experimental.nlp_concat_heads(a), [1, S, Q_WIDTH])

    def _mlp(self, x, h, w, mc):
        """Residual + SwiGLU over an ALREADY-NORMED `h`. Shared by prefill and decode.

        `h` and `mc` are both passed in rather than decided here, because they are exactly what the
        two paths do differently, and both are load-bearing:
          * `h` -- decode norms width-sharded (`_norm_dec`), prefill interleaved (`_norm`). See
            _norm for why that distinction has bitten twice.
          * `mc` -- decode keeps intermediates in L1 (see _L1, worth 0.9 ms). Prefill cannot: its
            `g` is [1,S,9216] and S reaches 384, i.e. 6.8 MB, so it passes DRAM.

        `activation="silu"` on the FF1 matmul is bit-identical to a separate `ttnn.silu` and one
        dispatch cheaper.
        """
        g = ttnn.linear(h, w["w1"], activation="silu", compute_kernel_config=COMPUTE_CONFIG,
                        memory_config=mc)
        u = ttnn.multiply(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG,
                                         memory_config=mc), memory_config=mc)
        return ttnn.add(x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG,
                                       memory_config=mc), memory_config=mc)

    def _layer(self, x, w, S, cos, sin, mask, cache=None):
        """x [1,S,3072] -> same. Pre-norm GQA with RoPE + causal mask, then SwiGLU.

        Rows are already folded (batch 1 here), so every linear reads its weights once. Attention is
        the ONLY row-mixing op: it runs on the unfolded [1, heads, S, d] view. Any future row-mixing
        op must go inside that same window -- see ttnn_voxtral_flow._block.

        `cache` is (k, v) [1,8,MAX,128] and is FILLED here, not read: prefill attends to the k/v it
        just computed, so the cache is purely an output. It receives all S PADDED rows; the garbage
        past the real length is never read back because `step` masks everything above `self.pos`.
        """
        qh, kh, vh = self._qkv(x, w, S, cos, sin)
        if cache is not None:
            ttnn.fill_cache(cache[0], kh, 0)     # update_idx 0, so the tile-alignment rule is moot
            ttnn.fill_cache(cache[1], vh, 0)
        a = self._attend(qh, kh, vh, S, mask)
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG))
        return self._mlp(x, self._norm(x, w["fn"]), w, ttnn.DRAM_MEMORY_CONFIG)

    # ----------------------------------------------------------------------------
    # DECODE PATH -- one frame at a time; THIS is the hot loop
    # ----------------------------------------------------------------------------
    def _layer_step(self, x, w, cos, sin, cache, pos_t):
        """One decode position. x [1,1,3072] -> same, against `cache` written up to `pos_t`.

        Head layout is [1, batch, heads, head_dim] here, not prefill's [1, heads, seq, head_dim] --
        see _QKV_SHARD for why that makes the whole interior glue-free. `pos_t` is a DEVICE tensor
        because paged_update_cache and sdpa_decode both take the position that way.
        """
        qkv = ttnn.linear(self._norm_dec(x, w["an"]), w["wqkv"],
                          compute_kernel_config=COMPUTE_CONFIG)
        qkv = ttnn.to_memory_config(ttnn.reshape(qkv, [1, 1, 1, _QKV_WIDTH]), _QKV_SHARD)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS)
        qh = ttnn.experimental.rotary_embedding_hf(qh, cos, sin, is_decode_mode=True,
                                                   compute_kernel_config=COMPUTE_CONFIG)
        kh = ttnn.experimental.rotary_embedding_hf(kh, cos, sin, is_decode_mode=True,
                                                   compute_kernel_config=COMPUTE_CONFIG)
        ttnn.experimental.paged_update_cache(cache[0], kh, update_idxs_tensor=pos_t)
        ttnn.experimental.paged_update_cache(cache[1], vh, update_idxs_tensor=pos_t)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            qh, cache[0], cache[1], cur_pos_tensor=pos_t, scale=SCALE,
            compute_kernel_config=COMPUTE_CONFIG)
        # o -> DRAM and not _L1 on purpose: L1 here measures 0.999x, see _L1.
        a = ttnn.reshape(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG), [1, 1, Q_WIDTH])
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG,
                                    memory_config=_L1), memory_config=_L1)
        return self._mlp(x, self._norm_dec(x, w["fn"]), w, _L1)

    @torch.no_grad()
    def prefill(self, embeds, apply_final_norm=True, last_only=False):
        """embeds torch [1,S,3072] -> hidden torch [1,S,3072], or [1,1,3072] if `last_only`.

        No cache yet (increment 4). `last_only` is what the pipeline actually wants -- Block 2
        only ever sees the final position -- and it keeps the [1,S,3072] readback off the host.

        PADDING IS OURS TO CHOOSE, unlike the `tt_transformers` path which needs a 256-multiple as
        a hard constraint. Correctness needs only TILE; PREFILL_MULTIPLE is set above it to keep
        the kernel-shape count down. Zeros are safe: the causal mask keeps real positions from
        attending to the pad, and the padded QUERY rows are computed-then-discarded (each still
        attends to real keys, so no all-masked row and no NaN).
        """
        S = embeds.shape[1]
        Sp = (S + PREFILL_MULTIPLE - 1) // PREFILL_MULTIPLE * PREFILL_MULTIPLE
        if self.caches and Sp > self.max_seq_len:
            raise ValueError(f"prompt pads to {Sp} but the KV cache holds {self.max_seq_len}")
        if Sp != S:
            embeds = torch.cat([embeds, embeds.new_zeros(1, Sp - S, DIM)], dim=1)
        cosb, sinb = rope_tables(Sp)
        up = lambda t, d=None: ttnn.from_torch(t.contiguous(), dtype=d or self.dtype,
                                              layout=ttnn.TILE_LAYOUT, device=self.device)
        cos = up(cosb.reshape(1, 1, Sp, HEAD_DIM))
        sin = up(sinb.reshape(1, 1, Sp, HEAD_DIM))
        m = torch.full((Sp, Sp), float("-inf")).triu(1).reshape(1, 1, Sp, Sp)
        mask = up(m, ttnn.bfloat16)
        x = up(embeds.reshape(1, Sp, DIM))
        for i, w in enumerate(self.layers):
            x = self._layer(x, w, Sp, cos, sin, mask, self.caches[i] if self.caches else None)
        # Decode continues from the REAL length, not the padded one, or the first generated frame
        # would attend to the zero rows the pad wrote into the cache.
        self.pos = S
        if last_only:
            x = ttnn.slice(x, [0, S - 1, 0], [1, S, DIM])
        if apply_final_norm:
            x = ttnn.rms_norm(x, weight=self.norm, epsilon=NORM_EPS,
                              compute_kernel_config=COMPUTE_CONFIG)
        if last_only:
            return ttnn.to_torch(x).float().reshape(1, 1, DIM)
        return ttnn.to_torch(x).float().reshape(1, Sp, DIM)[:, :S]

    def prefill_last(self, embeds):
        """[1,P,3072] -> hidden of the LAST position [1,1,3072]. The pipeline's entry point; it is
        all Block 2 ever sees."""
        return self.prefill(embeds, last_only=True)

    @torch.no_grad()
    def step(self, embed):
        """embed torch [1,1,3072] (one frame) -> hidden torch [1,1,3072]. Advances self.pos.

        Mirrors `IncrementalBackbone.step`. No causal mask and no cache slice: every cached
        position is in the past, and sdpa_decode reads the whole cache bounded by `pos_t`.
        """
        if not self.caches:
            raise RuntimeError("step() needs a KV cache; construct with max_seq_len > 0")
        if self.pos >= self.max_seq_len:
            raise ValueError(f"KV cache full at {self.max_seq_len} positions")
        pos = self.pos
        cosb, sinb = rope_tables(1, offset=pos)
        up = lambda t, d=None: ttnn.from_torch(t.contiguous(), dtype=d or self.dtype,
                                              layout=ttnn.TILE_LAYOUT, device=self.device)
        # cos/sin sharded: rotary_embedding_hf's decode mode requires it. pos on device: both
        # paged_update_cache and sdpa_decode take the position as a tensor.
        cos = ttnn.to_memory_config(up(cosb.reshape(1, 1, 1, HEAD_DIM)), _ROPE_SHARD)
        sin = ttnn.to_memory_config(up(sinb.reshape(1, 1, 1, HEAD_DIM)), _ROPE_SHARD)
        pos_t = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=self.device)
        x = up(embed.reshape(1, 1, DIM))
        for i, w in enumerate(self.layers):
            x = self._layer_step(x, w, cos, sin, self.caches[i], pos_t)
        x = self._norm_dec(x, self.norm)
        self.pos = pos + 1
        return ttnn.to_torch(x).float().reshape(1, 1, DIM)


# --------------------------------------------------------------------------------
# GATES -- each increment of this port had to pass one before the next started.
# --------------------------------------------------------------------------------
