# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-TTS Block 1 (3.4B AR backbone) on TTNN, ours rather than `tt_transformers`.

Block 1, ours. It replaced a `models/tt_transformers` wrapper, which is gone -- see git history
if you need it (`tt/ttnn_voxtral_backbone.py`, removed once this beat it on every metric). The
rationale for owning this is in STATUS.md's Block 1 section.

Measured against the fp32 CPU reference on REAL prompts (never random ones -- STATUS.md trap #12),
with `tt_transformers` on the same metric for comparison:

                             ours (mixed W)   ours BFP8 W   tt_transformers
    prefill, last position     0.999881        0.999881       0.999564
    decode step                0.99991         0.99986        0.981
    decode ms/step             38.7            33.6           48
    end-to-end natural WER     0.88%           87.39%         1.17%

ALL-BFP8 IS THE FAST ONE AND WE DO NOT SHIP IT. It triggers a card-wedging hang in multi-utterance
runs and drives fixture case 4 into a repetition loop; both vanish under the mixed default. See
WEIGHT_DTYPE and ttnn_voxtral_pipeline.CLEAR_PROGRAM_CACHE.

The decode gap against tt_transformers (0.99991 vs 0.981) is real and reproduces at every weight
dtype we tried, so it is not ours to explain -- but note it is NOT caused by sdpa_decode, which is
nearly free at these shapes.

Mirrors `voxtral_backbone_ref._layer` op-for-op. Structurally this is Block 2's `_block` plus
three things -- RoPE, a causal mask, and a KV cache -- so what Block 2 already proved carries over
unchanged: the row fold, k+v fused into one weight, `nlp_create_qkv_heads`, HiFi4 with
fp32_dest_acc_en, bf16 activations. And so does its central lesson, which decode needed twice: a
BATCHED matmul costs per batch element, so fold whatever you can into ROWS (see `_layer_step`).

DECODE IS BOUND BY WEIGHT BYTES, so the weight dtype is the speed. The six linears alone measure
31.1 ms at bf16 for 6.05 GB, i.e. 194 GB/s -- the same ceiling a plain interleaved matmul reaches
after the Block 2 work, so there is no layout trick left in them. Everything else in the step is
overhead to be removed, and BFP8 halves the floor itself.

WHERE THE TIME GOES, per decode frame, steady state on one N150 (measure with
scratch probe_perf.py; the numbers below are case 2, 448 frames):

    six linears, weight streaming     ~23.6 ms   AT THE CEILING -- 194 GB/s, and a plain
                                                 interleaved matmul cannot do better here.
                                                 Tuned matmul program configs are SLOWER
                                                 (169 vs 193 GB/s on wq). Only fewer BYTES help.
    rms_norm x2                         6.0 ms   fp32 accumulation is load-bearing; the cheap
                                                 variant is 2.4x faster and drops model PCC to
                                                 0.992. Closed.
    everything else                    ~5.3 ms   qkv heads, rope, cache write, sdpa_decode,
                                                 reshapes, residual adds
    ------------------------------------------
    Block 1 total                      34.9 ms   (Block 2 is 42.5, so Block 2 is the larger half)

So the only remaining lever on Block 1 is WEIGHT BYTES, and that is capped by the hang: BFP8 on
FF1_FF3 is safe, adding FF2 reintroduces it (see WEIGHT_DTYPE). Everything else is small change.

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

import os

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
KV_WIDTH = N_KV_HEADS * HEAD_DIM      # 1024
TILE = 32
# Prefill pads its sequence to this. TILE is all correctness needs -- unlike the `tt_transformers`
# path, nothing here shards the sequence across the core grid, which is what forced ITS 256.
# 128 is a SHAPE-CHURN choice, not a hardware one: every distinct padded
# length is a distinct set of kernels, and a first-time compile costs ~6 s against ~1.3 s for a
# warm prefill. At 32 the 15 fixture prompts want 11 distinct shapes; at 128 they want 3, and the
# extra attention work on the padding is a fraction of one prefill that happens once per utterance.
PREFILL_MULTIPLE = 128
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
DTYPE = ttnn.bfloat16

# Decode runs in ttnn's DECODE-NATIVE head layout, [1, batch, heads, head_dim], and these are the
# memory configs those ops demand. At batch 1 the layout lines everything up for free:
# nlp_create_qkv_heads_decode emits q exactly as sdpa_decode wants it and k/v exactly as
# paged_update_cache wants them, already sharded -- so there is no permute, no hand-built shard,
# no cache slice and no attention mask (sdpa_decode bounds the cache with cur_pos instead).
# Worth 6.6 ms/frame over a hand-rolled interior, at the same decode PCC.
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
# BFP8 on FF1_FF3 only, bf16 everywhere else: exactly tt_transformers' precision_override.
# ALL-BFP8 is ~5 ms/frame faster and TRIGGERS A CARD-WEDGING HANG in multi-utterance runs, and
# adding FF2 alone is enough to bring it back (see ttnn_voxtral_pipeline for the full diagnosis).
#
#     weights                decode PCC   ms/frame   natural WER   hang?
#     bf16 (all)               0.99991      45.8         -         no
#     mixed (this)             0.99991      38.7       0.88%       no
#     BFP8 (all)               0.99986      33.6      87.39%       HANGS
#
# Decode is bandwidth-limited on weight bytes -- the six linears alone reach 194 GB/s, the same
# ceiling a plain interleaved matmul hits -- so the weight dtype IS the speed. FF1_FF3 is where
# BFP8 buys most: those two 3072x9216 matrices are about half the layer's bytes.
WEIGHT_DTYPE = ttnn.bfloat16          # everything except...
FF_WEIGHT_DTYPE = ttnn.bfloat8_b      # ...FF1 and FF3


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

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, dtype=DTYPE, weight_dtype=None,
                 n_layers=N_LAYERS, state=None, max_seq_len=2048):
        """`state` takes an already-loaded `load_backbone_state` dict. Pass it when the caller
        also needs the fp32 reference weights: that dict is ~13 GB, and loading it twice is the
        difference between comfortable and swapping.

        `max_seq_len` sizes the KV cache: 2 x n_layers x 8 x max_seq x 128 bf16, i.e. 218 MB at
        2048. Pass 0 to skip it (prefill-only harnesses; `step` then raises).
        """
        self.device = device
        self.dtype = dtype
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pos = 0
        self._dec_trace = None      # (tid, x_in, cos_in, sin_in, mask_in, pos_t, out, L, host)
        self._warm = None           # persistent trace inputs, allocated by warmup_decode
        wd = weight_dtype or WEIGHT_DTYPE or dtype
        w = state if state is not None else load_backbone_state(ckpt_path)

        up = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT,
                                         device=device)
        ffd = FF_WEIGHT_DTYPE or wd                         # FF1_FF3 may differ; see WEIGHT_DTYPE
        vec = lambda t: up(t.reshape(1, 1, -1), dtype)      # norm gammas: no bandwidth, keep bf16
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
                "wqkv": lin(torch.cat([wq, wk, w[p + "attention.wv"]], dim=0)),
                "wo": lin(w[p + "attention.wo"]),
                "w1": lin(w[p + "feed_forward.w1"], ffd),
                "w2": lin(w[p + "feed_forward.w2"]),
                "w3": lin(w[p + "feed_forward.w3"], ffd),
            })
        self._assert_shapes()
        # Allocated once and written in place, so a generation never reallocates. Zero-init is not
        # relied on for correctness -- `step` masks everything above self.pos.
        z = torch.zeros(1, N_KV_HEADS, max_seq_len, HEAD_DIM)
        self.caches = [(up(z, dtype), up(z, dtype)) for _ in range(n_layers)] if max_seq_len else []

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
        """
        return ttnn.rms_norm(x, weight=gamma, epsilon=NORM_EPS,
                             compute_kernel_config=COMPUTE_CONFIG)

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

    def _mlp(self, x, w):
        """Residual + pre-norm SwiGLU. Shared by prefill and decode; only attention differs.

        `activation="silu"` on the FF1 matmul is bit-identical to a separate `ttnn.silu` and one
        dispatch cheaper.
        """
        h = self._norm(x, w["fn"])
        g = ttnn.linear(h, w["w1"], activation="silu", compute_kernel_config=COMPUTE_CONFIG)
        u = ttnn.multiply(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG))
        return ttnn.add(x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG))

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
        return self._mlp(ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG)), w)

    # ----------------------------------------------------------------------------
    # DECODE PATH -- one frame at a time; THIS is the hot loop
    # Two interiors: _layer_step_native (default, ttnn decode-native ops) and _layer_step (hand-rolled reference).
    # ----------------------------------------------------------------------------
    def _layer_step(self, x, w, cos, sin, cache, pos_t):
        """One decode position. x [1,1,3072] -> same, against `cache` written up to `pos_t`.

        Head layout is [1, batch, heads, head_dim] here, not prefill's [1, heads, seq, head_dim] --
        see _QKV_SHARD for why that makes the whole interior glue-free. `pos_t` is a DEVICE tensor
        because paged_update_cache and sdpa_decode both take the position that way.
        """
        qkv = ttnn.linear(self._norm(x, w["an"]), w["wqkv"], compute_kernel_config=COMPUTE_CONFIG)
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
        a = ttnn.reshape(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG), [1, 1, Q_WIDTH])
        return self._mlp(ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG)), w)

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
        x = ttnn.rms_norm(x, weight=self.norm, epsilon=NORM_EPS,
                          compute_kernel_config=COMPUTE_CONFIG)
        self.pos = pos + 1
        return ttnn.to_torch(x).float().reshape(1, 1, DIM)


# --------------------------------------------------------------------------------
# GATES -- each increment of this port had to pass one before the next started.
# --------------------------------------------------------------------------------
def fixture_embeds(case_idx, w):
    """Fixture case -> real prompt embeds [1,P,3072], exactly as the pipeline builds them.

    REAL PROMPTS ARE NOT OPTIONAL for an accuracy number (STATUS.md trap #12): random embeddings
    are off-manifold and reported PCC 0.892 where these give 0.9994 on the same weights.
    """
    import json
    import os

    from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    case = json.load(open(os.path.join(here, "tests", "prompt_fixture.json")))["cases"][case_idx]
    ids = torch.tensor(case["ids"], dtype=torch.long)
    return pref.build_inputs_embeds(ids, pref.load_voice(case["voice"]), w), case


def gate_wiring(dev, ref):
    """Increment 2: ONE layer against the reference. A RoPE convention error shows up here."""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
        causal_bias, pcc, rope_cis)

    S = 128
    gen = TtVoxtralGPT(dev, n_layers=1)
    w = ref.load_backbone_state()
    torch.manual_seed(0)
    x = torch.randn(1, S, DIM) * 0.02
    exp = ref._layer(x, w, "layers.0.", rope_cis(S, HEAD_DIM, ROPE_THETA),
                     causal_bias(S, torch.float32))
    got = gen.prefill(x, apply_final_norm=False)
    print(f"  [1 layer prefill] PCC {pcc(got, exp):.8f}  "
          f"maxabs {(got - exp).abs().max():.3e}")
    print("  NOTE: random inputs are a pessimistic proxy (trap #12) -- this gate is for WIRING")
    print("  and the RoPE convention. Judge accuracy on real prompts at 26 layers.")


def gate_prefill26(dev, ref, cases, n_layers=N_LAYERS):
    """Increment 3: the full stack, prefill, on REAL prompts vs `reference_forward`.

    Reports the LAST position separately because that is the only one Block 2 consumes; the
    all-positions number is there to catch a bug that only touches part of the sequence.
    """
    import time

    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    print(f"  loading fp32 reference weights (~13 GB) -- shared with the device upload")
    w = ref.load_backbone_state()
    gen = TtVoxtralGPT(dev, n_layers=n_layers, state=w)
    print(f"  {n_layers} layers on device\n")
    print(f"  {'case':>4} {'voice':>16} {'P':>5} {'PCC all':>12} {'PCC last':>12} "
          f"{'worst last':>11} {'worst pos (which)':>20} {'device':>9}")
    for ci in cases:
        embeds, case = fixture_embeds(ci, w)
        P = embeds.shape[1]
        exp = ref.reference_forward(embeds, w, n_layers=n_layers)
        t0 = time.perf_counter()
        got = gen.prefill(embeds)
        dt = time.perf_counter() - t0
        el, xl = got[:, -1:], exp[:, -1:]
        worst = (el - xl).abs().max().item() / xl.abs().max().item() * 100
        # Per position, because a pooled PCC over the whole prompt is dominated by whichever
        # positions have the largest magnitude and hides which ones are actually weak.
        per = [pcc(got[:, i], exp[:, i]) for i in range(P)]
        wi = min(range(P), key=lambda i: per[i])
        print(f"  {ci:>4} {case['voice']:>16} {P:>5} {pcc(got, exp):>12.6f} "
              f"{pcc(el, xl):>12.6f} {worst:>10.2f}% {per[wi]:>13.6f} (@{wi:>4}) {dt:>8.2f}s")
    print("\n  reference for comparison, same metric on the LAST position (STATUS.md, Block 1):")
    print("    tt_transformers, FF1_FF3 BFP8: 0.999564 at P=200, 0.999579 at P=312")


def gate_decode(dev, ref, cases, n_steps=8, n_layers=N_LAYERS):
    """Increment 4: on-device KV cache + decode steps vs `IncrementalBackbone.step()`.

    TEACHER-FORCED on REAL frames (`tests/real_frames_fixture.pt`, genuine Block 1+2 output): both
    sides advance on the SAME embedding every step, so each step is an independent measurement.
    Feeding each its own codes instead compares two diverging trajectories and tells you nothing
    (the same trap `ttnn_voxtral_pipeline.compare_codes` documents).
    """
    import os
    import time

    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames = torch.load(os.path.join(here, "tests", "real_frames_fixture.pt")).long()
    w = ref.load_backbone_state()
    gen = TtVoxtralGPT(dev, n_layers=n_layers, state=w, max_seq_len=1024)
    for ci in cases:
        embeds, case = fixture_embeds(ci, w)
        P = embeds.shape[1]
        print(f"\n  case {ci} ({case['voice']}, P={P}), {n_steps} real frames teacher-forced")
        print(f"  {'step':>6} {'pos':>5} {'PCC':>11} {'worst':>8} {'ms':>8}")
        inc = ref.IncrementalBackbone(w, n_layers=n_layers)
        h_ref = inc.prefill(embeds)
        gen.reset()
        h_dev = gen.prefill(embeds, last_only=True)
        assert gen.pos == inc.pos == P, f"position mismatch after prefill: {gen.pos} vs {inc.pos}"
        worst = (h_dev - h_ref).abs().max().item() / h_ref.abs().max().item() * 100
        print(f"  {'prefill':>6} {P:>5} {pcc(h_dev, h_ref):>11.6f} {worst:>7.2f}%")
        for t in range(min(n_steps, frames.shape[0])):
            emb = ref.embed_frame(w, frames[t])
            h_ref = inc.step(emb)
            t0 = time.perf_counter()
            h_dev = gen.step(emb)
            dt = (time.perf_counter() - t0) * 1e3
            worst = (h_dev - h_ref).abs().max().item() / h_ref.abs().max().item() * 100
            print(f"  {t:>6} {gen.pos - 1:>5} {pcc(h_dev, h_ref):>11.6f} {worst:>7.2f}% {dt:>7.1f}")
    print("\n  reference for comparison (STATUS.md, Block 1): tt_transformers decode PCC 0.981,")
    print("  48 ms/step. The 0.981 is unexplained there; this path should not reproduce it.")


def main():
    import argparse

    from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as ref

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", default="wiring", choices=("wiring", "prefill26", "decode"),
                    help="wiring = increment 2 (fast, one layer, checks the RoPE convention); "
                         "prefill26 = increment 3 (26 layers on real prompts, ~13 GB host RAM); "
                         "decode = increment 4 (KV cache + steps vs IncrementalBackbone)")
    ap.add_argument("--cases", default="0,2", help="prompt_fixture.json indices")
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--steps", type=int, default=8, help="decode steps for --gate decode")
    args = ap.parse_args()

    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

    dev = open_device()
    try:
        cases = [int(c) for c in args.cases.split(",")]
        if args.gate == "wiring":
            gate_wiring(dev, ref)
        elif args.gate == "prefill26":
            gate_prefill26(dev, ref, cases, args.layers)
        else:
            gate_decode(dev, ref, cases, args.steps, args.layers)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
