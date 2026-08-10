# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-TTS Block 1: the 3.4B autoregressive backbone, on TTNN.

See NOTES.md [gpt-01].
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

# NOTES.md [gpt-02] -- Prefill pads its sequence to this...
PREFILL_MULTIPLE = 128
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
DTYPE = ttnn.bfloat16

# NOTES.md [gpt-03] -- DECODE'S INTERMEDIATES LIVE IN L1, not DRAM -- the same...
_L1 = ttnn.L1_MEMORY_CONFIG

# NOTES.md [gpt-04] -- the decode RMSNorm is NOT width-sharded on Blackhole. Sharding it wins
# on Wormhole and LOSES here, by the same margin and for the same reason -- the reshard is the
# tax, and the p150's interleaved kernel made the reduction cheap enough that the tax is pure
# loss. That is why _NORM_SHARD / _NORM_PRG / _norm_dec are gone from this branch.

# NOTES.md [gpt-05] -- Decode runs in ttnn's DECODE-NATIVE head layout, [1...
_QKV_WIDTH = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM      # 6144, one fused projection
# One number used twice. On Blackhole FEWER cores is faster -- 1 beats 8 by 0.46 ms/step, and
# the grid provably never reaches the consumers (they always emit 1 core, (32,128)). [gpt-05].
_QKV_GRID_X = 1
_QKV_SHARD = ttnn.create_sharded_memory_config(
    (TILE, _QKV_WIDTH // _QKV_GRID_X), core_grid=ttnn.CoreGrid(y=1, x=_QKV_GRID_X),
    strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True)
# rotary_embedding_hf's decode mode requires cos/sin sharded as well as the input
# ("Cos must be sharded in decode mode"), one tile row on one core at batch 1.
_ROPE_SHARD = ttnn.create_sharded_memory_config(
    (TILE, HEAD_DIM), core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
# NOTES.md [gpt-19] -- _V_SHARD is GONE. It existed only to let paged_fused_update_cache accept
# K and V, and on Blackhole that fused write is 0.687 ms/step SLOWER than two plain writes. Its
# silent failure mode (RoPE on a core whose cos/sin table lives elsewhere returns 3.4e38 from
# uninitialised L1) goes with it. STATUS.md 6.44.

# NOTES.md [gpt-20] -- wo has NO program config on Blackhole. 6.25 hand-tuned one for the N150
# (+0.196 ms/frame); here it is worth nothing measurable and is deleted. Removing it is bit-exact,
# so the only question was speed, and no instrument could find any. STATUS.md 6.43.

# NOTES.md [gpt-21] -- k=512 on an 8x2 grid, 1.63x. THE FASTER ONES ARE NOT SAFE -- position sweep...
_SDPA_PRG = ttnn.SDPAProgramConfig(
    q_chunk_size=TILE, k_chunk_size=512,
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 2))

# NOTES.md [gpt-26] -- DECODE matmul program configs. Two separate p150 findings, one fix:
#   * activation="silu" IS NOT FUSED. It costs 98.8 us against a plain matmul's 85.5 -- the same
#     +14.9 as writing ttnn.silu() as its own op, which is what it evidently does. Passing
#     UnaryWithParam or UnaryOpType instead changes nothing (100.6 / 100.2). ONLY a program
#     config's fused_activation actually folds it in, at 88.1. That one op is worth 2.42 ms/frame
#     over the 47 w1 calls in the two blocks, and is slightly MORE accurate besides (PCC
#     0.9999984 vs 0.9999970) because the value stays in the dest registers.
#   * the ttnn heuristic collapses on deep reductions -- 144-147 GB/s at Kt=128/288 against 352 at
#     Kt=96, under half of this chip's measured 367 GB/s. A tuned in0_block_w recovers ~350.
# ONE 12x6 grid serves all four shapes: it ties a set of per-shape isolated winners in the block
# (36.99 vs 37.04 ms, against a 0.070 noise floor), so no shape earns its own grid. Whole-block
# A/B, three runs: -4.66 / -4.38 / -4.24 ms/frame. STATUS.md 6.52.
#
# DECODE ONLY. per_core_M=1 and fuse_batch=True assume ONE tile of rows -- true for Block 1's 1
# row and Block 2's 3-or-6, false for prefill. _mlp is shared with prefill, so the configs are
# passed IN rather than read from module scope; prefill passes nothing and keeps the heuristic.
_MM_GRID = (12, 6)                    # 72 of the 130 cores; 13x10 measured 0.31 ms WORSE


def _mm1d(in0_block_w, per_core_n, activation=None):
    """1D multicast: split N across the grid, broadcast in0. The batch-1 decode shape."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=_MM_GRID, in0_block_w=in0_block_w, out_subblock_h=1,
        # Largest legal width. Two hard rules, both TT_FATAL in ttnn: osh*osw <= 4 (8
        # normally -- fp32_dest_acc_en halves the dest register file), and per_core_N must
        # be divisible by it. 3 belongs in this list: ttnn's own SUBBLOCK_HW_CHOICES has
        # {3,1}, and leaving it out dropped wqkv (per_core_N=3) to 1. STATUS.md 6.61.
        out_subblock_w=next(s for s in (4, 3, 2, 1) if per_core_n % s == 0),
        per_core_M=1, per_core_N=per_core_n, fuse_batch=True,
        fused_activation=activation, mcast_in0=True)


#                       in0_block_w   per_core_N = ceil(N_tiles / 72)
_PRG_QKV = _mm1d(2, 3)              # K=3072  N=6144   Nt=192
_PRG_WO = _mm1d(4, 2)               # K=4096  N=3072   Nt= 96
_PRG_W1 = _mm1d(2, 4, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU))   # K=3072 N=9216 Nt=288
_PRG_W3 = _mm1d(2, 4)               # same shape as w1, no activation
_PRG_W2 = _mm1d(4, 2)               # K=9216  N=3072   Nt= 96 -- the deepest reduction in the model
DECODE_PRG = {"wqkv": _PRG_QKV, "wo": _PRG_WO, "w1": _PRG_W1, "w3": _PRG_W3, "w2": _PRG_W2}


def _pc(prg, key):
    """program_config kwarg for `key`, or nothing at all when prg is empty (the prefill path)."""
    return {"program_config": prg[key]} if prg else {}

# NOTES.md [gpt-06] -- WEIGHT PRECISION -- load-bearing for CORRECTNESS, not...
WEIGHT_DTYPE = ttnn.bfloat16          # w2 -- bf16 for ACCURACY (6.16), not for the hang (6.13)
FF_WEIGHT_DTYPE = ttnn.bfloat8_b      # FF1 and FF3
ATTN_WEIGHT_DTYPE = ttnn.bfloat8_b    # wqkv and wo


def interleaved_to_halfsplit(t, n_heads):
    """Mistral-native (interleaved-pair) q/k weight -> half-split layout, so `rotate_half` applies.

    See NOTES.md [gpt-07].
    """
    d1, d2 = t.shape
    return t.view(n_heads, d1 // n_heads // 2, 2, d2).transpose(1, 2).reshape(d1, d2)


def rope_tables(seq_len, offset=0, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """-> (cos, sin) torch [seq_len, head_dim], each half duplicated for the half-split form.

    See NOTES.md [gpt-08].
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

        See NOTES.md [gpt-09].
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
                # NOTES.md [gpt-10] -- q, k and v fused into ONE weight: one matmul and one...
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

        See NOTES.md [gpt-11].
        """
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=False,
                                                     compute_kernel_config=COMPUTE_CONFIG)

    def _norm(self, x, gamma):
        """RMSNorm.

        See NOTES.md [gpt-12].
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

        See NOTES.md [gpt-13].
        """
        rep = N_HEADS // N_KV_HEADS
        kr, vr = ttnn.repeat_interleave(kh, rep, dim=1), ttnn.repeat_interleave(vh, rep, dim=1)
        s = ttnn.matmul(qh, ttnn.transpose(kr, -2, -1), compute_kernel_config=COMPUTE_CONFIG)
        s = ttnn.add(ttnn.multiply(s, SCALE), mask)
        a = ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        a = ttnn.matmul(a, vr, compute_kernel_config=COMPUTE_CONFIG)
        # bit-identical to permute(0,2,1,3) + reshape, in one dispatch
        return ttnn.reshape(ttnn.experimental.nlp_concat_heads(a), [1, S, Q_WIDTH])

    def _mlp(self, x, h, w, mc, prg=None):
        """Residual + SwiGLU over an ALREADY-NORMED `h`. Shared by prefill and decode.

        `prg` is DECODE_PRG on the decode path and None on the prefill one -- see NOTES.md
        [gpt-26]; those configs assume a single tile of rows, which prefill violates.

        See NOTES.md [gpt-14].
        """
        prg = prg or {}
        # NOTES.md [gpt-22] -- w1 and w3 stay SEPARATE -- fusing them is 4x SLOWER, and why...
        # NOTES.md [gpt-26] -- silu rides in the program config, NOT activation="silu", which is
        # not fused. With no config (prefill) fall back to the kwarg.
        g = (ttnn.linear(h, w["w1"], program_config=prg["w1"],
                         compute_kernel_config=COMPUTE_CONFIG, memory_config=mc) if prg else
             ttnn.linear(h, w["w1"], activation="silu", compute_kernel_config=COMPUTE_CONFIG,
                         memory_config=mc))
        # NOTES.md [gpt-25] -- IN PLACE. g and x are both dead immediately after, and on Blackhole
        # the allocation is ~12 us of a ~65 us op. Worth 0.929 ms/step with the two adds, and
        # bit-identical. 6.37 measured this at +0.001 ms on the N150. STATUS.md 6.47.
        u = ttnn.multiply_(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG,
                                          memory_config=mc, **_pc(prg, "w3")))
        return ttnn.add_(x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG,
                                        memory_config=mc, **_pc(prg, "w2")))

    def _layer(self, x, w, S, cos, sin, mask, cache=None):
        """x [1,S,3072] -> same. Pre-norm GQA with RoPE + causal mask, then SwiGLU.

        See NOTES.md [gpt-15].
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

        See NOTES.md [gpt-16].
        """
        qkv = ttnn.linear(self._norm(x, w["an"]), w["wqkv"], program_config=DECODE_PRG["wqkv"],
                          compute_kernel_config=COMPUTE_CONFIG)
        qkv = ttnn.to_memory_config(ttnn.reshape(qkv, [1, 1, 1, _QKV_WIDTH]), _QKV_SHARD)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS)
        qh = ttnn.experimental.rotary_embedding_hf(qh, cos, sin, is_decode_mode=True,
                                                   compute_kernel_config=COMPUTE_CONFIG)
        # NOTES.md [gpt-23] -- two calls, not the fused q+k rope: convention mismatch, 0.236 ms/frame...
        kh = ttnn.experimental.rotary_embedding_hf(kh, cos, sin, is_decode_mode=True,
                                                   compute_kernel_config=COMPUTE_CONFIG)
        # NOTES.md [gpt-24] -- TWO writes, not the fused one: on Blackhole the fused write LOSES
        # 0.687 ms/step and 6.20/6.22's whole V-move chain goes with it. STATUS.md 6.44.
        ttnn.experimental.paged_update_cache(cache[0], kh, update_idxs_tensor=pos_t)
        ttnn.experimental.paged_update_cache(cache[1], vh, update_idxs_tensor=pos_t)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            qh, cache[0], cache[1], cur_pos_tensor=pos_t, scale=SCALE,
            compute_kernel_config=COMPUTE_CONFIG, program_config=_SDPA_PRG)
        # NOTES.md [gpt-03b] -- no memory_config move: L1 here measures 0.999x, see [gpt-03]...
        a = ttnn.reshape(o, [1, 1, Q_WIDTH])
        # in place -- see NOTES.md [gpt-25]. Safe: `x` is the layer input and is dead the moment
        # this returns, and _norm below is evaluated BEFORE _mlp mutates anything.
        x = ttnn.add_(x, ttnn.linear(a, w["wo"], program_config=DECODE_PRG["wo"],
                                     compute_kernel_config=COMPUTE_CONFIG, memory_config=_L1))
        return self._mlp(x, self._norm(x, w["fn"]), w, _L1, DECODE_PRG)

    @torch.no_grad()
    def prefill(self, embeds, apply_final_norm=True, last_only=False):
        """embeds torch [1,S,3072] -> hidden torch [1,S,3072], or [1,1,3072] if `last_only`.

        See NOTES.md [gpt-17].
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

        See NOTES.md [gpt-18].
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
        x = self._norm(x, self.norm)
        self.pos = pos + 1
        return ttnn.to_torch(x).float().reshape(1, 1, DIM)


# --------------------------------------------------------------------------------
# GATES -- each increment of this port had to pass one before the next started.
# --------------------------------------------------------------------------------
