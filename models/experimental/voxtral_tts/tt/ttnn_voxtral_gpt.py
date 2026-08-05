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

# NOTES.md [gpt-04] -- RMSNORM, WIDTH-SHARDED, for the DECODE shape...
# Grid and blocking for the sharded norm. NOTES.md [gpt-04] has the sweep; the short version is
# that `block_w` is NOT a free knob (it is DIM // cores // TILE, so it only moves when the core
# count does), `subblock_w` IS free but inert (1/2/3/4 within 0.02 ms, and >=6 will not build), and
# and the core count has a MINIMUM AT 32, not a monotone trend: 2/4/8/16/24/32/48 cores measure
# 25.53/24.84/24.56/24.45/24.42/24.41/24.44 ms/step. The count must DIVIDE the 96 width-tiles of a
# 32x3072 tensor, since block_w is tiles-per-core and a tile is indivisible -- so 40, 56 and 64 do
# not build at all (96/64 = 1.5), while 48 is legal and simply loses.
_NORM_GRID_X, _NORM_GRID_Y, _NORM_SUBBLOCK_W = 8, 4, 1
_NORM_CORES = _NORM_GRID_X * _NORM_GRID_Y
_NORM_SHARD = ttnn.create_sharded_memory_config(
    shape=(1, 1, TILE, DIM), core_grid=ttnn.CoreGrid(y=_NORM_GRID_Y, x=_NORM_GRID_X),
    strategy=ttnn.ShardStrategy.WIDTH)
_NORM_PRG = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(_NORM_GRID_X, _NORM_GRID_Y),
    subblock_w=_NORM_SUBBLOCK_W, block_h=1,
    block_w=DIM // _NORM_CORES // TILE, inplace=False)

# NOTES.md [gpt-05] -- Decode runs in ttnn's DECODE-NATIVE head layout, [1...
_QKV_WIDTH = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM      # 6144, one fused projection
# The core count is inert for speed -- 6 to 48 all land inside a 0.020 ms spread, NOTES.md [gpt-05].
# One number, used twice: the literal 8 used to appear in both the shard width and the grid, and
# changing one without the other yields a silently wrong shard rather than an error.
_QKV_GRID_X = 8
_QKV_SHARD = ttnn.create_sharded_memory_config(
    (TILE, _QKV_WIDTH // _QKV_GRID_X), core_grid=ttnn.CoreGrid(y=1, x=_QKV_GRID_X),
    strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True)
# rotary_embedding_hf's decode mode requires cos/sin sharded as well as the input
# ("Cos must be sharded in decode mode"), one tile row on one core at batch 1.
_ROPE_SHARD = ttnn.create_sharded_memory_config(
    (TILE, HEAD_DIM), core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
# V's parking space, core (1,0). paged_fused_update_cache writes both caches in one kernel but
# refuses K and V on the same core, and nlp_create_qkv_heads_decode puts q, k and v all on (0,0), so
# ONE of them has to move. MOVE V, NOT K -- the choice matters more than it looks:
#   * moving K is what overlap_qk_coregrid=False does. It is 0.047 ms/frame faster, and it costs two
#     coupling hazards: it asserts a whole head per core (pinning _QKV_SHARD to 48 cores, load-bearing
#     and non-obvious), and K then goes through RoPE on a core whose cos/sin table is elsewhere --
#     which does NOT raise, it returns 3.4e38 from uninitialised L1.
#   * V never touches RoPE and imposes nothing on the shard width. One reshard per layer, 26 a frame,
#     ~2.1 us each because it is an 8 KB hop between adjacent cores.
# 0.405 vs 0.452 ms/frame -- 0.09% of a frame to delete a silent-garbage failure mode. Taken from
# lserbedzija/xtts-gpt-ttnn, which does the same thing in xtts_v2/tt/ttnn_xtts_gpt.py.
_V_SHARD = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}),
                   (TILE, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR))

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

    def _mlp(self, x, h, w, mc):
        """Residual + SwiGLU over an ALREADY-NORMED `h`. Shared by prefill and decode.

        See NOTES.md [gpt-14].
        """
        g = ttnn.linear(h, w["w1"], activation="silu", compute_kernel_config=COMPUTE_CONFIG,
                        memory_config=mc)
        u = ttnn.multiply(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG,
                                         memory_config=mc), memory_config=mc)
        return ttnn.add(x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG,
                                       memory_config=mc), memory_config=mc)

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
        qkv = ttnn.linear(self._norm_dec(x, w["an"]), w["wqkv"],
                          compute_kernel_config=COMPUTE_CONFIG)
        qkv = ttnn.to_memory_config(ttnn.reshape(qkv, [1, 1, 1, _QKV_WIDTH]), _QKV_SHARD)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS)
        qh = ttnn.experimental.rotary_embedding_hf(qh, cos, sin, is_decode_mode=True,
                                                   compute_kernel_config=COMPUTE_CONFIG)
        # Two calls, not ttnn's fused q+k rope: that one implements the INTERLEAVED convention via a
        # trans_mat, and our wq/wk are permuted to HALF-SPLIT at load. Measured 0.236 ms/frame for
        # reverting that permute, disjoint q/k cores and losing bit-exactness -- STATUS.md 6.23.
        kh = ttnn.experimental.rotary_embedding_hf(kh, cos, sin, is_decode_mode=True,
                                                   compute_kernel_config=COMPUTE_CONFIG)
        # ONE fused write, not two: 26 launches a frame instead of 52, worth 0.405 ms and
        # bit-identical. V is moved to core (1,0) first because the op refuses an overlap -- see
        # _V_SHARD for why V and not K.
        ttnn.experimental.paged_fused_update_cache(
            cache[0], kh, cache[1], ttnn.to_memory_config(vh, _V_SHARD),
            update_idxs_tensor=pos_t)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            qh, cache[0], cache[1], cur_pos_tensor=pos_t, scale=SCALE,
            compute_kernel_config=COMPUTE_CONFIG)
        # o -> DRAM and not _L1 on purpose, and NOT because DRAM is better: sdpa_decode already
        # emits o as interleaved DRAM, so L1 would mean MOVING it (+6.3 us) to save ~1.1 us on a
        # matmul whose activation is 0.06% of its read traffic. 0.999x. See NOTES.md [gpt-03].
        a = ttnn.reshape(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG), [1, 1, Q_WIDTH])
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG,
                                    memory_config=_L1), memory_config=_L1)
        return self._mlp(x, self._norm_dec(x, w["fn"]), w, _L1)

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
        x = self._norm_dec(x, self.norm)
        self.pos = pos + 1
        return ttnn.to_torch(x).float().reshape(1, 1, DIM)


# --------------------------------------------------------------------------------
# GATES -- each increment of this port had to pass one before the next started.
# --------------------------------------------------------------------------------
