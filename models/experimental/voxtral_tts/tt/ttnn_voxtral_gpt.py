# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-TTS backbone: the 3.4B autoregressive transformer, on TTNN.

prefill() runs the whole prompt and fills the KV cache; step() decodes one frame against it.
Mirrors reference/voxtral_backbone_ref._layer op-for-op. Tests: tests/pcc/test_backbone_*_pcc.py.
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
Q_WIDTH = N_HEADS * HEAD_DIM  # 4096, deliberately != DIM
TILE = 32

# Prefill pads its sequence to a multiple of this: a tile multiple keeps the explicit causal mask
# tile-aligned, and a coarser step keeps the number of compiled prefill shapes down.
PREFILL_MULTIPLE = 128
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
DTYPE = ttnn.bfloat16

# Decode keeps its intermediates in L1: they are small and consumed within an op or two.
_L1 = ttnn.L1_MEMORY_CONFIG

# Memory configs for ttnn's decode-native head layout, [1, batch, heads, head_dim].
_QKV_WIDTH = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM  # 6144, one fused projection
# Sets both the shard width and the grid, so the two cannot disagree. One core: the decode head op
# emits a single-core output whatever this is, so a wider input shard only adds cost.
_QKV_GRID_X = 1
_QKV_SHARD = ttnn.create_sharded_memory_config(
    (TILE, _QKV_WIDTH // _QKV_GRID_X),
    core_grid=ttnn.CoreGrid(y=1, x=_QKV_GRID_X),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
# rotary_embedding_hf's decode mode requires cos/sin sharded as well as the input
# ("Cos must be sharded in decode mode"), one tile row on one core at batch 1.
_ROPE_SHARD = ttnn.create_sharded_memory_config(
    (TILE, HEAD_DIM),
    core_grid=ttnn.CoreGrid(y=1, x=1),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
# sdpa_decode program config. Faster chunk/grid choices exist but are not exact at every cache
# position; change it only against a sweep over positions.
_SDPA_PRG = ttnn.SDPAProgramConfig(
    q_chunk_size=TILE, k_chunk_size=512, compute_with_storage_grid_size=ttnn.CoreCoord(8, 2)
)

# Decode matmul program configs. DECODE ONLY: per_core_M=1 and fuse_batch=True assume one tile of
# rows, so _mlp takes them as an argument. SiLU fuses via fused_activation, not activation="silu".
_MM_CORES = 72  # every per_core_N below splits N over this many cores, whatever the grid


def decode_grid(device_grid):
    """-> the decode matmuls' core grid: the widest, up to 12 columns, with _MM_CORES cores that fits
    `device_grid`. The per-core split stays fixed, so every grid computes identical results."""
    for x in range(min(device_grid.x, 12), 0, -1):
        y = -(-_MM_CORES // x)
        if y <= device_grid.y:
            return (x, y)
    raise RuntimeError(f"device grid {device_grid.x}x{device_grid.y} has no rectangle of {_MM_CORES} cores")


def _mm1d(grid, in0_block_w, per_core_n, activation=None):
    """1D multicast: split N across the grid, broadcast in0. The batch-1 decode shape."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        # largest legal width: osh*osw <= 4 and per_core_N % osw == 0, both TT_FATAL.
        out_subblock_w=next(s for s in (4, 3, 2, 1) if per_core_n % s == 0),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=activation,
        mcast_in0=True,
    )


# (in0_block_w, per_core_N = ceil(N_tiles / _MM_CORES), fused activation) per decode matmul
_DECODE_SPLIT = {
    "wqkv": (2, 3, None),  # K=3072  N=6144   Nt=192
    "wo": (4, 2, None),  # K=4096  N=3072   Nt= 96
    "w1": (2, 4, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)),  # K=3072 N=9216 Nt=288
    "w3": (2, 4, None),  # same shape as w1, no activation
    "w2": (4, 2, None),  # K=9216  N=3072   Nt= 96 -- the deepest reduction in the model
}


def decode_program_configs(grid):
    """-> {matmul name: program config} on `grid` (see decode_grid)."""
    return {name: _mm1d(grid, *split) for name, split in _DECODE_SPLIT.items()}


def _pc(prg, key):
    """program_config kwarg for `key`, or nothing at all when prg is empty (the prefill path)."""
    return {"program_config": prg[key]} if prg else {}


# Width-sharded decode RMSNorm: the interleaved one splits its work by rows, and decode has one.
# DECODE ONLY: the shard spec fixes the height at one tile.
_NORM_GRID = (8, 4)  # 32 cores x block_w 3 == 96 tiles
_NORM_SHARD = ttnn.create_sharded_memory_config(
    (TILE, DIM // (_NORM_GRID[0] * _NORM_GRID[1])),
    core_grid=ttnn.CoreGrid(y=_NORM_GRID[1], x=_NORM_GRID[0]),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
_NORM_PRG = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=_NORM_GRID,
    subblock_w=1,
    block_h=1,
    block_w=DIM // TILE // (_NORM_GRID[0] * _NORM_GRID[1]),
    inplace=False,
)


def check_device_grid(device):
    """Raise, naming the constant, if any hardcoded core grid does not fit this device."""
    g = device.compute_with_storage_grid_size()
    need = {
        "_NORM_GRID": _NORM_GRID,
        "_SDPA_PRG": (_SDPA_PRG.compute_with_storage_grid_size.x, _SDPA_PRG.compute_with_storage_grid_size.y),
    }
    bad = {k: v for k, v in need.items() if v[0] > g.x or v[1] > g.y}
    if bad:
        raise RuntimeError(
            f"device compute grid is {g.x}x{g.y}; these do not fit: {bad}. "
            f"Check `tt-smi -s` ENABLED_TENSIX_COL (0x3fff = all 14 columns)."
        )


def sharded_norm(x, gamma, eps, mc):
    """Width-sharded RMSNorm for ONE tile of rows; falls back to interleaved for prefill."""
    if x.shape[-2] > TILE:
        return ttnn.rms_norm(x, weight=gamma, epsilon=eps, compute_kernel_config=COMPUTE_CONFIG)
    r = ttnn.rms_norm(
        ttnn.to_memory_config(x, _NORM_SHARD),
        weight=gamma,
        epsilon=eps,
        program_config=_NORM_PRG,
        memory_config=_NORM_SHARD,
        compute_kernel_config=COMPUTE_CONFIG,
    )
    return ttnn.to_memory_config(r, mc)


# Weight precision decides accuracy as well as speed. Decode is bound by weight bandwidth, so every
# matrix that tolerates it is BFP8; w2 does not.
WEIGHT_DTYPE = ttnn.bfloat16  # w2, kept bf16 for accuracy
FF_WEIGHT_DTYPE = ttnn.bfloat8_b  # FF1 and FF3
ATTN_WEIGHT_DTYPE = ttnn.bfloat8_b  # wqkv and wo


def interleaved_to_halfsplit(t, n_heads):
    """Mistral-native (interleaved-pair) q/k weight -> half-split layout, so `rotate_half` applies."""
    d1, d2 = t.shape
    return t.view(n_heads, d1 // n_heads // 2, 2, d2).transpose(1, 2).reshape(d1, d2)


def rope_tables(seq_len, offset=0, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """-> (cos, sin) torch [seq_len, head_dim], each half duplicated for the half-split form."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    ang = torch.outer(torch.arange(offset, offset + seq_len, dtype=torch.float64), inv)
    return (torch.cat([ang.cos(), ang.cos()], dim=-1).float(), torch.cat([ang.sin(), ang.sin()], dim=-1).float())


class TtVoxtralGPT:
    """The backbone on device. prefill(embeds) -> hidden; step(embed) -> hidden, sharing a KV
    cache."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, n_layers=N_LAYERS, state=None, max_seq_len=2048):
        """`state` takes an already-loaded `load_backbone_state` dict so the fp32 weights load once;
        `max_seq_len=0` skips the KV cache."""
        check_device_grid(device)
        self.device = device
        self.decode_prg = decode_program_configs(decode_grid(device.compute_with_storage_grid_size()))
        self.dtype = DTYPE
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pos = 0
        wd = WEIGHT_DTYPE
        attnd = ATTN_WEIGHT_DTYPE
        w = state if state is not None else load_backbone_state(ckpt_path)

        up = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT, device=device)
        ffd = FF_WEIGHT_DTYPE or wd  # FF1_FF3 may differ; see WEIGHT_DTYPE
        vec = lambda t: up(t.reshape(1, 1, -1), DTYPE)  # norm gammas: no bandwidth, keep bf16
        lin = lambda t, d=None: up(t.t(), d or wd)  # torch [out,in] -> ttnn wants [in,out]

        self.norm = vec(w["norm"])
        self.layers = []
        for i in range(n_layers):
            p = f"layers.{i}."
            wq = interleaved_to_halfsplit(w[p + "attention.wq"], N_HEADS)
            wk = interleaved_to_halfsplit(w[p + "attention.wk"], N_KV_HEADS)  # n_kv, not n_heads
            self.layers.append(
                {
                    "an": vec(w[p + "attention_norm"]),
                    "fn": vec(w[p + "ffn_norm"]),
                    # q, k and v fused into one weight: one matmul, in the layout the decode head op expects.
                    "wqkv": lin(torch.cat([wq, wk, w[p + "attention.wv"]], dim=0), attnd),
                    "wo": lin(w[p + "attention.wo"], attnd),
                    "w1": lin(w[p + "feed_forward.w1"], ffd),
                    "w2": lin(w[p + "feed_forward.w2"]),
                    "w3": lin(w[p + "feed_forward.w3"], ffd),
                }
            )
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
        exp = {
            "wqkv": (DIM, _QKV_WIDTH),
            "wo": (Q_WIDTH, DIM),
            "w1": (DIM, HIDDEN_DIM),
            "w3": (DIM, HIDDEN_DIM),
            "w2": (HIDDEN_DIM, DIM),
        }
        for i, L in enumerate(self.layers):
            for k, e in exp.items():
                got = tuple(L[k].shape)[-2:]
                assert got == e, f"layer {i} {k}: expected {e}, got {got}"

    # ----------------------------------------------------------------------------
    # SHARED PRIMITIVES -- used by both prefill and decode
    # ----------------------------------------------------------------------------
    def _rope(self, x, cos, sin):
        """Half-split RoPE on [1, heads, S, head_dim]: x*cos + rotate_half(x)*sin. Correct only because
        wq/wk were permuted to half-split at load (interleaved_to_halfsplit)."""
        return ttnn.experimental.rotary_embedding_hf(
            x, cos, sin, is_decode_mode=False, compute_kernel_config=COMPUTE_CONFIG
        )

    def _norm(self, x, gamma):
        """RMSNorm. The HiFi4 / fp32-accumulate compute config inside is load-bearing for accuracy."""
        return sharded_norm(x, gamma, NORM_EPS, _L1)

    # ----------------------------------------------------------------------------
    # PREFILL PATH -- whole prompt at once, fills the KV cache
    # Runs once per utterance, not once per frame.
    # ----------------------------------------------------------------------------
    def _qkv(self, x, w, S, cos, sin):
        """Pre-norm + fused QKV + RoPE. -> (q,k,v) as [1, heads, S, head_dim], v un-rotated."""
        h = self._norm(x, w["an"])
        qkv = ttnn.linear(h, w["wqkv"], compute_kernel_config=COMPUTE_CONFIG)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(qkv, [1, 1, S, _QKV_WIDTH]),
            num_heads=N_HEADS,
            num_kv_heads=N_KV_HEADS,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._rope(qh, cos, sin), self._rope(kh, cos, sin), vh  # v carries no RoPE

    def _attend(self, qh, kh, vh, S, mask):
        """PREFILL attention: [1,32,S,128] x [1,8,S,128] -> merged [1,S,4096], `mask` additive.

        Scale, mask and softmax stay three ops: the fused scale_mask_softmax reads only row 0 of the
        mask unless told it is causal, which silently breaks a triangular mask.
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
        """Residual + SwiGLU over an already-normed `h`; `prg` is decode_prg on decode, None on
        prefill."""
        prg = prg or {}
        # w1 and w3 stay separate matmuls: fused into one wide weight, the matmul loses most of its bandwidth.
        # silu rides in the program config, not activation="silu", which is not fused. With no
        # config (prefill) fall back to the kwarg.
        g = (
            ttnn.linear(h, w["w1"], program_config=prg["w1"], compute_kernel_config=COMPUTE_CONFIG, memory_config=mc)
            if prg
            else ttnn.linear(h, w["w1"], activation="silu", compute_kernel_config=COMPUTE_CONFIG, memory_config=mc)
        )
        # In place, saving an allocation: g and x are both dead immediately after.
        u = ttnn.multiply_(
            g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG, memory_config=mc, **_pc(prg, "w3"))
        )
        # Residual as the matmul bias on decode (one row) only; a bias broadcasts over rows, so
        # prefill keeps the add.
        if prg:
            return ttnn.linear(
                u,
                w["w2"],
                bias=ttnn.reshape(x, [1, DIM]),
                program_config=prg["w2"],
                compute_kernel_config=COMPUTE_CONFIG,
                memory_config=mc,
            )
        return ttnn.add_(
            x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG, memory_config=mc, **_pc(prg, "w2"))
        )

    def _layer(self, x, w, S, cos, sin, mask, cache=None):
        """x [1,S,3072] -> same. Pre-norm GQA with RoPE + causal mask, then SwiGLU. Fills `cache`
        with all padded rows; decode reads it only up to `self.pos`."""
        qh, kh, vh = self._qkv(x, w, S, cos, sin)
        if cache is not None:
            ttnn.fill_cache(cache[0], kh, 0)  # update_idx 0, so the tile-alignment rule is moot
            ttnn.fill_cache(cache[1], vh, 0)
        a = self._attend(qh, kh, vh, S, mask)
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG))
        return self._mlp(x, self._norm(x, w["fn"]), w, ttnn.DRAM_MEMORY_CONFIG)

    # ----------------------------------------------------------------------------
    # DECODE PATH -- one frame at a time; THIS is the hot loop
    # ----------------------------------------------------------------------------
    def _layer_step(self, x, w, cos, sin, cache, pos_t):
        """One decode position. x [1,1,3072] -> same, against `cache` written up to `pos_t`. `pos_t` is
        a device tensor: the cache writes and sdpa_decode both take the position that way."""
        qkv = ttnn.linear(
            self._norm(x, w["an"]),
            w["wqkv"],
            program_config=self.decode_prg["wqkv"],
            compute_kernel_config=COMPUTE_CONFIG,
        )
        qkv = ttnn.to_memory_config(ttnn.reshape(qkv, [1, 1, 1, _QKV_WIDTH]), _QKV_SHARD)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS)
        qh = ttnn.experimental.rotary_embedding_hf(
            qh, cos, sin, is_decode_mode=True, compute_kernel_config=COMPUTE_CONFIG
        )
        # Two calls, not the fused q+k rope, which uses the interleaved convention.
        kh = ttnn.experimental.rotary_embedding_hf(
            kh, cos, sin, is_decode_mode=True, compute_kernel_config=COMPUTE_CONFIG
        )
        # Two plain cache writes: the fused k+v write is slower here.
        ttnn.experimental.paged_update_cache(cache[0], kh, update_idxs_tensor=pos_t)
        ttnn.experimental.paged_update_cache(cache[1], vh, update_idxs_tensor=pos_t)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            qh,
            cache[0],
            cache[1],
            cur_pos_tensor=pos_t,
            scale=SCALE,
            compute_kernel_config=COMPUTE_CONFIG,
            program_config=_SDPA_PRG,
        )
        # No memory_config move: sdpa already emits the layout wo reads.
        a = ttnn.reshape(o, [1, 1, Q_WIDTH])
        # Residual as bias: decode is M=1, so the residual is a row vector.
        x = ttnn.linear(
            a,
            w["wo"],
            bias=ttnn.reshape(x, [1, DIM]),
            program_config=self.decode_prg["wo"],
            compute_kernel_config=COMPUTE_CONFIG,
            memory_config=_L1,
        )
        return self._mlp(x, self._norm(x, w["fn"]), w, _L1, self.decode_prg)

    @torch.no_grad()
    def prefill(self, embeds, apply_final_norm=True, last_only=False):
        """embeds torch [1,S,3072] -> hidden torch [1,S,3072], or [1,1,3072] if `last_only`. Pads with
        zero rows to PREFILL_MULTIPLE; the causal mask keeps real positions from seeing them."""
        S = embeds.shape[1]
        Sp = (S + PREFILL_MULTIPLE - 1) // PREFILL_MULTIPLE * PREFILL_MULTIPLE
        if self.caches and Sp > self.max_seq_len:
            raise ValueError(f"prompt pads to {Sp} but the KV cache holds {self.max_seq_len}")
        if Sp != S:
            embeds = torch.cat([embeds, embeds.new_zeros(1, Sp - S, DIM)], dim=1)
        cosb, sinb = rope_tables(Sp)
        up = lambda t, d=None: ttnn.from_torch(
            t.contiguous(), dtype=d or self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
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
            x = ttnn.rms_norm(x, weight=self.norm, epsilon=NORM_EPS, compute_kernel_config=COMPUTE_CONFIG)
        if last_only:
            return ttnn.to_torch(x).float().reshape(1, 1, DIM)
        return ttnn.to_torch(x).float().reshape(1, Sp, DIM)[:, :S]

    def prefill_last(self, embeds):
        """[1,P,3072] -> hidden of the LAST position [1,1,3072]. The pipeline's entry point; it is
        all the flow model ever sees."""
        return self.prefill(embeds, last_only=True)

    @torch.no_grad()
    def step(self, embed):
        """embed torch [1,1,3072] (one frame) -> hidden torch [1,1,3072]. Advances self.pos. sdpa_decode
        bounds the cache at `pos_t`, so nothing above it, prefill's padded rows included, is read."""
        if not self.caches:
            raise RuntimeError("step() needs a KV cache; construct with max_seq_len > 0")
        if self.pos >= self.max_seq_len:
            raise ValueError(f"KV cache full at {self.max_seq_len} positions")
        pos = self.pos
        cosb, sinb = rope_tables(1, offset=pos)
        up = lambda t, d=None: ttnn.from_torch(
            t.contiguous(), dtype=d or self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
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
