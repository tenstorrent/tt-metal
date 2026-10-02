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


def _rope_shard_for(batch, device_grid):
    """Height-sharded one tile per user over `batch` cores: the layout rotary_embedding_hf's decode
    mode reads for cos/sin and the one nlp_concat_heads_decode takes for the sdpa output (the same
    spec tt_transformers' RotarySetupHF builds). batch=1 is _ROPE_SHARD, unchanged."""
    if batch == 1:
        return _ROPE_SHARD
    # ONE rectangle of exactly `batch` cores. nlp_concat_heads_decode dereferences an optional
    # sub-core grid when the input grid is more than one range (bad optional access on a 13-wide
    # chip, where num_cores_to_corerangeset(32) is 13+13+6), so lay the users out 8 per row like
    # tt_transformers' attention-output shard does.
    if batch <= 8:
        x, y = batch, 1
    elif batch % 8 == 0:
        x, y = 8, batch // 8
    else:
        raise ValueError(f"max_batch must be <= 8 or a multiple of 8 for a rectangular user grid, got {batch}")
    if x > device_grid.x or y > device_grid.y:
        raise RuntimeError(f"device grid {device_grid.x}x{device_grid.y} cannot hold a {x}x{y} user grid")
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})
    return ttnn.create_sharded_memory_config(
        (TILE, HEAD_DIM),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# sdpa_decode program config. Faster chunk/grid choices exist but are not exact at every cache
# position; change it only against a sweep over positions.
_SDPA_PRG = ttnn.SDPAProgramConfig(
    q_chunk_size=TILE, k_chunk_size=512, compute_with_storage_grid_size=ttnn.CoreCoord(8, 2)
)


def _sdpa_prg_for(batch, device_grid):
    """sdpa_decode needs at least one core per user (sdpa_decode_program_factory: cores >= B), so
    the 8x2 batch-1 grid cannot serve B > 16. B > 1 takes 8x8 (tt_transformers' 32-user choice)
    when the chip has it, else 8 x ceil(B/8). batch=1 is _SDPA_PRG, unchanged."""
    if batch == 1:
        return _SDPA_PRG
    x = min(8, device_grid.x)
    y = max(-(-batch // x), min(8, device_grid.y))
    if x * y < batch:
        raise RuntimeError(f"device grid {device_grid.x}x{device_grid.y} has no {batch}-core grid for sdpa_decode")
    return ttnn.SDPAProgramConfig(
        q_chunk_size=TILE, k_chunk_size=512, compute_with_storage_grid_size=ttnn.CoreCoord(x, y), exp_approx_mode=False
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


def _mm1d(grid, in0_block_w, per_core_n, activation=None, per_core_m=1):
    """1D multicast: split N across the grid, broadcast in0. The batch-1 decode shape at
    per_core_m=1; `per_core_m` tiles of rows per core when the input carries more than one tile
    (the flow model at B users folds 2*B*3 rows)."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        # largest legal width: osh*osw <= 4 and per_core_N % osw == 0, both TT_FATAL.
        out_subblock_w=next(s for s in (4, 3, 2, 1) if per_core_n % s == 0),
        per_core_M=per_core_m,
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


def decode_program_configs(grid, m_tiles=1):
    """-> {matmul name: program config} on `grid` (see decode_grid). `m_tiles` is the number of
    32-row tiles the input carries: 1 for decode (unchanged), more for the flow model at B users."""
    return {name: _mm1d(grid, *split, per_core_m=int(m_tiles)) for name, split in _DECODE_SPLIT.items()}


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

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, n_layers=N_LAYERS, state=None, max_seq_len=2048, max_batch=1):
        """`state` takes an already-loaded `load_backbone_state` dict so the fp32 weights load once;
        `max_seq_len=0` skips the KV cache. `max_batch` is the number of users decoded together
        (one row each, at most one tile of rows); 1 keeps every code path exactly as before."""
        check_device_grid(device)
        if not 1 <= int(max_batch) <= TILE:
            raise ValueError(f"max_batch must be in 1..{TILE} (one tile of decode rows), got {max_batch}")
        self.device = device
        self.decode_prg = decode_program_configs(decode_grid(device.compute_with_storage_grid_size()))
        self.dtype = DTYPE
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.max_batch = int(max_batch)
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
        z = torch.zeros(self.max_batch, N_KV_HEADS, max_seq_len, HEAD_DIM)
        self.caches = [(up(z, DTYPE), up(z, DTYPE)) for _ in range(n_layers)] if max_seq_len else []
        # Batched decode gathers each user's cos/sin row from these tables ON DEVICE (the
        # tt_transformers RotarySetupHF pattern); `step()` at max_batch=1 keeps the host path.
        self._rope_mem = _rope_shard_for(self.max_batch, device.compute_with_storage_grid_size())
        self.sdpa_prg = _sdpa_prg_for(self.max_batch, device.compute_with_storage_grid_size())
        if max_seq_len:
            cos_t, sin_t = rope_tables(max_seq_len)
            self._cos_tab = ttnn.from_torch(
                cos_t.contiguous(), dtype=DTYPE, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            self._sin_tab = ttnn.from_torch(
                sin_t.contiguous(), dtype=DTYPE, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )

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
        # Residual as the matmul bias on decode (one row) only; a bias broadcasts ONE row over every
        # output row, so prefill and batched decode (max_batch > 1) keep the add. B=1 is unchanged.
        if prg and self.max_batch == 1:
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

    def _layer(self, x, w, S, cos, sin, mask, cache=None, user=0):
        """x [1,S,3072] -> same. Pre-norm GQA with RoPE + causal mask, then SwiGLU. Fills `cache`
        row `user` with all padded rows; decode reads it only up to that user's position."""
        qh, kh, vh = self._qkv(x, w, S, cos, sin)
        if cache is not None:
            ttnn.fill_cache(cache[0], kh, user)  # one batch row; prefill runs one user at a time
            ttnn.fill_cache(cache[1], vh, user)
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
        B = self.max_batch
        # One row per user; B <= 32 rows fill the same (32, 6144) shard the batch-1 path used.
        qkv = ttnn.to_memory_config(ttnn.reshape(qkv, [1, 1, B, _QKV_WIDTH]), _QKV_SHARD)
        if B == 1:
            qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS)
        else:
            # Heads land one tile per user on B cores, matching the cos/sin shard the rope op reads.
            qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS, memory_config=self._rope_mem
            )
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
            program_config=self.sdpa_prg,
        )
        if B == 1:
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
        else:
            # [1, B, heads, hd] -> one tile per user -> [1, 1, B, heads*hd] -> [1, B, 4096]; then a
            # per-row residual add, because a bias would broadcast row 0 over all B rows.
            o = ttnn.to_memory_config(o, self._rope_mem)
            a = ttnn.experimental.nlp_concat_heads_decode(o, num_heads=N_HEADS)
            a = ttnn.to_memory_config(a, _L1)  # [1, 1, 32, 4096]: the op pads the user axis to 32
            if B < TILE:
                a = ttnn.slice(a, [0, 0, 0, 0], [1, 1, B, Q_WIDTH])
            a = ttnn.reshape(a, [1, B, Q_WIDTH])
            x = ttnn.add(
                x,
                ttnn.linear(
                    a,
                    w["wo"],
                    program_config=self.decode_prg["wo"],
                    compute_kernel_config=COMPUTE_CONFIG,
                    memory_config=_L1,
                ),
                memory_config=_L1,
            )
        return self._mlp(x, self._norm(x, w["fn"]), w, _L1, self.decode_prg)

    # ---- batched decode (max_batch > 1): per-user positions, everything on device ----------
    def rot_mats(self, pos_u32):
        """pos_u32: device uint32 [1, Bpad] (Bpad = B rounded up to 32) -> (cos, sin), each
        [1, B, 1, 128] height-sharded one tile per user, gathered from the device tables."""
        B = self.max_batch
        cos = ttnn.embedding(pos_u32, self._cos_tab, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin = ttnn.embedding(pos_u32, self._sin_tab, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)  # [1, Bpad, 1(pad 32), 128]
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)
        if B % TILE:
            cos, sin = cos[:, :B, :, :], sin[:, :B, :, :]
        return ttnn.interleaved_to_sharded(cos, self._rope_mem), ttnn.interleaved_to_sharded(sin, self._rope_mem)

    def step_device(self, x, pos_u32, pos_i32):
        """The whole decode step on device tensors: x [1, B, 3072], pos_u32 [1, Bpad] uint32 for the
        rope gather, pos_i32 [B] int32 for the cache write and sdpa. -> normed hidden [1, B, 3072]
        on device. No host work inside, so it can be captured as a trace."""
        cos, sin = self.rot_mats(pos_u32)
        for i, w in enumerate(self.layers):
            x = self._layer_step(x, w, cos, sin, self.caches[i], pos_i32)
        return self._norm(x, self.norm)

    @staticmethod
    def pos_tensors(positions, device):
        """torch int [B] -> (pos_u32 [1, Bpad] uint32, pos_i32 [B] int32), both on device."""
        positions = torch.as_tensor(positions, dtype=torch.int32).reshape(-1)
        B = positions.shape[0]
        Bpad = -(-B // TILE) * TILE
        padded = torch.zeros(1, Bpad, dtype=torch.int32)
        padded[0, :B] = positions
        pos_u32 = ttnn.from_torch(padded, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        pos_i32 = ttnn.from_torch(positions, dtype=ttnn.int32, device=device)
        return pos_u32, pos_i32

    @torch.no_grad()
    def step_batched(self, embeds, positions):
        """embeds torch [1, B, 3072] (one frame per user), positions int [B] (each user's own cache
        position) -> hidden torch [1, B, 3072]. Leaves `self.pos` alone: the caller owns positions."""
        if not self.caches:
            raise RuntimeError("step_batched() needs a KV cache; construct with max_seq_len > 0")
        B = self.max_batch
        if tuple(embeds.shape) != (1, B, DIM):
            raise ValueError(f"embeds must be [1, {B}, {DIM}], got {tuple(embeds.shape)}")
        positions = torch.as_tensor(positions, dtype=torch.int32).reshape(-1)
        if positions.shape[0] != B or int(positions.max()) >= self.max_seq_len:
            raise ValueError(f"positions must be [{B}] and below {self.max_seq_len}, got {positions.tolist()}")
        x = ttnn.from_torch(embeds.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        pos_u32, pos_i32 = self.pos_tensors(positions, self.device)
        return ttnn.to_torch(self.step_device(x, pos_u32, pos_i32)).float().reshape(1, B, DIM)

    @torch.no_grad()
    def prefill(self, embeds, apply_final_norm=True, last_only=False, user=0):
        """embeds torch [1,S,3072] -> hidden torch [1,S,3072], or [1,1,3072] if `last_only`. Pads with
        zero rows to PREFILL_MULTIPLE; the causal mask keeps real positions from seeing them.
        `user` picks the KV-cache row written (0 on the batch-1 path); prefill is one user at a time."""
        if not 0 <= int(user) < self.max_batch:
            raise ValueError(f"user must be in 0..{self.max_batch - 1}, got {user}")
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
            x = self._layer(x, w, Sp, cos, sin, mask, self.caches[i] if self.caches else None, user=int(user))
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

    # ----------------------------------------------------------------------------
    # BATCHED PREFILL -- a group of users in one pass (mixed voices and lengths)
    # ----------------------------------------------------------------------------
    @staticmethod
    def batched_prefill_fits(B, Sp, limit_bytes=1 << 30):
        """Keep a group's attention working set (q/k/v heads [B,32,Sp,128] bf16 and the fused sdpa's
        scratch) under `limit_bytes`; longer prompts take the per-user path."""
        return B * N_HEADS * Sp * HEAD_DIM * 2 * 4 <= limit_bytes

    def _layer_batched(self, x, w, B, S, cos, sin, first_row, cache=None):
        """x [1,B*S,3072] (a group of B users stacked) -> same. Fused causal sdpa over the padded
        prompts (a pad row only ever attends to real rows before it). Writes the group's K/V into
        cache rows [first_row, first_row+B) with one concat + copy per tensor; rows above S keep what
        they held (decode overwrites a row before reading it)."""
        M = B * S
        cc = COMPUTE_CONFIG
        h = self._norm(x, w["an"])
        qkv = ttnn.linear(h, w["wqkv"], compute_kernel_config=cc)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(qkv, [B, 1, S, _QKV_WIDTH]),
            num_heads=N_HEADS,
            num_kv_heads=N_KV_HEADS,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # RoPE with the users folded into the heads axis, so the [1,1,S,hd] tables broadcast as on
        # the batch-1 path (a free view: the last two dims are unchanged).
        qh = ttnn.reshape(
            self._rope(ttnn.reshape(qh, [1, B * N_HEADS, S, HEAD_DIM]), cos, sin), [B, N_HEADS, S, HEAD_DIM]
        )
        kh = ttnn.reshape(
            self._rope(ttnn.reshape(kh, [1, B * N_KV_HEADS, S, HEAD_DIM]), cos, sin), [B, N_KV_HEADS, S, HEAD_DIM]
        )
        if cache is not None:
            Bc, L = self.max_batch, self.max_seq_len
            for t, c in ((kh, cache[0]), (vh, cache[1])):
                parts = []
                if first_row > 0:
                    parts.append(ttnn.slice(c, [0, 0, 0, 0], [first_row, N_KV_HEADS, L, HEAD_DIM]))
                if S < L:
                    tail = ttnn.slice(c, [first_row, 0, S, 0], [first_row + B, N_KV_HEADS, L, HEAD_DIM])
                    parts.append(ttnn.concat([t, tail], dim=2))
                    ttnn.deallocate(tail)
                else:
                    parts.append(t)
                if first_row + B < Bc:
                    parts.append(ttnn.slice(c, [first_row + B, 0, 0, 0], [Bc, N_KV_HEADS, L, HEAD_DIM]))
                full = ttnn.concat(parts, dim=0) if len(parts) > 1 else parts[0]
                ttnn.copy(full, c)
                for q in parts + [full]:
                    if q is not t and q is not c:
                        ttnn.deallocate(q)
        a = ttnn.transformer.scaled_dot_product_attention(qh, kh, vh, is_causal=True, compute_kernel_config=cc)
        a = ttnn.reshape(ttnn.experimental.nlp_concat_heads(a), [1, M, Q_WIDTH])
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=cc))
        return self._mlp(x, self._norm(x, w["fn"]), w, ttnn.DRAM_MEMORY_CONFIG)

    @torch.no_grad()
    def prefill_batched(self, embeds_list, first_row=0):
        """[torch [1,S_b,3072]] (a group of B <= max_batch users) -> hidden of each user's LAST
        position, torch fp32 [B,3072], with KV-cache rows first_row..first_row+B-1 filled. One pass
        for the group: prompts right-padded to a common tile multiple, causal attention keeps real
        positions from seeing the pads, the pads' K/V land above each user's length where decode
        overwrites them before reading. Mixed voices and lengths. Row-equivalent to
        prefill(embeds[b], last_only=True, user=first_row+b) (same ops, batched)."""
        B = len(embeds_list)
        if not 1 <= B <= self.max_batch - first_row:
            raise ValueError(f"group of {B} users does not fit rows {first_row}.. of max_batch={self.max_batch}")
        lens = [int(e.shape[1]) for e in embeds_list]
        Sp = -(-max(lens) // self.PREFILL_GROUP_MULTIPLE) * self.PREFILL_GROUP_MULTIPLE
        if self.caches and Sp > self.max_seq_len:
            raise ValueError(f"prompt pads to {Sp} but the KV cache holds {self.max_seq_len}")
        xh = torch.zeros(B, Sp, DIM)
        for b, e in enumerate(embeds_list):
            xh[b, : lens[b]] = e.reshape(lens[b], DIM)
        cosb, sinb = rope_tables(Sp)
        up = lambda t, d=None: ttnn.from_torch(
            t.contiguous(), dtype=d or self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        cos = up(cosb.reshape(1, 1, Sp, HEAD_DIM))
        sin = up(sinb.reshape(1, 1, Sp, HEAD_DIM))
        x = up(xh.reshape(1, B * Sp, DIM))
        for i, w in enumerate(self.layers):
            x = self._layer_batched(x, w, B, Sp, cos, sin, first_row, self.caches[i] if self.caches else None)
        x = ttnn.reshape(x, [B, Sp, DIM])
        rows = [ttnn.slice(x, [b, lens[b] - 1, 0], [b + 1, lens[b], DIM]) for b in range(B)]
        h = ttnn.concat(rows, dim=0) if B > 1 else rows[0]  # [B,1,3072]
        h = ttnn.rms_norm(h, weight=self.norm, epsilon=NORM_EPS, compute_kernel_config=COMPUTE_CONFIG)
        return ttnn.to_torch(h).float().reshape(B, DIM)

    PREFILL_GROUP_MULTIPLE = 128  # a prompt's padded length is its own bucket, not the batch's

    @staticmethod
    def prefill_groups(lens):
        """Users sorted by prompt length and grouped by length bucket (128-token multiples), each
        group prefilled in one pass padded to its bucket. The padded length is a property of the
        request, not of the batch, so two identical prompts in one batch land in the same group and
        get identical results, and a request's prefill does not change with the batch's other prompts
        (up to matmul blocking, which can vary with the group size). -> (order, bounds): user indices
        sorted by length, group boundaries in that order."""
        m = TtVoxtralGPT.PREFILL_GROUP_MULTIPLE
        order = sorted(range(len(lens)), key=lambda i: lens[i])
        bounds = [0]
        for k in range(1, len(order)):
            if -(-lens[order[k]] // m) != -(-lens[order[k - 1]] // m):
                bounds.append(k)
        bounds.append(len(order))
        return order, bounds

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
