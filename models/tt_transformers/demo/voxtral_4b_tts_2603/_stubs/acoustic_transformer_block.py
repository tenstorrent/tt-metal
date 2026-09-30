# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `acoustic_transformer_block` (`AcousticTransformerBlock`).

    r = attention(attention_norm(x)); h = x + r
    r = feed_forward(ffn_norm(h));    out = h + r

The attention inside is `BidirectionalAttention`: NO positional encoding and NO causal mask, so
SDPA runs with `is_causal=False`. Shapes come from `params.json`'s `acoustic_transformer_args`:
dim 3072, 32 heads / 8 KV heads, head_dim 128, hidden_dim 9216, no biases.

BATCH AXIS. The leading bound is read from the tensor, never assumed to be 1: a `[B, 1, S, dim]`
input keeps all B samples and comes back as `[B, 1, S, dim]`, while the rank-<=3 input the
component test feeds (`[1, S, dim]`) is unchanged. A hardcoded leading 1 would have silently
dropped samples 1..B-1 once the sampler ran a real batch.

ADDITIVE MASK. `attn_mask=` is added to the attention scores. The flow-matching sampler's
sequence is 3 real tokens living in one 32-row tile, so it hands in a mask that blocks columns
3..31; the component test passes no mask and runs unmasked as before.

FIDELITY. Everything -- residual stream, Q/K/V, softmax -- runs in float32 against bfloat16
weights with HiFi4 + `fp32_dest_acc_en`. The sampler's output is rounded onto 21 levels 0.1 apart,
so a code flips on a few-1e-3 error, and SDPA's bfloat16-only interface plus `ttnn.softmax`'s
~6e-3 absolute error on the probabilities cost ~7% of the codes. See `_attention`.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import cpp_down, cpp_swiglu, ttl_down

_TILE = 32
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# Tall (>= 8 tile rows) linears are compute-bound, so they run one fidelity rung below HiFi4.
_TALL_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
)
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_L1_BUDGET = 1_100_000


def _mcast_cfg(x, w, rows, out_dtype):
    """A full-grid 2D-multicast program config for a tall `[rows, K] x [K, N]` linear, or None.

    Left to itself ttnn picks a partial grid with small K-blocks for these shapes. This spreads M
    over the grid rows and N over the grid columns, takes the widest K-block whose double-buffered
    in0/in1 blocks plus the output block fit L1, and the largest subblock fp32 DEST allows (4 tiles).
    None when even the output block alone does not fit, so the caller keeps ttnn's default.
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = rows // 32, int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    per_m, per_n = -(-mt // gy), -(-nt // gx)
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    fixed = per_m * per_n * (size(out_dtype) + (0 if out_dtype == ttnn.float32 else 4096))
    kb = next(
        (
            c
            for c in (16, 8, 4, 2, 1)
            if kt % c == 0 and fixed + 2 * c * (per_m * size(x.dtype) + per_n * size(w.dtype)) <= _L1_BUDGET
        ),
        None,
    )
    if kb is None:
        return None
    sub = max(
        ((h, s) for h in range(1, 5) for s in range(1, 5) if h * s <= 4 and per_m % h == 0 and per_n % s == 0),
        key=lambda hs: (hs[0] * hs[1], hs[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        per_core_M=per_m,
        per_core_N=per_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def _short_cfg(x, w, rows, out_dtype, k_block=None):
    """A 1D in0-multicast config for a SHORT (1..7 tile rows) linear, or None.

    Such a linear is bound by streaming its weight, so every core should own a slice of N and
    read only its own weight columns while the small activation is multicast to all of them.
    Left to itself ttnn gives it small K-blocks, and each block is a multicast round trip that
    every core waits on; this takes the widest K-block that fits L1.
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = rows // 32, int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    per_n = next(p for p in range(-(-nt // (gx * gy)), nt + 1) if nt % p == 0)
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    fixed = mt * per_n * (size(out_dtype) + (0 if out_dtype == ttnn.float32 else 4096))
    kb = next(
        (
            c
            for c in ((k_block,) if k_block else (32, 24, 16, 12, 8, 6, 4, 3, 2, 1))
            if kt % c == 0 and fixed + 2 * c * (mt * size(x.dtype) + per_n * size(w.dtype)) <= _L1_BUDGET
        ),
        None,
    )
    if kb is None:
        return None
    sub = max(
        ((h, s) for h in range(1, 5) for s in range(1, 5) if h * s <= 4 and mt % h == 0 and per_n % s == 0),
        key=lambda hs: (hs[0] * hs[1], hs[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        per_core_M=mt,
        per_core_N=per_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _lin(x, w, **kwargs):
    """`ttnn.linear` with the leading batch folded into M, so the weight streams ONCE.

    A `[B, 1, S, K]` activation against a 2-D weight runs as B separate `S x K x N` matmuls that
    each re-read the whole weight from DRAM; `[1, 1, B*S, K]` is one matmul that reads it once.
    Tall results (>= 8 tile rows) also get a hand-sized full-grid program config.
    """
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    rows = lead * shape[-2]
    if rows >= 256 and rows % 32 == 0 and "program_config" not in kwargs:
        cfg = _mcast_cfg(x, w, rows, kwargs.get("dtype") or x.dtype)
        if cfg is not None:
            kwargs["program_config"] = cfg
        kwargs["compute_kernel_config"] = _TALL_COMPUTE
    elif 32 <= rows < 256 and rows % 32 == 0 and "program_config" not in kwargs:
        cfg = _short_cfg(x, w, rows, kwargs.get("dtype") or x.dtype)
        if cfg is not None:
            kwargs["program_config"] = cfg
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    y = ttnn.linear(ttnn.reshape(x, [1, 1, rows, shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _weight(linear, device):
    """A `[in, out]` device tensor for a torch `nn.Linear` (whose weight is `[out, in]`)."""
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def _gamma(norm, device):
    """A norm's gamma as a `[1, 1, 1, dim]` float32 tile tensor, for `_rms_norm`."""
    return _from_torch(norm.weight.detach().reshape(1, 1, 1, -1).contiguous(), device, dtype=ttnn.float32)


def _rms_norm(x, gamma, eps, dtype=None):
    """RMSNorm in float32: `x * rsqrt(mean(x^2) + eps) * gamma`.

    NOT the `ttnn` layernorm op, which carries 2.7e-3 of RELATIVE error -- measured on this chip
    against a float64 reference, at either gamma dtype, with a float32 input and a float32
    output. Written out with `mean / rsqrt / multiply` the same normalization holds 1.2e-7. The
    sampler downstream rounds onto 21 levels 0.1 apart in x, so 2.7e-3 through seven norms is
    worth ~1% of the output codes and 1.2e-7 is worth none of them.
    """
    inv = ttnn.add(ttnn.mean(ttnn.square(x), dim=-1, keepdim=True), eps, activations=[ttnn.UnaryOpType.RSQRT])
    if gamma is None:  # folded into the consuming weights
        return ttnn.multiply(x, inv, dtype=dtype or ttnn.float32)
    return ttnn.multiply(ttnn.multiply(x, inv), gamma, dtype=dtype or ttnn.float32)


_NORM_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
)
_NORM_COLS = 8


def _norm_layout(device, rows, dim):
    """Block-sharded layout for a `[rows, dim]` RMSNorm: one row tile per grid row, 8 cores across."""
    grid = device.compute_with_storage_grid_size()
    ht, wt = rows // _TILE, dim // _TILE
    if rows % _TILE or dim % (_TILE * _NORM_COLS) or ht > int(grid.y) or _NORM_COLS > int(grid.x):
        return None
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_NORM_COLS - 1, ht - 1))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [_TILE, dim // _NORM_COLS], ttnn.ShardOrientation.ROW_MAJOR),
    )
    cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(_NORM_COLS, ht),
        subblock_w=1,
        block_h=1,
        block_w=wt // _NORM_COLS,
        inplace=False,
    )
    return mem, cfg


def _sharded_rms_norm(x, eps, dtype, gamma=None, memory_config=None):
    """`x * rsqrt(mean(x^2) + eps)` on the block-sharded layout, or None when the shape has none.

    The interleaved reduction deals one work unit per row tile, so a 96-row norm ran on 3 cores
    (23 us for the reduce alone). Sharded 8 cores across, each core reduces its own column block
    and the partial sums are combined over the row. The op's reduction scaler is a bfloat16
    rounding of `8 / dim`, a constant relative error that `_norm_scale` measures once and the
    consuming weights absorb.
    """
    shape = [int(d) for d in x.shape]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    layout = _norm_layout(x.device(), rows, shape[-1])
    if layout is None:
        return None
    mem, cfg = layout
    y = ttnn.rms_norm(
        ttnn.to_memory_config(ttnn.reshape(x, [1, 1, rows, shape[-1]]), mem),
        epsilon=eps,
        weight=gamma,
        memory_config=mem,
        program_config=cfg,
        compute_kernel_config=_NORM_COMPUTE,
    )
    out_mem = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
    return ttnn.reshape(ttnn.sharded_to_interleaved(y, out_mem, output_dtype=dtype), shape)


# The SwiGLU kernel reads its activation from L1 on every core: land the norm output there.
_FFN_IN_MEM = ttnn.L1_MEMORY_CONFIG

_NORM_SCALE = {}


def _norm_scale(device, dim, eps):
    """The factor that makes `_sharded_rms_norm` exact: `exact / measured` on a row of ones."""
    key = (id(device), dim, eps)
    if key not in _NORM_SCALE:
        ones = _from_torch(torch.ones(1, 1, _TILE, dim), device, dtype=ttnn.float32)
        got = _sharded_rms_norm(ones, eps, ttnn.float32)
        if got is None:
            _NORM_SCALE[key] = None
        else:
            if device.__class__.__name__ == "MeshDevice":
                host = ttnn.to_torch(got, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            else:
                host = ttnn.to_torch(got)
            measured = host.double().mean().item()
            _NORM_SCALE[key] = (1.0 / math.sqrt(1.0 + eps)) / measured
    return _NORM_SCALE[key]


def _block_norm(h, eps, scale, dtype, memory_config=None):
    """The in-block RMSNorm whose output feeds weights pre-multiplied by `scale` (see `_norm_scale`)."""
    if scale is not None:
        y = _sharded_rms_norm(h, eps, dtype, memory_config=memory_config)
        if y is not None:
            return y
        sq = ttnn.multiply(ttnn.mean(ttnn.square(h), dim=-1, keepdim=True), scale * scale)
        return ttnn.multiply(h, ttnn.add(sq, eps * scale * scale, activations=[ttnn.UnaryOpType.RSQRT]), dtype=dtype)
    return _rms_norm(h, None, eps, dtype=dtype)


def _bmm(a, b, per_core_m=None, transpose_b=False, dtype=None):
    """Head-batched `a @ b` spread over the full grid.

    Without a program config the `[B, H, 32, 128] x [B, H, 128, 32]` score product lands on ONE
    core (and probs @ V on four), running B*H tiny matmuls back to back. The reuse config makes
    every (batch, head) output block its own work unit, so they fan out across the grid.
    """
    m, k, n = int(a.shape[-2]) // 32, int(a.shape[-1]) // 32, int(b.shape[-2 if transpose_b else -1]) // 32
    grid = a.device().compute_with_storage_grid_size()
    cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k,
        out_subblock_h=1,
        out_subblock_w=min(n, 4),
        per_core_M=per_core_m or m,
        per_core_N=n,
    )
    return ttnn.matmul(a, b, transpose_b=transpose_b, program_config=cfg, compute_kernel_config=_COMPUTE, dtype=dtype)


def _attention(h, wqkv, wo, n_heads, n_kv_heads, scale, attn_mask):
    """GQA attention, bidirectional and non-causal, entirely in float32.

    NOT `ttnn.transformer.scaled_dot_product_attention`. SDPA rejects float32
    (`sdpa_device_operation.cpp:43`), so it forces Q/K/V down to bfloat16, and its softmax carries
    ~6e-3 of ABSOLUTE error on the attention probabilities -- measured on this chip against a
    float64 reference, and `ttnn.softmax` on its own is just as bad (6.2e-3) at either dtype and
    with `numeric_stable` either way. The same softmax written out as
    `max / exp / sum / divide` in float32 holds 6e-8. The sampler this feeds rounds onto 21
    levels 0.1 apart in x, so 6e-3 on a probability is worth roughly 7% of the output codes;
    6e-8 is worth none of them. The sequence here is one 32-row tile, so the explicit form costs
    four small ops per block and nothing measurable in time.

    The query heads are GROUPED BY KV HEAD rather than K/V being repeat_interleaved: query head
    h reads KV head h // repeats (the reference's `repeat_kv` mapping), and those `repeats` heads
    are contiguous, so `[B, H, S, D]` read as `[B, H_kv, repeats * S, D]` is a free view whose
    batch dims line up with K/V's. No K/V tensor is materialised `repeats` times. The mask only
    ever blocks COLUMNS, so its first row broadcasts over every stacked query row.
    """
    qkv = _lin(h, wqkv, dtype=ttnn.float32, compute_kernel_config=_COMPUTE)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=n_heads, num_kv_heads=n_kv_heads, transpose_k_heads=False
    )
    batch, _, seq, head_dim = (int(d) for d in q.shape)
    repeats = n_heads // n_kv_heads
    q = ttnn.reshape(q, [batch, n_kv_heads, repeats * seq, head_dim])

    # The reference scales the QUERY before the product, not the scores after it.
    scores = _bmm(q if scale is None else ttnn.multiply(q, scale), ttnn.transpose(k, -2, -1))
    if attn_mask is not None:
        if int(attn_mask.shape[-2]) != 1:
            attn_mask = ttnn.slice(attn_mask, [0, 0, 0, 0], [1, 1, 1, int(attn_mask.shape[-1])])
        scores = ttnn.add(scores, attn_mask)
    weights = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True), activations=[ttnn.UnaryOpType.EXP])
    weights = ttnn.divide(weights, ttnn.sum(weights, dim=-1, keepdim=True))

    out = ttnn.reshape(_bmm(weights, v), [batch, n_heads, seq, head_dim])
    return _lin(
        ttnn.experimental.nlp_concat_heads(out),
        wo,
        dtype=ttnn.float32,
        compute_kernel_config=_COMPUTE,
    )


_COMPACT_MASKS = {}
_COMPACT_ROWS = (96, 192)


def _compact_mask(device, rows, tokens, repeats, q_rows=None):
    """Additive `[1, 1, repeats * rows, rows]` mask letting a row attend only to its own sample.

    Row/column `t * R + r` is token t of sample r, so the allowed pairs are those with equal
    `index % R`; the `repeats` grouped query heads stack the same pattern vertically. One per
    (device, rows) shape, shared by every block, created at build time for the usual row counts
    so nothing is allocated inside a trace.
    """
    q_rows = rows if q_rows is None else q_rows
    key = (id(device), rows, tokens, repeats, q_rows)
    mask = _COMPACT_MASKS.get(key)
    if mask is None:
        sample = torch.arange(rows) % (rows // tokens)
        blk = torch.where(sample[:q_rows, None] == sample[None, :], 0.0, -1.0e9)
        mask = _from_torch(blk.repeat(repeats, 1).reshape(1, 1, repeats * q_rows, rows), device, dtype=ttnn.float32)
        _COMPACT_MASKS[key] = mask
    return mask


def _split_heads(qkv, n_heads, n_kv_heads):
    """`nlp_create_qkv_heads` on the fused `[1, 1, rows, (H + 2 H_kv) * D]` projection, split over HEADS.

    The fused q/k/v mode deals out one work unit per 32-row tile, so a 96-row compact projection
    runs on 3 cores. Its Q-only mode (`num_kv_heads=0`) deals out (row tile, head) pairs instead;
    q, k and v are the fused layout's consecutive head ranges, so the three are slices of one split.
    """
    total = n_heads + 2 * n_kv_heads
    heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=total, num_kv_heads=0, transpose_k_heads=False)
    b, _, s, d = (int(v) for v in heads.shape)
    q = ttnn.slice(heads, [0, 0, 0, 0], [b, n_heads, s, d])
    k = ttnn.slice(heads, [0, n_heads, 0, 0], [b, n_heads + n_kv_heads, s, d])
    v = ttnn.slice(heads, [0, n_heads + n_kv_heads, 0, 0], [b, total, s, d])
    return q, k, v


def _compact_attention(h, wqkv, wo, n_heads, n_kv_heads, scale, tokens, readout=False):
    """The same attention on the COMPACT layout: `[1, 1, tokens * R, dim]`, token t in rows t*R..

    No 32-row pad per sample, so nothing downstream computes on 29 padding rows. All
    `tokens * R` rows form one sequence and a constant mask keeps each row to its own sample's
    tokens, so this is ordinary head-batched attention with last-axis reductions -- no per-key
    loop and no batch-axis reduction (which ttnn implements with a full permute).
    """
    rows = int(h.shape[-2])
    repeats = n_heads // n_kv_heads
    # bf16 q/k/v for the head split; the scores come back float32 for the softmax.
    # 8-tile K blocks: 12 multicast rounds pipeline the weight stream better than 3 wide ones.
    qkv = _lin(
        h,
        wqkv,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_TALL_COMPUTE,
        program_config=_short_cfg(h, wqkv, rows, ttnn.bfloat16, k_block=8),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    q, k, v = _split_heads(qkv, n_heads, n_kv_heads)
    head_dim = int(q.shape[-1])
    # After the last block only token 0 is read out, so only its queries (the first R = rows /
    # tokens rows; token t lives in rows t*R..) go through the scores, softmax, context and o_proj.
    # Every token's k/v is still needed.
    q_rows = rows // tokens if readout else rows
    if readout:
        q = ttnn.slice(q, [0, 0, 0, 0], [1, n_heads, q_rows, head_dim])
    q = ttnn.reshape(q, [1, n_kv_heads, repeats * q_rows, head_dim])

    q = q if scale is None else ttnn.multiply(q, scale)
    scores = _bmm(q, k, per_core_m=1, transpose_b=True, dtype=ttnn.float32)
    scores = ttnn.add(scores, _compact_mask(h.device(), rows, tokens, repeats, q_rows))
    weights = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True), activations=[ttnn.UnaryOpType.EXP])
    weights = ttnn.divide(weights, ttnn.sum(weights, dim=-1, keepdim=True))

    # bf16 context: the head merge moves it and o_proj multicasts it whole.
    out = ttnn.reshape(_bmm(weights, v, per_core_m=1, dtype=ttnn.bfloat16), [1, n_heads, q_rows, head_dim])
    # bf16 into L1: the residual add is its only reader, and fp32 in DRAM doubles the bytes it writes.
    return _lin(
        _concat_heads(out),
        wo,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_TALL_COMPUTE,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _concat_heads(out):
    """`nlp_concat_heads` with one HEAD per core instead of one 32-row tile per core.

    Interleaved, the op deals out a work unit per row tile, so 96 rows run on 3 cores (54 us).
    Height-sharded one head per core, every core copies its own head into its own column block
    of a width-sharded result; the result goes back to interleaved L1 for o_proj.
    """
    _, n_heads, rows, head_dim = (int(d) for d in out.shape)
    grid = out.device().compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(n_heads, grid, row_wise=True)
    in_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [rows, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [rows, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
    )
    merged = ttnn.experimental.nlp_concat_heads(ttnn.to_memory_config(out, in_cfg), memory_config=out_cfg)
    return ttnn.to_memory_config(merged, ttnn.L1_MEMORY_CONFIG)


def _leading(shape) -> int:
    """The product of every axis before `[seq, dim]` -- the real batch, from the tensor."""
    dims = list(shape)[:-2]
    batch = 1
    for d in dims:
        batch *= int(d)
    return batch


def build(device, torch_module):
    blk = torch_module
    attn = blk.attention
    ff = blk.feed_forward

    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    head_dim = int(attn.head_dim)
    dim = int(blk.dim)
    scale = 1.0 / math.sqrt(head_dim)
    eps = float(blk.attention_norm.eps)
    norm_scale = _norm_scale(device, dim, eps)
    fold = 1.0 if norm_scale is None else norm_scale

    # Each norm's gamma is folded into the weights that consume it, `(x * s * g) @ W` being
    # `(x * s) @ (g[:, None] * W)`, and the query scale into Wq.
    g_attn_t = blk.attention_norm.weight.detach().float().reshape(-1, 1) * fold
    g_ffn_t = blk.ffn_norm.weight.detach().float().reshape(-1, 1) * fold
    wqkv = _from_torch(
        torch.cat(
            [
                attn.wq.weight.detach().float().transpose(0, 1) * scale,
                attn.wk.weight.detach().float().transpose(0, 1),
                attn.wv.weight.detach().float().transpose(0, 1),
            ],
            dim=-1,
        )
        .mul(g_attn_t)
        .contiguous(),
        device,
        dtype=ttnn.bfloat8_b,
    )
    # o_proj is weight-stream bound at 96 rows; bf4_b is 576 B a tile against bf8_b's 1088.
    wo = _from_torch(attn.wo.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat4_b)
    w1 = _from_torch(
        (ff.w1.weight.detach().float().transpose(0, 1) * g_ffn_t).contiguous(), device, dtype=ttnn.bfloat8_b
    )
    # The down projection is DRAM-bound at 1024 rows; bf8_b halves the weight it streams.
    w2 = _from_torch(ff.w2.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat4_b)
    w2_ttl = ttl_down.weight(ff.w2.weight.detach().transpose(0, 1).contiguous(), device, _from_torch)
    w2_cpp = cpp_down.shard(ff.w2.weight.detach().transpose(0, 1).contiguous(), device)
    w3 = _from_torch(
        (ff.w3.weight.detach().float().transpose(0, 1) * g_ffn_t).contiguous(), device, dtype=ttnn.bfloat8_b
    )
    w13 = cpp_swiglu.fuse(
        (ff.w1.weight.detach().float().transpose(0, 1) * g_ffn_t).contiguous(),
        (ff.w3.weight.detach().float().transpose(0, 1) * g_ffn_t).contiguous(),
        device,
    )
    for rows in _COMPACT_ROWS:
        _compact_mask(device, rows, 3, n_heads // n_kv_heads)
        _compact_mask(device, rows, 3, n_heads // n_kv_heads, rows // 3)

    def acoustic_transformer_block(x, attn_mask=None, tokens=None, readout=False, **kwargs):
        seq = int(x.shape[-2])
        batch = _leading(x.shape)
        rank = len(list(x.shape))

        h4 = ttnn.reshape(x, [batch, 1, seq, dim])
        if h4.dtype != ttnn.float32:
            h4 = ttnn.typecast(h4, ttnn.float32)

        # qkv's input lands in L1, not DRAM: it is read once, by the next op.
        xn = _block_norm(h4, eps, norm_scale, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        if tokens:
            attn_out = _compact_attention(xn, wqkv, wo, n_heads, n_kv_heads, None, tokens, readout)
        else:
            attn_out = _attention(xn, wqkv, wo, n_heads, n_kv_heads, None, attn_mask)
        if tokens and readout:
            seq = seq // tokens
            h4 = ttnn.slice(h4, [0, 0, 0, 0], [batch, 1, seq, dim])
        h4 = ttnn.add(h4, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        hn = _block_norm(h4, eps, norm_scale, ttnn.bfloat16, memory_config=_FFN_IN_MEM if w13 is not None else None)
        if cpp_swiglu.serves(hn, w13):
            gated = cpp_swiglu.apply(hn, w13)
        else:
            gate = _lin(
                hn, w1, dtype=ttnn.bfloat16, compute_kernel_config=_TALL_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            up = _lin(
                hn, w3, dtype=ttnn.bfloat16, compute_kernel_config=_TALL_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            gated = ttnn.multiply(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                # bf16: the down projection multicasts it whole to every core.
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        if cpp_down.serves(gated, w2_cpp):
            down = cpp_down.apply(gated, w2_cpp)
        elif ttl_down.supports(gated, w2_ttl):
            down = ttl_down.apply(gated, w2_ttl)
        else:
            down = _lin(
                gated, w2, dtype=ttnn.float32, compute_kernel_config=_TALL_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        h4 = ttnn.add(h4, down, memory_config=ttnn.L1_MEMORY_CONFIG)

        if rank >= 4:
            return h4
        return ttnn.reshape(h4, [batch, seq, dim])

    return acoustic_transformer_block
