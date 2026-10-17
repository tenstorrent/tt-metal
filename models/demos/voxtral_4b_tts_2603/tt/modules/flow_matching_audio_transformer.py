# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `flow_matching_audio_transformer` (`acoustic_transformer`).

The deterministic core of the flow-matching sampler -- the velocity field plus the semantic head:

    t_proj   = time_projection(time_embedding(t))
    llm_proj = llm_projection(llm_hidden)
    seq      = [input_projection(x_t), t_proj, llm_proj]     # THREE tokens
    velocity = acoustic_codebook_output(norm(blocks(seq))[:, 0, :])
    semantic = semantic_codebook_output(llm_hidden)          # reads the RAW hidden state

The Euler loop around this is not ported here because it is not a function of its inputs: it draws
`x_0 = torch.randn(...)` inside the module. `x_t` is supplied instead, which leaves every weight
under test (see the note in `tests/pcc/test_flow_matching_audio_transformer.py`).

**The sequence is padded to one tile.** The real sequence is 3 tokens, which in TILE layout lives
inside a 32-row tile with 29 rows of padding. Rather than hope an op respects a sub-tile logical
length, the sequence is explicitly built 32 rows long and attention is given an additive mask that
blocks columns 3..31 -- so rows 0..2 attend to exactly the three real tokens, which is what the
reference computes. Only row 0 is ever read out.

The blocks are `AcousticTransformerBlock`s: bidirectional, RoPE-free, GQA 32/8, head_dim 128,
dim 3072, hidden 9216, `norm_eps` 1e-5, no biases.

The activation path runs in float32 (weights stay bfloat16 -- `ttnn.linear` takes a float32
activation against a bfloat16 weight). All-bfloat16 cleared the 0.99 target by only 0.0007, and
that margin is not worth holding: the residual accumulates over three blocks and the read-out is a
single row. The attention is written out rather than calling SDPA, which rejects float32
(`sdpa_device_operation.cpp:43`) -- see `_attention` -- because the sampler around this field
rounds onto 21 levels and SDPA's bfloat16 softmax was worth ~7% of the output codes.

THE 29-ROW TILE PAD IS A BUILD-TIME BUFFER. It used to be a `ttnn.zeros(...)` on every call,
which cannot live inside a captured trace (a trace replays kernels, it cannot allocate). It is
now a persistent device buffer, created once per distinct batch and reused; `build(..., batch=N)`
pre-creates the one the caller will actually use so nothing is allocated on the first traced
call. The numerics are unchanged -- same shape, same zeros, same concat.
"""

from __future__ import annotations

import math

import torch

import ttnn

_TILE = 32
_MASK_NEG = -1.0e9


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
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


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


_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# Tall (>= 8 tile rows) linears get their own config object; fidelity stays HiFi4 (fp32 accumulation),
# like every acoustic linear, because the stage's output is rounded onto 21 code levels.
_TALL_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
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


def _bmm(a, b, per_core_m=None, transpose_b=False, dtype=None):
    """Head-batched `a @ b` spread over the full grid.

    Without a program config the `[B, H, 32, 128] x [B, H, 128, 32]` score product lands on ONE
    core (and probs @ V on four), running B*H tiny matmuls back to back. The reuse config makes
    every (batch, head) output block its own work unit, so they fan out across the grid.
    """
    m, k, n = int(a.shape[-2]) // 32, int(a.shape[-1]) // 32, int(b.shape[-2 if transpose_b else -1]) // 32
    grid = a.device().compute_with_storage_grid_size()
    # At most one (batch, head, M-block) work unit per core: with more units than cores this config
    # returns WRONG values (192 units on the 130-core grid gave PCC 0.80 against torch; 96 is exact).
    lead = 1
    for d in list(a.shape)[:-2]:
        lead *= int(d)
    per_core_m = per_core_m or m
    while m % per_core_m or lead * (m // per_core_m) > int(grid.x) * int(grid.y):
        per_core_m += 1
    cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k,
        out_subblock_h=1,
        out_subblock_w=max(d for d in range(1, min(n, 4) + 1) if n % d == 0),
        per_core_M=per_core_m,
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
    # Widest K blocks: the bf16 output is also the L1 partial-sum format, so fewer blocks round fewer times.
    qkv = _lin(
        h,
        wqkv,
        dtype=ttnn.float32,
        compute_kernel_config=_TALL_COMPUTE,
        program_config=_short_cfg(h, wqkv, rows, ttnn.float32),
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

    # bf16 context: the head merge moves it and o_proj multicasts it whole; the scores and the
    # softmax that produced it stay float32.
    out = ttnn.reshape(_bmm(weights, v, per_core_m=1, dtype=ttnn.float32), [1, n_heads, q_rows, head_dim])
    # bf16 into L1: the residual add is its only reader, and fp32 in DRAM doubles the bytes it writes.
    return _lin(
        _concat_heads(out),
        wo,
        dtype=ttnn.float32,
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


def _residual_add(h, x):
    """The residual stream stays in L1: each block's norm and next add read it right back."""
    return ttnn.add(h, x, memory_config=ttnn.L1_MEMORY_CONFIG)


def _compile_block(device, blk, mask):
    """One `AcousticTransformerBlock` as a callable on `[B, 1, TILE, dim]`."""
    attn = blk.attention
    ff = blk.feed_forward
    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    head_dim = int(attn.head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    # Each norm's gamma is folded into the weights that consume it: `(x * s * g) @ W` is
    # `(x * s) @ (g[:, None] * W)`, one full-width multiply fewer per norm.
    eps = float(blk.attention_norm.eps)
    g_attn_t = blk.attention_norm.weight.detach().float().reshape(-1, 1)
    g_ffn_t = blk.ffn_norm.weight.detach().float().reshape(-1, 1)
    wqkv = _from_torch(
        torch.cat(
            [
                # The query scale folded into Wq: the reference scales the query before the product.
                attn.wq.weight.detach().float().transpose(0, 1) * scale,
                attn.wk.weight.detach().float().transpose(0, 1),
                attn.wv.weight.detach().float().transpose(0, 1),
            ],
            dim=-1,
        )
        .mul(g_attn_t)
        .contiguous(),
        device,
        dtype=ttnn.bfloat16,
    )
    wo = _from_torch(attn.wo.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat16)
    # bfloat16 FFN weights at HiFi4: the 8-bit / HiFi2 variant was faster but cost the stage its accuracy.
    w1_t, w3_t = ((m.weight.detach().float().transpose(0, 1) * g_ffn_t).contiguous() for m in (ff.w1, ff.w3))
    w1, w3 = (_from_torch(t, device, dtype=ttnn.bfloat16) for t in (w1_t, w3_t))
    # The down projection stays bfloat16: bf8_b weights here cost the acoustic stage its accuracy.
    w2 = _from_torch(ff.w2.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat16)
    for rows in _COMPACT_ROWS:
        _compact_mask(device, rows, 3, n_heads // n_kv_heads)
        _compact_mask(device, rows, 3, n_heads // n_kv_heads, rows // 3)

    def run(h, tokens=None, readout=False):
        xn = _rms_norm(h, None, eps, dtype=ttnn.float32)
        if tokens:
            attn_out = _compact_attention(xn, wqkv, wo, n_heads, n_kv_heads, None, tokens, readout)
        else:
            attn_out = _attention(xn, wqkv, wo, n_heads, n_kv_heads, None, mask)
        if tokens and readout:
            h = ttnn.slice(h, [0, 0, 0, 0], [1, 1, int(h.shape[-2]) // tokens, int(h.shape[-1])])
        h = _residual_add(h, attn_out)

        hn = _rms_norm(h, None, eps, dtype=ttnn.float32)
        gated = ttnn.multiply(
            _lin(hn, w1, dtype=ttnn.float32, compute_kernel_config=_TALL_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG),
            _lin(hn, w3, dtype=ttnn.float32, compute_kernel_config=_TALL_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG),
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            # Consumed once, by the down projection: hand it over in L1, not through DRAM.
            dtype=ttnn.float32,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return _residual_add(
            h,
            _lin(
                gated, w2, dtype=ttnn.float32, compute_kernel_config=_TALL_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG
            ),
        )

    return run


def build(device, torch_module, batch=None, layers=None):
    at = getattr(torch_module, "inner", torch_module)
    args = at.acoustic_transformer_args
    dim = int(args.dim)
    n_real_tokens = 3

    # float32, not bfloat16: `inv_freq` is COMPUTED by the module (exp of an arange), not a
    # checkpoint tensor, so bfloat16 would put 0.4% of error into the sinusoidal phase that
    # nothing downstream can recover.
    inv_freq = _from_torch(at.time_embedding.inv_freq.detach().reshape(1, -1).contiguous(), device, dtype=ttnn.float32)
    w_time = _weight(at.time_projection, device)
    w_llm = _weight(at.llm_projection, device)
    w_input = _weight(at.input_projection, device)
    # The final norm's gamma, folded into the only weight that reads the normalised rows.
    w_acoustic = _from_torch(
        (
            at.acoustic_codebook_output.weight.detach().float().transpose(0, 1)
            * at.norm.weight.detach().float().reshape(-1, 1)
        ).contiguous(),
        device,
    )
    w_semantic = _weight(at.semantic_codebook_output, device)
    semantic_bias = None
    if at.semantic_codebook_output.bias is not None:
        semantic_bias = _from_torch(at.semantic_codebook_output.bias.detach().reshape(1, 1, 1, -1), device)

    # Columns 3..31 of the padded tile are not real tokens; block them so rows 0..2 attend to
    # exactly the three the reference builds.
    mask_torch = torch.zeros(1, 1, 1, _TILE)
    mask_torch[:, :, :, n_real_tokens:] = _MASK_NEG
    mask = _from_torch(mask_torch, device, dtype=ttnn.float32)

    # The 29 pad rows of the one-tile sequence, as a PERSISTENT buffer rather than a per-call
    # `ttnn.zeros` (which a trace cannot replay). One buffer per distinct batch.
    pad_rows = _TILE - n_real_tokens
    _pads = {}

    def _pad_for(rows):
        buf = _pads.get(rows)
        if buf is None:
            buf = _from_torch(torch.zeros(rows, 1, pad_rows, dim), device, dtype=ttnn.float32)
            _pads[rows] = buf
        return buf

    if batch is not None:
        _pad_for(int(batch))

    blocks = [_compile_block(device, at.layers[str(i)], mask) for i in at.layers_ids[:layers]]
    g_final = None
    eps_final = float(at.norm.eps)

    acoustic_out = int(at.acoustic_codebook_output.out_features)
    semantic_out = int(at.semantic_codebook_output.out_features)

    def _time_projection(t):
        """`time_projection(time_embedding(t))` as `[1, 1, rows, dim]`, for a `[rows, 1]` timestep.

        Depends on nothing but `t`, so a caller with fixed timesteps can compute it once.
        """
        rows = int(t.shape[0])
        freqs = ttnn.matmul(
            ttnn.typecast(ttnn.reshape(t, [1, 1, rows, int(t.shape[-1])]), ttnn.float32),
            ttnn.typecast(inv_freq, ttnn.float32),
            compute_kernel_config=_COMPUTE,
        )
        t_emb = ttnn.concat([ttnn.cos(freqs), ttnn.sin(freqs)], dim=-1)
        return _lin(t_emb, w_time, compute_kernel_config=_COMPUTE)

    def _compact_forward(llm_hidden, x_t, t, batch, cache=None, t_proj=None):
        # Token t of every sample in rows t*batch.. -- no per-sample 29-row tile pad.
        # `llm_projection(llm_hidden)` and the semantic head read only `llm_hidden`, which is the
        # same at every Euler step of a frame: a caller-owned `cache` computes them once per frame.
        if cache is not None and "llm_proj" in cache:
            semantic, llm_proj = cache["semantic"], cache["llm_proj"]
        else:
            h_in = ttnn.typecast(ttnn.reshape(llm_hidden, [1, 1, batch, dim]), ttnn.float32)
            semantic = _lin(h_in, w_semantic, compute_kernel_config=_COMPUTE)
            if semantic_bias is not None:
                semantic = ttnn.add(semantic, semantic_bias)
            semantic = ttnn.reshape(semantic, [batch, semantic_out])
            llm_proj = _lin(h_in, w_llm, compute_kernel_config=_COMPUTE)
            if cache is not None:
                cache["semantic"], cache["llm_proj"] = semantic, llm_proj

        if t_proj is None:
            t_proj = _time_projection(t)
        x_in = ttnn.typecast(ttnn.reshape(x_t, [1, 1, batch, int(x_t.shape[-1])]), ttnn.float32)
        h = ttnn.concat(
            [
                _lin(x_in, w_input, compute_kernel_config=_COMPUTE),
                t_proj,
                llm_proj,
            ],
            dim=2,
        )
        for i, block in enumerate(blocks):
            h = block(h, tokens=n_real_tokens, readout=i == len(blocks) - 1)
        # The norm is per row and only token 0 is read out, so normalise just those rows.
        first = _rms_norm(ttnn.slice(h, [0, 0, 0, 0], [1, 1, batch, dim]), g_final, eps_final)
        velocity = ttnn.reshape(_lin(first, w_acoustic, compute_kernel_config=_COMPUTE), [batch, acoustic_out])
        return velocity, semantic

    def flow_matching_audio_transformer(llm_hidden, x_t=None, t=None, step_cache=None, t_proj=None, **kwargs):
        batch = int(llm_hidden.shape[0])
        if batch % _TILE == 0:
            return _compact_forward(llm_hidden, x_t, t, batch, step_cache, t_proj)
        h_in = ttnn.reshape(llm_hidden, [batch, 1, 1, dim])

        h_in = ttnn.typecast(h_in, ttnn.float32)
        semantic = _lin(h_in, w_semantic, compute_kernel_config=_COMPUTE)
        if semantic_bias is not None:
            semantic = ttnn.add(semantic, semantic_bias)
        semantic = ttnn.reshape(semantic, [batch, semantic_out])

        # TimeEmbedding: outer product t (x) inv_freq, then cat(cos, sin).
        freqs = ttnn.matmul(
            ttnn.typecast(ttnn.reshape(t, [batch, 1, 1, int(t.shape[-1])]), ttnn.float32),
            ttnn.typecast(inv_freq, ttnn.float32),
            compute_kernel_config=_COMPUTE,
        )
        t_emb = ttnn.concat([ttnn.cos(freqs), ttnn.sin(freqs)], dim=-1)
        t_proj = _lin(t_emb, w_time, compute_kernel_config=_COMPUTE)
        llm_proj = _lin(h_in, w_llm, compute_kernel_config=_COMPUTE)
        x_proj = _lin(
            ttnn.typecast(ttnn.reshape(x_t, [batch, 1, 1, int(x_t.shape[-1])]), ttnn.float32),
            w_input,
            compute_kernel_config=_COMPUTE,
        )

        h = ttnn.concat([x_proj, t_proj, llm_proj, _pad_for(batch)], dim=2)

        for block in blocks:
            h = block(h)
        h = _rms_norm(h, g_final, eps_final)

        first = ttnn.slice(h, [0, 0, 0, 0], [batch, 1, 1, dim])
        velocity = ttnn.reshape(
            _lin(first, w_acoustic, compute_kernel_config=_COMPUTE),
            [batch, acoustic_out],
        )
        return velocity, semantic

    flow_matching_audio_transformer.time_projection = _time_projection
    return flow_matching_audio_transformer
