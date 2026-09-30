# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `bidirectional_attention` (`acoustic_transformer.layers.0.attention`).

The canonical `models/tt_transformers/tt/attention.py` is not usable here: it builds itself from
`ModelArgs`, which resolves the model through `AutoConfig`, and this checkpoint is a native Mistral
`consolidated.safetensors` with no `config.json` / `model_type` -- `ModelArgs` raises before
reading a weight. Its math is also the wrong math (see below). So this is a direct ttnn forward
over the resolved submodule's own weights.

Despite tensor names identical to the text backbone (`wq/wk/wv/wo`), this attention is
**bidirectional and RoPE-free**: no positional encoding and no causal mask, so SDPA runs with
`is_causal=False` and nothing is rotated. The `rope_theta: 10000.0` under
`acoustic_transformer_args` in `params.json` is dead config -- `AcousticTransformerArgs` has no
such field.

GQA 32 query heads over 8 KV heads, head_dim 128, dim 3072, no biases. The reference returns
`wo(...).squeeze(0)`, i.e. `[S, dim]`; a leading-1 reshape is metadata only, so keeping rank
`[1, S, dim]` compares the same elements in the same order.

BATCH AXIS. The leading bound is read from the tensor, never assumed to be 1: a `[B, 1, S, dim]`
input keeps all B samples and comes back at rank 4, while the rank-<=3 input the component test
feeds is unchanged. A hardcoded leading 1 would silently drop samples 1..B-1.

`attn_mask=` is an additive mask applied to the scores (the flow-matching sampler pads its
3-token sequence to one 32-row tile and blocks columns 3..31). The whole attention runs in
float32 against bfloat16 weights at HiFi4 + `fp32_dest_acc_en` -- see `_attention` for why it is
written out instead of calling SDPA.
"""

from __future__ import annotations

import math

import torch

import ttnn

_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# Tall (>= 8 tile rows) linears are compute-bound, so they run one fidelity rung below HiFi4.
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
    cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k,
        out_subblock_h=1,
        out_subblock_w=min(n, 4),
        per_core_M=per_core_m or m,
        per_core_N=n,
    )
    return ttnn.matmul(a, b, transpose_b=transpose_b, program_config=cfg, compute_kernel_config=_COMPUTE, dtype=dtype)


def _leading(shape) -> int:
    """The product of every axis before `[seq, dim]` -- the real batch, from the tensor."""
    batch = 1
    for d in list(shape)[:-2]:
        batch *= int(d)
    return batch


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

    # bf16 context: the head merge moves it and o_proj multicasts it whole.
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


def build(device, torch_module):
    attn = torch_module
    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    head_dim = int(attn.head_dim)
    dim = int(attn.wq.in_features)
    out_dim = int(attn.wo.out_features)
    scale = 1.0 / math.sqrt(head_dim)

    wqkv = _from_torch(
        torch.cat(
            [
                # The query scale folded into Wq: the reference scales the query before the product.
                attn.wq.weight.detach().float().transpose(0, 1) * scale,
                attn.wk.weight.detach().float().transpose(0, 1),
                attn.wv.weight.detach().float().transpose(0, 1),
            ],
            dim=-1,
        ).contiguous(),
        device,
        dtype=ttnn.bfloat16,
    )
    wo = _from_torch(attn.wo.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat16)
    for rows in _COMPACT_ROWS:
        _compact_mask(device, rows, 3, n_heads // n_kv_heads)
        _compact_mask(device, rows, 3, n_heads // n_kv_heads, rows // 3)

    def bidirectional_attention(x, attn_mask=None, tokens=None, readout=False, **kwargs):
        if tokens:
            return _compact_attention(x, wqkv, wo, n_heads, n_kv_heads, None, tokens, readout)
        seq = int(x.shape[-2])
        batch = _leading(x.shape)
        rank = len(list(x.shape))

        # A bf16 input (a norm that already narrowed its output) feeds the qkv matmul as-is; the
        # projection writes float32, so the attention itself stays float32 either way.
        h = ttnn.reshape(x, [batch, 1, seq, dim])
        out = _attention(h, wqkv, wo, n_heads, n_kv_heads, None, attn_mask)
        if rank >= 4:
            return ttnn.reshape(out, [batch, 1, seq, out_dim])
        return ttnn.reshape(out, [batch, seq, out_dim])

    return bidirectional_attention
