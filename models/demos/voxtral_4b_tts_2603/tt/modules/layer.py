# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `layer` -- one text-backbone `MistralDecoderLayer`
(`model.layers.0`).

    h = x + self_attn(input_layernorm(x));   out = h + mlp(post_attention_layernorm(h))

GQA 32 query heads over 8 KV heads, head_dim **128** (explicitly 128, not 3072/32=96), hidden_dim
9216, `norm_eps` 1e-5, no biases. RoPE is `rotate_half` against the `(cos, sin)` the caller passes;
the text stack stages them on the device, so the forward makes no torch call and can be captured in a
trace.

Causality comes from SDPA's `is_causal`, so the additive `attention_mask` argument is accepted and
ignored. That matches the reference for an unpadded batch; a mask carrying real PADDING would need
to be fed through as an `attn_mask` instead.

TWO phases, one set of weights. With no `kv_cache` this is a plain causal prefill. Given one it ALSO
seeds it from its own post-RoPE k/v, and `decode=True` then runs the cached single-token path, which
reads the resident history instead of recomputing it."""

from __future__ import annotations

import torch

import ttnn

# SDPA takes bfloat16 and nothing wider (`sdpa_device_operation.cpp:43`), and the KV cache is read
# by the same op family, so q/k/v and the cache are bf16 while the residual stream stays float32.
_SDPA_DTYPE = ttnn.bfloat16
# The resident KV cache: every decode step streams all of it through both attention bmms.
_CACHE_DTYPE = ttnn.bfloat8_b

# `ttnn.linear` on its DEFAULTS leaves `fp32_dest_acc_en` off, which rounds the matmul
# accumulator to bfloat16 at every step even though the activations are float32. The decode linears
# pass this instead (the tall prefill linears take `_TALL_COMPUTE` below).
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# Tall (prefill) linears are compute-bound, so they run at LoFi rather than the HiFi4 the
# rest of this file uses; the one-token decode linears are weight-bandwidth-bound and keep HiFi4.
_TALL_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
)
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_L1_BUDGET = 1_100_000


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _mcast_cfg(x, w, rows, out_dtype):
    """A full-grid 2D-multicast program config for a tall `[rows, K] x [K, N]` linear, or None.

    M goes over the grid rows and N over the grid columns. Per-core M/N are searched a few tiles
    above the minimum (a slightly larger block often divides into better subblocks), and when the
    whole per-core output does not fit L1 it is split into out-blocks. Ranked by per-core work,
    then the tiles each core re-reads across out-blocks, then a K-block of at least 4, then subblock
    area (16-bit DEST allows 8 tiles).
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = rows // 32, int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    xs, ws = size(x.dtype), size(w.dtype)
    # 16-bit DEST: no separate float32 accumulation buffer beside the output block.
    os_ = size(out_dtype)
    best = None
    for pm in range(-(-mt // gy), -(-mt // gy) + 5):
        if -(-mt // pm) > gy:
            continue
        for pn in range(-(-nt // gx), -(-nt // gx) + 5):
            if -(-nt // pn) > gx:
                continue
            for bh in _divisors(pm):
                for bw in _divisors(pn):
                    kb = next(
                        (
                            c
                            for c in (8, 4, 2, 1)
                            if kt % c == 0 and bh * bw * os_ + 2 * c * (bh * xs + bw * ws) <= _L1_BUDGET
                        ),
                        None,
                    )
                    if kb is None:
                        continue
                    sub = max(
                        (
                            (h, s)
                            for h in range(1, 9)
                            for s in range(1, 9)
                            if h * s <= 8 and bh % h == 0 and bw % s == 0
                        ),
                        key=lambda hs: (hs[0] * hs[1], hs[1]),
                    )
                    reads = kt * (pm * (pn // bw) + pn * (pm // bh))
                    score = (pm * pn, reads, -min(kb, 4), -sub[0] * sub[1], -kb)
                    if best is None or score < best[0]:
                        best = (score, pm, pn, bh, bw, kb, sub)
    if best is None:
        return None
    _, pm, pn, bh, bw, kb, sub = best
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        out_block_h=bh,
        out_block_w=bw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
    )


def _row_cfg(x, w, out_dtype):
    """A 1D in0-multicast config for a ONE-tile-row (decode) linear, or None.

    Every core owns a slice of N and streams only its own weight columns while the single
    activation row is multicast; the widest K block that fits L1 keeps the multicast rounds few.
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    kt, nt = int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    per_n = next(p for p in range(-(-nt // (gx * gy)), nt + 1) if nt % p == 0)
    if per_n > 2:  # wide N (gate/up): ttnn's own choice measured faster
        return None
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    fixed = per_n * (size(out_dtype) + (0 if out_dtype == ttnn.float32 else 4096))
    kb = next(
        (
            c
            for c in (32, 24, 16, 12, 8, 6, 4, 3, 2, 1)
            if kt % c == 0 and fixed + 2 * c * (size(x.dtype) + per_n * size(w.dtype)) <= _L1_BUDGET
        ),
        None,
    )
    if kb is None:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=1,
        out_subblock_w=max(s for s in range(1, 5) if per_n % s == 0),
        per_core_M=1,
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
    elif rows == 32 and "program_config" not in kwargs:
        cfg = _row_cfg(x, w, kwargs.get("dtype") or x.dtype)
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


_STATS_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
)


def _sq_mean(x):
    """`mean(x^2, -1)` of a float32 `x`. A tall (prefill) `x` gets its row sums of squares from
    `rms_norm_pre_all_gather` -- ONE read of `x`, float32 accumulation and a float32 `[rows, 32]`
    result whose column 0 is the sum -- instead of writing `x^2` out and reading it back to reduce."""
    shape = [int(d) for d in x.shape]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    if rows < 256:
        return ttnn.mean(ttnn.square(x), dim=-1, keepdim=True)
    stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=_STATS_COMPUTE, dtype=ttnn.float32)
    total = ttnn.slice(stats, [0] * len(shape), shape[:-1] + [1])
    return ttnn.multiply(total, 1.0 / shape[-1])


def _rms_norm(x, gamma, eps, dtype=None, memory_config=None):
    """`x * rsqrt(mean(x^2) + eps) * gamma`, spelled out, entirely in float32.

    NOT `ttnn.rms_norm`. On the real layer-0 input the stock op lands at 9.65e-4 relative error
    against the reference and these four ops at 6.6e-8. The error is RELATIVE, so it rescales the
    whole branch that follows, and this stack runs two per layer over 26 layers; left in, the
    prefill's last hidden state came back at PCC 0.9971 and the 21-level acoustic quantiser
    downstream turned that into ~26% wrong audio codes.

    A zero pad row stays zero: `mean(x^2)` is 0 and `0 * rsqrt(eps)` is 0, so an off-tile
    sequence padded up to a tile multiple neither NaNs nor leaks into a real row.
    """
    scale = ttnn.add(_sq_mean(x), eps, activations=[ttnn.UnaryOpType.RSQRT])
    if gamma is None:  # folded into the consuming weights
        return ttnn.multiply(x, scale, dtype=dtype or ttnn.float32, memory_config=memory_config)
    return ttnn.multiply(ttnn.multiply(x, scale), gamma, dtype=dtype or ttnn.float32, memory_config=memory_config)


def _view4(x, dim):
    """`[..., seq, dim]` -> `([lead, 1, seq, dim], lead, seq, rank)`.

    THE LEADING BOUND IS READ OFF THE TENSOR. This used to be a literal
    `ttnn.reshape(x, [1, 1, seq, dim])`, which is right at a batch of 1 and wrong for every batched
    caller: at B=32 the reshape either raises on volume
    or -- worse, once a leading 1 is folded in elsewhere -- keeps only row 0 and silently drops
    samples 1..31. Everything downstream of here is per-row, so collapsing every leading axis into
    one `lead` is exact for `[B, S, D]`, `[B, 1, S, D]` and the decode stream's `[1, 1, B, D]`.
    """
    shape = [int(s) for s in x.shape]
    seq = shape[-2]
    lead = 1
    for size in shape[:-2]:
        lead *= size
    return ttnn.reshape(x, [lead, 1, seq, dim]), lead, seq, len(shape)


def _restore(x, lead, seq, rank, dim):
    """Put a `[lead, 1, seq, dim]` result back into the RANK the caller handed in."""
    return ttnn.reshape(x, [lead, seq, dim] if rank == 3 else [lead, 1, seq, dim])


def _broadcast4(t, seq, width):
    """A `(cos, sin)` table as `[lead, 1, seq, width]`, its leading bound read off the tensor."""
    volume = 1
    for size in t.shape:
        volume *= int(size)
    return ttnn.reshape(t, [volume // (seq * width), 1, seq, width])


def _rope(x, cos, sin, half):
    """`x * cos + rotate_half(x) * sin` -- the convention `apply_rotary_pos_emb` uses.

    `rotate_half` is `cat(-x[..., half:], x[..., :half])`; both halves are multiples of the tile
    width, so the two slices are tile-aligned.
    """
    ends = list(x.shape)
    lower = ttnn.slice(x, [0, 0, 0, 0], [ends[0], ends[1], ends[2], half])
    upper = ttnn.slice(x, [0, 0, 0, half], ends)
    rotated = ttnn.concat([ttnn.neg(upper), lower], dim=-1)
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))


def _rope_signed(x, cos, sin_signed, half):
    """`_rope` with the rotate-half sign already on the table: `x * cos + cat(x2, x1) * sin_signed`,
    where `sin_signed = cat(-sin1, sin2)` -- no negation of half of `x`."""
    ends = list(x.shape)
    lower = ttnn.slice(x, [0, 0, 0, 0], [ends[0], ends[1], ends[2], half])
    upper = ttnn.slice(x, [0, 0, 0, half], ends)
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(ttnn.concat([upper, lower], dim=-1), sin_signed))


def _rope_prefill(q, k, cos, sin, half):
    """Prefill RoPE on q and k as ONE fused kernel each when the table is shared by the batch.

    `ttnn.experimental.rotary_embedding` computes the same `x * cos + rotate_half(x) * sin` in a
    single pass, where `_rope` spends six ops (two slices, a neg, a concat, two multiplies and an
    add) and a DRAM round-trip per op over `[B, H, S, head_dim]`. It wants a `[1, 1, S, head_dim]`
    table in the input's dtype; a per-row table (explicit position ids) keeps the spelled-out path.
    """
    if int(cos.shape[0]) != 1 or q.dtype != ttnn.bfloat16 or k.dtype != ttnn.bfloat16:
        return _rope(q, cos, sin, half), _rope(k, cos, sin, half)
    if cos.dtype != ttnn.bfloat16:
        cos, sin = ttnn.typecast(cos, ttnn.bfloat16), ttnn.typecast(sin, ttnn.bfloat16)
    return (
        ttnn.experimental.rotary_embedding(q, cos, sin),
        ttnn.experimental.rotary_embedding(k, cos, sin),
    )


def _bmm(a, b, transpose_b=False):
    """Head-batched decode attention `a @ b` spread over the full grid.

    Without a program config the `[B, n_kv, groups, C]` products land on a handful of cores (probs @ V
    on four) and run the B * n_kv small matmuls back to back. The reuse config makes every
    (batch, kv-head) output block its own work unit, so they fan out across the grid.
    """
    # Ceil, not floor: the grouped query is `[B, n_kv, groups, head_dim]` with groups=4 rows, one padded tile.
    m, k, n = (-(-int(d) // 32) for d in (a.shape[-2], a.shape[-1], b.shape[-2 if transpose_b else -1]))
    grid = a.device().compute_with_storage_grid_size()
    cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k,
        out_subblock_h=1,
        out_subblock_w=max(s for s in range(1, 5) if n % s == 0),
        per_core_M=m,
        per_core_N=n,
    )
    return ttnn.matmul(a, b, transpose_b=transpose_b, program_config=cfg, compute_kernel_config=_COMPUTE)


def _sdpa_cfg(q):
    """Prefill SDPA on the full grid with the widest q/k chunk (<= 128) that divides the sequence.

    With no program config SDPA takes small default chunks, so each (batch, head) row is many
    tiny work units that re-stream K/V per chunk.
    """
    grid = q.device().compute_with_storage_grid_size()
    seq = int(q.shape[-2])
    chunk = next(c for c in (128, 64, 32) if seq % c == 0)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
    )


def _per_sample(t, batch, real, padded):
    """A compact `[1, H, batch * real, D]` k/v as the cache's per-sample `[batch, H, padded, D]`.

    Row `b * real + i` is sample b's i-th tail position; the rows past `real` are zeros, which
    the decode slot mask keeps closed until a step writes them.
    """
    heads, width = int(t.shape[1]), int(t.shape[-1])
    rows = ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), [heads, batch, real, width])
    rows = ttnn.permute(rows, (1, 0, 2, 3))
    if padded != real:
        rows = ttnn.pad(rows, [(0, 0), (0, 0), (0, padded - real), (0, 0)], 0.0)
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)


def _prefill_sdpa(q, k, v, kv_cache):
    """Prefill SDPA, `(attn, k, v)` -- the k/v to seed the cache with, or None to seed nothing.

    With no `kv_cache["prefix_phase"]` this is the plain causal SDPA. A text stack running a SHARED
    prompt prefix once calls every layer twice: "stash" (the batch-1 prefix) keeps its k/v in the
    cache dict and seeds nothing, and "extend" (the per-row tail at the full batch) puts that k/v in
    front of its own for every row and attends through `kv_cache["prefix_mask"]` -- prefix columns
    open, the tail's own columns causal -- so the cache is seeded with the whole prompt.
    """
    phase = kv_cache.get("prefix_phase") if kv_cache is not None else None
    compact = kv_cache.get("prefix_compact") if kv_cache is not None else None
    if phase == "extend" and compact is not None:
        # COMPACT tail: `[1, H, batch * real, D]`, only the positions the rows disagree on. All
        # of it is one sequence behind the batch-1 prefix k/v, and the staged mask keeps each row
        # to the shared prefix plus its own sample's earlier tail positions.
        batch, real, padded = compact
        pk, pv = kv_cache.pop("prefix_kv")
        grid = q.device().compute_with_storage_grid_size()
        a = ttnn.transformer.scaled_dot_product_attention(
            q,
            ttnn.concat([pk, k], dim=2),
            ttnn.concat([pv, v], dim=2),
            is_causal=False,
            attn_mask=kv_cache["prefix_mask"],
            scale=1.0,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                exp_approx_mode=False,
                q_chunk_size=next(c for c in (128, 64, 32) if int(q.shape[-2]) % c == 0),
                k_chunk_size=32,
            ),
        )
        rep = ttnn.Shape([batch, 1, 1, 1])
        k = ttnn.concat([ttnn.repeat(pk, rep), _per_sample(k, batch, real, padded)], dim=2)
        v = ttnn.concat([ttnn.repeat(pv, rep), _per_sample(v, batch, real, padded)], dim=2)
        ttnn.deallocate(pk)
        ttnn.deallocate(pv)
        return a, k, v
    if phase == "extend":
        pk, pv = kv_cache.pop("prefix_kv")
        rep = ttnn.Shape([int(q.shape[0]), 1, 1, 1])
        k = ttnn.concat([ttnn.repeat(pk, rep), k], dim=2)
        v = ttnn.concat([ttnn.repeat(pv, rep), v], dim=2)
        ttnn.deallocate(pk)
        ttnn.deallocate(pv)
        a = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, attn_mask=kv_cache["prefix_mask"], scale=1.0, program_config=_sdpa_cfg(q)
        )
        return a, k, v
    a = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0, program_config=_sdpa_cfg(q))
    if phase == "stash":
        kv_cache["prefix_kv"] = (k, v)
        return a, None, None
    return a, k, v


def _decode_shard(device, rows, width):
    """HEIGHT-sharded over the batch, one user per core -- the decode op set's layout.

    `nlp_create_qkv_heads_decode`, decode-mode `rotary_embedding_hf` and
    `nlp_concat_heads_decode` are a matched set: each wants one 32-row tile per user, and the RoPE
    op rejects a merely-interleaved tensor outright, so this is part of the contract.
    """
    grid = device.compute_with_storage_grid_size()
    cols = min(int(grid.x), int(rows))
    while rows % cols:
        cols -= 1
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, int(width)),
        core_grid=ttnn.CoreGrid(y=rows // cols, x=cols),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# THE ZERO TAIL IS A PERSISTENT BUFFER, NOT A PER-CALL `ttnn.zeros`.
# `ttnn.zeros` builds the tensor on the host and enqueues a WRITE to get it onto the device, and a
# write is exactly what a captured trace cannot replay: capturing a prefill that seeded its cache
# this way died on `TT_FATAL: Writes are not supported during trace capture`. The tail is the same
# shape of the same zeros on every call, so it is created once per (device, shape) and reused --
# and because the pad is now shared it is NEVER deallocated by the caller, which used to free it
# after the concat. `flow_matching_audio_transformer` hoists its tile pad for the same reason.
_ZERO_TAIL = {}


def _zero_tail(device, b, h, rows, width):
    key = (id(device), b, h, rows, width)
    buf = _ZERO_TAIL.get(key)
    if buf is None:
        buf = ttnn.zeros([b, h, rows, width], dtype=_CACHE_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        _ZERO_TAIL[key] = buf
    return buf


def _seed_cache(kv, k, v):
    """Hand the prefill's POST-RoPE k/v to the cache, widened out to `kv["capacity"]`.

    No copy and no second source of truth: the cache IS the prefill's own k/v with a zero tail
    concatenated on the sequence axis, so the resident history cannot disagree with the prefill
    that produced it. The tail slots are never READ before they are written -- a decode step
    writes slot `position` and then attends to `[0, position]` -- so they only have to exist.
    """
    capacity = int(kv.get("capacity") or 0)
    for key, tensor in (("k", k), ("v", v)):
        if tensor.dtype != _CACHE_DTYPE:
            tensor = ttnn.typecast(tensor, _CACHE_DTYPE)
        shape = [int(s) for s in tensor.shape]
        if capacity > shape[-2]:
            pad = _zero_tail(tensor.device(), shape[0], shape[1], capacity - shape[-2], shape[-1])
            tensor = ttnn.concat([tensor, pad], dim=2)
        elif capacity and capacity < shape[-2]:
            raise ValueError(f"kv capacity {capacity} is shorter than the prefill's {shape[-2]}")
        stale = kv.get(key)
        if stale is not None:
            try:
                ttnn.deallocate(stale)
            except Exception:  # noqa: BLE001 - an already-freed buffer is fine to skip
                pass
        kv[key] = tensor
    kv["filled"] = int(k.shape[-2])


# The additive mask for one decode position, `[1, 1, 1, C]`: 0 through `position`, a large
# negative beyond it. The row comes off the `[C, C]` float32 table the text stack uploaded ONCE at
# build time and handed to every block in its `kv` dict -- built here it would be a torch call
# inside the forward, and a host write inside a captured trace.


def _decode_mask(kv_cache, position, cap):
    table = kv_cache.get("mask")
    if table is None:
        raise RuntimeError(
            "the decode step needs kv_cache['mask'] -- the [capacity, capacity] additive table "
            "the text stack stages at build time; without it the zero tail of the cache is "
            "attended as if it were real keys"
        )
    shared = kv_cache.get("mask_row")
    if shared is not None and shared[0] == int(position) and int(shared[1].shape[-1]) == cap:
        return shared[1]
    row = ttnn.slice(table, [int(position), 0], [int(position) + 1, cap])
    return ttnn.to_layout(ttnn.reshape(row, [1, 1, 1, cap]), ttnn.TILE_LAYOUT)


def _swiglu_pairs(gate, up, tile=32):
    """`[K, N]` gate and up weights as ONE `[K, 2N]` weight of interleaved column-tile pairs
    `[gate_t0, up_t0, gate_t1, up_t1, ...]` -- the layout `minimal_matmul(fuse_swiglu=True)` reads
    to emit `silu(gate) * up` straight from the matmul."""
    rows, n = int(gate.shape[0]), int(gate.shape[-1])
    pairs = torch.stack([gate.reshape(rows, n // tile, tile), up.reshape(rows, n // tile, tile)], dim=2)
    return pairs.reshape(rows, 2 * n).contiguous()


def _fused_swiglu(h, w_gu):
    """Prefill `silu(h @ Wg) * (h @ Wu)` as ONE matmul: no gate/up tensors are written and there
    is no separate multiply pass over them."""
    grid = h.device().compute_with_storage_grid_size()
    rows = 1
    for d in list(h.shape)[:-1]:
        rows *= int(d)
    # Half of each core's M share per block (vs the default 8 rows) halves the weight re-reads and
    # 18-tile N blocks split each core's 54 fused columns (27 output tiles x 2) into exactly 3 blocks
    # (12 left a half-empty 5th block); 4-tile K blocks keep the L1 footprint near 1 MB.
    share = -(-(rows // 32) // int(grid.y))
    # The fewest grid rows that keep that share: rows past ceil(M / share) would only compute padding
    # (32 M tiles over 10 rows is 4 each, and rows 8-9 held the pad).
    grid = ttnn.CoreCoord(int(grid.x), -(-(rows // 32) // share))
    # A share of <= 8 tiles fits one M block, so each core streams its weight columns ONCE. Longer shares
    # split into the fewest equal blocks of <= 8 tiles: each M tile of a block costs ~124 KB of L1
    # circular buffer, so an unbounded block (10 tiles at M = 181 tiles, a ~180-token tail at batch 32)
    # overflows L1. The split changes only how often the weight is re-read, not the arithmetic.
    m_blk = -(-share // -(-share // 8))
    # At <= 2 M tiles a core the weight's blocks are small enough for 8-tile K blocks (half the K
    # steps of 4): 8 x 18 bf8_b tiles double-buffered is ~313 KB, within the ~1 MB of L1 that traces.
    wide = m_blk <= 2
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=m_blk,
        K_block_size=8 if wide else 4,
        N_block_size=18,
        subblock_h=1,
        subblock_w=6,
        compute_with_storage_grid_size=grid,
    )
    return ttnn.experimental.minimal_matmul(
        h, w_gu, fuse_swiglu=True, config=cfg, dtype=ttnn.bfloat16, compute_kernel_config=_TALL_COMPUTE
    )


def build(device, torch_module):
    layer = torch_module
    attn = layer.self_attn
    mlp = layer.mlp

    dim = int(attn.q_proj.in_features)
    n_heads = int(attn.config.num_attention_heads)
    n_kv_heads = int(attn.config.num_key_value_heads)
    head_dim = int(attn.head_dim)
    half = head_dim // 2
    scale = float(attn.scaling)

    # Each RMSNorm's gamma is FOLDED into the input rows of the weights it feeds (`x*s*g @ W` ==
    # `x*s @ diag(g) W`), so the norm is one scaling pass over the residual, not two. The product
    # is formed in float32 and rounded to bfloat16 once.
    g_in_t = layer.input_layernorm.weight.detach().float().reshape(-1, 1)
    g_post_t = layer.post_attention_layernorm.weight.detach().float().reshape(-1, 1)

    def _folded(linear, g):
        return _from_torch((linear.weight.detach().float().transpose(0, 1) * g).contiguous(), device)

    wqkv = _from_torch(
        torch.cat(
            [
                # The attention scale folded into Wq (RoPE is linear, so it commutes).
                attn.q_proj.weight.detach().float().transpose(0, 1) * float(attn.scaling),
                attn.k_proj.weight.detach().float().transpose(0, 1),
                attn.v_proj.weight.detach().float().transpose(0, 1),
            ],
            dim=-1,
        )
        .mul(g_in_t)
        .contiguous(),
        device,
        dtype=ttnn.bfloat8_b,
    )
    wo = _from_torch(attn.o_proj.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    # Decode-only now (prefill runs the fused `w_gu` below): a one-token step streams these from
    # DRAM every step, so bf8_b halves what it reads.
    w_gate, w_up = (
        _from_torch((p.weight.detach().float().transpose(0, 1) * g_post_t).contiguous(), device, dtype=ttnn.bfloat8_b)
        for p in (mlp.gate_proj, mlp.up_proj)
    )
    # Prefill's fused SwiGLU weight, bf8_b against the bf16 norm output (minimal_matmul takes mixed
    # dtypes); decode keeps the separate gate/up above for its float32 activation. Not bf4_b: 4-bit
    # gate/up weights cost the prefill hidden ~0.01 PCC (ar_male 0.980 vs 0.991 at bf8_b).
    w_gu = _from_torch(
        _swiglu_pairs(
            mlp.gate_proj.weight.detach().float().transpose(0, 1) * g_post_t,
            mlp.up_proj.weight.detach().float().transpose(0, 1) * g_post_t,
        ),
        device,
        dtype=ttnn.bfloat8_b,
    )
    # bf8_b halves the weight both the prefill (LoFi) and decode down projections unpack.
    w_down = _from_torch(mlp.down_proj.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    g_in = None
    g_post = None
    eps_in = float(layer.input_layernorm.variance_epsilon)
    eps_post = float(layer.post_attention_layernorm.variance_epsilon)

    def _decode_attn(xn, position_embeddings, kv_cache, position):
        """ONE token per user, attending to the RESIDENT cache instead of recomputing the prefix.

        The stream arrives folded as `[1, 1, B, dim]`: `[B, 1, dim]` pads its middle dim out to a
        whole tile, so every op would touch 32x the data the step carries.
        """
        # THE USER COUNT IS THE VOLUME OVER THE WIDTH, not any single leading dim. A decode step
        # reaches here as `[1, 1, B, dim]` (the folded stream) or as `[B, 1, 1, dim]`, and reading
        # `shape[-2]` returns B for the first and 1 for the second -- which would quietly process
        # one user and drop the other 31.
        held = [int(size) for size in xn.shape]
        batch = 1
        for size in held[:-1]:
            batch *= size
        # FLOAT32 ALL THE WAY THROUGH. `_SDPA_DTYPE` is bfloat16 because SDPA takes nothing wider,
        # and this path no longer calls SDPA -- so the narrowing bought nothing and cost a bfloat16
        # ulp on every q, k and v. The three ops that forced it are gone with it:
        # `nlp_create_qkv_heads_decode` (its head split is three slices and a reshape, and it
        # hands back a HEIGHT-SHARDED tensor this path would only have to interleave again) and
        # decode-mode `rotary_embedding_hf` (which typecasts its input to bfloat16 outright --
        # `models/tt_transformers/tt/attention.py:664`). `_rope` is the SAME rotate-half the
        # prefill branch below runs, in float32, and every user in a step shares one position so
        # `cos`/`sin` are `[1, 1, 1, head_dim]` and broadcast.
        groups = n_heads // n_kv_heads
        flat = ttnn.reshape(xn, [1, 1, batch, dim])
        fused = _lin(flat, wqkv, dtype=ttnn.float32, compute_kernel_config=_COMPUTE)
        q_width, kv_width = n_heads * head_dim, n_kv_heads * head_dim
        rows = ttnn.to_layout(fused, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(fused)

        def _head_split(start, width, heads_per_kv):
            part = ttnn.slice(rows, [0, 0, 0, start], [1, 1, batch, start + width])
            return ttnn.to_layout(ttnn.reshape(part, [batch, n_kv_heads, heads_per_kv, head_dim]), ttnn.TILE_LAYOUT)

        # The grouped query is `groups` rows of a 32-row tile. Zero-padding it to a LOGICAL full tile
        # costs nothing physically and lets every softmax reduction below skip its FillPad pass.
        q_rows = -(-groups // 32) * 32
        # q is RoPE'd with the heads as ROWS, `[B, 1, n_heads, head_dim]` (one tile per user), and
        # only then regrouped by kv head: the grouped `[B, n_kv, groups, head_dim]` form pads each
        # group of `groups` rows out to a 32-row tile, 8x the tiles for the six RoPE ops.
        q = ttnn.to_layout(
            ttnn.reshape(ttnn.slice(rows, [0, 0, 0, 0], [1, 1, batch, q_width]), [batch, 1, n_heads, head_dim]),
            ttnn.TILE_LAYOUT,
        )

        # k and v go straight into the cache's own `[1, B, n_kv, head_dim]` layout: the n_kv heads
        # share ONE tile per user there, where `[B, n_kv, 1, head_dim]` pads every head to its own
        # 32-row tile and k's RoPE would run on 32x the data.
        def _cache_split(start):
            part = ttnn.slice(rows, [0, 0, 0, start], [1, 1, batch, start + kv_width])
            return ttnn.to_layout(ttnn.reshape(part, [1, batch, n_kv_heads, head_dim]), ttnn.TILE_LAYOUT)

        k = _cache_split(q_width)
        v = _cache_split(q_width + kv_width)
        ttnn.deallocate(rows)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            signed = kv_cache.get("rope_signed") if kv_cache is not None else None
            if signed is not None and signed[0] == int(position):
                q = _rope_signed(q, cos, signed[1], half)
                k = _rope_signed(k, cos, signed[1], half)
            else:
                q = _rope(q, cos, sin, half)
                k = _rope(k, cos, sin, half)
        q = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT), [batch, n_kv_heads, groups, head_dim]),
            ttnn.TILE_LAYOUT,
        )
        if q_rows != groups:
            q = ttnn.pad(q, [(0, 0), (0, 0), (0, q_rows - groups), (0, 0)], 0.0)
        # A split prefill leaves dead slots before the tail: position p lives at slot p + offset.
        idxs = [int(position) + int(kv_cache.get("slot_offset", 0))] * batch
        # `paged_update_cache` wants the decode layout `[1, B, n_kv, head_dim]` AND it wants that
        # tensor HEIGHT-SHARDED, one user per core -- it is part of the decode op set even though
        # the rest of that set is gone from this path ("Expect input_tensor to be sharded"). The
        # k/v split above already produced that layout.
        for slot, tensor in (("k", k), ("v", v)):
            ttnn.experimental.paged_update_cache(
                kv_cache[slot],
                ttnn.to_memory_config(
                    tensor,
                    _decode_shard(tensor.device(), batch, head_dim),
                ),
                update_idxs=idxs,
            )
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # FLASH-DECODE ATTENUATES, so this path spells the attention out instead.
        # `scaled_dot_product_attention_decode` came back with its output norm SHORT of the
        # reference's at every layer -- measured against torch on this model, one decode step, the
        # reference fed the same cache: norm ratio 0.9478 / 0.9781 / 0.9888 / 0.9867 / 0.9895 /
        # 0.9942 at PCC 0.9999, i.e. almost pure attenuation, worst where the softmax is flattest.
        # It is a denominator that carries mass the numerator does not. That is invisible wherever
        # the residual is large, and this model puts its first three layers at hidden norms of
        # 1.1 / 2.6 / 4.0 before layer 3 jumps to 287, so the whole error lands there: the cached
        # decode step measured PCC 0.93 against the reference while THIS STACK'S OWN PREFILL path,
        # same weights and same token, measured 0.9999.
        #
        # The query heads are GROUPED BY KV HEAD rather than repeat_interleaved: reading q as
        # `[B, n_kv, groups, head_dim]` makes the batch dims line up with the cache's
        # `[B, n_kv, C, head_dim]`, so the whole thing is two batched matmuls and no cache
        # tensor is ever materialised `n_heads` times. `head // groups` IS the reference's
        # `repeat_kv` mapping, so the grouping is the same one HF uses.
        cap = int(kv_cache["k"].shape[-2])
        # The bmm reads the cache transposed in place; an explicit transpose re-wrote the whole
        # [B, n_kv, C, head_dim] K cache every step.
        # Scale the [B, n_kv, 32, head_dim] query, not the [B, n_kv, 32, C] scores: one pass over
        # a tensor C/head_dim times smaller.
        scores = _bmm(q, kv_cache["k"], transpose_b=True)
        ttnn.deallocate(q)
        # The cache tail beyond `position` is zeros, and a zero key scores ZERO -- which is a
        # perfectly ordinary logit, not a small one. It has to be masked explicitly.
        scores = ttnn.add(scores, _decode_mask(kv_cache, position, cap))
        # The softmax is spelled out, not `ttnn.softmax`: measured on this build against float64, the
        # stock op's rows do not sum to 1 (mean 0.9943, worst 0.9611), which attenuates the output.
        # Normalise AFTER P@V: dividing the [B, n_kv, 32, head_dim] context by the row sums is the
        # same arithmetic as dividing the C/head_dim-times larger [B, n_kv, 32, C] weights first.
        e = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True), activations=[ttnn.UnaryOpType.EXP])
        ttnn.deallocate(scores)
        ctx = ttnn.divide(_bmm(e, kv_cache["v"]), ttnn.sum(e, dim=-1, keepdim=True))
        ttnn.deallocate(e)
        merged = ttnn.to_layout(
            ttnn.reshape(
                ttnn.slice(
                    ttnn.to_layout(ctx, ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0, 0], [batch, n_kv_heads, groups, head_dim]
                ),
                [1, 1, batch, n_heads * head_dim],
            ),
            ttnn.TILE_LAYOUT,
        )
        ttnn.deallocate(ctx)
        out = _lin(
            ttnn.reshape(merged, [1, 1, batch, n_heads * head_dim]),
            wo,
            dtype=xn.dtype,
            compute_kernel_config=_COMPUTE,
        )
        ttnn.deallocate(merged)
        # Back in the caller's own shape, so the residual add downstream lines up whichever form
        # of the one-token stream came in.
        return ttnn.reshape(out, held[:-1] + [dim])

    def _prefill_attn(xn, position_embeddings, kv_cache, seq):
        qkv = _lin(xn, wqkv, dtype=_SDPA_DTYPE, compute_kernel_config=_COMPUTE)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=n_heads, num_kv_heads=n_kv_heads, transpose_k_heads=False
        )
        ttnn.deallocate(qkv)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = _broadcast4(cos, seq, head_dim)
            sin = _broadcast4(sin, seq, head_dim)
            q, k = _rope_prefill(q, k, cos, sin, half)
        a, k, v = _prefill_sdpa(q, k, v, kv_cache)
        if kv_cache is not None and k is not None:
            _seed_cache(kv_cache, k, v)
        return _lin(
            ttnn.experimental.nlp_concat_heads(a),
            wo,
            dtype=xn.dtype,
            compute_kernel_config=_COMPUTE,
        )

    def layer_forward(
        hidden_states,
        position_embeddings=None,
        kv_cache=None,
        position=None,
        decode=False,
        trim=None,
        **kwargs,
    ):
        h, lead, seq, rank = _view4(hidden_states, dim)

        xn = _rms_norm(h, g_in, eps_in, dtype=None if decode else ttnn.bfloat16)
        if decode:
            attn_out = _decode_attn(xn, position_embeddings, kv_cache, position)
        else:
            attn_out = _prefill_attn(xn, position_embeddings, kv_cache, seq)
        ttnn.deallocate(xn)
        h = ttnn.add(h, attn_out)
        ttnn.deallocate(attn_out)
        if trim is not None:
            # The stack reads only some of this block's rows (its last block): `trim` keeps those as
            # `[1, 1, rows, dim]` (or None when none are read), and the FFN runs on them as a decode
            # step's would.
            kept = trim(h)
            ttnn.deallocate(h)
            if kept is None:
                return None
            h, decode = kept, True

        # Prefill's norm output is read only by the fused SwiGLU, once per N block: it lands in L1.
        hn = _rms_norm(
            h,
            g_post,
            eps_post,
            dtype=None if decode else ttnn.bfloat16,
            memory_config=None if decode else ttnn.L1_MEMORY_CONFIG,
        )
        if decode:
            gated = ttnn.multiply(
                _lin(hn, w_gate, dtype=ttnn.bfloat16, compute_kernel_config=_COMPUTE),
                _lin(hn, w_up, dtype=ttnn.bfloat16, compute_kernel_config=_COMPUTE),
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )
        else:
            gated = _fused_swiglu(hn, w_gu)
        ttnn.deallocate(hn)
        # Prefill hands the down projection over in bf16, as every wo already does; the residual it is
        # added into stays float32.
        down_dtype = h.dtype if decode else ttnn.bfloat16
        h = ttnn.add(h, _lin(gated, w_down, dtype=down_dtype, compute_kernel_config=_COMPUTE))
        ttnn.deallocate(gated)

        return h if trim is not None else _restore(h, lead, seq, rank, dim)

    return layer_forward
