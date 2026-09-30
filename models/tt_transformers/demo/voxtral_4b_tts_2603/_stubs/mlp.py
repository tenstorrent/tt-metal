# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mlp` -- the text backbone's `MistralMLP` (`model.layers.0.mlp`).

SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`, 3072 -> 9216 -> 3072, no biases.

The canonical `models/tt_transformers/tt/mlp.py` is not reusable here: it builds itself from
`ModelArgs`, which resolves the model through `AutoConfig`, and this checkpoint is a native Mistral
`consolidated.safetensors` with no `config.json` / `model_type` -- `ModelArgs` raises before
reading a weight.

The leading bound comes from the tensor (`_view4`), so a batched `[B, 1, S, D]` stream keeps all B
rows; `down_proj` packs back to the dtype the caller handed in, which is what lets a float32
residual stream feed bfloat16 weights without a widening cast on the way out."""

from __future__ import annotations

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import ttl_swiglu

# `ttnn.linear`/`ttnn.matmul` on their DEFAULTS leave `fp32_dest_acc_en` off, so the accumulator
# rounds to bfloat16 at every step even when the activations are float32. The consumer of this
# stack resolves a top-1/top-2 margin of a few hundredths, and the audio path rounds onto 21
# levels 0.1 apart, so that rounding decides real codes. Every matmul below passes this.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


_SHARD_HEIGHT = 32

# SDPA takes bfloat16 and nothing wider (`sdpa_device_operation.cpp:43`), and the KV cache is read
# by the same op family, so q/k/v and the cache are bf16 while the residual stream stays float32.
_SDPA_DTYPE = ttnn.bfloat16
# The resident KV cache: every decode step streams all of it through both attention bmms.
_CACHE_DTYPE = ttnn.bfloat8_b


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


def _weight(linear, device):
    """A `[in, out]` device tensor for a torch `nn.Linear` (whose weight is `[out, in]`)."""
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def _norm_weight(norm, device):
    """Gamma in the `[1, 1, dim // 32, 32]` ROW_MAJOR form `ttnn.rms_norm` requires."""
    return _from_torch(norm.weight.detach().reshape(1, 1, -1, _SHARD_HEIGHT), device, layout=ttnn.ROW_MAJOR_LAYOUT)


def _view4(x, dim):
    """`[..., seq, dim]` -> `([lead, 1, seq, dim], lead, seq, rank)`.

    THE LEADING BOUND IS READ OFF THE TENSOR. This used to be a literal
    `ttnn.reshape(x, [1, 1, seq, dim])`, which is right at the batch of 1 the per-component PCC
    harness feeds and wrong for every batched caller: at B=32 the reshape either raises on volume
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


def _swiglu_pairs(gate, up, tile=32):
    """`[K, N]` gate and up weights as ONE `[K, 2N]` weight of interleaved column-tile pairs
    `[gate_t0, up_t0, gate_t1, up_t1, ...]` -- the layout `minimal_matmul(fuse_swiglu=True)` reads
    to emit `silu(gate) * up` straight from the matmul."""
    rows, n = int(gate.shape[0]), int(gate.shape[-1])
    pairs = torch.stack([gate.reshape(rows, n // tile, tile), up.reshape(rows, n // tile, tile)], dim=2)
    return pairs.reshape(rows, 2 * n).contiguous()


def _ttl_swiglu_weights(gate, up, device):
    if not ttl_swiglu.enabled():
        return None
    return tuple(_from_torch(w.contiguous(), device, dtype=ttnn.bfloat16) for w in (gate, up))


def _fused_swiglu(h, w_gu, w_ttl=None):
    """Prefill `silu(h @ Wg) * (h @ Wu)` as ONE matmul: no gate/up tensors are written and there
    is no separate multiply pass over them."""
    if w_ttl is not None and ttl_swiglu.supports(h, w_ttl[0]):
        return ttl_swiglu.apply(h, *w_ttl)
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
    # A share of <= 8 tiles fits one M block, so each core streams its weight columns ONCE.
    m_blk = share if share <= 8 else -(-share // 2)
    # At <= 2 M tiles a core the bf4_b weight's blocks are small enough for 8-tile K blocks
    # (half the K steps of 4) within the ~1 MB of L1 that still traces.
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
    mlp = torch_module
    dim = int(mlp.gate_proj.in_features)
    out_dim = int(mlp.down_proj.out_features)

    # Decode-only now (tall bf16 prefill runs the fused `w_gu` below): bf8_b halves the bytes a
    # one-token step streams.
    w_gate, w_up = (
        _from_torch(p.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
        for p in (mlp.gate_proj, mlp.up_proj)
    )
    # bf8_b halves the weight both the prefill (LoFi) and decode down projections unpack.
    w_down = _from_torch(mlp.down_proj.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    # Prefill's fused SwiGLU weight, bf4_b against the bf16 norm output; the float32 decode
    # activation keeps the separate pair above.
    w_gu = _from_torch(
        _swiglu_pairs(
            mlp.gate_proj.weight.detach().float().transpose(0, 1),
            mlp.up_proj.weight.detach().float().transpose(0, 1),
        ),
        device,
        dtype=ttnn.bfloat4_b,
    )
    w_ttl = _ttl_swiglu_weights(
        mlp.gate_proj.weight.detach().float().transpose(0, 1),
        mlp.up_proj.weight.detach().float().transpose(0, 1),
        device,
    )

    def mlp_forward(x, **kwargs):
        h, lead, seq, rank = _view4(x, dim)
        if h.dtype == ttnn.bfloat16 and lead * seq >= 256:
            gated = _fused_swiglu(h, w_gu, w_ttl)
        else:
            gated = ttnn.multiply(
                _lin(h, w_gate, dtype=ttnn.bfloat16, compute_kernel_config=_COMPUTE),
                _lin(h, w_up, dtype=ttnn.bfloat16, compute_kernel_config=_COMPUTE),
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )
        out = _lin(gated, w_down, dtype=x.dtype, compute_kernel_config=_COMPUTE)
        ttnn.deallocate(gated)
        return _restore(out, lead, seq, rank, out_dim)

    return mlp_forward
