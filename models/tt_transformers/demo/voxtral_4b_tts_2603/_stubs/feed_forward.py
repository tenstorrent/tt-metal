# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `feed_forward` (`acoustic_transformer.layers.0.feed_forward`).

SwiGLU: `w2(silu(w1(x)) * w3(x))`, dim 3072 -> hidden 9216 -> 3072. Only `w2` can carry a bias
(`use_biases`, false in this checkpoint), so it is applied conditionally rather than assumed away.

Note the naming: this module's `w1` is the GATE and `w3` is the up-projection, the opposite of the
`{gate,up}_proj` ordering the checkpoint's key map implies -- `feed_forward.w1/w2/w3` maps to
`mlp.{gate,down,up}_proj`.

BATCH AXIS. The leading bound is read from the tensor: a `[B, 1, S, dim]` input keeps all B
samples and comes back at rank 4, while the rank-<=3 input the component test feeds is unchanged.
HiFi4 + `fp32_dest_acc_en` on a float32 activation against bfloat16 weights, because the
flow-matching sampler this feeds rounds its output onto 21 levels.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import cpp_down, cpp_swiglu, ttl_down

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


def _short_cfg(x, w, rows, out_dtype):
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
            for c in (32, 24, 16, 12, 8, 6, 4, 3, 2, 1)
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


def _weight(linear, device):
    """A `[in, out]` device tensor for a torch `nn.Linear` (whose weight is `[out, in]`)."""
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def build(device, torch_module):
    ff = torch_module
    dim = int(ff.w1.in_features)
    out_dim = int(ff.w2.out_features)

    w1 = _from_torch(ff.w1.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat16)
    # The down projection is DRAM-bound at 1024 rows; bf8_b halves the weight it streams.
    w2 = _from_torch(ff.w2.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat16)
    w2_ttl = ttl_down.weight(ff.w2.weight.detach().transpose(0, 1).contiguous(), device, _from_torch)
    w2_cpp = cpp_down.shard(ff.w2.weight.detach().transpose(0, 1).contiguous(), device)
    w3 = _from_torch(ff.w3.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat16)
    w13 = cpp_swiglu.fuse(ff.w1.weight.detach().transpose(0, 1), ff.w3.weight.detach().transpose(0, 1), device)
    bias = None
    if ff.w2.bias is not None:
        bias = _from_torch(ff.w2.bias.detach().reshape(1, 1, 1, out_dim), device)

    def feed_forward(x, **kwargs):
        seq = int(x.shape[-2])
        batch = _leading(x.shape)
        rank = len(list(x.shape))

        h = ttnn.reshape(x, [batch, 1, seq, dim])
        # HiFi2 on the bf8_b weights (two phases cover a bf8_b mantissa); the hidden activation is
        # handed over in L1 and in bf16, since the down projection multicasts it whole.
        gated = (
            cpp_swiglu.apply(h, w13)
            if cpp_swiglu.serves(h, w13)
            else ttnn.multiply(
                _lin(
                    h, w1, dtype=ttnn.float32, compute_kernel_config=_TALL_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG
                ),
                _lin(
                    h, w3, dtype=ttnn.float32, compute_kernel_config=_TALL_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG
                ),
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.float32,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
        if cpp_down.serves(gated, w2_cpp):
            out = cpp_down.apply(gated, w2_cpp)
        elif ttl_down.supports(gated, w2_ttl):
            out = ttl_down.apply(gated, w2_ttl)
        else:
            out = _lin(gated, w2, dtype=ttnn.float32, compute_kernel_config=_TALL_COMPUTE)
        if bias is not None:
            out = ttnn.add(out, bias)
        if rank >= 4:
            return ttnn.reshape(out, [batch, 1, seq, out_dim])
        return ttnn.reshape(out, [batch, seq, out_dim])

    return feed_forward
