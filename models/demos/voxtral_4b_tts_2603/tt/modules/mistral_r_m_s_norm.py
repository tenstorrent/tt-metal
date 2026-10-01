# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mistral_r_m_s_norm` (`model.layers.0.input_layernorm`).

`x * rsqrt(mean(x^2) + eps) * weight`, over dim 3072 with `eps = 1e-5`. `MistralRMSNorm` does the
reduction in float32 and casts back, so the activation is widened here to match rather than
normalising in bfloat16.

SPELLED OUT IN FOUR OPS, NOT `ttnn.rms_norm`. Measured against the reference on the real layer-0
input, the stock op lands at 9.65e-4 relative error and the four ops below at 6.6e-8 -- four
orders of magnitude apart. It matters because the stack runs 52 of these and the error is
RELATIVE, so it rescales the whole branch output that follows: at 9.65e-4 apiece the final hidden
state came back at PCC 0.9971, and a 21-level acoustic quantiser (code edges 0.1 apart in x, with
a fifth of all values landing within 0.01 of one) turned that into ~26% wrong audio codes. Every
other term is already smaller -- the float32-activation x bfloat16-weight matmul sits at a 4.9e-4
hardware floor per linear, and the bfloat16 Q/K/V cast SDPA forces is 5.7e-4 end to end.
`mistral_model.py::_rms_norm` is the same four ops, for the same reason.

Tile padding on an off-tile sequence is safe: a padded row is all zeros, so `mean(x^2)` is 0 and
`0 * rsqrt(eps)` stays 0 -- no NaN, and nothing leaks into a real row.

Gamma therefore goes up as `[1, 1, 1, dim]` float32 TILE, the form the final multiply broadcasts
against, rather than the `[1, 1, dim // 32, 32]` ROW_MAJOR that only existed to satisfy
`ttnn.rms_norm`'s `gamma.padded_shape[-1] == TILE_WIDTH` assert
(`layernorm_device_operation.cpp:106`).

THE LEADING BOUND IS READ OFF THE TENSOR. This was `ttnn.reshape(hidden_states, [1, 1, seq, dim])`
-- correct at the batch of 1 the per-component harness feeds, and wrong for every batched caller:
at B=32 that reshape either raises on volume or, once a leading 1 is folded in elsewhere, keeps
row 0 and silently drops samples 1..31. An RMS norm reduces over the LAST dim only, so collapsing
every leading axis into one bound is exact for `[B, S, D]`, `[B, 1, S, D]` and the decode stream's
`[1, 1, B, D]` alike, and the output goes back in the rank it arrived in.
"""

from __future__ import annotations

import torch

import ttnn


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


def build(device, torch_module):
    norm = torch_module
    dim = int(norm.weight.shape[-1])
    eps = float(norm.variance_epsilon)
    gamma = _from_torch(norm.weight.detach().reshape(1, 1, 1, dim), device, dtype=ttnn.float32)
    # A unit gamma (one the caller folded into the consuming weights) costs a full pass for nothing.
    unit = bool(torch.all(norm.weight.detach() == 1))

    def mistral_r_m_s_norm(hidden_states, dtype=None, **kwargs):
        """`dtype` narrows only the OUTPUT (e.g. bf16 for a norm that feeds a matmul); the
        statistics and the scaling are float32 either way."""
        shape = [int(s) for s in hidden_states.shape]
        seq = shape[-2]
        lead = 1
        for size in shape[:-2]:
            lead *= size
        x = ttnn.reshape(hidden_states, [lead, 1, seq, dim])
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        scale = ttnn.add(_sq_mean(x), eps, activations=[ttnn.UnaryOpType.RSQRT])
        if unit:
            out = ttnn.multiply(x, scale, dtype=dtype or ttnn.float32)
        else:
            out = ttnn.multiply(ttnn.multiply(x, scale), gamma, dtype=dtype or ttnn.float32)
        return ttnn.reshape(out, [lead, seq, dim] if len(shape) == 3 else [lead, 1, seq, dim])

    return mistral_r_m_s_norm
