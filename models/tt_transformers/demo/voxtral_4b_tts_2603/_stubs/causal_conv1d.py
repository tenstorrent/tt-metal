# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `causal_conv1d` (`audio_tokenizer.decoder_blocks.0`).

`CausalConv1d` left-pads by `effective_kernel_size - stride` (plus whatever trailing padding is
needed to make the frame count come out whole) and then runs a plain `Conv1d` with no padding of
its own. Here: 292 -> 1024 channels, kernel 3, stride 1, `pad_mode="replicate"`, no bias, and the
weight is the `weight_norm` reconstruction (reading `conv.weight` in `build` runs the
parametrization on the host, which is where weight prep belongs).

The convolution itself is done as **shifted matmuls**: with the sequence laid out channels-last as
`[1, 1, L, C]`, output position `t` is `sum_k x[t + k] @ W_k`, so each tap is one `[C_in, C_out]`
matmul over a row-shifted slice of the padded input. That keeps the whole forward in ttnn matmul /
slice / concat -- no conv-specific sharding config to get wrong -- and the row-shifted slices are
exact even at non-tile-aligned offsets.
"""

from __future__ import annotations

import math

import torch

import ttnn

# A PERSISTENT ZERO BUFFER, NOT A PER-CALL `ttnn.zeros`.
# `ttnn.zeros` builds its tensor on the host and enqueues a WRITE to land it on the device, and a
# captured trace cannot replay a write -- capturing this stage died on `TT_FATAL: Writes are not
# supported during trace capture`. The shape is a function of the input shape, which a trace pins,
# so the buffer is created once per (device, shape, dtype) and reused. It is READ-ONLY here and is
# therefore never deallocated by a caller.
_ZEROS = {}


def _zeros_like_buf(device, shape, dtype):
    key = (id(device), tuple(int(s) for s in shape), str(dtype))
    buf = _ZEROS.get(key)
    if buf is None:
        buf = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        _ZEROS[key] = buf
    return buf


# FLOAT32 WEIGHTS. Measured on this device against a float64 reference, one matmul at M=32,
# K=N=3072, HiFi4 + `fp32_dest_acc_en`: a bfloat16 weight costs 1.738e-3 relative where a float32
# weight costs 1.169e-3. That 1.5x is small per op and this codec stacks eight residual blocks on
# top of five convolutions, where it is the last error source left after the softmax and the RMS
# norm were spelled out. Nothing here is an `ttnn.embedding` table (which would have to stay
# bfloat16); those live in the codebook stubs.
def _from_torch(t, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
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


# fp32 accumulation in DEST. The codec's activation path is float32 -- bfloat16 end to end put the
# full chain at PCC 0.9873 over eight residual blocks plus five convolutions -- and the default
# configuration would accumulate a float32 matmul in a narrower DEST.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _edge_pad(x4, rows, *, at_start):
    """`rows` copies of the first (or last) sequence row of a `[B, 1, L, C]` tensor.

    This is the `pad_mode="replicate"` edge: the padded value is the boundary sample itself. The
    leading bound is read off the tensor -- every sample in the batch gets its own edge.
    """
    batch, length, channels = int(x4.shape[0]), int(x4.shape[-2]), int(x4.shape[-1])
    row_start = 0 if at_start else length - 1
    row = ttnn.slice(x4, [0, 0, row_start, 0], [batch, 1, row_start + 1, channels])
    return row if rows == 1 else ttnn.repeat(row, [1, 1, rows, 1])


def _tap_linear(x, w, **kwargs):
    """`ttnn.linear` for one tap with the leading batch folded into M, so the tap streams ONCE.

    A `[B, 1, L, C]` slice against a 2-D tap runs as B separate `L x C x C_out` matmuls that each
    re-read the whole tap; `[1, 1, B*L, C]` is one matmul. An L that is not tile-aligned makes the
    fold a real relayout each way, still far cheaper than re-reading the tap B times.
    """
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    y = ttnn.linear(ttnn.reshape(x, [1, 1, lead * shape[-2], shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


def build(device, torch_module):
    blk = torch_module
    conv = blk.conv

    weight = conv.weight.detach()
    out_channels, in_channels, kernel = (int(v) for v in weight.shape)
    stride = int(conv.stride[0])
    dilation = int(conv.dilation[0])
    effective_kernel = int(blk._effective_kernel_size)
    padding_total = int(blk._padding_total)
    pad_mode = str(blk.pad_mode)
    if pad_mode not in ("replicate", "constant", "zeros"):
        raise NotImplementedError(f"causal_conv1d pad_mode {pad_mode!r} is not ported")
    replicate = pad_mode == "replicate"

    taps = [_from_torch(weight[:, :, i].transpose(0, 1).contiguous(), device) for i in range(kernel)]
    bias = None
    if conv.bias is not None:
        bias = _from_torch(conv.bias.detach().reshape(1, 1, 1, out_channels), device)

    def causal_conv1d(x, **kwargs):
        # `[B, C, L]` in, `[B, C_out, L']` out. The leading bound comes from the TENSOR, never from
        # a literal 1: the pipeline stacks 32 independent samples on axis 0.
        shape = [int(v) for v in x.shape]
        length = shape[-1]
        batch = shape[0] if len(shape) >= 3 else 1
        x4 = ttnn.reshape(ttnn.transpose(x, -2, -1), [batch, 1, length, in_channels])

        n_frames = (length - effective_kernel + padding_total) / stride + 1
        target = (math.ceil(n_frames) - 1) * stride + (effective_kernel - padding_total)
        extra = target - length

        pieces = []
        if padding_total > 0:
            pieces.append(
                _edge_pad(x4, padding_total, at_start=True)
                if replicate
                else _zeros_like_buf(device, [batch, 1, padding_total, in_channels], x4.dtype)
            )
        pieces.append(x4)
        if extra > 0:
            pieces.append(
                _edge_pad(x4, extra, at_start=False)
                if replicate
                else _zeros_like_buf(device, [batch, 1, extra, in_channels], x4.dtype)
            )
        padded = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2)

        padded_len = length + padding_total + extra
        out_len = (padded_len - effective_kernel) // stride + 1

        acc = None
        for i, tap in enumerate(taps):
            begin = i * dilation
            end = begin + (out_len - 1) * stride + 1
            seg = ttnn.slice(
                padded,
                [0, 0, begin, 0],
                [batch, 1, end, in_channels],
                [1, 1, stride, 1] if stride > 1 else None,
            )
            term = _tap_linear(seg, tap, compute_kernel_config=_COMPUTE)
            acc = term if acc is None else ttnn.add(acc, term)
        if bias is not None:
            acc = ttnn.add(acc, bias)

        return ttnn.reshape(ttnn.transpose(acc, -2, -1), [batch, out_channels, out_len])

    return causal_conv1d
