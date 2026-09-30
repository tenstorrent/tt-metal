# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `causal_conv_transpose1d` (`audio_tokenizer.decoder_blocks.2`).

`CausalConvTranspose1d` runs `ConvTranspose1d(1024, 1024, kernel=4, stride=2)` -- no bias, weight
from the `weight_norm` reconstruction -- and then trims `ceil((kernel - stride) * trim_ratio) = 2`
samples off the RIGHT (`trim_ratio=1.0`, so nothing comes off the left).

Written as taps + an interleave rather than a transposed convolution kernel. With the sequence
channels-last as `[1, 1, L, C]`, `out[i * 2 + k] += x[i] @ W[:, :, k]`, so for stride 2 the even
output positions take taps 0 and 2 and the odd ones take taps 1 and 3, each with the tap-2/3 term
delayed by one input step:

    even[m] = x[m] @ W0 + x[m-1] @ W2
    odd[m]  = x[m] @ W1 + x[m-1] @ W3

The two trimmed samples are exactly `even[L]` and `odd[L]`, so those are never computed. The
interleave is free: concatenating `even` and `odd` on the CHANNEL axis and reshaping
`[1, 1, L, 2C] -> [1, 1, 2L, C]` is, in row-major order, precisely `even[0], odd[0], even[1], ...`.
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


# fp32 accumulation in DEST. The codec's activation path is float32 -- bfloat16 end to end put the
# full chain at PCC 0.9873 over eight residual blocks plus five convolutions -- and the default
# configuration would accumulate a float32 matmul in a narrower DEST.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


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
    in_channels, out_channels, kernel = (int(v) for v in weight.shape)
    stride = int(conv.stride[0])
    if stride != 2 or kernel != 4:
        raise NotImplementedError(f"only kernel 4 / stride 2 is ported, got {kernel}/{stride}")

    total_padding = kernel - stride
    right_trim = math.ceil(total_padding * float(blk.trim_ratio))
    left_trim = total_padding - right_trim
    if left_trim != 0:
        raise NotImplementedError(f"a non-zero left trim ({left_trim}) is not ported")

    taps = [_from_torch(weight[:, :, i].contiguous(), device) for i in range(kernel)]
    bias = None
    if conv.bias is not None:
        bias = _from_torch(conv.bias.detach().reshape(1, 1, 1, out_channels), device)

    def causal_conv_transpose1d(x, **kwargs):
        # `[B, C, L]` in, `[B, C, 2L]` out. The leading bound comes from the TENSOR, never from a
        # literal 1: the pipeline stacks 32 independent samples on axis 0.
        shape = [int(v) for v in x.shape]
        length = shape[-1]
        batch = shape[0] if len(shape) >= 3 else 1
        x4 = ttnn.reshape(ttnn.transpose(x, -2, -1), [batch, 1, length, in_channels])

        zero_row = _zeros_like_buf(device, [batch, 1, 1, out_channels], x4.dtype)

        def _delayed(tap):
            """`tap` applied to the PREVIOUS input step: a zero row, then rows 0..L-2."""
            z = _tap_linear(
                ttnn.slice(x4, [0, 0, 0, 0], [batch, 1, length - 1, in_channels]),
                tap,
                compute_kernel_config=_COMPUTE,
            )
            return ttnn.concat([zero_row, z], dim=2)

        even = ttnn.add(_tap_linear(x4, taps[0], compute_kernel_config=_COMPUTE), _delayed(taps[2]))
        odd = ttnn.add(_tap_linear(x4, taps[1], compute_kernel_config=_COMPUTE), _delayed(taps[3]))

        interleaved = ttnn.reshape(ttnn.concat([even, odd], dim=-1), [batch, 1, length * stride, out_channels])
        if bias is not None:
            interleaved = ttnn.add(interleaved, bias)

        return ttnn.reshape(ttnn.transpose(interleaved, -2, -1), [batch, out_channels, length * stride])

    return causal_conv_transpose1d
