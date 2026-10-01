# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `parametrized_conv_transpose1d` (`audio_tokenizer.decoder_blocks.2.conv`).

The bare `weight_norm`-parametrized `ConvTranspose1d` INSIDE `CausalConvTranspose1d`:
1024 -> 1024 channels, kernel 4, stride 2, no bias, and **no trimming** -- the causal wrapper is
what drops the trailing 2 samples, so this produces the full `2L + 2`.

Done as taps plus an interleave. With the sequence channels-last as `[1, 1, L, C]`,
`out[i * 2 + k] += x[i] @ W[:, :, k]`, so for stride 2 the even output positions take taps 0 and 2
and the odd ones take taps 1 and 3, with the tap-2/3 term delayed by one input step:

    even[m] = x[m] @ W0 + x[m-1] @ W2        m = 0 .. L   (x[L] and x[-1] read as zero)
    odd[m]  = x[m] @ W1 + x[m-1] @ W3

The interleave is free: concatenating `even` and `odd` on the CHANNEL axis and reshaping
`[1, 1, L+1, 2C] -> [1, 1, 2L+2, C]` is, in row-major order, exactly
`even[0], odd[0], even[1], odd[1], ...`.
"""

from __future__ import annotations

import torch

import ttnn

# A PERSISTENT ZERO BUFFER, NOT A PER-CALL `ttnn.zeros`.
# `ttnn.zeros` builds its tensor on the host and enqueues a WRITE to land it on the device, and a
# captured trace cannot replay a write (`TT_FATAL: Writes are not supported during trace capture`).
# The shape follows the input shape, which a trace pins, so it is created once per
# (device, shape, dtype) and reused. Read-only, so it is never deallocated by a caller.
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
    conv = torch_module

    weight = conv.weight.detach()
    in_channels, out_channels, kernel = (int(v) for v in weight.shape)
    stride = int(conv.stride[0])
    if stride != 2 or kernel != 4:
        raise NotImplementedError(f"only kernel 4 / stride 2 is ported, got {kernel}/{stride}")
    if int(conv.padding[0]) or int(conv.output_padding[0]) or int(conv.groups) != 1:
        raise NotImplementedError("padding / output_padding / grouped transposed conv is not ported")

    taps = [_from_torch(weight[:, :, i].contiguous(), device) for i in range(kernel)]
    bias = None
    if conv.bias is not None:
        bias = _from_torch(conv.bias.detach().reshape(1, 1, 1, out_channels), device)

    def parametrized_conv_transpose1d(x, **kwargs):
        # `[B, C, L]` in, `[B, C, 2L + 2]` out. The leading bound comes from the TENSOR, never from
        # a literal 1: the pipeline stacks 32 independent samples on axis 0.
        shape = [int(v) for v in x.shape]
        length = shape[-1]
        batch = shape[0] if len(shape) >= 3 else 1
        x4 = ttnn.reshape(ttnn.transpose(x, -2, -1), [batch, 1, length, in_channels])

        zero_row = _zeros_like_buf(device, [batch, 1, 1, out_channels], x4.dtype)

        def _now(tap):
            """`tap` applied to input steps 0..L-1, then a zero row for output step L."""
            return ttnn.concat([_tap_linear(x4, tap, compute_kernel_config=_COMPUTE), zero_row], dim=2)

        def _delayed(tap):
            """`tap` applied to the PREVIOUS input step: a zero row, then steps 0..L-1."""
            return ttnn.concat([zero_row, _tap_linear(x4, tap, compute_kernel_config=_COMPUTE)], dim=2)

        even = ttnn.add(_now(taps[0]), _delayed(taps[2]))
        odd = ttnn.add(_now(taps[1]), _delayed(taps[3]))

        out_len = length * stride + (kernel - stride)
        interleaved = ttnn.reshape(ttnn.concat([even, odd], dim=-1), [batch, 1, out_len, out_channels])
        if bias is not None:
            interleaved = ttnn.add(interleaved, bias)

        return ttnn.reshape(ttnn.transpose(interleaved, -2, -1), [batch, out_channels, out_len])

    return parametrized_conv_transpose1d
