# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `norm_conv_transpose1d` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_tokenizer.decoder.upsample_layers.1.0.convtr`,
a `vibevoice.modular.modular_vibevoice_tokenizer.NormConvTranspose1d` wrapping
`nn.ConvTranspose1d(2048, 1024, kernel_size=16, stride=8, padding=0,
output_padding=0, dilation=1, groups=1)` with `norm='none'` (Identity).

A stride-`s` groups=1 ConvTranspose1d is equivalent to a stride-1 forward
Conv1d over the input "dilated" by inserting `s-1` zeros between consecutive
time steps and zero-padded by `K-1` on both sides, using the *transpose*
weight `W[C_in, C_out, K]` directly (no channel-swap/no explicit flip needed,
since indexing the K-1-tap already accounts for the flip):

    x_dilated = insert (stride-1) zeros between time steps of x   # length (T-1)*stride + 1
    x_pad = zero-pad x_dilated by (K-1) on both sides
    y[:, t, :] = sum_tap  x_pad[:, t + tap, :] @ W[:, :, K-1-tap]   (+ bias)

Verified: this reproduces `T_out = (T_in-1)*stride + K` (the padding=0,
output_padding=0, dilation=1 case of the standard ConvTranspose1d formula).
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs._trace_pad import cached_zeros

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_ROW_MAJOR = ttnn.ROW_MAJOR_LAYOUT
_TILE = ttnn.TILE_LAYOUT


def build(device, torch_module):
    """Bind the trained conv-transpose weight/bias and return a native ttnn forward closure."""
    convtr = torch_module.convtr  # nn.ConvTranspose1d
    w = convtr.weight.detach().float()  # [C_in, C_out, K]
    c_in, c_out, k = w.shape
    stride = int(convtr.stride[0])
    if convtr.groups != 1 or int(convtr.dilation[0]) != 1:
        raise RuntimeError(
            f"norm_conv_transpose1d native port supports groups=1, dilation=1 only (got groups={convtr.groups}, dilation={convtr.dilation[0]})"
        )
    if int(convtr.padding[0]) != 0 or int(convtr.output_padding[0]) != 0:
        raise RuntimeError("norm_conv_transpose1d native port assumes padding=0, output_padding=0")

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # taps[tap'] = W[:, :, K-1-tap'] used directly as the [C_in, C_out] matmul
    # matrix (no transpose: `W` is already [C_in, C_out, K]).
    taps = [
        ttnn.from_torch(w[:, :, k - 1 - tap].contiguous(), dtype=ttnn.float32, layout=_TILE, device=device)
        for tap in range(k)
    ]
    bias = None
    if convtr.bias is not None:
        bias = ttnn.from_torch(
            convtr.bias.detach().reshape(1, 1, c_out).contiguous().float(),
            dtype=ttnn.float32,
            layout=_TILE,
            device=device,
        )

    _zc: dict = {}  # cached zero blocks (dilation + causal pad); trace-capture safe

    def _dilate_time(x_tc_rm, T_in, C):
        # x_tc_rm: [1, T_in, C] row-major -> interleave (stride-1) zero rows
        # after every time step, then drop the trailing (stride-1) zeros so
        # the result has length (T_in - 1) * stride + 1.
        if stride == 1:
            return x_tc_rm
        x_exp = ttnn.reshape(x_tc_rm, (1, T_in, 1, C))
        zeros_block = cached_zeros(_zc, (1, T_in, stride - 1, C), ttnn.float32, _ROW_MAJOR, device)
        interleaved = ttnn.concat([x_exp, zeros_block], dim=2, memory_config=_DRAM)  # [1, T_in, stride, C]
        flat = ttnn.reshape(interleaved, (1, T_in * stride, C))
        return ttnn.slice(flat, [0, 0, 0], [1, (T_in - 1) * stride + 1, C])

    def _pad_time_rm(x_tc_rm, pad_l, pad_r, C):
        if pad_l == 0 and pad_r == 0:
            return x_tc_rm
        pieces = []
        if pad_l:
            pieces.append(cached_zeros(_zc, (1, pad_l, C), ttnn.float32, _ROW_MAJOR, device))
        pieces.append(x_tc_rm)
        if pad_r:
            pieces.append(cached_zeros(_zc, (1, pad_r, C), ttnn.float32, _ROW_MAJOR, device))
        return ttnn.concat(pieces, dim=1, memory_config=_DRAM)

    def forward(x, *args, **kwargs):
        # x: [1, C_in, T_in] channels-first -> [1, T_in, C_in]
        x_tc = ttnn.transpose(x, 1, 2)
        if x_tc.get_dtype() != ttnn.float32:
            x_tc = ttnn.typecast(x_tc, ttnn.float32)
        T_in = int(x_tc.shape[1])

        x_tc_rm = ttnn.to_layout(x_tc, _ROW_MAJOR)
        x_dilated = _dilate_time(x_tc_rm, T_in, c_in)
        x_pad = _pad_time_rm(x_dilated, k - 1, k - 1, c_in)
        x_pad = ttnn.to_layout(x_pad, _TILE)

        Lp = int(x_pad.shape[1])
        t_out = Lp - (k - 1)
        y = None
        for tap in range(k):
            xs = ttnn.slice(x_pad, [0, tap, 0], [1, tap + t_out, c_in])
            term = ttnn.matmul(xs, taps[tap], compute_kernel_config=compute_config, memory_config=_DRAM)
            y = term if y is None else ttnn.add(y, term, memory_config=_DRAM)
        if bias is not None:
            y = ttnn.add(y, bias, memory_config=_DRAM)

        return ttnn.transpose(y, 1, 2)  # [1, C_out, T_out] channels-first

    return forward


def norm_conv_transpose1d(*args, **kwargs):
    raise RuntimeError(
        "norm_conv_transpose1d requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
