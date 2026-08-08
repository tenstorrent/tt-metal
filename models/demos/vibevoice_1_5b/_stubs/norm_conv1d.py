# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `norm_conv1d` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv`,
a `vibevoice.modular.modular_vibevoice_tokenizer.NormConv1d` wrapping
`nn.Conv1d(1, 32, kernel_size=7, stride=1, groups=1)` with `norm='none'`
(Identity) -- the tokenizer stem's first conv, no padding (padding is applied
by the enclosing `SConv1d`, not here).

Implemented as the same matmul shifted tap-accumulate recipe as
`kokoro_82m/_stubs/parametrized_conv1d.py` (stride-1, groups-1 Conv1d):

    y[:, t, :] = sum_tap  x[:, t + tap, :] @ W[:, :, tap]^T   (+ bias)
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained conv weight/bias and return a native ttnn forward closure."""
    conv = torch_module.conv  # nn.Conv1d
    w = conv.weight.detach().float()  # [C_out, C_in, K]
    c_out, c_in, k = w.shape
    if int(conv.stride[0]) != 1 or conv.groups != 1:
        raise RuntimeError(
            f"norm_conv1d native port supports stride-1 groups-1 only (got stride={conv.stride[0]}, groups={conv.groups})"
        )
    if int(conv.padding[0]) != 0:
        raise RuntimeError("norm_conv1d native port assumes no padding")

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    taps = [
        ttnn.from_torch(w[:, :, tap].t().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        for tap in range(k)
    ]
    bias = None
    if conv.bias is not None:
        bias = ttnn.from_torch(
            conv.bias.detach().reshape(1, 1, c_out).contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def forward(x, *args, **kwargs):
        # x: [1, C_in, T] channels-first -> [1, T, C_in]
        x_tc = ttnn.transpose(x, 1, 2)
        if x_tc.get_dtype() != ttnn.float32:
            x_tc = ttnn.typecast(x_tc, ttnn.float32)
        T = int(x_tc.shape[1])
        t_out = T - (k - 1)
        y = None
        for tap in range(k):
            xs = ttnn.slice(x_tc, [0, tap, 0], [1, tap + t_out, c_in])
            yt = ttnn.matmul(xs, taps[tap], compute_kernel_config=compute_config, memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        if bias is not None:
            y = ttnn.add(y, bias, memory_config=_DRAM)
        # [1, T, C_out] -> [1, C_out, T] channels-first
        return ttnn.transpose(y, 1, 2)

    return forward


def norm_conv1d(*args, **kwargs):
    raise RuntimeError(
        "norm_conv1d requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
