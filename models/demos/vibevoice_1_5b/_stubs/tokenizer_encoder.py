# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `tokenizer_encoder` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_tokenizer.encoder`, a
`vibevoice.modular.modular_vibevoice_tokenizer.TokenizerEncoder`:

    for i in range(len(depths)):
        x = downsample_layers[i][0](x)      # SConv1d, causal; stride==1 for the
                                             # stem (i==0), stride==ratios[i-1]>1
                                             # for every later downsample layer
        for block in stages[i]:
            x = block(x)                    # Block1D
    x = norm(x)                             # Identity for this checkpoint
    x = head(x)                             # SConv1d, stride==1

The stem and head are stride-1 groups-1 causal `SConv1d`s, ported by reusing
the already-graduated `s_conv1d` build (drop-in per-instance forward). The
strided downsample layers need a stride-aware variant (`s_conv1d` only
supports stride==1) implemented here as the same shifted-tap matmul
accumulate, but with a strided time-axis slice per tap
(`ttnn.slice(..., slice_step=[1, stride, 1])`) instead of a unit-stride one,
plus the extra right-padding `SConv1d._forward_non_streaming` adds so the
padded length divides evenly into `stride`-sized frames
(`get_extra_padding_for_conv1d`).
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs._trace_pad import cached_zeros
from models.demos.vibevoice_1_5b._stubs.block1_d import build as _build_block1d
from models.demos.vibevoice_1_5b._stubs.s_conv1d import build as _build_s_conv1d

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_TILE = ttnn.TILE_LAYOUT


def _extra_padding_for_conv1d(length, kernel_size, stride, padding_total):
    import math

    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def _build_strided_s_conv1d(device, sconv):
    """Native port of a causal, groups=1, stride>1 `SConv1d` instance."""
    conv = sconv.conv.conv  # nn.Conv1d
    w = conv.weight.detach().float()  # [C_out, C_in, K]
    c_out, c_in, k = w.shape
    if conv.groups != 1:
        raise RuntimeError(f"tokenizer_encoder strided-conv port supports groups=1 only (got groups={conv.groups})")
    if type(sconv.conv.norm).__name__ != "Identity":
        raise RuntimeError(
            f"tokenizer_encoder strided-conv port assumes norm='none' (Identity), got {type(sconv.conv.norm).__name__}"
        )
    if not bool(sconv.causal):
        raise RuntimeError("tokenizer_encoder strided-conv port assumes a causal conv")
    stride = int(conv.stride[0])
    padding_total = int(sconv.padding_total)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    taps = [
        ttnn.from_torch(w[:, :, tap].t().contiguous(), dtype=ttnn.float32, layout=_TILE, device=device)
        for tap in range(k)
    ]
    bias = None
    if conv.bias is not None:
        bias = ttnn.from_torch(
            conv.bias.detach().reshape(1, 1, c_out).contiguous().float(),
            dtype=ttnn.float32,
            layout=_TILE,
            device=device,
        )
    _zc: dict = {}  # cached causal zero-pad buffers (trace-capture safe)

    def forward(x, *args, **kwargs):
        # x: [1, C_in, T] channels-first -> [1, T, C_in]
        x_tc = ttnn.transpose(x, 1, 2)
        if x_tc.get_dtype() != ttnn.float32:
            x_tc = ttnn.typecast(x_tc, ttnn.float32)

        T_in = int(x_tc.shape[1])
        extra = _extra_padding_for_conv1d(T_in, k, stride, padding_total)
        pad_left, pad_right = padding_total, extra

        pieces = []
        if pad_left > 0:
            pieces.append(cached_zeros(_zc, (1, pad_left, c_in), ttnn.float32, _TILE, device))
        pieces.append(x_tc)
        if pad_right > 0:
            pieces.append(cached_zeros(_zc, (1, pad_right, c_in), ttnn.float32, _TILE, device))
        x_pad = ttnn.concat(pieces, dim=1, memory_config=_DRAM) if (pad_left or pad_right) else x_tc

        Lp = int(x_pad.shape[1])
        t_out = (Lp - k) // stride + 1
        y = None
        for tap in range(k):
            last = tap + stride * (t_out - 1) + 1
            xs = ttnn.slice(x_pad, [0, tap, 0], [1, last, c_in], slice_step=[1, stride, 1])
            term = ttnn.matmul(xs, taps[tap], compute_kernel_config=compute_config, memory_config=_DRAM)
            y = term if y is None else ttnn.add(y, term, memory_config=_DRAM)
        if bias is not None:
            y = ttnn.add(y, bias, memory_config=_DRAM)
        return ttnn.transpose(y, 1, 2)  # [1, C_out, T_out] channels-first

    return forward


def build(device, torch_module):
    """Bind every child submodule's trained weights and return a native ttnn forward closure."""
    m = torch_module
    depths = list(m.depths)

    downsample_forwards = []
    for layer_seq in m.downsample_layers:
        layer = layer_seq[0]
        stride = int(layer.conv.conv.stride[0])
        if stride == 1:
            downsample_forwards.append(_build_s_conv1d(device, layer))
        else:
            downsample_forwards.append(_build_strided_s_conv1d(device, layer))

    stage_forwards = [[_build_block1d(device, blk) for blk in stage] for stage in m.stages]

    if type(m.norm).__name__ != "Identity":
        raise RuntimeError(f"tokenizer_encoder native port assumes norm='none' (Identity), got {type(m.norm).__name__}")

    head_forward = _build_s_conv1d(device, m.head)

    def forward(x, *args, **kwargs):
        for i in range(len(depths)):
            x = downsample_forwards[i](x)
            for blk_fwd in stage_forwards[i]:
                x = blk_fwd(x)
        x = head_forward(x)
        return x

    return forward


def tokenizer_encoder(*args, **kwargs):
    raise RuntimeError(
        "tokenizer_encoder requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
