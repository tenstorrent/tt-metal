# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Time-packed 1-D ops for the vocoder's narrow late bands.

With 8 or 16 channels a ``(T, C)`` fp32 row is a 32- or 64-byte DRAM page and every op in the band runs at a few
GB/s. Packing ``k`` consecutive time steps into one row, ``(T, C) -> (T / k, k * C)`` (a plain row-major reshape),
gives 128-byte pages and ``k``x fewer of them. Every op in an AMP block is shift-invariant with an integer rate, so
each has an exact packed form as a dense conv over packed rows:

* a dilated ``Conv1d`` becomes a ``Conv1d`` on ``k * C_in -> k * C_out`` with a few taps;
* the 2x anti-alias upsampler becomes ``k * C -> 2k * C``, the 2x downsampler ``2k * C -> k * C``;
* ``SnakeBeta`` keeps working elementwise with its per-channel parameters tiled ``k`` times.

The packed weights are read off the reference op's impulse responses (``packed_weight``), so any shift-invariant
torch op can be packed without deriving index arithmetic by hand. The dense form multiplies zeros where the
original coupling is absent, but conv3d already pads the channel dim to 32, so at ``k * C = 32`` the FLOPs are
unchanged while the channel pad/trim ops disappear. Only the global sequence ends differ from the unpacked op: a
replicate pad of packed rows repeats the last ``k`` samples instead of the last one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn

from .audio_ops import Conv1dViaConv3d, SnakeBeta, _make_kaiser_sinc_kernel_1d
from .module import Module


def packed_weight(
    op, *, c_in: int, c_out: int, k_in: int, k_out: int, support: int, q_half: int | None = None
) -> torch.Tensor:
    """Dense packed weight ``(k_out * c_out, k_in * c_in, K')`` of a shift-invariant torch op.

    ``op`` maps ``(1, c_in, L)`` to ``(1, c_out, L * k_out / k_in)`` (no bias) and commutes with shifts by ``k_in``
    input / ``k_out`` output samples. ``support`` bounds the op's receptive field in input samples. The result is a
    "same"-padded odd-length kernel (zero taps pad an asymmetric reach), so ``F.conv1d(X', W', padding=K'//2)`` on
    packed rows reproduces ``op`` away from the sequence ends. ``q_half`` fixes ``K' = 2 * q_half + 1`` (a weight
    with chance zero taps must keep the kernel its module was built for); ``None`` trims zero taps symmetrically.
    """
    q_max = -(-(support + k_in) // k_in) + 1
    rows = 2 * q_max + 5
    t0 = rows // 2
    width = k_out * c_out
    taps = torch.zeros(2 * q_max + 1, k_in * c_in, width, dtype=torch.float64)
    for r in range(k_in):
        for ci in range(c_in):
            x = torch.zeros(1, c_in, rows * k_in, dtype=torch.float64)
            x[0, ci, t0 * k_in + r] = 1.0
            with torch.no_grad():
                y = op(x)
            assert y.shape[-1] == rows * k_out, f"op must map {rows * k_in} -> {rows * k_out} samples, got {y.shape}"
            y_rows = y[0].transpose(0, 1).reshape(rows, width)  # (T', k_out * c_out), slot-major
            for d in range(-q_max, q_max + 1):
                taps[q_max - d, r * c_in + ci] = y_rows[t0 + d]
    if q_half is None:
        # Trim zero taps symmetrically so the kernel stays "same"-padded.
        nonzero = [q for q in range(taps.shape[0]) if taps[q].abs().max() > 0]
        assert nonzero, "op has no response"
        trim = min(nonzero[0], taps.shape[0] - 1 - nonzero[-1])
    else:
        trim = q_max - q_half
        assert trim >= 0, f"q_half {q_half} exceeds the probed reach {q_max}"
        assert taps[:trim].abs().max() == 0 and taps[taps.shape[0] - trim :].abs().max() == 0, "response outside q_half"
    taps = taps[trim : taps.shape[0] - trim]
    return taps.permute(2, 1, 0).contiguous().float()  # (k_out*c_out, k_in*c_in, K') = Conv1d weight layout


def conv1d_same(weight: torch.Tensor, dilation: int):
    """Bias-free "same" ``Conv1d`` closure for ``packed_weight``."""
    k = weight.shape[-1]
    pad = (k - 1) * dilation // 2
    w = weight.double()
    return lambda x: F.conv1d(x, w, padding=pad, dilation=dilation)


def upsample2x_ref(taps: torch.Tensor, channels: int):
    """BigVGAN ``UpSample1d`` (ratio 2) as a bias-free closure: replicate pad, ``2 * conv_transpose1d``, crop."""
    k = taps.numel()
    pad = k // 2 - 1
    crop = pad * 2 + (k - 2) // 2
    w = taps.double().reshape(1, 1, k).expand(channels, 1, k).contiguous()

    def op(x):
        xp = F.pad(x, (pad, pad), mode="replicate")
        y = 2.0 * F.conv_transpose1d(xp, w, stride=2, groups=channels)
        return y[..., crop : y.shape[-1] - crop]

    return op


def downsample2x_ref(taps: torch.Tensor, channels: int):
    """BigVGAN ``DownSample1d`` (ratio 2): replicate pad ``(k/2 - 1, k/2)``, strided depthwise conv."""
    k = taps.numel()
    w = taps.double().reshape(1, 1, k).expand(channels, 1, k).contiguous()

    def op(x):
        xp = F.pad(x, (k // 2 - 1, k // 2), mode="replicate")
        return F.conv1d(xp, w, stride=2, groups=channels)

    return op


def kaiser_taps(ratio: int = 2, kernel_size: int = 12) -> torch.Tensor:
    return _make_kaiser_sinc_kernel_1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size)


class PackedConv1d(Conv1dViaConv3d):
    """A dilated "same" ``Conv1d`` on ``k``-packed rows: ``(B, T/k, k*C_in) -> (B, T/k, k*C_out)``.

    Loads the ordinary ``(C_out, C_in, K)`` torch weight and bias and packs them at load time.
    """

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, dilation: int = 1, pack: int, **kw):
        support = (kernel_size - 1) * dilation + 1
        probe = packed_weight(
            conv1d_same(torch.ones(out_channels, in_channels, kernel_size), dilation),
            c_in=in_channels,
            c_out=out_channels,
            k_in=pack,
            k_out=pack,
            support=support,
        )
        super().__init__(
            pack * in_channels, pack * out_channels, kernel_size=probe.shape[-1], dilation=1, padding_mode="zeros", **kw
        )
        self.pack = pack
        self.orig = (out_channels, in_channels, kernel_size)
        self.orig_dilation = dilation
        self.q_half = probe.shape[-1] // 2

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            w = state["weight"]
            assert tuple(w.shape) == self.orig, (tuple(w.shape), self.orig)
            state["weight"] = packed_weight(
                conv1d_same(w, self.orig_dilation),
                c_in=self.orig[1],
                c_out=self.orig[0],
                k_in=self.pack,
                k_out=self.pack,
                support=(self.orig[2] - 1) * self.orig_dilation + 1,
                q_half=self.q_half,
            )
        if "bias" in state:
            state["bias"] = state["bias"].reshape(-1).repeat(self.pack)
        super()._prepare_torch_state(state)


def _packed_resample_weight(taps, channels, k_in, k_out, up, q_half=None):
    ref = (upsample2x_ref if up else downsample2x_ref)(taps, channels)
    return packed_weight(
        ref, c_in=channels, c_out=channels, k_in=k_in, k_out=k_out, support=taps.numel() * 2 + 2, q_half=q_half
    )


class PackedResample(Conv1dViaConv3d):
    """A fixed depthwise 2x resampler (up or down) as a dense conv on packed rows.

    ``k_in`` input slots per row, ``k_out`` output slots per row (``k_out = 2 * k_in`` up, ``k_in / 2`` down). The
    taps come from the checkpoint filter when it carries one, else the kaiser-sinc default. Replicate padding at the
    sequence ends becomes a replicate halo of packed rows when T-sharded (zeros when unsharded).
    """

    def __init__(self, channels: int, *, k_in: int, k_out: int, up: bool, **kw):
        taps = kaiser_taps()
        probe = _packed_resample_weight(taps, channels, k_in, k_out, up)
        super().__init__(
            k_in * channels, k_out * channels, kernel_size=probe.shape[-1], padding_mode="zeros", bias=False, **kw
        )
        self.channels, self.k_in, self.k_out, self.up = channels, k_in, k_out, up
        self._taps = taps
        self.q_half = probe.shape[-1] // 2
        self.halo_padding_mode = "replicate"

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        taps = state.pop("filter", None)
        if taps is not None:
            self._taps = taps.reshape(-1).float()
        state["weight"] = _packed_resample_weight(
            self._taps, self.channels, self.k_in, self.k_out, self.up, q_half=self.q_half
        )
        super()._prepare_torch_state(state)


class PackedActivation1d(Module):
    """``UpSample1d(2x) -> SnakeBeta -> DownSample1d(2x)`` on ``k``-packed rows: three ops plus the snake's layout
    round trip, against ~25 today. Loads the unpacked ``Activation1d`` state (``act.alpha``, ``act.beta`` and the
    optional resampler filters), tiling the per-channel snake parameters over the ``2k`` slots of the upsampled row.
    """

    def __init__(
        self,
        *,
        channels: int,
        pack: int,
        mesh_device,
        dtype=ttnn.float32,
        parallel_config=None,
        ccl_manager=None,
        split_mode="off",
        resampler_split_mode: str | None = None,
    ):
        super().__init__()
        self.channels, self.pack = channels, pack
        # The resamplers' fixed kaiser taps tolerate a cheaper split than the learned convs: measured on the H3
        # decoder, no split on them costs 2.3 dB (67.3 -> 65.1) and saves 72 ms of 414.
        rs_mode = split_mode if resampler_split_mode is None else resampler_split_mode
        common = dict(mesh_device=mesh_device, dtype=dtype, parallel_config=parallel_config, ccl_manager=ccl_manager)
        self.up = PackedResample(channels, k_in=pack, k_out=2 * pack, up=True, split_mode=rs_mode, **common)
        self.act = SnakeBeta(
            2 * pack * channels,
            alpha_logscale=True,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
        )
        self.down = PackedResample(channels, k_in=2 * pack, k_out=pack, up=False, split_mode=rs_mode, **common)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("act.alpha", "act.beta"):
            if name in state:
                state[name] = state[name].reshape(-1).repeat(2 * self.pack)
        if "upsample.filter" in state:
            state["up.filter"] = state.pop("upsample.filter")
        if "downsample.lowpass.filter" in state:
            state["down.filter"] = state.pop("downsample.lowpass.filter")

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        y = self.up(x_BTC)
        y = self.act(y)
        if y.layout != ttnn.ROW_MAJOR_LAYOUT:
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        return self.down(y)
