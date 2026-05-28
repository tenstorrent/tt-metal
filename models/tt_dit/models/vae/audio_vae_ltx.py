# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 Audio VAE *decoder* on Tenstorrent (TTNN).

This ports the **audio VAE decoder only** (audio latent -> log-mel spectrogram)
to device. The vocoder (mel -> waveform) stays on CPU for now; the mel produced
here is handed back to the host and fed to the reference vocoder unchanged.

Design notes
------------
* The audio decoder is *tiny* (latent is ``(1, z, ~151, 16)``) so there is no
  benefit to mesh sharding. Weights are **replicated** across the mesh and the
  conv stack runs single-device (identically on every device).
* The module is built directly *from a loaded reference* ``AudioDecoder``
  (``from_reference``), reading channel sizes, padding, and weights straight off
  the torch layers. This avoids re-deriving the checkpoint config and guarantees
  structural parity with the reference.
* The reference uses ``PixelNorm`` (per-location RMS over channels) — *not*
  GroupNorm — because causal conv blocks forbid GroupNorm. PixelNorm has no
  learnable params, so only the conv weights are loaded.
* Trivial host-side bookkeeping (per-channel de-normalisation, patchify/
  unpatchify, output crop/pad) is delegated back to the reference module's own
  helpers; only the conv/attention stack runs on device.

Reference: ``LTX-2/packages/ltx-core/src/ltx_core/model/audio_vae/``
  - ``audio_vae.py``        (AudioDecoder forward + helpers)
  - ``causal_conv_2d.py``   (CausalConv2d padding semantics)
  - ``resnet.py``           (ResnetBlock: norm -> silu -> conv x2 + shortcut)
  - ``attention.py``        (AttnBlock: 1x1 q/k/v + softmax + proj_out)
  - ``upsample.py``         (nearest x2 + conv + causal-axis trim)

WARNING: This has not yet been validated on hardware. The conv/groupnorm/memory
configs and op layouts almost certainly need a few rounds of on-device debugging
(PCC + OOM/layout fixes) before the parity test passes. See
``tests/.../test_audio_vae_ltx.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ...utils.tensor import from_torch as tt_from_torch

if TYPE_CHECKING:
    pass

# PixelNorm epsilon (matches ltx_core PixelNorm: build_normalization_layer eps=1e-6)
_PIXEL_NORM_EPS = 1e-6

_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def _to_tile(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    return x


def _to_rm(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return x


def _pixel_norm(x: ttnn.Tensor, eps: float = _PIXEL_NORM_EPS) -> ttnn.Tensor:
    """Per-location RMS norm over the channel (last) dim: x / sqrt(mean(x^2)+eps)."""
    x = _to_tile(x)
    sq = ttnn.mul(x, x)
    mean_sq = ttnn.mean(sq, dim=-1, keepdim=True)
    inv_rms = ttnn.rsqrt(ttnn.add(mean_sq, eps))
    return ttnn.mul(x, inv_rms)


class _TtConv2d:
    """A single-device ttnn.conv2d wrapper that mirrors a torch conv.

    Accepts either a reference ``CausalConv2d`` (asymmetric ``F.pad`` handled
    explicitly via ``ttnn.pad``) or a plain ``torch.nn.Conv2d`` (1x1 attention
    projections). Activations are NHWC with ``H = time``, ``W = freq``.
    """

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple[int, int],
        kernel_size: tuple[int, int],
        pad_nhwc: tuple[int, int, int, int],  # (top, bottom, left, right) over (H=time, W=freq)
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        self.mesh_device = mesh_device
        self.out_channels, self.in_channels, self.kh, self.kw = weight.shape
        self.stride = stride
        self.kernel_size = kernel_size
        self.pad_nhwc = pad_nhwc

        # ttnn.conv2d preps weights itself; keep them host + row-major (OIHW),
        # replicated across the mesh (all mesh axes None == replicate).
        self.weight = tt_from_torch(
            weight.contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[None] * weight.ndim,
            on_host=True,
        )
        if bias is not None:
            bias_4d = bias.reshape(1, 1, 1, -1).contiguous()
            self.bias = tt_from_torch(
                bias_4d,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_axes=[None] * bias_4d.ndim,
                on_host=True,
            )
        else:
            self.bias = None

    @classmethod
    def from_causal_conv(cls, cc, mesh_device: ttnn.MeshDevice) -> "_TtConv2d":
        # cc.padding is the F.pad tuple (pad_left, pad_right, pad_top, pad_bottom)
        # over the torch NCHW layout (last dim = freq = W, 2nd-last = time = H).
        pl, pr, pt, pb = cc.padding
        conv = cc.conv
        return cls(
            weight=conv.weight.detach().float(),
            bias=conv.bias.detach().float() if conv.bias is not None else None,
            stride=tuple(conv.stride),
            kernel_size=tuple(conv.kernel_size),
            pad_nhwc=(pt, pb, pl, pr),
            mesh_device=mesh_device,
        )

    @classmethod
    def from_conv(cls, conv: torch.nn.Conv2d, mesh_device: ttnn.MeshDevice) -> "_TtConv2d":
        ph, pw = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        return cls(
            weight=conv.weight.detach().float(),
            bias=conv.bias.detach().float() if conv.bias is not None else None,
            stride=tuple(conv.stride),
            kernel_size=tuple(conv.kernel_size),
            pad_nhwc=(ph, ph, pw, pw),
            mesh_device=mesh_device,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = _to_rm(x)
        top, bottom, left, right = self.pad_nhwc
        if any((top, bottom, left, right)):
            x = ttnn.pad(x, [(0, 0), (top, bottom), (left, right), (0, 0)], value=0.0)

        b, h, w, _ = x.shape
        out, [oh, ow] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.mesh_device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
            batch_size=b,
            input_height=h,
            input_width=w,
            conv_config=ttnn.Conv2dConfig(act_block_h_override=32),
            compute_config=_COMPUTE_CONFIG,
            return_output_dim=True,
        )
        return ttnn.reshape(out, (b, oh, ow, self.out_channels))


class _TtResnetBlock:
    """norm -> silu -> conv1 -> norm -> silu -> conv2 (+ optional 1x1/3x3 shortcut)."""

    def __init__(self, ref, mesh_device: ttnn.MeshDevice) -> None:
        self.conv1 = _TtConv2d.from_causal_conv(ref.conv1, mesh_device)
        self.conv2 = _TtConv2d.from_causal_conv(ref.conv2, mesh_device)
        self.shortcut: _TtConv2d | None = None
        if ref.in_channels != ref.out_channels:
            if getattr(ref, "use_conv_shortcut", False):
                self.shortcut = _TtConv2d.from_causal_conv(ref.conv_shortcut, mesh_device)
            else:
                self.shortcut = _TtConv2d.from_causal_conv(ref.nin_shortcut, mesh_device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = _pixel_norm(x)
        h = ttnn.silu(h)
        h = self.conv1(h)
        h = _pixel_norm(h)
        h = ttnn.silu(h)
        h = self.conv2(h)
        residual = self.shortcut(x) if self.shortcut is not None else x
        return ttnn.add(_to_tile(residual), _to_tile(h))


class _TtAttnBlock:
    """Vanilla spatial self-attention (single head over H*W) matching AttnBlock."""

    def __init__(self, ref, mesh_device: ttnn.MeshDevice) -> None:
        self.q = _TtConv2d.from_conv(ref.q, mesh_device)
        self.k = _TtConv2d.from_conv(ref.k, mesh_device)
        self.v = _TtConv2d.from_conv(ref.v, mesh_device)
        self.proj_out = _TtConv2d.from_conv(ref.proj_out, mesh_device)
        self.in_channels = ref.in_channels

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        residual = x
        h = _pixel_norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, height, width, c = q.shape
        hw = height * width
        scale = float(c) ** -0.5

        q = _to_tile(ttnn.reshape(q, (b, hw, c)))
        k = _to_tile(ttnn.reshape(k, (b, hw, c)))
        v = _to_tile(ttnn.reshape(v, (b, hw, c)))

        # scores[b,i,j] = sum_c q[b,i,c] k[b,j,c]  -> (b, hw, hw), softmax over keys (j)
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        scores = ttnn.mul(scores, scale)
        attn = ttnn.softmax(scores, dim=-1)
        # out[b,i,c] = sum_j attn[b,i,j] v[b,j,c]
        out = ttnn.matmul(attn, v)
        out = ttnn.reshape(out, (b, height, width, c))
        out = self.proj_out(out)
        return ttnn.add(_to_tile(residual), _to_tile(out))


class _TtUpsample:
    """Nearest x2 (both axes) -> conv -> drop first element along causality axis."""

    def __init__(self, ref, mesh_device: ttnn.MeshDevice) -> None:
        self.with_conv = ref.with_conv
        self.conv = _TtConv2d.from_causal_conv(ref.conv, mesh_device) if ref.with_conv else None
        # CausalityAxis enum value string: "none" | "height" | "width" | "width_compatibility"
        self.causality_axis = getattr(ref.causality_axis, "value", str(ref.causality_axis)).lower()

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = _to_rm(x)
        x = ttnn.upsample(x, scale_factor=2)
        if self.conv is not None:
            x = self.conv(x)
            b, h, w, c = x.shape
            if self.causality_axis == "height":
                x = ttnn.slice(x, [0, 1, 0, 0], [b, h, w, c])
            elif self.causality_axis == "width":
                x = ttnn.slice(x, [0, 0, 1, 0], [b, h, w, c])
            # "none"/"width_compatibility": no trim
        return x


class _TtMaybeAttn:
    """Mirror reference ``mid.attn_1`` which may be ``nn.Identity``."""

    def __init__(self, ref, mesh_device: ttnn.MeshDevice) -> None:
        # AttnBlock has a ``proj_out`` attribute; Identity does not.
        self.attn = _TtAttnBlock(ref, mesh_device) if hasattr(ref, "proj_out") else None

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.attn(x) if self.attn is not None else x


class TtAudioDecoder:
    """On-device LTX-2 audio VAE decoder (latent -> log-mel spectrogram).

    Build via :meth:`from_reference` with a *loaded* reference ``AudioDecoder``.
    Call with the same torch latent the reference decoder expects, shaped
    ``(B, z_channels, frames, mel_bins_latent)``; returns the reference-shaped
    mel spectrogram as a torch tensor (ready for the CPU vocoder).
    """

    def __init__(self, ref, mesh_device: ttnn.MeshDevice) -> None:
        self.ref = ref  # reference torch AudioDecoder (for host pre/post helpers)
        self.mesh_device = mesh_device
        self.num_resolutions = ref.num_resolutions

        self.conv_in = _TtConv2d.from_causal_conv(ref.conv_in, mesh_device)

        self.mid_block_1 = _TtResnetBlock(ref.mid.block_1, mesh_device)
        self.mid_attn_1 = _TtMaybeAttn(ref.mid.attn_1, mesh_device)
        self.mid_block_2 = _TtResnetBlock(ref.mid.block_2, mesh_device)

        # up[level] = {"blocks": [...], "attns": [...]|[], "upsample": _TtUpsample|None}
        self.up: list[dict] = []
        for level in range(self.num_resolutions):
            stage = ref.up[level]
            blocks = [_TtResnetBlock(b, mesh_device) for b in stage.block]
            attns = [_TtAttnBlock(a, mesh_device) for a in stage.attn] if len(stage.attn) > 0 else []
            upsample = _TtUpsample(stage.upsample, mesh_device) if (level != 0 and hasattr(stage, "upsample")) else None
            self.up.append({"blocks": blocks, "attns": attns, "upsample": upsample})

        self.conv_out = _TtConv2d.from_causal_conv(ref.conv_out, mesh_device)

    @classmethod
    def from_reference(cls, ref, *, mesh_device: ttnn.MeshDevice) -> "TtAudioDecoder":
        return cls(ref, mesh_device)

    def _run_device_stack(self, x_nchw: torch.Tensor) -> torch.Tensor:
        b = x_nchw.shape[0]
        # NCHW (b, c, time, freq) -> NHWC (b, time, freq, c)
        x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
        x = tt_from_torch(
            x_nhwc,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[None] * x_nhwc.ndim,
        )

        x = self.conv_in(x)
        x = self.mid_block_1(x)
        x = self.mid_attn_1(x)
        x = self.mid_block_2(x)

        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage["blocks"]):
                x = block(x)
                if stage["attns"]:
                    x = stage["attns"][block_idx](x)
            if stage["upsample"] is not None:
                x = stage["upsample"](x)

        x = _pixel_norm(x)
        x = ttnn.silu(x)
        x = self.conv_out(x)

        # Weights/inputs are replicated, so every device holds the same result;
        # read back the first shard.
        out = ttnn.to_torch(ttnn.get_device_tensors(x)[0])[:b]
        # NHWC (b, time, freq, out_ch) -> NCHW (b, out_ch, time, freq)
        return out.permute(0, 3, 1, 2).contiguous().float()

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # Host: per-channel de-normalisation + target-shape derivation (cheap).
        denorm, target_shape = self.ref._denormalize_latents(sample)
        # Device: conv/attention stack.
        decoded = self._run_device_stack(denorm)
        # Host: crop/pad to exact target shape + optional tanh (cheap).
        if getattr(self.ref, "tanh_out", False):
            decoded = torch.tanh(decoded)
        return self.ref._adjust_output_shape(decoded, target_shape)
