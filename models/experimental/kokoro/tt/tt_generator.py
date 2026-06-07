# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro istftnet :class:`~models.experimental.kokoro.reference.istftnet.Generator`
(``TorchSTFT`` path only — ``disable_complex=True`` / ``CustomSTFT`` is out of scope here).

Reference pipeline:

    f0 = upsample_nearest(f0[:, None]).transpose(1, 2)              # [B, T_har, 1]
    har_source, _, _ = m_source(f0)                                 # [B, T_har, 1]
    har_spec, har_phase = stft.transform(har_source.squeeze(-1))    # [B, K, F] each
    har = cat([har_spec, har_phase], dim=1)                         # [B, 2K, F]

    for i in range(num_upsamples):
        x = leaky_relu(x, 0.1)
        x_source = noise_res[i](noise_convs[i](har), s)
        x = ups[i](x)                                               # ConvTranspose1d, ×upsample_rates[i]
        if i == num_upsamples - 1:
            x = reflection_pad_left(x, 1)                           # ReflectionPad1d((1, 0))
        x = x + x_source
        xs = sum(resblocks[i*nk + j](x, s) for j in range(num_kernels)) / num_kernels
        x = xs

    x = leaky_relu(x)
    x = conv_post(x)                                                # Conv1d
    spec  = exp(x[:, :K_post, :])
    phase = sin(x[:, K_post:, :])
    return stft.inverse(spec, phase)                                # [B, 1, audio_len]

Internally everything is in NLC ``[B, T, C]``; we permute at the iSTFT/STFT boundaries only.

PyTorch is used only at preprocess time (weight upload + ``weight_norm`` folding).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from torch.nn.utils import parametrize

import ttnn

from .tt_adain_resblock1 import (
    TTAdaINResBlock1,
    TTAdaINResBlock1Params,
    preprocess_tt_adain_resblock1,
)
from .tt_conv import (
    TTConv1dParams,
    TTConvTranspose1dParams,
    tt_conv1d_nlc,
    tt_conv_transpose1d_nlc,
)
from .tt_source_module_hn_nsf import (
    TTSourceModuleHnNSFParams,
    TTSourceModuleHnNSF,
    preprocess_tt_source_module_hn_nsf,
)
from .tt_torch_stft import (
    TTTorchSTFT,
    TTTorchSTFTParams,
    preprocess_tt_torch_stft,
)


# ---------------------------------------------------------------------------
# weight upload helpers
# ---------------------------------------------------------------------------


def _strip_weight_norm(m: nn.Module) -> None:
    """Fold ``weight_norm`` into ``.weight`` (``parametrizations`` or legacy hook). Idempotent."""
    if parametrize.is_parametrized(m, "weight"):
        parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        return
    try:
        nn.utils.remove_weight_norm(m, "weight")
    except ValueError:
        pass


def _conv1d_to_tt_params(conv: nn.Conv1d, *, weights_dtype) -> TTConv1dParams:
    """Upload a ``nn.Conv1d`` (after any ``weight_norm`` is folded) for :func:`tt_conv1d_nlc`."""
    w = conv.weight.detach().cpu().unsqueeze(-1)  # [out, in/g, k] -> [out, in/g, k, 1]
    w_tt = ttnn.from_torch(
        w,
        dtype=weights_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    b_tt = None
    if conv.bias is not None:
        b_tt = ttnn.from_torch(
            conv.bias.detach().cpu().reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    return TTConv1dParams(
        weight=w_tt,
        bias=b_tt,
        in_channels=int(conv.in_channels),
        out_channels=int(conv.out_channels),
        kernel_size=int(conv.kernel_size[0]),
        stride=int(conv.stride[0]),
        padding=int(conv.padding[0]),
        groups=int(conv.groups),
        dilation=int(conv.dilation[0]),
    )


def _conv_transpose1d_to_tt_params(m: nn.ConvTranspose1d, _device, *, weights_dtype) -> TTConvTranspose1dParams:
    """Upload a ``nn.ConvTranspose1d`` for :func:`tt_conv_transpose1d_nlc` (``spatial_style="height"``).

    Uses the same height-style layout as :func:`tt_adain_resblk_1d._conv_transpose_pool_to_tt_params`
    (which the existing TT tests cover) — weight shape ``[in, out/g, k, 1]``. Works for both
    depthwise (groups=in) and regular (groups=1) transpose convs.
    """
    w = m.weight.detach().cpu().unsqueeze(-1)  # [in, out/g, k] -> [in, out/g, k, 1]
    # Keep transpose-conv weights/bias in host ROW_MAJOR; TTNN conv_transpose2d expects to
    # preprocess this raw IOHW form for the current invocation shape.
    w_tt = ttnn.from_torch(
        w,
        dtype=weights_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    b_tt = None
    if m.bias is not None:
        b_tt = ttnn.from_torch(
            m.bias.detach().cpu().reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    return TTConvTranspose1dParams(
        weight=w_tt,
        bias=b_tt,
        in_channels=int(m.in_channels),
        out_channels=int(m.out_channels),
        kernel_size=int(m.kernel_size[0]),
        stride=int(m.stride[0]),
        padding=int(m.padding[0]),
        output_padding=int(m.output_padding[0]),
        groups=int(m.groups),
        mirror_kernel=True,
        spatial_style="height",
    )


# ---------------------------------------------------------------------------
# params + preprocess
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTGeneratorUpsampleStageParams:
    """Weights for one of the ``num_upsamples`` stages."""

    ups: TTConvTranspose1dParams
    noise_conv: TTConv1dParams
    noise_res: TTAdaINResBlock1Params
    resblocks: tuple[TTAdaINResBlock1Params, ...]


@dataclass(frozen=True)
class TTGeneratorParams:
    """Device-resident weights for :class:`TTGenerator`."""

    m_source: TTSourceModuleHnNSFParams
    stft: TTTorchSTFTParams
    stages: tuple[TTGeneratorUpsampleStageParams, ...]
    conv_post: TTConv1dParams

    upsample_rates: tuple[int, ...]
    upsample_scale_full: int  # ``prod(upsample_rates) * gen_istft_hop_size``
    post_n_fft: int
    num_kernels: int
    num_upsamples: int
    time_len_x: int  # Input ``x`` time axis (mel-frame count) baked into ``m_source`` / ``stft``.


def preprocess_tt_generator(
    module: nn.Module,
    device: ttnn.Device,
    *,
    time_len_x: int,
    weights_dtype=ttnn.float32,
    conv_weights_dtype=ttnn.float32,
    source_stft_dtype=ttnn.float32,
) -> TTGeneratorParams:
    """Upload a reference ``Generator`` to device for :class:`TTGenerator`.

    ``time_len_x`` is the input ``x`` time axis (mel-frame count); ``m_source`` and ``stft`` need
    fixed time axes baked at construction time. The harmonic source length is
    ``time_len_x * upsample_scale_full`` and the STFT input/output length matches.

    ``source_stft_dtype`` controls the precision of :class:`TTSourceModuleHnNSF` and
    :class:`TTTorchSTFT` weights. The default is fp32 because the trained Kokoro sine source has
    per-sample RMS ≈ 0.06, putting the STFT magnitudes near the bf16 precision floor — phase from
    ``atan2`` of bf16 real/imag pairs is essentially random there. fp32 keeps the harmonic ``har``
    PCC well above 0.99 (vs ~0.2 in bf16) without otherwise changing the model.
    """
    upsample_rates = (
        tuple(int(r) for r in module.f0_upsamp.scale_factor)
        if isinstance(module.f0_upsamp.scale_factor, (list, tuple))
        else (int(module.f0_upsamp.scale_factor),)
    )
    # ``f0_upsamp`` was built with the prod of upsample_rates × hop_size as a single scalar; recover
    # the original list from ``module.ups`` strides.
    upsample_rates = tuple(int(u.stride[0]) for u in module.ups)
    hop_size = int(module.stft.hop_length)
    post_n_fft = int(module.post_n_fft)
    upsample_scale_full = int(module.f0_upsamp.scale_factor)
    expected = math.prod(upsample_rates) * hop_size
    assert (
        upsample_scale_full == expected
    ), f"f0_upsamp scale {upsample_scale_full} != prod(upsample_rates)*hop {expected}"

    num_upsamples = len(module.ups)
    num_kernels = len(module.resblocks) // num_upsamples
    assert num_upsamples * num_kernels == len(
        module.resblocks
    ), f"resblock count {len(module.resblocks)} not divisible by num_upsamples {num_upsamples}"

    har_time_len = time_len_x * upsample_scale_full

    # m_source: SineGen + Linear + Tanh + output noise scale. fp32 weights to keep the small
    # sine source amplitude (per-sample RMS ≈ 0.06 with trained Kokoro weights) numerically stable
    # for the downstream STFT.
    m_source = preprocess_tt_source_module_hn_nsf(
        module.m_source,
        device,
        sampling_rate=float(module.m_source.l_sin_gen.sampling_rate),
        upsample_scale=upsample_scale_full,
        harmonic_num=int(module.m_source.l_sin_gen.harmonic_num),
        voiced_threshold=float(module.m_source.l_sin_gen.voiced_threshold),
        time_len=har_time_len,
        weights_dtype=source_stft_dtype,
    )

    # STFT: same input length used for both transform and inverse. fp32 matrices to keep
    # ``atan2`` of small real/imag pairs precise.
    stft = preprocess_tt_torch_stft(
        filter_length=int(module.stft.filter_length),
        hop_length=hop_size,
        win_length=int(module.stft.win_length),
        input_length=har_time_len,
        device=device,
        weights_dtype=source_stft_dtype,
    )

    stages: list[TTGeneratorUpsampleStageParams] = []
    for i in range(num_upsamples):
        ups_conv = module.ups[i]
        _strip_weight_norm(ups_conv)
        ups_p = _conv_transpose1d_to_tt_params(ups_conv, device, weights_dtype=conv_weights_dtype)

        noise_conv = module.noise_convs[i]
        _strip_weight_norm(noise_conv)  # No-op for plain Conv1d
        noise_conv_p = _conv1d_to_tt_params(noise_conv, weights_dtype=conv_weights_dtype)

        noise_res_p = preprocess_tt_adain_resblock1(
            module.noise_res[i],
            device,
            weights_dtype=weights_dtype,
            conv_weights_dtype=conv_weights_dtype,
        )

        resblocks_p = tuple(
            preprocess_tt_adain_resblock1(
                module.resblocks[i * num_kernels + j],
                device,
                weights_dtype=weights_dtype,
                conv_weights_dtype=conv_weights_dtype,
            )
            for j in range(num_kernels)
        )

        stages.append(
            TTGeneratorUpsampleStageParams(
                ups=ups_p,
                noise_conv=noise_conv_p,
                noise_res=noise_res_p,
                resblocks=resblocks_p,
            )
        )

    conv_post = module.conv_post
    _strip_weight_norm(conv_post)
    conv_post_p = _conv1d_to_tt_params(conv_post, weights_dtype=conv_weights_dtype)

    return TTGeneratorParams(
        m_source=m_source,
        stft=stft,
        stages=tuple(stages),
        conv_post=conv_post_p,
        upsample_rates=upsample_rates,
        upsample_scale_full=upsample_scale_full,
        post_n_fft=post_n_fft,
        num_kernels=num_kernels,
        num_upsamples=num_upsamples,
        time_len_x=int(time_len_x),
    )


# ---------------------------------------------------------------------------
# small ops
# ---------------------------------------------------------------------------


def _reflection_pad_left_1_nlc(x_nlc: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """Match ``nn.ReflectionPad1d((1, 0))`` on a ``[B, T, C]`` NLC tensor (output ``[B, T+1, C]``).

    Reflection pad of 1 on the left maps ``out[0] = x[1]`` then concatenates the rest of ``x``.
    """
    B, _T, C = (int(d) for d in x_nlc.shape)
    head = ttnn.slice(x_nlc, [0, 1, 0], [B, 2, C], [1, 1, 1], memory_config=memory_config)
    out = ttnn.concat([head, x_nlc], dim=1, memory_config=memory_config)
    ttnn.deallocate(head)
    return out


def _upsample_nearest_axis1(x_nlc: ttnn.Tensor, *, scale: int, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """Nearest-neighbour upsample along axis 1 (matches ``nn.Upsample(scale_factor=scale)``)."""
    if scale == 1:
        return x_nlc
    return ttnn.repeat_interleave(x_nlc, scale, 1, memory_config=memory_config)


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------


class TTGenerator:
    """TTNN port of ``Generator`` (``TorchSTFT`` only).

    ``use_torch_stft_fallback=True`` routes the entire STFT ``transform`` through CPU ``torch.stft``.
    ``use_torch_phase_fallback=True`` runs the SineGen phase chain on CPU float32.
    """

    def __init__(
        self,
        device: ttnn.Device,
        params: TTGeneratorParams,
        *,
        use_torch_stft_fallback: bool = False,
        use_torch_phase_fallback: bool = False,
    ) -> None:
        self.device = device
        self.params = params
        # HiFi4 (with fp32_dest_acc_en) measured better than HiFi3 for the STFT precision
        # bottleneck (cos(phase) PCC 0.78 vs 0.77; near-zero sign match 0.70 vs 0.69). Apply the
        # same setting to the surrounding convs / resblocks so the chain stays consistent.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._m_source = TTSourceModuleHnNSF(
            device,
            params.m_source,
            use_torch_phase_fallback=use_torch_phase_fallback,
        )
        self._stft = TTTorchSTFT(
            device,
            params.stft,
            use_torch_stft_fallback=use_torch_stft_fallback,
        )
        # Keep the full harmonic-source path (SineGen + Source linear + STFT) on the same
        # precision profile as the rest of the generator to avoid mixed-fidelity phase drift.
        self._m_source.compute_kernel_config = self.compute_kernel_config
        self._m_source._sinegen.compute_kernel_config = self.compute_kernel_config
        self._noise_res = tuple(TTAdaINResBlock1(device, sp.noise_res) for sp in params.stages)
        self._resblocks = tuple(tuple(TTAdaINResBlock1(device, rb) for rb in sp.resblocks) for sp in params.stages)

    def _harmonic_source_path(
        self,
        f0: ttnn.Tensor,
        *,
        sinegen_rand_ini: Optional[ttnn.Tensor],
        sinegen_noise_raw: Optional[ttnn.Tensor],
        source_noise_raw: Optional[ttnn.Tensor],
        memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        """``f0 -> f0_upsamp -> m_source -> stft.transform -> cat(spec, phase, dim=channel)``.

        Returns ``har`` in **NLC** ``[B, F, 2K]`` so subsequent convs/resblocks operate channel-last.
        (The reference cats along channel in BCT, which is the last dim in NLC.)
        """
        p = self.params

        # ``f0`` in: ``[B, T_f0]`` or ``[B, T_f0, 1]``. We need ``[B, T_har, 1]``.
        f_shape = list(f0.shape)
        if len(f_shape) == 2:
            f0_b_t_1 = ttnn.unsqueeze(f0, 2)
        else:
            f0_b_t_1 = f0
        # Upcast to fp32 so the small downstream sine source survives ``atan2`` in :class:`TTTorchSTFT`
        # (trained Kokoro ``sine_merge`` has per-sample RMS ≈ 0.06; bf16 phase is then ~random).
        f0_fp32 = ttnn.typecast(f0_b_t_1, ttnn.float32, memory_config=memory_config)
        if len(f_shape) == 2:
            ttnn.deallocate(f0_b_t_1)
        f0_b_t_1 = f0_fp32
        f0_har = _upsample_nearest_axis1(
            f0_b_t_1,
            scale=p.upsample_scale_full,
            memory_config=memory_config,
        )
        ttnn.deallocate(f0_b_t_1)

        # m_source -> har_source ``[B, T_har, 1]``
        har_source, _noise_out, _uv = self._m_source.forward(
            f0_har,
            sinegen_rand_ini=sinegen_rand_ini,
            sinegen_noise_raw=sinegen_noise_raw,
            out_noise_raw=source_noise_raw,
            memory_config=memory_config,
        )
        ttnn.deallocate(f0_har)
        ttnn.deallocate(_noise_out)
        ttnn.deallocate(_uv)

        # ``har_source.transpose(1, 2).squeeze(1)`` in the reference -> ``[B, T_har]``.
        # Drop the trailing singleton channel so STFT sees ``[B, T_har]`` like the Torch path.
        har_flat = ttnn.squeeze(har_source, 2)
        ttnn.deallocate(har_source)

        # STFT: ``transform`` returns ``(mag, phase)`` each ``[B, K, F]`` (BCT-style). Cast to
        # the configured ``activation_dtype`` of the source module (fp32 by default — see the
        # rationale in :func:`preprocess_tt_generator`).
        if har_flat.dtype != p.m_source.sinegen.activation_dtype:
            har_flat_cast = ttnn.typecast(
                har_flat,
                p.m_source.sinegen.activation_dtype,
                memory_config=memory_config,
            )
            ttnn.deallocate(har_flat)
            har_flat = har_flat_cast
        mag, phase = self._stft.transform(har_flat)
        ttnn.deallocate(har_flat)

        # ``cat([mag, phase], dim=1)`` -> ``[B, 2K, F]`` (BCT). Permute to NLC ``[B, F, 2K]``.
        har_bct = ttnn.concat([mag, phase], dim=1, memory_config=memory_config)
        ttnn.deallocate(mag)
        ttnn.deallocate(phase)
        har_nlc = ttnn.permute(har_bct, (0, 2, 1), memory_config=memory_config)
        ttnn.deallocate(har_bct)
        return har_nlc

    def forward(
        self,
        x_nlc: ttnn.Tensor,
        s_bs: ttnn.Tensor,
        f0: ttnn.Tensor,
        *,
        sinegen_rand_ini: Optional[ttnn.Tensor] = None,
        sinegen_noise_raw: Optional[ttnn.Tensor] = None,
        source_noise_raw: Optional[ttnn.Tensor] = None,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """
        Args:
            x_nlc: ``[B, T_x, C_in]`` initial spectral input (``C_in == upsample_initial_channel``).
            s_bs: ``[B, style_dim]`` style embedding (consumed by AdaIN inside the resblocks).
            f0: ``[B, T_f0]`` or ``[B, T_f0, 1]`` fundamental frequency (``T_f0 == T_x``).

        Returns:
            ``[B, 1, audio_len]`` reconstructed audio. ``audio_len = T_x * prod(upsample_rates) * hop_size``.
        """
        p = self.params
        ck = self.compute_kernel_config

        har_nlc = self._harmonic_source_path(
            f0,
            sinegen_rand_ini=sinegen_rand_ini,
            sinegen_noise_raw=sinegen_noise_raw,
            source_noise_raw=source_noise_raw,
            memory_config=memory_config,
        )
        # ``har_nlc`` shape: [B, F, 2K]

        # Propagate the harmonic source's dtype (fp32 by default) through the network so the
        # downstream convs don't bf16-quantize the STFT outputs before noise_conv reads them.
        target_dtype = har_nlc.dtype
        if x_nlc.dtype != target_dtype:
            x_cast = ttnn.typecast(x_nlc, target_dtype, memory_config=memory_config)
            x = x_cast
        else:
            x = x_nlc
        for i, stage in enumerate(p.stages):
            x_act = ttnn.leaky_relu(x, negative_slope=0.1, memory_config=memory_config)
            if x_act is not x:
                ttnn.deallocate(x)
            x = x_act

            x_source = tt_conv1d_nlc(
                x_nlc=har_nlc,
                params=stage.noise_conv,
                device=self.device,
                compute_config=ck,
                memory_config=memory_config,
                preserve_input_dtype=True,
            )
            x_source = self._noise_res[i].forward(x_source, s_bs, memory_config=memory_config)

            x_up = tt_conv_transpose1d_nlc(
                x_nlc=x,
                params=stage.ups,
                device=self.device,
                compute_config=ck,
                memory_config=memory_config,
            )
            ttnn.deallocate(x)
            x = x_up

            if i == p.num_upsamples - 1:
                x_padded = _reflection_pad_left_1_nlc(x, memory_config=memory_config)
                ttnn.deallocate(x)
                x = x_padded

            x_sum = ttnn.add(x, x_source, memory_config=memory_config)
            ttnn.deallocate(x)
            ttnn.deallocate(x_source)
            x = x_sum

            xs: Optional[ttnn.Tensor] = None
            for resblk in self._resblocks[i]:
                r = resblk.forward(x, s_bs, memory_config=memory_config)
                if xs is None:
                    xs = r
                else:
                    new_xs = ttnn.add(xs, r, memory_config=memory_config)
                    ttnn.deallocate(xs)
                    ttnn.deallocate(r)
                    xs = new_xs
            ttnn.deallocate(x)
            x = ttnn.multiply(xs, 1.0 / p.num_kernels, memory_config=memory_config)
            ttnn.deallocate(xs)

        ttnn.deallocate(har_nlc)

        x_act = ttnn.leaky_relu(x, negative_slope=0.01, memory_config=memory_config)
        ttnn.deallocate(x)
        x = x_act

        # ``conv_post`` returns NLC ``[B, T_out, n_fft + 2]``.
        x_post = tt_conv1d_nlc(
            x_nlc=x,
            params=p.conv_post,
            device=self.device,
            compute_config=ck,
            memory_config=memory_config,
            preserve_input_dtype=True,
        )
        ttnn.deallocate(x)

        # ``spec = exp(x[:, :K, :])``, ``phase = sin(x[:, K:, :])`` along the channel axis (last in NLC).
        K = p.post_n_fft // 2 + 1
        B = int(x_post.shape[0])
        T_post = int(x_post.shape[1])
        spec_nlc = ttnn.slice(x_post, [0, 0, 0], [B, T_post, K], [1, 1, 1], memory_config=memory_config)
        phase_nlc = ttnn.slice(
            x_post,
            [0, 0, K],
            [B, T_post, 2 * K],
            [1, 1, 1],
            memory_config=memory_config,
        )
        ttnn.deallocate(x_post)

        spec_nlc = ttnn.exp(spec_nlc, memory_config=memory_config)
        phase_nlc = ttnn.sin(phase_nlc, memory_config=memory_config)

        # iSTFT expects BCT-style ``[B, K, F]``; permute from NLC.
        spec_bct = ttnn.permute(spec_nlc, (0, 2, 1), memory_config=memory_config)
        phase_bct = ttnn.permute(phase_nlc, (0, 2, 1), memory_config=memory_config)
        ttnn.deallocate(spec_nlc)
        ttnn.deallocate(phase_nlc)

        audio = self._stft.inverse(spec_bct, phase_bct)  # [B, 1, audio_len]
        ttnn.deallocate(spec_bct)
        ttnn.deallocate(phase_bct)
        return audio

    __call__ = forward
