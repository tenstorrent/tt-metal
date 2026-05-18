# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro ``Decoder`` from ``reference/istftnet.py``.

The Decoder prepares harmonic-source conditioning (F0 + noise) before the Generator:

    F0  = F0_conv(F0_curve[:,None])          # [B,1,T_mel]  stride-2 downsample
    N   = N_conv(N_curve[:,None])             # [B,1,T_mel]
    x   = cat([asr, F0, N], dim=channel)     # [B, dim_in+2, T_mel]
    x   = encode(x, s)                       # [B, 1024, T_mel]
    asr_res = asr_res_conv(asr)              # [B, 64,   T_mel]
    for block in decode:                      # first 3: cat+resblk; last: upsample resblk
        if res: x = cat([x, asr_res, F0, N], dim=channel)
        x = block(x, s)
        if block.upsample: res = False
    audio = generator(x, s, F0_curve)        # [B, 1, audio_len]

All activations are **NLC** ``[B, T, C]``; PyTorch appears only in
:func:`preprocess_tt_decoder` (weight upload).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from torch.nn.utils import parametrize

import ttnn

from .tt_adain_resblk_1d import TTAdainResBlk1d, TTAdainResBlk1dParams, preprocess_tt_adain_resblk_1d
from .tt_conv import TTConv1dParams, tt_conv1d_nlc
from .tt_generator import (
    TTGenerator,
    TTGeneratorParams,
    preprocess_tt_generator,
)


# ---------------------------------------------------------------------------
# weight-norm helper (mirrors tt_generator._strip_weight_norm)
# ---------------------------------------------------------------------------


def _to_interleaved(t: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """Move ``t`` to interleaved DRAM if it is sharded (conv1d output is L1-sharded)."""
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return ttnn.to_memory_config(t, memory_config)
    return t


def _strip_wn(m: nn.Module) -> None:
    if parametrize.is_parametrized(m, "weight"):
        parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        return
    try:
        nn.utils.remove_weight_norm(m, "weight")
    except ValueError:
        pass


def _conv1d_to_tt(conv: nn.Conv1d, device, *, weights_dtype) -> TTConv1dParams:
    """Upload a ``nn.Conv1d`` (weight-norm already stripped) for :func:`tt_conv1d_nlc`."""

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
        dilation=int(conv.dilation[0]) if hasattr(conv.dilation, "__getitem__") else int(conv.dilation),
    )


# ---------------------------------------------------------------------------
# params dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTDecoderParams:
    """Device-resident weights for :class:`TTDecoder`."""

    encode: TTAdainResBlk1dParams
    decode: tuple[TTAdainResBlk1dParams, ...]
    F0_conv: TTConv1dParams
    N_conv: TTConv1dParams
    asr_res: TTConv1dParams
    generator: TTGeneratorParams


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def preprocess_tt_decoder(
    module: nn.Module,
    device: ttnn.Device,
    *,
    time_len_asr: int,
    weights_dtype=ttnn.float32,
    conv_weights_dtype=ttnn.float32,
    source_stft_dtype=ttnn.float32,
) -> TTDecoderParams:
    """Upload a reference ``Decoder`` to device for :class:`TTDecoder`.

    Args:
        module: Reference ``Decoder`` (``nn.Module``), already in eval mode.
        device: TT device.
        time_len_asr: Mel-frame count of the ``asr`` input (``T_mel``).  The
            last decode block upsamples by 2×, so the generator receives
            ``time_len_asr * 2`` mel frames as its ``time_len_x``.
        weights_dtype: Dtype for AdaIN / norm weights (default fp32).
        conv_weights_dtype: Dtype for Conv1d / ConvTranspose1d weights (default fp32).
        source_stft_dtype: Dtype for SineGen + STFT weights inside generator (default fp32).
    """
    encode_p = preprocess_tt_adain_resblk_1d(
        module.encode,
        device,
        weights_dtype=weights_dtype,
        conv_weights_dtype=conv_weights_dtype,
    )

    decode_p = tuple(
        preprocess_tt_adain_resblk_1d(
            blk,
            device,
            weights_dtype=weights_dtype,
            conv_weights_dtype=conv_weights_dtype,
        )
        for blk in module.decode
    )

    # F0_conv and N_conv both carry weight_norm
    _strip_wn(module.F0_conv)
    _strip_wn(module.N_conv)
    F0_conv_p = _conv1d_to_tt(module.F0_conv, device, weights_dtype=conv_weights_dtype)
    N_conv_p = _conv1d_to_tt(module.N_conv, device, weights_dtype=conv_weights_dtype)

    # asr_res = nn.Sequential(weight_norm(Conv1d(512, 64, 1)))
    asr_conv = module.asr_res[0]
    _strip_wn(asr_conv)
    asr_res_p = _conv1d_to_tt(asr_conv, device, weights_dtype=conv_weights_dtype)

    # Generator receives x after the decode-upsample block: T_mel * 2 mel frames
    gen_p = preprocess_tt_generator(
        module.generator,
        device,
        time_len_x=time_len_asr * 2,
        weights_dtype=weights_dtype,
        conv_weights_dtype=conv_weights_dtype,
        source_stft_dtype=source_stft_dtype,
    )

    return TTDecoderParams(
        encode=encode_p,
        decode=decode_p,
        F0_conv=F0_conv_p,
        N_conv=N_conv_p,
        asr_res=asr_res_p,
        generator=gen_p,
    )


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------


class TTDecoder:
    """TTNN port of ``Decoder`` (``TorchSTFT`` path; ``disable_complex=True`` is out of scope).

    ``use_torch_stft_fallback`` / ``use_torch_phase_fallback`` are passed through to
    :class:`TTGenerator`; both are required together to achieve PCC > 0.99 on BH hardware
    (see :class:`TTGenerator` and ``test_tt_generator_pcc`` for the rationale).
    """

    def __init__(
        self,
        device: ttnn.Device,
        params: TTDecoderParams,
        *,
        use_torch_stft_fallback: bool = False,
        use_torch_phase_fallback: bool = False,
    ) -> None:
        self.device = device
        self.params = params
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._encode = TTAdainResBlk1d(device, params.encode)
        self._decode = tuple(TTAdainResBlk1d(device, p) for p in params.decode)
        self._generator = TTGenerator(
            device,
            params.generator,
            use_torch_stft_fallback=use_torch_stft_fallback,
            use_torch_phase_fallback=use_torch_phase_fallback,
        )

    def forward(
        self,
        asr_nlc: ttnn.Tensor,
        F0_curve: ttnn.Tensor,
        N_curve: ttnn.Tensor,
        s_bs: ttnn.Tensor,
        *,
        sinegen_rand_ini: Optional[ttnn.Tensor] = None,
        sinegen_noise_raw: Optional[ttnn.Tensor] = None,
        source_noise_raw: Optional[ttnn.Tensor] = None,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """
        Args:
            asr_nlc: ``[B, T_mel, 512]`` ASR features (NLC).
            F0_curve: ``[B, T_f0]`` fundamental frequency (T_f0 = 2 * T_mel).
            N_curve: ``[B, T_f0]`` noise curve.
            s_bs: ``[B, style_dim]`` style embedding.

        Returns:
            ``[B, 1, audio_len]`` reconstructed audio.
        """
        p = self.params
        ck = self.compute_kernel_config
        dev = self.device

        # -- F0/N downsampling --------------------------------------------------
        # Reference: F0 = F0_conv(F0_curve[:,None])  [B,1,T_f0] -stride2-> [B,1,T_mel]
        # In NLC: unsqueeze last dim → [B, T_f0, 1] → conv → [B, T_mel, 1]
        f0_shape = list(F0_curve.shape)
        if len(f0_shape) == 2:
            f0_nlc = ttnn.unsqueeze(F0_curve, 2)  # [B, T_f0, 1]
        else:
            f0_nlc = F0_curve

        n_shape = list(N_curve.shape)
        if len(n_shape) == 2:
            n_nlc = ttnn.unsqueeze(N_curve, 2)  # [B, T_f0, 1]
        else:
            n_nlc = N_curve

        F0_down = _to_interleaved(
            tt_conv1d_nlc(
                x_nlc=f0_nlc,
                params=p.F0_conv,
                device=dev,
                compute_config=ck,
                memory_config=memory_config,
                preserve_input_dtype=True,
            ),
            memory_config,
        )  # [B, T_mel, 1]
        N_down = _to_interleaved(
            tt_conv1d_nlc(
                x_nlc=n_nlc,
                params=p.N_conv,
                device=dev,
                compute_config=ck,
                memory_config=memory_config,
                preserve_input_dtype=True,
            ),
            memory_config,
        )  # [B, T_mel, 1]

        if len(f0_shape) == 2:
            ttnn.deallocate(f0_nlc)
        if len(n_shape) == 2:
            ttnn.deallocate(n_nlc)

        # dtype of the conditioning tensors — cast x back to this after each block so that
        # TTAdainResBlk1d's default bfloat16 output doesn't create a dtype mismatch in concat.
        cond_dtype = F0_down.dtype

        # -- Encode block -------------------------------------------------------
        # x = cat([asr, F0, N], dim=channel)  [B, T_mel, dim_in+2]
        x = ttnn.concat([asr_nlc, F0_down, N_down], dim=2, memory_config=memory_config)
        x = self._encode.forward(x, s_bs, memory_config=memory_config)  # [B, T_mel, 1024]
        if x.dtype != cond_dtype:
            x_cast = ttnn.typecast(x, cond_dtype, memory_config=memory_config)
            ttnn.deallocate(x)
            x = x_cast

        # -- ASR residual projection (1x1 conv) ---------------------------------
        asr_res = _to_interleaved(
            tt_conv1d_nlc(
                x_nlc=asr_nlc,
                params=p.asr_res,
                device=dev,
                compute_config=ck,
                memory_config=memory_config,
                preserve_input_dtype=True,
            ),
            memory_config,
        )  # [B, T_mel, 64]

        # -- Decode blocks ------------------------------------------------------
        res = True
        for blk in self._decode:
            if res:
                x_cat = ttnn.concat([x, asr_res, F0_down, N_down], dim=2, memory_config=memory_config)
                ttnn.deallocate(x)
                x = x_cat
            x = blk.forward(x, s_bs, memory_config=memory_config)
            # Cast back so the next iteration's concat sees a uniform dtype.
            if x.dtype != cond_dtype:
                x_cast = ttnn.typecast(x, cond_dtype, memory_config=memory_config)
                ttnn.deallocate(x)
                x = x_cast
            if blk._params.layer_type != "none":
                res = False

        ttnn.deallocate(asr_res)
        ttnn.deallocate(F0_down)
        ttnn.deallocate(N_down)

        # -- Generator ----------------------------------------------------------
        audio = self._generator.forward(
            x,
            s_bs,
            F0_curve,
            sinegen_rand_ini=sinegen_rand_ini,
            sinegen_noise_raw=sinegen_noise_raw,
            source_noise_raw=source_noise_raw,
            memory_config=memory_config,
        )
        ttnn.deallocate(x)
        return audio

    __call__ = forward
