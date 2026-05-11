# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full TTNN Kokoro ISTFTNet ``Decoder`` (front + body + generator); ``m_source`` uses device ``KokoroTtnnSineGen``."""

from __future__ import annotations

from typing import Any

import ttnn

from ..reference.kokoro_decoder_front_preprocess import preprocess_kokoro_decoder_front_parameters
from ..reference.kokoro_generator_preprocess import preprocess_kokoro_generator_parameters
from .ttnn_kokoro_decoder_body import KokoroDecoderBody, preprocess_kokoro_decoder_body_parameters
from .ttnn_kokoro_decoder_front import KokoroDecoderFront
from .ttnn_kokoro_generator import KokoroGenerator


def preprocess_kokoro_decoder_tt_parameters(
    decoder: Any,
    device,
    *,
    f0_coarse_time: int,
    disable_complex: bool = False,
) -> dict[str, Any]:
    """
    Preprocess full ``Decoder`` for :class:`KokoroDecoderTt` / :class:`KokoroIstftNetTt`.

    Args:
        decoder: ``kokoro_istftnet.Decoder``.
        f0_coarse_time: Length ``Tf`` of the coarse F0 curve (``F0_curve.shape[1]``) before ``f0_upsamp``.
        disable_complex: Passed through to generator / STFT preprocess (``CustomSTFT`` path).

    Returns:
        Dict with ``front``, ``body``, ``generator`` keys. Mutates ``decoder`` submodules in place
        (weight norm removal), same as the individual preprocess helpers.
    """
    gen = decoder.generator
    f0_scale = (
        float(gen.f0_upsamp.scale_factor)
        if isinstance(gen.f0_upsamp.scale_factor, (int, float))
        else float(gen.f0_upsamp.scale_factor[0])
    )
    sf_int = int(round(f0_scale))
    if abs(f0_scale - float(sf_int)) > 1e-6:
        raise ValueError(f"TTNN decoder expects integer f0 upsample scale, got {f0_scale!r}")
    f0_upsampled_time = int(f0_coarse_time) * sf_int

    front_p = preprocess_kokoro_decoder_front_parameters(decoder, device)
    body_p = preprocess_kokoro_decoder_body_parameters(decoder, device)
    gen_p = preprocess_kokoro_generator_parameters(
        gen,
        device,
        f0_upsampled_time=f0_upsampled_time,
        disable_complex=disable_complex,
    )
    return {"front": front_p, "body": body_p, "generator": gen_p}


class KokoroDecoderTt:
    """
    TTNN ``Decoder.forward`` (``F0_conv`` / ``N_conv`` / ``asr_res`` → ``encode`` / ``decode`` → ``generator``).

    Tensor layout matches existing Kokoro TT pieces: ``asr`` ``(B, 512, T_asr)``,
    ``f0_curve`` / ``n`` ``(B, Tf, 1)``, ``s`` ``(B, style_dim)``, float32 TILE, L1.
    """

    def __init__(self, device, parameters: dict[str, Any]):
        self.device = device
        self.front = KokoroDecoderFront(device, parameters["front"])
        self.body = KokoroDecoderBody(device, parameters["body"])
        self.generator = KokoroGenerator(device, parameters["generator"])

    def __call__(
        self,
        asr: ttnn.Tensor,
        f0_curve: ttnn.Tensor,
        n: ttnn.Tensor,
        s: ttnn.Tensor,
        *,
        deterministic: bool = False,
    ) -> ttnn.Tensor:
        """
        Args:
            asr: ``(B, 512, T_asr)``.
            f0_curve: ``(B, Tf, 1)`` coarse F0 (same as ``KokoroGenerator`` ``f0_coarse`` layout).
            n: ``(B, Tf, 1)`` noise curve for ``N_conv``.
            s: ``(B, style_dim)`` (128 for Kokoro).
        """
        l1 = ttnn.L1_MEMORY_CONFIG
        asr = ttnn.to_memory_config(asr, l1)
        if asr.dtype != ttnn.float32:
            asr = ttnn.typecast(asr, ttnn.float32, memory_config=l1)
        f0_curve = ttnn.to_memory_config(f0_curve, l1)
        if f0_curve.dtype != ttnn.float32:
            f0_curve = ttnn.typecast(f0_curve, ttnn.float32, memory_config=l1)
        n = ttnn.to_memory_config(n, l1)
        if n.dtype != ttnn.float32:
            n = ttnn.typecast(n, ttnn.float32, memory_config=l1)
        s = ttnn.to_memory_config(s, l1)
        if s.dtype != ttnn.float32:
            s = ttnn.typecast(s, ttnn.float32, memory_config=l1)

        bsz = int(asr.shape[0])
        t_asr = int(asr.shape[2])
        tf = int(f0_curve.shape[1])

        f0_b1t = ttnn.permute(f0_curve, [0, 2, 1], memory_config=l1)
        n_b1t = ttnn.permute(n, [0, 2, 1], memory_config=l1)

        f0_f = self.front.f0_conv(f0_b1t, bsz, tf)
        n_f = self.front.n_conv(n_b1t, bsz, tf)
        asr_res = self.front.asr_res(asr, bsz, t_asr)
        x0 = ttnn.concat([asr, f0_f, n_f], dim=1, memory_config=l1)
        x = self.body(x0, s, asr_res, f0_f, n_f)
        return self.generator(x, s, f0_curve, deterministic=deterministic)


class KokoroIstftNetTt:
    """TTNN analogue of ``KokoroIstftNet``: ``ref_s[:, :128]`` then :class:`KokoroDecoderTt`."""

    def __init__(self, device, parameters: dict[str, Any]):
        self.device = device
        self.decoder = KokoroDecoderTt(device, parameters)

    def __call__(
        self,
        *,
        asr: ttnn.Tensor,
        f0_pred: ttnn.Tensor,
        n_pred: ttnn.Tensor,
        ref_s: ttnn.Tensor,
        deterministic: bool = False,
    ) -> ttnn.Tensor:
        b = int(ref_s.shape[0])
        sd = min(128, int(ref_s.shape[1]))
        s = ttnn.slice(ref_s, [0, 0], [b, sd], memory_config=ttnn.L1_MEMORY_CONFIG)
        if s.dtype != ttnn.float32:
            s = ttnn.typecast(s, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        return self.decoder(asr, f0_pred, n_pred, s, deterministic=deterministic)
