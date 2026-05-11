# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN ``Decoder`` front blocks: ``F0_conv``, ``N_conv``, ``asr_res`` (stride / 1×1 conv1d)."""

from __future__ import annotations

from typing import Any

import ttnn

from .ttnn_kokoro_generator import _StridedNoiseConv1d


class KokoroDecoderFront:
    """
    Matches ``Decoder.forward`` preprocessing: ``F0_conv``, ``N_conv``, ``asr_res``.

    Inputs are ``(B, C, L)`` float32 TILE on device, same layout as other Kokoro TT conv paths.
    """

    def __init__(self, device, parameters: dict[str, Any]):
        self.device = device
        self._f0 = _StridedNoiseConv1d(device, parameters["f0_conv"])
        self._n = _StridedNoiseConv1d(device, parameters["n_conv"])
        self._asr_res_conv = _StridedNoiseConv1d(device, parameters["asr_res"])

    def f0_conv(self, f0_b1t: ttnn.Tensor, batch_size: int, input_len: int) -> ttnn.Tensor:
        """``f0_b1t``: ``(B, 1, Tf)`` — same as ``F0_curve.unsqueeze(1)``."""
        return self._f0(f0_b1t, batch_size, input_len)

    def n_conv(self, n_b1t: ttnn.Tensor, batch_size: int, input_len: int) -> ttnn.Tensor:
        """``n_b1t``: ``(B, 1, Tf)`` — same as ``N.unsqueeze(1)``."""
        return self._n(n_b1t, batch_size, input_len)

    def asr_res(self, asr_bct: ttnn.Tensor, batch_size: int, input_len: int) -> ttnn.Tensor:
        """``asr_bct``: ``(B, 512, T_asr)`` ASR features."""
        return self._asr_res_conv(asr_bct, batch_size, input_len)
