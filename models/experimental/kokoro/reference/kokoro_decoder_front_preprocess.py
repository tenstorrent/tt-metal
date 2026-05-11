# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host preprocessing for TTNN ``Decoder`` front convs: ``F0_conv``, ``N_conv``, ``asr_res``."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import ttnn


def _safe_remove_weight_norm(m: nn.Module) -> None:
    try:
        nn_utils.remove_weight_norm(m)
    except ValueError:
        pass


def _stride_conv1d_spec(conv: nn.Conv1d) -> dict[str, Any]:
    """Spec dict for ``ttnn_kokoro_generator._StridedNoiseConv1d`` (conv2d-on-time)."""
    w = conv.weight.data.unsqueeze(-1).contiguous()
    return {
        "weight": ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
        "bias": None
        if conv.bias is None
        else ttnn.from_torch(
            torch.reshape(conv.bias.data, (1, 1, 1, conv.out_channels)),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "stride": int(conv.stride[0]),
        "kernel_size": int(conv.kernel_size[0]),
        "padding": int(conv.padding[0]),
        "in_channels": int(conv.in_channels),
        "out_channels": int(conv.out_channels),
    }


def preprocess_kokoro_decoder_front_parameters(decoder: nn.Module, device) -> dict[str, Any]:
    """
    Args:
        decoder: ``kokoro_istftnet.Decoder`` (``F0_conv``, ``N_conv``, ``asr_res`` only).

    Returns:
        Dict with keys ``f0_conv``, ``n_conv``, ``asr_res`` for ``KokoroDecoderFront``.
    """
    _safe_remove_weight_norm(decoder.F0_conv)
    _safe_remove_weight_norm(decoder.N_conv)
    _safe_remove_weight_norm(decoder.asr_res[0])

    return {
        "f0_conv": _stride_conv1d_spec(decoder.F0_conv),
        "n_conv": _stride_conv1d_spec(decoder.N_conv),
        "asr_res": _stride_conv1d_spec(decoder.asr_res[0]),
    }
