# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side preprocessing for TTNN Kokoro STFT (conv DFT weights; same as `CustomSTFT`)."""

from __future__ import annotations

from typing import Any


import ttnn

from .kokoro_istftnet import CustomSTFT


def preprocess_kokoro_conv_stft_parameters(
    device,
    *,
    filter_length: int = 800,
    hop_length: int = 200,
    win_length: int = 800,
    center: bool = True,
    pad_mode: str = "replicate",
) -> dict[str, Any]:
    """
    Build TTNN conv weights matching `CustomSTFT` in `kokoro_istftnet.py`.

    Kokoro checkpoints list `TorchSTFT` in the module tree; the conv formulation is the
    deterministic, complex-free path used for TT bring-up and matches `CustomSTFT` numerics.
    """
    if pad_mode != "replicate":
        raise ValueError(f"Only pad_mode='replicate' is supported, got {pad_mode!r}")

    m = CustomSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        pad_mode=pad_mode,
    )

    def conv_weight_rm(name: str) -> ttnn.Tensor:
        w = getattr(m, name).data.unsqueeze(-1).contiguous()
        return ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

    pad_len = m.n_fft // 2 if center else 0
    return {
        "weight_forward_real": conv_weight_rm("weight_forward_real"),
        "weight_forward_imag": conv_weight_rm("weight_forward_imag"),
        "weight_backward_real": conv_weight_rm("weight_backward_real"),
        "weight_backward_imag": conv_weight_rm("weight_backward_imag"),
        "n_fft": int(m.n_fft),
        "hop_length": int(m.hop_length),
        "freq_bins": int(m.freq_bins),
        "center": bool(center),
        "pad_len": int(pad_len),
        "eps": 1e-14,
    }
