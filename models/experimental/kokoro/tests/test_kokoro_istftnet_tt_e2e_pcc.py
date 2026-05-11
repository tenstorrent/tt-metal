# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end PCC: full TTNN ``KokoroDecoderTt`` vs PyTorch ``Decoder`` (``disable_complex=True``).

    pytest models/experimental/kokoro/tests/test_kokoro_istftnet_tt_e2e_pcc.py --confcutdir=models/experimental/kokoro -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.experimental.kokoro.tt import KokoroDecoderTt, preprocess_kokoro_decoder_tt_parameters


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


def test_kokoro_decoder_tt_e2e_waveform_pcc(ttnn_device):
    """Full decoder on TTNN vs PyTorch reference waveform (deterministic ``m_source``)."""
    dec_ref = load_decoder_from_huggingface(device="cpu", disable_complex=True).decoder
    dec_tt = load_decoder_from_huggingface(device="cpu", disable_complex=True).decoder

    batch, time_asr = 1, 8
    tf = 2 * time_asr
    torch.manual_seed(42)
    dim_in = dec_ref.asr_res[0].in_channels
    asr = torch.randn(batch, dim_in, time_asr, dtype=torch.float32)
    f0_curve = torch.randn(batch, tf, dtype=torch.float32) * 100.0 + 120.0
    n = torch.randn(batch, tf, dtype=torch.float32)
    s = torch.randn(batch, 128, dtype=torch.float32)

    def zeros_rand(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
        return torch.zeros(*args, **kwargs)

    def zeros_randn(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
        return torch.zeros(*args, **kwargs)

    def zeros_randn_like(t, **kwargs):
        return torch.zeros_like(t)

    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        with torch.no_grad():
            y_ref = dec_ref(asr, f0_curve, n, s)

    params = preprocess_kokoro_decoder_tt_parameters(
        dec_tt,
        ttnn_device,
        f0_coarse_time=tf,
        disable_complex=True,
    )
    tt_dec = KokoroDecoderTt(ttnn_device, params)
    l1 = ttnn.L1_MEMORY_CONFIG

    asr_tt = ttnn.from_torch(
        asr,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )
    f0_tt = ttnn.from_torch(
        f0_curve.unsqueeze(-1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )
    n_tt = ttnn.from_torch(
        n.unsqueeze(-1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )
    s_tt = ttnn.from_torch(
        s,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )

    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        y_tt = tt_dec(asr_tt, f0_tt, n_tt, s_tt, deterministic=True)

    y_hat = ttnn.to_torch(y_tt).reshape(y_ref.shape)
    assert y_hat.shape == y_ref.shape
    assert torch.isfinite(y_hat).all()
    ok, p = comp_pcc(y_ref, y_hat, pcc=0.90)
    print(f"decoder_tt e2e PCC={p:.6f} pass={ok}")
    assert ok, f"E2E waveform PCC {p} expected >= 0.90"
