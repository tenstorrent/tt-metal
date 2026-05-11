# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC: TTNN ``KokoroDecoderBody`` (encode + decode) vs PyTorch ``Decoder`` for the same
``asr``, ``F0``, ``N``, ``s`` (front conv outputs fed identically to both paths).

    pytest models/experimental/kokoro/tests/test_kokoro_decoder_body_pcc.py --confcutdir=models/experimental/kokoro -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.experimental.kokoro.tt import (
    KokoroDecoderBody,
    preprocess_kokoro_decoder_body_parameters,
)


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def kokoro_decoder_cpu_disable_complex():
    return load_decoder_from_huggingface(device="cpu", disable_complex=True)


def test_kokoro_decoder_body_pcc(ttnn_device, kokoro_decoder_cpu_disable_complex):
    dec = kokoro_decoder_cpu_disable_complex.decoder
    batch, time_asr = 2, 16
    tf = 2 * time_asr
    torch.manual_seed(0)
    dim_in = dec.asr_res[0].in_channels
    asr = torch.randn(batch, dim_in, time_asr, dtype=torch.float32)
    f0_curve = torch.randn(batch, tf, dtype=torch.float32) * 100.0 + 120.0
    n = torch.randn(batch, tf, dtype=torch.float32)
    s = torch.randn(batch, 128, dtype=torch.float32)

    with torch.no_grad():
        f0 = dec.F0_conv(f0_curve.unsqueeze(1))
        n_b = dec.N_conv(n.unsqueeze(1))
        asr_res = dec.asr_res(asr)
        x0 = torch.cat([asr, f0, n_b], dim=1)
        x = dec.encode(x0, s)
        res = True
        for block in dec.decode:
            if res:
                x = torch.cat([x, asr_res, f0, n_b], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        x_ref = x

    body_p = preprocess_kokoro_decoder_body_parameters(dec, ttnn_device)
    body = KokoroDecoderBody(ttnn_device, body_p)
    l1 = ttnn.L1_MEMORY_CONFIG

    x0_tt = ttnn.from_torch(
        x0,
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
    asr_res_tt = ttnn.from_torch(
        asr_res,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )
    f0_tt = ttnn.from_torch(
        f0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )
    n_tt = ttnn.from_torch(
        n_b,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )

    x_tt = body(x0_tt, s_tt, asr_res_tt, f0_tt, n_tt)
    x_hat = ttnn.to_torch(x_tt).reshape(x_ref.shape)
    assert x_hat.shape == x_ref.shape
    assert int(x_hat.shape[2]) == 2 * time_asr

    ok, p = comp_pcc(x_ref, x_hat, pcc=0.99)
    print(f"decoder_body PCC={p:.6f} pass={ok}")
    assert ok, f"decoder body PCC {p} expected >= 0.99"
