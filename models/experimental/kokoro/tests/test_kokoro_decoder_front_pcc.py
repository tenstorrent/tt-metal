# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests: TTNN ``KokoroDecoderFront`` (``F0_conv``, ``N_conv``, ``asr_res``) vs PyTorch.

    cd <tt-metal-root>
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    pytest models/experimental/kokoro/tests/test_kokoro_decoder_front_pcc.py --confcutdir=models/experimental/kokoro -v
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
from models.experimental.kokoro.tt import KokoroDecoderFront, preprocess_kokoro_decoder_front_parameters


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def kokoro_decoder_cpu_disable_complex():
    return load_decoder_from_huggingface(device="cpu", disable_complex=True)


def test_decoder_front_f0_n_asr_pcc(ttnn_device, kokoro_decoder_cpu_disable_complex):
    dec = kokoro_decoder_cpu_disable_complex.decoder
    batch, time_asr = 2, 16
    tf = 2 * time_asr
    torch.manual_seed(0)
    dim_in = dec.asr_res[0].in_channels
    asr = torch.randn(batch, dim_in, time_asr, dtype=torch.float32)
    f0_curve = torch.randn(batch, tf, dtype=torch.float32) * 100.0 + 120.0
    n = torch.randn(batch, tf, dtype=torch.float32)

    with torch.no_grad():
        f0_ref = dec.F0_conv(f0_curve.unsqueeze(1))
        n_ref = dec.N_conv(n.unsqueeze(1))
        asr_res_ref = dec.asr_res(asr)

    params = preprocess_kokoro_decoder_front_parameters(dec, ttnn_device)
    front = KokoroDecoderFront(ttnn_device, params)
    l1 = ttnn.L1_MEMORY_CONFIG

    f0_in = ttnn.from_torch(
        f0_curve.unsqueeze(1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )
    f0_tt = front.f0_conv(f0_in, batch, tf)
    f0_hat = ttnn.to_torch(f0_tt).reshape(f0_ref.shape)
    ttnn.deallocate(f0_in)
    ttnn.deallocate(f0_tt)

    n_in = ttnn.from_torch(
        n.unsqueeze(1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )
    n_tt = front.n_conv(n_in, batch, tf)
    n_hat = ttnn.to_torch(n_tt).reshape(n_ref.shape)
    ttnn.deallocate(n_in)
    ttnn.deallocate(n_tt)

    asr_in = ttnn.from_torch(
        asr,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=l1,
    )
    asr_tt = front.asr_res(asr_in, batch, time_asr)
    asr_hat = ttnn.to_torch(asr_tt).reshape(asr_res_ref.shape)
    ttnn.deallocate(asr_in)
    ttnn.deallocate(asr_tt)

    min_pcc = 0.99
    for name, ref, hat in (
        ("F0_conv", f0_ref, f0_hat),
        ("N_conv", n_ref, n_hat),
        ("asr_res", asr_res_ref, asr_hat),
    ):
        ok, p = comp_pcc(ref, hat, pcc=min_pcc)
        print(f"{name} PCC={p:.6f} pass={ok}")
        assert ok, f"{name} PCC {p} expected >= {min_pcc}"
