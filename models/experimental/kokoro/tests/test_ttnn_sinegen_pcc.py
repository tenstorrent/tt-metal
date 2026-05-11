# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: ``KokoroTtnnSineGen`` vs PyTorch ``SineGen`` (deterministic RNG zeros).

Usage:
    pytest models/experimental/kokoro/tests/test_ttnn_sinegen_pcc.py --confcutdir=models/experimental/kokoro -v
"""

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.experimental.kokoro.reference.kokoro_source_module_preprocess import (
    preprocess_source_module_hn_nsf_parameters,
)
from models.experimental.kokoro.tt.ttnn_sinegen import KokoroTtnnSineGen


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def kokoro_decoder_cpu():
    return load_decoder_from_huggingface(device="cpu")


def test_ttnn_sinegen_deterministic_pcc(ttnn_device, kokoro_decoder_cpu):
    sg = kokoro_decoder_cpu.decoder.generator.m_source.l_sin_gen
    ups = int(sg.upsample_scale)
    time_len = ups * 2
    batch = 2
    torch.manual_seed(0)
    f0 = torch.rand(batch, time_len, 1, dtype=torch.float32) * 400.0 + 80.0

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
            sw_ref, uv_ref, noise_ref = sg(f0)

    m = kokoro_decoder_cpu.decoder.generator.m_source
    params = preprocess_source_module_hn_nsf_parameters(m, ttnn_device, time_len)
    tt_sg = KokoroTtnnSineGen(ttnn_device, params)
    f0_tt = ttnn.from_torch(
        f0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    sw_tt, uv_tt, noise_tt = tt_sg(f0_tt, deterministic=True)

    sw_hat = ttnn.to_torch(sw_tt).reshape(sw_ref.shape)
    uv_hat = ttnn.to_torch(uv_tt).reshape(uv_ref.shape)
    noise_hat = ttnn.to_torch(noise_tt).reshape(noise_ref.shape)

    min_pcc = {"sine_waves": 0.97, "uv": 0.99, "noise": 0.99}
    for name, ref, hat in (
        ("sine_waves", sw_ref, sw_hat),
        ("uv", uv_ref, uv_hat),
        ("noise", noise_ref, noise_hat),
    ):
        ok, p = comp_pcc(ref, hat, pcc=min_pcc[name])
        print(f"{name} PCC={p:.6f} pass={ok} (min {min_pcc[name]})")
        assert ok, f"{name} PCC {p} expected >= {min_pcc[name]}"
