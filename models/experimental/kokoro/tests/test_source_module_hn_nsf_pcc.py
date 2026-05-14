# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN ``SourceModuleHnNSF`` vs PyTorch (deterministic path, zeros for all RNG).

``SourceModuleHnNSF`` always uses device ``KokoroTtnnSineGen`` — no CPU SineGen fallback.

Setup:
    cd <tt-metal-root>
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

Usage:
    pytest models/experimental/kokoro/tests/test_source_module_hn_nsf_pcc.py --confcutdir=models/experimental/kokoro -v
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
from models.experimental.kokoro.tt import SourceModuleHnNSF


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def kokoro_decoder_cpu():
    return load_decoder_from_huggingface(device="cpu")


def test_source_module_hn_nsf_deterministic_pcc(ttnn_device, kokoro_decoder_cpu):
    """Device ``KokoroTtnnSineGen`` + TTNN linear + tanh vs PyTorch (deterministic zeros)."""
    m = kokoro_decoder_cpu.decoder.generator.m_source
    ups = int(m.l_sin_gen.upsample_scale)
    time_len = ups * 2
    batch = 2

    torch.manual_seed(23)
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
            har_ref, noise_ref, uv_ref = m(f0)

    params = preprocess_source_module_hn_nsf_parameters(m, ttnn_device, time_len)
    tt_mod = SourceModuleHnNSF(ttnn_device, params)
    f0_tt = ttnn.from_torch(
        f0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    har_tt, noise_tt, uv_tt = tt_mod(f0_tt, deterministic=True)

    har_hat = ttnn.to_torch(har_tt).reshape(har_ref.shape)
    noise_hat = ttnn.to_torch(noise_tt).reshape(noise_ref.shape)
    uv_hat = ttnn.to_torch(uv_tt).reshape(uv_ref.shape)

    # Device SineGen + TTNN linear + tanh (merge linear uses same HiFi4+fp32 dest as generator convs).
    min_pcc = {"har_source": 0.99, "noise_merge": 0.99, "uv": 0.99}
    for name, ref, hat in (
        ("har_source", har_ref, har_hat),
        ("noise_merge", noise_ref, noise_hat),
        ("uv", uv_ref, uv_hat),
    ):
        pcc = min_pcc[name]
        ok, p = comp_pcc(ref, hat, pcc=pcc)
        print(f"{name} PCC={p:.6f} pass={ok} (min {pcc})")
        assert ok, f"{name} PCC {p} expected >= {pcc}"
