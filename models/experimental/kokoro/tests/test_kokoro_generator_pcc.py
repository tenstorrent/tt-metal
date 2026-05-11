# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN ``KokoroGenerator`` vs PyTorch: waveform PCC, shape, finiteness (``disable_complex=True``).

``SourceModuleHnNSF`` uses CPU PyTorch for the harmonic source; remaining drift is STFT / AdaIN /
ups / post. Upsampling PCC also lives in ``test_kokoro_generator_ups_pcc.py``.
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
from models.experimental.kokoro.reference.kokoro_generator_preprocess import preprocess_kokoro_generator_parameters
from models.experimental.kokoro.tt import KokoroGenerator


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def kokoro_decoder_cpu_disable_complex():
    return load_decoder_from_huggingface(device="cpu", disable_complex=True)


def _decoder_tensors_for_generator(dec, batch: int, time_asr: int, seed: int):
    """Match ``Decoder.forward`` tensor sizes up to the generator call."""
    torch.manual_seed(seed)
    dim_in = dec.asr_res[0].in_channels
    # ``F0_conv`` / ``N_conv`` stride 2: length ``(Tf + 2 - 3) // 2 + 1`` must equal ``time_asr``.
    # Use ``Tf = 2 * time_asr`` (not ``2 * time_asr - 1``) so the generator harmonic path matches upsampling.
    tf = 2 * time_asr
    asr = torch.randn(batch, dim_in, time_asr, dtype=torch.float32)
    f0_curve = torch.randn(batch, tf, dtype=torch.float32) * 100.0 + 120.0
    n = torch.randn(batch, tf, dtype=torch.float32)
    s = torch.randn(batch, 128, dtype=torch.float32)
    with torch.no_grad():
        f0 = dec.F0_conv(f0_curve.unsqueeze(1))
        n_b = dec.N_conv(n.unsqueeze(1))
        x = torch.cat([asr, f0, n_b], dim=1)
        x = dec.encode(x, s)
        asr_res = dec.asr_res(asr)
        res = True
        for block in dec.decode:
            if res:
                x = torch.cat([x, asr_res, f0, n_b], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
    return x, s, f0_curve


def test_kokoro_generator_forward_smoke(ttnn_device, kokoro_decoder_cpu_disable_complex):
    """Same tensors as PyTorch ref; assert waveform PCC, shape match, and finite TT output."""
    dec = kokoro_decoder_cpu_disable_complex.decoder
    gen = dec.generator
    time_asr = 8
    batch = 1
    x, s, f0_curve = _decoder_tensors_for_generator(dec, batch, time_asr, seed=42)

    sf = int(round(float(gen.f0_upsamp.scale_factor)))
    f0_up_t = f0_curve.shape[1] * sf

    def zeros_rand(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
        return torch.zeros(*args, **kwargs)

    def zeros_randn(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
        return torch.zeros(*args, **kwargs)

    def zeros_randn_like(t, **kwargs):
        return torch.zeros_like(t)

    params = preprocess_kokoro_generator_parameters(
        gen,
        ttnn_device,
        f0_upsampled_time=f0_up_t,
        disable_complex=True,
    )
    tt_gen = KokoroGenerator(ttnn_device, params)

    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        with torch.no_grad():
            y_ref = gen(x, s, f0_curve)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    s_tt = ttnn.from_torch(
        s,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    f0_tt = ttnn.from_torch(
        f0_curve.unsqueeze(-1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    y_tt = tt_gen(x_tt, s_tt, f0_tt, deterministic=True)
    y_hat = ttnn.to_torch(y_tt).reshape(y_ref.shape)
    assert y_hat.shape == y_ref.shape
    assert torch.isfinite(y_hat).all(), "TTNN generator output has non-finite values"
    ok, p = comp_pcc(y_ref, y_hat, pcc=0.90)
    assert ok, f"generator waveform PCC {p} (expected >= 0.90)"
