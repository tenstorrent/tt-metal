# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Staged PCC for Kokoro generator: STFT on ref har, noise_conv (isolates E2E drift)."""

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.kokoro_generator_preprocess import (
    _safe_remove_weight_norm,
    preprocess_kokoro_generator_parameters,
)
from models.experimental.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.experimental.kokoro.tt import KokoroConvStft
from models.experimental.kokoro.tt.ttnn_kokoro_generator import _StridedNoiseConv1d


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def kokoro_decoder_cpu_disable_complex():
    return load_decoder_from_huggingface(device="cpu", disable_complex=True)


def _gen_inputs(dec, batch: int, time_asr: int, seed: int):
    torch.manual_seed(seed)
    dim_in = dec.asr_res[0].in_channels
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


def test_generator_stft_pcc_reference_har(ttnn_device, kokoro_decoder_cpu_disable_complex):
    """``KokoroConvStft`` matches PyTorch ``CustomSTFT`` on the same reference harmonic waveform."""
    dec = kokoro_decoder_cpu_disable_complex.decoder
    gen = dec.generator
    time_asr = 8
    _, _, f0_curve = _gen_inputs(dec, 1, time_asr, seed=42)
    f0_up_t = f0_curve.shape[1] * int(round(float(gen.f0_upsamp.scale_factor)))

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
            f0_u = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
            har_t, _, _ = gen.m_source(f0_u)
            har_t = har_t.transpose(1, 2).squeeze(1)
            mag_r, ph_r = gen.stft.transform(har_t)
            har_cat_ref = torch.cat([mag_r, ph_r], dim=1)

    params = preprocess_kokoro_generator_parameters(gen, ttnn_device, f0_upsampled_time=f0_up_t, disable_complex=True)
    stft = KokoroConvStft(ttnn_device, params["stft"])
    har_rm = ttnn.from_torch(
        har_t.unsqueeze(1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    mag_tt, ph_tt = stft.transform(har_rm)
    ttnn.deallocate(har_rm)
    har_cat_tt = ttnn.concat([mag_tt, ph_tt], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(ph_tt)
    har_hat = ttnn.to_torch(har_cat_tt).reshape(har_cat_ref.shape)
    ttnn.deallocate(har_cat_tt)

    nb = int(mag_r.shape[1])
    ok_mag, p_mag = comp_pcc(mag_r, har_hat[:, :nb, :], pcc=0.999)
    ok_ph, p_ph = comp_pcc(ph_r, har_hat[:, nb:, :], pcc=0.85)
    ok_full, p_full = comp_pcc(har_cat_ref, har_hat, pcc=0.86)
    print(f"STFT(ref har) mag PCC={p_mag:.4f} phase PCC={p_ph:.4f} full PCC={p_full:.4f}")
    assert ok_mag and ok_ph, f"STFT split PCC mag={p_mag} ph={p_ph}"
    assert ok_full, f"STFT concat PCC {p_full}"


@pytest.mark.parametrize("noise_idx", [0, 1])
def test_generator_noise_conv_pcc(ttnn_device, kokoro_decoder_cpu_disable_complex, noise_idx: int):
    """Each ``noise_convs[i]`` matches PyTorch on the same ``har_cat``."""
    dec = kokoro_decoder_cpu_disable_complex.decoder
    gen = dec.generator
    time_asr = 8
    _, _, f0_curve = _gen_inputs(dec, 1, time_asr, seed=42)
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

    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        with torch.no_grad():
            f0_u = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
            har_t, _, _ = gen.m_source(f0_u)
            har_t = har_t.transpose(1, 2).squeeze(1)
            mag_r, ph_r = gen.stft.transform(har_t)
            har_cat = torch.cat([mag_r, ph_r], dim=1)
            F = har_cat.shape[2]

    for c in gen.noise_convs:
        _safe_remove_weight_norm(c)
    params = preprocess_kokoro_generator_parameters(gen, ttnn_device, f0_upsampled_time=f0_up_t, disable_complex=True)

    har_tt = ttnn.from_torch(
        har_cat,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    m = gen.noise_convs[noise_idx]
    spec = params["noise_convs"][noise_idx]
    tt_c = _StridedNoiseConv1d(ttnn_device, spec)
    with torch.no_grad():
        y_ref = m(har_cat)
    y_tt = tt_c(har_tt, 1, F)
    y_hat = ttnn.to_torch(y_tt).reshape(y_ref.shape)
    ok, p = comp_pcc(y_ref, y_hat, pcc=0.98)
    print(f"noise_conv[{noise_idx}] PCC={p:.6f} out={tuple(y_ref.shape)}")
    assert ok, f"noise_conv[{noise_idx}] PCC {p}"
    ttnn.deallocate(y_tt)
    ttnn.deallocate(har_tt)
