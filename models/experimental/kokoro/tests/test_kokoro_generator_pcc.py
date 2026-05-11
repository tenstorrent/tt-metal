# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN ``KokoroGenerator`` vs PyTorch: waveform PCC, shape, finiteness (``disable_complex=True``).

``KokoroGenerator`` uses device ``KokoroTtnnSineGen`` by default; set ``use_torch_sinegen=True`` in
``preprocess_kokoro_generator_parameters`` to run the reference PyTorch ``SineGen`` on CPU for harmonics
(see ``test_kokoro_generator_waveform_pcc_sinegen_modes``). The full waveform PCC vs PyTorch is lower with
pure device SineGen under deterministic zeros; tighter checks live in ``test_ttnn_sinegen_pcc.py`` and
``test_source_module_hn_nsf_pcc.py``. Upsampling PCC also lives in ``test_kokoro_generator_ups_pcc.py``.

Decoder prefix: TTNN ``F0_conv`` / ``N_conv`` / ``asr_res``, then TTNN ``encode`` + ``decode`` (:class:`KokoroDecoderBody`).
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
from models.experimental.kokoro.tests.kokoro_generator_pcc_inputs import decoder_tensors_for_generator
from models.experimental.kokoro.tt import (
    KokoroDecoderBody,
    KokoroDecoderFront,
    KokoroGenerator,
    preprocess_kokoro_decoder_body_parameters,
    preprocess_kokoro_decoder_front_parameters,
    preprocess_kokoro_generator_parameters,
)


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def kokoro_decoder_cpu_disable_complex():
    return load_decoder_from_huggingface(device="cpu", disable_complex=True)


@pytest.mark.parametrize("use_torch_sinegen,min_pcc", [(False, 0.61), (True, 0.85)])
def test_kokoro_generator_waveform_pcc_sinegen_modes(
    ttnn_device, kokoro_decoder_cpu_disable_complex, use_torch_sinegen: bool, min_pcc: float
):
    """Generator waveform vs PyTorch ref; ``use_torch_sinegen`` selects harmonic backend."""
    dec = kokoro_decoder_cpu_disable_complex.decoder
    gen = dec.generator
    time_asr = 8
    batch = 1
    front_p = preprocess_kokoro_decoder_front_parameters(dec, ttnn_device)
    tt_front = KokoroDecoderFront(ttnn_device, front_p)
    body_p = preprocess_kokoro_decoder_body_parameters(dec, ttnn_device)
    tt_body = KokoroDecoderBody(ttnn_device, body_p)

    x, s, f0_curve = decoder_tensors_for_generator(
        dec, batch, time_asr, seed=42, tt_front=tt_front, tt_body=tt_body, ttnn_device=ttnn_device
    )

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
        use_torch_sinegen=use_torch_sinegen,
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
    ok, p = comp_pcc(y_ref, y_hat, pcc=min_pcc)
    mode = "torch_cpu_sinegen" if use_torch_sinegen else "ttnn_device_sinegen"
    print(f"generator waveform PCC mode={mode} pcc={p:.6f} pass={ok} (min {min_pcc})")
    assert ok, f"generator waveform PCC {p} (mode={mode}, min {min_pcc})"
