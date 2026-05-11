# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN ``KokoroGenerator`` vs PyTorch: waveform PCC, shape, finiteness (``disable_complex=True``).

``KokoroGenerator`` uses device ``KokoroTtnnSineGen`` (see ``ttnn_kokoro_generator``). The full
waveform PCC vs PyTorch is dominated by SineGen/STFT numerics under deterministic zeros; tighter
checks live in ``test_ttnn_sinegen_pcc.py`` and ``test_source_module_hn_nsf_pcc.py``. Upsampling PCC
also lives in ``test_kokoro_generator_ups_pcc.py``.

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


def _decoder_tensors_for_generator(
    dec,
    batch: int,
    time_asr: int,
    seed: int,
    *,
    tt_front: KokoroDecoderFront | None = None,
    tt_body: KokoroDecoderBody | None = None,
    ttnn_device=None,
):
    """Match ``Decoder.forward`` tensor sizes up to the generator call.

    When ``tt_front``, ``tt_body``, and ``ttnn_device`` are set, the full prefix through decode runs on TTNN.

    When only ``tt_front`` and ``ttnn_device`` are set, ``encode``/``decode`` stay PyTorch.
    """
    torch.manual_seed(seed)
    dim_in = dec.asr_res[0].in_channels
    # ``F0_conv`` / ``N_conv`` stride 2: length ``(Tf + 2 - 3) // 2 + 1`` must equal ``time_asr``.
    # Use ``Tf = 2 * time_asr`` (not ``2 * time_asr - 1``) so the generator harmonic path matches upsampling.
    tf = 2 * time_asr
    asr = torch.randn(batch, dim_in, time_asr, dtype=torch.float32)
    f0_curve = torch.randn(batch, tf, dtype=torch.float32) * 100.0 + 120.0
    n = torch.randn(batch, tf, dtype=torch.float32)
    s = torch.randn(batch, 128, dtype=torch.float32)
    l1 = ttnn.L1_MEMORY_CONFIG

    with torch.no_grad():
        if tt_front is not None and tt_body is not None and ttnn_device is not None:
            asr_tt = ttnn.from_torch(
                asr,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            f0_in = ttnn.from_torch(
                f0_curve.unsqueeze(1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            f0_tt = tt_front.f0_conv(f0_in, batch, tf)
            ttnn.deallocate(f0_in)
            n_in = ttnn.from_torch(
                n.unsqueeze(1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            n_tt = tt_front.n_conv(n_in, batch, tf)
            ttnn.deallocate(n_in)
            asr_res_tt = tt_front.asr_res(asr_tt, batch, time_asr)
            x0_tt = ttnn.concat([asr_tt, f0_tt, n_tt], dim=1, memory_config=l1)
            s_tt = ttnn.from_torch(
                s,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            x_tt = tt_body(x0_tt, s_tt, asr_res_tt, f0_tt, n_tt)
            x = ttnn.to_torch(x_tt).reshape(batch, 512, 2 * time_asr)
        elif tt_front is not None and ttnn_device is not None:
            f0_in = ttnn.from_torch(
                f0_curve.unsqueeze(1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            f0_tt = tt_front.f0_conv(f0_in, batch, tf)
            f0 = ttnn.to_torch(f0_tt).reshape(batch, 1, time_asr)
            ttnn.deallocate(f0_in)
            ttnn.deallocate(f0_tt)

            n_in = ttnn.from_torch(
                n.unsqueeze(1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            n_tt = tt_front.n_conv(n_in, batch, tf)
            n_b = ttnn.to_torch(n_tt).reshape(batch, 1, time_asr)
            ttnn.deallocate(n_in)
            ttnn.deallocate(n_tt)

            asr_in = ttnn.from_torch(
                asr,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                memory_config=l1,
            )
            asr_tt = tt_front.asr_res(asr_in, batch, time_asr)
            asr_res = ttnn.to_torch(asr_tt).reshape(batch, 64, time_asr)
            ttnn.deallocate(asr_in)
            ttnn.deallocate(asr_tt)

            x = torch.cat([asr, f0, n_b], dim=1)
            x = dec.encode(x, s)
            res = True
            for block in dec.decode:
                if res:
                    x = torch.cat([x, asr_res, f0, n_b], dim=1)
                x = block(x, s)
                if block.upsample_type != "none":
                    res = False
        else:
            f0 = dec.F0_conv(f0_curve.unsqueeze(1))
            n_b = dec.N_conv(n.unsqueeze(1))
            asr_res = dec.asr_res(asr)

            x = torch.cat([asr, f0, n_b], dim=1)
            x = dec.encode(x, s)
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
    front_p = preprocess_kokoro_decoder_front_parameters(dec, ttnn_device)
    tt_front = KokoroDecoderFront(ttnn_device, front_p)
    body_p = preprocess_kokoro_decoder_body_parameters(dec, ttnn_device)
    tt_body = KokoroDecoderBody(ttnn_device, body_p)

    x, s, f0_curve = _decoder_tensors_for_generator(
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
    # CPU ``SineGen`` vs ``KokoroTtnnSineGen`` + STFT chain: deterministic PCC is ~0.62 here (WH B0).
    ok, p = comp_pcc(y_ref, y_hat, pcc=0.61)
    print(f"generator waveform PCC={p:.6f} pass={ok} (min 0.61)")
    assert ok, f"generator waveform PCC {p} (expected >= 0.61; CPU vs device harmonic source)"
