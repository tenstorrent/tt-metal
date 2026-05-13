# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-run PCC table: TTNN ``KokoroTtnnSineGen`` vs PyTorch ``SineGen`` (``use_torch_sinegen``).

Run with capture disabled so the Markdown table appears in logs::

    pytest models/experimental/kokoro/tests/test_kokoro_pcc_sinegen_comparison_report.py \\
        --confcutdir=models/experimental/kokoro -v -s

Assertions reuse the same per-scenario floors as the parametrized PCC tests.
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
from models.experimental.kokoro.reference import KokoroConfig, KokoroFullReference
from models.experimental.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.experimental.kokoro.tests.kokoro_generator_pcc_inputs import decoder_tensors_for_generator
from models.experimental.kokoro.tt import (
    KokoroDecoderBody,
    KokoroDecoderFront,
    KokoroDecoderTt,
    KokoroGenerator,
    preprocess_kokoro_decoder_body_parameters,
    preprocess_kokoro_decoder_front_parameters,
    preprocess_kokoro_decoder_tt_parameters,
    preprocess_kokoro_generator_parameters,
)


def _zeros_noise_patches():
    def zeros_rand(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
        return torch.zeros(*args, **kwargs)

    def zeros_randn(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
        return torch.zeros(*args, **kwargs)

    def zeros_randn_like(t, **kwargs):
        return torch.zeros_like(t)

    return zeros_rand, zeros_randn, zeros_randn_like


def _pcc_generator(mesh_device, use_torch_sinegen: bool) -> float:
    dec_wrap = load_decoder_from_huggingface(device="cpu", disable_complex=True)
    dec = dec_wrap.decoder
    gen = dec.generator
    time_asr, batch = 8, 1
    front_p = preprocess_kokoro_decoder_front_parameters(dec, mesh_device)
    tt_front = KokoroDecoderFront(mesh_device, front_p)
    body_p = preprocess_kokoro_decoder_body_parameters(dec, mesh_device)
    tt_body = KokoroDecoderBody(mesh_device, body_p)
    x, s, f0_curve = decoder_tensors_for_generator(
        dec, batch, time_asr, seed=42, tt_front=tt_front, tt_body=tt_body, ttnn_device=mesh_device
    )
    sf = int(round(float(gen.f0_upsamp.scale_factor)))
    f0_up_t = f0_curve.shape[1] * sf
    zeros_rand, zeros_randn, zeros_randn_like = _zeros_noise_patches()
    params = preprocess_kokoro_generator_parameters(
        gen,
        mesh_device,
        f0_upsampled_time=f0_up_t,
        disable_complex=True,
        use_torch_sinegen=use_torch_sinegen,
    )
    tt_gen = KokoroGenerator(mesh_device, params)
    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        with torch.no_grad():
            y_ref = gen(x, s, f0_curve)
    l1 = ttnn.L1_MEMORY_CONFIG
    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=l1)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=l1)
    f0_tt = ttnn.from_torch(
        f0_curve.unsqueeze(-1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=l1
    )
    y_tt = tt_gen(x_tt, s_tt, f0_tt, deterministic=True)
    y_hat = ttnn.to_torch(y_tt).reshape(y_ref.shape)
    assert y_hat.shape == y_ref.shape
    assert torch.isfinite(y_hat).all()
    _ok, p = comp_pcc(y_ref, y_hat, pcc=0.0)
    return float(p)


def _pcc_decoder_tt_e2e(mesh_device, use_torch_sinegen: bool) -> float:
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
    zeros_rand, zeros_randn, zeros_randn_like = _zeros_noise_patches()
    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        with torch.no_grad():
            y_ref = dec_ref(asr, f0_curve, n, s)
    params = preprocess_kokoro_decoder_tt_parameters(
        dec_tt, mesh_device, f0_coarse_time=tf, disable_complex=True, use_torch_sinegen=use_torch_sinegen
    )
    tt_dec = KokoroDecoderTt(mesh_device, params)
    l1 = ttnn.L1_MEMORY_CONFIG
    asr_tt = ttnn.from_torch(asr, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=l1)
    f0_tt = ttnn.from_torch(
        f0_curve.unsqueeze(-1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=l1
    )
    n_tt = ttnn.from_torch(
        n.unsqueeze(-1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=l1
    )
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=l1)
    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        y_tt = tt_dec(asr_tt, f0_tt, n_tt, s_tt, deterministic=True)
    y_hat = ttnn.to_torch(y_tt).reshape(y_ref.shape)
    assert torch.isfinite(y_hat).all()
    _ok, p = comp_pcc(y_ref, y_hat, pcc=0.0)
    return float(p)


def _pcc_full_pipeline(mesh_device, use_torch_sinegen: bool) -> float:
    pytest.importorskip("kokoro")
    from kokoro import KPipeline

    from models.experimental.kokoro.tt.ttnn_kokoro_full_pipeline import KokoroFullTtnn

    text = "Hi."
    voice = "af_heart"
    pipe = KPipeline(lang_code="a", model=False)
    results = list(pipe(text, voice=voice, speed=2.0))
    assert results
    phonemes = results[0].phonemes
    assert phonemes
    pack = pipe.load_voice(voice)
    ref_s = pack[len(phonemes) - 1].to("cpu")
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    zeros_rand, zeros_randn, zeros_randn_like = _zeros_noise_patches()
    ref = KokoroFullReference(repo_id=KokoroConfig.repo_id, device="cpu", disable_complex=True)
    tt_model = KokoroFullTtnn(
        mesh_device,
        repo_id=KokoroConfig.repo_id,
        disable_complex=True,
        use_torch_sinegen=use_torch_sinegen,
    )
    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        with torch.no_grad():
            out_ref = ref(phonemes=phonemes, ref_s=ref_s, speed=2.0)
    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        with torch.no_grad():
            out_tt = tt_model(phonemes=phonemes, ref_s=ref_s, speed=2.0, deterministic=True)
    assert out_ref.audio.shape == out_tt.audio.shape
    assert torch.isfinite(out_tt.audio).all()
    _ok, p = comp_pcc(out_ref.audio, out_tt.audio, pcc=0.0)
    return float(p)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_kokoro_sinegen_pcc_comparison_table(mesh_device):
    rows: list[tuple[str, float, float, float, float, float]] = []
    # (name, p_ttnn, min_ttnn, p_torch, min_torch, delta)
    # Full pipeline first: it is most sensitive to device state after heavy vocoder work.
    # Floors track current achievable PCC on WH B0 after:
    #   * SineGen downsample switched to sparse fp32 gather-lerp (decoder e2e 0.583 → 0.799 with TT SG).
    #   * Predictor + text_encoder compute_kernel_config switched from HiFi4 to HiFi3 + fp32 dest acc
    #     (avoids the WH HiFi4-fp32-accum HW bug; full-pipeline 0.137 → 0.81).
    #   * Duration-projection weights pinned to fp32 so ``round(dur)`` aligns with PyTorch.
    # Tighten as upstream LSTM / duration_encoder precision improves.
    scenarios = [
        ("KokoroFullTtnn vs KokoroFullReference", _pcc_full_pipeline, 0.80, 0.80),
        ("Generator vs PyTorch ref (decoder prefix on TTNN)", _pcc_generator, 0.75, 0.85),
        ("KokoroDecoderTt e2e vs PyTorch Decoder", _pcc_decoder_tt_e2e, 0.75, 0.90),
    ]
    for name, fn, min_ttnn, min_torch in scenarios:
        p_ttnn = fn(mesh_device, False)
        p_torch = fn(mesh_device, True)
        delta = p_torch - p_ttnn
        rows.append((name, p_ttnn, min_ttnn, p_torch, min_torch, delta))
        assert p_ttnn >= min_ttnn, f"{name} TTNN SineGen PCC {p_ttnn} < {min_ttnn}"
        assert p_torch >= min_torch, f"{name} torch SineGen PCC {p_torch} < {min_torch}"

    lines = [
        "",
        "### Kokoro experimental — waveform PCC: TTNN SineGen vs torch SineGen",
        "",
        "| Scenario | PCC (TTNN SineGen) | min | PCC (torch SineGen) | min | Δ (torch − TTNN) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, p_ttnn, min_ttnn, p_torch, min_torch, delta in rows:
        safe = name.replace("|", "\\|")
        lines.append(f"| {safe} | {p_ttnn:.6f} | {min_ttnn:.2f} | {p_torch:.6f} | {min_torch:.2f} | {delta:+.6f} |")
    lines.append("")
    lines.append(
        "Other `comp_pcc` tests under `models/experimental/kokoro/tests/` "
        "(PL-BERT, predictor, decoder front/body, unit SineGen, STFT, AdaIN, …) "
        "do not branch on `use_torch_sinegen`; only the generator harmonic path does."
    )
    lines.append("")
    print("\n".join(lines))
