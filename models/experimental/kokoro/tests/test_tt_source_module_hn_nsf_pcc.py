# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_source_module_hn_nsf.TTSourceModuleHnNSF`
vs reference :class:`~models.experimental.kokoro.reference.istftnet.SourceModuleHnNSF`."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.istftnet import SourceModuleHnNSF
from models.experimental.kokoro.tt.tt_source_module_hn_nsf import (
    TTSourceModuleHnNSF,
    preprocess_tt_source_module_hn_nsf,
)


@contextmanager
def _torch_random_zeros():
    """Patch ``torch.rand`` / ``torch.randn_like`` to return zeros for deterministic comparison."""
    real_rand = torch.rand
    real_randn_like = torch.randn_like

    def fake_rand(*size, **kwargs):
        return torch.zeros(*size, **kwargs)

    def fake_randn_like(t, **kwargs):
        return torch.zeros_like(t, **kwargs)

    torch.rand = fake_rand
    torch.randn_like = fake_randn_like
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def _make_ref(*, sampling_rate, upsample_scale, harmonic_num, sine_amp=0.1, noise_std=0.003, voiced_threshold=0.0):
    return SourceModuleHnNSF(
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        sine_amp=sine_amp,
        add_noise_std=noise_std,
        voiced_threshod=voiced_threshold,
    ).eval()


def _run_deterministic_pcc(device, *, sampling_rate, upsample_scale, harmonic_num, time_len, B, seed):
    torch.manual_seed(seed)
    ref = _make_ref(sampling_rate=sampling_rate, upsample_scale=upsample_scale, harmonic_num=harmonic_num)
    f0 = torch.relu(torch.randn(B, time_len, 1) * 200.0)

    with torch.no_grad(), _torch_random_zeros():
        sine_merge_ref, noise_ref, uv_ref = ref(f0)

    params = preprocess_tt_source_module_hn_nsf(
        ref,
        device,
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        voiced_threshold=0.0,
        time_len=time_len,
    )
    tt_mod = TTSourceModuleHnNSF(device, params)

    f0_tt = ttnn.from_torch(f0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sine_merge_tt, noise_tt, uv_tt = tt_mod(f0_tt)

    sm_h = ttnn.to_torch(sine_merge_tt).float()
    n_h = ttnn.to_torch(noise_tt).float()
    uv_h = ttnn.to_torch(uv_tt).float()
    for arr, ref_t in ((sm_h, sine_merge_ref), (n_h, noise_ref), (uv_h, uv_ref)):
        while arr.dim() > ref_t.dim():
            arr.squeeze_(0)
    ttnn.deallocate(sine_merge_tt)
    ttnn.deallocate(noise_tt)
    ttnn.deallocate(uv_tt)
    ttnn.deallocate(f0_tt)

    assert sm_h.shape == sine_merge_ref.shape, (sm_h.shape, sine_merge_ref.shape)
    assert uv_h.shape == uv_ref.shape, (uv_h.shape, uv_ref.shape)

    _, pcc_sm = comp_pcc(sine_merge_ref, sm_h, pcc=0.0)
    _, pcc_uv = comp_pcc(uv_ref, uv_h, pcc=0.0)
    noise_abs = n_h.abs().max().item()
    print(
        f"TTSourceModuleHnNSF (sr={sampling_rate}, up={upsample_scale}, harm={harmonic_num}, T={time_len}, B={B}) "
        f"sine_merge PCC: {pcc_sm:.6f}, uv PCC: {pcc_uv:.6f}, noise |max|: {noise_abs:.2e}"
    )
    assert pcc_sm > 0.99, f"sine_merge PCC too low: {pcc_sm}"
    assert pcc_uv > 0.99, f"uv PCC too low: {pcc_uv}"
    assert noise_abs < 1e-3, f"noise should be ~0 in deterministic mode (got {noise_abs})"


def test_tt_source_module_hn_nsf_default(device):
    """Default Kokoro: ``harmonic_num=0`` (``dim=1``)."""
    _run_deterministic_pcc(device, sampling_rate=24000.0, upsample_scale=4, harmonic_num=0, time_len=64, B=1, seed=0)


def test_tt_source_module_hn_nsf_with_harmonics(device):
    """``harmonic_num=2`` — exercises the Linear(dim, 1) merge."""
    _run_deterministic_pcc(device, sampling_rate=24000.0, upsample_scale=4, harmonic_num=2, time_len=64, B=1, seed=1)


def test_tt_source_module_hn_nsf_larger_upsample(device):
    """``upsample_scale=10`` with ``T=160``."""
    _run_deterministic_pcc(device, sampling_rate=24000.0, upsample_scale=10, harmonic_num=1, time_len=160, B=2, seed=2)


def test_tt_source_module_hn_nsf_noise_path(device):
    """Inject a deterministic ``out_noise_raw`` and verify the noise branch matches the reference."""
    torch.manual_seed(3)
    sampling_rate = 24000.0
    upsample_scale = 4
    # ``harmonic_num >= 1`` so SineGen's internal ``randn_like(sine_waves)`` ([B, T, dim]) has a
    # different shape from the output ``randn_like(uv)`` ([B, T, 1]); the shape-based filter below
    # then injects ``fixed_noise`` only into the latter.
    harmonic_num = 1
    time_len = 64
    B = 1

    ref = _make_ref(sampling_rate=sampling_rate, upsample_scale=upsample_scale, harmonic_num=harmonic_num)
    f0 = torch.relu(torch.randn(B, time_len, 1) * 200.0)

    fixed_noise = torch.randn(B, time_len, 1)

    # On the reference side: ``randn_like(uv) * sine_amp / 3`` is what produces the output ``noise``.
    # The SineGen side uses its own ``rand`` and ``randn_like`` — we zero those.
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    # Track which call we're on: SineGen's internal ``randn_like`` (zero) vs ours (``fixed_noise``).
    state = {"calls": 0}

    def fake_randn_like(t, **kwargs):
        # SineGen calls ``randn_like(sine_waves)`` first ([B, T, dim]) — return zeros.
        # SourceModuleHnNSF then calls ``randn_like(uv)`` ([B, T, 1]) — return ``fixed_noise``.
        if t.shape == fixed_noise.shape:
            return fixed_noise.to(t.dtype)
        return torch.zeros_like(t, **kwargs)

    torch.randn_like = fake_randn_like
    try:
        with torch.no_grad():
            sine_merge_ref, noise_ref, uv_ref = ref(f0)
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like

    params = preprocess_tt_source_module_hn_nsf(
        ref,
        device,
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        voiced_threshold=0.0,
        time_len=time_len,
    )
    tt_mod = TTSourceModuleHnNSF(device, params)

    f0_tt = ttnn.from_torch(f0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    noise_tt_in = ttnn.from_torch(fixed_noise, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sine_merge_tt, noise_tt, uv_tt = tt_mod(f0_tt, out_noise_raw=noise_tt_in)

    sm_h = ttnn.to_torch(sine_merge_tt).float()
    n_h = ttnn.to_torch(noise_tt).float()
    while sm_h.dim() > sine_merge_ref.dim():
        sm_h.squeeze_(0)
    while n_h.dim() > noise_ref.dim():
        n_h.squeeze_(0)
    ttnn.deallocate(sine_merge_tt)
    ttnn.deallocate(noise_tt)
    ttnn.deallocate(uv_tt)
    ttnn.deallocate(noise_tt_in)
    ttnn.deallocate(f0_tt)

    _, pcc_sm = comp_pcc(sine_merge_ref, sm_h, pcc=0.0)
    _, pcc_n = comp_pcc(noise_ref, n_h, pcc=0.0)
    print(f"TTSourceModuleHnNSF (noise path) sine_merge PCC: {pcc_sm:.6f}, noise PCC: {pcc_n:.6f}")
    assert pcc_sm > 0.99, f"sine_merge PCC too low: {pcc_sm}"
    assert pcc_n > 0.99, f"noise PCC too low: {pcc_n}"


def test_tt_source_module_hn_nsf_tanh_only_vs_linear_fallback(device):
    """Compare source-module PCC when only tanh falls back vs when linear falls back.

    Mirrors the generator-side fallback investigation using Kokoro-like source settings:
    ``upsample_scale=300`` and ``harmonic_num=8`` (``dim=9``).
    """
    torch.manual_seed(11)
    sampling_rate = 24000.0
    upsample_scale = 300
    harmonic_num = 8
    time_len = 1500
    B = 1

    ref = _make_ref(sampling_rate=sampling_rate, upsample_scale=upsample_scale, harmonic_num=harmonic_num)
    f0 = torch.relu(torch.randn(B, time_len, 1) * 200.0)

    with torch.no_grad(), _torch_random_zeros():
        sine_merge_ref, _noise_ref, _uv_ref = ref(f0)

    params = preprocess_tt_source_module_hn_nsf(
        ref,
        device,
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        voiced_threshold=0.0,
        time_len=time_len,
    )

    f0_tt = ttnn.from_torch(f0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_linear = TTSourceModuleHnNSF(device, params, use_torch_linear_fallback=True, use_torch_tanh_fallback=False)
    tt_tanh_only = TTSourceModuleHnNSF(device, params, use_torch_linear_fallback=False, use_torch_tanh_fallback=True)
    tt_both = TTSourceModuleHnNSF(device, params, use_torch_linear_fallback=True, use_torch_tanh_fallback=True)

    sm_linear_tt, noise_linear_tt, uv_linear_tt = tt_linear(f0_tt)
    sm_tanh_tt, noise_tanh_tt, uv_tanh_tt = tt_tanh_only(f0_tt)
    sm_both_tt, noise_both_tt, uv_both_tt = tt_both(f0_tt)

    sm_linear_h = ttnn.to_torch(sm_linear_tt).float()
    sm_tanh_h = ttnn.to_torch(sm_tanh_tt).float()
    sm_both_h = ttnn.to_torch(sm_both_tt).float()
    while sm_linear_h.dim() > sine_merge_ref.dim():
        sm_linear_h.squeeze_(0)
    while sm_tanh_h.dim() > sine_merge_ref.dim():
        sm_tanh_h.squeeze_(0)
    while sm_both_h.dim() > sine_merge_ref.dim():
        sm_both_h.squeeze_(0)

    _, pcc_linear = comp_pcc(sine_merge_ref, sm_linear_h, pcc=0.0)
    _, pcc_tanh_only = comp_pcc(sine_merge_ref, sm_tanh_h, pcc=0.0)
    _, pcc_both = comp_pcc(sine_merge_ref, sm_both_h, pcc=0.0)
    delta = pcc_linear - pcc_tanh_only
    print(
        "TTSourceModuleHnNSF fallback comparison: "
        f"linear-only PCC={pcc_linear:.6f}, tanh-only PCC={pcc_tanh_only:.6f}, "
        f"both-fallbacks PCC={pcc_both:.6f}, delta={delta:.6f}"
    )

    # Matches the generator-level observation: preserving TTNN linear keeps the dominant BF16
    # dot-product error, so tanh-only fallback should not outperform linear fallback.
    assert pcc_linear >= pcc_tanh_only, (
        f"Expected linear fallback PCC >= tanh-only fallback PCC; got "
        f"linear={pcc_linear:.6f}, tanh-only={pcc_tanh_only:.6f}"
    )

    ttnn.deallocate(sm_linear_tt)
    ttnn.deallocate(noise_linear_tt)
    ttnn.deallocate(uv_linear_tt)
    ttnn.deallocate(sm_tanh_tt)
    ttnn.deallocate(noise_tanh_tt)
    ttnn.deallocate(uv_tanh_tt)
    ttnn.deallocate(sm_both_tt)
    ttnn.deallocate(noise_both_tt)
    ttnn.deallocate(uv_both_tt)
    ttnn.deallocate(f0_tt)
