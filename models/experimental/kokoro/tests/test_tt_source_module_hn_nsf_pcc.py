# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_source_module_hn_nsf.TTSourceModuleHnNSF`
vs reference :class:`~models.experimental.kokoro.reference.istftnet.SourceModuleHnNSF`."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F_torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.m_source_rng import (
    MSourceRngTensors,
    deallocate_m_source_rng_tt,
    patched_m_source_torch_rng,
    upload_m_source_rng,
)
from models.experimental.kokoro.reference.istftnet import SourceModuleHnNSF
from models.experimental.kokoro.tt.tt_source_module_hn_nsf import (
    TTSourceModuleHnNSF,
    preprocess_tt_source_module_hn_nsf,
)

_CKPT_CANDIDATES = (
    Path("/home/ubuntu/ign-tt/kokoro/examples/checkpoints/kokoro-v1_0.pth"),
    Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots",
)


def _find_checkpoint() -> Path | None:
    for p in _CKPT_CANDIDATES:
        if p.is_file():
            return p
        if p.is_dir():
            for child in p.rglob("kokoro-v1_0.pth"):
                return child
    return None


def _pcc_ref_tt(ref_t: torch.Tensor, tt_t: ttnn.Tensor) -> float:
    ref_f = ref_t.float()
    tt_h = ttnn.to_torch(tt_t).float()
    while tt_h.dim() > ref_f.dim():
        tt_h = tt_h.squeeze(0)
    if ref_f.shape != tt_h.shape:
        raise ValueError(f"shape mismatch ref={tuple(ref_f.shape)} tt={tuple(tt_h.shape)}")
    _, pcc = comp_pcc(ref_f, tt_h, pcc=0.0)
    return float(pcc)


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

    params = preprocess_tt_source_module_hn_nsf(
        ref,
        device,
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        voiced_threshold=0.0,
        time_len=time_len,
    )
    with _torch_random_zeros():
        with torch.no_grad():
            sine_merge_ref, noise_ref, uv_ref = ref(f0)
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
    """Verify the noise branch using noise generated at init time."""
    torch.manual_seed(3)
    sampling_rate = 24000.0
    upsample_scale = 4
    # ``harmonic_num >= 1`` so SineGen's dummy ([B, T, dim]) has a different shape from the
    # output dummy ([B, T, 1]); the shape filter injects ``fixed_noise`` only into the latter.
    harmonic_num = 1
    time_len = 64
    B = 1

    ref = _make_ref(sampling_rate=sampling_rate, upsample_scale=upsample_scale, harmonic_num=harmonic_num)
    f0 = torch.relu(torch.randn(B, time_len, 1) * 200.0)

    fixed_noise = torch.randn(B, time_len, 1)

    def fake_randn_like(t, **kwargs):
        # SineGen dummy/call ([B, T, dim=2]) → zeros; SourceModule dummy/call ([B, T, 1]) → fixed_noise.
        if t.shape == fixed_noise.shape:
            return fixed_noise.to(t.dtype)
        return torch.zeros_like(t, **kwargs)

    params = preprocess_tt_source_module_hn_nsf(
        ref,
        device,
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        voiced_threshold=0.0,
        time_len=time_len,
    )
    # Apply the same randn_like patch for both TT init and reference forward:
    #   sinegen._noise_raw = zeros, _out_noise_raw = fixed_noise (from init)
    #   reference SineGen noise = zeros, SourceModule output noise = fixed_noise
    real_randn_like = torch.randn_like
    torch.randn_like = fake_randn_like
    try:
        tt_mod = TTSourceModuleHnNSF(device, params)
        with torch.no_grad():
            sine_merge_ref, noise_ref, uv_ref = ref(f0)
    finally:
        torch.randn_like = real_randn_like

    f0_tt = ttnn.from_torch(f0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sine_merge_tt, noise_tt, uv_tt = tt_mod(f0_tt)

    sm_h = ttnn.to_torch(sine_merge_tt).float()
    n_h = ttnn.to_torch(noise_tt).float()
    while sm_h.dim() > sine_merge_ref.dim():
        sm_h.squeeze_(0)
    while n_h.dim() > noise_ref.dim():
        n_h.squeeze_(0)
    ttnn.deallocate(sine_merge_tt)
    ttnn.deallocate(noise_tt)
    ttnn.deallocate(uv_tt)
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

    params = preprocess_tt_source_module_hn_nsf(
        ref,
        device,
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        voiced_threshold=0.0,
        time_len=time_len,
    )
    with _torch_random_zeros():
        with torch.no_grad():
            sine_merge_ref, _noise_ref, _uv_ref = ref(f0)
        tt_linear = TTSourceModuleHnNSF(device, params, use_torch_linear_fallback=True, use_torch_tanh_fallback=False)
        tt_tanh_only = TTSourceModuleHnNSF(
            device, params, use_torch_linear_fallback=False, use_torch_tanh_fallback=True
        )
        tt_both = TTSourceModuleHnNSF(device, params, use_torch_linear_fallback=True, use_torch_tanh_fallback=True)

    f0_tt = ttnn.from_torch(f0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

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


def _kokoro_source_module_from_ckpt(ckpt_path: Path) -> SourceModuleHnNSF:
    """Trained Kokoro ``Generator.m_source`` (upsample_scale=300, harmonic_num=8)."""
    ref = _make_ref(sampling_rate=24000.0, upsample_scale=300, harmonic_num=8, voiced_threshold=10.0)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    prefix = "module.generator.m_source."
    sd = {k[len(prefix) :]: v for k, v in ckpt["decoder"].items() if k.startswith(prefix)}
    ref.load_state_dict(sd, strict=False)
    return ref.eval()


def _run_m_source_per_op_diagnostic(
    device,
    *,
    use_torch_linear_fallback: bool,
    use_torch_sinegen_fallback: bool = False,
) -> dict[str, float]:
    """Run per-op PCC diagnostic; return ``{checkpoint_name: pcc}``."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")

    ref = _kokoro_source_module_from_ckpt(ckpt_path)
    time_len = 1500
    upsample_scale = 300
    B = 1
    dim = ref.l_sin_gen.dim

    torch.manual_seed(42)
    t = torch.arange(time_len, dtype=torch.float32)
    f0 = (100.0 + 50.0 * torch.sin(2 * torch.pi * t / 200.0)).unsqueeze(0).unsqueeze(-1)

    rand_ini = torch.rand(B, dim)
    rand_ini[:, 0] = 0.0
    rng_cpu = MSourceRngTensors(
        rand_ini=rand_ini,
        sinegen_noise=torch.randn(B, time_len, dim),
        source_noise=torch.randn(B, time_len, 1),
    )

    params = preprocess_tt_source_module_hn_nsf(
        ref,
        device,
        sampling_rate=24000.0,
        upsample_scale=upsample_scale,
        harmonic_num=8,
        voiced_threshold=10.0,
        time_len=time_len,
    )
    tt_mod = TTSourceModuleHnNSF(
        device,
        params,
        use_torch_sinegen_fallback=use_torch_sinegen_fallback,
        use_torch_linear_fallback=use_torch_linear_fallback,
        use_torch_tanh_fallback=False,
    )
    assert tt_mod._use_torch_linear_fallback == use_torch_linear_fallback
    assert not tt_mod._use_torch_tanh_fallback
    assert not tt_mod._sinegen.use_torch_phase_fallback
    assert tt_mod._sinegen.use_torch_sinegen_fallback == use_torch_sinegen_fallback

    mc = ttnn.DRAM_MEMORY_CONFIG
    pcc_log: list[tuple[str, float]] = []

    def _log(name: str, ref_t: torch.Tensor, tt_t: ttnn.Tensor) -> None:
        pcc_log.append((name, _pcc_ref_tt(ref_t, tt_t)))

    with torch.no_grad(), patched_m_source_torch_rng(rng_cpu):
        sine_wavs_ref, uv_ref, _ = ref.l_sin_gen(f0)
        linear_pre_ref = ref.l_linear(sine_wavs_ref)
        sine_merge_ref = ref.l_tanh(linear_pre_ref)
        noise_ref = torch.randn_like(uv_ref) * (ref.sine_amp / 3.0)
        sine_merge_e2e, noise_e2e, uv_e2e = ref(f0)

    rng_tt = upload_m_source_rng(rng_cpu, device, memory_config=mc)
    f0_tt = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    sine_wavs_tt, uv_tt, sinegen_noise_tt = tt_mod._sinegen.forward(
        f0_tt,
        rand_ini=rng_tt.rand_ini,
        noise_raw=rng_tt.sinegen_noise,
        memory_config=mc,
    )
    _log("sinegen.sine_wavs", sine_wavs_ref, sine_wavs_tt)
    _log("sinegen.uv", uv_ref, uv_tt)
    ttnn.deallocate(sinegen_noise_tt)

    if use_torch_linear_fallback:
        x_cpu = ttnn.to_torch(sine_wavs_tt).float().reshape(B * time_len, dim)
        w_cpu = ttnn.to_torch(params.linear_weight).float().reshape(1, dim)
        b_cpu = ttnn.to_torch(params.linear_bias).float().flatten()[:1]
        ttnn.deallocate(sine_wavs_tt)
        merged_cpu = F_torch.linear(x_cpu, w_cpu, b_cpu).reshape(B, time_len, 1)
        merged = ttnn.from_torch(
            merged_cpu.contiguous(),
            dtype=params.sinegen.activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
    else:
        merged = ttnn.linear(
            sine_wavs_tt,
            params.linear_weight,
            bias=params.linear_bias,
            transpose_b=True,
            memory_config=mc,
            compute_kernel_config=tt_mod.compute_kernel_config,
        )
        ttnn.deallocate(sine_wavs_tt)
        while len(merged.shape) > 3:
            merged = ttnn.squeeze(merged, 0)

    _log("linear_pre_tanh", linear_pre_ref, merged)

    # Isolate linear: CPU float32 on *reference* sine_wavs should match ref linear exactly.
    x_ref_cpu = sine_wavs_ref.float().reshape(B * time_len, dim)
    w_cpu = ttnn.to_torch(params.linear_weight).float().reshape(1, dim)
    b_cpu = ttnn.to_torch(params.linear_bias).float().flatten()[:1]
    merged_ref_sine_cpu = F_torch.linear(x_ref_cpu, w_cpu, b_cpu).reshape(B, time_len, 1)
    merged_ref_sine_tt = ttnn.from_torch(
        merged_ref_sine_cpu.contiguous(),
        dtype=params.sinegen.activation_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    _log("linear_pre_tanh.cpu_ref_sine", linear_pre_ref, merged_ref_sine_tt)
    ttnn.deallocate(merged_ref_sine_tt)

    sine_merge_tt = ttnn.tanh(merged, memory_config=mc)
    ttnn.deallocate(merged)
    _log("sine_merge", sine_merge_ref, sine_merge_tt)

    noise_tt = ttnn.multiply(rng_tt.source_noise, params.noise_scale, memory_config=mc)
    _log("output_noise", noise_ref, noise_tt)

    _log("end_to_end.sine_merge", sine_merge_e2e, sine_merge_tt)
    _log("end_to_end.uv", uv_e2e, uv_tt)
    _log("end_to_end.output_noise", noise_e2e, noise_tt)

    ttnn.deallocate(f0_tt)
    ttnn.deallocate(uv_tt)
    ttnn.deallocate(sine_merge_tt)
    ttnn.deallocate(noise_tt)
    deallocate_m_source_rng_tt(rng_tt)

    if use_torch_sinegen_fallback:
        mode = "sinegen_fallback"
    else:
        mode = "linear_fallback" if use_torch_linear_fallback else "pure_ttnn"
    print(f"\nTTSourceModuleHnNSF per-op PCC ({mode}, T={time_len}, up={upsample_scale}, dim={dim}):")
    for name, val in pcc_log:
        print(f"  {name:32s} {val:.6f}")

    return dict(pcc_log)


def test_tt_source_module_hn_nsf_per_op_no_fallback_diagnostic(device):
    """Per-op PCC for pure-TTNN :class:`TTSourceModuleHnNSF` (no fallbacks)."""
    by_name = _run_m_source_per_op_diagnostic(device, use_torch_linear_fallback=False)
    assert by_name["sinegen.uv"] > 0.99
    assert by_name["output_noise"] > 0.99
    assert by_name["end_to_end.uv"] > 0.99
    assert by_name["end_to_end.output_noise"] > 0.99
    assert by_name["linear_pre_tanh"] > 0.88, f"linear_pre_tanh PCC too low: {by_name['linear_pre_tanh']:.6f}"


def test_tt_source_module_hn_nsf_per_op_linear_fallback_diagnostic(device):
    """Per-op PCC with ``use_torch_linear_fallback=True`` (CPU float32 ``l_linear``).

    With Kokoro SineGen at ``upsample_scale=300``, ``sine_wavs`` PCC ~0.94 dominates;
    linear fallback fixes BF16 MAC error on ``dim=9`` but does not raise ``linear_pre_tanh``
    much until SineGen/phase is tighter (see ``linear_pre_tanh.cpu_ref_sine``).
    """
    by_no_fb = _run_m_source_per_op_diagnostic(device, use_torch_linear_fallback=False)
    by_lin = _run_m_source_per_op_diagnostic(device, use_torch_linear_fallback=True)
    print(
        "\nlinear_fallback delta (linear_pre_tanh): " f"{by_lin['linear_pre_tanh'] - by_no_fb['linear_pre_tanh']:+.6f}"
    )

    assert by_lin["sinegen.uv"] > 0.99
    assert by_lin["output_noise"] > 0.99
    assert by_lin["end_to_end.uv"] > 0.99
    assert by_lin["end_to_end.output_noise"] > 0.99
    # CPU linear on ref sine_wavs proves fallback path is wired correctly.
    assert by_lin["linear_pre_tanh.cpu_ref_sine"] > 0.99
    # End-to-end chain still limited by TT SineGen input (~0.90), same as pure TTNN linear.
    assert by_lin["linear_pre_tanh"] > 0.88
    assert by_lin["linear_pre_tanh"] >= by_no_fb["linear_pre_tanh"] - 1e-4


def test_tt_source_module_hn_nsf_per_op_sinegen_fallback_diagnostic(device):
    """Show that long-sequence degradation starts in SineGen and improves with fallback.

    Keeps linear/tanh on TTNN to isolate SineGen fallback impact only.
    """
    by_no_fb = _run_m_source_per_op_diagnostic(
        device,
        use_torch_linear_fallback=False,
        use_torch_sinegen_fallback=False,
    )
    by_sinegen_fb = _run_m_source_per_op_diagnostic(
        device,
        use_torch_linear_fallback=False,
        use_torch_sinegen_fallback=True,
    )
    delta = by_sinegen_fb["sinegen.sine_wavs"] - by_no_fb["sinegen.sine_wavs"]
    print(f"\nsinegen_fallback delta (sinegen.sine_wavs): {delta:+.6f}")
    assert by_sinegen_fb["sinegen.sine_wavs"] > by_no_fb["sinegen.sine_wavs"] + 0.05
    assert by_sinegen_fb["end_to_end.sine_merge"] >= by_no_fb["end_to_end.sine_merge"]
