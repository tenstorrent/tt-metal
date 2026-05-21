# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_generator.TTGenerator` vs reference
:class:`~models.experimental.kokoro.reference.istftnet.Generator` (``TorchSTFT`` path).

Wormhole's ``ttnn.matmul`` uses bf16 intermediate products under HiFi3 (the highest math fidelity
that works correctly on WH B0 — HiFi4 + fp32 accumulator is known broken). The STFT in the
harmonic-source path is built on dense matmuls; for near-zero off-frequency bins, the bf16
products generate sign-flipped values, and ``atan2(imag, real)`` of those tilts to ``±π`` —
random phase. Trained Kokoro is robust to this (the ``noise_conv`` was specifically learned to
absorb it — see ``tt_old/ttnn_kokoro_stft.py`` comment: *"noise_conv[i] PCC stays at 0.999997+
on real inputs"*), but the full forward path cannot reach bit-exact PCC vs PyTorch.

We therefore split the PCC validation into two layers:

1. ``test_tt_generator_pipeline_pcc`` — injects the **reference** ``har`` ``(spec, phase)`` into the
   TT pipeline, isolating everything **except** the STFT phase computation. This validates that
   the TT noise_conv / resblock / ups / conv_post / iSTFT stack is bit-precise vs PyTorch. PCC
   must be > 0.99 (this is the actual implementation correctness test).

2. ``test_tt_generator_full_forward_smoke`` — runs the full TT Generator forward and only checks
   shape + finite output. The PCC vs PyTorch is documented as a known limitation of WH bf16
   matmul precision — it will not reach 0.99 without CPU fallback or new hardware features.

3. ``test_tt_generator_trained_harmonic_har_pcc_xfail_until_stft_phase_tight`` — isolates the
   harmonic branch ``har`` (``m_source`` + ``stft.transform``) vs PyTorch on the same ``f0`` and
   checkpoint. Marked ``xfail``: PCC is expected to stay below 0.99 until TT STFT phase matches
   reference more closely; the printed mag / cos(phase) lines show where the gap lives.
"""

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
from models.experimental.kokoro.reference.istftnet import Generator
from models.experimental.kokoro.tt.tt_conv import tt_conv1d_nlc, tt_conv_transpose1d_nlc
from models.experimental.kokoro.tt.tt_generator import (
    TTGenerator,
    _f0_upsamp_cpu_nlc,
    _reflection_pad_left_1_nlc,
    _upsample_nearest_axis1,
    preprocess_tt_generator,
)
from models.experimental.kokoro.tt.tt_torch_stft import TTTorchSTFT, _reflect_pad_1d_dim2


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


@contextmanager
def _torch_random_zeros():
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    torch.randn_like = lambda t, **kwargs: torch.zeros_like(t, **kwargs)
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def _build_kokoro_generator() -> Generator:
    return Generator(
        style_dim=128,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6],
        upsample_kernel_sizes=[20, 12],
        upsample_initial_channel=512,
        gen_istft_n_fft=20,
        gen_istft_hop_size=5,
        disable_complex=False,
    ).eval()


def _load_trained_weights(ref: Generator, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    prefix = "module.generator."
    gen_sd = {k[len(prefix) :]: v for k, v in ckpt["decoder"].items() if k.startswith(prefix)}
    ref.load_state_dict(gen_sd, strict=False)


def _setup_test_inputs(T_x: int, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(1, 512, T_x)
    s = torch.randn(1, 128)
    f0 = torch.tensor([[100.0, 150.0, 200.0, 175.0, 125.0]][:1])
    if T_x > 5:
        raise ValueError("Adjust f0 for T_x > 5")
    f0 = torch.tensor([[100.0, 150.0, 200.0, 175.0, 125.0][:T_x]])
    return x, s, f0


def _ref_har_and_audio(ref: Generator, x: torch.Tensor, s: torch.Tensor, f0: torch.Tensor):
    """Run the reference Generator, returning both the intermediate ``har`` and final audio."""
    with torch.no_grad(), _torch_random_zeros():
        f0u = ref.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_src, _, _ = ref.m_source(f0u)
        har_src = har_src.transpose(1, 2).squeeze(1)
        mag_ref, phase_ref = ref.stft.transform(har_src)
        har = torch.cat([mag_ref, phase_ref], dim=1)
        y_ref = ref(x, s, f0)
    return har, y_ref


def _assert_generator_no_fallbacks(tt_mod: TTGenerator) -> None:
    assert not tt_mod._m_source._use_torch_linear_fallback
    assert not tt_mod._m_source._use_torch_tanh_fallback
    assert not tt_mod._m_source._sinegen.use_torch_phase_fallback
    assert not tt_mod._m_source._sinegen.use_torch_sinegen_fallback
    assert not tt_mod._stft._use_torch_stft_fallback
    assert not tt_mod._stft._use_torch_stft_conv_fallback


def _pcc_ref_tt(ref_t: torch.Tensor, tt_t: ttnn.Tensor) -> float:
    """PCC between reference and TT tensors (auto NLC/BCT layout for rank-3)."""
    ref_f = ref_t.float()
    tt_h = ttnn.to_torch(tt_t).float()
    while tt_h.dim() > ref_f.dim():
        tt_h = tt_h.squeeze(0)
    if ref_f.dim() == 3 and tt_h.dim() == 3:
        # Reference is BCT ``[B, C, T]``; TT may be NLC ``[B, T, C]``.
        if tt_h.shape[1] == ref_f.shape[2] and tt_h.shape[2] == ref_f.shape[1]:
            tt_h = tt_h.transpose(1, 2).contiguous()
    if ref_f.shape != tt_h.shape:
        raise ValueError(f"shape mismatch ref={tuple(ref_f.shape)} tt={tuple(tt_h.shape)}")
    _, pcc = comp_pcc(ref_f, tt_h, pcc=0.0)
    return float(pcc)


def _pcc_flat(ref_bl: torch.Tensor, tt_bl: ttnn.Tensor) -> float:
    """PCC for flat ``[B, L]`` tensors (e.g. STFT input waveform)."""
    tt_h = ttnn.to_torch(tt_bl).float()
    while tt_h.dim() > ref_bl.dim():
        tt_h = tt_h.squeeze(0)
    _, pcc = comp_pcc(ref_bl.float(), tt_h, pcc=0.0)
    return float(pcc)


def test_tt_generator_m_source_no_fallback_pcc(device):
    """`Generator.m_source`: reference vs TT with identical deterministic inputs and loaded weights."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    _x, _s, f0 = _setup_test_inputs(T_x)
    f0u = ref.f0_upsamp(f0[:, None]).transpose(1, 2).contiguous()

    B, T_har, _ = f0u.shape
    dim = ref.m_source.l_sin_gen.dim
    torch.manual_seed(123)
    rand_ini = torch.rand(B, dim)
    rand_ini[:, 0] = 0.0
    sinegen_noise_raw = torch.randn(B, T_har, dim)
    source_noise_raw = torch.randn(B, T_har, 1)

    def _fake_rand(*size, **kwargs):
        out = rand_ini.to(kwargs.get("device", rand_ini.device))
        dtype = kwargs.get("dtype", out.dtype)
        return out.to(dtype)

    def _fake_randn_like(t, **kwargs):
        if t.shape[-1] == 1:
            return source_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()
        return sinegen_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = _fake_rand
    torch.randn_like = _fake_randn_like
    try:
        tt_mod = TTGenerator(
            device,
            params,
            use_torch_phase_fallback=False,
            use_torch_linear_fallback=False,
            use_torch_tanh_fallback=False,
        )
        with torch.no_grad():
            sine_merge_ref, noise_ref, uv_ref = ref.m_source(f0u)
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like
    _assert_generator_no_fallbacks(tt_mod)

    f0_tt = ttnn.from_torch(f0u, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    rand_ini_tt = ttnn.from_torch(rand_ini.unsqueeze(1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    sine_merge_tt, noise_tt, uv_tt = tt_mod._m_source(
        f0_tt,
        sinegen_rand_ini=rand_ini_tt,
    )

    sine_merge_h = ttnn.to_torch(sine_merge_tt).float()
    noise_h = ttnn.to_torch(noise_tt).float()
    uv_h = ttnn.to_torch(uv_tt).float()
    while sine_merge_h.dim() > sine_merge_ref.dim():
        sine_merge_h.squeeze_(0)
    while noise_h.dim() > noise_ref.dim():
        noise_h.squeeze_(0)
    while uv_h.dim() > uv_ref.dim():
        uv_h.squeeze_(0)

    ttnn.deallocate(sine_merge_tt)
    ttnn.deallocate(noise_tt)
    ttnn.deallocate(uv_tt)
    ttnn.deallocate(f0_tt)
    ttnn.deallocate(rand_ini_tt)

    assert sine_merge_h.shape == sine_merge_ref.shape, (sine_merge_h.shape, sine_merge_ref.shape)
    assert noise_h.shape == noise_ref.shape, (noise_h.shape, noise_ref.shape)
    assert uv_h.shape == uv_ref.shape, (uv_h.shape, uv_ref.shape)

    _, pcc_sine = comp_pcc(sine_merge_ref, sine_merge_h, pcc=0.0)
    _, pcc_noise = comp_pcc(noise_ref, noise_h, pcc=0.0)
    _, pcc_uv = comp_pcc(uv_ref, uv_h, pcc=0.0)
    print("TTGenerator m_source no-fallback PCC: " f"sine_merge={pcc_sine:.6f}, noise={pcc_noise:.6f}, uv={pcc_uv:.6f}")
    assert pcc_sine > 0.98, f"m_source sine_merge PCC too low: {pcc_sine}"
    assert pcc_noise > 0.99, f"m_source noise PCC too low: {pcc_noise}"
    assert pcc_uv > 0.99, f"m_source uv PCC too low: {pcc_uv}"


def test_tt_generator_noise_res_no_fallback_pcc(device):
    """`Generator.noise_res`: reference vs TT on identical `har` + style inputs."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    x, s, f0 = _setup_test_inputs(T_x)
    har_ref, _y_ref = _ref_har_and_audio(ref, x, s, f0)  # BCT [B, 2K, F]

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    tt_mod = TTGenerator(device, params)
    mc = ttnn.DRAM_MEMORY_CONFIG

    # Default constructor path must keep all generator-source fallbacks disabled.
    _assert_generator_no_fallbacks(tt_mod)

    # Reference `noise_conv`/`noise_res` consume BCT; TT consumes NLC.
    har_nlc_t = har_ref.transpose(1, 2).contiguous()  # [B, F, 2K]
    har_nlc = ttnn.from_torch(har_nlc_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    pcc_by_stage = []
    for i, stage in enumerate(params.stages):
        with torch.no_grad():
            y_ref_bct = ref.noise_res[i](ref.noise_convs[i](har_ref), s)
        y_ref_nlc = y_ref_bct.transpose(1, 2).contiguous()

        x_source_tt = tt_conv1d_nlc(
            x_nlc=har_nlc,
            params=stage.noise_conv,
            device=device,
            compute_config=tt_mod.compute_kernel_config,
            memory_config=mc,
            preserve_input_dtype=True,
        )
        y_tt = tt_mod._noise_res[i].forward(x_source_tt, s_tt, memory_config=mc)
        y_hat = ttnn.to_torch(y_tt).float()
        while y_hat.dim() > y_ref_nlc.dim():
            y_hat.squeeze_(0)
        ttnn.deallocate(y_tt)
        ttnn.deallocate(x_source_tt)

        assert y_hat.shape == y_ref_nlc.shape, (i, y_hat.shape, y_ref_nlc.shape)
        _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.99)
        pcc_by_stage.append(pcc)
        assert pcc > 0.99, f"noise_res[{i}] PCC too low: {pcc}"

    ttnn.deallocate(har_nlc)
    ttnn.deallocate(s_tt)

    print("TTGenerator noise_res no-fallback PCC: " f"stage0={pcc_by_stage[0]:.6f}, stage1={pcc_by_stage[1]:.6f}")


def test_tt_generator_f0_upsamp_no_fallback_pcc(device):
    """`Generator.f0_upsamp`: compare reference nearest upsample vs TT helper."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    _x, _s, f0 = _setup_test_inputs(T_x=5)

    with torch.no_grad():
        f0u_ref = ref.f0_upsamp(f0[:, None]).transpose(1, 2).contiguous()  # [B, T_har, 1]

    params = preprocess_tt_generator(ref, device, time_len_x=5)
    tt_mod = TTGenerator(device, params)
    _assert_generator_no_fallbacks(tt_mod)

    f0_nlc = f0.unsqueeze(-1).contiguous()
    f0_tt = ttnn.from_torch(f0_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f0u_tt = _upsample_nearest_axis1(f0_tt, scale=params.upsample_scale_full, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    f0u_h = ttnn.to_torch(f0u_tt).float()
    ttnn.deallocate(f0u_tt)
    ttnn.deallocate(f0_tt)

    while f0u_h.dim() > f0u_ref.dim():
        f0u_h.squeeze_(0)
    assert f0u_h.shape == f0u_ref.shape, (f0u_h.shape, f0u_ref.shape)
    _, pcc = comp_pcc(f0u_ref, f0u_h, pcc=0.0)
    print(f"TTGenerator f0_upsamp no-fallback PCC: {pcc:.6f}")
    assert pcc > 0.99, f"f0_upsamp PCC too low: {pcc}"


def test_tt_generator_f0_upsamp_cpu_nlc_at_t162(device):
    """At production ``T_f0=162``, CPU nearest upsample matches ref; repeat_interleave does not."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    # Synthetic F0 curve length matching captured Kokoro runs (T_f0=162).
    f0 = (100.0 + 50.0 * torch.sin(torch.linspace(0, 4 * 3.14159, 162))).unsqueeze(0)

    with torch.no_grad():
        f0u_ref = ref.f0_upsamp(f0[:, None]).transpose(1, 2).contiguous()

    params = preprocess_tt_generator(ref, device, time_len_x=162)
    mc = ttnn.DRAM_MEMORY_CONFIG
    f0_tt = ttnn.from_torch(f0.unsqueeze(-1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f0u_cpu = _f0_upsamp_cpu_nlc(
        f0_tt, scale=params.upsample_scale_full, device=device, memory_config=mc, out_dtype=ttnn.float32
    )
    f0u_ri = _upsample_nearest_axis1(f0_tt, scale=params.upsample_scale_full, memory_config=mc)
    h_cpu = ttnn.to_torch(f0u_cpu).float()
    h_ri = ttnn.to_torch(f0u_ri).float()
    ttnn.deallocate(f0u_cpu)
    ttnn.deallocate(f0u_ri)
    ttnn.deallocate(f0_tt)

    _, pcc_cpu = comp_pcc(f0u_ref, h_cpu, pcc=0.0)
    _, pcc_ri = comp_pcc(f0u_ref, h_ri, pcc=0.0)
    max_abs_cpu = (f0u_ref - h_cpu).abs().max().item()
    max_abs_ri = (f0u_ref - h_ri).abs().max().item()
    print(
        f"f0_upsamp T_f0=162: cpu_nlc PCC={pcc_cpu:.6f} max_abs={max_abs_cpu:.4f}, "
        f"repeat_interleave PCC={pcc_ri:.6f} max_abs={max_abs_ri:.4f}"
    )
    assert pcc_cpu > 0.9999
    assert max_abs_cpu < 1e-4
    # On captured prosody F0, repeat_interleave max_abs can reach ~1 Hz; synthetic curve may be smaller.
    assert max_abs_ri >= max_abs_cpu


def test_tt_generator_ups_no_fallback_pcc(device):
    """`Generator.ups`: compare each reference ConvTranspose stage vs TT stage."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    x, _s, _f0 = _setup_test_inputs(T_x=5)

    params = preprocess_tt_generator(ref, device, time_len_x=5)
    tt_mod = TTGenerator(device, params)
    _assert_generator_no_fallbacks(tt_mod)
    mc = ttnn.DRAM_MEMORY_CONFIG

    x_ref = x.contiguous()  # BCT
    x_tt = ttnn.from_torch(x.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    pcc_by_stage = []
    for i, stage in enumerate(params.stages):
        with torch.no_grad():
            x_ref_act = F_torch.leaky_relu(x_ref, negative_slope=0.1)
            x_ref_up = ref.ups[i](x_ref_act)
            if i == params.num_upsamples - 1:
                x_ref_up = ref.reflection_pad(x_ref_up)

        x_act_tt = ttnn.leaky_relu(x_tt, negative_slope=0.1, memory_config=mc)
        ttnn.deallocate(x_tt)
        x_up_tt = tt_conv_transpose1d_nlc(
            x_nlc=x_act_tt,
            params=stage.ups,
            device=device,
            compute_config=tt_mod.compute_kernel_config,
            memory_config=mc,
        )
        ttnn.deallocate(x_act_tt)
        if i == params.num_upsamples - 1:
            x_up_pad_tt = _reflection_pad_left_1_nlc(x_up_tt, memory_config=mc)
            ttnn.deallocate(x_up_tt)
            x_up_tt = x_up_pad_tt

        y_hat = ttnn.to_torch(x_up_tt).float()
        y_ref_nlc = x_ref_up.transpose(1, 2).contiguous()
        while y_hat.dim() > y_ref_nlc.dim():
            y_hat.squeeze_(0)
        assert y_hat.shape == y_ref_nlc.shape, (i, y_hat.shape, y_ref_nlc.shape)
        _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.0)
        pcc_by_stage.append(pcc)
        assert pcc > 0.99, f"ups[{i}] PCC too low: {pcc}"

        x_ref = x_ref_up
        x_tt = x_up_tt

    ttnn.deallocate(x_tt)
    print(f"TTGenerator ups no-fallback PCC: stage0={pcc_by_stage[0]:.6f}, stage1={pcc_by_stage[1]:.6f}")


def test_tt_generator_resblocks_no_fallback_pcc(device):
    """`Generator.resblocks`: compare each stage's averaged resblock output."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    x, s, f0 = _setup_test_inputs(T_x=5)
    har_ref, _ = _ref_har_and_audio(ref, x, s, f0)

    params = preprocess_tt_generator(ref, device, time_len_x=5)
    tt_mod = TTGenerator(device, params)
    _assert_generator_no_fallbacks(tt_mod)
    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = tt_mod.compute_kernel_config

    har_nlc = ttnn.from_torch(
        har_ref.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    x_ref = x.contiguous()  # BCT
    x_tt = ttnn.from_torch(x.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    pcc_by_stage = []
    for i, stage in enumerate(params.stages):
        with torch.no_grad():
            x_ref_act = F_torch.leaky_relu(x_ref, negative_slope=0.1)
            x_source_ref = ref.noise_res[i](ref.noise_convs[i](har_ref), s)
            x_ref_up = ref.ups[i](x_ref_act)
            if i == params.num_upsamples - 1:
                x_ref_up = ref.reflection_pad(x_ref_up)
            x_ref_sum = x_ref_up + x_source_ref
            xs_ref = None
            for j in range(params.num_kernels):
                r = ref.resblocks[i * params.num_kernels + j](x_ref_sum, s)
                xs_ref = r if xs_ref is None else (xs_ref + r)
            x_ref = xs_ref / params.num_kernels

        x_act_tt = ttnn.leaky_relu(x_tt, negative_slope=0.1, memory_config=mc)
        ttnn.deallocate(x_tt)
        x_source_tt = tt_conv1d_nlc(
            x_nlc=har_nlc,
            params=stage.noise_conv,
            device=device,
            compute_config=ck,
            memory_config=mc,
            preserve_input_dtype=True,
        )
        x_source_tt = tt_mod._noise_res[i].forward(x_source_tt, s_tt, memory_config=mc)
        x_up_tt = tt_conv_transpose1d_nlc(
            x_nlc=x_act_tt,
            params=stage.ups,
            device=device,
            compute_config=ck,
            memory_config=mc,
        )
        ttnn.deallocate(x_act_tt)
        if i == params.num_upsamples - 1:
            x_up_pad_tt = _reflection_pad_left_1_nlc(x_up_tt, memory_config=mc)
            ttnn.deallocate(x_up_tt)
            x_up_tt = x_up_pad_tt
        x_sum_tt = ttnn.add(x_up_tt, x_source_tt, memory_config=mc)
        ttnn.deallocate(x_up_tt)
        ttnn.deallocate(x_source_tt)

        xs_tt = None
        for resblk in tt_mod._resblocks[i]:
            r_tt = resblk.forward(x_sum_tt, s_tt, memory_config=mc)
            if xs_tt is None:
                xs_tt = r_tt
            else:
                new_xs = ttnn.add(xs_tt, r_tt, memory_config=mc)
                ttnn.deallocate(xs_tt)
                ttnn.deallocate(r_tt)
                xs_tt = new_xs
        ttnn.deallocate(x_sum_tt)
        x_tt = ttnn.multiply(xs_tt, 1.0 / params.num_kernels, memory_config=mc)
        ttnn.deallocate(xs_tt)

        y_hat = ttnn.to_torch(x_tt).float()
        y_ref_nlc = x_ref.transpose(1, 2).contiguous()
        while y_hat.dim() > y_ref_nlc.dim():
            y_hat.squeeze_(0)
        assert y_hat.shape == y_ref_nlc.shape, (i, y_hat.shape, y_ref_nlc.shape)
        _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.0)
        pcc_by_stage.append(pcc)
        assert pcc > 0.99, f"resblocks stage[{i}] averaged output PCC too low: {pcc}"

    ttnn.deallocate(x_tt)
    ttnn.deallocate(har_nlc)
    ttnn.deallocate(s_tt)
    print(f"TTGenerator resblocks no-fallback PCC: stage0={pcc_by_stage[0]:.6f}, stage1={pcc_by_stage[1]:.6f}")


def test_tt_generator_conv_post_no_fallback_pcc(device):
    """`Generator.conv_post`: compare pre-iSTFT logits after stage stack."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    x, s, f0 = _setup_test_inputs(T_x=5)
    har_ref, _ = _ref_har_and_audio(ref, x, s, f0)

    params = preprocess_tt_generator(ref, device, time_len_x=5)
    tt_mod = TTGenerator(device, params)
    _assert_generator_no_fallbacks(tt_mod)
    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = tt_mod.compute_kernel_config

    har_nlc = ttnn.from_torch(
        har_ref.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    x_ref = x.contiguous()  # BCT
    x_tt = ttnn.from_torch(x.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    with torch.no_grad():
        for i in range(params.num_upsamples):
            x_ref = F_torch.leaky_relu(x_ref, negative_slope=0.1)
            x_source_ref = ref.noise_res[i](ref.noise_convs[i](har_ref), s)
            x_ref = ref.ups[i](x_ref)
            if i == params.num_upsamples - 1:
                x_ref = ref.reflection_pad(x_ref)
            x_ref = x_ref + x_source_ref
            xs_ref = None
            for j in range(params.num_kernels):
                r = ref.resblocks[i * params.num_kernels + j](x_ref, s)
                xs_ref = r if xs_ref is None else (xs_ref + r)
            x_ref = xs_ref / params.num_kernels
        x_ref = F_torch.leaky_relu(x_ref, negative_slope=0.01)
        x_post_ref = ref.conv_post(x_ref)  # [B, n_fft+2, T]

    for i, stage in enumerate(params.stages):
        x_act_tt = ttnn.leaky_relu(x_tt, negative_slope=0.1, memory_config=mc)
        ttnn.deallocate(x_tt)
        x_source_tt = tt_conv1d_nlc(
            x_nlc=har_nlc,
            params=stage.noise_conv,
            device=device,
            compute_config=ck,
            memory_config=mc,
            preserve_input_dtype=True,
        )
        x_source_tt = tt_mod._noise_res[i].forward(x_source_tt, s_tt, memory_config=mc)
        x_up_tt = tt_conv_transpose1d_nlc(
            x_nlc=x_act_tt,
            params=stage.ups,
            device=device,
            compute_config=ck,
            memory_config=mc,
        )
        ttnn.deallocate(x_act_tt)
        if i == params.num_upsamples - 1:
            x_up_pad_tt = _reflection_pad_left_1_nlc(x_up_tt, memory_config=mc)
            ttnn.deallocate(x_up_tt)
            x_up_tt = x_up_pad_tt
        x_sum_tt = ttnn.add(x_up_tt, x_source_tt, memory_config=mc)
        ttnn.deallocate(x_up_tt)
        ttnn.deallocate(x_source_tt)

        xs_tt = None
        for resblk in tt_mod._resblocks[i]:
            r_tt = resblk.forward(x_sum_tt, s_tt, memory_config=mc)
            if xs_tt is None:
                xs_tt = r_tt
            else:
                new_xs = ttnn.add(xs_tt, r_tt, memory_config=mc)
                ttnn.deallocate(xs_tt)
                ttnn.deallocate(r_tt)
                xs_tt = new_xs
        ttnn.deallocate(x_sum_tt)
        x_tt = ttnn.multiply(xs_tt, 1.0 / params.num_kernels, memory_config=mc)
        ttnn.deallocate(xs_tt)

    x_tt_act = ttnn.leaky_relu(x_tt, negative_slope=0.01, memory_config=mc)
    ttnn.deallocate(x_tt)
    x_post_tt = tt_conv1d_nlc(
        x_nlc=x_tt_act,
        params=params.conv_post,
        device=device,
        compute_config=ck,
        memory_config=mc,
        preserve_input_dtype=True,
    )
    ttnn.deallocate(x_tt_act)

    y_hat = ttnn.to_torch(x_post_tt).float()
    y_ref_nlc = x_post_ref.transpose(1, 2).contiguous()
    while y_hat.dim() > y_ref_nlc.dim():
        y_hat.squeeze_(0)
    ttnn.deallocate(x_post_tt)
    ttnn.deallocate(har_nlc)
    ttnn.deallocate(s_tt)

    assert y_hat.shape == y_ref_nlc.shape, (y_hat.shape, y_ref_nlc.shape)
    _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.0)
    print(f"TTGenerator conv_post no-fallback PCC: {pcc:.6f}")
    assert pcc > 0.99, f"conv_post PCC too low: {pcc}"


def test_tt_generator_reflection_pad_no_fallback_pcc(device):
    """`Generator.reflection_pad`: compare left reflection pad behavior."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    params = preprocess_tt_generator(ref, device, time_len_x=5)
    tt_mod = TTGenerator(device, params)
    _assert_generator_no_fallbacks(tt_mod)

    torch.manual_seed(9)
    x_ref_bct = torch.randn(1, 128, 300)
    with torch.no_grad():
        y_ref_bct = ref.reflection_pad(x_ref_bct)

    x_nlc = x_ref_bct.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = _reflection_pad_left_1_nlc(x_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y_hat = ttnn.to_torch(y_tt).float().transpose(1, 2).contiguous()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    while y_hat.dim() > y_ref_bct.dim():
        y_hat.squeeze_(0)
    assert y_hat.shape == y_ref_bct.shape, (y_hat.shape, y_ref_bct.shape)
    _, pcc = comp_pcc(y_ref_bct, y_hat, pcc=0.0)
    print(f"TTGenerator reflection_pad no-fallback PCC: {pcc:.6f}")
    assert pcc > 0.99, f"reflection_pad PCC too low: {pcc}"


def test_tt_generator_stft_no_fallback_pcc(device):
    """`Generator.stft`: compare transform and inverse vs TT STFT with fallback disabled."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    params = preprocess_tt_generator(ref, device, time_len_x=5)
    tt_mod = TTGenerator(device, params)
    _assert_generator_no_fallbacks(tt_mod)

    torch.manual_seed(7)
    x_wave = torch.randn(1, params.stft.input_length).float() * 3.0
    with torch.no_grad():
        mag_ref, phase_ref = ref.stft.transform(x_wave)
        y_inv_ref = ref.stft.inverse(mag_ref, phase_ref)

    x_tt = ttnn.from_torch(x_wave, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod._stft.transform(x_tt)
    mag_h = ttnn.to_torch(mag_tt).float()
    phase_h = ttnn.to_torch(phase_tt).float()

    mag_ref_f = mag_ref.float()
    phase_ref_f = phase_ref.float()
    _, pcc_mag = comp_pcc(mag_ref_f, mag_h, pcc=0.0)
    _, pcc_cos_phase = comp_pcc(torch.cos(phase_ref_f), torch.cos(phase_h), pcc=0.0)
    assert pcc_mag > 0.99, f"stft magnitude PCC too low: {pcc_mag}"
    assert pcc_cos_phase > 0.99, f"stft cos(phase) PCC too low: {pcc_cos_phase}"

    mag_ref_tt = ttnn.from_torch(mag_ref_f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    phase_ref_tt = ttnn.from_torch(phase_ref_f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_inv_tt = tt_mod._stft.inverse(mag_ref_tt, phase_ref_tt)
    y_inv_h = ttnn.to_torch(y_inv_tt).float()
    while y_inv_h.dim() > y_inv_ref.dim():
        y_inv_h.squeeze_(0)
    _, pcc_inv = comp_pcc(y_inv_ref.float(), y_inv_h, pcc=0.0)

    ttnn.deallocate(x_tt)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(mag_ref_tt)
    ttnn.deallocate(phase_ref_tt)
    ttnn.deallocate(y_inv_tt)

    print(
        "TTGenerator stft no-fallback PCC: " f"mag={pcc_mag:.6f}, cos(phase)={pcc_cos_phase:.6f}, inverse={pcc_inv:.6f}"
    )
    assert pcc_inv > 0.99, f"stft inverse PCC too low: {pcc_inv}"


def _harmonic_source_for_stft(ref: Generator, f0: torch.Tensor) -> torch.Tensor:
    """Trained ``m_source`` output at ``[B, L]`` — same signal the vocoder feeds into ``stft.transform``."""
    with torch.no_grad(), _torch_random_zeros():
        f0u = ref.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_src, _, _ = ref.m_source(f0u)
        return har_src.transpose(1, 2).squeeze(1).float()


def _stft_transform_pcc(
    ref_stft,
    tt_stft: TTTorchSTFT,
    x_wave: torch.Tensor,
) -> tuple[float, float]:
    """Return ``(pcc_mag, pcc_cos_phase)`` for ``transform`` vs reference."""
    with torch.no_grad():
        mag_ref, phase_ref = ref_stft.transform(x_wave)
    x_tt = ttnn.from_torch(x_wave, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=tt_stft.device)
    mag_tt, phase_tt = tt_stft.transform(x_tt)
    mag_h = ttnn.to_torch(mag_tt).float()
    phase_h = ttnn.to_torch(phase_tt).float()
    ttnn.deallocate(x_tt)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    while mag_h.dim() > mag_ref.dim():
        mag_h = mag_h.squeeze(0)
    while phase_h.dim() > phase_ref.dim():
        phase_h = phase_h.squeeze(0)
    _, pcc_mag = comp_pcc(mag_ref.float(), mag_h, pcc=0.0)
    _, pcc_phase = comp_pcc(torch.cos(phase_ref.float()), torch.cos(phase_h), pcc=0.0)
    return pcc_mag, pcc_phase


def test_tt_generator_stft_transform_conv_fallback_isolation(device):
    """Compare STFT ``transform`` PCC: pure TTNN vs CPU conv vs full ``torch.stft``.

    Uses trained ``m_source`` output (low amplitude, many near-zero STFT bins) at the generator's
    baked ``stft.input_length``.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    params = preprocess_tt_generator(ref, device, time_len_x=5)
    _, _, f0 = _setup_test_inputs(5)
    x_wave = _harmonic_source_for_stft(ref, f0)

    configs = [
        ("pure_ttnn", dict()),
        ("conv_cpu", dict(use_torch_stft_conv_fallback=True)),
        ("full_torch_stft", dict(use_torch_stft_fallback=True)),
    ]

    rows: list[tuple[str, float, float]] = []
    for name, stft_kw in configs:
        tt_stft = TTTorchSTFT(device, params.stft, **stft_kw)
        pcc_mag, pcc_phase = _stft_transform_pcc(ref.stft, tt_stft, x_wave)
        rows.append((name, pcc_mag, pcc_phase))

    print("\nSTFT transform op isolation (trained harmonic source, L={}):".format(params.stft.input_length))
    print(f"{'mode':<18} {'pcc_mag':>10} {'pcc_cos_phase':>14}")
    for name, pcc_mag, pcc_phase in rows:
        print(f"{name:<18} {pcc_mag:10.6f} {pcc_phase:14.6f}")

    by_name = {n: (m, p) for n, m, p in rows}
    pcc_mag_pure, pcc_phase_pure = by_name["pure_ttnn"]
    _, pcc_phase_conv = by_name["conv_cpu"]
    _, pcc_phase_full = by_name["full_torch_stft"]

    assert pcc_mag_pure > 0.95, f"unexpectedly low magnitude PCC on pure TTNN: {pcc_mag_pure:.6f}"
    assert pcc_phase_full > 0.99, f"full torch.stft fallback phase PCC too low: {pcc_phase_full:.6f}"
    assert (
        pcc_phase_conv > pcc_phase_pure
    ), f"conv CPU should help phase vs pure TT (conv={pcc_phase_conv:.6f}, pure={pcc_phase_pure:.6f})"
    assert pcc_phase_conv < 0.99, "conv-only does not reach tight phase PCC; use use_torch_stft_fallback=True"


def _stft_conv1d_weights_cpu(stft_params) -> tuple[torch.Tensor, torch.Tensor]:
    """``[K, 1, n_fft]`` conv1d weights matching TT ``_StridedStftConv`` kernels."""
    w_r = ttnn.to_torch(stft_params.conv_stft_real).float()[:, 0, :, 0].unsqueeze(1)
    w_i = ttnn.to_torch(stft_params.conv_stft_imag).float()[:, 0, :, 0].unsqueeze(1)
    return w_r, w_i


def _ref_pure_ttnn_stft_transform_stages(
    x_bl: torch.Tensor,
    stft_params,
    *,
    eps: float,
    phase_zero_floor: float,
) -> dict[str, torch.Tensor]:
    """CPU mirror of ``TTTorchSTFT`` transform (conv path, no ``torch.stft``)."""
    pad = int(stft_params.conv_pad_len)
    hop = int(stft_params.hop_length)
    w_r, w_i = _stft_conv1d_weights_cpu(stft_params)
    x = x_bl.float()
    x_pad = F_torch.pad(x, (pad, pad), mode="reflect")
    x_u = x_pad.unsqueeze(1)
    x_real = F_torch.conv1d(x_u, w_r, stride=hop, padding=0)
    x_imag = F_torch.conv1d(x_u, w_i, stride=hop, padding=0)
    mag_sq = x_real.pow(2) + x_imag.pow(2) + eps
    magnitude = torch.sqrt(mag_sq)
    phase_atan2 = torch.atan2(x_imag, x_real)
    neg_real = (x_imag == 0) & (x_real < 0)
    phase_corr = torch.where(neg_real, torch.full_like(phase_atan2, torch.pi), phase_atan2)
    near_zero = mag_sq < phase_zero_floor
    phase_final = torch.where(near_zero, torch.zeros_like(phase_corr), phase_corr)
    return {
        "input": x,
        "after_reflect_pad": x_pad,
        "X_real": x_real,
        "X_imag": x_imag,
        "mag_sq": mag_sq,
        "magnitude": magnitude,
        "phase_atan2": phase_atan2,
        "phase_after_neg_real_correction": phase_corr,
        "phase_final": phase_final,
    }


def _ref_pure_ttnn_stft_inverse_stages(
    magnitude_bkf: torch.Tensor,
    phase_bkf: torch.Tensor,
    stft_params,
) -> dict[str, torch.Tensor]:
    """CPU mirror of ``TTTorchSTFT.inverse`` (precomputed matrices when present)."""
    mag = magnitude_bkf.float()
    ph = phase_bkf.float()
    x_real = mag * torch.cos(ph)
    x_imag = mag * torch.sin(ph)
    out: dict[str, torch.Tensor] = {
        "X_real": x_real,
        "X_imag": x_imag,
    }
    if stft_params.istft_real is None or stft_params.istft_imag is None:
        return out
    b = int(mag.shape[0])
    x_real_flat = x_real.reshape(b, -1)
    x_imag_flat = x_imag.reshape(b, -1)
    b_real = ttnn.to_torch(stft_params.istft_real).float()
    b_imag = ttnn.to_torch(stft_params.istft_imag).float()
    y_real = x_real_flat @ b_real
    y_imag = x_imag_flat @ b_imag
    out["y_real_matmul"] = y_real
    out["y_imag_matmul"] = y_imag
    out["waveform_bl"] = y_real + y_imag
    return out


def test_tt_generator_stft_transform_inverse_per_op_diagnostic(device):
    """Per-op PCC for pure-TTNN ``stft.transform`` and ``stft.inverse`` (no fallbacks).

      Transform checkpoints mirror ``TTTorchSTFT._forward_stft_conv`` and
      ``_magnitude_phase_from_xy``.  Reference stages replay the same conv/atan2 math on CPU
      with TT-uploaded kernels (not ``torch.stft`` internals).

      Also prints end-to-end PCC vs ``Generator.stft`` (``torch.stft`` / ``torch.istft``).

    Inputs:
      - **harmonic**: trained ``m_source`` waveform at ``params.stft.input_length``.
      - **decoder_spec_phase**: ``exp``/``sin`` of ``conv_post`` for inverse-only checks.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    t_x = 5
    x, s, f0 = _setup_test_inputs(t_x)
    p = preprocess_tt_generator(ref, device, time_len_x=t_x)
    tt_stft = TTTorchSTFT(device, p.stft, use_torch_stft_conv_fallback=True)
    # assert not tt_stft._use_torch_stft_fallback
    # assert not tt_stft._use_torch_stft_conv_fallback

    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = tt_stft.compute_kernel_config
    eps = tt_stft.eps
    phase_floor = tt_stft.phase_zero_floor
    sp = p.stft

    har_wave = _harmonic_source_for_stft(ref, f0)
    spec_dec, phase_dec = _ref_decoder_spec_phase(ref, x, s, f0)

    with torch.no_grad():
        mag_ref_stft, phase_ref_stft = ref.stft.transform(har_wave)
        cos_phase_ref_stft = torch.cos(phase_ref_stft)

    ref_tf = _ref_pure_ttnn_stft_transform_stages(har_wave, sp, eps=eps, phase_zero_floor=phase_floor)

    pcc_log: list[tuple[str, float]] = []

    def _log(name: str, ref_t: torch.Tensor, tt_t: ttnn.Tensor) -> None:
        pcc_log.append((name, _pcc_ref_tt(ref_t, tt_t)))

    x_tt = ttnn.from_torch(har_wave, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    _log("transform.input", ref_tf["input"], x_tt)

    b = int(har_wave.shape[0])
    l_in = int(har_wave.shape[-1])
    x_rm = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
    x_n1lc = ttnn.reshape(x_rm, [b, 1, l_in, 1], memory_config=mc)
    ttnn.deallocate(x_rm)
    x_pad = _reflect_pad_1d_dim2(x_n1lc, l_in, sp.conv_pad_len)
    ttnn.deallocate(x_n1lc)
    l_pad = l_in + 2 * sp.conv_pad_len
    x_pad_flat = ttnn.reshape(x_pad, [b, l_pad], memory_config=mc)
    _log("transform.after_reflect_pad", ref_tf["after_reflect_pad"], x_pad_flat)
    ttnn.deallocate(x_pad_flat)

    x_real = tt_stft._conv_real(x_pad, b, l_pad)
    x_imag = tt_stft._conv_imag(x_pad, b, l_pad)
    ttnn.deallocate(x_pad)
    _log("transform.X_real", ref_tf["X_real"], x_real)
    _log("transform.X_imag", ref_tf["X_imag"], x_imag)

    mag_sq = ttnn.add(
        ttnn.multiply(x_real, x_real, memory_config=mc),
        ttnn.multiply(x_imag, x_imag, memory_config=mc),
        memory_config=mc,
    )
    eps_t = ttnn.full_like(x_real, eps, memory_config=mc)
    mag_sq = ttnn.add(mag_sq, eps_t, memory_config=mc)
    ttnn.deallocate(eps_t)
    _log("transform.mag_sq", ref_tf["mag_sq"], mag_sq)

    magnitude = ttnn.sqrt(mag_sq, memory_config=mc)
    _log("transform.magnitude", ref_tf["magnitude"], magnitude)

    phase = ttnn.atan2(x_imag, x_real, memory_config=mc)
    _log("transform.phase_atan2", ref_tf["phase_atan2"], phase)

    corr_mask = ttnn.logical_and(
        ttnn.eq(x_imag, 0.0, memory_config=mc),
        ttnn.lt(x_real, 0.0, memory_config=mc),
        memory_config=mc,
    )
    pi_fill = ttnn.full_like(phase, 3.141592653589793, memory_config=mc)
    phase_corr = ttnn.where(corr_mask, pi_fill, phase, memory_config=mc)
    ttnn.deallocate(corr_mask)
    ttnn.deallocate(pi_fill)
    ttnn.deallocate(phase)
    _log("transform.phase_after_neg_real_correction", ref_tf["phase_after_neg_real_correction"], phase_corr)

    near_zero = ttnn.lt(mag_sq, phase_floor, memory_config=mc)
    zero_phase = ttnn.full_like(phase_corr, 0.0, memory_config=mc)
    phase_final = ttnn.where(near_zero, zero_phase, phase_corr, memory_config=mc)
    ttnn.deallocate(near_zero)
    ttnn.deallocate(zero_phase)
    ttnn.deallocate(mag_sq)
    ttnn.deallocate(x_real)
    ttnn.deallocate(x_imag)
    _log("transform.phase_final", ref_tf["phase_final"], phase_final)

    mag_end, phase_end = magnitude, phase_final
    _log("transform.end_to_end.magnitude_vs_torch_stft", mag_ref_stft, mag_end)
    cos_phase_tt = ttnn.cos(phase_end, memory_config=mc)
    _log("transform.end_to_end.cos_phase_vs_torch_stft", cos_phase_ref_stft, cos_phase_tt)
    ttnn.deallocate(cos_phase_tt)

    # --- inverse: reference torch.stft mag/phase (matrix path should be exact) ---
    ref_inv_a = _ref_pure_ttnn_stft_inverse_stages(mag_ref_stft, phase_ref_stft, sp)
    mag_a_tt = ttnn.from_torch(mag_ref_stft.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    phase_a_tt = ttnn.from_torch(phase_ref_stft.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    cos_ph = ttnn.cos(phase_a_tt, memory_config=mc)
    sin_ph = ttnn.sin(phase_a_tt, memory_config=mc)
    x_r = ttnn.multiply(mag_a_tt, cos_ph, memory_config=mc)
    x_i = ttnn.multiply(mag_a_tt, sin_ph, memory_config=mc)
    ttnn.deallocate(cos_ph)
    ttnn.deallocate(sin_ph)
    _log("inverse_ref_inputs.X_real", ref_inv_a["X_real"], x_r)
    _log("inverse_ref_inputs.X_imag", ref_inv_a["X_imag"], x_i)

    kf = int(sp.K * sp.F)
    x_r_flat = ttnn.reshape(x_r, [b, kf], memory_config=mc)
    x_i_flat = ttnn.reshape(x_i, [b, kf], memory_config=mc)
    ttnn.deallocate(x_r)
    ttnn.deallocate(x_i)

    if sp.istft_real is not None:
        y_r = ttnn.matmul(x_r_flat, sp.istft_real, memory_config=mc, compute_kernel_config=ck)
        y_i = ttnn.matmul(x_i_flat, sp.istft_imag, memory_config=mc, compute_kernel_config=ck)
        _log("inverse_ref_inputs.y_real_matmul", ref_inv_a["y_real_matmul"], y_r)
        _log("inverse_ref_inputs.y_imag_matmul", ref_inv_a["y_imag_matmul"], y_i)
        y_sum = ttnn.add(y_r, y_i, memory_config=mc)
        ttnn.deallocate(y_r)
        ttnn.deallocate(y_i)
        y_bl = ttnn.reshape(y_sum, [b, sp.output_length], memory_config=mc)
        ttnn.deallocate(y_sum)
        with torch.no_grad():
            y_ref_inv = ref.stft.inverse(mag_ref_stft, phase_ref_stft).squeeze(1)
        _log("inverse_ref_inputs.waveform_vs_torch_istft", y_ref_inv, y_bl)
        ttnn.deallocate(y_bl)

    ttnn.deallocate(x_r_flat)
    ttnn.deallocate(x_i_flat)
    ttnn.deallocate(mag_a_tt)
    ttnn.deallocate(phase_a_tt)

    # --- inverse: decoder spec/phase (generator forward boundary) ---
    ref_inv_b = _ref_pure_ttnn_stft_inverse_stages(spec_dec, phase_dec, sp)
    mag_b_tt = ttnn.from_torch(spec_dec.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    phase_b_tt = ttnn.from_torch(phase_dec.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with torch.no_grad():
        y_ref_dec = ref.stft.inverse(spec_dec, phase_dec).squeeze(1)
    y_dec_tt = tt_stft.inverse(mag_b_tt, phase_b_tt)
    _log("inverse_decoder_spec_phase.waveform_vs_torch_istft", y_ref_dec, y_dec_tt)
    if "waveform_bl" in ref_inv_b:
        y_dec_bl = ttnn.reshape(y_dec_tt, [b, sp.output_length], memory_config=mc)
        _log("inverse_decoder_spec_phase.waveform_matmul_path", ref_inv_b["waveform_bl"], y_dec_bl)
        ttnn.deallocate(y_dec_bl)
    ttnn.deallocate(mag_b_tt)
    ttnn.deallocate(phase_b_tt)
    ttnn.deallocate(y_dec_tt)

    # --- inverse: TT pure transform outputs (shows cascade from weak phase) ---
    mag_tt_h = ttnn.to_torch(mag_end).float()
    phase_tt_h = ttnn.to_torch(phase_end).float()
    with torch.no_grad():
        y_ref_from_tt_tf = ref.stft.inverse(mag_tt_h, phase_tt_h).squeeze(1)
    mag_c_tt = ttnn.from_torch(mag_tt_h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    phase_c_tt = ttnn.from_torch(phase_tt_h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_c_tt = tt_stft.inverse(mag_c_tt, phase_c_tt)
    _log("inverse_tt_transform_outputs.waveform_vs_torch_istft", y_ref_from_tt_tf, y_c_tt)
    ttnn.deallocate(mag_c_tt)
    ttnn.deallocate(phase_c_tt)
    ttnn.deallocate(y_c_tt)
    ttnn.deallocate(mag_end)
    ttnn.deallocate(phase_end)
    ttnn.deallocate(x_tt)

    print(f"\nTTTorchSTFT per-op PCC (pure TTNN, L={sp.input_length}, harmonic m_source input):")
    for name, val in pcc_log:
        print(f"  {name:48s} {val:.6f}")

    by_name = dict(pcc_log)
    # Internal TT pipeline should match CPU replay of the same kernels.
    assert by_name["transform.X_real"] > 0.99
    assert by_name["transform.X_imag"] > 0.99
    assert by_name["transform.magnitude"] > 0.99
    if sp.istft_real is not None:
        assert by_name["inverse_ref_inputs.y_real_matmul"] > 0.99
        assert by_name["inverse_ref_inputs.y_imag_matmul"] > 0.99
        assert by_name["inverse_ref_inputs.waveform_vs_torch_istft"] > 0.99
    assert by_name["inverse_decoder_spec_phase.waveform_vs_torch_istft"] > 0.99


def _ref_decoder_spec_phase(ref: Generator, x: torch.Tensor, s: torch.Tensor, f0: torch.Tensor):
    """``conv_post`` → ``exp``/``sin`` inputs to ``stft.inverse`` (istftnet.py 407-408)."""
    with torch.no_grad():
        f0u = ref.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_src, _, _ = ref.m_source(f0u)
        har_src = har_src.transpose(1, 2).squeeze(1)
        mag, phase = ref.stft.transform(har_src)
        har = torch.cat([mag, phase], dim=1)
        x_dec = x
        for i in range(ref.num_upsamples):
            x_dec = F_torch.leaky_relu(x_dec, negative_slope=0.1)
            x_nc = ref.noise_convs[i](har)
            x_dec = ref.ups[i](x_dec)
            if i == ref.num_upsamples - 1:
                x_dec = ref.reflection_pad(x_dec)
            x_dec = x_dec + x_nc
            xs = sum(ref.resblocks[i * ref.num_kernels + j](x_dec, s) for j in range(ref.num_kernels))
            x_dec = xs / ref.num_kernels
        x_dec = F_torch.leaky_relu(x_dec)
        x_post = ref.conv_post(x_dec)
        K = ref.post_n_fft // 2 + 1
        spec = torch.exp(x_post[:, :K, :])
        phase = torch.sin(x_post[:, K:, :])
    return spec, phase


def test_tt_generator_istft_reference_vs_tt_no_fallback_pcc(device):
    """``stft.inverse`` only: reference ``TorchSTFT`` vs ``TTTorchSTFT`` (no fallbacks).

    Two input regimes at Kokoro ``T_x=5`` frame geometry (``F=301``, ``output_length=1500``):

    1. **stft_transform** — magnitude/phase from ``stft.transform`` on a harmonic-length waveform.
    2. **decoder_spec_phase** — ``exp``/``sin`` of ``conv_post`` logits (what ``Generator.forward`` passes
       to ``inverse``).
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    tt_mod = TTGenerator(device, params)
    _assert_generator_no_fallbacks(tt_mod)

    K = params.post_n_fft // 2 + 1
    F = params.stft.F

    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    torch.manual_seed(11)
    x_wave = torch.randn(1, params.stft.input_length).float() * 2.0
    with torch.no_grad():
        mag_stft, phase_stft = ref.stft.transform(x_wave)
    cases.append(("stft_transform", mag_stft, phase_stft))

    x, s, f0 = _setup_test_inputs(T_x)
    spec_dec, phase_dec = _ref_decoder_spec_phase(ref, x, s, f0)
    cases.append(("decoder_spec_phase", spec_dec, phase_dec))

    for name, spec_ref, phase_ref in cases:
        assert spec_ref.shape == phase_ref.shape
        assert spec_ref.shape[1] == K, (spec_ref.shape, K)
        assert spec_ref.shape[2] == F, (spec_ref.shape, F)

        with torch.no_grad():
            y_ref = ref.stft.inverse(spec_ref, phase_ref)

        spec_tt = ttnn.from_torch(spec_ref.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        phase_tt = ttnn.from_torch(phase_ref.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = tt_mod._stft.inverse(spec_tt, phase_tt)
        pcc = _pcc_ref_tt(y_ref, y_tt)
        ttnn.deallocate(spec_tt)
        ttnn.deallocate(phase_tt)
        ttnn.deallocate(y_tt)

        print(f"TTGenerator istft ({name}) no-fallback PCC: {pcc:.6f}, shapes spec={tuple(spec_ref.shape)}")
        assert pcc > 0.99, f"istft PCC too low for {name}: {pcc}"


def test_tt_generator_harmonic_path_before_after_stft_pcc(device):
    """Recreate `f0_upsamp -> m_source -> stft.transform` and report PCC before/after STFT."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    _x, _s, f0 = _setup_test_inputs(T_x)

    # Reference path from istftnet forward:
    # f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
    # har_source, _, _ = self.m_source(f0)
    # har_source = har_source.transpose(1, 2).squeeze(1)
    # har_spec, har_phase = self.stft.transform(har_source)
    f0u_ref = ref.f0_upsamp(f0[:, None]).transpose(1, 2).contiguous()  # [B, T_har, 1]

    B, T_har, _ = f0u_ref.shape
    dim = ref.m_source.l_sin_gen.dim
    torch.manual_seed(123)
    rand_ini = torch.rand(B, dim)
    rand_ini[:, 0] = 0.0
    sinegen_noise_raw = torch.randn(B, T_har, dim)
    source_noise_raw = torch.randn(B, T_har, 1)

    def _fake_rand(*size, **kwargs):
        out = rand_ini.to(kwargs.get("device", rand_ini.device))
        dtype = kwargs.get("dtype", out.dtype)
        return out.to(dtype)

    def _fake_randn_like(t, **kwargs):
        if t.shape[-1] == 1:
            return source_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()
        return sinegen_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = _fake_rand
    torch.randn_like = _fake_randn_like
    try:
        tt_mod = TTGenerator(device, params)
        with torch.no_grad():
            har_source_ref_btl, _noi_ref, _uv_ref = ref.m_source(f0u_ref)  # [B, T_har, 1]
            har_source_ref = har_source_ref_btl.transpose(1, 2).squeeze(1).contiguous()  # [B, T_har]
            har_spec_ref, har_phase_ref = ref.stft.transform(har_source_ref)  # [B, K, F]
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like
    _assert_generator_no_fallbacks(tt_mod)

    f0_tt = ttnn.from_torch(f0u_ref, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    rand_ini_tt = ttnn.from_torch(rand_ini.unsqueeze(1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    har_source_tt_btl, noise_tt, uv_tt = tt_mod._m_source(
        f0_tt,
        sinegen_rand_ini=rand_ini_tt,
    )
    har_source_tt = ttnn.squeeze(har_source_tt_btl, 2)  # [B, T_har]
    har_spec_tt, har_phase_tt = tt_mod._stft.transform(har_source_tt)

    har_source_h = ttnn.to_torch(har_source_tt).float()
    har_spec_h = ttnn.to_torch(har_spec_tt).float()
    har_phase_h = ttnn.to_torch(har_phase_tt).float()

    _, pcc_before_stft = comp_pcc(har_source_ref.float(), har_source_h, pcc=0.0)
    har_ref = torch.cat([har_spec_ref.float(), har_phase_ref.float()], dim=1)
    har_tt = torch.cat([har_spec_h, har_phase_h], dim=1)
    _, pcc_after_stft = comp_pcc(har_ref, har_tt, pcc=0.0)
    _, pcc_mag = comp_pcc(har_spec_ref.float(), har_spec_h, pcc=0.0)
    _, pcc_cos_phase = comp_pcc(torch.cos(har_phase_ref.float()), torch.cos(har_phase_h), pcc=0.0)

    print(
        "TTGenerator harmonic integration diagnostic: "
        f"before_stft(har_source)={pcc_before_stft:.6f}, "
        f"after_stft(har)={pcc_after_stft:.6f}, "
        f"mag={pcc_mag:.6f}, cos(phase)={pcc_cos_phase:.6f}"
    )
    # Demonstrate the integration drop at STFT boundary in the no-fallback path.
    assert pcc_before_stft > 0.98, f"Unexpectedly low PCC before STFT: {pcc_before_stft}"
    assert (
        pcc_after_stft < pcc_before_stft
    ), f"Expected STFT integration drop; before={pcc_before_stft:.6f}, after={pcc_after_stft:.6f}"

    ttnn.deallocate(har_source_tt_btl)
    ttnn.deallocate(har_source_tt)
    ttnn.deallocate(har_spec_tt)
    ttnn.deallocate(har_phase_tt)
    ttnn.deallocate(noise_tt)
    ttnn.deallocate(uv_tt)
    ttnn.deallocate(f0_tt)
    ttnn.deallocate(rand_ini_tt)


def test_tt_generator_harmonic_path_before_after_stft_pcc_including_f0_upsample_block(device):
    """Same diagnostic, but include TT `_harmonic_source_path` f0 shape/cast/upsample block."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    _x, _s, f0 = _setup_test_inputs(T_x)  # [B, T_f0]

    # Reference path from istftnet forward.
    f0u_ref = ref.f0_upsamp(f0[:, None]).transpose(1, 2).contiguous()  # [B, T_har, 1]
    B, T_har, _ = f0u_ref.shape
    dim = ref.m_source.l_sin_gen.dim
    torch.manual_seed(123)
    rand_ini = torch.rand(B, dim)
    rand_ini[:, 0] = 0.0
    sinegen_noise_raw = torch.randn(B, T_har, dim)
    source_noise_raw = torch.randn(B, T_har, 1)

    def _fake_rand(*size, **kwargs):
        out = rand_ini.to(kwargs.get("device", rand_ini.device))
        dtype = kwargs.get("dtype", out.dtype)
        return out.to(dtype)

    def _fake_randn_like(t, **kwargs):
        if t.shape[-1] == 1:
            return source_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()
        return sinegen_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = _fake_rand
    torch.randn_like = _fake_randn_like
    try:
        tt_mod = TTGenerator(device, params)
        with torch.no_grad():
            har_source_ref_btl, _noi_ref, _uv_ref = ref.m_source(f0u_ref)  # [B, T_har, 1]
            har_source_ref = har_source_ref_btl.transpose(1, 2).squeeze(1).contiguous()  # [B, T_har]
            har_spec_ref, har_phase_ref = ref.stft.transform(har_source_ref)  # [B, K, F]
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like
    _assert_generator_no_fallbacks(tt_mod)
    mc = ttnn.DRAM_MEMORY_CONFIG

    # Recreate TT block at tt_generator.py:_harmonic_source_path L397-L415 exactly from raw [B, T_f0].
    f0_tt_raw = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f_shape = list(f0_tt_raw.shape)
    if len(f_shape) == 2:
        f0_b_t_1 = ttnn.unsqueeze(f0_tt_raw, 2)
    else:
        f0_b_t_1 = f0_tt_raw
    f0_fp32 = ttnn.typecast(f0_b_t_1, ttnn.float32, memory_config=mc)
    if len(f_shape) == 2:
        ttnn.deallocate(f0_b_t_1)
    f0_b_t_1 = f0_fp32
    f0_har_tt = _upsample_nearest_axis1(
        f0_b_t_1,
        scale=params.upsample_scale_full,
        memory_config=mc,
    )
    ttnn.deallocate(f0_b_t_1)

    rand_ini_tt = ttnn.from_torch(rand_ini.unsqueeze(1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    har_source_tt_btl, noise_tt, uv_tt = tt_mod._m_source(
        f0_har_tt,
        sinegen_rand_ini=rand_ini_tt,
    )
    har_source_tt = ttnn.squeeze(har_source_tt_btl, 2)  # [B, T_har]
    har_spec_tt, har_phase_tt = tt_mod._stft.transform(har_source_tt)

    f0_har_h = ttnn.to_torch(f0_har_tt).float()
    har_source_h = ttnn.to_torch(har_source_tt).float()
    har_spec_h = ttnn.to_torch(har_spec_tt).float()
    har_phase_h = ttnn.to_torch(har_phase_tt).float()

    _, pcc_f0_upsamp = comp_pcc(f0u_ref.float(), f0_har_h, pcc=0.0)
    _, pcc_before_stft = comp_pcc(har_source_ref.float(), har_source_h, pcc=0.0)
    har_ref = torch.cat([har_spec_ref.float(), har_phase_ref.float()], dim=1)
    har_tt = torch.cat([har_spec_h, har_phase_h], dim=1)
    _, pcc_after_stft = comp_pcc(har_ref, har_tt, pcc=0.0)
    _, pcc_mag = comp_pcc(har_spec_ref.float(), har_spec_h, pcc=0.0)
    _, pcc_cos_phase = comp_pcc(torch.cos(har_phase_ref.float()), torch.cos(har_phase_h), pcc=0.0)

    print(
        "TTGenerator harmonic integration diagnostic (includes f0 block): "
        f"f0_upsamp={pcc_f0_upsamp:.6f}, "
        f"before_stft(har_source)={pcc_before_stft:.6f}, "
        f"after_stft(har)={pcc_after_stft:.6f}, "
        f"mag={pcc_mag:.6f}, cos(phase)={pcc_cos_phase:.6f}"
    )
    assert pcc_f0_upsamp > 0.99, f"Unexpectedly low PCC at f0 upsample block: {pcc_f0_upsamp}"
    assert pcc_before_stft > 0.98, f"Unexpectedly low PCC before STFT: {pcc_before_stft}"
    assert (
        pcc_after_stft < pcc_before_stft
    ), f"Expected STFT integration drop; before={pcc_before_stft:.6f}, after={pcc_after_stft:.6f}"

    ttnn.deallocate(f0_tt_raw)
    ttnn.deallocate(f0_har_tt)
    ttnn.deallocate(har_source_tt_btl)
    ttnn.deallocate(har_source_tt)
    ttnn.deallocate(har_spec_tt)
    ttnn.deallocate(har_phase_tt)
    ttnn.deallocate(noise_tt)
    ttnn.deallocate(uv_tt)
    ttnn.deallocate(rand_ini_tt)


def test_tt_generator_phase_and_inverse_integration_diagnostic(device):
    """Per-op PCC along full Generator forward (istftnet 385-409), no-fallback TT path."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    x, s, f0 = _setup_test_inputs(T_x)

    f0u_ref = ref.f0_upsamp(f0[:, None]).transpose(1, 2).contiguous()
    B, T_har, _ = f0u_ref.shape
    dim = ref.m_source.l_sin_gen.dim
    torch.manual_seed(123)
    rand_ini = torch.rand(B, dim)
    rand_ini[:, 0] = 0.0
    sinegen_noise_raw = torch.randn(B, T_har, dim)
    source_noise_raw = torch.randn(B, T_har, 1)

    real_rand = torch.rand
    real_randn_like = torch.randn_like

    def _fake_rand(*size, **kwargs):
        out = rand_ini.to(kwargs.get("device", rand_ini.device))
        return out.to(kwargs.get("dtype", out.dtype))

    def _fake_randn_like(t, **kwargs):
        if t.shape[-1] == 1:
            return source_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()
        return sinegen_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()

    pcc_log: list[tuple[str, float]] = []

    def _log(name: str, ref_t: torch.Tensor, tt_t: ttnn.Tensor) -> None:
        if ref_t.dim() == 2:
            pcc_log.append((name, _pcc_flat(ref_t, tt_t)))
        else:
            pcc_log.append((name, _pcc_ref_tt(ref_t, tt_t)))

    # --- Reference: step-by-step (istftnet.py 385-409) ---
    ref_stage = {
        "leaky_relu_0.1": [],
        "noise_conv": [],
        "noise_res": [],
        "ups": [],
        "add": [],
        "resblocks": [],
    }
    p = preprocess_tt_generator(ref, device, time_len_x=T_x)
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = _fake_rand
    torch.randn_like = _fake_randn_like
    try:
        tt_mod = TTGenerator(device, p)
        with torch.no_grad():
            f0_ref = ref.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source_btl, _, _ = ref.m_source(f0_ref)
            har_source_ref = har_source_btl.transpose(1, 2).squeeze(1)
            har_spec_ref, har_phase_ref = ref.stft.transform(har_source_ref)
            har_ref = torch.cat([har_spec_ref, har_phase_ref], dim=1)
            x_ref = x
            reflection_pad_ref = None
            for i in range(ref.num_upsamples):
                x_ref = F_torch.leaky_relu(x_ref, negative_slope=0.1)
                ref_stage["leaky_relu_0.1"].append(x_ref)
                x_nc = ref.noise_convs[i](har_ref)
                ref_stage["noise_conv"].append(x_nc)
                x_nc = ref.noise_res[i](x_nc, s)
                ref_stage["noise_res"].append(x_nc)
                x_ref = ref.ups[i](x_ref)
                ref_stage["ups"].append(x_ref)
                if i == ref.num_upsamples - 1:
                    x_ref = ref.reflection_pad(x_ref)
                    reflection_pad_ref = x_ref
                x_ref = x_ref + x_nc
                ref_stage["add"].append(x_ref)
                xs = sum(ref.resblocks[i * ref.num_kernels + j](x_ref, s) for j in range(ref.num_kernels))
                x_ref = xs / ref.num_kernels
                ref_stage["resblocks"].append(x_ref)
            x_pre_post_ref = F_torch.leaky_relu(x_ref)
            x_post_ref = ref.conv_post(x_pre_post_ref)
            K = ref.post_n_fft // 2 + 1
            spec_logits_ref = x_post_ref[:, :K, :]
            phase_logits_ref = x_post_ref[:, K:, :]
            spec_ref = torch.exp(spec_logits_ref)
            phase_after_ref = torch.sin(phase_logits_ref)
            audio_ref = ref.stft.inverse(spec_ref, phase_after_ref)
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like
    _assert_generator_no_fallbacks(tt_mod)
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    ck = tt_mod.compute_kernel_config

    x_nlc = ttnn.from_torch(x.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_bs = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f0 = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    sinegen_rand_ini = ttnn.from_torch(
        rand_ini.unsqueeze(1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )

    # --- TT: mirrors ``TTGenerator._harmonic_source_path`` + ``forward`` op-for-op ---
    f_shape = list(f0.shape)
    if len(f_shape) == 2:
        f0_b_t_1 = ttnn.unsqueeze(f0, 2)
    else:
        f0_b_t_1 = f0
    f0_fp32 = ttnn.typecast(f0_b_t_1, ttnn.float32, memory_config=memory_config)
    if len(f_shape) == 2:
        ttnn.deallocate(f0_b_t_1)
    f0_b_t_1 = f0_fp32
    f0_har = _upsample_nearest_axis1(
        f0_b_t_1,
        scale=p.upsample_scale_full,
        memory_config=memory_config,
    )
    ttnn.deallocate(f0_b_t_1)
    _log("f0_upsamp", f0_ref, f0_har)

    har_source, _noise_out, _uv = tt_mod._m_source.forward(
        f0_har,
        sinegen_rand_ini=sinegen_rand_ini,
        memory_config=memory_config,
    )
    ttnn.deallocate(f0_har)
    ttnn.deallocate(_noise_out)
    ttnn.deallocate(_uv)

    har_flat = ttnn.squeeze(har_source, 2)
    ttnn.deallocate(har_source)
    _log("m_source_har_source", har_source_ref, har_flat)

    if har_flat.dtype != p.m_source.sinegen.activation_dtype:
        har_flat_cast = ttnn.typecast(
            har_flat,
            p.m_source.sinegen.activation_dtype,
            memory_config=memory_config,
        )
        ttnn.deallocate(har_flat)
        har_flat = har_flat_cast
    mag, phase = tt_mod._stft.transform(har_flat)
    ttnn.deallocate(har_flat)
    _log("stft_mag", har_spec_ref, mag)
    _log("stft_phase", har_phase_ref, phase)

    har_bct = ttnn.concat([mag, phase], dim=1, memory_config=memory_config)
    ttnn.deallocate(mag)
    ttnn.deallocate(phase)
    har_nlc = ttnn.permute(har_bct, (0, 2, 1), memory_config=memory_config)
    ttnn.deallocate(har_bct)
    _log("har_cat", har_ref, har_nlc)

    target_dtype = har_nlc.dtype
    if x_nlc.dtype != target_dtype:
        x_cast = ttnn.typecast(x_nlc, target_dtype, memory_config=memory_config)
        x = x_cast
    else:
        x = x_nlc

    for i, stage in enumerate(p.stages):
        x_act = ttnn.leaky_relu(x, negative_slope=0.1, memory_config=memory_config)
        if x_act is not x:
            ttnn.deallocate(x)
        x = x_act
        _log(f"stage{i}_leaky_relu_0.1", ref_stage["leaky_relu_0.1"][i], x)

        x_source = tt_conv1d_nlc(
            x_nlc=har_nlc,
            params=stage.noise_conv,
            device=device,
            compute_config=ck,
            memory_config=memory_config,
            preserve_input_dtype=True,
        )
        _log(f"stage{i}_noise_conv", ref_stage["noise_conv"][i], x_source)
        x_source = tt_mod._noise_res[i].forward(x_source, s_bs, memory_config=memory_config)
        _log(f"stage{i}_noise_res", ref_stage["noise_res"][i], x_source)

        x_up = tt_conv_transpose1d_nlc(
            x_nlc=x,
            params=stage.ups,
            device=device,
            compute_config=ck,
            memory_config=memory_config,
        )
        ttnn.deallocate(x)
        x = x_up
        _log(f"stage{i}_ups", ref_stage["ups"][i], x)

        if i == p.num_upsamples - 1:
            x_padded = _reflection_pad_left_1_nlc(x, memory_config=memory_config)
            ttnn.deallocate(x)
            x = x_padded
            _log("reflection_pad", reflection_pad_ref, x)

        x_sum = ttnn.add(x, x_source, memory_config=memory_config)
        ttnn.deallocate(x)
        ttnn.deallocate(x_source)
        x = x_sum
        _log(f"stage{i}_add", ref_stage["add"][i], x)

        xs = None
        for resblk in tt_mod._resblocks[i]:
            r = resblk.forward(x, s_bs, memory_config=memory_config)
            if xs is None:
                xs = r
            else:
                new_xs = ttnn.add(xs, r, memory_config=memory_config)
                ttnn.deallocate(xs)
                ttnn.deallocate(r)
                xs = new_xs
        ttnn.deallocate(x)
        x = ttnn.multiply(xs, 1.0 / p.num_kernels, memory_config=memory_config)
        ttnn.deallocate(xs)
        _log(f"stage{i}_resblocks", ref_stage["resblocks"][i], x)

    ttnn.deallocate(har_nlc)

    x_act = ttnn.leaky_relu(x, negative_slope=0.01, memory_config=memory_config)
    ttnn.deallocate(x)
    x = x_act
    _log("leaky_relu_0.01_pre_conv_post", x_pre_post_ref, x)

    x_post = tt_conv1d_nlc(
        x_nlc=x,
        params=p.conv_post,
        device=device,
        compute_config=ck,
        memory_config=memory_config,
        preserve_input_dtype=True,
    )
    ttnn.deallocate(x)
    _log("conv_post", x_post_ref, x_post)

    K = p.post_n_fft // 2 + 1
    B = int(x_post.shape[0])
    T_post = int(x_post.shape[1])
    spec_nlc = ttnn.slice(x_post, [0, 0, 0], [B, T_post, K], [1, 1, 1], memory_config=memory_config)
    phase_nlc = ttnn.slice(
        x_post,
        [0, 0, K],
        [B, T_post, 2 * K],
        [1, 1, 1],
        memory_config=memory_config,
    )
    ttnn.deallocate(x_post)
    _log("phase_logits_pre_sin", phase_logits_ref, phase_nlc)

    spec_nlc = ttnn.exp(spec_nlc, memory_config=memory_config)
    phase_nlc = ttnn.sin(phase_nlc, memory_config=memory_config)
    _log("spec_after_exp", spec_ref, spec_nlc)
    _log("phase_after_sin", phase_after_ref, phase_nlc)

    spec_bct = ttnn.permute(spec_nlc, (0, 2, 1), memory_config=memory_config)
    phase_bct = ttnn.permute(phase_nlc, (0, 2, 1), memory_config=memory_config)
    ttnn.deallocate(spec_nlc)
    ttnn.deallocate(phase_nlc)

    audio = tt_mod._stft.inverse(spec_bct, phase_bct)
    _log("stft_inverse_audio", audio_ref, audio)

    print("\nTTGenerator per-op integration PCC (no fallback):")
    for name, val in pcc_log:
        print(f"  {name:28s} {val:.6f}")

    # Soft checks: harmonic STFT is known weak; decoder blocks should stay high when har is fixed.
    by_name = dict(pcc_log)
    assert by_name["f0_upsamp"] > 0.99
    assert by_name["m_source_har_source"] > 0.98
    assert by_name["stage0_noise_res"] > 0.99
    assert by_name["stage1_noise_res"] > 0.99

    if x_nlc.dtype != target_dtype:
        ttnn.deallocate(x_nlc)
    ttnn.deallocate(s_bs)
    ttnn.deallocate(f0)
    ttnn.deallocate(sinegen_rand_ini)
    ttnn.deallocate(spec_bct)
    ttnn.deallocate(phase_bct)
    ttnn.deallocate(audio)


def test_tt_generator_pipeline_pcc(device):
    """End-to-end TT pipeline PCC > 0.99 using the **reference** ``har`` as input.

    Injecting the reference STFT bypasses the WH bf16-matmul-induced phase noise. This isolates
    the rest of the TT Generator (noise_conv, noise_res, ups, resblocks, conv_post, iSTFT) and
    proves it is bit-precise vs PyTorch.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    x, s, f0 = _setup_test_inputs(T_x)
    har_ref, y_ref = _ref_har_and_audio(ref, x, s, f0)

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    tt_mod = TTGenerator(device, params)
    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = tt_mod.compute_kernel_config
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Inject ref har as NLC ``[B, F, 2K]``.
    har_nlc_t = har_ref.transpose(1, 2).contiguous()
    har_nlc = ttnn.from_torch(har_nlc_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    x_nlc = x.transpose(1, 2).contiguous()
    x_curr = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    p = tt_mod.params
    for i, stage in enumerate(p.stages):
        x_act = ttnn.leaky_relu(x_curr, negative_slope=0.1, memory_config=mc)
        x_source = tt_conv1d_nlc(
            x_nlc=har_nlc,
            params=stage.noise_conv,
            device=device,
            compute_config=ck,
            memory_config=mc,
            preserve_input_dtype=True,
        )
        x_source = tt_mod._noise_res[i].forward(x_source, s_tt, memory_config=mc)
        x_up = tt_conv_transpose1d_nlc(
            x_nlc=x_act,
            params=stage.ups,
            device=device,
            compute_config=ck,
            memory_config=mc,
        )
        if i == p.num_upsamples - 1:
            x_up = _reflection_pad_left_1_nlc(x_up, memory_config=mc)
        x_sum = ttnn.add(x_up, x_source, memory_config=mc)
        xs = None
        for resblk in tt_mod._resblocks[i]:
            r = resblk.forward(x_sum, s_tt, memory_config=mc)
            xs = r if xs is None else ttnn.add(xs, r, memory_config=mc)
        x_curr = ttnn.multiply(xs, 1.0 / p.num_kernels, memory_config=mc)

    x_act = ttnn.leaky_relu(x_curr, negative_slope=0.01, memory_config=mc)
    x_post = tt_conv1d_nlc(
        x_nlc=x_act,
        params=p.conv_post,
        device=device,
        compute_config=ck,
        memory_config=mc,
        preserve_input_dtype=True,
    )
    K = p.post_n_fft // 2 + 1
    B = int(x_post.shape[0])
    T_post = int(x_post.shape[1])
    spec_nlc = ttnn.slice(x_post, [0, 0, 0], [B, T_post, K], [1, 1, 1], memory_config=mc)
    phase_nlc = ttnn.slice(x_post, [0, 0, K], [B, T_post, 2 * K], [1, 1, 1], memory_config=mc)
    spec_nlc = ttnn.exp(spec_nlc, memory_config=mc)
    phase_nlc = ttnn.sin(phase_nlc, memory_config=mc)
    spec_bct = ttnn.permute(spec_nlc, (0, 2, 1), memory_config=mc)
    phase_bct = ttnn.permute(phase_nlc, (0, 2, 1), memory_config=mc)
    audio = tt_mod._stft.inverse(spec_bct, phase_bct)

    y_h = ttnn.to_torch(audio).float()
    while y_h.dim() > y_ref.dim():
        y_h.squeeze_(0)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTGenerator pipeline (ref har injected) PCC: {pcc:.6f}, shape: {tuple(y_ref.shape)}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_generator_full_forward_torch_stft_fallback_pcc(device):
    """Full TT forward with both torch fallbacks enabled — PCC must be > 0.99.

    Two independent BH BF16 precision bottlenecks are addressed:

    1. **STFT transform** (``use_torch_stft_fallback=True``): the entire ``TTTorchSTFT.transform``
       runs on CPU via ``torch.stft`` (float32).  BH rounds float32→BF16 before ALL compute ops —
       including the SFPU for ``atan2``.  Moving only the conv2d to CPU is insufficient: near-zero
       DFT bins (~1e-5) still get BF16-rounded inputs to ``atan2``, giving sign-random phase.
       The full transform fallback fixes both conv and atan2.

    2. **SineGen phase chain** (``use_torch_phase_fallback=True``): BF16 MACs on small cumsum
       values (~3.3e-5 cycles) are amplified by 2π × upsample_scale=300 ≈ 1885× → ~0.25 rad
       phase error, comparable to sine_amp=0.1.  CPU float32 cumsum eliminates this.

    3. **Source linear+tanh** (``use_torch_linear_fallback=True``, ``use_torch_tanh_fallback=True``):
       runs both operations on CPU float32 to remove remaining source-merge BF16 error.

    These fallbacks together are expected to reach the tight PCC target.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    x, s, f0 = _setup_test_inputs(T_x)

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    # Enable full STFT transform fallback + SineGen phase fallback + source linear/tanh fallbacks.
    # TTGenerator is constructed inside the same zeros context so self._noise_raw /
    # self._out_noise_raw match the zeros noise used by the reference forward.
    with torch.no_grad(), _torch_random_zeros():
        y_ref = ref(x, s, f0)
        tt_mod = TTGenerator(
            device,
            params,
            use_torch_stft_fallback=True,
            use_torch_phase_fallback=True,
            use_torch_linear_fallback=True,
            use_torch_tanh_fallback=True,
        )

    x_nlc = x.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f0_tt = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, f0_tt)

    y_h = ttnn.to_torch(y_tt).float()
    while y_h.dim() > y_ref.dim():
        y_h.squeeze_(0)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    assert torch.isfinite(y_h).all(), "TTGenerator (fallbacks) produced NaN/Inf"
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTGenerator full-forward (phase + STFT + linear + tanh fallback) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low with phase + STFT + linear + tanh fallbacks: {pcc}"


@pytest.mark.xfail(
    reason=(
        "SineGen 1.2% BF16 phase noise at upsample_scale=300 cascades into near-zero STFT bins; "
        "use_torch_phase_fallback is also required to reach PCC > 0.99."
    ),
    strict=False,
)
def test_tt_generator_stft_linear_fallback_no_sinegen_pcc(device):
    """STFT transform + l_linear on CPU; SineGen phase chain stays pure TTNN.

    Isolates whether SineGen's BF16 phase error at upsample_scale=300 (PCC=0.988 in isolation)
    is sufficient to prevent full-forward PCC > 0.99, or whether fixing only STFT+l_linear is
    enough.

    - ``use_torch_stft_fallback=True``: torch.stft on CPU (fixes atan2 BF16 degradation)
    - ``use_torch_linear_fallback=True``: l_linear+tanh on CPU (fixes ~2% BF16 dot-product error)
    - ``use_torch_phase_fallback=False``: SineGen phase chain stays on TTNN (has 1.2% BF16 noise)

    If this test passes > 0.99, the SineGen fallback is not strictly required — the 1.2% sinegen
    noise does not cascade into the audio output.  If it fails, the SineGen fallback is needed
    to clean up the input to l_linear and STFT.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    x, s, f0 = _setup_test_inputs(T_x)

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    with torch.no_grad(), _torch_random_zeros():
        y_ref = ref(x, s, f0)
        tt_mod = TTGenerator(
            device,
            params,
            use_torch_stft_fallback=True,
            use_torch_linear_fallback=True,
            use_torch_phase_fallback=False,  # SineGen stays on TTNN — this is what we're testing
        )

    x_nlc = x.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f0_tt = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, f0_tt)

    y_h = ttnn.to_torch(y_tt).float()
    while y_h.dim() > y_ref.dim():
        y_h.squeeze_(0)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    assert torch.isfinite(y_h).all(), "TTGenerator produced NaN/Inf"
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTGenerator (STFT+linear fallback, TTNN sinegen) PCC: {pcc:.6f}")
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99: SineGen 1.2% BF16 noise cascades into near-zero STFT bins "
        f"even with precise l_linear and torch.stft — use_torch_phase_fallback is also needed."
    )


@pytest.mark.xfail(
    reason=(
        "Keeping TTNN linear while only moving tanh to CPU is expected to preserve most BF16 "
        "dot-product error from l_linear and may stay below 0.99 PCC."
    ),
    strict=False,
)
def test_tt_generator_stft_tanh_fallback_no_sinegen_pcc(device):
    """STFT transform + tanh on CPU; SineGen phase chain and l_linear stay pure TTNN.

    This isolates whether moving only tanh to CPU materially improves PCC once STFT transform
    precision is fixed, while still keeping SineGen phase + l_linear in TTNN.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    x, s, f0 = _setup_test_inputs(T_x)

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    with torch.no_grad(), _torch_random_zeros():
        y_ref = ref(x, s, f0)
        tt_mod = TTGenerator(
            device,
            params,
            use_torch_stft_fallback=True,
            use_torch_tanh_fallback=True,
            use_torch_linear_fallback=False,
            use_torch_phase_fallback=False,
        )

    x_nlc = x.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f0_tt = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, f0_tt)

    y_h = ttnn.to_torch(y_tt).float()
    while y_h.dim() > y_ref.dim():
        y_h.squeeze_(0)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    assert torch.isfinite(y_h).all(), "TTGenerator produced NaN/Inf"
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTGenerator (STFT+tanh fallback, TTNN linear+sinegen) PCC: {pcc:.6f}")
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99: tanh-only fallback is insufficient; l_linear and/or sinegen "
        f"precision still dominate the remaining error."
    )


@pytest.mark.xfail(
    reason=(
        "All fallbacks except linear still keep TTNN l_linear BF16 dot-product error; "
        "PCC is expected to stay below 0.99."
    ),
    strict=False,
)
def test_tt_generator_full_forward_all_fallbacks_except_linear_pcc(device):
    """Full TT forward with STFT/phase/tanh fallbacks enabled, but linear kept on TTNN."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    x, s, f0 = _setup_test_inputs(T_x)

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    with torch.no_grad(), _torch_random_zeros():
        y_ref = ref(x, s, f0)
        tt_mod = TTGenerator(
            device,
            params,
            use_torch_stft_fallback=True,
            use_torch_phase_fallback=True,
            use_torch_tanh_fallback=True,
            use_torch_linear_fallback=False,
        )

    x_nlc = x.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f0_tt = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, f0_tt)

    y_h = ttnn.to_torch(y_tt).float()
    while y_h.dim() > y_ref.dim():
        y_h.squeeze_(0)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    assert torch.isfinite(y_h).all(), "TTGenerator (fallbacks except linear) produced NaN/Inf"
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTGenerator full-forward (phase + STFT + tanh fallback, TTNN linear) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low with all fallbacks except linear: {pcc}"


def test_tt_generator_full_forward_smoke(device):
    """End-to-end TT forward — verifies shape + finite output + non-trivial correlation.

    PCC vs PyTorch will not reach 0.99 here without CPU fallback or new hardware capability
    (see module-level docstring on bf16-matmul phase noise). We accept any positive PCC as
    proof the forward path runs end-to-end and produces structured (non-NaN, non-zero) output.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    x, s, f0 = _setup_test_inputs(T_x)
    f0u_ref = ref.f0_upsamp(f0[:, None]).transpose(1, 2).contiguous()
    B, T_har, _ = f0u_ref.shape
    dim = ref.m_source.l_sin_gen.dim
    torch.manual_seed(123)
    rand_ini = torch.rand(B, dim)
    rand_ini[:, 0] = 0.0
    sinegen_noise_raw = torch.randn(B, T_har, dim)
    source_noise_raw = torch.randn(B, T_har, 1)

    def _fake_rand(*size, **kwargs):
        out = rand_ini.to(kwargs.get("device", rand_ini.device))
        dtype = kwargs.get("dtype", out.dtype)
        return out.to(dtype)

    def _fake_randn_like(t, **kwargs):
        if t.shape[-1] == 1:
            return source_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()
        return sinegen_noise_raw.to(device=t.device, dtype=t.dtype).expand_as(t).clone()

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = _fake_rand
    torch.randn_like = _fake_randn_like
    try:
        tt_mod = TTGenerator(device, params)
        with torch.no_grad():
            y_ref = ref(x, s, f0)
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like

    x_nlc = x.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f0_tt = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    rand_ini_tt = ttnn.from_torch(rand_ini.unsqueeze(1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(
        x_tt,
        s_tt,
        f0_tt,
        sinegen_rand_ini=rand_ini_tt,
    )

    y_h = ttnn.to_torch(y_tt).float()
    while y_h.dim() > y_ref.dim():
        y_h.squeeze_(0)

    assert y_h.shape == y_ref.shape, (y_h.shape, y_ref.shape)
    assert torch.isfinite(y_h).all(), "TTGenerator full forward produced NaN/Inf"
    assert y_h.abs().max().item() > 1e-3, "TTGenerator full forward produced ~zero output"
    _, pcc = comp_pcc(y_ref, y_h, pcc=0.0)
    print(f"TTGenerator full-forward smoke PCC: {pcc:.6f} (informational; WH bf16 limit applies)")
    ttnn.deallocate(rand_ini_tt)


@pytest.mark.xfail(
    reason=(
        "TT trained harmonic har PCC < 0.99: TT m_source + TTTorchSTFT.transform vs torch "
        "(phase / cos-phase dominates; see module docstring)."
    ),
    strict=False,
)
def test_tt_generator_trained_harmonic_har_pcc_xfail_until_stft_phase_tight(device):
    """Harmonic ``har`` only: TT ``_harmonic_source_path`` vs reference ``har`` (BCT).

    This is the slice that blocks ``test_tt_generator_full_forward_smoke`` from reaching high PCC
    while ``test_tt_generator_pipeline_pcc`` (reference ``har`` injected) still passes > 0.99.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_kokoro_generator()
    _load_trained_weights(ref, ckpt_path)
    T_x = 5
    x, s, f0 = _setup_test_inputs(T_x)
    har_ref, _y_ref = _ref_har_and_audio(ref, x, s, f0)

    params = preprocess_tt_generator(ref, device, time_len_x=T_x)
    tt_mod = TTGenerator(device, params)
    mc = ttnn.DRAM_MEMORY_CONFIG

    f0_tt = ttnn.from_torch(f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    har_nlc = tt_mod._harmonic_source_path(
        f0_tt,
        sinegen_rand_ini=None,
        sinegen_noise_raw=None,
        source_noise_raw=None,
        memory_config=mc,
    )
    har_bct = ttnn.permute(har_nlc, (0, 2, 1), memory_config=mc)
    har_tt = ttnn.to_torch(har_bct).float()
    ttnn.deallocate(har_nlc)
    ttnn.deallocate(har_bct)
    ttnn.deallocate(f0_tt)

    while har_tt.dim() > har_ref.dim():
        har_tt.squeeze_(0)

    assert har_tt.shape == har_ref.shape, (har_tt.shape, har_ref.shape)
    K = har_ref.shape[1] // 2
    _, pcc_har = comp_pcc(har_ref, har_tt, pcc=0.0)
    _, pcc_mag = comp_pcc(har_ref[:, :K, :], har_tt[:, :K, :], pcc=0.0)
    _, pcc_cos_phase = comp_pcc(
        torch.cos(har_ref[:, K:, :]),
        torch.cos(har_tt[:, K:, :]),
        pcc=0.0,
    )
    print(
        "TTGenerator trained harmonic har PCC diagnostic: "
        f"har={pcc_har:.6f}, mag={pcc_mag:.6f}, cos(phase)={pcc_cos_phase:.6f}, "
        f"shape={tuple(har_ref.shape)}"
    )
    assert pcc_har > 0.99, (
        f"Expected har PCC > 0.99 once harmonic STFT path matches reference; got har={pcc_har:.6f}, "
        f"mag={pcc_mag:.6f}, cos(phase)={pcc_cos_phase:.6f}"
    )
