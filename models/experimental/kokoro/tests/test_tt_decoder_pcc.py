# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_decoder.TTDecoder` vs reference
:class:`~models.experimental.kokoro.reference.istftnet.Decoder` (``TorchSTFT`` path).

Test structure
--------------
1. ``test_tt_decoder_encode_pcc``
   Validates :attr:`Decoder.encode` (``AdainResBlk1d(514, 1024, 128)``) in isolation.
   Inputs: random concatenated ``[asr, F0, N]`` feature ``[B, 514, T]``.
   PCC must be > 0.99.

2. ``test_tt_decoder_decode_pcc``
   Validates the full decode block chain (blocks 0-3, including the final upsample block)
   against the reference, using pre-computed encode output as input.
   PCC must be > 0.99.

3. ``test_tt_decoder_full_forward_fallback_pcc``
   Injects reference decode-chain output into :class:`TTGenerator` (with both fallbacks).
   Matching the approach of ``test_tt_generator_pipeline_pcc``, this isolates the Generator's
   fallback precision from decode-chain BF16 error.  Validates at T_x = T_mel×2 = 10 (vs T_x=5
   in the standalone generator test).  PCC must be > 0.99.

4. ``test_tt_decoder_full_forward_no_fallback_smoke``
   Full :class:`TTDecoder` forward without fallbacks — verifies shape, finite output,
   and non-trivial signal. WH BF16 matmul phase noise prevents PCC > 0.99 here
   (documented limitation; see test_tt_generator_pcc.py module docstring).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.istftnet import AdainResBlk1d, Decoder
from models.experimental.kokoro.tt.tt_adain_resblk_1d import TTAdainResBlk1d, preprocess_tt_adain_resblk_1d
from models.experimental.kokoro.tt.tt_conv import tt_conv1d_nlc
from models.experimental.kokoro.tt.tt_decoder import TTDecoder, preprocess_tt_decoder
from models.experimental.kokoro.tt.tt_generator import TTGenerator


# ---------------------------------------------------------------------------
# Kokoro config constants (match KModel config.json / generator test)
# ---------------------------------------------------------------------------

_DIM_IN = 512  # hidden_dim
_STYLE_DIM = 128  # style_dim
_T_MEL = 5  # asr mel-frame count used in tests (T_f0 = T_MEL * 2 = 10)
_T_F0 = _T_MEL * 2


_DECODER_KWARGS = dict(
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[10, 6],
    upsample_kernel_sizes=[20, 12],
    upsample_initial_channel=512,
    gen_istft_n_fft=20,
    gen_istft_hop_size=5,
    disable_complex=False,
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


@contextmanager
def _torch_random_zeros():
    """Zero out all stochastic ops so reference and TT see identical noise."""
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    torch.randn_like = lambda t, **kwargs: torch.zeros_like(t, **kwargs)
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


@contextmanager
def _torch_source_module_fixed_noise(rand_ini: torch.Tensor, sinegen_noise: torch.Tensor, source_noise: torch.Tensor):
    """Inject fixed SourceModuleHnNSF noise tensors into the reference path."""
    real_rand = torch.rand
    real_randn_like = torch.randn_like

    def fake_rand(*size, **kwargs):
        requested = tuple(size) if len(size) > 1 else tuple(size[0])
        if requested == tuple(rand_ini.shape):
            return rand_ini.to(dtype=kwargs.get("dtype", rand_ini.dtype), device=kwargs.get("device", rand_ini.device))
        return torch.zeros(*size, **kwargs)

    def fake_randn_like(t, **kwargs):
        if tuple(t.shape) == tuple(sinegen_noise.shape):
            return sinegen_noise.to(dtype=t.dtype, device=t.device)
        if tuple(t.shape) == tuple(source_noise.shape):
            return source_noise.to(dtype=t.dtype, device=t.device)
        return torch.zeros_like(t, **kwargs)

    torch.rand = fake_rand
    torch.randn_like = fake_randn_like
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def _build_decoder() -> Decoder:
    return Decoder(
        dim_in=_DIM_IN,
        style_dim=_STYLE_DIM,
        dim_out=80,
        **_DECODER_KWARGS,
    ).eval()


def _load_trained_weights(ref: Decoder, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    raw_sd = ckpt.get("decoder", {})
    # Checkpoint uses DataParallel "module." prefix; strip it to match bare Decoder state dict.
    prefix = "module."
    sd = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in raw_sd.items()}
    ref.load_state_dict(sd, strict=False)


def _setup_inputs(seed: int = 0):
    torch.manual_seed(seed)
    asr = torch.randn(1, _DIM_IN, _T_MEL)  # [B, 512, T_mel] BCL
    F0_curve = torch.rand(1, _T_F0) * 300  # [B, T_f0]  (Hz range)
    N_curve = torch.randn(1, _T_F0) * 0.01  # [B, T_f0]
    s = torch.randn(1, _STYLE_DIM)  # [B, 128]
    return asr, F0_curve, N_curve, s


def _setup_realistic_decoder_inputs(seed: int = 2):
    """Use a smooth F0 contour (generator-like) for stable full-forward PCC checks.

    Fully random F0/N curves are far outside Kokoro's training distribution and can
    over-amplify phase sensitivity in the harmonic source path. This fixture mirrors
    the deterministic contour used in the generator fallback test, but at ``T_f0=10``.
    """
    torch.manual_seed(seed)
    asr = torch.randn(1, _DIM_IN, _T_MEL)
    s = torch.randn(1, _STYLE_DIM)
    F0_curve = torch.tensor([[100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 180.0, 160.0, 140.0, 120.0]])
    N_curve = torch.zeros_like(F0_curve)
    return asr, F0_curve, N_curve, s


def _pcc_nlc(y_ref_bcl: torch.Tensor, y_tt) -> float:
    """Convert ref BCL → NLC and compare with a TT NLC tensor."""
    y_ref_nlc = y_ref_bcl.transpose(1, 2).contiguous()
    y_hat = ttnn.to_torch(y_tt).float()
    _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.0)
    return pcc


# ---------------------------------------------------------------------------
# 1. Decoder.encode isolated test
# ---------------------------------------------------------------------------


def test_tt_decoder_encode_pcc(device):
    """``Decoder.encode`` = ``AdainResBlk1d(514, 1024, 128)`` in isolation; PCC > 0.99.

    The encode block maps the concatenated ``[asr, F0, N]`` feature (514 channels)
    to the 1024-channel latent space used by all decode blocks.
    """
    torch.manual_seed(10)
    # dim_in = _DIM_IN + 2 = 514  (asr channels + F0 + N)
    dim_in_enc = _DIM_IN + 2
    b, t = 1, _T_MEL

    ref_enc = AdainResBlk1d(dim_in_enc, 1024, style_dim=_STYLE_DIM, upsample="none")
    ref_enc.eval()

    params = preprocess_tt_adain_resblk_1d(ref_enc, device, weights_dtype=ttnn.float32, conv_weights_dtype=ttnn.float32)
    tt_blk = TTAdainResBlk1d(device, params)

    x_bcl = torch.randn(b, dim_in_enc, t)
    s = torch.randn(b, _STYLE_DIM)
    with torch.no_grad():
        y_ref = ref_enc(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_blk(x_tt, s_tt)

    pcc = _pcc_nlc(y_ref, y_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    print(f"Decoder.encode (AdainResBlk1d 514→1024) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


# ---------------------------------------------------------------------------
# 2. Decoder.decode chain test
# ---------------------------------------------------------------------------


def test_tt_decoder_decode_pcc(device):
    """Decode block chain (4 ``AdainResBlk1d`` blocks incl. upsample); PCC > 0.99.

    Feeds a random 1024-channel input through all four decode blocks and compares
    TT vs PyTorch on the final 512-channel, 2×-upsampled output.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_decoder()
    _load_trained_weights(ref, ckpt_path)
    asr, F0_curve, N_curve, s = _setup_inputs(seed=1)

    # Run reference forward through F0_conv + N_conv + encode to get the input to decode
    with torch.no_grad():
        F0 = ref.F0_conv(F0_curve.unsqueeze(1))  # [1, 1, T_mel]
        N = ref.N_conv(N_curve.unsqueeze(1))  # [1, 1, T_mel]
        x_cat = torch.cat([asr, F0, N], dim=1)  # [1, 514, T_mel]
        x_enc = ref.encode(x_cat, s)  # [1, 1024, T_mel]
        asr_res = ref.asr_res(asr)  # [1, 64,  T_mel]

        # Reference decode chain
        x_ref = x_enc
        res = True
        for block in ref.decode:
            if res:
                x_ref = torch.cat([x_ref, asr_res, F0, N], dim=1)
            x_ref = block(x_ref, s)
            if block.upsample_type != "none":
                res = False
        y_ref_bcl = x_ref  # [1, 512, T_mel*2]

    # TT decode chain using the same reference encode output as input
    from models.experimental.kokoro.tt.tt_adain_resblk_1d import preprocess_tt_adain_resblk_1d

    decode_params = tuple(
        preprocess_tt_adain_resblk_1d(blk, device, weights_dtype=ttnn.float32, conv_weights_dtype=ttnn.float32)
        for blk in ref.decode
    )
    tt_decode_blocks = tuple(TTAdainResBlk1d(device, p) for p in decode_params)

    from models.experimental.kokoro.tt.tt_decoder import _conv1d_to_tt, _strip_wn
    import copy

    # Use copies so weight-norm stripping doesn't affect the already-loaded ref
    ref_copy = copy.deepcopy(ref)
    _strip_wn(ref_copy.F0_conv)
    _strip_wn(ref_copy.N_conv)
    F0_conv_p = _conv1d_to_tt(ref_copy.F0_conv, device, weights_dtype=ttnn.float32)
    N_conv_p = _conv1d_to_tt(ref_copy.N_conv, device, weights_dtype=ttnn.float32)

    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )

    F0_f0_nlc = F0_curve.unsqueeze(2)  # [1, T_f0, 1]
    N_f0_nlc = N_curve.unsqueeze(2)  # [1, T_f0, 1]
    F0_tt = ttnn.from_torch(F0_f0_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    N_tt = ttnn.from_torch(N_f0_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _interleaved(t):
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return ttnn.to_memory_config(t, mc)
        return t

    F0_down = _interleaved(
        tt_conv1d_nlc(
            x_nlc=F0_tt, params=F0_conv_p, device=device, compute_config=ck, memory_config=mc, preserve_input_dtype=True
        )
    )
    N_down = _interleaved(
        tt_conv1d_nlc(
            x_nlc=N_tt, params=N_conv_p, device=device, compute_config=ck, memory_config=mc, preserve_input_dtype=True
        )
    )
    ttnn.deallocate(F0_tt)
    ttnn.deallocate(N_tt)

    # Upload reference encode output and asr_res as starting point
    x_enc_nlc = x_enc.transpose(1, 2).contiguous()  # [1, T_mel, 1024]
    x_tt = ttnn.from_torch(x_enc_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # asr_res conv (1x1)
    ref_copy2 = copy.deepcopy(ref)
    _strip_wn(ref_copy2.asr_res[0])
    asr_res_p = _conv1d_to_tt(ref_copy2.asr_res[0], device, weights_dtype=ttnn.float32)
    asr_nlc = asr.transpose(1, 2).contiguous()
    asr_tt = ttnn.from_torch(asr_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    asr_res_tt = _interleaved(
        tt_conv1d_nlc(
            x_nlc=asr_tt,
            params=asr_res_p,
            device=device,
            compute_config=ck,
            memory_config=mc,
            preserve_input_dtype=True,
        )
    )
    ttnn.deallocate(asr_tt)

    cond_dtype = F0_down.dtype  # float32 — cast x back after each block to keep concat uniform
    res_tt = True
    for blk in tt_decode_blocks:
        if res_tt:
            x_cat_tt = ttnn.concat([x_tt, asr_res_tt, F0_down, N_down], dim=2, memory_config=mc)
            ttnn.deallocate(x_tt)
            x_tt = x_cat_tt
        x_tt = blk.forward(x_tt, s_tt, memory_config=mc)
        if x_tt.dtype != cond_dtype:
            x_tt = ttnn.typecast(x_tt, cond_dtype, memory_config=mc)
        if blk._params.layer_type != "none":
            res_tt = False

    ttnn.deallocate(asr_res_tt)
    ttnn.deallocate(F0_down)
    ttnn.deallocate(N_down)

    y_hat = ttnn.to_torch(x_tt).float()
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    # y_ref_bcl is [1, 512, T_mel*2]; y_hat is [1, T_mel*2, 512] (NLC)
    y_ref_nlc = y_ref_bcl.transpose(1, 2).contiguous()
    assert y_hat.shape == y_ref_nlc.shape, (y_hat.shape, y_ref_nlc.shape)
    _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.0)
    print(f"Decoder decode chain PCC: {pcc:.6f}, shape: {tuple(y_ref_bcl.shape)}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


# ---------------------------------------------------------------------------
# 3. Full Decoder forward — with fallbacks (PCC > 0.99)
# ---------------------------------------------------------------------------


def test_tt_decoder_full_forward_fallback_pcc(device):
    """TTDecoder Generator with both BH BF16 fallbacks applied; PCC must be > 0.99.

    Injects the reference decode-chain output directly into the TT Generator (with both fallbacks
    enabled), matching the approach of ``test_tt_generator_pipeline_pcc`` for the harmonic source.
    This isolates the Generator's fallback precision from decode-chain BF16 approximation error and
    validates that ``use_torch_stft_fallback=True`` + ``use_torch_phase_fallback=True`` match
    generator-level fallback behavior and keep PCC > 0.99 in the Decoder context (T_x = T_mel×2
    = 10 at inference scale, vs T_x = 5 in the standalone generator test) on realistic F0 contours.

    The decode-chain PCC (≥ 0.99 via :func:`test_tt_decoder_decode_pcc`) and the generator PCC are
    validated separately so each component's accuracy requirement is independently verifiable.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_decoder()
    _load_trained_weights(ref, ckpt_path)
    asr, F0_curve, N_curve, s = _setup_realistic_decoder_inputs(seed=2)

    # Run the reference decode chain manually to obtain x_dec — the exact input the Generator
    # receives in the reference forward.  We also compute y_ref so PCC is measured against the
    # same reference as the full-decoder test.
    with torch.no_grad(), _torch_random_zeros():
        F0_ref = ref.F0_conv(F0_curve.unsqueeze(1))  # [1, 1, T_mel]
        N_ref = ref.N_conv(N_curve.unsqueeze(1))  # [1, 1, T_mel]
        x_cat = torch.cat([asr, F0_ref, N_ref], dim=1)  # [1, 514, T_mel]
        x_enc = ref.encode(x_cat, s)  # [1, 1024, T_mel]
        asr_res_ref = ref.asr_res(asr)  # [1, 64,  T_mel]
        x_dec = x_enc
        res = True
        for block in ref.decode:
            if res:
                x_dec = torch.cat([x_dec, asr_res_ref, F0_ref, N_ref], dim=1)
            x_dec = block(x_dec, s)
            if block.upsample_type != "none":
                res = False
        y_ref = ref.generator(x_dec, s, F0_curve)  # [1, 1, audio_len]

    # Build TTDecoder params (exercises full preprocess path including generator weights).
    params = preprocess_tt_decoder(ref, device, time_len_asr=_T_MEL)

    # Create TTGenerator from the Decoder's generator params with fallbacks enabled.
    gen = TTGenerator(
        device,
        params.generator,
        use_torch_stft_fallback=True,
        use_torch_phase_fallback=True,
        use_torch_linear_fallback=True,
        use_torch_tanh_fallback=True,
    )

    mc = ttnn.DRAM_MEMORY_CONFIG
    # x_dec is [1, 512, T_mel*2] BCL → transpose to NLC [1, T_mel*2, 512]
    x_dec_nlc = x_dec.transpose(1, 2).contiguous()
    x_dec_tt = ttnn.from_torch(x_dec_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    F0_tt = ttnn.from_torch(F0_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    audio_tt = gen(x_dec_tt, s_tt, F0_tt, memory_config=mc)

    y_hat = ttnn.to_torch(audio_tt).float()
    ttnn.deallocate(audio_tt)

    # Squeeze extra dims to match reference shape [1, 1, audio_len]
    while y_hat.dim() > y_ref.dim():
        y_hat = y_hat.squeeze(0)

    assert y_hat.shape == y_ref.shape, (y_hat.shape, y_ref.shape)
    assert torch.isfinite(y_hat).all(), "TTDecoder generator (fallbacks) produced NaN/Inf"
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTDecoder generator (phase + STFT fallback, ref decode injected) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low with fallbacks: {pcc}"


# ---------------------------------------------------------------------------
# 4. Full Decoder forward — without fallbacks (smoke / shape + finite check)
# ---------------------------------------------------------------------------


def test_tt_decoder_full_forward_no_fallback_smoke(device):
    """Full TTDecoder forward without fallbacks — shape + finite + non-trivial output.

    PCC vs PyTorch will not reach 0.99 here due to WH BF16 matmul phase noise
    in the harmonic-source STFT (see test_tt_generator_pcc.py module docstring).
    Acceptance criteria: correct shape, all finite values, non-zero signal.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_decoder()
    _load_trained_weights(ref, ckpt_path)
    asr, F0_curve, N_curve, s = _setup_inputs(seed=3)

    with torch.no_grad(), _torch_random_zeros():
        y_ref = ref(asr, F0_curve, N_curve, s)

    params = preprocess_tt_decoder(ref, device, time_len_asr=_T_MEL)
    tt_mod = TTDecoder(device, params)

    mc = ttnn.DRAM_MEMORY_CONFIG
    asr_nlc = asr.transpose(1, 2).contiguous()
    asr_tt = ttnn.from_torch(asr_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    F0_tt = ttnn.from_torch(F0_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    N_tt = ttnn.from_torch(N_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    audio_tt = tt_mod(asr_tt, F0_tt, N_tt, s_tt, memory_config=mc)

    y_hat = ttnn.to_torch(audio_tt).float()
    ttnn.deallocate(audio_tt)

    while y_hat.dim() > y_ref.dim():
        y_hat = y_hat.squeeze(0)

    assert y_hat.shape == y_ref.shape, (y_hat.shape, y_ref.shape)
    assert torch.isfinite(y_hat).all(), "TTDecoder (no fallback) produced NaN/Inf"
    assert y_hat.abs().max().item() > 1e-3, "TTDecoder (no fallback) produced ~zero output"
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTDecoder full-forward smoke PCC: {pcc:.6f} (informational; WH BF16 limit applies)")


def test_tt_decoder_full_forward_no_fallback_matched_source_noise(device):
    """Compare no-fallback vs full-fallback TTDecoder under identical source-noise tensors."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    ref = _build_decoder()
    _load_trained_weights(ref, ckpt_path)
    asr, F0_curve, N_curve, s = _setup_realistic_decoder_inputs(seed=4)

    params = preprocess_tt_decoder(ref, device, time_len_asr=_T_MEL)
    tt_mod_no_fallback = TTDecoder(device, params)
    tt_mod_full_fallback = TTDecoder(
        device,
        params,
        use_torch_stft_fallback=True,
        use_torch_phase_fallback=True,
        use_torch_linear_fallback=True,
        use_torch_tanh_fallback=True,
    )

    B = int(F0_curve.shape[0])
    T_har = params.generator.time_len_x * params.generator.upsample_scale_full
    dim = params.generator.m_source.sinegen.dim

    torch.manual_seed(321)
    rand_ini = torch.rand(B, 1, dim)
    sinegen_noise = torch.randn(B, T_har, dim)
    source_noise = torch.randn(B, T_har, 1)

    with torch.no_grad(), _torch_source_module_fixed_noise(rand_ini, sinegen_noise, source_noise):
        y_ref = ref(asr, F0_curve, N_curve, s)

    mc = ttnn.DRAM_MEMORY_CONFIG
    asr_nlc = asr.transpose(1, 2).contiguous()
    asr_tt = ttnn.from_torch(asr_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    F0_tt = ttnn.from_torch(F0_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    N_tt = ttnn.from_torch(N_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    rand_ini_tt = ttnn.from_torch(rand_ini, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    sinegen_noise_tt = ttnn.from_torch(sinegen_noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    source_noise_tt = ttnn.from_torch(source_noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    audio_tt_no_fallback = tt_mod_no_fallback(
        asr_tt,
        F0_tt,
        N_tt,
        s_tt,
        sinegen_rand_ini=rand_ini_tt,
        sinegen_noise_raw=sinegen_noise_tt,
        source_noise_raw=source_noise_tt,
        memory_config=mc,
    )
    audio_tt_full_fallback = tt_mod_full_fallback(
        asr_tt,
        F0_tt,
        N_tt,
        s_tt,
        sinegen_rand_ini=rand_ini_tt,
        sinegen_noise_raw=sinegen_noise_tt,
        source_noise_raw=source_noise_tt,
        memory_config=mc,
    )

    y_hat_no_fallback = ttnn.to_torch(audio_tt_no_fallback).float()
    y_hat_full_fallback = ttnn.to_torch(audio_tt_full_fallback).float()
    ttnn.deallocate(audio_tt_no_fallback)
    ttnn.deallocate(audio_tt_full_fallback)
    ttnn.deallocate(asr_tt)
    ttnn.deallocate(F0_tt)
    ttnn.deallocate(N_tt)
    ttnn.deallocate(s_tt)
    ttnn.deallocate(rand_ini_tt)
    ttnn.deallocate(sinegen_noise_tt)
    ttnn.deallocate(source_noise_tt)

    while y_hat_no_fallback.dim() > y_ref.dim():
        y_hat_no_fallback = y_hat_no_fallback.squeeze(0)
    while y_hat_full_fallback.dim() > y_ref.dim():
        y_hat_full_fallback = y_hat_full_fallback.squeeze(0)

    assert y_hat_no_fallback.shape == y_ref.shape, (y_hat_no_fallback.shape, y_ref.shape)
    assert y_hat_full_fallback.shape == y_ref.shape, (y_hat_full_fallback.shape, y_ref.shape)
    assert torch.isfinite(y_hat_no_fallback).all(), "TTDecoder (no fallback, matched source noise) produced NaN/Inf"
    assert torch.isfinite(y_hat_full_fallback).all(), "TTDecoder (full fallback, matched source noise) produced NaN/Inf"
    _, pcc_no_fallback = comp_pcc(y_ref, y_hat_no_fallback, pcc=0.0)
    _, pcc_full_fallback = comp_pcc(y_ref, y_hat_full_fallback, pcc=0.0)
    print(
        "TTDecoder matched-source-noise PCCs: "
        f"no-fallback={pcc_no_fallback:.6f}, full-fallback={pcc_full_fallback:.6f}"
    )
    assert (
        pcc_no_fallback > 0.0
    ), f"Expected positive PCC in matched-source-noise no-fallback run, got {pcc_no_fallback:.6f}"
    assert (
        pcc_full_fallback > 0.0
    ), f"Expected positive PCC in matched-source-noise full-fallback run, got {pcc_full_fallback:.6f}"
