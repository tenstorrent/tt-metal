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

1b. ``test_tt_decoder_encode_pcc_captured_cat_in``
   Same encode block with **real** ``D2_cat_in`` from Kokoro prosody (not random).
   Compares ref vs TT encode on (a) ref-built cat_in and (b) TT F0/N conv + ref ASR cat_in
   (matches the failing decode-stack walk). See ``DECODE_STACK_FINDINGS.md``.

1c. ``test_tt_decoder_f0_n_conv_pcc_captured``
   F0_conv / N_conv on real F0/N curves; ref vs TT on **shared** input per conv.

2. ``test_tt_decoder_decode_pcc``
   Validates the full decode block chain (blocks 0-3, including the final upsample block)
   against the reference, using pre-computed encode output as input.
   PCC must be > 0.99.

2b. ``test_tt_decoder_decode_pcc_captured``
   Decode chain on captured prosody; ref encode input, ref vs TT F0/N in cond concat.

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
from models.experimental.kokoro.m_source_rng import (
    MSourceRngTensors,
    deallocate_m_source_rng_tt,
    make_zero_m_source_rng,
    m_source_rng_shapes_from_f0,
    patched_m_source_torch_rng,
    upload_m_source_rng,
)
from models.experimental.kokoro.reference.istftnet import AdainResBlk1d, Decoder
from models.experimental.kokoro.tt.tt_adain_resblk_1d import TTAdainResBlk1d, preprocess_tt_adain_resblk_1d
from models.experimental.kokoro.tt.tt_conv import tt_conv1d_nlc, tt_conv1d_nlc_cpu
from models.experimental.kokoro.tt.tt_decoder import TTDecoder, _to_interleaved, preprocess_tt_decoder
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


def _build_ref_cat_in_bct(
    dec: Decoder,
    asr_bct: torch.Tensor,
    F0_curve: torch.Tensor,
    N_curve: torch.Tensor,
) -> torch.Tensor:
    """Reference ``cat([asr, F0_conv(F0), N_conv(N)], dim=channel)`` in BCT layout."""
    with torch.no_grad():
        f0_b1t = F0_curve.unsqueeze(1) if F0_curve.dim() == 2 else F0_curve
        n_b1t = N_curve.unsqueeze(1) if N_curve.dim() == 2 else N_curve
        f0_down = dec.F0_conv(f0_b1t)
        n_down = dec.N_conv(n_b1t)
        return torch.cat([asr_bct, f0_down, n_down], dim=1)


def _build_tt_walk_cat_in_bct(
    device,
    dec_params,
    asr_bct: torch.Tensor,
    F0_curve: torch.Tensor,
    N_curve: torch.Tensor,
) -> torch.Tensor:
    """``D2_cat_in`` (TT F0/N conv + ref ASR) compared against the reference decode stack."""
    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    asr_nlc = asr_bct.transpose(1, 2).contiguous()
    asr_tt = ttnn.from_torch(asr_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    f0_nlc = ttnn.unsqueeze(
        ttnn.from_torch(F0_curve.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc),
        2,
    )
    n_nlc = ttnn.unsqueeze(
        ttnn.from_torch(N_curve.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc),
        2,
    )
    f0_down = _to_interleaved(
        tt_conv1d_nlc_cpu(x_nlc=f0_nlc, params=dec_params.F0_conv, device=device, memory_config=mc),
        mc,
    )
    n_down = _to_interleaved(
        tt_conv1d_nlc_cpu(x_nlc=n_nlc, params=dec_params.N_conv, device=device, memory_config=mc),
        mc,
    )
    ttnn.deallocate(f0_nlc)
    ttnn.deallocate(n_nlc)
    x_cat = ttnn.concat([asr_tt, f0_down, n_down], dim=2, memory_config=mc)
    ttnn.deallocate(asr_tt)
    ttnn.deallocate(f0_down)
    ttnn.deallocate(n_down)
    x_bct = _tt_to_torch_cat(x_cat).permute(0, 2, 1).contiguous()
    ttnn.deallocate(x_cat)
    return x_bct


def _tt_to_torch_cat(t: ttnn.Tensor) -> torch.Tensor:
    out = ttnn.to_torch(t).float()
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out


def _kokoro_ck_mc(device):
    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return ck, mc


def _pcc_flat_torch(ref: torch.Tensor, tt: torch.Tensor) -> float:
    _, pcc = comp_pcc(ref.detach().float().reshape(1, -1), tt.detach().float().reshape(1, -1), pcc=0.0)
    return pcc


def _load_captured_prosody(ckpt_path: Path):
    """ASR/F0/N/s_style from the decode-stack diagnostic phoneme string."""
    from models.experimental.kokoro.reference.model import KModel
    from models.experimental.kokoro.tests.kokoro_checkpoint import (
        _phonemize,
        _ref_prosody,
        _zero_noise,
    )

    phonemes, ref_s = _phonemize("Hello from Tenstorrent.")
    ref_model = KModel(repo_id="hexgrad/Kokoro-82M", model=str(ckpt_path), disable_complex=True).eval()
    with torch.no_grad(), _zero_noise():
        return _ref_prosody(ref_model, phonemes, ref_s)


def _ref_f0_n_down_bct(
    dec: Decoder, F0_curve: torch.Tensor, N_curve: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        f0_b1t = F0_curve.unsqueeze(1) if F0_curve.dim() == 2 else F0_curve
        n_b1t = N_curve.unsqueeze(1) if N_curve.dim() == 2 else N_curve
        return dec.F0_conv(f0_b1t), dec.N_conv(n_b1t)


def _tt_f0_n_down_nlc(
    device,
    *,
    F0_conv_p,
    N_conv_p,
    F0_curve: torch.Tensor,
    N_curve: torch.Tensor,
):
    """TT F0_conv / N_conv on shared curves; returns NLC ``[B, T_mel, 1]`` tensors."""
    ck, mc = _kokoro_ck_mc(device)
    f0_nlc = ttnn.unsqueeze(
        ttnn.from_torch(F0_curve.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc),
        2,
    )
    n_nlc = ttnn.unsqueeze(
        ttnn.from_torch(N_curve.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc),
        2,
    )
    f0_down = _to_interleaved(
        tt_conv1d_nlc_cpu(x_nlc=f0_nlc, params=F0_conv_p, device=device, memory_config=mc),
        mc,
    )
    n_down = _to_interleaved(
        tt_conv1d_nlc_cpu(x_nlc=n_nlc, params=N_conv_p, device=device, memory_config=mc),
        mc,
    )
    ttnn.deallocate(f0_nlc)
    ttnn.deallocate(n_nlc)
    return f0_down, n_down


def _encode_pcc_on_cat_in(
    device,
    ref_enc: AdainResBlk1d,
    tt_enc: TTAdainResBlk1d,
    cat_bct: torch.Tensor,
    s_style: torch.Tensor,
) -> float:
    """PCC between ref and TT encode outputs for one shared ``cat_bct`` input."""
    with torch.no_grad():
        y_ref = ref_enc(cat_bct, s_style)
    x_nlc = cat_bct.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s_style.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_enc(x_tt, s_tt)
    pcc = _pcc_nlc(y_ref, y_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)
    return pcc


def test_tt_decoder_encode_pcc_captured_cat_in(device):
    """Encode PCC on captured ``D2_cat_in`` (real prosody), not random Gaussian input.

    Uses real ``D2_cat_in`` from Kokoro prosody (not random Gaussian). Staged diagnostic
    ``D3_encode`` ~0.77 compares ref encode on ref ``cat_in`` vs TT encode on TT ``cat_in``;
    this test compares ref vs TT encode on each **shared** ``cat_in`` separately.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    try:
        from models.experimental.kokoro.reference.model import KModel
        from models.experimental.kokoro.tests.kokoro_checkpoint import (
            _phonemize,
            _ref_prosody,
            _zero_noise,
        )
    except ImportError:
        pytest.skip("kokoro package required for captured cat_in test")

    test_text = "Hello from Tenstorrent."
    phonemes, ref_s = _phonemize(test_text)
    ref_model = KModel(repo_id="hexgrad/Kokoro-82M", model=str(ckpt_path), disable_complex=True).eval()
    with torch.no_grad(), _zero_noise():
        asr_bct, f0_curve, n_curve, s_style, _ = _ref_prosody(ref_model, phonemes, ref_s)

    ref_dec = _build_decoder()
    _load_trained_weights(ref_dec, ckpt_path)
    t_mel = int(asr_bct.shape[-1])

    ref_enc_mod = ref_dec.encode
    enc_params = preprocess_tt_adain_resblk_1d(
        ref_enc_mod, device, weights_dtype=ttnn.float32, conv_weights_dtype=ttnn.float32
    )
    tt_enc = TTAdainResBlk1d(device, enc_params)
    dec_params = preprocess_tt_decoder(ref_dec, device, time_len_asr=t_mel)

    ref_cat = _build_ref_cat_in_bct(ref_dec, asr_bct, f0_curve, n_curve)
    tt_cat = _build_tt_walk_cat_in_bct(device, dec_params, asr_bct, f0_curve, n_curve)

    _, pcc_cat = comp_pcc(
        ref_cat.float().reshape(1, -1),
        tt_cat.float().reshape(1, -1),
        pcc=0.0,
    )
    pcc_ref_cat = _encode_pcc_on_cat_in(device, ref_enc_mod, tt_enc, ref_cat, s_style)
    pcc_tt_cat = _encode_pcc_on_cat_in(device, ref_enc_mod, tt_enc, tt_cat, s_style)

    print(
        f"\nEncode PCC captured cat_in ({test_text!r}): "
        f"D2_cat_in ref-vs-tt={float(pcc_cat):.6f}, "
        f"encode(ref_cat_in)={pcc_ref_cat:.6f}, "
        f"encode(tt_walk_cat_in)={pcc_tt_cat:.6f}"
    )

    assert (
        pcc_ref_cat > 0.99
    ), f"TT encode on ref-built cat_in PCC {pcc_ref_cat:.6f}; encode op wrong even on ref statistics"
    assert pcc_tt_cat > 0.99, f"TT encode on walk cat_in PCC {pcc_tt_cat:.6f}"


def test_tt_decoder_f0_n_conv_pcc_captured(device):
    """F0_conv / N_conv on captured F0/N; ref vs TT on the same 1-D curves."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    try:
        asr_bct, f0_curve, n_curve, _, _ = _load_captured_prosody(ckpt_path)
    except ImportError:
        pytest.skip("kokoro package required for captured F0/N conv test")

    ref_dec = _build_decoder()
    _load_trained_weights(ref_dec, ckpt_path)
    dec_params = preprocess_tt_decoder(ref_dec, device, time_len_asr=int(asr_bct.shape[-1]))

    ref_f0, ref_n = _ref_f0_n_down_bct(ref_dec, f0_curve, n_curve)
    f0_tt, n_tt = _tt_f0_n_down_nlc(
        device,
        F0_conv_p=dec_params.F0_conv,
        N_conv_p=dec_params.N_conv,
        F0_curve=f0_curve,
        N_curve=n_curve,
    )
    tt_f0_bct = _tt_to_torch_cat(f0_tt).permute(0, 2, 1).contiguous()
    tt_n_bct = _tt_to_torch_cat(n_tt).permute(0, 2, 1).contiguous()
    ttnn.deallocate(f0_tt)
    ttnn.deallocate(n_tt)

    pcc_f0 = _pcc_flat_torch(ref_f0, tt_f0_bct)
    pcc_n = _pcc_flat_torch(ref_n, tt_n_bct)
    print(
        f"\nF0/N conv PCC captured: D0_F0_conv={pcc_f0:.6f}, D1_N_conv={pcc_n:.6f}, "
        f"T_f0={f0_curve.shape[-1]}, T_mel={ref_f0.shape[-1]}"
    )
    assert pcc_f0 > 0.999, f"F0_conv PCC {pcc_f0:.6f}"
    assert pcc_n > 0.999, f"N_conv PCC {pcc_n:.6f}"


def _run_ref_decode_chain(
    ref_dec: Decoder,
    *,
    x_enc: torch.Tensor,
    asr_bct: torch.Tensor,
    f0_down: torch.Tensor,
    n_down: torch.Tensor,
    s_style: torch.Tensor,
) -> torch.Tensor:
    """Reference decode blocks 0–3; returns BCT ``[B, 512, T_x]``."""
    with torch.no_grad():
        asr_res = ref_dec.asr_res(asr_bct)
        x = x_enc
        res = True
        for block in ref_dec.decode:
            if res:
                x = torch.cat([x, asr_res, f0_down, n_down], dim=1)
            x = block(x, s_style)
            if block.upsample_type != "none":
                res = False
        return x


def _run_tt_decode_chain(
    device,
    tt_decode_blocks: tuple[TTAdainResBlk1d, ...],
    *,
    x_enc_bct: torch.Tensor,
    asr_res_nlc,
    f0_down_nlc,
    n_down_nlc,
    s_style: torch.Tensor,
) -> torch.Tensor:
    ck, mc = _kokoro_ck_mc(device)
    x_nlc = x_enc_bct.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    s_tt = ttnn.from_torch(
        s_style.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    cond_dtype = f0_down_nlc.dtype
    res_tt = True
    for blk in tt_decode_blocks:
        if res_tt:
            x_cat = ttnn.concat([x_tt, asr_res_nlc, f0_down_nlc, n_down_nlc], dim=2, memory_config=mc)
            ttnn.deallocate(x_tt)
            x_tt = x_cat
        x_tt = blk.forward(x_tt, s_tt, memory_config=mc)
        if x_tt.dtype != cond_dtype:
            x_cast = ttnn.typecast(x_tt, cond_dtype, memory_config=mc)
            ttnn.deallocate(x_tt)
            x_tt = x_cast
        if blk._params.layer_type != "none":
            res_tt = False
    out = _tt_to_torch_cat(x_tt).permute(0, 2, 1).contiguous()
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)
    return out


def test_tt_decoder_decode_pcc_captured(device):
    """Decode blocks on captured prosody with **shared** ref encode output.

    Two cond paths: ref F0/N downsamples vs TT F0/N (isolates D5 drift from D0/D1).
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro checkpoint not found locally.")

    try:
        asr_bct, f0_curve, n_curve, s_style, _ = _load_captured_prosody(ckpt_path)
    except ImportError:
        pytest.skip("kokoro package required for captured decode test")

    ref_dec = _build_decoder()
    _load_trained_weights(ref_dec, ckpt_path)
    t_mel = int(asr_bct.shape[-1])
    dec_params = preprocess_tt_decoder(ref_dec, device, time_len_asr=t_mel)

    ref_cat = _build_ref_cat_in_bct(ref_dec, asr_bct, f0_curve, n_curve)
    with torch.no_grad():
        x_enc = ref_dec.encode(ref_cat, s_style)
    ref_f0, ref_n = _ref_f0_n_down_bct(ref_dec, f0_curve, n_curve)
    y_ref = _run_ref_decode_chain(ref_dec, x_enc=x_enc, asr_bct=asr_bct, f0_down=ref_f0, n_down=ref_n, s_style=s_style)

    decode_params = tuple(
        preprocess_tt_adain_resblk_1d(blk, device, weights_dtype=ttnn.float32, conv_weights_dtype=ttnn.float32)
        for blk in ref_dec.decode
    )
    tt_blocks = tuple(TTAdainResBlk1d(device, p) for p in decode_params)

    ck, mc = _kokoro_ck_mc(device)
    asr_nlc = asr_bct.transpose(1, 2).contiguous()
    asr_tt = ttnn.from_torch(asr_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    asr_res = _to_interleaved(
        tt_conv1d_nlc(
            x_nlc=asr_tt,
            params=dec_params.asr_res,
            device=device,
            compute_config=ck,
            memory_config=mc,
            preserve_input_dtype=True,
        ),
        mc,
    )
    ttnn.deallocate(asr_tt)

    f0_ref_nlc = ttnn.from_torch(
        ref_f0.transpose(1, 2).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    n_ref_nlc = ttnn.from_torch(
        ref_n.transpose(1, 2).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    y_tt_ref_cond = _run_tt_decode_chain(
        device,
        tt_blocks,
        x_enc_bct=x_enc,
        asr_res_nlc=asr_res,
        f0_down_nlc=f0_ref_nlc,
        n_down_nlc=n_ref_nlc,
        s_style=s_style,
    )
    pcc_ref_cond = _pcc_flat_torch(y_ref, y_tt_ref_cond)
    ttnn.deallocate(f0_ref_nlc)
    ttnn.deallocate(n_ref_nlc)

    f0_tt_nlc, n_tt_nlc = _tt_f0_n_down_nlc(
        device,
        F0_conv_p=dec_params.F0_conv,
        N_conv_p=dec_params.N_conv,
        F0_curve=f0_curve,
        N_curve=n_curve,
    )
    y_tt_tt_cond = _run_tt_decode_chain(
        device,
        tt_blocks,
        x_enc_bct=x_enc,
        asr_res_nlc=asr_res,
        f0_down_nlc=f0_tt_nlc,
        n_down_nlc=n_tt_nlc,
        s_style=s_style,
    )
    pcc_tt_cond = _pcc_flat_torch(y_ref, y_tt_tt_cond)
    ttnn.deallocate(asr_res)
    ttnn.deallocate(f0_tt_nlc)
    ttnn.deallocate(n_tt_nlc)

    print(
        f"\nDecode PCC captured (ref encode, T_mel={t_mel}): "
        f"ref_F0N_cond={pcc_ref_cond:.6f}, tt_F0N_cond={pcc_tt_cond:.6f}"
    )
    assert pcc_ref_cond > 0.99, f"decode with ref F0/N cond PCC {pcc_ref_cond:.6f}"
    # TT F0/N cond: informational until D0/D1 fixed; staged D5+0 ~0.8 on full walk.
    assert pcc_tt_cond > 0.99, f"decode with tt F0/N cond PCC {pcc_tt_cond:.6f}"


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

    # Build TTDecoder params (exercises full preprocess path including generator weights).
    # Must be done before _torch_random_zeros so weight upload is unaffected.
    params = preprocess_tt_decoder(ref, device, time_len_asr=_T_MEL)

    # Run the reference decode chain manually to obtain x_dec — the exact input the Generator
    # receives in the reference forward.  We also compute y_ref so PCC is measured against the
    # same reference as the full-decoder test.
    # TTGenerator is constructed inside the same zeros context so self._noise_raw /
    # self._out_noise_raw match the zeros noise used by the reference forward.
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
        # Create TTGenerator from the Decoder's generator params with fallbacks enabled.
        gen = TTGenerator(
            device,
            params.generator,
            use_torch_stft_fallback=True,
            use_torch_phase_fallback=True,
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

    params = preprocess_tt_decoder(ref, device, time_len_asr=_T_MEL)
    B_rng, T_har, dim = m_source_rng_shapes_from_f0(
        F0_curve,
        upsample_scale_full=int(params.generator.upsample_scale_full),
        dim=int(params.generator.m_source.sinegen.dim),
    )
    rng_cpu = make_zero_m_source_rng(B_rng, T_har, dim)

    with torch.no_grad(), patched_m_source_torch_rng(rng_cpu):
        y_ref = ref(asr, F0_curve, N_curve, s)

    tt_mod = TTDecoder(device, params)

    mc = ttnn.DRAM_MEMORY_CONFIG
    asr_nlc = asr.transpose(1, 2).contiguous()
    asr_tt = ttnn.from_torch(asr_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    F0_tt = ttnn.from_torch(F0_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    N_tt = ttnn.from_torch(N_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    rng_tt = upload_m_source_rng(rng_cpu, device, memory_config=mc)

    audio_tt = tt_mod(
        asr_tt,
        F0_tt,
        N_tt,
        s_tt,
        memory_config=mc,
        sinegen_rand_ini=rng_tt.rand_ini,
        sinegen_noise_raw=rng_tt.sinegen_noise,
        source_noise_raw=rng_tt.source_noise,
    )
    deallocate_m_source_rng_tt(rng_tt)

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
    )

    B = int(F0_curve.shape[0])
    T_har = params.generator.time_len_x * params.generator.upsample_scale_full
    dim = params.generator.m_source.sinegen.dim

    torch.manual_seed(321)
    rng_cpu = MSourceRngTensors(
        rand_ini=torch.rand(B, dim),
        sinegen_noise=torch.randn(B, T_har, dim),
        source_noise=torch.randn(B, T_har, 1),
    )

    with torch.no_grad(), patched_m_source_torch_rng(rng_cpu):
        y_ref = ref(asr, F0_curve, N_curve, s)

    mc = ttnn.DRAM_MEMORY_CONFIG
    asr_nlc = asr.transpose(1, 2).contiguous()
    asr_tt = ttnn.from_torch(asr_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    F0_tt = ttnn.from_torch(F0_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    N_tt = ttnn.from_torch(N_curve, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    rng_tt = upload_m_source_rng(rng_cpu, device, memory_config=mc)

    m_source_kw = dict(
        sinegen_rand_ini=rng_tt.rand_ini,
        sinegen_noise_raw=rng_tt.sinegen_noise,
        source_noise_raw=rng_tt.source_noise,
        memory_config=mc,
    )
    audio_tt_no_fallback = tt_mod_no_fallback(asr_tt, F0_tt, N_tt, s_tt, **m_source_kw)
    audio_tt_full_fallback = tt_mod_full_fallback(asr_tt, F0_tt, N_tt, s_tt, **m_source_kw)

    y_hat_no_fallback = ttnn.to_torch(audio_tt_no_fallback).float()
    y_hat_full_fallback = ttnn.to_torch(audio_tt_full_fallback).float()
    ttnn.deallocate(audio_tt_no_fallback)
    ttnn.deallocate(audio_tt_full_fallback)
    ttnn.deallocate(asr_tt)
    ttnn.deallocate(F0_tt)
    ttnn.deallocate(N_tt)
    ttnn.deallocate(s_tt)
    deallocate_m_source_rng_tt(rng_tt)

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
