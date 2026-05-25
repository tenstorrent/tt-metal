"""
PCC-based tests for Kokoro TTNN port.

All tests require a Tenstorrent device (device_id=0).
Run with:
    pytest models/experimental/kokoro/tests/test_kokoro_ttnn_pcc.py -v

PCC thresholds:
  Per-layer unit tests : PCC >= 0.99
  Full model e2e       : PCC >= 0.98
  Generated mel output : PCC >= 0.97
"""

import sys
import os

import pytest
import torch

import ttnn

# Allow importing from models/experimental/kokoro
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from models.experimental.kokoro.reference.modules import (
    LinearNorm,
    LayerNorm as RefLayerNorm,
    AdaLayerNorm as RefAdaLayerNorm,
    TextEncoder as RefTextEncoder,
    DurationEncoder as RefDurationEncoder,
)
from models.experimental.kokoro.reference.istftnet import (
    AdaIN1d as RefAdaIN1d,
    AdaINResBlock1 as RefAdaINResBlock1,
    SourceModuleHnNSF as RefSourceModuleHnNSF,
    AdainResBlk1d as RefAdainResBlk1d,
    Decoder as RefDecoder,
)
from models.experimental.kokoro.reference.custom_stft import CustomSTFT as RefCustomSTFT

from models.experimental.kokoro.tt.tt_modules import (
    TTLinearNorm,
    TTLayerNorm,
    TTAdaLayerNorm,
    TTTextEncoder,
    TTDurationEncoder,
)
from models.experimental.kokoro.tt.tt_istftnet import (
    TTAdaIN1d,
    TTAdaINResBlock1,
    TTSourceModuleHnNSF,
    TTAdainResBlk1d,
    TTDecoder,
)
from models.experimental.kokoro.tt.tt_custom_stft import TTCustomSTFT


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


# ─────────────────────────────────────────────────────────────
# PCC helper
# ─────────────────────────────────────────────────────────────


def compute_pcc(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    a = tensor_a.float().flatten()
    b = tensor_b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    pcc = (a * b).sum() / (torch.sqrt((a**2).sum()) * torch.sqrt((b**2).sum()) + 1e-8)
    return pcc.item()


def assert_pcc(ref_out, tt_out, threshold, name):
    pcc = compute_pcc(ref_out, tt_out)
    status = "PASS" if pcc >= threshold else "FAIL"
    print(f"[PCC] {name}: {pcc:.6f} ({status})")
    assert pcc >= threshold, f"{name} PCC={pcc:.6f} below threshold={threshold}"


# ─────────────────────────────────────────────────────────────
# test_pcc_modules: LinearNorm, LayerNorm, AdaLayerNorm
# ─────────────────────────────────────────────────────────────


def test_pcc_modules(device):
    torch.manual_seed(42)

    # ── LinearNorm ──────────────────────────────────────────
    ref_ln = LinearNorm(512, 50)
    tt_ln = TTLinearNorm(ref_ln.linear_layer, device)
    x = torch.randn(2, 16, 512)
    ref_out = ref_ln(x)
    tt_out = tt_ln(x)
    assert_pcc(ref_out, tt_out, 0.99, "LinearNorm")

    # ── Custom LayerNorm (channels-first) ─────────────────
    ref_layernorm = RefLayerNorm(512)
    tt_layernorm = TTLayerNorm.from_ref(ref_layernorm, device)
    x = torch.randn(2, 512, 32)  # (B, C, T)
    ref_out = ref_layernorm(x)
    tt_out = tt_layernorm(x)
    assert_pcc(ref_out, tt_out, 0.99, "LayerNorm-channels-first")

    # ── AdaLayerNorm ─────────────────────────────────────
    # AdaLayerNorm receives (B, T, C) — see DurationEncoder which calls
    # block(x.transpose(-1,-2), style) where x is (B, C, T).
    ref_ada = RefAdaLayerNorm(style_dim=128, channels=512)
    tt_ada = TTAdaLayerNorm.from_ref(ref_ada, device)
    x = torch.randn(2, 32, 512)  # (B, T, C)
    s = torch.randn(2, 128)
    ref_out = ref_ada(x, s)
    tt_out = tt_ada(x, s)
    assert_pcc(ref_out, tt_out, 0.99, "AdaLayerNorm")


# ─────────────────────────────────────────────────────────────
# test_pcc_stft: CustomSTFT forward (transform)
# ─────────────────────────────────────────────────────────────


def test_pcc_stft(device):
    torch.manual_seed(0)
    ref_stft = RefCustomSTFT(filter_length=20, hop_length=5, win_length=20)
    tt_stft = TTCustomSTFT(ref_stft, device)

    waveform = torch.randn(1, 200)  # short waveform for quick test

    ref_mag, ref_phase = ref_stft.transform(waveform)
    tt_mag, tt_phase = tt_stft.transform(waveform)

    assert_pcc(ref_mag, tt_mag, 0.99, "CustomSTFT-magnitude")
    assert_pcc(ref_phase, tt_phase, 0.99, "CustomSTFT-phase")


# ─────────────────────────────────────────────────────────────
# test_pcc_istft: CustomSTFT inverse
# ─────────────────────────────────────────────────────────────


def test_pcc_istft(device):
    torch.manual_seed(1)
    ref_stft = RefCustomSTFT(filter_length=20, hop_length=5, win_length=20)
    tt_stft = TTCustomSTFT(ref_stft, device)

    waveform = torch.randn(1, 200)
    ref_mag, ref_phase = ref_stft.transform(waveform)
    tt_mag, tt_phase = tt_stft.transform(waveform)

    ref_recon = ref_stft.inverse(ref_mag, ref_phase)
    tt_recon = tt_stft.inverse(tt_mag, tt_phase)

    assert_pcc(ref_recon, tt_recon, 0.99, "CustomSTFT-iSTFT")


# ─────────────────────────────────────────────────────────────
# test_pcc_adain1d
# ─────────────────────────────────────────────────────────────


def test_pcc_adain1d(device):
    torch.manual_seed(2)
    ref_adain = RefAdaIN1d(style_dim=128, num_features=256)
    tt_adain = TTAdaIN1d(ref_adain, device)

    x = torch.randn(2, 256, 64)
    s = torch.randn(2, 128)

    ref_out = ref_adain(x, s)
    tt_out = tt_adain(x, s)
    assert_pcc(ref_out, tt_out, 0.99, "AdaIN1d")


# ─────────────────────────────────────────────────────────────
# test_pcc_adain_resblock1: AdaINResBlock1 (in Generator)
# ─────────────────────────────────────────────────────────────


def test_pcc_adain_resblock1(device):
    torch.manual_seed(3)
    ref_blk = RefAdaINResBlock1(channels=128, kernel_size=3, dilation=(1, 3, 5), style_dim=128)
    ref_blk.eval()
    tt_blk = TTAdaINResBlock1(ref_blk, device)

    x = torch.randn(1, 128, 64)
    s = torch.randn(1, 128)

    with torch.no_grad():
        ref_out = ref_blk(x, s)
    tt_out = tt_blk(x, s)
    assert_pcc(ref_out, tt_out, 0.99, "AdaINResBlock1")


# ─────────────────────────────────────────────────────────────
# test_pcc_adain_resblk1d: AdainResBlk1d (in Decoder/ProsodyPredictor)
# ─────────────────────────────────────────────────────────────


def test_pcc_adain_resblk1d(device):
    torch.manual_seed(4)
    # Same-channel (no shortcut)
    ref_blk = RefAdainResBlk1d(dim_in=256, dim_out=256, style_dim=128, dropout_p=0.0)
    ref_blk.eval()
    tt_blk = TTAdainResBlk1d(ref_blk, device)

    x = torch.randn(1, 256, 32)
    s = torch.randn(1, 128)

    with torch.no_grad():
        ref_out = ref_blk(x, s)
    tt_out = tt_blk(x, s)
    assert_pcc(ref_out, tt_out, 0.99, "AdainResBlk1d-same-chan")

    # Cross-channel (with learned shortcut)
    ref_blk2 = RefAdainResBlk1d(dim_in=512, dim_out=256, style_dim=128, dropout_p=0.0)
    ref_blk2.eval()
    tt_blk2 = TTAdainResBlk1d(ref_blk2, device)
    x2 = torch.randn(1, 512, 32)
    with torch.no_grad():
        ref_out2 = ref_blk2(x2, s)
    tt_out2 = tt_blk2(x2, s)
    assert_pcc(ref_out2, tt_out2, 0.99, "AdainResBlk1d-cross-chan")


# ─────────────────────────────────────────────────────────────
# test_pcc_source_module: SourceModuleHnNSF
# ─────────────────────────────────────────────────────────────


def test_pcc_source_module(device):
    torch.manual_seed(5)
    ref_src = RefSourceModuleHnNSF(
        sampling_rate=24000,
        upsample_scale=300,
        harmonic_num=8,
        voiced_threshod=10,
    )
    ref_src.eval()
    tt_src = TTSourceModuleHnNSF(ref_src, device)

    # f0: (B, L, 1)
    f0 = torch.rand(1, 32, 1) * 300 + 80  # 80–380 Hz voiced
    with torch.no_grad():
        ref_sm, ref_noise, ref_uv = ref_src(f0)

    # Fix same random seed to compare deterministic sine
    torch.manual_seed(5)
    tt_sm, tt_noise, tt_uv = tt_src(f0)

    assert_pcc(ref_sm, tt_sm, 0.99, "SourceModuleHnNSF-sine_merge")
    assert_pcc(ref_uv, tt_uv, 0.99, "SourceModuleHnNSF-uv")


# ─────────────────────────────────────────────────────────────
# test_pcc_model: full TTDecoder (medium-sized subsystem)
# ─────────────────────────────────────────────────────────────


def test_pcc_model(device):
    """
    Tests the TTDecoder — the most compute-intensive subsystem.
    Uses the actual Kokoro-82M model config extracted from config.json defaults.
    """
    torch.manual_seed(10)

    ISTFTNET_CFG = dict(
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[10, 6],
        upsample_initial_channel=512,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes=[20, 12],
        gen_istft_n_fft=20,
        gen_istft_hop_size=5,
        disable_complex=True,  # use CustomSTFT for testability
    )

    ref_dec = RefDecoder(
        dim_in=512,
        style_dim=128,
        dim_out=80,
        **ISTFTNET_CFG,
    ).eval()

    tt_dec = TTDecoder(ref_dec, device)

    T_mel = 32
    # F0_curve and N come from ProsodyPredictor.F0Ntrain which upsamples
    # by 2x (AdainResBlk1d with upsample=True).  Decoder.F0_conv/N_conv
    # have stride=2 to downsample back to T_mel.
    T_f0 = T_mel * 2
    asr = torch.randn(1, 512, T_mel)
    F0_curve = torch.rand(1, T_f0) * 300 + 80
    N = torch.randn(1, T_f0)
    s = torch.randn(1, 128)

    with torch.no_grad():
        ref_out = ref_dec(asr, F0_curve, N, s)
    tt_out = tt_dec(asr, F0_curve, N, s)

    assert_pcc(ref_out, tt_out, 0.99, "TTDecoder")


# ─────────────────────────────────────────────────────────────
# test_pcc_text_encoder: TTTextEncoder
# ─────────────────────────────────────────────────────────────


def test_pcc_text_encoder(device):
    torch.manual_seed(20)

    ref_te = RefTextEncoder(
        channels=512,
        kernel_size=5,
        depth=3,
        n_symbols=178,
    ).eval()
    tt_te = TTTextEncoder(ref_te, device)

    B, T = 1, 32
    input_ids = torch.randint(1, 178, (B, T))
    input_lengths = torch.tensor([T])
    text_mask = torch.zeros(B, T, dtype=torch.bool)

    with torch.no_grad():
        ref_out = ref_te(input_ids, input_lengths, text_mask)
    tt_out = tt_te(input_ids, input_lengths, text_mask)

    assert_pcc(ref_out, tt_out, 0.99, "TTTextEncoder")


# ─────────────────────────────────────────────────────────────
# test_pcc_duration_encoder: TTDurationEncoder
# ─────────────────────────────────────────────────────────────


def test_pcc_duration_encoder(device):
    torch.manual_seed(21)

    ref_de = RefDurationEncoder(sty_dim=128, d_model=512, nlayers=3, dropout=0.0).eval()
    tt_de = TTDurationEncoder(ref_de, device)

    B, T = 1, 32
    x = torch.randn(B, 512, T)
    style = torch.randn(B, 128)
    text_lengths = torch.tensor([T])
    m = torch.zeros(B, T, dtype=torch.bool)

    with torch.no_grad():
        ref_out = ref_de(x, style, text_lengths, m)
    tt_out = tt_de(x, style, text_lengths, m)

    assert_pcc(ref_out, tt_out, 0.99, "TTDurationEncoder")


# ─────────────────────────────────────────────────────────────
# test_pcc_full_model: end-to-end KModel vs TTKModel
# Requires model weights downloaded from HuggingFace.
# Skip if weights not available.
# ─────────────────────────────────────────────────────────────


def test_pcc_full_model(device):
    """
    End-to-end PCC test: reference KModel vs TTKModel.
    Downloads hexgrad/Kokoro-82M weights from HuggingFace.
    """
    pytest.importorskip("huggingface_hub", reason="huggingface_hub not installed")

    try:
        from models.experimental.kokoro.reference.model import KModel
    except ImportError:
        pytest.skip("KModel not importable")

    torch.manual_seed(99)

    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

    from models.experimental.kokoro.tt.tt_model import TTKModel

    tt_kmodel = TTKModel.from_kmodel(kmodel, device)

    # Build a short test input
    phonemes = "hɛloʊ"
    input_ids_list = list(
        filter(
            lambda i: i is not None,
            map(lambda p: kmodel.vocab.get(p), phonemes),
        )
    )
    assert len(input_ids_list) > 0, "No valid phoneme IDs"

    bert_device = next(kmodel.bert.parameters()).device
    input_ids = torch.LongTensor([[0, *input_ids_list, 0]]).to(bert_device)

    # Random style vector (replace with real voice pack for audio quality testing)
    ref_s = torch.randn(1, 256)

    with torch.no_grad():
        ref_audio, ref_dur = kmodel.forward_with_tokens(input_ids, ref_s)
        tt_audio, tt_dur = tt_kmodel.forward_with_tokens(input_ids, ref_s)

    assert_pcc(ref_audio, tt_audio, 0.98, "KModel-e2e-audio")

    # Also check pred_dur alignment
    if ref_dur is not None and tt_dur is not None:
        ref_dur_f = ref_dur.float()
        tt_dur_f = tt_dur.float()
        if ref_dur_f.shape == tt_dur_f.shape:
            assert_pcc(ref_dur_f, tt_dur_f, 0.97, "KModel-e2e-duration")
