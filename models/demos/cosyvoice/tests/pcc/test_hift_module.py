"""PCC tests for CosyVoice HiFT vocoder (Phase 2c).

Tests:
  1. mel→waveform PCC ≥ 0.99 vs golden/hift/<mode>.pt['waveform']
  2. mel→f0 PCC ≥ 0.99 vs golden/hift/<mode>.pt['f0']
  3. Logs mel-cepstral distance (MCD) for audio quality monitoring.

Run:
  pytest models/demos/cosyvoice/tests/pcc/test_hift_module.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

DEMO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = DEMO_ROOT / "model_data" / "golden" / "hift"
HIFT_PT = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B" / "hift.pt"

_COSYVOICE_SRC = str(DEMO_ROOT / "model_data" / "CosyVoice_src")
_MATCHA = str(DEMO_ROOT / "model_data" / "CosyVoice_src" / "third_party" / "Matcha-TTS")
if _COSYVOICE_SRC not in sys.path:
    sys.path.insert(0, _COSYVOICE_SRC)
if _MATCHA not in sys.path:
    sys.path.append(_MATCHA)

from models.common.utility_functions import comp_pcc

MODES = ["zero_shot", "cross_lingual", "instruct2", "sft"]
PCC_THRESHOLD = 0.99


@pytest.fixture(scope="module")
def vocoder():
    from models.demos.cosyvoice.tt.hifigan.generator import HiFTVocoder

    return HiFTVocoder.from_checkpoint(str(HIFT_PT))


def _mel_cepstral_distance(wav_a: torch.Tensor, wav_b: torch.Tensor, sr: int = 24000) -> float:
    """Compute mel-cepstral distance (MCD) in dB between two waveforms."""

    n_fft = 1024
    hop = 256
    n_mels = 80

    def _melspec(wav):
        wav = wav.squeeze().float()
        spec = torch.stft(
            wav, n_fft, hop_length=hop, win_length=n_fft, window=torch.hann_window(n_fft), return_complex=True
        )
        power = spec.abs().pow(2)
        mel_basis = torch.from_numpy(__import__("librosa").filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)).float()
        mel = mel_basis @ power
        log_mel = torch.log(mel.clamp(min=1e-10))
        return log_mel

    mcd_a = _melspec(wav_a)
    mcd_b = _melspec(wav_b)
    min_len = min(mcd_a.shape[-1], mcd_b.shape[-1])
    diff = mcd_a[:, :min_len] - mcd_b[:, :min_len]
    mcd = (10.0 / torch.log(torch.tensor(10.0))) * torch.sqrt((diff**2).sum(dim=0)).mean()
    return mcd.item()


@pytest.mark.parametrize("mode", MODES)
def test_hift_waveform_pcc(vocoder, mode):
    """Vocoder mel→waveform PCC ≥ 0.99."""
    golden_path = GOLDEN_DIR / f"{mode}.pt"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    g = torch.load(str(golden_path), map_location="cpu", weights_only=False)
    mel_in = g["mel_in"]
    golden_waveform = g["waveform"]

    torch.manual_seed(1986)
    waveform, source = vocoder.inference(mel_in)

    pcc_val, _ = comp_pcc(golden_waveform, waveform)
    print(f"\n[{mode}] waveform PCC = {pcc_val:.6f}, shape = {waveform.shape}")
    assert pcc_val >= PCC_THRESHOLD, f"PCC {pcc_val:.6f} < {PCC_THRESHOLD}"


@pytest.mark.parametrize("mode", MODES)
def test_hift_f0_pcc(vocoder, mode):
    """Vocoder f0 predictor PCC ≥ 0.99."""
    golden_path = GOLDEN_DIR / f"{mode}.pt"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    g = torch.load(str(golden_path), map_location="cpu", weights_only=False)
    mel_in = g["mel_in"]
    golden_f0 = g["f0"]

    with torch.inference_mode():
        f0 = vocoder.model.f0_predictor(mel_in)

    pcc_val, _ = comp_pcc(golden_f0, f0)
    print(f"\n[{mode}] f0 PCC = {pcc_val:.6f}")
    assert pcc_val >= PCC_THRESHOLD, f"PCC {pcc_val:.6f} < {PCC_THRESHOLD}"


@pytest.mark.parametrize("mode", MODES)
def test_hift_mcd(vocoder, mode):
    """Log mel-cepstral distance (informational, no hard gate)."""
    golden_path = GOLDEN_DIR / f"{mode}.pt"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    g = torch.load(str(golden_path), map_location="cpu", weights_only=False)
    mel_in = g["mel_in"]
    golden_waveform = g["waveform"]

    torch.manual_seed(1986)
    waveform, _ = vocoder.inference(mel_in)

    try:
        mcd = _mel_cepstral_distance(waveform, golden_waveform)
        print(f"\n[{mode}] MCD = {mcd:.2f} dB")
    except Exception as e:
        print(f"\n[{mode}] MCD computation skipped: {e}")
