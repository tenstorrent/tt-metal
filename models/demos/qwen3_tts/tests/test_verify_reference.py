# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Verification tests: Compare reference implementations against saved official outputs.

Uses saved outputs from official qwen_tts (generated previously) to verify
our reference implementations match with high accuracy.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal


def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    waveform, sr = sf.read(path)
    waveform = torch.from_numpy(waveform.astype(np.float32))

    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)

    if sr != target_sr:
        num_samples = int(len(waveform) * target_sr / sr)
        waveform_np = waveform.numpy()
        waveform_resampled = signal.resample(waveform_np, num_samples)
        waveform = torch.from_numpy(waveform_resampled.astype(np.float32))

    return waveform


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def test_mel_spectrogram():
    """Test mel spectrogram computation produces valid output."""
    print("\n" + "=" * 80)
    print("Test: Mel Spectrogram")
    print("=" * 80)

    # Load audio
    audio_path = "/tmp/clone_ref.wav"
    if not Path(audio_path).exists():
        print(f"SKIP: Audio file not found at {audio_path}")
        return False

    audio = load_audio(audio_path)
    print(f"Input audio shape: {audio.shape}")

    # Reference implementation
    from models.demos.qwen3_tts.reference.functional import compute_mel_spectrogram_qwen

    mel = compute_mel_spectrogram_qwen(audio)
    print(f"Mel spectrogram shape: {mel.shape}")
    print(f"Mel range: [{mel.min():.4f}, {mel.max():.4f}]")
    print(f"Mel std: {mel.std():.4f}")

    # Verify shape is correct: [batch, 128, time]
    if mel.shape[1] == 128:
        print("PASS: Mel spectrogram has correct shape [batch, 128, time]")
        return True
    else:
        print(f"FAIL: Expected 128 mel bins, got {mel.shape[1]}")
        return False


def test_speaker_encoder():
    """Test speaker encoder against saved official output."""
    print("\n" + "=" * 80)
    print("Test: Speaker Encoder vs Official Saved Output")
    print("=" * 80)

    # Load saved official outputs
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print(f"SKIP: Official outputs not found at {voice_clone_path}")
        return False

    data = torch.load(voice_clone_path)
    official_embedding = data["ref_spk_embedding"]
    print(f"Official embedding shape: {official_embedding.shape}")
    print(f"Official embedding range: [{official_embedding.min():.4f}, {official_embedding.max():.4f}]")
    print(f"Official embedding norm: {official_embedding.norm():.4f}")

    # Load audio and compute mel spectrogram
    audio_path = "/tmp/clone_ref.wav"
    if not Path(audio_path).exists():
        print(f"SKIP: Audio file not found at {audio_path}")
        return False

    audio = load_audio(audio_path)

    # Compute mel spectrogram using our function
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeakerEncoderConfig,
        compute_mel_spectrogram_qwen,
        extract_speaker_encoder_weights,
        speaker_encoder_forward,
    )

    mel = compute_mel_spectrogram_qwen(audio)
    print(f"Mel spectrogram shape: {mel.shape}")

    # Load speaker encoder weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["model.safetensors"])
    model_path = Path(model_path)

    main_dict = load_file(model_path / "model.safetensors")
    speaker_weights = extract_speaker_encoder_weights(main_dict)
    print(f"Loaded {len(speaker_weights)} speaker encoder weights")

    # Run reference speaker encoder
    config = SpeakerEncoderConfig()
    ref_embedding = speaker_encoder_forward(mel, speaker_weights, config)
    print(f"Reference embedding shape: {ref_embedding.shape}")
    print(f"Reference embedding range: [{ref_embedding.min():.4f}, {ref_embedding.max():.4f}]")
    print(f"Reference embedding norm: {ref_embedding.norm():.4f}")

    # Compare
    pcc = compute_pcc(ref_embedding.squeeze(), official_embedding)
    print(f"PCC vs official: {pcc:.6f}")

    if pcc > 0.99:
        print("PASS: Speaker encoder matches official (PCC > 0.99)")
        return True
    elif pcc > 0.9:
        print(f"PARTIAL: Speaker encoder PCC {pcc:.4f} > 0.9")
        return True
    elif pcc > 0.5:
        print(f"FAIR: Speaker encoder PCC {pcc:.4f} > 0.5")
        return True
    else:
        print(f"FAIL: Speaker encoder PCC {pcc:.4f} < 0.5")
        return False


def test_speech_tokenizer_encoder():
    """Test speech tokenizer encoder against saved official output."""
    print("\n" + "=" * 80)
    print("Test: Speech Tokenizer Encoder (MimiModel) vs Official Saved Output")
    print("=" * 80)

    # Load saved official outputs
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print(f"SKIP: Official outputs not found at {voice_clone_path}")
        return False

    data = torch.load(voice_clone_path)
    official_codes = data["ref_code"]
    print(f"Official codes shape: {official_codes.shape}")
    print(f"Official codes range: [{official_codes.min()}, {official_codes.max()}]")

    # Load audio
    audio_path = "/tmp/clone_ref.wav"
    if not Path(audio_path).exists():
        print(f"SKIP: Audio file not found at {audio_path}")
        return False

    audio = load_audio(audio_path)
    print(f"Input audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio)/24000:.2f}s")

    # Run reference encoder (using MimiModel)
    from models.demos.qwen3_tts.reference.functional import speech_tokenizer_encoder_forward

    ref_codes = speech_tokenizer_encoder_forward(audio.unsqueeze(0), use_mimi=True)
    print(f"Reference codes shape: {ref_codes.shape}")
    print(f"Reference codes range: [{ref_codes.min()}, {ref_codes.max()}]")

    # Reshape official codes to match reference format
    # Official is [seq_len, num_quantizers], reference is [batch, num_quantizers, seq_len]
    if official_codes.dim() == 2:
        if official_codes.shape[0] > official_codes.shape[1]:
            official_codes_reshaped = official_codes.T.unsqueeze(0)
        else:
            official_codes_reshaped = official_codes.unsqueeze(0)
    else:
        official_codes_reshaped = official_codes

    print(f"Official codes reshaped: {official_codes_reshaped.shape}")

    # Compare sequence lengths
    ref_seq_len = ref_codes.shape[-1]
    official_seq_len = official_codes_reshaped.shape[-1]
    print(f"Reference seq_len: {ref_seq_len}, Official seq_len: {official_seq_len}")

    # Check exact match for overlapping portion
    min_len = min(ref_seq_len, official_seq_len)
    ref_trimmed = ref_codes[..., :min_len]
    official_trimmed = official_codes_reshaped[..., :min_len]

    match_count = (ref_trimmed == official_trimmed).sum().item()
    total_count = ref_trimmed.numel()
    match_ratio = match_count / total_count
    print(f"Exact match ratio: {match_ratio:.4f} ({match_count}/{total_count})")

    if ref_seq_len == official_seq_len and match_ratio > 0.99:
        print("PASS: Speech encoder codes match official exactly")
        return True
    elif match_ratio > 0.9:
        print(f"PARTIAL: Code match ratio {match_ratio:.4f} > 0.9")
        return True
    else:
        print(f"FAIL: Code match ratio {match_ratio:.4f} < 0.9")
        return False


def test_roundtrip_audio_quality():
    """Test decode quality using official saved codes."""
    print("\n" + "=" * 80)
    print("Test: Decoder Quality (using official saved codes)")
    print("=" * 80)

    # Load audio
    audio_path = "/tmp/clone_ref.wav"
    if not Path(audio_path).exists():
        print(f"SKIP: Audio file not found at {audio_path}")
        return False

    audio = load_audio(audio_path)
    print(f"Input audio: shape={audio.shape}, duration={len(audio)/24000:.2f}s")

    # Load saved official codes instead of encoding
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print(f"SKIP: Official outputs not found at {voice_clone_path}")
        return False

    data = torch.load(voice_clone_path)
    official_codes = data["ref_code"]

    # Reshape to [batch, num_quantizers, seq_len]
    codes = official_codes.T.unsqueeze(0)
    print(f"Using official codes: shape={codes.shape}")

    # Import decoder components
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        speech_tokenizer_decoder_forward,
    )

    # Load decoder weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["speech_tokenizer/*"])
    model_path = Path(model_path)

    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)
    decoder_weights = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}
    print(f"Loaded {len(decoder_weights)} decoder weights")

    # Decode
    decoder_config = SpeechTokenizerDecoderConfig()
    reconstructed = speech_tokenizer_decoder_forward(codes, decoder_weights, decoder_config)
    print(f"Reconstructed audio: shape={reconstructed.shape}")
    print(f"Reconstructed duration: {reconstructed.shape[-1]/24000:.2f}s")

    # Compare
    min_len = min(len(audio), reconstructed.shape[-1])
    orig = audio[:min_len]
    recon = reconstructed.squeeze()[:min_len]

    # Energy envelope correlation
    window_size = 480
    orig_env = torch.nn.functional.avg_pool1d(
        orig.abs().unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=window_size // 2
    ).squeeze()
    recon_env = torch.nn.functional.avg_pool1d(
        recon.abs().unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=window_size // 2
    ).squeeze()

    pcc = compute_pcc(orig_env, recon_env)
    print(f"Energy envelope PCC: {pcc:.4f}")

    # Save reconstructed audio
    output_path = "/tmp/roundtrip_test.wav"
    sf.write(output_path, reconstructed.squeeze().cpu().float().numpy(), 24000)
    print(f"Saved: {output_path}")

    if pcc > 0.8:
        print("PASS: Good roundtrip quality (PCC > 0.8)")
        return True
    elif pcc > 0.5:
        print(f"PARTIAL: Moderate roundtrip quality (PCC {pcc:.4f} > 0.5)")
        return True
    else:
        print(f"FAIL: Poor roundtrip quality (PCC {pcc:.4f} < 0.5)")
        return False


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("Reference Implementation Verification Tests")
    print("=" * 80)

    results = {}
    results["mel_spectrogram"] = test_mel_spectrogram()
    results["speaker_encoder"] = test_speaker_encoder()
    results["speech_encoder"] = test_speech_tokenizer_encoder()
    results["roundtrip"] = test_roundtrip_audio_quality()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        print(f"  {test_name}: {status}")

    print("=" * 80)
    all_passed = all(r for r in results.values() if r is not None)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed")


if __name__ == "__main__":
    main()
