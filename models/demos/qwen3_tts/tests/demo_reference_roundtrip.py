# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Demo: Reference Implementation Roundtrip Test

Tests the full encode-decode pipeline with real audio:
1. Load reference audio (clone_ref.wav)
2. Encode to RVQ codes using Speech Tokenizer Encoder
3. Decode back to audio using Speech Tokenizer Decoder
4. Compare and save output
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal


def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    # Load with soundfile
    waveform, sr = sf.read(path)

    # Convert to tensor
    waveform = torch.from_numpy(waveform.astype(np.float32))

    # Handle stereo -> mono
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)

    # Resample if needed
    if sr != target_sr:
        # Use scipy for resampling
        num_samples = int(len(waveform) * target_sr / sr)
        waveform_np = waveform.numpy()
        waveform_resampled = signal.resample(waveform_np, num_samples)
        waveform = torch.from_numpy(waveform_resampled.astype(np.float32))

    # Add dimensions: [1, 1, samples]
    waveform = waveform.unsqueeze(0).unsqueeze(0)

    return waveform


def compute_mel_spectrogram(audio: torch.Tensor, n_mels: int = 128, sample_rate: int = 24000) -> torch.Tensor:
    """Compute mel spectrogram for speaker encoder using scipy."""
    # Remove batch dimensions
    waveform = audio.squeeze().numpy()

    # STFT parameters
    n_fft = 1024
    hop_length = 256
    win_length = 1024

    # Compute STFT
    f, t, Zxx = signal.stft(waveform, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length)

    # Power spectrogram
    power_spec = np.abs(Zxx) ** 2

    # Create mel filterbank
    mel_basis = _create_mel_filterbank(sample_rate, n_fft, n_mels, f_min=0, f_max=8000)

    # Apply mel filterbank
    mel_spec = np.dot(mel_basis, power_spec)

    # Log mel
    mel_spec = np.log(np.maximum(mel_spec, 1e-5))

    # Convert to tensor [1, n_mels, time]
    mel = torch.from_numpy(mel_spec.astype(np.float32)).unsqueeze(0)

    return mel


def _create_mel_filterbank(sr, n_fft, n_mels, f_min=0, f_max=None):
    """Create a mel filterbank matrix."""
    if f_max is None:
        f_max = sr / 2

    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Compute mel points
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Frequency bins
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Create filterbank
    filterbank = np.zeros((n_mels, len(freq_bins)))
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        for j, freq in enumerate(freq_bins):
            if left <= freq < center:
                filterbank[i, j] = (freq - left) / (center - left)
            elif center <= freq < right:
                filterbank[i, j] = (right - freq) / (right - center)

    return filterbank


def main():
    print("=" * 80)
    print("Reference Implementation Roundtrip Demo")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeakerEncoderConfig,
        SpeechTokenizerDecoderConfig,
        SpeechTokenizerEncoderConfig,
        extract_speaker_encoder_weights,
        extract_speech_tokenizer_encoder_weights,
        speaker_encoder_forward,
        speech_tokenizer_decoder_forward,
        speech_tokenizer_encoder_forward,
    )

    # Load weights
    print("\n1. Loading model weights...")
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    # Speech tokenizer weights
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)

    encoder_weights = extract_speech_tokenizer_encoder_weights(raw_dict)
    decoder_weights = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}
    print(f"   Loaded {len(encoder_weights)} encoder weights, {len(decoder_weights)} decoder weights")

    # Speaker encoder weights
    main_model_path = model_path / "model.safetensors"
    main_dict = load_file(main_model_path)
    speaker_weights = extract_speaker_encoder_weights(main_dict)
    print(f"   Loaded {len(speaker_weights)} speaker encoder weights")

    # Load reference audio
    print("\n2. Loading reference audio...")
    audio_path = "/tmp/clone_ref.wav"
    if not Path(audio_path).exists():
        print(f"   ERROR: {audio_path} not found!")
        print("   Please provide a reference audio file.")
        return

    audio = load_audio(audio_path, target_sr=24000)
    duration = audio.shape[-1] / 24000
    print(f"   Input audio: shape={audio.shape}, duration={duration:.2f}s")
    print(f"   Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Test Speaker Encoder
    print("\n3. Testing Speaker Encoder...")
    mel = compute_mel_spectrogram(audio, n_mels=128)
    print(f"   Mel spectrogram: shape={mel.shape}")

    with torch.no_grad():
        speaker_embedding = speaker_encoder_forward(mel, speaker_weights, SpeakerEncoderConfig())
    print(f"   Speaker embedding: shape={speaker_embedding.shape}")
    print(f"   Embedding range: [{speaker_embedding.min():.4f}, {speaker_embedding.max():.4f}]")
    print(f"   Embedding norm: {speaker_embedding.norm():.4f}")

    # Encode audio to RVQ codes
    print("\n4. Encoding audio to RVQ codes...")
    encoder_config = SpeechTokenizerEncoderConfig()

    with torch.no_grad():
        codes = speech_tokenizer_encoder_forward(audio, encoder_weights, encoder_config)

    print(f"   Output codes: shape={codes.shape}")
    print(f"   Code range: [{codes.min()}, {codes.max()}]")
    expected_seq_len = audio.shape[-1] // 1920  # 24kHz / 12.5Hz = 1920
    print(f"   Expected seq_len: ~{expected_seq_len} (got {codes.shape[-1]})")

    # Decode codes back to audio
    print("\n5. Decoding RVQ codes to audio...")
    decoder_config = SpeechTokenizerDecoderConfig()

    with torch.no_grad():
        reconstructed = speech_tokenizer_decoder_forward(codes, decoder_weights, decoder_config)

    print(f"   Reconstructed audio: shape={reconstructed.shape}")
    print(f"   Audio range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    print(f"   Audio std: {reconstructed.std():.4f}")

    # Save outputs
    print("\n6. Saving audio files...")

    # Save reconstructed audio
    output_path = "/tmp/reference_roundtrip.wav"
    audio_np = reconstructed.squeeze().cpu().float().numpy()
    sf.write(output_path, audio_np, 24000)
    print(f"   Saved: {output_path}")

    # Compare durations
    original_duration = audio.shape[-1] / 24000
    reconstructed_duration = reconstructed.shape[-1] / 24000
    print(f"\n7. Comparison:")
    print(f"   Original duration: {original_duration:.2f}s")
    print(f"   Reconstructed duration: {reconstructed_duration:.2f}s")

    # Compute energy correlation
    # Trim to same length
    min_len = min(audio.shape[-1], reconstructed.shape[-1])
    orig_trimmed = audio[..., :min_len].squeeze()
    recon_trimmed = reconstructed[..., :min_len].squeeze()

    # Compute envelope (using Hilbert transform approximation with abs)
    window_size = 480  # 20ms at 24kHz
    orig_env = torch.nn.functional.avg_pool1d(
        orig_trimmed.abs().unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=window_size // 2
    ).squeeze()
    recon_env = torch.nn.functional.avg_pool1d(
        recon_trimmed.abs().unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=window_size // 2
    ).squeeze()

    # PCC
    orig_env = orig_env - orig_env.mean()
    recon_env = recon_env - recon_env.mean()
    pcc = (orig_env * recon_env).sum() / (orig_env.norm() * recon_env.norm() + 1e-8)
    print(f"   Energy envelope PCC: {pcc:.4f}")

    if pcc > 0.8:
        print("\n✓ Roundtrip test PASSED! High correlation between original and reconstructed audio.")
    elif pcc > 0.5:
        print("\n⚠ Roundtrip test PARTIAL. Moderate correlation - some reconstruction quality.")
    else:
        print("\n✗ Roundtrip test FAILED. Low correlation - reconstruction may have issues.")

    print("\n" + "=" * 80)
    print("Listen to the files to verify quality:")
    print(f"  Original: {audio_path}")
    print(f"  Reconstructed: {output_path}")
    print("=" * 80)

    return codes, reconstructed, speaker_embedding


if __name__ == "__main__":
    main()
