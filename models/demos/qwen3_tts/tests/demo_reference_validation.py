# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Demo: Reference Implementation Validation

Validates reference implementations using official qwen_tts outputs:
1. Decoder: Uses codes from official encoder to verify audio output
2. Speaker Encoder: Computes embedding from audio and compares with official
3. Full pipeline: End-to-end validation

This provides the baseline for TTNN PCC comparison.
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


def compute_mel_spectrogram(audio: torch.Tensor, n_mels: int = 128, sample_rate: int = 24000) -> torch.Tensor:
    """
    Compute mel spectrogram for speaker encoder using exact official qwen_tts params.

    Official params:
    - n_fft=1024, num_mels=128, sampling_rate=24000
    - hop_size=256, win_size=1024
    - fmin=0, fmax=12000
    - Uses librosa mel filterbank with slaney norm
    - Uses Hann window, reflect padding
    - Returns: [batch, time, n_mels] (transposed)
    """
    try:
        from librosa.filters import mel as librosa_mel_fn
    except ImportError:
        print("Warning: librosa not available, using scipy mel")
        return _compute_mel_scipy(audio, n_mels, sample_rate)

    n_fft = 1024
    hop_size = 256
    win_size = 1024
    fmin = 0
    fmax = 12000

    waveform = audio.unsqueeze(0) if audio.dim() == 1 else audio

    # Create mel filterbank (librosa with slaney norm)
    mel_basis = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel_basis).float()

    # Hann window
    hann_window = torch.hann_window(win_size)

    # Reflect padding
    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    # STFT
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    # Apply mel filterbank
    mel_spec = torch.matmul(mel_basis, spec)

    # Dynamic range compression
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    # Transpose to [batch, time, n_mels] as expected by speaker encoder
    mel_spec = mel_spec.transpose(1, 2)

    return mel_spec


def _compute_mel_scipy(audio: torch.Tensor, n_mels: int = 128, sample_rate: int = 24000) -> torch.Tensor:
    """Fallback mel computation using scipy (less accurate)."""
    waveform = audio.numpy()

    n_fft = 1024
    hop_length = 256
    win_length = 1024

    f, t, Zxx = signal.stft(waveform, fs=sample_rate, nperseg=win_length, noverlap=win_length - hop_length)
    power_spec = np.abs(Zxx) ** 2

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(12000)  # fmax=12000
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
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

    mel_spec = np.dot(filterbank, power_spec)
    mel_spec = np.log(np.maximum(mel_spec, 1e-5))

    # Return as [batch, time, n_mels]
    return torch.from_numpy(mel_spec.T.astype(np.float32)).unsqueeze(0)


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    print("=" * 80)
    print("Reference Implementation Validation Demo")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeakerEncoderConfig,
        SpeechTokenizerDecoderConfig,
        extract_speaker_encoder_weights,
        speaker_encoder_forward,
        speech_tokenizer_decoder_forward,
    )

    # Load saved tensors from official qwen_tts
    print("\n1. Loading official qwen_tts outputs...")
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print(f"   ERROR: {voice_clone_path} not found!")
        print("   Please run the official qwen_tts encoder first to generate these.")
        return

    data = torch.load(voice_clone_path)
    official_codes = data["ref_code"]  # [101, 16]
    official_embedding = data["ref_spk_embedding"]  # [2048]
    ref_text = data["ref_text"]

    print(f"   Official codes: {official_codes.shape}")
    print(f"   Official speaker embedding: {official_embedding.shape}")
    print(f"   Reference text: '{ref_text[:50]}...'")

    # Load model weights
    print("\n2. Loading model weights...")
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)
    decoder_weights = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}

    main_model_path = model_path / "model.safetensors"
    main_dict = load_file(main_model_path)
    speaker_weights = extract_speaker_encoder_weights(main_dict)

    print(f"   Decoder weights: {len(decoder_weights)}")
    print(f"   Speaker encoder weights: {len(speaker_weights)}")

    # Test Decoder with official codes
    print("\n3. Testing Reference Decoder with official codes...")
    codes = official_codes.T.unsqueeze(0)  # [1, 16, 101]
    print(f"   Input codes shape: {codes.shape}")

    decoder_config = SpeechTokenizerDecoderConfig()
    with torch.no_grad():
        audio = speech_tokenizer_decoder_forward(codes, decoder_weights, decoder_config)

    print(f"   Output audio shape: {audio.shape}")
    print(f"   Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"   Audio std: {audio.std():.4f}")

    # Save decoder output
    output_path = "/tmp/reference_decoder_validation.wav"
    audio_np = audio.squeeze().cpu().float().numpy()
    sf.write(output_path, audio_np, 24000)
    print(f"   Saved: {output_path}")

    # Compare with official decoder output (clone_ref.wav)
    print("\n4. Comparing with official decoder output...")
    official_audio_path = "/tmp/clone_ref.wav"
    if Path(official_audio_path).exists():
        official_audio = load_audio(official_audio_path, target_sr=24000)
        ref_audio = audio.squeeze()

        # Trim to same length
        min_len = min(len(official_audio), len(ref_audio))
        official_trimmed = official_audio[:min_len]
        ref_trimmed = ref_audio[:min_len]

        # Compute PCC
        pcc = compute_pcc(official_trimmed, ref_trimmed)
        print(f"   Direct waveform PCC: {pcc:.4f}")

        # Compute energy envelope PCC (more robust to phase differences)
        window_size = 480

        def get_envelope(x):
            return torch.nn.functional.avg_pool1d(
                x.abs().unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=window_size // 2
            ).squeeze()

        off_env = get_envelope(official_trimmed)
        ref_env = get_envelope(ref_trimmed)
        env_pcc = compute_pcc(off_env, ref_env)
        print(f"   Energy envelope PCC: {env_pcc:.4f}")

        if env_pcc > 0.9:
            print("   ✓ Decoder output matches official very well!")
        elif env_pcc > 0.7:
            print("   ⚠ Decoder output has moderate match with official")
        else:
            print("   ✗ Decoder output differs significantly from official")
    else:
        print(f"   Official audio not found at {official_audio_path}")

    # Test Speaker Encoder
    print("\n5. Testing Reference Speaker Encoder...")
    if Path(official_audio_path).exists():
        audio_raw = load_audio(official_audio_path, target_sr=24000)
        mel = compute_mel_spectrogram(audio_raw, n_mels=128)
        # Transpose from [batch, time, n_mels] to [batch, n_mels, time] for speaker encoder
        mel = mel.transpose(1, 2)
        print(f"   Mel spectrogram shape: {mel.shape}")

        speaker_config = SpeakerEncoderConfig()
        with torch.no_grad():
            embedding = speaker_encoder_forward(mel, speaker_weights, speaker_config)

        print(f"   Output embedding shape: {embedding.shape}")
        print(f"   Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print(f"   Embedding norm: {embedding.norm():.4f}")

        # Compare with official embedding
        ref_emb = embedding.squeeze()
        official_emb = official_embedding

        emb_pcc = compute_pcc(ref_emb, official_emb)
        print(f"   Embedding PCC vs official: {emb_pcc:.4f}")

        if emb_pcc > 0.9:
            print("   ✓ Speaker embedding matches official very well!")
        elif emb_pcc > 0.7:
            print("   ⚠ Speaker embedding has moderate match with official")
        else:
            print("   ✗ Speaker embedding differs significantly from official")
            print("   Note: This may be due to different mel spectrogram computation")
    else:
        print(f"   Audio not found at {official_audio_path}")

    print("\n" + "=" * 80)
    print("Summary:")
    print("  - Reference decoder produces audio from official codes")
    print("  - Reference speaker encoder produces embeddings from mel")
    print("  - These can be used as baseline for TTNN PCC comparison")
    print("=" * 80)


if __name__ == "__main__":
    main()
