# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Complete Reference Implementation Demo for Qwen3-TTS

This demo tests ALL reference blocks from functional.py:
1. Mel Spectrogram computation
2. Speaker Encoder (ECAPA-TDNN)
3. Speech Tokenizer Encoder (MimiModel-based)
4. Speech Tokenizer Decoder

It generates audio using only our reference implementations and compares
against official qwen_tts outputs to verify accuracy.

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_complete_reference.py
"""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    from scipy import signal

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


def main():
    """Run complete reference demo."""
    print("=" * 80)
    print("Complete Reference Implementation Demo")
    print("=" * 80)

    # Check for input audio
    audio_path = "/tmp/clone_ref.wav"
    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found at {audio_path}")
        print("Run the official qwen_tts demo first to generate this file.")
        return

    # Import reference implementations
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeakerEncoderConfig,
        SpeechTokenizerDecoderConfig,
        compute_mel_spectrogram_qwen,
        extract_speaker_encoder_weights,
        speaker_encoder_forward,
        speech_tokenizer_decoder_forward,
        speech_tokenizer_encoder_forward_mimi,
    )

    # =========================================================================
    # Step 1: Load and process audio
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Load Audio")
    print("=" * 80)

    audio = load_audio(audio_path)
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio duration: {len(audio) / 24000:.2f}s")
    print(f"  Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # =========================================================================
    # Step 2: Compute Mel Spectrogram
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Compute Mel Spectrogram (Reference)")
    print("=" * 80)

    start_time = time.time()
    mel = compute_mel_spectrogram_qwen(audio)
    mel_time = time.time() - start_time

    print(f"  Mel shape: {mel.shape}")
    print(f"  Mel range: [{mel.min():.4f}, {mel.max():.4f}]")
    print(f"  Time: {mel_time * 1000:.1f} ms")

    # =========================================================================
    # Step 3: Extract Speaker Embedding
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Extract Speaker Embedding (Reference)")
    print("=" * 80)

    # Load speaker encoder weights
    print("  Loading speaker encoder weights...")
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["model.safetensors"])
    model_path = Path(model_path)
    main_dict = load_file(model_path / "model.safetensors")
    speaker_weights = extract_speaker_encoder_weights(main_dict)
    print(f"  Loaded {len(speaker_weights)} speaker encoder weights")

    # Run speaker encoder
    start_time = time.time()
    config = SpeakerEncoderConfig()
    speaker_embedding = speaker_encoder_forward(mel, speaker_weights, config)
    speaker_time = time.time() - start_time

    print(f"  Speaker embedding shape: {speaker_embedding.shape}")
    print(f"  Speaker embedding range: [{speaker_embedding.min():.4f}, {speaker_embedding.max():.4f}]")
    print(f"  Speaker embedding norm: {speaker_embedding.norm():.4f}")
    print(f"  Time: {speaker_time * 1000:.1f} ms")

    # Compare with official if available
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if voice_clone_path.exists():
        data = torch.load(voice_clone_path, weights_only=False)
        official_embedding = data["ref_spk_embedding"]
        pcc = compute_pcc(speaker_embedding.squeeze(), official_embedding)
        print(f"  PCC vs official: {pcc:.6f}")

    # =========================================================================
    # Step 4: Encode Audio to Codes
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Encode Audio to RVQ Codes (Reference)")
    print("=" * 80)

    start_time = time.time()
    codes = speech_tokenizer_encoder_forward_mimi(audio.unsqueeze(0))
    encoder_time = time.time() - start_time

    print(f"  Codes shape: {codes.shape}")  # Should be [1, 16, seq_len]
    print(f"  Codes range: [{codes.min()}, {codes.max()}]")
    print(f"  Time: {encoder_time * 1000:.1f} ms")

    # Compare with official if available
    if voice_clone_path.exists():
        official_codes = data["ref_code"].T.unsqueeze(0)  # [1, 16, seq_len]
        min_len = min(codes.shape[-1], official_codes.shape[-1])
        ref_trimmed = codes[..., :min_len]
        official_trimmed = official_codes[..., :min_len]
        match_ratio = (ref_trimmed == official_trimmed).float().mean().item()
        print(f"  Match ratio vs official: {match_ratio:.6f}")

    # =========================================================================
    # Step 5: Decode Codes to Audio
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Decode RVQ Codes to Audio (Reference)")
    print("=" * 80)

    # Load decoder weights
    print("  Loading decoder weights...")
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)
    decoder_weights = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}
    print(f"  Loaded {len(decoder_weights)} decoder weights")

    # Run decoder
    start_time = time.time()
    decoder_config = SpeechTokenizerDecoderConfig()
    reconstructed = speech_tokenizer_decoder_forward(codes, decoder_weights, decoder_config)
    decoder_time = time.time() - start_time

    print(f"  Reconstructed audio shape: {reconstructed.shape}")
    print(f"  Reconstructed duration: {reconstructed.shape[-1] / 24000:.2f}s")
    print(f"  Time: {decoder_time * 1000:.1f} ms")

    # =========================================================================
    # Step 6: Quality Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Quality Analysis")
    print("=" * 80)

    # Compare original and reconstructed
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

    energy_pcc = compute_pcc(orig_env, recon_env)
    print(f"  Energy envelope PCC: {energy_pcc:.4f}")

    # =========================================================================
    # Step 7: Save Output
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Save Output")
    print("=" * 80)

    output_path = "/tmp/reference_complete_roundtrip.wav"
    audio_np = reconstructed.squeeze().detach().cpu().float().numpy()
    sf.write(output_path, audio_np, 24000)
    print(f"  Saved to: {output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Input: {audio_path}")
    print(f"  Output: {output_path}")
    print(f"")
    print(f"  Mel Spectrogram:     {mel_time * 1000:6.1f} ms")
    print(f"  Speaker Encoder:     {speaker_time * 1000:6.1f} ms")
    print(f"  Speech Encoder:      {encoder_time * 1000:6.1f} ms")
    print(f"  Speech Decoder:      {decoder_time * 1000:6.1f} ms")
    print(f"  Total:               {(mel_time + speaker_time + encoder_time + decoder_time) * 1000:6.1f} ms")
    print(f"")
    print(f"  Energy PCC:          {energy_pcc:.4f}")
    print(f"")
    print("Listen to the output to verify audio quality.")
    print("If it sounds like the input, all reference blocks are working!")
    print("=" * 80)


if __name__ == "__main__":
    main()
