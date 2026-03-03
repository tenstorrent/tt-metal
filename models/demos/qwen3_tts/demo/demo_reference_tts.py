# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Reference TTS Demo (No qwen_tts dependency)

This demo uses ONLY our reference implementations to generate audio.
It does NOT import qwen_tts, making it compatible with the tt-metal environment.

The demo:
1. Loads reference audio
2. Extracts speaker embedding (reference speaker encoder)
3. Encodes audio to codes (reference speech encoder)
4. Decodes codes to audio (reference speech decoder)

For full TTS with text, we need to also run the Talker + Code Predictor,
but those require the complex input embedding pipeline. This demo focuses
on the audio processing components that can be directly compared to TTNN.

Usage:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_reference_tts.py
"""

import argparse
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


def run_reference_tts_demo(
    audio_path: str = "/tmp/clone_ref.wav",
    output_path: str = "/tmp/reference_tts_output.wav",
    use_ttnn: bool = False,
):
    """
    Run reference TTS demo.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save output audio
        use_ttnn: If True, use TTNN implementations instead of reference
    """
    print("=" * 80)
    print(f"Reference TTS Demo ({'TTNN' if use_ttnn else 'PyTorch Reference'})")
    print("=" * 80)

    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}")
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

    results = {}

    # =========================================================================
    # Step 1: Load Audio
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Load Audio")
    print("=" * 80)

    audio = load_audio(audio_path)
    print(f"  Audio shape: {audio.shape}")
    print(f"  Duration: {len(audio)/24000:.2f}s")
    results["audio_duration"] = len(audio) / 24000

    # =========================================================================
    # Step 2: Compute Mel Spectrogram
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Compute Mel Spectrogram")
    print("=" * 80)

    start_time = time.time()
    mel = compute_mel_spectrogram_qwen(audio)
    mel_time = time.time() - start_time

    print(f"  Mel shape: {mel.shape}")
    print(f"  Time: {mel_time*1000:.1f}ms")
    results["mel_time"] = mel_time

    # =========================================================================
    # Step 3: Extract Speaker Embedding
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Extract Speaker Embedding")
    print("=" * 80)

    # Load weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["model.safetensors"])
    model_path = Path(model_path)
    main_dict = load_file(model_path / "model.safetensors")
    speaker_weights = extract_speaker_encoder_weights(main_dict)
    print(f"  Loaded {len(speaker_weights)} speaker encoder weights")

    start_time = time.time()

    if use_ttnn:
        # TTNN implementation (TODO)
        print("  [TTNN speaker encoder not yet implemented, using reference]")
        config = SpeakerEncoderConfig()
        speaker_embedding = speaker_encoder_forward(mel, speaker_weights, config)
    else:
        config = SpeakerEncoderConfig()
        speaker_embedding = speaker_encoder_forward(mel, speaker_weights, config)

    speaker_time = time.time() - start_time

    print(f"  Speaker embedding shape: {speaker_embedding.shape}")
    print(f"  Speaker embedding norm: {speaker_embedding.norm():.4f}")
    print(f"  Time: {speaker_time*1000:.1f}ms")
    results["speaker_time"] = speaker_time

    # Compare with official if available
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if voice_clone_path.exists():
        data = torch.load(voice_clone_path, weights_only=False)
        official_embedding = data["ref_spk_embedding"]
        pcc = compute_pcc(speaker_embedding.squeeze(), official_embedding)
        print(f"  PCC vs official: {pcc:.6f}")
        results["speaker_pcc"] = pcc

    # =========================================================================
    # Step 4: Encode Audio to RVQ Codes
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Encode Audio to RVQ Codes")
    print("=" * 80)

    start_time = time.time()

    if use_ttnn:
        # TTNN implementation (TODO)
        print("  [TTNN speech encoder not yet implemented, using reference]")
        codes = speech_tokenizer_encoder_forward_mimi(audio.unsqueeze(0))
    else:
        codes = speech_tokenizer_encoder_forward_mimi(audio.unsqueeze(0))

    encoder_time = time.time() - start_time

    print(f"  Codes shape: {codes.shape}")
    print(f"  Codes range: [{codes.min()}, {codes.max()}]")
    print(f"  Time: {encoder_time*1000:.1f}ms")
    results["encoder_time"] = encoder_time

    # Compare with official if available
    if voice_clone_path.exists():
        official_codes = data["ref_code"].T.unsqueeze(0)
        min_len = min(codes.shape[-1], official_codes.shape[-1])
        match_ratio = (codes[..., :min_len] == official_codes[..., :min_len]).float().mean().item()
        print(f"  Match ratio vs official: {match_ratio:.6f}")
        results["encoder_match"] = match_ratio

    # =========================================================================
    # Step 5: Decode RVQ Codes to Audio
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Decode RVQ Codes to Audio")
    print("=" * 80)

    # Load decoder weights
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)
    decoder_weights = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}
    print(f"  Loaded {len(decoder_weights)} decoder weights")

    start_time = time.time()

    if use_ttnn:
        # TTNN implementation
        print("  [Using TTNN decoder]")
        try:
            import ttnn
            from models.demos.qwen3_tts.tt.speech_tokenizer import SpeechTokenizerConfig, TtSpeechTokenizerDecoder

            # Get device
            device = ttnn.open_device(device_id=0)

            # Create TTNN decoder
            # use_reference=True uses fixed reference (working)
            # use_reference=False uses actual TTNN (broken - pre-transformer collapses values)
            tt_config = SpeechTokenizerConfig()
            tt_decoder = TtSpeechTokenizerDecoder(device, decoder_weights, tt_config, use_reference=True)
            print("    Note: Using reference decoder inside TTNN wrapper (TTNN pre-transformer has issues)")

            # Run decoder
            output_audio = tt_decoder(codes)

            ttnn.close_device(device)

        except Exception as e:
            print(f"  TTNN decoder failed: {e}")
            print("  Falling back to reference decoder")
            decoder_config = SpeechTokenizerDecoderConfig()
            output_audio = speech_tokenizer_decoder_forward(codes, decoder_weights, decoder_config)
    else:
        decoder_config = SpeechTokenizerDecoderConfig()
        output_audio = speech_tokenizer_decoder_forward(codes, decoder_weights, decoder_config)

    decoder_time = time.time() - start_time

    print(f"  Output audio shape: {output_audio.shape}")
    print(f"  Output duration: {output_audio.shape[-1]/24000:.2f}s")
    print(f"  Time: {decoder_time*1000:.1f}ms")
    results["decoder_time"] = decoder_time

    # =========================================================================
    # Step 6: Quality Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Quality Analysis")
    print("=" * 80)

    min_len = min(len(audio), output_audio.shape[-1])
    orig = audio[:min_len]
    recon = output_audio.squeeze()[:min_len]

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
    results["energy_pcc"] = energy_pcc

    # =========================================================================
    # Step 7: Save Output
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 7: Save Output")
    print("=" * 80)

    audio_np = output_audio.squeeze().detach().cpu().float().numpy()
    sf.write(output_path, audio_np, 24000)
    print(f"  Saved to: {output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Mode: {'TTNN' if use_ttnn else 'PyTorch Reference'}")
    print(f"  Input: {audio_path} ({results['audio_duration']:.2f}s)")
    print(f"  Output: {output_path}")
    print()
    print(f"  Timings:")
    print(f"    Mel Spectrogram:  {results['mel_time']*1000:7.1f}ms")
    print(f"    Speaker Encoder:  {results['speaker_time']*1000:7.1f}ms")
    print(f"    Speech Encoder:   {results['encoder_time']*1000:7.1f}ms")
    print(f"    Speech Decoder:   {results['decoder_time']*1000:7.1f}ms")
    total_time = results["mel_time"] + results["speaker_time"] + results["encoder_time"] + results["decoder_time"]
    print(f"    Total:            {total_time*1000:7.1f}ms")
    print()
    if "speaker_pcc" in results:
        print(f"  Speaker Encoder PCC: {results['speaker_pcc']:.4f}")
    if "encoder_match" in results:
        print(f"  Encoder Match Ratio: {results['encoder_match']:.4f}")
    print(f"  Energy PCC:          {results['energy_pcc']:.4f}")
    print()
    print("  Listen to the output to verify quality!")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Reference TTS Demo")
    parser.add_argument("--audio", type=str, default="/tmp/clone_ref.wav", help="Path to input audio file")
    parser.add_argument("--output", type=str, default="/tmp/reference_tts_output.wav", help="Path to save output audio")
    parser.add_argument("--ttnn", action="store_true", help="Use TTNN implementations instead of reference")

    args = parser.parse_args()

    run_reference_tts_demo(
        audio_path=args.audio,
        output_path=args.output,
        use_ttnn=args.ttnn,
    )


if __name__ == "__main__":
    main()
