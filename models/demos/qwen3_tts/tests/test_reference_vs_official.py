# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Compare our reference implementation against the official qwen_tts package.

This test identifies where our implementation diverges from official.
"""

from pathlib import Path

import numpy as np
import torch


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    cov = (a_centered * b_centered).sum()
    std_a = torch.sqrt((a_centered**2).sum())
    std_b = torch.sqrt((b_centered**2).sum())
    if std_a == 0 or std_b == 0:
        return 0.0
    return (cov / (std_a * std_b)).item()


def test_official_decoder_works():
    """First verify the official qwen_tts decoder produces correct audio."""
    print("=" * 80)
    print("Test 1: Verify Official Decoder Works")
    print("=" * 80)

    try:
        import soundfile as sf
        from qwen_tts import Qwen3TTS

        # Load model
        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        model = Qwen3TTS.from_pretrained(model_id)
        model.eval()

        # Get test codes from voice clone prompt
        voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
        if voice_clone_path.exists():
            data = torch.load(voice_clone_path)
            ref_code = data["ref_code"]  # [101, 16]
            print(f"  Loaded ref_code shape: {ref_code.shape}")

            # Decode with official decoder
            with torch.no_grad():
                # The official speech tokenizer expects [batch, num_codebooks, seq_len]
                codes = ref_code.T.unsqueeze(0)  # [1, 16, 101]
                print(f"  Input codes shape: {codes.shape}")

                # Get the official decoder
                decoder = model.speech_tokenizer.decoder
                print(f"  Official decoder type: {type(decoder)}")

                # Decode
                audio = decoder.decode(codes)
                print(f"  Official output shape: {audio.shape}")
                print(f"  Official output range: [{audio.min():.4f}, {audio.max():.4f}]")

                # Save audio
                audio_np = audio.squeeze().cpu().numpy()
                sf.write("/tmp/official_decoder_test.wav", audio_np, 24000)
                print(f"  Saved to /tmp/official_decoder_test.wav")
                print("  LISTEN to verify it's correct audio!")

                return audio, codes
        else:
            print("  No test codes found, generating random codes")
            codes = torch.randint(0, 2048, (1, 16, 50))
            with torch.no_grad():
                decoder = model.speech_tokenizer.decoder
                audio = decoder.decode(codes)
                print(f"  Official output shape: {audio.shape}")
                return audio, codes

    except ImportError as e:
        print(f"  qwen_tts not available: {e}")
        return None, None
    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def test_compare_codebook_lookup():
    """Compare codebook lookup between official and reference."""
    print("\n" + "=" * 80)
    print("Test 2: Compare Codebook Lookup")
    print("=" * 80)

    try:
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTS
        from safetensors.torch import load_file

        from models.demos.qwen3_tts.reference.functional import codebook_lookup_rvq

        # Load model
        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        model = Qwen3TTS.from_pretrained(model_id)
        model.eval()

        # Load weights
        model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
        model_path = Path(model_path)
        speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
        raw_dict = load_file(speech_tokenizer_path)

        # Strip decoder prefix
        state_dict = {}
        for k, v in raw_dict.items():
            if k.startswith("decoder."):
                state_dict[k[8:]] = v
            else:
                state_dict[k] = v

        # Get test codes
        voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
        if voice_clone_path.exists():
            data = torch.load(voice_clone_path)
            ref_code = data["ref_code"]
            codes = ref_code.T.unsqueeze(0)  # [1, 16, 101]
        else:
            codes = torch.randint(0, 2048, (1, 16, 50))

        print(f"  Input codes shape: {codes.shape}")

        # Official codebook lookup
        with torch.no_grad():
            decoder = model.speech_tokenizer.decoder

            # Get official quantizer output
            official_quantizer = decoder.quantizer
            print(f"  Official quantizer type: {type(official_quantizer)}")

            # Call official decode_code method
            official_emb = official_quantizer.decode_code(codes)
            print(f"  Official embeddings shape: {official_emb.shape}")
            print(f"  Official embeddings range: [{official_emb.min():.4f}, {official_emb.max():.4f}]")

        # Reference codebook lookup
        rvq_first_codebook = state_dict.get("quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
        rvq_first_output_proj = state_dict.get("quantizer.rvq_first.output_proj.weight")
        rvq_rest_codebooks = []
        for i in range(15):
            key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
            if key in state_dict:
                rvq_rest_codebooks.append(state_dict[key])
        rvq_rest_output_proj = state_dict.get("quantizer.rvq_rest.output_proj.weight")

        ref_emb = codebook_lookup_rvq(
            codes,
            rvq_first_codebook,
            rvq_rest_codebooks,
            rvq_first_output_proj,
            rvq_rest_output_proj,
        )
        print(f"  Reference embeddings shape: {ref_emb.shape}")
        print(f"  Reference embeddings range: [{ref_emb.min():.4f}, {ref_emb.max():.4f}]")

        # Compare - need to handle different shapes
        # Official might be [batch, channels, seq_len] and ours is [batch, seq_len, channels]
        if official_emb.shape != ref_emb.shape:
            print(f"  Shape mismatch: official {official_emb.shape} vs ref {ref_emb.shape}")
            # Try transposing reference to match official
            if official_emb.dim() == 3 and ref_emb.dim() == 3:
                ref_emb_transposed = ref_emb.transpose(1, 2)  # [batch, channels, seq_len]
                if official_emb.shape == ref_emb_transposed.shape:
                    pcc = compute_pcc(official_emb, ref_emb_transposed)
                    print(f"  PCC (after transpose): {pcc:.6f}")
                    return pcc

        pcc = compute_pcc(official_emb, ref_emb)
        print(f"  PCC: {pcc:.6f}")
        return pcc

    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_trace_official_decoder():
    """Trace through official decoder to understand the architecture."""
    print("\n" + "=" * 80)
    print("Test 3: Trace Official Decoder Architecture")
    print("=" * 80)

    try:
        from qwen_tts import Qwen3TTS

        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        model = Qwen3TTS.from_pretrained(model_id)
        model.eval()

        decoder = model.speech_tokenizer.decoder

        # Print decoder structure
        print("\nOfficial Decoder Structure:")
        for name, module in decoder.named_children():
            print(f"  {name}: {type(module).__name__}")
            if hasattr(module, "named_children"):
                for subname, submodule in module.named_children():
                    print(f"    {subname}: {type(submodule).__name__}")

        # Check for decode method
        print(f"\n  Has decode method: {hasattr(decoder, 'decode')}")
        print(f"  Has forward method: {hasattr(decoder, 'forward')}")

        # Get test codes
        voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
        if voice_clone_path.exists():
            data = torch.load(voice_clone_path)
            ref_code = data["ref_code"]
            codes = ref_code.T.unsqueeze(0)
        else:
            codes = torch.randint(0, 2048, (1, 16, 50))

        # Trace through decoder step by step
        print("\n  Tracing through decoder...")
        with torch.no_grad():
            # Step 1: Quantizer decode
            print(f"\n  Step 1: Quantizer decode_code")
            z = decoder.quantizer.decode_code(codes)
            print(f"    Output shape: {z.shape}")
            print(f"    Output range: [{z.min():.4f}, {z.max():.4f}]")

            # Step 2: Pre-transformer (if exists)
            if hasattr(decoder, "pre_transformer") and decoder.pre_transformer is not None:
                print(f"\n  Step 2: Pre-transformer")
                print(f"    Pre-transformer type: {type(decoder.pre_transformer)}")
                # Check input shape expected
                z_pre = decoder.pre_transformer(z)
                print(f"    Output shape: {z_pre.shape}")
                print(f"    Output range: [{z_pre.min():.4f}, {z_pre.max():.4f}]")
                z = z_pre
            else:
                print(f"\n  Step 2: No pre_transformer found")

            # Step 3: Pre-conv (if exists)
            if hasattr(decoder, "pre_conv") and decoder.pre_conv is not None:
                print(f"\n  Step 3: Pre-conv")
                z_conv = decoder.pre_conv(z)
                print(f"    Output shape: {z_conv.shape}")
                z = z_conv
            else:
                print(f"\n  Step 3: No pre_conv found")

            # Step 4: Upsample (if exists)
            if hasattr(decoder, "upsample") and decoder.upsample is not None:
                print(f"\n  Step 4: Upsample")
                z_up = decoder.upsample(z)
                print(f"    Output shape: {z_up.shape}")
                z = z_up
            else:
                print(f"\n  Step 4: No upsample found")

            # Step 5: Decoder (conv decoder)
            if hasattr(decoder, "decoder") and decoder.decoder is not None:
                print(f"\n  Step 5: Conv decoder")
                audio = decoder.decoder(z)
                print(f"    Output shape: {audio.shape}")
                print(f"    Output range: [{audio.min():.4f}, {audio.max():.4f}]")
            else:
                print(f"\n  Step 5: No decoder found")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()


def test_reference_decoder_with_official_intermediates():
    """Test our reference decoder using official intermediate outputs."""
    print("\n" + "=" * 80)
    print("Test 4: Reference Decoder with Official Intermediates")
    print("=" * 80)

    try:
        import soundfile as sf
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTS
        from safetensors.torch import load_file

        from models.demos.qwen3_tts.reference.functional import (
            SpeechTokenizerDecoderConfig,
            speech_tokenizer_decoder_forward,
        )

        # Load model
        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        model = Qwen3TTS.from_pretrained(model_id)
        model.eval()

        # Load weights
        model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
        model_path = Path(model_path)
        speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
        raw_dict = load_file(speech_tokenizer_path)

        # Strip decoder prefix
        state_dict = {}
        for k, v in raw_dict.items():
            if k.startswith("decoder."):
                state_dict[k[8:]] = v

        config = SpeechTokenizerDecoderConfig()

        # Get test codes
        voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
        if voice_clone_path.exists():
            data = torch.load(voice_clone_path)
            ref_code = data["ref_code"]
            codes = ref_code.T.unsqueeze(0)
        else:
            codes = torch.randint(0, 2048, (1, 16, 50))

        print(f"  Input codes shape: {codes.shape}")

        # Get official output
        with torch.no_grad():
            decoder = model.speech_tokenizer.decoder
            official_audio = decoder.decode(codes)
            print(f"  Official audio shape: {official_audio.shape}")
            print(f"  Official audio range: [{official_audio.min():.4f}, {official_audio.max():.4f}]")

        # Get reference output
        with torch.no_grad():
            ref_audio = speech_tokenizer_decoder_forward(codes, state_dict, config)
            print(f"  Reference audio shape: {ref_audio.shape}")
            print(f"  Reference audio range: [{ref_audio.min():.4f}, {ref_audio.max():.4f}]")

        # Compare
        pcc = compute_pcc(official_audio, ref_audio)
        print(f"\n  PCC (full decoder): {pcc:.6f}")

        # Save both for listening
        official_np = official_audio.squeeze().cpu().numpy()
        ref_np = ref_audio.squeeze().cpu().float().numpy()

        sf.write("/tmp/test_official.wav", official_np, 24000)
        sf.write("/tmp/test_reference.wav", ref_np, 24000)
        print(f"\n  Saved /tmp/test_official.wav and /tmp/test_reference.wav")
        print("  LISTEN to both to compare!")

        if pcc < 0.99:
            print(f"\n  *** PCC IS LOW - REFERENCE HAS BUG ***")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()


def test_mel_spectrogram_vs_official():
    """Test mel spectrogram computation against official qwen_tts."""
    print("\n" + "=" * 80)
    print("Test 5: Mel Spectrogram vs Official")
    print("=" * 80)

    import soundfile as sf
    from scipy import signal

    def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
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

    # Load audio
    audio_path = "/tmp/clone_ref.wav"
    if not Path(audio_path).exists():
        print(f"  SKIP: Audio file not found at {audio_path}")
        return

    audio = load_audio(audio_path)
    print(f"  Input audio shape: {audio.shape}")

    # Reference implementation
    from models.demos.qwen3_tts.reference.functional import compute_mel_spectrogram_qwen

    ref_mel = compute_mel_spectrogram_qwen(audio)
    print(f"  Reference mel shape: {ref_mel.shape}")
    print(f"  Reference mel range: [{ref_mel.min():.4f}, {ref_mel.max():.4f}]")

    # Official implementation
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram as official_mel_fn

        official_mel = official_mel_fn(
            audio.unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        )
        print(f"  Official mel shape: {official_mel.shape}")
        print(f"  Official mel range: [{official_mel.min():.4f}, {official_mel.max():.4f}]")

        # Compare
        pcc = compute_pcc(ref_mel, official_mel)
        print(f"  PCC: {pcc:.6f}")

        if pcc > 0.99:
            print("  PASS: Mel spectrogram matches official (PCC > 0.99)")
        else:
            print(f"  FAIL: Mel spectrogram PCC {pcc:.4f} < 0.99")
    except ImportError:
        print("  SKIP: qwen_tts package not installed")


def test_speaker_encoder_vs_official():
    """Test speaker encoder against official qwen_tts."""
    print("\n" + "=" * 80)
    print("Test 6: Speaker Encoder vs Official")
    print("=" * 80)

    import soundfile as sf
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from scipy import signal

    def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
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

    # Load saved official outputs
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print(f"  SKIP: Official outputs not found at {voice_clone_path}")
        return

    data = torch.load(voice_clone_path)
    official_embedding = data["ref_spk_embedding"]
    print(f"  Official embedding shape: {official_embedding.shape}")
    print(f"  Official embedding range: [{official_embedding.min():.4f}, {official_embedding.max():.4f}]")

    # Load audio and compute mel spectrogram
    audio_path = "/tmp/clone_ref.wav"
    if not Path(audio_path).exists():
        print(f"  SKIP: Audio file not found at {audio_path}")
        return

    audio = load_audio(audio_path)

    # Compute mel spectrogram using our function
    from models.demos.qwen3_tts.reference.functional import (
        SpeakerEncoderConfig,
        compute_mel_spectrogram_qwen,
        extract_speaker_encoder_weights,
        speaker_encoder_forward,
    )

    mel = compute_mel_spectrogram_qwen(audio)
    print(f"  Mel spectrogram shape: {mel.shape}")

    # Load speaker encoder weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["model.safetensors"])
    model_path = Path(model_path)

    main_dict = load_file(model_path / "model.safetensors")
    speaker_weights = extract_speaker_encoder_weights(main_dict)
    print(f"  Loaded {len(speaker_weights)} speaker encoder weights")

    # Run reference speaker encoder
    config = SpeakerEncoderConfig()
    ref_embedding = speaker_encoder_forward(mel, speaker_weights, config)
    print(f"  Reference embedding shape: {ref_embedding.shape}")
    print(f"  Reference embedding range: [{ref_embedding.min():.4f}, {ref_embedding.max():.4f}]")

    # Compare
    pcc = compute_pcc(ref_embedding.squeeze(), official_embedding)
    print(f"  PCC: {pcc:.6f}")

    if pcc > 0.99:
        print("  PASS: Speaker encoder matches official (PCC > 0.99)")
    elif pcc > 0.9:
        print(f"  PARTIAL: Speaker encoder PCC {pcc:.4f} > 0.9")
    else:
        print(f"  FAIL: Speaker encoder PCC {pcc:.4f} < 0.9")


def test_speech_tokenizer_encoder_vs_official():
    """Test speech tokenizer encoder against official qwen_tts."""
    print("\n" + "=" * 80)
    print("Test 7: Speech Tokenizer Encoder vs Official")
    print("=" * 80)

    import soundfile as sf
    from scipy import signal

    def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
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

    # Load saved official outputs
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print(f"  SKIP: Official outputs not found at {voice_clone_path}")
        return

    data = torch.load(voice_clone_path)
    official_codes = data["ref_code"]
    print(f"  Official codes shape: {official_codes.shape}")
    print(f"  Official codes range: [{official_codes.min()}, {official_codes.max()}]")

    # Load audio
    audio_path = "/tmp/clone_ref.wav"
    if not Path(audio_path).exists():
        print(f"  SKIP: Audio file not found at {audio_path}")
        return

    audio = load_audio(audio_path)
    print(f"  Input audio shape: {audio.shape}")

    # Run reference encoder (using MimiModel)
    from models.demos.qwen3_tts.reference.functional import speech_tokenizer_encoder_forward

    ref_codes = speech_tokenizer_encoder_forward(audio.unsqueeze(0), use_mimi=True)
    print(f"  Reference codes shape: {ref_codes.shape}")
    print(f"  Reference codes range: [{ref_codes.min()}, {ref_codes.max()}]")

    # Reshape official codes to match reference format
    # Official is [seq_len, num_quantizers], reference is [batch, num_quantizers, seq_len]
    if official_codes.dim() == 2:
        if official_codes.shape[0] > official_codes.shape[1]:
            official_codes_reshaped = official_codes.T.unsqueeze(0)
        else:
            official_codes_reshaped = official_codes.unsqueeze(0)
    else:
        official_codes_reshaped = official_codes

    print(f"  Official codes reshaped: {official_codes_reshaped.shape}")

    # Compare sequence lengths
    ref_seq_len = ref_codes.shape[-1]
    official_seq_len = official_codes_reshaped.shape[-1]
    print(f"  Reference seq_len: {ref_seq_len}, Official seq_len: {official_seq_len}")

    # Check exact match for overlapping portion
    min_len = min(ref_seq_len, official_seq_len)
    ref_trimmed = ref_codes[..., :min_len]
    official_trimmed = official_codes_reshaped[..., :min_len]

    match_count = (ref_trimmed == official_trimmed).sum().item()
    total_count = ref_trimmed.numel()
    match_ratio = match_count / total_count
    print(f"  Exact match ratio: {match_ratio:.4f} ({match_count}/{total_count})")

    if ref_seq_len == official_seq_len and match_ratio > 0.99:
        print("  PASS: Speech encoder codes match official exactly")
    elif match_ratio > 0.9:
        print(f"  PARTIAL: Code match ratio {match_ratio:.4f} > 0.9")
    else:
        print(f"  FAIL: Code match ratio {match_ratio:.4f} < 0.9")


def main():
    print("Testing Reference vs Official Implementation")
    print("=" * 80)

    # Test 1: Verify official works
    official_audio, codes = test_official_decoder_works()

    # Test 2: Compare codebook lookup
    test_compare_codebook_lookup()

    # Test 3: Trace official decoder
    test_trace_official_decoder()

    # Test 4: Compare full decoder
    test_reference_decoder_with_official_intermediates()

    # Test 5: Mel spectrogram
    test_mel_spectrogram_vs_official()

    # Test 6: Speaker encoder
    test_speaker_encoder_vs_official()

    # Test 7: Speech tokenizer encoder
    test_speech_tokenizer_encoder_vs_official()


if __name__ == "__main__":
    main()
