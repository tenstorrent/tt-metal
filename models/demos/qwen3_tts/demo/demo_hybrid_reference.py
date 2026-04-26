# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Hybrid Reference TTS Demo

This demo:
1. Uses official qwen_tts for input preparation and generation
2. Captures intermediate tensors at every step
3. Runs our reference implementations in parallel
4. Compares PCC between official and reference at each step
5. Uses our reference decoder to generate audio

This allows us to verify reference implementations match official,
and then swap in TTNN implementations one by one.

Usage:
    # In qwen_tts environment with tt-metal PYTHONPATH:
    source /tmp/qwen_tts_env/bin/activate
    export PYTHONPATH=/home/ttuser/ssinghal/PR-fix/main/debug/tt-metal:$PYTHONPATH
    python models/demos/qwen3_tts/demo/demo_hybrid_reference.py

Or in tt-metal env (if qwen_tts installed):
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/qwen3_tts/demo/demo_hybrid_reference.py
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


@dataclass
class CapturedTensors:
    """Container for captured intermediate tensors."""

    # Voice clone inputs
    ref_audio: Optional[torch.Tensor] = None
    ref_mel: Optional[torch.Tensor] = None
    ref_spk_embedding: Optional[torch.Tensor] = None
    ref_codes: Optional[torch.Tensor] = None

    # Talker inputs/outputs
    talker_input_embeds: Optional[torch.Tensor] = None
    talker_attention_mask: Optional[torch.Tensor] = None
    talker_hidden_states: Optional[torch.Tensor] = None
    talker_codes: Optional[torch.Tensor] = None

    # Code predictor inputs/outputs
    code_predictor_input: Optional[torch.Tensor] = None
    code_predictor_hidden_states: Optional[torch.Tensor] = None
    all_codes: Optional[torch.Tensor] = None

    # Decoder output
    audio_output: Optional[torch.Tensor] = None


def run_hybrid_demo(
    ref_audio_path: str = "/tmp/clone_ref.wav",
    ref_text: str = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    text: str = "Hello, this is a test of the text to speech system.",
    output_path: str = "/tmp/hybrid_reference_output.wav",
    max_new_tokens: int = 256,
):
    """Run hybrid reference demo."""
    print("=" * 80)
    print("Hybrid Reference TTS Demo")
    print("=" * 80)

    captured = CapturedTensors()

    # =========================================================================
    # Step 1: Load Official Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Load Official qwen_tts Model")
    print("=" * 80)

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        print("ERROR: qwen_tts not found.")
        print("Install with: pip install qwen_tts")
        return None

    print("Loading model...")
    start_time = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.1f}s")

    # =========================================================================
    # Step 2: Voice Clone Prompt Creation (with capturing)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Create Voice Clone Prompt (Official)")
    print("=" * 80)

    # Load reference audio
    audio, sr = sf.read(ref_audio_path)
    audio = torch.from_numpy(audio.astype(np.float32))
    if audio.dim() == 2:
        audio = audio.mean(dim=1)
    captured.ref_audio = audio
    print(f"  Reference audio: {len(audio)/24000:.2f}s")

    # Create voice clone prompt
    start_time = time.time()
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )
    prompt_item = prompt_items[0] if isinstance(prompt_items, list) else prompt_items
    prompt_time = time.time() - start_time

    captured.ref_codes = prompt_item.ref_code
    captured.ref_spk_embedding = prompt_item.ref_spk_embedding

    print(f"  ref_code shape: {captured.ref_codes.shape}")
    print(f"  ref_spk_embedding shape: {captured.ref_spk_embedding.shape}")
    print(f"  Time: {prompt_time*1000:.1f}ms")

    # =========================================================================
    # Step 3: Compare Voice Clone Components with Reference
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Compare Voice Clone Components (Official vs Reference)")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeakerEncoderConfig,
        compute_mel_spectrogram_qwen,
        extract_speaker_encoder_weights,
        speaker_encoder_forward,
        speech_tokenizer_encoder_forward_mimi,
    )

    # Mel spectrogram
    print("\n  [Mel Spectrogram]")
    ref_mel = compute_mel_spectrogram_qwen(audio)
    captured.ref_mel = ref_mel
    print(f"    Reference mel shape: {ref_mel.shape}")

    # Speaker embedding
    print("\n  [Speaker Encoder]")
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["model.safetensors"])
    model_path = Path(model_path)
    main_dict = load_file(model_path / "model.safetensors")
    speaker_weights = extract_speaker_encoder_weights(main_dict)

    config = SpeakerEncoderConfig()
    ref_embedding = speaker_encoder_forward(ref_mel, speaker_weights, config)

    pcc = compute_pcc(ref_embedding.squeeze(), captured.ref_spk_embedding)
    print(f"    Reference embedding shape: {ref_embedding.shape}")
    print(f"    PCC vs official: {pcc:.6f}")

    # Speech encoder
    print("\n  [Speech Tokenizer Encoder]")
    ref_codes_mine = speech_tokenizer_encoder_forward_mimi(audio.unsqueeze(0))

    # Compare codes
    official_codes = captured.ref_codes.T.unsqueeze(0)  # [1, 16, seq_len]
    min_len = min(ref_codes_mine.shape[-1], official_codes.shape[-1])
    match_ratio = (ref_codes_mine[..., :min_len] == official_codes[..., :min_len]).float().mean().item()
    print(f"    Reference codes shape: {ref_codes_mine.shape}")
    print(f"    Code match ratio: {match_ratio:.6f}")

    # =========================================================================
    # Step 4: Generate with Official Model (capturing intermediates)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Generate Audio (Official Model)")
    print("=" * 80)
    print(f"  Text: {text}")

    # Set up hooks to capture intermediate tensors
    tts_model = model.model
    talker = tts_model.talker

    captured_intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            if name not in captured_intermediates:
                if isinstance(output, torch.Tensor):
                    captured_intermediates[name] = output.detach().clone()
                elif hasattr(output, "last_hidden_state"):
                    captured_intermediates[name] = output.last_hidden_state.detach().clone()
                elif isinstance(output, tuple) and len(output) > 0:
                    captured_intermediates[name] = (
                        output[0].detach().clone() if isinstance(output[0], torch.Tensor) else None
                    )

        return hook

    # Register hooks
    handles = []
    if hasattr(talker, "model") and hasattr(talker.model, "layers"):
        handles.append(talker.model.layers[0].register_forward_hook(make_hook("talker_layer_0")))
        handles.append(talker.model.layers[-1].register_forward_hook(make_hook("talker_layer_27")))
        handles.append(talker.model.norm.register_forward_hook(make_hook("talker_norm")))

    # Generate
    start_time = time.time()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=max_new_tokens,
    )
    gen_time = time.time() - start_time

    # Remove hooks
    for h in handles:
        h.remove()

    official_audio = wavs[0]
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Official audio shape: {official_audio.shape}")
    print(f"  Official audio duration: {len(official_audio)/sr:.2f}s")

    # Save official output
    official_output_path = output_path.replace(".wav", "_official.wav")
    sf.write(official_output_path, official_audio, sr)
    print(f"  Saved official: {official_output_path}")

    # Show captured intermediates
    print(f"\n  Captured {len(captured_intermediates)} intermediate tensors:")
    for name, tensor in captured_intermediates.items():
        if tensor is not None:
            print(f"    {name}: {tensor.shape}")

    # =========================================================================
    # Step 5: Extract Generated Codes and Decode with Reference
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 5: Decode with Reference Implementation")
    print("=" * 80)

    # We need to get the generated codes from the official model
    # Since we can't easily extract them during generation, let's use the
    # speech tokenizer to re-encode the official audio and then decode

    # Actually, let's capture the codes by hooking the speech tokenizer decode
    # For now, let's encode the official audio to get codes, then decode with reference

    print("  Encoding official audio to get codes...")
    official_audio_tensor = torch.from_numpy(official_audio.astype(np.float32))
    generated_codes = speech_tokenizer_encoder_forward_mimi(official_audio_tensor.unsqueeze(0))
    print(f"  Generated codes shape: {generated_codes.shape}")

    # Decode with reference decoder
    print("  Decoding with reference decoder...")
    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        speech_tokenizer_decoder_forward,
    )

    # Load decoder weights
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)
    decoder_weights = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}

    start_time = time.time()
    decoder_config = SpeechTokenizerDecoderConfig()
    reference_audio = speech_tokenizer_decoder_forward(generated_codes, decoder_weights, decoder_config)
    decode_time = time.time() - start_time

    print(f"  Reference audio shape: {reference_audio.shape}")
    print(f"  Reference decode time: {decode_time*1000:.1f}ms")

    # Save reference output
    reference_audio_np = reference_audio.squeeze().detach().cpu().float().numpy()
    sf.write(output_path, reference_audio_np, 24000)
    print(f"  Saved reference: {output_path}")

    # =========================================================================
    # Step 6: Compare Official vs Reference Decoder
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Compare Official vs Reference Decoder Output")
    print("=" * 80)

    # Compare audio quality
    min_len = min(len(official_audio), len(reference_audio_np))

    # Energy envelope correlation
    window_size = 480
    off_env = torch.nn.functional.avg_pool1d(
        torch.from_numpy(np.abs(official_audio[:min_len])).unsqueeze(0).unsqueeze(0).float(),
        kernel_size=window_size,
        stride=window_size // 2,
    ).squeeze()
    ref_env = torch.nn.functional.avg_pool1d(
        torch.from_numpy(np.abs(reference_audio_np[:min_len])).unsqueeze(0).unsqueeze(0).float(),
        kernel_size=window_size,
        stride=window_size // 2,
    ).squeeze()

    energy_pcc = compute_pcc(off_env, ref_env)
    print(f"  Energy envelope PCC: {energy_pcc:.4f}")

    # Direct waveform comparison
    waveform_pcc = compute_pcc(
        torch.from_numpy(official_audio[:min_len]), torch.from_numpy(reference_audio_np[:min_len])
    )
    print(f"  Waveform PCC: {waveform_pcc:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Reference audio: {ref_audio_path}")
    print(f"  Text: {text}")
    print(f"")
    print(f"  Official output: {official_output_path}")
    print(f"  Reference output: {output_path}")
    print(f"")
    print(f"  Voice Clone Components:")
    print(f"    Speaker Encoder PCC: {pcc:.4f}")
    print(f"    Speech Encoder Match: {match_ratio:.4f}")
    print(f"")
    print(f"  Decoder Comparison:")
    print(f"    Energy PCC: {energy_pcc:.4f}")
    print(f"    Waveform PCC: {waveform_pcc:.4f}")
    print(f"")
    print("  Listen to both outputs to verify quality!")
    print("=" * 80)

    return captured, captured_intermediates


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Reference TTS Demo")
    parser.add_argument("--ref-audio", type=str, default="/tmp/clone_ref.wav")
    parser.add_argument(
        "--ref-text",
        type=str,
        default="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    )
    parser.add_argument("--text", type=str, default="Hello, this is a test of the text to speech system.")
    parser.add_argument("--output", type=str, default="/tmp/hybrid_reference_output.wav")
    parser.add_argument("--max-tokens", type=int, default=256)

    args = parser.parse_args()

    run_hybrid_demo(
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        text=args.text,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
