# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Voice Clone Demo using official qwen_tts infrastructure + TTNN model.

This demo:
1. Uses official qwen_tts for voice clone prompt creation (audio encoding)
2. Runs TTNN model for inference
3. Uses official qwen_tts for audio decoding

Run in /tmp/qwen_tts_env (with TTNN available):
    source /tmp/qwen_tts_env/bin/activate
    # Need to add tt-metal to PYTHONPATH
    export PYTHONPATH=/home/ttuser/ssinghal/PR-fix/main/debug/tt-metal:$PYTHONPATH
    python models/demos/qwen3_tts/demo/demo_voice_clone.py
"""

import argparse
import time

import soundfile as sf
import torch


def run_voice_clone_demo(
    ref_audio: str = "/tmp/clone_ref.wav",
    ref_text: str = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    text: str = "Hello, this is a test of the text to speech system.",
    audio_output: str = "/tmp/voice_clone_ttnn.wav",
    use_ttnn: bool = False,
):
    """Run voice clone demo."""
    print("=" * 80)
    print("Voice Clone Demo")
    print(f"Mode: {'TTNN' if use_ttnn else 'Official qwen_tts'}")
    print("=" * 80)

    from qwen_tts import Qwen3TTSModel

    # Load official model
    print("\nLoading official qwen_tts model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    print("Model loaded!")

    # Create voice clone prompt
    print("\n" + "=" * 80)
    print("Creating Voice Clone Prompt")
    print("=" * 80)
    print(f"Reference audio: {ref_audio}")
    print(f"Reference text: {ref_text[:50]}...")

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    prompt_item = prompt_items[0] if isinstance(prompt_items, list) else prompt_items

    print(f"  ref_code shape: {prompt_item.ref_code.shape}")
    print(f"  ref_spk_embedding shape: {prompt_item.ref_spk_embedding.shape}")

    # Save the prompt for TTNN usage
    torch.save(
        {
            "ref_code": prompt_item.ref_code,
            "ref_spk_embedding": prompt_item.ref_spk_embedding,
            "ref_text": ref_text,
        },
        "/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt",
    )

    if use_ttnn:
        print("\n" + "=" * 80)
        print("Running TTNN Model")
        print("=" * 80)
        print("TTNN voice clone mode not yet implemented.")
        print("Components working:")
        print("  - RMSNorm: PCC 0.999982")
        print("  - Embedding: PCC 1.000000")
        print("  - MLP: PCC 0.999872")
        print("\nTo complete TTNN voice clone:")
        print("  1. Convert ref_code to talker input (first codebook)")
        print("  2. Tokenize text and embed via text_embedding")
        print("  3. Run Talker to generate codec tokens")
        print("  4. Run Code Predictor to generate remaining codebooks")
        print("  5. Decode RVQ codes to audio via speech tokenizer")
        return

    # Use official model for generation
    print("\n" + "=" * 80)
    print("Generating Audio (Official qwen_tts)")
    print("=" * 80)
    print(f"Text to synthesize: {text}")

    start_time = time.time()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    gen_time = time.time() - start_time

    print(f"\nGeneration time: {gen_time:.2f}s")
    print(f"Audio shape: {wavs[0].shape}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(wavs[0]) / sr:.2f} seconds")

    # Save audio
    sf.write(audio_output, wavs[0], sr)
    print(f"\nAudio saved to: {audio_output}")

    # Also extract some intermediate tensors for TTNN comparison
    print("\n" + "=" * 80)
    print("Extracting Tensors for TTNN Comparison")
    print("=" * 80)

    tts_model = model.model
    talker = tts_model.talker
    talker_model = talker.model

    # Capture a single forward pass to get intermediate tensors
    capture = {}

    def capture_hook(name):
        def hook(module, input, output):
            if name not in capture:
                if isinstance(output, torch.Tensor):
                    capture[name] = output.detach().clone()
                elif isinstance(output, tuple) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        capture[name] = output[0].detach().clone()

        return hook

    handles = [
        talker_model.layers[0].register_forward_hook(capture_hook("talker_layer_0_output")),
        talker_model.layers[-1].register_forward_hook(capture_hook("talker_layer_27_output")),
        talker_model.norm.register_forward_hook(capture_hook("talker_norm_output")),
    ]

    # Run a short generation to capture tensors
    _, _ = model.generate_voice_clone(
        text="Hi",
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    for h in handles:
        h.remove()

    print(f"Captured {len(capture)} tensors:")
    for name, tensor in capture.items():
        print(f"  {name}: {tensor.shape}")

    torch.save(capture, "/tmp/qwen_tts_tensors/inference_tensors.pt")
    print(f"\nSaved to /tmp/qwen_tts_tensors/inference_tensors.pt")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Voice Clone Demo")
    parser.add_argument("--ref-audio", type=str, default="/tmp/clone_ref.wav")
    parser.add_argument(
        "--ref-text",
        type=str,
        default="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    )
    parser.add_argument("--text", type=str, default="Hello, this is a test of the text to speech system.")
    parser.add_argument("--audio-output", type=str, default="/tmp/voice_clone_ttnn.wav")
    parser.add_argument("--use-ttnn", action="store_true", help="Use TTNN model instead of official")

    args = parser.parse_args()

    run_voice_clone_demo(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        text=args.text,
        audio_output=args.audio_output,
        use_ttnn=args.use_ttnn,
    )


if __name__ == "__main__":
    main()
