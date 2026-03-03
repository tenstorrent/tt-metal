# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Extract intermediate tensors from official qwen_tts for PCC comparison.

Run in /tmp/qwen_tts_env:
    source /tmp/qwen_tts_env/bin/activate
    python models/demos/qwen3_tts/demo/extract_intermediates.py
"""

from pathlib import Path

import torch


def hook_model_internals():
    """Hook into model internals to capture intermediate tensors."""
    from qwen_tts import Qwen3TTSModel

    print("=" * 80)
    print("Extracting Intermediate Tensors from Official qwen_tts")
    print("=" * 80)

    # Storage for captured tensors
    captured = {}

    # Load model
    print("\nLoading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    print("Model loaded!")

    # Inspect model structure
    print("\n" + "=" * 80)
    print("Model Structure")
    print("=" * 80)

    # The model has a 'model' attribute containing the actual TTS model
    tts_model = model.model
    print(f"TTS model type: {type(tts_model)}")

    # Check for Talker
    if hasattr(tts_model, "talker"):
        talker = tts_model.talker
        print(f"\nTalker type: {type(talker)}")
        print(f"Talker attributes: {[a for a in dir(talker) if not a.startswith('_')][:20]}")

        if hasattr(talker, "model"):
            talker_model = talker.model
            print(f"Talker.model type: {type(talker_model)}")

            # Check embeddings
            if hasattr(talker_model, "codec_embedding"):
                print(f"  codec_embedding: {talker_model.codec_embedding.weight.shape}")
            if hasattr(talker_model, "text_embedding"):
                print(f"  text_embedding: {talker_model.text_embedding.weight.shape}")
            if hasattr(talker_model, "layers"):
                print(f"  num_layers: {len(talker_model.layers)}")

        # Check code predictor
        if hasattr(talker, "code_predictor"):
            code_pred = talker.code_predictor
            print(f"\nCode Predictor type: {type(code_pred)}")
            if hasattr(code_pred, "model"):
                print(f"  Code Predictor model layers: {len(code_pred.model.layers)}")
            if hasattr(code_pred, "lm_head"):
                print(f"  LM heads: {len(code_pred.lm_head)}")

    # Check speech tokenizer
    if hasattr(tts_model, "speech_tokenizer"):
        st = tts_model.speech_tokenizer
        print(f"\nSpeech Tokenizer type: {type(st)}")
        print(f"Speech Tokenizer attributes: {[a for a in dir(st) if not a.startswith('_')][:20]}")

    # Set up hooks to capture tensors
    print("\n" + "=" * 80)
    print("Setting up hooks...")
    print("=" * 80)

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captured[f"{name}_output"] = output.detach().clone()
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    captured[f"{name}_output"] = output[0].detach().clone()
            if isinstance(input, tuple) and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    captured[f"{name}_input"] = input[0].detach().clone()

        return hook

    handles = []

    # Hook Talker layers
    if hasattr(tts_model, "talker") and hasattr(tts_model.talker, "model"):
        talker_model = tts_model.talker.model

        # Hook first and last layer
        if hasattr(talker_model, "layers") and len(talker_model.layers) > 0:
            handles.append(talker_model.layers[0].register_forward_hook(make_hook("talker_layer_0")))
            handles.append(talker_model.layers[-1].register_forward_hook(make_hook("talker_layer_last")))
            print(f"  Hooked talker layers 0 and {len(talker_model.layers)-1}")

        # Hook norm
        if hasattr(talker_model, "norm"):
            handles.append(talker_model.norm.register_forward_hook(make_hook("talker_norm")))
            print("  Hooked talker norm")

        # Hook embeddings
        if hasattr(talker_model, "codec_embedding"):
            handles.append(talker_model.codec_embedding.register_forward_hook(make_hook("codec_embedding")))
            print("  Hooked codec_embedding")
        if hasattr(talker_model, "text_embedding"):
            handles.append(talker_model.text_embedding.register_forward_hook(make_hook("text_embedding")))
            print("  Hooked text_embedding")

    # Hook Code Predictor
    if hasattr(tts_model, "talker") and hasattr(tts_model.talker, "code_predictor"):
        code_pred = tts_model.talker.code_predictor
        if hasattr(code_pred, "model") and hasattr(code_pred.model, "layers"):
            handles.append(code_pred.model.layers[0].register_forward_hook(make_hook("code_pred_layer_0")))
            handles.append(code_pred.model.layers[-1].register_forward_hook(make_hook("code_pred_layer_last")))
            print(f"  Hooked code_predictor layers")

    # Generate audio
    print("\n" + "=" * 80)
    print("Generating audio...")
    print("=" * 80)

    ref_audio = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    text = "Hello, this is a test of the text to speech system."

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    print(f"Generated audio: {wavs[0].shape}, sr={sr}")

    # Remove hooks
    for h in handles:
        h.remove()

    # Print captured tensors
    print("\n" + "=" * 80)
    print("Captured Tensors")
    print("=" * 80)

    for name, tensor in captured.items():
        print(f"  {name}: {tensor.shape}, dtype={tensor.dtype}")

    # Save tensors
    output_dir = Path("/tmp/qwen_tts_tensors")
    output_dir.mkdir(exist_ok=True)

    torch.save(captured, output_dir / "intermediates.pt")
    print(f"\nSaved {len(captured)} tensors to {output_dir / 'intermediates.pt'}")

    return captured


def main():
    hook_model_internals()


if __name__ == "__main__":
    main()
