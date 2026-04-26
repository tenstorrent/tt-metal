# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Extract the actual input format from official qwen_tts.

Run in /tmp/qwen_tts_env:
    source /tmp/qwen_tts_env/bin/activate
    python models/demos/qwen3_tts/demo/extract_input_format.py
"""

from pathlib import Path

import torch


def extract_input_format():
    """Extract the actual input format used by official model."""
    from qwen_tts import Qwen3TTSModel

    print("=" * 80)
    print("Extracting Input Format from Official qwen_tts")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    print("Model loaded!")

    tts_model = model.model
    talker = tts_model.talker
    talker_model = talker.model

    # Capture input to the model
    captured = {}

    def capture_hook(name):
        def hook(module, input, output):
            if name not in captured:
                if isinstance(input, tuple) and len(input) > 0:
                    captured[name] = input[0].detach().clone() if isinstance(input[0], torch.Tensor) else input
                if isinstance(output, torch.Tensor):
                    captured[f"{name}_output"] = output.detach().clone()

        return hook

    # Hook embedding layers
    handles = []
    handles.append(talker_model.codec_embedding.register_forward_hook(capture_hook("codec_embedding")))
    handles.append(talker_model.text_embedding.register_forward_hook(capture_hook("text_embedding")))

    # Hook the first layer to see what actually goes in
    handles.append(talker_model.layers[0].register_forward_hook(capture_hook("layer_0")))

    # Generate
    print("\nGenerating (capturing input format)...")
    ref_audio = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    text = "Hello, this is a test of the text to speech system."

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    for h in handles:
        h.remove()

    print(f"Generated audio: {wavs[0].shape}")

    # Analyze captured data
    print("\n" + "=" * 80)
    print("Captured Input Format")
    print("=" * 80)

    for name, tensor in captured.items():
        if isinstance(tensor, torch.Tensor):
            print(f"\n{name}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            if tensor.dtype in [torch.int32, torch.int64, torch.long]:
                print(f"  Min: {tensor.min().item()}, Max: {tensor.max().item()}")
                if tensor.numel() <= 20:
                    print(f"  Values: {tensor.flatten().tolist()}")
                else:
                    print(f"  First 10: {tensor.flatten()[:10].tolist()}")
                    print(f"  Last 10: {tensor.flatten()[-10:].tolist()}")
        else:
            print(f"\n{name}: {type(tensor)}")

    # Save for analysis
    output_dir = Path("/tmp/qwen_tts_tensors")
    torch.save(captured, output_dir / "input_format.pt")
    print(f"\nSaved to {output_dir / 'input_format.pt'}")

    # Try to understand the model's generation flow
    print("\n" + "=" * 80)
    print("Model Generation Analysis")
    print("=" * 80)

    # Check the generate method
    if hasattr(tts_model, "generate"):
        print("Model has generate method")
    if hasattr(talker, "generate"):
        print("Talker has generate method")

    # Check config for special tokens
    if hasattr(model, "config"):
        config = model.config
        print(f"\nModel config attributes: {[a for a in dir(config) if not a.startswith('_')][:20]}")
        if hasattr(config, "bos_token_id"):
            print(f"  bos_token_id: {config.bos_token_id}")
        if hasattr(config, "eos_token_id"):
            print(f"  eos_token_id: {config.eos_token_id}")
        if hasattr(config, "pad_token_id"):
            print(f"  pad_token_id: {config.pad_token_id}")

    return captured


def main():
    extract_input_format()


if __name__ == "__main__":
    main()
