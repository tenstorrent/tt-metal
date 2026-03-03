# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Extract the FULL input sequence from official qwen_tts during prefill.

Run in /tmp/qwen_tts_env:
    source /tmp/qwen_tts_env/bin/activate
    python models/demos/qwen3_tts/demo/extract_full_input.py
"""

from pathlib import Path

import torch


def extract_full_input():
    """Extract the full input sequence."""
    from qwen_tts import Qwen3TTSModel

    print("=" * 80)
    print("Extracting Full Input Sequence from Official qwen_tts")
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

    # Capture the FIRST call to embeddings (prefill)
    prefill_captured = {"captured": False}

    def capture_prefill_codec(module, input, output):
        if "codec_input_ids" not in prefill_captured:
            input_ids = input[0] if isinstance(input, tuple) else input
            if isinstance(input_ids, torch.Tensor):
                # Capture the FIRST codec embedding call
                prefill_captured["codec_input_ids"] = input_ids.detach().clone()
                prefill_captured["codec_embeddings"] = output.detach().clone()
                print(f"  Captured codec: {input_ids.shape}, values={input_ids.flatten()[:10].tolist()}")

    def capture_prefill_text(module, input, output):
        if "text_input_ids" not in prefill_captured:
            input_ids = input[0] if isinstance(input, tuple) else input
            if isinstance(input_ids, torch.Tensor):
                # Capture the FIRST text embedding call
                prefill_captured["text_input_ids"] = input_ids.detach().clone()
                prefill_captured["text_embeddings"] = output.detach().clone()
                print(f"  Captured text: {input_ids.shape}, first_10={input_ids.flatten()[:10].tolist()}")

    def capture_prefill_layer0(module, input, output):
        if not prefill_captured["captured"]:
            hidden = input[0] if isinstance(input, tuple) else input
            if isinstance(hidden, torch.Tensor) and hidden.shape[1] > 10:
                prefill_captured["layer0_input"] = hidden.detach().clone()
                prefill_captured["captured"] = True
                print(f"  Captured layer0 prefill: {hidden.shape}")

    handles = []
    handles.append(talker_model.codec_embedding.register_forward_hook(capture_prefill_codec))
    handles.append(talker_model.text_embedding.register_forward_hook(capture_prefill_text))
    handles.append(talker_model.layers[0].register_forward_hook(capture_prefill_layer0))

    # Generate
    print("\nGenerating...")
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

    # Analyze
    print("\n" + "=" * 80)
    print("Prefill Input Analysis")
    print("=" * 80)

    for name, tensor in prefill_captured.items():
        if isinstance(tensor, torch.Tensor):
            print(f"\n{name}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            if tensor.dtype in [torch.int32, torch.int64, torch.long]:
                print(f"  Min: {tensor.min().item()}, Max: {tensor.max().item()}")
                # Print segments
                flat = tensor.flatten()
                if flat.shape[0] > 0:
                    print(f"  First 20: {flat[:20].tolist()}")
                    if flat.shape[0] > 40:
                        print(f"  Middle 20: {flat[flat.shape[0]//2-10:flat.shape[0]//2+10].tolist()}")
                    print(f"  Last 20: {flat[-20:].tolist()}")

    # Save
    output_dir = Path("/tmp/qwen_tts_tensors")
    torch.save(prefill_captured, output_dir / "prefill_input.pt")
    print(f"\nSaved to {output_dir / 'prefill_input.pt'}")

    return prefill_captured


def main():
    extract_full_input()


if __name__ == "__main__":
    main()
