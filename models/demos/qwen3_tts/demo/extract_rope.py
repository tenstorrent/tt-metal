# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Extract MROPE cos/sin tensors from official qwen_tts.

Run in /tmp/qwen_tts_env:
    source /tmp/qwen_tts_env/bin/activate
    python models/demos/qwen3_tts/demo/extract_rope.py
"""

from pathlib import Path

import torch


def extract_rope_tensors():
    """Extract MROPE cos/sin tensors used during prefill."""
    from qwen_tts import Qwen3TTSModel

    print("=" * 80)
    print("Extracting MROPE Tensors from Official qwen_tts")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    print("Model loaded!")

    talker = model.model.talker
    talker_model = talker.model

    # Print config
    config = talker_model.config
    print(f"\nrope_scaling: {config.rope_scaling}")
    print(f"rope_theta: {config.rope_theta}")

    # Capture cos/sin during attention
    captured_data = {}
    prefill_done = [False]

    def make_attn_hook(layer_idx):
        def hook(module, args, kwargs, output):
            if prefill_done[0]:
                return

            # Get hidden_states shape from first arg
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None or hidden_states.shape[1] <= 10:
                return

            # Find position_ids and cos/sin
            position_ids = kwargs.get("position_ids")
            cos_cache = kwargs.get("cos_cache") if "cos_cache" in kwargs else None
            sin_cache = kwargs.get("sin_cache") if "sin_cache" in kwargs else None

            key = f"layer_{layer_idx}"
            if key not in captured_data:
                captured_data[key] = {}

                if position_ids is not None:
                    captured_data[key]["position_ids"] = position_ids.clone()
                    print(f"  Layer {layer_idx}: position_ids shape={position_ids.shape}")

                if cos_cache is not None:
                    captured_data[key]["cos_cache"] = cos_cache.clone()
                    print(f"  Layer {layer_idx}: cos_cache shape={cos_cache.shape}")

                if sin_cache is not None:
                    captured_data[key]["sin_cache"] = sin_cache.clone()
                    print(f"  Layer {layer_idx}: sin_cache shape={sin_cache.shape}")

                # Mark prefill done after capturing first layer
                if layer_idx == 27:
                    prefill_done[0] = True

        return hook

    # Also capture from rotary_emb directly
    rotary_data = {}

    def rotary_hook(module, inputs, output):
        if prefill_done[0]:
            return

        x = inputs[0]
        position_ids = inputs[1] if len(inputs) > 1 else None

        if position_ids is not None and position_ids.shape[-1] > 10:
            rotary_data["position_ids"] = position_ids.clone()
            print(f"\nRotary Emb: position_ids shape={position_ids.shape}")
            if len(position_ids.shape) == 3:
                print(f"  Dim 0 (temporal): {position_ids[0, 0, :10]}...")
                print(f"  Dim 1 (height): {position_ids[1, 0, :10]}...")
                print(f"  Dim 2 (width): {position_ids[2, 0, :10]}...")

            if isinstance(output, tuple) and len(output) == 2:
                cos, sin = output
                rotary_data["cos"] = cos.clone()
                rotary_data["sin"] = sin.clone()
                print(f"  cos shape={cos.shape}, sin shape={sin.shape}")

    # Add hooks
    handles = []
    for i, layer in enumerate(talker_model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(i), with_kwargs=True))

    handles.append(talker_model.rotary_emb.register_forward_hook(rotary_hook))

    # Generate
    print("\nGenerating...")
    ref_audio = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    text = "Hello, this is a test of the text to speech system."

    try:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        print(f"Generated audio: {wavs[0].shape}")
    except Exception as e:
        print(f"Error during generation: {e}")

    for h in handles:
        h.remove()

    # Save
    print("\n" + "=" * 80)
    print("Captured Data")
    print("=" * 80)

    output_dir = Path("/tmp/qwen_tts_tensors")
    output_dir.mkdir(exist_ok=True)

    if rotary_data:
        print("\nRotary Data:")
        for k, v in rotary_data.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        torch.save(rotary_data, output_dir / "rotary_data.pt")
        print(f"\nSaved rotary data to {output_dir / 'rotary_data.pt'}")

    return rotary_data


if __name__ == "__main__":
    extract_rope_tensors()
