# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Extract Q and K tensors after RoPE from official qwen_tts.

Run in /tmp/qwen_tts_env:
    source /tmp/qwen_tts_env/bin/activate
    python models/demos/qwen3_tts/demo/extract_qk_after_rope.py
"""

from pathlib import Path

import torch


def extract_qk_tensors():
    """Extract Q/K tensors before and after RoPE."""
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

    print("=" * 80)
    print("Extracting Q/K Tensors Before/After RoPE")
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
    config = talker_model.config

    print(f"\nConfig:")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  num_kv_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)}")
    print(f"  rope_scaling: {config.rope_scaling}")

    # Capture tensors
    captured_data = {}
    prefill_done = [False]

    def make_attn_hook(layer_idx):
        def hook(module, args, kwargs, output):
            if prefill_done[0]:
                return

            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None or hidden_states.shape[1] <= 10:
                return

            key = f"layer_{layer_idx}"
            if key in captured_data:
                return

            captured_data[key] = {}

            bsz, q_len, _ = hidden_states.size()
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
            hidden_shape = (bsz, q_len, num_heads, head_dim)
            kv_shape = (bsz, q_len, num_kv_heads, head_dim)

            # Get position embeddings
            position_ids = kwargs.get("position_ids")
            position_embeddings = kwargs.get("position_embeddings")

            if position_embeddings is None:
                # Compute position embeddings
                value_states = module.v_proj(hidden_states)
                cos, sin = talker_model.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings

            # Compute Q and K (matching official forward exactly)
            query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = module.k_norm(module.k_proj(hidden_states).view(kv_shape)).transpose(1, 2)

            captured_data[key]["q_before_rope"] = query_states.clone()
            captured_data[key]["k_before_rope"] = key_states.clone()
            captured_data[key]["cos"] = cos.clone()
            captured_data[key]["sin"] = sin.clone()

            print(f"  Layer {layer_idx}: q shape={query_states.shape}, cos shape={cos.shape}")

            # Apply RoPE
            q_rotated, k_rotated = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                config.rope_scaling["mrope_section"],
                config.rope_scaling["interleaved"],
            )

            captured_data[key]["q_after_rope"] = q_rotated.clone()
            captured_data[key]["k_after_rope"] = k_rotated.clone()

            print(f"  Layer {layer_idx}: q_after_rope shape={q_rotated.shape}")

            if layer_idx == 0:
                captured_data[key]["hidden_input"] = hidden_states.clone()

            if layer_idx == 27:
                prefill_done[0] = True

        return hook

    # Add hooks
    handles = []
    for i, layer in enumerate(talker_model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(i), with_kwargs=True))

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
        import traceback

        print(f"Error during generation: {e}")
        traceback.print_exc()

    for h in handles:
        h.remove()

    # Save
    print("\n" + "=" * 80)
    print("Saving Captured Data")
    print("=" * 80)

    output_dir = Path("/tmp/qwen_tts_tensors")
    output_dir.mkdir(exist_ok=True)

    # Save layer 0 data
    if "layer_0" in captured_data:
        layer0_data = captured_data["layer_0"]
        torch.save(layer0_data, output_dir / "layer0_attention_tensors.pt")
        print(f"\nSaved layer 0 tensors:")
        for k, v in layer0_data.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    return captured_data


if __name__ == "__main__":
    extract_qk_tensors()
