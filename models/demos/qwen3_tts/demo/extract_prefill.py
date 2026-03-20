# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Extract prefill phase tensors from official qwen_tts for PCC comparison.

This captures the full context (ref_audio tokens + text tokens) during prefill,
not just the single-token decode phase.

Run in /tmp/qwen_tts_env:
    source /tmp/qwen_tts_env/bin/activate
    python models/demos/qwen3_tts/demo/extract_prefill.py
"""

from pathlib import Path

import torch


def extract_prefill_tensors():
    """Extract prefill phase tensors."""
    from qwen_tts import Qwen3TTSModel

    print("=" * 80)
    print("Extracting Prefill Phase Tensors")
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

    # Storage
    captured = {"prefill": {}}
    capture_enabled = [True]  # Use list for mutable closure

    def make_hook(name, storage_dict):
        def hook(module, input, output):
            if not capture_enabled[0]:
                return
            if isinstance(output, torch.Tensor) and output.shape[1] > 1:
                # Only capture if seq_len > 1 (prefill)
                storage_dict[f"{name}_output"] = output.detach().clone()
                if isinstance(input, tuple) and len(input) > 0:
                    if isinstance(input[0], torch.Tensor):
                        storage_dict[f"{name}_input"] = input[0].detach().clone()

        return hook

    handles = []

    # Hook all talker layers
    for i, layer in enumerate(talker_model.layers):
        handles.append(layer.register_forward_hook(make_hook(f"talker_layer_{i}", captured["prefill"])))

    # Hook norm
    handles.append(talker_model.norm.register_forward_hook(make_hook("talker_norm", captured["prefill"])))

    # Hook embeddings
    handles.append(
        talker_model.codec_embedding.register_forward_hook(make_hook("codec_embedding", captured["prefill"]))
    )
    handles.append(talker_model.text_embedding.register_forward_hook(make_hook("text_embedding", captured["prefill"])))

    print(f"  Hooked {len(talker_model.layers)} talker layers + norm + embeddings")

    # Also hook code predictor
    code_pred = talker.code_predictor
    for i, layer in enumerate(code_pred.model.layers):
        handles.append(layer.register_forward_hook(make_hook(f"code_pred_layer_{i}", captured["prefill"])))
    handles.append(code_pred.model.norm.register_forward_hook(make_hook("code_pred_norm", captured["prefill"])))

    # Hook LM heads
    for i, head in enumerate(code_pred.lm_head):
        handles.append(head.register_forward_hook(make_hook(f"code_pred_lm_head_{i}", captured["prefill"])))

    print(f"  Hooked {len(code_pred.model.layers)} code_pred layers + norm + {len(code_pred.lm_head)} LM heads")

    # Generate
    print("\n" + "=" * 80)
    print("Generating (capturing prefill)...")
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

    # Disable capture
    capture_enabled[0] = False

    # Remove hooks
    for h in handles:
        h.remove()

    print(f"Generated audio: {wavs[0].shape}, sr={sr}")

    # Print captured prefill tensors
    print("\n" + "=" * 80)
    print("Captured Prefill Tensors")
    print("=" * 80)

    prefill_tensors = captured["prefill"]
    print(f"Total captured: {len(prefill_tensors)}")

    for name, tensor in sorted(prefill_tensors.items()):
        print(f"  {name}: {tensor.shape}, dtype={tensor.dtype}")

    # Save
    output_dir = Path("/tmp/qwen_tts_tensors")
    output_dir.mkdir(exist_ok=True)

    torch.save(prefill_tensors, output_dir / "prefill_tensors.pt")
    print(f"\nSaved to {output_dir / 'prefill_tensors.pt'}")

    # Also save model weights for comparison
    print("\n" + "=" * 80)
    print("Extracting Model Weights")
    print("=" * 80)

    weights = {}

    # Talker layer 0 weights
    layer0 = talker_model.layers[0]
    weights["talker.layer_0.input_layernorm.weight"] = layer0.input_layernorm.weight.detach().clone()
    weights["talker.layer_0.self_attn.q_proj.weight"] = layer0.self_attn.q_proj.weight.detach().clone()
    weights["talker.layer_0.self_attn.k_proj.weight"] = layer0.self_attn.k_proj.weight.detach().clone()
    weights["talker.layer_0.self_attn.v_proj.weight"] = layer0.self_attn.v_proj.weight.detach().clone()
    weights["talker.layer_0.self_attn.o_proj.weight"] = layer0.self_attn.o_proj.weight.detach().clone()
    weights["talker.layer_0.self_attn.q_norm.weight"] = layer0.self_attn.q_norm.weight.detach().clone()
    weights["talker.layer_0.self_attn.k_norm.weight"] = layer0.self_attn.k_norm.weight.detach().clone()
    weights["talker.layer_0.post_attention_layernorm.weight"] = layer0.post_attention_layernorm.weight.detach().clone()
    weights["talker.layer_0.mlp.gate_proj.weight"] = layer0.mlp.gate_proj.weight.detach().clone()
    weights["talker.layer_0.mlp.up_proj.weight"] = layer0.mlp.up_proj.weight.detach().clone()
    weights["talker.layer_0.mlp.down_proj.weight"] = layer0.mlp.down_proj.weight.detach().clone()

    # Embeddings
    weights["talker.codec_embedding.weight"] = talker_model.codec_embedding.weight.detach().clone()
    weights["talker.text_embedding.weight"] = talker_model.text_embedding.weight.detach().clone()
    weights["talker.norm.weight"] = talker_model.norm.weight.detach().clone()

    # Codec head
    weights["talker.codec_head.weight"] = talker.codec_head.weight.detach().clone()

    # Code predictor layer 0
    cp_layer0 = code_pred.model.layers[0]
    weights["code_pred.layer_0.input_layernorm.weight"] = cp_layer0.input_layernorm.weight.detach().clone()
    weights["code_pred.layer_0.self_attn.q_proj.weight"] = cp_layer0.self_attn.q_proj.weight.detach().clone()
    weights["code_pred.layer_0.self_attn.k_proj.weight"] = cp_layer0.self_attn.k_proj.weight.detach().clone()
    weights["code_pred.layer_0.self_attn.v_proj.weight"] = cp_layer0.self_attn.v_proj.weight.detach().clone()
    weights["code_pred.layer_0.self_attn.o_proj.weight"] = cp_layer0.self_attn.o_proj.weight.detach().clone()

    # Code predictor LM heads
    for i, head in enumerate(code_pred.lm_head):
        weights[f"code_pred.lm_head_{i}.weight"] = head.weight.detach().clone()

    print(f"Extracted {len(weights)} weight tensors")

    torch.save(weights, output_dir / "model_weights.pt")
    print(f"Saved to {output_dir / 'model_weights.pt'}")

    return prefill_tensors, weights


def main():
    extract_prefill_tensors()


if __name__ == "__main__":
    main()
