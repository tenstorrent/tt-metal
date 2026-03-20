# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Extract hidden states from official qwen_tts for TTNN testing.

This extracts:
1. Hidden states BEFORE layer 0 (after embedding combination)
2. Hidden states AFTER each layer
3. Final outputs

These can be used to test TTNN layers directly, bypassing the
embedding/prompt construction complexity.

Run in /tmp/qwen_tts_env:
    source /tmp/qwen_tts_env/bin/activate
    python models/demos/qwen3_tts/demo/extract_hidden_states.py
"""

from pathlib import Path

import torch


def extract_hidden_states():
    """Extract hidden states at each layer."""
    from qwen_tts import Qwen3TTSModel

    print("=" * 80)
    print("Extracting Hidden States from Official qwen_tts")
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

    # Capture hidden states at each layer during prefill
    # We'll capture the first call where seq_len > 50 (prefill)
    hidden_states_captured = {}
    prefill_done = [False]

    def make_layer_hook(layer_idx):
        def hook(module, input, output):
            if prefill_done[0]:
                return

            hidden = input[0] if isinstance(input, tuple) else input
            if isinstance(hidden, torch.Tensor) and hidden.shape[1] > 50:
                key = f"layer_{layer_idx}_input"
                if key not in hidden_states_captured:
                    hidden_states_captured[key] = hidden.detach().clone()

                    out = output[0] if isinstance(output, tuple) else output
                    if isinstance(out, torch.Tensor):
                        hidden_states_captured[f"layer_{layer_idx}_output"] = out.detach().clone()

                    print(
                        f"  Captured layer {layer_idx}: input={hidden.shape}, output={out.shape if isinstance(out, torch.Tensor) else 'N/A'}"
                    )

                    if layer_idx == len(talker_model.layers) - 1:
                        prefill_done[0] = True

        return hook

    # Hook all layers
    handles = []
    for i, layer in enumerate(talker_model.layers):
        handles.append(layer.register_forward_hook(make_layer_hook(i)))

    # Hook norm
    def norm_hook(module, input, output):
        if prefill_done[0]:
            return
        hidden = input[0] if isinstance(input, tuple) else input
        if isinstance(hidden, torch.Tensor) and hidden.shape[1] > 50:
            if "norm_input" not in hidden_states_captured:
                hidden_states_captured["norm_input"] = hidden.detach().clone()
                hidden_states_captured["norm_output"] = output.detach().clone()
                print(f"  Captured norm: input={hidden.shape}, output={output.shape}")

    handles.append(talker_model.norm.register_forward_hook(norm_hook))

    # Hook codec_head
    def codec_head_hook(module, input, output):
        if prefill_done[0]:
            return
        hidden = input[0] if isinstance(input, tuple) else input
        if isinstance(hidden, torch.Tensor) and hidden.shape[1] > 50:
            if "codec_head_input" not in hidden_states_captured:
                hidden_states_captured["codec_head_input"] = hidden.detach().clone()
                hidden_states_captured["codec_head_output"] = output.detach().clone()
                # Also get predicted token
                last_logits = output[:, -1, :]
                predicted_token = torch.argmax(last_logits, dim=-1)
                hidden_states_captured["codec_head_predicted_token"] = predicted_token.item()
                print(
                    f"  Captured codec_head: input={hidden.shape}, output={output.shape}, predicted={predicted_token.item()}"
                )

    handles.append(talker.codec_head.register_forward_hook(codec_head_hook))

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

    print(f"\nGenerated audio: {wavs[0].shape}")

    # Summary
    print("\n" + "=" * 80)
    print("Captured Hidden States")
    print("=" * 80)

    for name, tensor in sorted(hidden_states_captured.items()):
        print(f"  {name}: {tensor.shape}")

    # Save
    output_dir = Path("/tmp/qwen_tts_tensors")
    torch.save(hidden_states_captured, output_dir / "layer_hidden_states.pt")
    print(f"\nSaved {len(hidden_states_captured)} tensors to {output_dir / 'layer_hidden_states.pt'}")

    return hidden_states_captured


def main():
    extract_hidden_states()


if __name__ == "__main__":
    main()
