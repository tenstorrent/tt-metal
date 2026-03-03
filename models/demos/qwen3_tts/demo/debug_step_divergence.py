# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug script to trace exact divergence point between official and reference talker.
Uses hooks on the inner model to avoid breaking generation.
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    print("=" * 80)
    print("Debug Step Divergence - Tracing Official vs Reference")
    print("=" * 80)

    # Load official model
    print("\n[1] Loading official model...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    tts_model = model.model
    talker = tts_model.talker

    # Load reference weights
    print("\n[2] Loading reference weights...")
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    main_dict = load_file(model_path / "model.safetensors")
    main_dict = {k: v.float() for k, v in main_dict.items()}

    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSConfig,
        compute_mrope_frequencies,
        decoder_layer,
        extract_talker_weights,
        rms_norm,
    )

    talker_weights = extract_talker_weights(main_dict)
    codec_head = main_dict["talker.codec_head.weight"]

    # Run official generation to capture states
    print("\n[3] Running official generation...")

    ref_audio_path = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    target_text = "Hello."

    captured_inputs = []
    captured_outputs = []

    # Hook the inner model (talker.model) to capture inputs/outputs
    def capture_model_forward(module, args, kwargs):
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            captured_inputs.append(kwargs["inputs_embeds"].clone().detach())
        return args, kwargs

    def capture_model_output(module, args, kwargs, output):
        if hasattr(output, "last_hidden_state"):
            captured_outputs.append(output.last_hidden_state[:, -1:, :].clone().detach())

    hook1 = talker.model.register_forward_pre_hook(capture_model_forward, with_kwargs=True)
    hook2 = talker.model.register_forward_hook(capture_model_output, with_kwargs=True)

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=5,
    )

    hook1.remove()
    hook2.remove()

    print(f"\n  Captured {len(captured_inputs)} inputs and {len(captured_outputs)} outputs")

    # Analyze captured data
    if len(captured_inputs) > 0:
        print("\n  Input shapes:")
        for i, inp in enumerate(captured_inputs[:5]):
            print(f"    Call {i}: {inp.shape}")

    if len(captured_outputs) > 0:
        print("\n  Output shapes:")
        for i, out in enumerate(captured_outputs[:5]):
            print(f"    Call {i}: {out.shape}")

    # Now run reference on the captured prefill input
    print("\n[4] Running reference on captured prefill input...")

    talker_config = Qwen3TTSConfig()
    talker_config.num_hidden_layers = 28
    head_dim = 128
    rope_theta = 1000000.0
    rms_norm_eps = 1e-6

    if len(captured_inputs) > 0:
        prefill_input = captured_inputs[0]
        print(f"\n  Prefill input shape: {prefill_input.shape}")

        x = prefill_input.clone()
        seq_len = x.shape[1]

        cos, sin = compute_mrope_frequencies(head_dim, seq_len, rope_theta, x.device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        attention_mask = (
            torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        for layer_idx in range(28):
            layer_prefix = f"layers.{layer_idx}."
            layer_weights = {
                k.replace(layer_prefix, ""): v for k, v in talker_weights.items() if k.startswith(layer_prefix)
            }
            x = decoder_layer(x, layer_weights, cos, sin, talker_config, attention_mask=attention_mask, use_mrope=True)

        x = rms_norm(x, talker_weights["norm.weight"], rms_norm_eps)
        ref_prefill_output = x[:, -1:, :]

        official_prefill_output = captured_outputs[0]
        pcc = compute_pcc(ref_prefill_output, official_prefill_output)
        print(f"  Prefill PCC: {pcc:.6f}")

    # Test step 1 (if we have enough captures)
    if len(captured_inputs) >= 2:
        print("\n[5] Testing Step 1...")

        # The official with KV cache: captured_inputs[1] is just [1, 1, 2048]
        # The reference needs to concatenate prefill + step0 and recompute

        step0_input = captured_inputs[1]  # [1, 1, 2048] - the new token embedding
        print(f"  Step 0 input (new token): {step0_input.shape}")

        # Build full sequence
        full_seq = torch.cat([prefill_input, step0_input], dim=1)
        print(f"  Full sequence shape: {full_seq.shape}")

        x = full_seq.clone()
        seq_len = x.shape[1]

        cos, sin = compute_mrope_frequencies(head_dim, seq_len, rope_theta, x.device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        attention_mask = (
            torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        for layer_idx in range(28):
            layer_prefix = f"layers.{layer_idx}."
            layer_weights = {
                k.replace(layer_prefix, ""): v for k, v in talker_weights.items() if k.startswith(layer_prefix)
            }
            x = decoder_layer(x, layer_weights, cos, sin, talker_config, attention_mask=attention_mask, use_mrope=True)

        x = rms_norm(x, talker_weights["norm.weight"], rms_norm_eps)
        ref_step0_output = x[:, -1:, :]

        official_step0_output = captured_outputs[1]
        pcc = compute_pcc(ref_step0_output, official_step0_output)
        print(f"  Step 0 output PCC: {pcc:.6f}")

        if pcc < 0.99:
            print(f"  *** DIVERGENCE DETECTED ***")
            print(f"  Reference: mean={ref_step0_output.mean():.4f}, std={ref_step0_output.std():.4f}")
            print(f"  Official: mean={official_step0_output.mean():.4f}, std={official_step0_output.std():.4f}")

            # Compare the generated codebook 0 token
            ref_logits = F.linear(ref_step0_output.squeeze(1), codec_head)
            off_logits = F.linear(official_step0_output.squeeze(1), codec_head)
            ref_token = ref_logits.argmax(dim=-1).item()
            off_token = off_logits.argmax(dim=-1).item()
            print(f"  Reference token 0: {ref_token}")
            print(f"  Official token 0: {off_token}")

    # Compare step by step until divergence
    if len(captured_inputs) >= 3:
        print("\n[6] Step-by-step comparison...")
        full_seq = prefill_input.clone()

        for step in range(min(4, len(captured_inputs) - 1)):
            step_input = captured_inputs[step + 1]  # New token for this step
            full_seq = torch.cat([full_seq, step_input], dim=1)

            x = full_seq.clone()
            seq_len = x.shape[1]

            cos, sin = compute_mrope_frequencies(head_dim, seq_len, rope_theta, x.device)
            cos = cos.to(x.dtype)
            sin = sin.to(x.dtype)

            attention_mask = (
                torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            for layer_idx in range(28):
                layer_prefix = f"layers.{layer_idx}."
                layer_weights = {
                    k.replace(layer_prefix, ""): v for k, v in talker_weights.items() if k.startswith(layer_prefix)
                }
                x = decoder_layer(
                    x, layer_weights, cos, sin, talker_config, attention_mask=attention_mask, use_mrope=True
                )

            x = rms_norm(x, talker_weights["norm.weight"], rms_norm_eps)
            ref_output = x[:, -1:, :]

            official_output = captured_outputs[step + 1]
            pcc = compute_pcc(ref_output, official_output)
            print(f"  Step {step}: PCC = {pcc:.6f}")

            if pcc < 0.99:
                print(f"    *** DIVERGENCE at step {step} ***")
                # Don't break - continue to see the pattern

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
