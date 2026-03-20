# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug generation loop v2 - compare with CORRECT 16-codebook sum approach.
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    print("=" * 80)
    print("Debug Generation Loop v2 - 16-Codebook Sum Approach")
    print("=" * 80)

    # =========================================================================
    # Load Official Model
    # =========================================================================
    print("\n[1] Loading official qwen_tts model...")
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        print("ERROR: qwen_tts not found")
        return

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    print("  Official model loaded")

    # =========================================================================
    # Load Reference Dependencies
    # =========================================================================
    print("\n[2] Loading reference components...")
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    main_dict = load_file(model_path / "model.safetensors")
    main_dict = {k: v.float() for k, v in main_dict.items()}
    print(f"  Loaded {len(main_dict)} weights")

    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSCodePredictorConfig,
        Qwen3TTSConfig,
        code_predictor_forward,
        compute_mrope_frequencies,
        decoder_layer,
        extract_code_predictor_weights,
        extract_talker_weights,
        rms_norm,
    )

    talker_weights = extract_talker_weights(main_dict)
    codec_head = main_dict["talker.codec_head.weight"]
    codec_embed_weight = main_dict["talker.model.codec_embedding.weight"]

    # Code predictor
    code_predictor_weights = extract_code_predictor_weights(main_dict)
    code_predictor_weights = {k.replace("model.", ""): v for k, v in code_predictor_weights.items()}
    code_predictor_config = Qwen3TTSCodePredictorConfig()

    mtp_proj_weight = code_predictor_weights.get("small_to_mtp_projection.weight")
    mtp_proj_bias = code_predictor_weights.get("small_to_mtp_projection.bias")

    def project_to_code_predictor(x):
        if mtp_proj_weight is not None:
            return F.linear(x, mtp_proj_weight, mtp_proj_bias)
        return x

    code_pred_embeds = []
    for i in range(15):
        key = f"codec_embedding.{i}.weight"
        if key in code_predictor_weights:
            code_pred_embeds.append(code_predictor_weights[key])

    lm_heads = []
    for i in range(15):
        key = f"lm_head.{i}.weight"
        if key in code_predictor_weights:
            lm_heads.append(code_predictor_weights[key])

    # TTS special embeddings
    text_embed_weight = main_dict["talker.model.text_embedding.weight"]
    text_proj_fc1_weight = main_dict["talker.text_projection.linear_fc1.weight"]
    text_proj_fc1_bias = main_dict["talker.text_projection.linear_fc1.bias"]
    text_proj_fc2_weight = main_dict["talker.text_projection.linear_fc2.weight"]
    text_proj_fc2_bias = main_dict["talker.text_projection.linear_fc2.bias"]

    def project_text(text_embeds):
        h = F.linear(text_embeds, text_proj_fc1_weight, text_proj_fc1_bias)
        h = F.silu(h)
        return F.linear(h, text_proj_fc2_weight, text_proj_fc2_bias)

    tts_pad_token_id = 151671
    tts_pad_embed = project_text(F.embedding(torch.tensor([[tts_pad_token_id]]), text_embed_weight))

    # =========================================================================
    # Run Official Generation and Capture Embeddings
    # =========================================================================
    print("\n[3] Running official generation...")

    ref_audio_path = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    target_text = "Hello, this is a test."

    tts_model = model.model
    talker = tts_model.talker

    captured = {}
    call_count = [0]
    captured_step_embeds = []
    official_hiddens = []

    def capture_talker_input(module, args, kwargs):
        call_count[0] += 1
        if "inputs_embeds" in kwargs:
            captured_step_embeds.append(kwargs["inputs_embeds"].clone().detach())
            if call_count[0] == 1:
                captured["official_input_embeds"] = kwargs["inputs_embeds"].clone().detach()
        return args, kwargs

    hook = talker.model.register_forward_pre_hook(capture_talker_input, with_kwargs=True)

    # Hook to capture generated tokens (all 16 codebooks)
    official_all_codes = []

    original_generate_loop = tts_model.talker.generate

    # Capture the final talker result which contains all codec_ids
    def capture_results(module, args, kwargs, output):
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            if isinstance(output.hidden_states, tuple) and len(output.hidden_states) > 1:
                codec_ids = output.hidden_states[1]  # [batch, 16]
                if codec_ids is not None:
                    official_all_codes.append(codec_ids.clone().detach())

    forward_hook = talker.register_forward_hook(capture_results, with_kwargs=True)

    original_forward = talker.model.forward

    def patched_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if hasattr(result, "last_hidden_state"):
            official_hiddens.append(result.last_hidden_state[:, -1:, :].clone().detach())
        return result

    talker.model.forward = patched_forward

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=20,
    )

    talker.model.forward = original_forward
    hook.remove()
    forward_hook.remove()

    print(f"  Official generated {len(official_hiddens)} tokens")
    print(f"  Captured {len(official_all_codes)} full code rows")
    official_input_embeds = captured.get("official_input_embeds")
    print(f"  Official input embeds shape: {official_input_embeds.shape}")

    # =========================================================================
    # Run Reference Generation Step by Step
    # =========================================================================
    print("\n[4] Running reference generation step by step...")

    num_hidden_layers = 28
    head_dim = 128
    rope_theta = 1000000.0
    rms_norm_eps = 1e-6

    talker_config = Qwen3TTSConfig()
    talker_config.num_hidden_layers = num_hidden_layers

    # Start with official input embeds
    hidden_states = official_input_embeds.clone()

    ref_tokens = []
    ref_hiddens = []

    codec_eos_id = 2150
    max_tokens = 20

    print("\n  Step-by-step comparison (using sum of 16 codebooks):")
    print("  " + "-" * 70)

    for step in range(max_tokens):
        current_seq_len = hidden_states.shape[1]

        # Compute RoPE
        cos, sin = compute_mrope_frequencies(head_dim, current_seq_len, rope_theta, hidden_states.device)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # Causal mask
        attention_mask = (
            torch.triu(
                torch.full(
                    (current_seq_len, current_seq_len),
                    float("-inf"),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Forward through all layers
        x = hidden_states
        for layer_idx in range(num_hidden_layers):
            layer_prefix = f"layers.{layer_idx}."
            layer_weights = {
                k.replace(layer_prefix, ""): v for k, v in talker_weights.items() if k.startswith(layer_prefix)
            }
            x = decoder_layer(x, layer_weights, cos, sin, talker_config, attention_mask=attention_mask, use_mrope=True)

        # Final norm
        x = rms_norm(x, talker_weights["norm.weight"], rms_norm_eps)

        # Get last hidden state
        last_hidden = x[:, -1:, :]
        ref_hiddens.append(last_hidden)

        # Generate codebook 0
        logits = F.linear(last_hidden.squeeze(1), codec_head)
        token_0 = logits.argmax(dim=-1).item()  # Greedy for comparison

        # Compare with official
        if step < len(official_hiddens):
            hidden_pcc = compute_pcc(last_hidden.squeeze(), official_hiddens[step].squeeze())
            if step < len(captured_step_embeds) - 1:
                # Compare input embeddings
                off_input = captured_step_embeds[step + 1]  # +1 because first is full prompt
                ref_input = hidden_states[:, -1:, :]
                input_pcc = compute_pcc(ref_input.squeeze(), off_input.squeeze())
                print(f"  Step {step:3d}: Token0={token_0:4d}  Hidden PCC={hidden_pcc:.4f}  Input PCC={input_pcc:.4f}")
            else:
                print(f"  Step {step:3d}: Token0={token_0:4d}  Hidden PCC={hidden_pcc:.4f}")

            if hidden_pcc < 0.99:
                print(f"           *** DIVERGENCE ***")
        else:
            print(f"  Step {step:3d}: Token0={token_0:4d}  (no official comparison)")

        # Check for EOS
        if token_0 == codec_eos_id:
            print(f"\n  Reference hit EOS at step {step}")
            break

        # Generate codebooks 1-15 using code predictor
        code_row = [token_0]

        hidden_proj = project_to_code_predictor(last_hidden)
        token_0_embed = F.embedding(torch.tensor([[token_0]]), codec_embed_weight)
        token_0_embed_proj = project_to_code_predictor(token_0_embed)
        cp_input = torch.cat([hidden_proj, token_0_embed_proj], dim=1)

        # Debug: print code predictor input stats for first 3 steps
        if step < 3:
            print(f"           CP input[0] (hidden_proj): mean={hidden_proj.mean():.4f}, std={hidden_proj.std():.4f}")
            print(
                f"           CP input[1] (token_0_proj): mean={token_0_embed_proj.mean():.4f}, std={token_0_embed_proj.std():.4f}, token={token_0}"
            )

            # Compare with official CP input if available
            if step < len(captured_step_embeds) - 1:
                # Official CP gets inputs after projection
                # We have captured_step_embeds which are the TALKER inputs, not CP inputs
                # Let me compare the token embedding
                pass

        all_cb_embeds = [token_0_embed]

        for cb_idx in range(15):
            cp_output = code_predictor_forward(cp_input, code_predictor_weights, code_predictor_config)
            cb_hidden = cp_output[:, -1, :]
            cb_logits = F.linear(cb_hidden, lm_heads[cb_idx])
            cb_token = cb_logits.argmax(dim=-1).item()  # Greedy
            code_row.append(cb_token)

            cb_embed = F.embedding(torch.tensor([[cb_token]]), code_pred_embeds[cb_idx])
            all_cb_embeds.append(cb_embed)

            if cb_idx < 14:
                cb_embed_proj = project_to_code_predictor(cb_embed)
                cp_input = torch.cat([cp_input, cb_embed_proj], dim=1)

        ref_tokens.append(code_row)

        # Compare codebook tokens with official
        if step < len(official_all_codes):
            off_codes = official_all_codes[step].squeeze().tolist()
            ref_codes = code_row
            match_count = sum(1 for a, b in zip(off_codes, ref_codes) if a == b)
            if match_count < 16:
                print(f"           Codebook match: {match_count}/16")
                print(f"           Off: {off_codes[:5]}...")
                print(f"           Ref: {ref_codes[:5]}...")

        # Build next input: SUM of all 16 codebook embeddings + tts_pad
        all_cb_stacked = torch.cat(all_cb_embeds, dim=1)  # [1, 16, 2048]
        next_embed = all_cb_stacked.sum(dim=1, keepdim=True)  # [1, 1, 2048]
        next_embed = next_embed + tts_pad_embed

        # Debug: compare next_embed with official's next input
        if step < len(captured_step_embeds) - 1:
            off_next_input = captured_step_embeds[step + 1]  # Official's input for next step
            embed_pcc = compute_pcc(next_embed.squeeze(), off_next_input.squeeze())
            if embed_pcc < 0.99:
                print(f"           Next embed PCC: {embed_pcc:.4f}")
                print(
                    f"           Ref: mean={next_embed.mean():.4f}, std={next_embed.std():.4f}, norm={next_embed.norm():.2f}"
                )
                print(
                    f"           Off: mean={off_next_input.mean():.4f}, std={off_next_input.std():.4f}, norm={off_next_input.norm():.2f}"
                )

                # Check if it's just the tts_pad that's wrong
                next_embed_no_pad = all_cb_stacked.sum(dim=1, keepdim=True)
                pcc_no_pad = compute_pcc(next_embed_no_pad.squeeze(), off_next_input.squeeze())
                print(f"           PCC without tts_pad: {pcc_no_pad:.4f}")

                # Check if official has different tts_pad
                diff = off_next_input - next_embed
                print(f"           Diff: mean={diff.mean():.4f}, std={diff.std():.4f}")

        hidden_states = torch.cat([hidden_states, next_embed], dim=1)

    print("  " + "-" * 70)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Official tokens generated: {len(official_hiddens)}")
    print(f"  Reference tokens generated: {len(ref_tokens)}")

    if len(ref_tokens) > 0:
        print(f"\n  First 5 reference tokens (codebook 0):")
        for i, row in enumerate(ref_tokens[:5]):
            print(f"    Step {i}: {row[0]}")

    print("=" * 80)


if __name__ == "__main__":
    main()
