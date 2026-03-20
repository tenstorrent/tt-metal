# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Full comparison: Run both official and reference on same input, compare outputs.
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
    print("Full Comparison: Official vs Reference")
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
    code_predictor = talker.code_predictor

    # Load reference weights
    print("\n[2] Loading reference weights...")
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    main_dict = load_file(model_path / "model.safetensors")
    main_dict = {k: v.float() for k, v in main_dict.items()}

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

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    talker_weights = extract_talker_weights(main_dict)
    codec_head = main_dict["talker.codec_head.weight"]
    codec_embed_weight = main_dict["talker.model.codec_embedding.weight"]
    text_embed_weight = main_dict["talker.model.text_embedding.weight"]
    text_proj_fc1_weight = main_dict["talker.text_projection.linear_fc1.weight"]
    text_proj_fc1_bias = main_dict["talker.text_projection.linear_fc1.bias"]
    text_proj_fc2_weight = main_dict["talker.text_projection.linear_fc2.weight"]
    text_proj_fc2_bias = main_dict["talker.text_projection.linear_fc2.bias"]

    code_predictor_weights = extract_code_predictor_weights(main_dict)
    code_predictor_weights = {k.replace("model.", ""): v for k, v in code_predictor_weights.items()}
    code_predictor_config = Qwen3TTSCodePredictorConfig()

    mtp_proj_weight = code_predictor_weights.get("small_to_mtp_projection.weight")
    mtp_proj_bias = code_predictor_weights.get("small_to_mtp_projection.bias")

    def project_text(text_embeds):
        h = F.linear(text_embeds, text_proj_fc1_weight, text_proj_fc1_bias)
        h = F.silu(h)
        return F.linear(h, text_proj_fc2_weight, text_proj_fc2_bias)

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

    # Run official generation, capture internal states
    print("\n[3] Running OFFICIAL generation...")

    ref_audio_path = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    target_text = "Hello."

    official_inputs = []
    official_outputs = []

    def capture_inputs(module, args, kwargs):
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            official_inputs.append(kwargs["inputs_embeds"].clone().detach())
        return args, kwargs

    def capture_outputs(module, args, kwargs, output):
        if hasattr(output, "last_hidden_state"):
            official_outputs.append(output.last_hidden_state[:, -1:, :].clone().detach())

    hook1 = talker.model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    hook2 = talker.model.register_forward_hook(capture_outputs, with_kwargs=True)

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=10,
    )

    hook1.remove()
    hook2.remove()

    print(f"  Official generated {len(official_inputs)} steps")
    official_prefill = official_inputs[0]
    print(f"  Prefill shape: {official_prefill.shape}")

    # Run REFERENCE with exact same prefill input
    print("\n[4] Running REFERENCE starting from official prefill...")

    talker_config = Qwen3TTSConfig()
    talker_config.num_hidden_layers = 28

    # Use official prefill as starting point
    hidden_states = official_prefill.clone()

    # Get tts_pad_embed
    tts_pad_token_id = 151671
    tts_pad_embed = project_text(F.embedding(torch.tensor([[tts_pad_token_id]]), text_embed_weight))

    ref_tokens = []
    max_steps = 10
    codec_eos_id = 2150
    head_dim = 128
    rope_theta = 1000000.0
    rms_norm_eps = 1e-6

    for step in range(max_steps):
        seq_len = hidden_states.shape[1]

        # Compute RoPE
        cos, sin = compute_mrope_frequencies(head_dim, seq_len, rope_theta, hidden_states.device)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # Causal mask
        attention_mask = (
            torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Forward through talker
        x = hidden_states
        for layer_idx in range(28):
            layer_prefix = f"layers.{layer_idx}."
            layer_weights = {
                k.replace(layer_prefix, ""): v for k, v in talker_weights.items() if k.startswith(layer_prefix)
            }
            x = decoder_layer(x, layer_weights, cos, sin, talker_config, attention_mask=attention_mask, use_mrope=True)

        x = rms_norm(x, talker_weights["norm.weight"], rms_norm_eps)
        last_hidden = x[:, -1:, :]

        # Compare with official at this step
        if step < len(official_outputs):
            pcc = compute_pcc(last_hidden, official_outputs[step])
            print(f"  Step {step}: Hidden PCC = {pcc:.6f}")

        # Sample codebook 0
        logits = F.linear(last_hidden.squeeze(1), codec_head)
        token_0 = logits.argmax(dim=-1).item()  # Greedy

        if token_0 == codec_eos_id:
            print(f"  EOS at step {step}")
            break

        ref_tokens.append([token_0])

        # Generate codebooks 1-15 with reference code predictor
        hidden_proj = project_to_code_predictor(last_hidden)
        token_0_embed = F.embedding(torch.tensor([[token_0]]), codec_embed_weight)
        token_0_embed_proj = project_to_code_predictor(token_0_embed)
        cp_input = torch.cat([hidden_proj, token_0_embed_proj], dim=1)

        all_cb_embeds = [token_0_embed]

        for cb_idx in range(15):
            cp_output = code_predictor_forward(cp_input, code_predictor_weights, code_predictor_config)
            cb_hidden = cp_output[:, -1, :]
            cb_logits = F.linear(cb_hidden, lm_heads[cb_idx])
            cb_token = cb_logits.argmax(dim=-1).item()
            ref_tokens[-1].append(cb_token)

            cb_embed = F.embedding(torch.tensor([[cb_token]]), code_pred_embeds[cb_idx])
            all_cb_embeds.append(cb_embed)

            if cb_idx < 14:
                cb_embed_proj = project_to_code_predictor(cb_embed)
                cp_input = torch.cat([cp_input, cb_embed_proj], dim=1)

        # Build next embed
        stacked = torch.cat(all_cb_embeds, dim=1)  # [1, 16, 2048]
        next_embed = stacked.sum(dim=1, keepdim=True) + tts_pad_embed

        # Compare with official next input
        if step + 1 < len(official_inputs):
            official_next = official_inputs[step + 1]
            pcc_next = compute_pcc(next_embed, official_next)
            print(f"  Step {step}: Next embed PCC = {pcc_next:.6f}")

            if pcc_next < 0.99:
                print(f"    *** DIVERGENCE in next embed ***")
                print(f"    Ref:  mean={next_embed.mean():.4f}, std={next_embed.std():.4f}")
                print(f"    Off:  mean={official_next.mean():.4f}, std={official_next.std():.4f}")
                diff = official_next - next_embed
                print(f"    Diff: mean={diff.mean():.4f}, std={diff.std():.4f}")

        hidden_states = torch.cat([hidden_states, next_embed], dim=1)

    print(f"\n  Reference generated {len(ref_tokens)} tokens")

    # Print first few tokens
    if len(ref_tokens) > 0:
        print("\n  First few codebook 0 tokens:")
        for i, row in enumerate(ref_tokens[:5]):
            print(f"    Step {i}: cb0={row[0]}, cb1={row[1]}")

    print("=" * 80)


if __name__ == "__main__":
    main()
