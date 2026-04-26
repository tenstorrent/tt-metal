# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug autoregressive generation loop - compare reference vs official step by step.

This script traces the generation loop beyond the first few tokens to find
where the reference implementation diverges from the official.
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
    print("Debug Generation Loop - Step by Step Comparison")
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
    from transformers import AutoTokenizer

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    main_dict = load_file(model_path / "model.safetensors")
    main_dict = {k: v.float() for k, v in main_dict.items()}
    print(f"  Loaded {len(main_dict)} weights")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSConfig,
        compute_mrope_frequencies,
        decoder_layer,
        extract_talker_weights,
        rms_norm,
    )

    talker_weights = extract_talker_weights(main_dict)
    codec_head = main_dict["talker.codec_head.weight"]
    codec_embed_weight = main_dict["talker.model.codec_embedding.weight"]

    # =========================================================================
    # Create test input using official model
    # =========================================================================
    print("\n[3] Creating voice clone prompt with official model...")

    ref_audio_path = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    target_text = "Hello, this is a test."

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )
    prompt_item = prompt_items[0] if isinstance(prompt_items, list) else prompt_items

    # =========================================================================
    # Get official input embeddings by hooking
    # =========================================================================
    print("\n[4] Getting official input embeddings...")

    tts_model = model.model
    talker = tts_model.talker

    # Hook to capture input_embeds - need to capture the FIRST call which has the full prompt
    captured = {}
    call_count = [0]  # Track call count

    def capture_talker_input(module, args, kwargs):
        call_count[0] += 1
        if "inputs_embeds" in kwargs and call_count[0] == 1:
            # First call has the full prompt
            captured["official_input_embeds"] = kwargs["inputs_embeds"].clone().detach()
            print(f"  [Hook] Captured input_embeds shape: {kwargs['inputs_embeds'].shape}")
        return args, kwargs

    # Register hook
    hook = talker.model.register_forward_pre_hook(capture_talker_input, with_kwargs=True)

    # Run official generation for a few tokens only
    print("  Running official generation with max_new_tokens=50...")

    # Patch the talker to capture intermediate states during generation
    official_tokens = []
    official_logits = []
    official_hiddens = []

    original_forward = talker.model.forward

    captured_step_embeds = []
    captured_tokens = []

    # Hook to capture generated tokens
    original_sample = None
    try:
        original_sample = talker.generate.__wrapped__
    except:
        pass

    def patched_forward(*args, **kwargs):
        # Log what's being passed to the forward
        if call_count[0] <= 5:
            if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                ie = kwargs["inputs_embeds"]
                print(f"  [Forward #{call_count[0]}] inputs_embeds shape: {ie.shape}")
                captured_step_embeds.append(ie.clone().detach())
            if "input_ids" in kwargs and kwargs["input_ids"] is not None:
                ii = kwargs["input_ids"]
                print(f"  [Forward #{call_count[0]}] input_ids shape: {ii.shape}, values: {ii.tolist()}")
            if len(args) > 0:
                if isinstance(args[0], torch.Tensor):
                    print(f"  [Forward #{call_count[0]}] positional arg[0] shape: {args[0].shape}")
        result = original_forward(*args, **kwargs)
        if hasattr(result, "last_hidden_state"):
            official_hiddens.append(result.last_hidden_state[:, -1:, :].clone().detach())
        return result

    talker.model.forward = patched_forward

    # Run generation
    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=50,
    )

    talker.model.forward = original_forward
    hook.remove()

    print(f"  Official generated {len(official_hiddens)} tokens")
    print(f"  Official audio: {wavs[0].shape}")

    # Get official input embeds
    official_input_embeds = captured.get("official_input_embeds")
    if official_input_embeds is None:
        print("  ERROR: Could not capture official input embeds")
        return

    print(f"  Official input embeds shape: {official_input_embeds.shape}")
    print(f"  Official prompt length: {official_input_embeds.shape[1]} tokens")

    # Get weights needed for embedding
    text_embed_weight = main_dict["talker.model.text_embedding.weight"]
    text_proj_fc1_weight = main_dict["talker.text_projection.linear_fc1.weight"]
    text_proj_fc1_bias = main_dict["talker.text_projection.linear_fc1.bias"]
    text_proj_fc2_weight = main_dict["talker.text_projection.linear_fc2.weight"]
    text_proj_fc2_bias = main_dict["talker.text_projection.linear_fc2.bias"]

    def project_text(text_embeds):
        h = F.linear(text_embeds, text_proj_fc1_weight, text_proj_fc1_bias)
        h = F.silu(h)
        return F.linear(h, text_proj_fc2_weight, text_proj_fc2_bias)

    # Get tts_pad embed for comparison
    tts_pad_token_id = 151671
    tts_pad_tokens = torch.tensor([[tts_pad_token_id]])
    tts_pad_embed = project_text(F.embedding(tts_pad_tokens, text_embed_weight))
    print(f"\n  tts_pad_embed: mean={tts_pad_embed.mean():.4f}, std={tts_pad_embed.std():.4f}")

    # Compare step 1 embedding (token 558)
    if len(captured_step_embeds) >= 2:
        step1_embed_official = captured_step_embeds[1]  # [1, 1, 2048]
        step1_embed_ref = F.embedding(torch.tensor([[558]]), codec_embed_weight)
        embed_pcc = compute_pcc(step1_embed_official.squeeze(), step1_embed_ref.squeeze())
        print(f"\n  Step 1 embedding comparison (token 558):")
        print(
            f"    Official: shape={step1_embed_official.shape}, mean={step1_embed_official.mean():.4f}, std={step1_embed_official.std():.4f}"
        )
        print(
            f"    Reference: shape={step1_embed_ref.shape}, mean={step1_embed_ref.mean():.4f}, std={step1_embed_ref.std():.4f}"
        )
        print(f"    PCC: {embed_pcc:.6f}")
        diff = (step1_embed_official.squeeze() - step1_embed_ref.squeeze()).abs()
        print(f"    Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

        # Check if official = ref + tts_pad
        step1_with_pad = step1_embed_ref + tts_pad_embed
        pcc_with_pad = compute_pcc(step1_embed_official.squeeze(), step1_with_pad.squeeze())
        print(f"\n    Testing: official = codec_embed + tts_pad?")
        print(f"    PCC(official, ref+tts_pad): {pcc_with_pad:.6f}")

        # What if tts_pad is added differently? Check stats
        print(f"\n    ref + tts_pad: mean={step1_with_pad.mean():.4f}, std={step1_with_pad.std():.4f}")

    # =========================================================================
    # Run reference generation step by step
    # =========================================================================
    print("\n[5] Running reference generation step by step...")

    # Config
    num_hidden_layers = 28
    head_dim = 128
    rope_theta = 1000000.0
    rms_norm_eps = 1e-6

    talker_config = Qwen3TTSConfig()
    talker_config.num_hidden_layers = num_hidden_layers

    # Start with official input embeds
    hidden_states = official_input_embeds.clone()

    # Get tts_pad embed for streaming mode
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
    tts_pad_tokens = torch.tensor([[tts_pad_token_id]])
    tts_pad_embed = project_text(F.embedding(tts_pad_tokens, text_embed_weight))

    # NOTE: Testing without tts_pad addition to see if that's the source of divergence
    # Add tts_pad to last position (streaming mode)
    # hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + tts_pad_embed
    print(f"  [DEBUG] NOT adding tts_pad to last position")

    # Autoregressive loop
    ref_tokens = []
    ref_logits_list = []
    ref_hiddens = []

    codec_eos_id = 2150
    max_tokens = 50

    print("\n  Step-by-step comparison:")
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
        ref_hiddens.append(last_hidden.squeeze(0))

        # Project to vocab (greedy for comparison)
        logits = F.linear(last_hidden.squeeze(1), codec_head)
        ref_logits_list.append(logits.clone())
        next_token = logits.argmax(dim=-1).item()

        # Compare with official if available
        if step < len(official_hiddens):
            hidden_pcc = compute_pcc(last_hidden.squeeze(), official_hiddens[step].squeeze())

            # We don't have official logits/tokens directly, but we can check hidden PCC
            print(f"  Step {step:3d}: Token={next_token:4d}  Hidden PCC={hidden_pcc:.6f}")

            if hidden_pcc < 0.99:
                print(f"           *** DIVERGENCE DETECTED at step {step} ***")
                print(f"           Hidden state PCC dropped below 0.99")

                # More detailed analysis
                ref_h = last_hidden.squeeze()
                off_h = official_hiddens[step].squeeze()
                print(f"           Ref hidden: mean={ref_h.mean():.4f}, std={ref_h.std():.4f}, norm={ref_h.norm():.4f}")
                print(f"           Off hidden: mean={off_h.mean():.4f}, std={off_h.std():.4f}, norm={off_h.norm():.4f}")
                diff = (ref_h - off_h).abs()
                print(f"           Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
        else:
            print(f"  Step {step:3d}: Token={next_token:4d}  (no official comparison available)")

        # Check for EOS
        if next_token == codec_eos_id:
            print(f"\n  Reference hit EOS at step {step}")
            break

        # Append token
        ref_tokens.append(next_token)
        next_embed = F.embedding(torch.tensor([[next_token]]), codec_embed_weight)
        hidden_states = torch.cat([hidden_states, next_embed], dim=1)

    print("  " + "-" * 70)
    print(f"\n  Reference generated {len(ref_tokens)} tokens")

    # =========================================================================
    # Compare token sequences
    # =========================================================================
    print("\n[6] Analyzing divergence...")

    # Try to extract official tokens from hidden states similarity
    # If hidden states diverge, the tokens will be different

    # Find first point of divergence
    divergence_step = None
    for step in range(min(len(ref_hiddens), len(official_hiddens))):
        pcc = compute_pcc(ref_hiddens[step], official_hiddens[step].squeeze())
        if pcc < 0.99:
            divergence_step = step
            break

    if divergence_step is not None:
        print(f"  First divergence at step {divergence_step}")
        print(f"  This means the autoregressive loop starts to drift around step {divergence_step}")
    else:
        print(f"  No significant divergence detected in first {min(len(ref_hiddens), len(official_hiddens))} steps")
        print(f"  Hidden states match well (PCC > 0.99)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Official tokens generated: {len(official_hiddens)}")
    print(f"  Reference tokens generated: {len(ref_tokens)}")
    print(f"  First divergence step: {divergence_step if divergence_step is not None else 'None detected'}")

    if len(ref_tokens) > 0:
        print(f"\n  First 10 reference tokens: {ref_tokens[:10]}")

    print("=" * 80)


if __name__ == "__main__":
    main()
