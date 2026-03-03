# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug code predictor generation loop - trace exact tokens at each step.
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
    print("Debug Code Predictor Generation Loop")
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

    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSCodePredictorConfig,
        code_predictor_forward,
        extract_code_predictor_weights,
    )

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    main_dict = load_file(Path(model_path) / "model.safetensors")
    main_dict = {k: v.float() for k, v in main_dict.items()}

    codec_embed_weight = main_dict["talker.model.codec_embedding.weight"]
    cp_weights = extract_code_predictor_weights(main_dict)
    cp_weights = {k.replace("model.", ""): v for k, v in cp_weights.items()}
    cp_config = Qwen3TTSCodePredictorConfig()

    mtp_proj_weight = cp_weights.get("small_to_mtp_projection.weight")
    mtp_proj_bias = cp_weights.get("small_to_mtp_projection.bias")

    def project_to_code_predictor(x):
        return F.linear(x, mtp_proj_weight, mtp_proj_bias)

    code_pred_embeds = []
    for i in range(15):
        key = f"codec_embedding.{i}.weight"
        if key in cp_weights:
            code_pred_embeds.append(cp_weights[key])

    lm_heads = []
    for i in range(15):
        key = f"lm_head.{i}.weight"
        if key in cp_weights:
            lm_heads.append(cp_weights[key])

    # Run official code predictor generate and capture intermediate states
    print("\n[3] Running official code predictor...")

    ref_audio_path = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    target_text = "Hello."

    # Capture talker's last hidden state
    captured = {"past_hidden": None, "generated_codes": None}

    # Hook to capture code_predictor.generate input/output
    cp_inputs = []
    cp_outputs = []

    def capture_cp_input(module, args, kwargs):
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            cp_inputs.append(kwargs["inputs_embeds"].clone().detach())
        return args, kwargs

    def capture_cp_output(module, args, kwargs, output):
        if hasattr(output, "last_hidden_state"):
            cp_outputs.append(output.last_hidden_state.clone().detach())

    hook1 = code_predictor.model.register_forward_pre_hook(capture_cp_input, with_kwargs=True)
    hook2 = code_predictor.model.register_forward_hook(capture_cp_output, with_kwargs=True)

    # Also capture from talker model
    talker_outputs = []

    def capture_talker_output(module, args, kwargs, output):
        if hasattr(output, "last_hidden_state"):
            talker_outputs.append(output.last_hidden_state[:, -1:, :].clone().detach())

    hook3 = talker.model.register_forward_hook(capture_talker_output, with_kwargs=True)

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=5,  # A few steps to analyze
    )

    hook1.remove()
    hook2.remove()
    hook3.remove()

    print(f"  Captured {len(cp_inputs)} CP inputs")
    print(f"  Captured {len(cp_outputs)} CP outputs")
    print(f"  Captured {len(talker_outputs)} talker outputs")

    # Analyze the first code predictor call
    if len(cp_inputs) > 0 and len(talker_outputs) > 0:
        print("\n[4] Analyzing first code predictor call...")

        # First talker output is after prefill
        past_hidden = talker_outputs[0]  # [1, 1, 2048]
        print(f"  past_hidden shape: {past_hidden.shape}")

        # First CP input should be [past_hidden_proj, token_0_embed_proj] = [1, 2, 1024]
        official_cp_input = cp_inputs[0]
        print(f"  Official CP input shape: {official_cp_input.shape}")

        # The official generates all 15 codebooks in one call to generate()
        # Let's see how many CP forward calls there are
        print(f"  Total CP forward calls: {len(cp_inputs)}")

        if len(cp_inputs) >= 16:
            print("\n  First 16 CP input shapes (for 15 codebooks + prefill):")
            for i in range(min(16, len(cp_inputs))):
                print(f"    CP call {i}: {cp_inputs[i].shape}")

    # Now reproduce with reference
    print("\n[5] Reproducing with reference...")

    if len(talker_outputs) > 0:
        past_hidden = talker_outputs[0]  # [1, 1, 2048]

        # Generate codebook 0 (we need to know what token_0 the official used)
        # For now, use a known value or compute from logits
        codec_head = main_dict["talker.codec_head.weight"]
        logits = F.linear(past_hidden.squeeze(1), codec_head)
        token_0 = logits.argmax(dim=-1).item()
        print(f"  Codebook 0 token: {token_0}")

        # Build reference CP input
        hidden_proj = project_to_code_predictor(past_hidden)
        token_0_embed = F.embedding(torch.tensor([[token_0]]), codec_embed_weight)
        token_0_embed_proj = project_to_code_predictor(token_0_embed)
        ref_cp_input = torch.cat([hidden_proj, token_0_embed_proj], dim=1)

        print(f"  Reference CP input shape: {ref_cp_input.shape}")

        # Compare with official
        if len(cp_inputs) > 0:
            official_cp_input = cp_inputs[0]
            pcc = compute_pcc(ref_cp_input, official_cp_input)
            print(f"  CP input PCC: {pcc:.6f}")

            if pcc < 0.99:
                print("  *** CP INPUT MISMATCH ***")
                print(f"  Ref: mean={ref_cp_input.mean():.4f}, std={ref_cp_input.std():.4f}")
                print(f"  Off: mean={official_cp_input.mean():.4f}, std={official_cp_input.std():.4f}")

        # Generate codebooks 1-15 with reference
        cp_input = ref_cp_input
        ref_tokens = [token_0]

        print("\n  Generating codebooks 1-15 with reference:")
        for cb_idx in range(15):
            cp_output = code_predictor_forward(cp_input, cp_weights, cp_config)
            cb_hidden = cp_output[:, -1, :]
            cb_logits = F.linear(cb_hidden, lm_heads[cb_idx])
            cb_token = cb_logits.argmax(dim=-1).item()
            ref_tokens.append(cb_token)

            if cb_idx < 3:
                print(f"    Codebook {cb_idx + 1}: token={cb_token}")

                # Compare with official
                if len(cp_outputs) > cb_idx:
                    official_hidden = cp_outputs[cb_idx][:, -1, :]
                    hidden_pcc = compute_pcc(cb_hidden, official_hidden)
                    official_logits = code_predictor.lm_head[cb_idx](official_hidden)
                    official_token = official_logits.argmax(dim=-1).item()
                    print(f"      Hidden PCC: {hidden_pcc:.6f}, Official token: {official_token}")

            # Extend input for next codebook
            if cb_idx < 14:
                cb_embed = F.embedding(torch.tensor([[cb_token]]), code_pred_embeds[cb_idx])
                cb_embed_proj = project_to_code_predictor(cb_embed)
                cp_input = torch.cat([cp_input, cb_embed_proj], dim=1)

        print(f"\n  Reference tokens: {ref_tokens}")

    print("=" * 80)


if __name__ == "__main__":
    main()
