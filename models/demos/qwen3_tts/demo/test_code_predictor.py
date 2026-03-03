# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test code predictor - compare official vs reference.
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
    print("Test Code Predictor - Official vs Reference")
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

    print("  Code predictor loaded")
    print(f"  Model hidden size: {code_predictor.config.hidden_size}")  # Should be 1024

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
        Qwen3TTSCodePredictorConfig,
        code_predictor_forward,
        extract_code_predictor_weights,
    )

    code_predictor_weights = extract_code_predictor_weights(main_dict)
    code_predictor_weights = {k.replace("model.", ""): v for k, v in code_predictor_weights.items()}
    code_predictor_config = Qwen3TTSCodePredictorConfig()

    # Get projection weights
    mtp_proj_weight = code_predictor_weights.get("small_to_mtp_projection.weight")
    mtp_proj_bias = code_predictor_weights.get("small_to_mtp_projection.bias")

    def project_to_code_predictor(x):
        return F.linear(x, mtp_proj_weight, mtp_proj_bias)

    print("  Reference weights loaded")

    # Create test inputs
    print("\n[3] Creating test inputs...")
    torch.manual_seed(42)
    # Simulate: past_hidden (talker output) + last_id_hidden (codebook 0 embedding)
    past_hidden = torch.randn(1, 1, 2048)  # Talker's last hidden state
    last_id_hidden = torch.randn(1, 1, 2048)  # Codebook 0 embedding

    inputs_embeds = torch.cat([past_hidden, last_id_hidden], dim=1)  # [1, 2, 2048]
    print(f"  Input shape: {inputs_embeds.shape}")

    # Test official code predictor forward
    print("\n[4] Testing official code predictor...")
    with torch.no_grad():
        # Project to 1024
        inputs_projected = code_predictor.small_to_mtp_projection(inputs_embeds)
        print(f"  Projected shape: {inputs_projected.shape}")

        # Forward through model
        official_output = code_predictor.model(inputs_embeds=inputs_projected)
        official_hidden = official_output.last_hidden_state
        print(f"  Official output shape: {official_hidden.shape}")
        print(f"  Official output: mean={official_hidden.mean():.4f}, std={official_hidden.std():.4f}")

        # Get logits for codebook 1
        official_logits = code_predictor.lm_head[0](official_hidden[:, -1, :])
        print(f"  Official logits shape: {official_logits.shape}")
        official_token = official_logits.argmax(dim=-1).item()
        print(f"  Official token (greedy): {official_token}")

    # Test reference code predictor forward
    print("\n[5] Testing reference code predictor...")
    with torch.no_grad():
        # Project to 1024 (same as official)
        ref_inputs_projected = project_to_code_predictor(inputs_embeds)
        pcc_proj = compute_pcc(inputs_projected, ref_inputs_projected)
        print(f"  Projection PCC: {pcc_proj:.6f}")

        # Forward through reference model
        ref_output = code_predictor_forward(ref_inputs_projected, code_predictor_weights, code_predictor_config)
        print(f"  Reference output shape: {ref_output.shape}")
        print(f"  Reference output: mean={ref_output.mean():.4f}, std={ref_output.std():.4f}")

        # Compare hidden states
        hidden_pcc = compute_pcc(official_hidden, ref_output)
        print(f"  Hidden state PCC: {hidden_pcc:.6f}")

        # Get logits for codebook 1 using reference
        ref_lm_head_weight = code_predictor_weights.get("lm_head.0.weight")
        ref_logits = F.linear(ref_output[:, -1, :], ref_lm_head_weight)
        print(f"  Reference logits shape: {ref_logits.shape}")
        ref_token = ref_logits.argmax(dim=-1).item()
        print(f"  Reference token (greedy): {ref_token}")

        # Compare logits
        logits_pcc = compute_pcc(official_logits, ref_logits)
        print(f"  Logits PCC: {logits_pcc:.6f}")

    # Test with REAL inputs from generation
    print("\n[6] Testing with real generation inputs...")

    # Generate with official model to get real inputs
    ref_audio_path = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    target_text = "Hello."

    captured_cp_inputs = []

    # Hook code predictor to capture inputs
    original_cp_forward = code_predictor.model.forward

    def capture_cp_forward(*args, **kwargs):
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            ie = kwargs["inputs_embeds"]
            if len(captured_cp_inputs) < 5:
                captured_cp_inputs.append(ie.clone().detach())
                print(
                    f"    CP call #{len(captured_cp_inputs)}: shape={ie.shape}, mean={ie.mean():.4f}, std={ie.std():.4f}"
                )
        return original_cp_forward(*args, **kwargs)

    code_predictor.model.forward = capture_cp_forward

    print("  Running official generation...")
    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=5,
    )

    code_predictor.model.forward = original_cp_forward

    print(f"\n  Captured {len(captured_cp_inputs)} code predictor calls")

    # Compare captured inputs vs what reference would compute
    if len(captured_cp_inputs) > 0:
        print("\n  Comparing first CP input:")
        official_cp_input = captured_cp_inputs[0]
        print(f"    Official: shape={official_cp_input.shape}")
        print(
            f"    Official[0]: mean={official_cp_input[:, 0, :].mean():.4f}, std={official_cp_input[:, 0, :].std():.4f}"
        )
        print(
            f"    Official[1]: mean={official_cp_input[:, 1, :].mean():.4f}, std={official_cp_input[:, 1, :].std():.4f}"
        )

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Projection PCC: {pcc_proj:.6f}")
    print(f"  Hidden PCC: {hidden_pcc:.6f}")
    print(f"  Logits PCC: {logits_pcc:.6f}")
    print(f"  Token match: {official_token == ref_token} (official={official_token}, ref={ref_token})")
    print("=" * 80)


if __name__ == "__main__":
    main()
