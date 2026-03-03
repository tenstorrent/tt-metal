# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug codebook 1 generation - single step comparison.
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
    print("Debug Codebook 1 Generation")
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

    mtp_proj_weight = code_predictor_weights.get("small_to_mtp_projection.weight")
    mtp_proj_bias = code_predictor_weights.get("small_to_mtp_projection.bias")
    codec_embed_weight = main_dict["talker.model.codec_embedding.weight"]

    def project_to_code_predictor(x):
        return F.linear(x, mtp_proj_weight, mtp_proj_bias)

    # Get code predictor embeddings
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

    # Run official generation and capture first step
    print("\n[3] Running official generation...")

    ref_audio_path = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    target_text = "Hello."

    captured = {}

    # Hook talker forward to capture past_hidden and codec 0 token
    original_talker_forward = talker.forward

    def capture_talker_forward(*args, **kwargs):
        result = original_talker_forward(*args, **kwargs)
        if hasattr(result, "past_hidden") and result.past_hidden is not None:
            if "past_hidden" not in captured:
                captured["past_hidden"] = result.past_hidden.clone().detach()
                captured["codec_ids"] = (
                    result.hidden_states[1].clone().detach() if result.hidden_states is not None else None
                )
                print(f"    Captured past_hidden: {result.past_hidden.shape}")
                if captured["codec_ids"] is not None:
                    print(f"    Captured codec_ids: {captured['codec_ids']}")
        return result

    talker.forward = capture_talker_forward

    # Hook code predictor to capture first input/output
    cp_calls = []
    original_cp_forward = code_predictor.model.forward

    def capture_cp_forward(*args, **kwargs):
        if "inputs_embeds" in kwargs:
            ie = kwargs["inputs_embeds"]
            result = original_cp_forward(*args, **kwargs)
            if len(cp_calls) < 2:
                cp_calls.append({"input": ie.clone().detach(), "output": result.last_hidden_state.clone().detach()})
                print(f"    CP call #{len(cp_calls)}: input={ie.shape}, output={result.last_hidden_state.shape}")
            return result
        return original_cp_forward(*args, **kwargs)

    code_predictor.model.forward = capture_cp_forward

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=2,
    )

    talker.forward = original_talker_forward
    code_predictor.model.forward = original_cp_forward

    print(f"\n  Captured {len(cp_calls)} CP calls")

    # Now reproduce with reference
    print("\n[4] Reproducing with reference...")

    if "past_hidden" in captured and len(cp_calls) > 0:
        past_hidden = captured["past_hidden"]
        codec_ids = captured["codec_ids"]
        token_0 = codec_ids[0, 0].item()

        print(f"  past_hidden shape: {past_hidden.shape}")
        print(f"  codec_ids: {codec_ids[0].tolist()}")
        print(f"  token_0: {token_0}")

        # Build reference CP input (matching official)
        # Official: inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1)
        # Then projected through small_to_mtp_projection

        last_id_hidden = F.embedding(torch.tensor([[token_0]]), codec_embed_weight)
        cp_input_2048 = torch.cat([past_hidden, last_id_hidden], dim=1)  # [1, 2, 2048]

        # Project to 1024 (matching official)
        cp_input_official = cp_calls[0]["input"]  # [1, 2, 1024] from official
        cp_input_ref = project_to_code_predictor(cp_input_2048)  # [1, 2, 1024] from reference

        input_pcc = compute_pcc(cp_input_official, cp_input_ref)
        print(f"\n  CP input PCC: {input_pcc:.6f}")
        print(f"  Official input: mean={cp_input_official.mean():.4f}, std={cp_input_official.std():.4f}")
        print(f"  Reference input: mean={cp_input_ref.mean():.4f}, std={cp_input_ref.std():.4f}")

        # Run reference forward
        cp_output_ref = code_predictor_forward(cp_input_ref, code_predictor_weights, code_predictor_config)
        cp_output_official = cp_calls[0]["output"]

        output_pcc = compute_pcc(cp_output_official, cp_output_ref)
        print(f"\n  CP output PCC: {output_pcc:.6f}")
        print(f"  Official output: mean={cp_output_official.mean():.4f}, std={cp_output_official.std():.4f}")
        print(f"  Reference output: mean={cp_output_ref.mean():.4f}, std={cp_output_ref.std():.4f}")

        # Compare logits for codebook 1
        official_logits = code_predictor.lm_head[0](cp_output_official[:, -1, :])
        ref_logits = F.linear(cp_output_ref[:, -1, :], lm_heads[0])

        logits_pcc = compute_pcc(official_logits, ref_logits)
        print(f"\n  Logits PCC: {logits_pcc:.6f}")

        official_token = official_logits.argmax(dim=-1).item()
        ref_token = ref_logits.argmax(dim=-1).item()
        print(f"  Official token: {official_token}")
        print(f"  Reference token: {ref_token}")
        print(f"  Match: {official_token == ref_token}")

    print("=" * 80)


if __name__ == "__main__":
    main()
