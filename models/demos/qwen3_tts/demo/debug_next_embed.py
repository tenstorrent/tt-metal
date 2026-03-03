# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug the next-step embedding construction.
Compares what official produces vs what reference would construct.
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
    print("Debug Next Embed Construction")
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

    # Get embeddings
    codec_embed_weight = main_dict["talker.model.codec_embedding.weight"]
    text_embed_weight = main_dict["talker.model.text_embedding.weight"]
    text_proj_fc1_weight = main_dict["talker.text_projection.linear_fc1.weight"]
    text_proj_fc1_bias = main_dict["talker.text_projection.linear_fc1.bias"]
    text_proj_fc2_weight = main_dict["talker.text_projection.linear_fc2.weight"]
    text_proj_fc2_bias = main_dict["talker.text_projection.linear_fc2.bias"]

    code_pred_embeds = []
    for i in range(15):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if key in main_dict:
            code_pred_embeds.append(main_dict[key])

    def project_text(text_embeds):
        h = F.linear(text_embeds, text_proj_fc1_weight, text_proj_fc1_bias)
        h = F.silu(h)
        return F.linear(h, text_proj_fc2_weight, text_proj_fc2_bias)

    # Get tts_pad_embed
    tts_pad_token_id = 151671
    tts_pad_embed = project_text(F.embedding(torch.tensor([[tts_pad_token_id]]), text_embed_weight))
    print(f"  tts_pad_embed: {tts_pad_embed.shape}")

    # Capture what official produces
    print("\n[3] Running official generation and capturing next embed construction...")

    captured = {
        "inputs": [],
        "outputs": [],
        "codec_ids_list": [],
    }

    ref_audio_path = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    target_text = "Hello."

    # Hook the inner model
    def capture_inputs(module, args, kwargs):
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            captured["inputs"].append(kwargs["inputs_embeds"].clone().detach())
        return args, kwargs

    def capture_outputs(module, args, kwargs, output):
        if hasattr(output, "last_hidden_state"):
            captured["outputs"].append(output.last_hidden_state[:, -1:, :].clone().detach())

    # Hook inner model only (not talker.forward to avoid breaking generate)
    hook1 = talker.model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    hook2 = talker.model.register_forward_hook(capture_outputs, with_kwargs=True)

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        max_new_tokens=3,
    )

    hook1.remove()
    hook2.remove()

    print(f"\n  Captured {len(captured['inputs'])} inputs")
    print(f"  Captured {len(captured['outputs'])} outputs")

    # Compare step 0
    if len(captured["inputs"]) >= 2:
        print("\n[4] Analyzing step 0 next embed...")

        official_step0_input = captured["inputs"][1]  # [1, 1, 2048]
        print(f"  Official step 0 input: {official_step0_input.shape}")

        # Reconstruct what reference would build
        # sum(16 codebook embeds) + tts_pad_embed

        all_cb_embeds = []
        for i in range(16):
            token_id = codec_ids[0, i].item()
            if i == 0:
                cb_embed = F.embedding(torch.tensor([[token_id]]), codec_embed_weight)
            else:
                cb_embed = F.embedding(torch.tensor([[token_id]]), code_pred_embeds[i - 1])
            all_cb_embeds.append(cb_embed)

        stacked = torch.cat(all_cb_embeds, dim=1)  # [1, 16, 2048]
        summed = stacked.sum(dim=1, keepdim=True)  # [1, 1, 2048]
        ref_next_embed = summed + tts_pad_embed  # [1, 1, 2048]

        print(f"\n  Reference reconstruction:")
        print(f"    Summed codebook embeds: mean={summed.mean():.4f}, std={summed.std():.4f}")
        print(f"    tts_pad_embed: mean={tts_pad_embed.mean():.4f}, std={tts_pad_embed.std():.4f}")
        print(f"    ref_next_embed: mean={ref_next_embed.mean():.4f}, std={ref_next_embed.std():.4f}")

        print(f"\n  Official step 0 input:")
        print(f"    mean={official_step0_input.mean():.4f}, std={official_step0_input.std():.4f}")

        pcc = compute_pcc(ref_next_embed, official_step0_input)
        print(f"\n  PCC: {pcc:.6f}")

        if pcc < 0.99:
            print(f"  *** MISMATCH DETECTED ***")
            diff = official_step0_input - ref_next_embed
            print(f"  Difference: mean={diff.mean():.4f}, std={diff.std():.4f}")

            # Check if it might be using trailing_text_hidden[0] instead of tts_pad
            # Find what text embedding was used
            text_contribution = official_step0_input - summed
            print(f"\n  Text contribution (official - summed):")
            print(f"    mean={text_contribution.mean():.4f}, std={text_contribution.std():.4f}")

            pcc_with_pad = compute_pcc(text_contribution, tts_pad_embed)
            print(f"    PCC with tts_pad_embed: {pcc_with_pad:.6f}")

    # Check what official uses for trailing_text_hidden
    print("\n[5] Checking official trailing_text_hidden...")

    # We need to capture this from the generate function
    # Let me check the config values

    print("  Checking if text > codec...")
    # In our test case: text = ref_text + target_text + eos = ~38 tokens
    # codec = codec_bos + ref_codes = 1 + 101 = 102 tokens
    # So text_lens < codec_lens, trailing_text_hidden should be tts_pad_embed

    print("=" * 80)


if __name__ == "__main__":
    main()
