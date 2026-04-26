# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Extract the generation flow from official qwen_tts to understand input/output formats.

This captures:
1. How reference audio becomes codec tokens
2. How text is combined with reference
3. What codec tokens look like (3072 vocab)
4. How codec tokens are converted to RVQ codes (2048×16 vocab)

Run in /tmp/qwen_tts_env:
    source /tmp/qwen_tts_env/bin/activate
    python models/demos/qwen3_tts/demo/extract_generation_flow.py
"""

from pathlib import Path

import torch


def extract_generation_flow():
    """Extract the generation flow details."""
    from qwen_tts import Qwen3TTSModel

    print("=" * 80)
    print("Extracting Generation Flow from Official qwen_tts")
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

    # Storage for tensors
    flow_data = {}

    # 1. Get the voice clone prompt
    print("\n" + "=" * 80)
    print("Step 1: Create Voice Clone Prompt")
    print("=" * 80)

    ref_audio = "/tmp/clone_ref.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    text = "Hello, this is a test of the text to speech system."

    # Create voice clone prompt
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    # prompt_items is a list, take first item
    if isinstance(prompt_items, list):
        prompt_item = prompt_items[0]
    else:
        prompt_item = prompt_items

    # The prompt_item contains ref_code [seq, 16] RVQ codes
    flow_data["ref_code"] = prompt_item.ref_code.clone()
    flow_data["ref_spk_embedding"] = prompt_item.ref_spk_embedding.clone()
    print(f"  ref_code: {flow_data['ref_code'].shape}")
    print(f"  ref_spk_embedding: {flow_data['ref_spk_embedding'].shape}")

    # 2. Understand the generation configuration
    print("\n" + "=" * 80)
    print("Step 2: Examine Model Configuration")
    print("=" * 80)

    # Check codec_head (3072 output vocab)
    print(f"  codec_head output vocab: {talker.codec_head.weight.shape[0]}")

    # Check code predictor LM heads (2048 output vocab each)
    for i, head in enumerate(talker.code_predictor.lm_head):
        if i < 3:  # Just show first few
            print(f"  code_predictor.lm_head_{i} output vocab: {head.weight.shape[0]}")

    # 3. Hook into generation to capture token flow
    print("\n" + "=" * 80)
    print("Step 3: Capture Generation Flow")
    print("=" * 80)

    capture_data = {
        "codec_head_outputs": [],
        "lm_head_outputs": [],
        "input_ids_during_gen": [],
    }

    # Hook codec_head to see what tokens it predicts
    def codec_head_hook(module, input, output):
        # output is logits [batch, 1, 3072]
        if output.shape[1] <= 2:  # Only during generation (not prefill)
            capture_data["codec_head_outputs"].append(output.detach().clone())

    handle_codec = talker.codec_head.register_forward_hook(codec_head_hook)

    # Hook LM heads to see code predictor outputs
    def lm_head_hook(idx):
        def hook(module, input, output):
            if output.shape[1] <= 2:
                if len(capture_data["lm_head_outputs"]) <= idx * 50:
                    capture_data["lm_head_outputs"].append({"head": idx, "output": output.detach().clone()})

        return hook

    handles_lm = []
    for i, head in enumerate(talker.code_predictor.lm_head[:3]):  # Hook first 3
        handles_lm.append(head.register_forward_hook(lm_head_hook(i)))

    # Generate
    print("Generating (capturing flow)...")
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    # Remove hooks
    handle_codec.remove()
    for h in handles_lm:
        h.remove()

    print(f"Generated audio: {wavs[0].shape}, sr={sr}")

    # Analyze captured data
    print("\n" + "=" * 80)
    print("Step 4: Analyze Captured Data")
    print("=" * 80)

    print(f"  codec_head calls captured: {len(capture_data['codec_head_outputs'])}")
    if capture_data["codec_head_outputs"]:
        example = capture_data["codec_head_outputs"][0]
        print(f"  codec_head output shape: {example.shape}")
        predicted_token = torch.argmax(example, dim=-1)
        print(f"  First predicted codec token: {predicted_token.item()}")

    print(f"  lm_head calls captured: {len(capture_data['lm_head_outputs'])}")
    if capture_data["lm_head_outputs"]:
        example = capture_data["lm_head_outputs"][0]
        print(f"  lm_head_{example['head']} output shape: {example['output'].shape}")
        predicted_token = torch.argmax(example["output"], dim=-1)
        print(f"  First predicted LM head tokens: {predicted_token[0].tolist()}")

    # 4. Understand token conversion
    print("\n" + "=" * 80)
    print("Step 5: Token Vocabulary Analysis")
    print("=" * 80)

    # The codec_head predicts tokens in range [0, 3072)
    # The code_predictor LM heads predict tokens in range [0, 2048)
    # These are then fed to the speech tokenizer decoder

    # Check if there's conversion in the model
    print("  Codec vocabulary: 3072 (codec_head)")
    print("  RVQ vocabulary: 2048 per codebook (code_predictor lm_heads)")

    # The 3072 vocab seems to be:
    # - base vocab for text tokens + special tokens
    # - OR a combined codec representation

    # Let's check the ref_code values
    ref_code = flow_data["ref_code"]
    print(f"\n  ref_code stats:")
    print(f"    shape: {ref_code.shape}")
    print(f"    min: {ref_code.min().item()}")
    print(f"    max: {ref_code.max().item()}")
    print(f"    dtype: {ref_code.dtype}")

    # First codebook (semantic)
    print(f"    codebook 0 range: [{ref_code[:, 0].min().item()}, {ref_code[:, 0].max().item()}]")
    # Acoustic codebooks
    for i in range(1, min(4, ref_code.shape[1])):
        print(f"    codebook {i} range: [{ref_code[:, i].min().item()}, {ref_code[:, i].max().item()}]")

    # Save flow data
    output_dir = Path("/tmp/qwen_tts_tensors")
    flow_data["codec_head_example"] = (
        capture_data["codec_head_outputs"][:5] if capture_data["codec_head_outputs"] else None
    )
    flow_data["lm_head_example"] = capture_data["lm_head_outputs"][:5] if capture_data["lm_head_outputs"] else None

    torch.save(flow_data, output_dir / "generation_flow.pt")
    print(f"\nSaved flow data to {output_dir / 'generation_flow.pt'}")

    # 5. Key insight: how does input to talker get constructed?
    print("\n" + "=" * 80)
    print("Step 6: Input Construction Analysis")
    print("=" * 80)

    # The talker takes codec tokens (3072 vocab) via codec_embedding
    # But ref_code is RVQ codes (2048 vocab per codebook)

    # Check if there's a conversion layer
    if hasattr(tts_model, "speech_tokenizer"):
        st = tts_model.speech_tokenizer
        if hasattr(st, "encode"):
            print("  Speech tokenizer has encode method (audio -> RVQ codes)")
        if hasattr(st, "decode"):
            print("  Speech tokenizer has decode method (RVQ codes -> audio)")

    # The key question: how are RVQ codes (2048×16) converted to codec tokens (3072)?
    # Looking at the model structure, the talker uses:
    # - codec_embedding [3072, 2048] for codec tokens
    # - text_embedding [151936, 2048] for text tokens

    # My hypothesis: The 3072 codec vocab might be:
    # - A mapping/quantization of the 16-codebook RVQ representation
    # - Or the first codebook is used directly (2048) plus some offset

    print("\n  Hypothesis: The 3072 codec vocab likely encodes the multi-codebook RVQ")
    print("  structure into a single token vocabulary for autoregressive generation.")

    return flow_data


def main():
    extract_generation_flow()


if __name__ == "__main__":
    main()
