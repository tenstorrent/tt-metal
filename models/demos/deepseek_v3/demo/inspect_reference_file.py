# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Simple script to inspect the contents of the teacher forcing reference file.
Run this to debug what tokens are stored in the reference file.
"""

import os
from pathlib import Path

import torch

# Try to load tokenizer for decoding tokens to text
try:
    from transformers import AutoTokenizer

    MODEL_PATH = Path(
        os.getenv(
            "DEEPSEEK_V3_HF_MODEL",
            "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528",
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    HAS_TOKENIZER = True
except Exception as e:
    print(f"Warning: Could not load tokenizer: {e}")
    tokenizer = None
    HAS_TOKENIZER = False

REFERENCE_FILE = Path(__file__).with_name("deepseek_v3_teacher_forcing.refpt")


def inspect_reference_file(reference_file: Path = REFERENCE_FILE):
    """Load and print contents of the reference file."""

    print(f"\n{'='*70}")
    print(f"Inspecting reference file: {reference_file}")
    print(f"{'='*70}\n")

    if not reference_file.exists():
        print(f"ERROR: Reference file does not exist at {reference_file}")
        print("Generate it first by running: python generate_teacher_forced_file.py")
        return

    # Load the reference file
    payload = torch.load(reference_file, weights_only=False)

    print("Keys in reference file:")
    for key in payload.keys():
        val = payload[key]
        if isinstance(val, torch.Tensor):
            print(f"  - {key}: Tensor {tuple(val.shape)} dtype={val.dtype}")
        elif isinstance(val, dict):
            print(f"  - {key}: dict with keys {list(val.keys())}")
        elif isinstance(val, str):
            print(f"  - {key}: str (len={len(val)})")
        else:
            print(f"  - {key}: {type(val).__name__} = {val}")
    print()

    # --- Prompt text ---
    if "prompt" in payload:
        print(f"Prompt text: {payload['prompt']!r}")
        print()

    # --- Token ID metadata ---
    if "token_ids_meta" in payload:
        meta = payload["token_ids_meta"]
        print("Token ID metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()

    # --- Lengths ---
    if "tf_prompt_len" in payload:
        tf_prompt_len = payload["tf_prompt_len"]
        print(f"tf_prompt_len (prompt length in tokens): {tf_prompt_len}")

    if "max_new_tokens" in payload:
        print(f"max_new_tokens: {payload['max_new_tokens']}")
    print()

    # --- Decoded generated text (stored in payload) ---
    if "decoded_generated_text" in payload:
        print(f"Decoded generated text (from file):")
        print(f"  {payload['decoded_generated_text']!r}")
        print()

    # --- Prompt tokens ---
    if "prompt_tokens" in payload:
        prompt_tokens = payload["prompt_tokens"]
        print(f"prompt_tokens shape: {tuple(prompt_tokens.shape)}, dtype: {prompt_tokens.dtype}")
        if prompt_tokens.dim() == 2:
            prompt_tokens = prompt_tokens[0]
        print(f"  Token IDs: {prompt_tokens.tolist()}")
        if HAS_TOKENIZER:
            decoded = tokenizer.decode(prompt_tokens.tolist(), skip_special_tokens=False)
            print(f"  Decoded: {decoded!r}")
        print()

    # --- Generated tokens ---
    if "generated_tokens" in payload:
        gen_tokens = payload["generated_tokens"]
        print(f"generated_tokens shape: {tuple(gen_tokens.shape)}, dtype: {gen_tokens.dtype}")
        if gen_tokens.dim() == 2:
            gen_tokens = gen_tokens[0]
        print(f"  Token IDs: {gen_tokens.tolist()}")
        if HAS_TOKENIZER:
            decoded = tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=False)
            print(f"  Decoded: {decoded!r}")
            print(f"\n  Token-by-token breakdown:")
            for i, tok_id in enumerate(gen_tokens.tolist()):
                tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
                print(f"    [{i:3d}] ID={tok_id:6d} -> {tok_text!r}")
        print()

    # --- Full reference tokens ---
    if "reference_tokens" in payload:
        ref_tokens = payload["reference_tokens"]
        print(f"reference_tokens shape: {tuple(ref_tokens.shape)}, dtype: {ref_tokens.dtype}")
        if ref_tokens.dim() == 2:
            ref_tokens = ref_tokens[0]
        print(f"  Total tokens: {len(ref_tokens)}")
        print(f"  First 10 IDs: {ref_tokens[:10].tolist()}")
        print(f"  Last 10 IDs: {ref_tokens[-10:].tolist()}")
        print()

    # --- Top-5 tokens ---
    if "top5_tokens" in payload:
        top5 = payload["top5_tokens"]
        print(f"top5_tokens shape: {tuple(top5.shape)}, dtype: {top5.dtype}")
        print("  (row i = model's top-5 prediction for token at position i, given context [0..i-1])")
        print("  (row 0 is zeros since there's no prediction for the first token)")

        # Find first non-zero entry (should be index 1)
        first_nonzero = None
        for i in range(len(top5)):
            if top5[i].sum() != 0:
                first_nonzero = i
                break

        if first_nonzero is not None:
            print(f"\n  First non-zero entry at position {first_nonzero}")
            print(f"  Showing positions {first_nonzero} to {min(first_nonzero + 10, len(top5) - 1)}:")
            for i in range(first_nonzero, min(first_nonzero + 10, len(top5))):
                ids = top5[i].tolist()
                print(f"    Position {i:4d}: {ids}")
                if HAS_TOKENIZER:
                    decoded = [tokenizer.decode([t], skip_special_tokens=False) for t in ids]
                    print(f"                  -> {decoded}")
        print()

    print(f"{'='*70}")
    print("Inspection complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    inspect_reference_file()
