# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch / HuggingFace reference inference for Devstral-2-123B (CPU, bfloat16)."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.experimental.devstral2_123B_instruct.reference.hf_reference_loader import (
    DEVSTRAL2_MODEL_ID,
    load_devstral2_causal_lm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single text inference with Devstral-2-123B.")
    parser.add_argument(
        "--prompt",
        default="Write a Python function to reverse a linked list.",
        help="Instruction for the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--offload-folder",
        default="./hf_offload_devstral2_123b",
        help="Folder used by Transformers/Accelerate for disk offloading.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading tokenizer for {DEVSTRAL2_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_MODEL_ID, trust_remote_code=True)

    print(f"Loading model for {DEVSTRAL2_MODEL_ID}...")
    model = load_devstral2_causal_lm(offload_folder=Path(args.offload_folder))
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": args.prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        encoded = tokenizer(args.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
        )

    new_tokens = output_ids[:, input_ids.shape[1] :]
    output_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Response ===")
    print(output_text)


if __name__ == "__main__":
    main()
