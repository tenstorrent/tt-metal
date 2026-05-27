# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate teacher-forcing reference files for readiness checks.

Uses book text as input and captures HuggingFace model's top-K predictions
at each token position. The generated reference file contains prompt tokens,
ground truth continuation tokens, and top-K predictions for validation.

CLI:
    python -m models.common.readiness_check.generate \\
        --hf-model meta-llama/Llama-3.1-8B-Instruct \\
        --prompt-len 128 \\
        --gen-len 256 \\
        --output llama31_8b.refpt

Python:
    from models.common.readiness_check.generate import generate_reference
    generate_reference(
        hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
        prompt_len=128,
        gen_len=256,
        output_path="llama31_8b.refpt",
    )
"""

from __future__ import annotations

import argparse
import bz2
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from models.common.readiness_check.schema import Reference, ReferenceEntry, save_reference

DEFAULT_K = 100
DEFAULT_PROMPT_LEN = 128
DEFAULT_GEN_LEN = 256


def _load_book_text() -> str:
    """Load the tale of two cities book text."""
    # Use the book from tt_transformers tests
    current_file_path = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    book_file = os.path.join(repo_root, "tt_transformers/tests/tale-of-two-cities.txt.bz2")

    if not os.path.exists(book_file):
        raise FileNotFoundError(
            f"Book text not found at {book_file}. " "Expected models/tt_transformers/tests/tale-of-two-cities.txt.bz2"
        )

    with bz2.open(book_file, "rt", encoding="utf-8") as f:
        return f.read()


def _generate_one_entry(
    model: torch.nn.Module,
    tokenizer,
    prompt_tokens: torch.Tensor,  # [prompt_len]
    gen_tokens: torch.Tensor,  # [gen_len]
    top_k: int,
    device: torch.device,
) -> ReferenceEntry:
    """
    Generate one reference entry using batch prefill.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt_tokens: 1D tensor of prompt token IDs
        gen_tokens: 1D tensor of continuation token IDs (ground truth)
        top_k: Number of top predictions to capture
        device: Device to run on

    Returns:
        ReferenceEntry with prompt, generated tokens, and top-K predictions
    """
    prompt_len = len(prompt_tokens)
    gen_len = len(gen_tokens)

    # Concatenate prompt + generated tokens for forward pass
    full_sequence = torch.cat([prompt_tokens, gen_tokens]).unsqueeze(0).to(device)  # [1, prompt_len + gen_len]

    # Forward pass to get logits at all positions
    with torch.no_grad():
        outputs = model(full_sequence)
        logits = outputs.logits  # [1, prompt_len + gen_len, vocab_size]

    # Extract logits at prompt positions (these predict the gen_tokens)
    # logits[0, i] predicts token at position i+1
    # So logits[0, prompt_len-1 : prompt_len+gen_len-1] predicts gen_tokens
    prediction_logits = logits[0, prompt_len - 1 : prompt_len + gen_len - 1, :]  # [gen_len, vocab_size]

    # Get top-K predictions at each position
    k = min(top_k, prediction_logits.shape[-1])
    topk_tokens = torch.topk(prediction_logits, k=k, dim=-1).indices  # [gen_len, k]
    topk_tokens = topk_tokens.to(torch.int32).cpu()

    # Pad with -1 if vocab is smaller than K
    if k < top_k:
        pad = torch.full((gen_len, top_k - k), -1, dtype=torch.int32)
        topk_tokens = torch.cat([topk_tokens, pad], dim=1)

    # Decode prompt text for debugging
    prompt_text = tokenizer.decode(prompt_tokens.tolist(), skip_special_tokens=False)

    return ReferenceEntry(
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens.unsqueeze(0).to(torch.int64),  # [1, prompt_len]
        generated_tokens=gen_tokens.unsqueeze(0).to(torch.int64),  # [1, gen_len]
        topk_tokens=topk_tokens,  # [gen_len, top_k]
        tf_prompt_len=prompt_len,
    )


def generate_reference(
    hf_model_id: str,
    prompt_len: int,
    gen_len: int,
    output_path: Path | str,
    top_k: int = DEFAULT_K,
    num_entries: int = 1,
    device: Optional[torch.device] = None,
) -> Path:
    """
    Generate a readiness reference file from book text.

    Args:
        hf_model_id: HuggingFace model ID
        prompt_len: Length of prompt in tokens
        gen_len: Length of continuation in tokens
        output_path: Where to save the .refpt file
        top_k: Number of top predictions to capture (default 100)
        num_entries: Number of entries to generate (default 1)
        device: Device to run on (default: cuda if available, else cpu)

    Returns:
        Path to saved reference file
    """
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be > 0, got {prompt_len}")
    if gen_len <= 0:
        raise ValueError(f"gen_len must be > 0, got {gen_len}")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if num_entries <= 0:
        raise ValueError(f"num_entries must be > 0, got {num_entries}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {hf_model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, trust_remote_code=True).eval().to(device)

    # Get token IDs
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        cfg_eos = getattr(model.config, "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            cfg_eos = cfg_eos[0]
        eos_id = cfg_eos
    if eos_id is None:
        raise RuntimeError("Could not determine eos_token_id from tokenizer or model config")

    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id

    # Load and tokenize book text
    print("Loading book text...")
    book_text = _load_book_text()
    print(f"Tokenizing book ({len(book_text)} characters)...")
    encoded_tokens = tokenizer.encode(book_text, add_special_tokens=True)
    print(f"Book tokenized to {len(encoded_tokens)} tokens")

    # Check we have enough tokens
    tokens_needed = num_entries * (prompt_len + gen_len)
    if len(encoded_tokens) < tokens_needed:
        raise ValueError(
            f"Book has {len(encoded_tokens)} tokens but need {tokens_needed} "
            f"for {num_entries} entries of {prompt_len}+{gen_len} tokens"
        )

    # Generate entries
    entries = []
    tokens_tensor = torch.tensor(encoded_tokens, dtype=torch.long)

    iterator = range(num_entries)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Generating entries", unit="entry")

    for i in iterator:
        start_idx = i * (prompt_len + gen_len)
        prompt_tokens = tokens_tensor[start_idx : start_idx + prompt_len]
        gen_tokens = tokens_tensor[start_idx + prompt_len : start_idx + prompt_len + gen_len]

        entry = _generate_one_entry(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            top_k=top_k,
            device=device,
        )
        entries.append(entry)

    # Create and save reference
    reference = Reference(
        k=top_k,
        hf_model_id=hf_model_id,
        entries=entries,
        token_ids_meta={
            "bos_id": int(bos_id) if bos_id is not None else None,
            "eos_id": int(eos_id),
            "pad_id": int(pad_id) if pad_id is not None else None,
        },
    )

    return save_reference(reference, output_path)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a readiness-check reference file using batch prefill on book text."
    )
    parser.add_argument("--hf-model", required=True, help="HuggingFace model id or local path.")
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=DEFAULT_PROMPT_LEN,
        help=f"Length of prompt in tokens (default {DEFAULT_PROMPT_LEN}).",
    )
    parser.add_argument(
        "--gen-len",
        type=int,
        default=DEFAULT_GEN_LEN,
        help=f"Length of generation (continuation) in tokens (default {DEFAULT_GEN_LEN}).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to output .refpt file.")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_K, help=f"Top-K to capture per position (default {DEFAULT_K})."
    )
    parser.add_argument(
        "--num-entries",
        type=int,
        default=1,
        help="Number of entries to generate from different positions in the book (default 1).",
    )
    args = parser.parse_args()

    path = generate_reference(
        hf_model_id=args.hf_model,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        output_path=args.output,
        top_k=args.top_k,
        num_entries=args.num_entries,
    )
    print(f"Reference saved to: {path}")


if __name__ == "__main__":
    _main()
