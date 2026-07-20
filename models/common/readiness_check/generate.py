# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate teacher-forcing reference files for readiness checks.

Uses book text or a chat-template AIME24 prompt as input and captures
HuggingFace model's top-K predictions at each token position. The generated
reference file contains prompt tokens, ground truth continuation tokens, and
top-K predictions for validation.

CLI:
    python -m models.common.readiness_check.generate \\
        --hf-model meta-llama/Llama-3.1-8B-Instruct \\
        --prompt-source aime24 \\
        --chat-template \\
        --gen-len 100 \\
        --output llama31_8b.refpt

Python:
    from models.common.readiness_check.generate import generate_reference
    generate_reference(
        hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
        prompt_source="aime24",
        chat_template=True,
        gen_len=100,
        output_path="llama31_8b.refpt",
    )
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
from collections.abc import Mapping
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
DEFAULT_GEN_LEN = 100
DEFAULT_PROMPT_SOURCE = "book"
DEFAULT_AIME24_PROMPTS_FILE = (
    Path(__file__).resolve().parents[2] / "demos" / "deepseek_v3" / "demo" / "aime_under_8k_prompts.json"
)
PROMPT_SOURCE_CHOICES = ("book", "aime24", "text", "file")


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


def _load_aime24_prompt(prompts_file: Path, prompt_index: int) -> str:
    """Load one AIME24 prompt from DeepSeek's curated teacher-forcing prompt set."""
    if prompt_index < 0:
        raise ValueError(f"aime24_prompt_index must be >= 0, got {prompt_index}")
    if not prompts_file.exists():
        raise FileNotFoundError(f"AIME24 prompts file not found: {prompts_file}")

    with prompts_file.open("r", encoding="utf-8") as f:
        prompt_items = json.load(f)

    if not isinstance(prompt_items, list) or not prompt_items:
        raise ValueError(f"AIME24 prompts file must contain a non-empty list: {prompts_file}")
    if prompt_index >= len(prompt_items):
        raise ValueError(f"AIME24 prompt index {prompt_index} out of range for {len(prompt_items)} prompts")

    item = prompt_items[prompt_index]
    if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
        raise ValueError(f"AIME24 prompt item {prompt_index} must contain a string 'prompt' field")
    task = item.get("task")
    if task not in {"aime24", "r1_aime24"}:
        raise ValueError(f"AIME24 prompt item {prompt_index} has unexpected task {task!r}")
    return item["prompt"]


def _resolve_prompt_text(
    prompt_source: str,
    *,
    prompt: Optional[str],
    prompt_file: Optional[Path],
    aime24_prompts_file: Path,
    aime24_prompt_index: int,
) -> str:
    if prompt_source == "aime24":
        return _load_aime24_prompt(aime24_prompts_file, aime24_prompt_index)
    if prompt_source == "text":
        if prompt is None:
            raise ValueError("--prompt is required when --prompt-source=text")
        return prompt
    if prompt_source == "file":
        if prompt_file is None:
            raise ValueError("--prompt-file is required when --prompt-source=file")
        return prompt_file.read_text(encoding="utf-8")
    raise ValueError(f"Prompt source {prompt_source!r} does not provide a direct prompt")


def _chat_or_plain_prompt_tokens(tokenizer, prompt_text: str, *, chat_template: bool) -> list[int]:
    if chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(f"Tokenizer {type(tokenizer).__name__} does not support apply_chat_template")
        prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
        )
    else:
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

    # Some fast tokenizers return a BatchEncoding from apply_chat_template even
    # when tokenize=True.  Normalize the single-prompt result before iterating;
    # otherwise iteration yields the mapping key ("input_ids") rather than IDs.
    if isinstance(prompt_tokens, Mapping):
        if "input_ids" not in prompt_tokens:
            raise ValueError("Chat-template tokenization did not return input_ids")
        prompt_tokens = prompt_tokens["input_ids"]
    if isinstance(prompt_tokens, torch.Tensor):
        prompt_tokens = prompt_tokens.tolist()
    if len(prompt_tokens) == 1 and isinstance(prompt_tokens[0], (list, tuple)):
        prompt_tokens = prompt_tokens[0]

    if not prompt_tokens:
        raise ValueError("Prompt tokenization produced no tokens")
    return [int(token_id) for token_id in prompt_tokens]


def _normal_token_ids(token_ids) -> list[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, int):
        return [token_ids]
    return [int(token_id) for token_id in token_ids if token_id is not None]


def _generation_stop_ids(tokenizer, model: torch.nn.Module) -> list[int]:
    stop_ids = _normal_token_ids(tokenizer.eos_token_id)
    stop_ids.extend(_normal_token_ids(getattr(model.config, "eos_token_id", None)))

    eot_id = getattr(tokenizer, "eot_token_id", None)
    if eot_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        for token in ("<|eot_id|>", "<end_of_turn>"):
            if vocab and token not in vocab:
                continue
            converted = tokenizer.convert_tokens_to_ids(token)
            if isinstance(converted, int) and converted >= 0 and converted != tokenizer.unk_token_id:
                eot_id = converted
                break
    stop_ids.extend(_normal_token_ids(eot_id))

    deduped = []
    for token_id in stop_ids:
        if token_id not in deduped:
            deduped.append(token_id)
    if not deduped:
        raise RuntimeError("Could not determine eos/eot token id from tokenizer or model config")
    return deduped


def _safe_pad_id(tokenizer, stop_ids: list[int]) -> Optional[int]:
    pad_id = tokenizer.pad_token_id
    if pad_id is not None and pad_id != tokenizer.bos_token_id:
        return int(pad_id)
    return int(stop_ids[0]) if stop_ids else None


def _generate_continuation_tokens(
    model: torch.nn.Module,
    tokenizer,
    prompt_tokens: torch.Tensor,
    gen_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a deterministic HF continuation to use as the teacher-forcing target."""
    input_ids = prompt_tokens.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    stop_ids = _generation_stop_ids(tokenizer, model)
    eos_token_id = stop_ids[0] if len(stop_ids) == 1 else stop_ids

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_len,
            do_sample=False,
            pad_token_id=_safe_pad_id(tokenizer, stop_ids),
            eos_token_id=eos_token_id,
            use_cache=True,
        )

    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    generated = sequences[0, prompt_tokens.numel() :].detach().cpu().to(torch.long)
    if generated.numel() == 0:
        raise RuntimeError("HF generation produced no continuation tokens")
    return generated


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
    prompt_len: int = DEFAULT_PROMPT_LEN,
    gen_len: int = DEFAULT_GEN_LEN,
    output_path: Path | str = "reference.refpt",
    top_k: int = DEFAULT_K,
    num_entries: int = 1,
    device: Optional[torch.device] = None,
    prompt_source: str = DEFAULT_PROMPT_SOURCE,
    chat_template: bool = False,
    prompt: Optional[str] = None,
    prompt_file: Optional[Path] = None,
    aime24_prompts_file: Path = DEFAULT_AIME24_PROMPTS_FILE,
    aime24_prompt_index: int = 0,
    fix_mistral_regex: bool = False,
) -> Path:
    """
    Generate a readiness reference file from book text or a prompt.

    Args:
        hf_model_id: HuggingFace model ID
        prompt_len: Length of prompt in tokens for prompt_source="book"
        gen_len: Length of continuation in tokens, or max new tokens for direct prompts
        output_path: Where to save the .refpt file
        top_k: Number of top predictions to capture (default 100)
        num_entries: Number of entries to generate (default 1)
        device: Device to run on (default: cuda if available, else cpu)
        prompt_source: "book", "aime24", "text", or "file"
        chat_template: Render prompt_source with tokenizer.apply_chat_template before generation
        prompt: Prompt text when prompt_source="text"
        prompt_file: Prompt file when prompt_source="file"
        aime24_prompts_file: DeepSeek AIME24 prompts JSON when prompt_source="aime24"
        aime24_prompt_index: Zero-based prompt index in the AIME24 prompts JSON
        fix_mistral_regex: Apply the corrected Mistral tokenizer regex

    Returns:
        Path to saved reference file
    """
    prompt_source = prompt_source.lower()
    if prompt_source not in PROMPT_SOURCE_CHOICES:
        raise ValueError(f"prompt_source must be one of {PROMPT_SOURCE_CHOICES}, got {prompt_source!r}")
    if prompt_source == "book" and prompt_len <= 0:
        raise ValueError(f"prompt_len must be > 0, got {prompt_len}")
    if gen_len <= 0:
        raise ValueError(f"gen_len must be > 0, got {gen_len}")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if num_entries <= 0:
        raise ValueError(f"num_entries must be > 0, got {num_entries}")
    if prompt_source != "book" and num_entries != 1:
        raise ValueError(f"num_entries must be 1 when prompt_source={prompt_source!r}; got {num_entries}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {hf_model_id} on {device}...")
    tokenizer_kwargs = {"trust_remote_code": True}
    if fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, trust_remote_code=True).eval().to(device)

    # Get token IDs
    stop_ids = _generation_stop_ids(tokenizer, model)
    eos_id = stop_ids[0]
    bos_id = tokenizer.bos_token_id
    pad_id = _safe_pad_id(tokenizer, stop_ids)

    # Generate entries
    entries = []
    if prompt_source == "book":
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
    else:
        prompt_text = _resolve_prompt_text(
            prompt_source,
            prompt=prompt,
            prompt_file=prompt_file,
            aime24_prompts_file=aime24_prompts_file,
            aime24_prompt_index=aime24_prompt_index,
        )
        prompt_tokens_list = _chat_or_plain_prompt_tokens(tokenizer, prompt_text, chat_template=chat_template)
        prompt_tokens = torch.tensor(prompt_tokens_list, dtype=torch.long)
        print(
            f"Using {prompt_source} prompt with {len(prompt_tokens)} tokens "
            f"after {'chat-template' if chat_template else 'plain'} tokenization"
        )
        gen_tokens = _generate_continuation_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            gen_len=gen_len,
            device=device,
        )
        print(f"Generated {len(gen_tokens)} continuation tokens")
        entries.append(
            _generate_one_entry(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                top_k=top_k,
                device=device,
            )
        )

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
        description="Generate a readiness-check reference file using batch prefill on book text or a chat prompt."
    )
    parser.add_argument("--hf-model", required=True, help="HuggingFace model id or local path.")
    parser.add_argument(
        "--prompt-source",
        choices=PROMPT_SOURCE_CHOICES,
        default=DEFAULT_PROMPT_SOURCE,
        help=(
            "Prompt source: book slices Tale of Two Cities; aime24 uses DeepSeek's AIME24 prompt JSON; "
            "text uses --prompt; file uses --prompt-file."
        ),
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Render direct prompts with tokenizer.apply_chat_template before generating the HF continuation.",
    )
    parser.add_argument(
        "--fix-mistral-regex",
        action="store_true",
        help="Use the corrected Mistral tokenizer regex when loading the tokenizer.",
    )
    parser.add_argument("--prompt", help="Direct prompt text for --prompt-source=text.")
    parser.add_argument("--prompt-file", type=Path, help="Prompt file for --prompt-source=file.")
    parser.add_argument(
        "--aime24-prompts-file",
        type=Path,
        default=DEFAULT_AIME24_PROMPTS_FILE,
        help=f"DeepSeek AIME24 prompts JSON (default {DEFAULT_AIME24_PROMPTS_FILE}).",
    )
    parser.add_argument(
        "--aime24-prompt-index",
        type=int,
        default=0,
        help="Zero-based DeepSeek AIME24 prompt index to use with --prompt-source=aime24 (default 0).",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=DEFAULT_PROMPT_LEN,
        help=f"Length of book prompt in tokens for --prompt-source=book (default {DEFAULT_PROMPT_LEN}).",
    )
    parser.add_argument(
        "--gen-len",
        type=int,
        default=DEFAULT_GEN_LEN,
        help=f"Length of continuation, or max new HF-generated tokens for direct prompts (default {DEFAULT_GEN_LEN}).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to output .refpt file.")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_K, help=f"Top-K to capture per position (default {DEFAULT_K})."
    )
    parser.add_argument(
        "--num-entries",
        type=int,
        default=1,
        help="Number of entries to generate from different positions in the book; only valid for --prompt-source=book.",
    )
    args = parser.parse_args()

    path = generate_reference(
        hf_model_id=args.hf_model,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        output_path=args.output,
        top_k=args.top_k,
        num_entries=args.num_entries,
        prompt_source=args.prompt_source,
        chat_template=args.chat_template,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        aime24_prompts_file=args.aime24_prompts_file,
        aime24_prompt_index=args.aime24_prompt_index,
        fix_mistral_regex=args.fix_mistral_regex,
    )
    print(f"Reference saved to: {path}")


if __name__ == "__main__":
    _main()
