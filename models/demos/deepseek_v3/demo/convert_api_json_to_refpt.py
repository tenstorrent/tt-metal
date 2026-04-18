#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Convert a DeepSeek API batch-results JSON into a compact multi-prompt .refpt
file that can be committed to the repo and consumed by
``test_demo_teacher_forced.py`` without any runtime JSON parsing.

Usage
-----
::

    python convert_api_json_to_refpt.py \\
        --input  results-512.json \\
        --output deepseek_r1_teacher_forcing_512.refpt \\
        --model-path /data/deepseek/DeepSeek-R1-0528-dequantized

The resulting ``.refpt`` stores only the tensors the test needs (prompt token
IDs, ground-truth generated token IDs, top-5 predictions) — no raw text — so
it is typically 2–4 MB instead of the ~93 MB JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm


def _api_token_to_id(tokenizer, token_str: str, token_bytes: Optional[list[int]]) -> int:
    """Resolve an API log-prob token string to a vocab ID."""
    try:
        vocab_id = tokenizer.convert_tokens_to_ids(token_str)
        if vocab_id is not None and vocab_id != getattr(tokenizer, "unk_token_id", None):
            return int(vocab_id)
    except Exception:
        pass

    def _try(text: str) -> Optional[int]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    tid = _try(token_str)
    if tid is not None:
        return tid

    if token_bytes:
        try:
            decoded = bytes(token_bytes).decode("utf-8", errors="replace")
            tid = _try(decoded)
            if tid is not None:
                return tid
        except Exception:
            pass

    return tokenizer.unk_token_id or 0


def _build_entries_from_api_json(
    results_json: Path,
    tokenizer,
    prompt_count: int,
    max_new_tokens: int,
) -> list[dict]:
    """Parse the API JSON and build per-prompt entry dicts."""
    with results_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "entries" in raw:
        items = raw["entries"]
    elif isinstance(raw, dict) and "results" in raw:
        items = raw["results"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(
            f"Unexpected JSON structure in {results_json}. "
            f"Expected a dict with 'entries' or 'results' key, or a list. "
            f"Got keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}"
        )

    if len(items) < prompt_count:
        raise ValueError(
            f"JSON has {len(items)} results but {prompt_count} were requested."
        )

    entries: list[dict] = []
    for item in tqdm(items[:prompt_count], desc="Building entries"):
        prompt_text: str = item.get("prompt", "")
        generated_text: str = item.get("generated_text", "")
        steps: list[dict] = item.get("top_logprobs_per_step", [])

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        if steps:
            generated_tokens = [
                _api_token_to_id(tokenizer, step.get("token", ""), step.get("bytes"))
                for step in steps[:max_new_tokens]
            ]
            if not generated_text:
                generated_text = "".join(step.get("token", "") for step in steps[:max_new_tokens])
        else:
            generated_tokens = tokenizer.encode(generated_text, add_special_tokens=False)[:max_new_tokens]

        if not generated_tokens:
            raise ValueError(
                f"Prompt has 0 generated tokens after processing. "
                f"prompt={prompt_text[:80]!r} generated_text={generated_text[:80]!r}"
            )

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long)
        gen_t = torch.tensor([generated_tokens], dtype=torch.long)
        total_len = len(prompt_ids) + len(generated_tokens)

        top5 = torch.zeros(total_len, 5, dtype=torch.long)
        for step_i, step in enumerate(steps[:max_new_tokens]):
            pos = len(prompt_ids) + step_i
            top_lps = step.get("top_logprobs", [])
            for k, lp_entry in enumerate(top_lps[:5]):
                tok_id = _api_token_to_id(
                    tokenizer, lp_entry.get("token", ""), lp_entry.get("bytes")
                )
                top5[pos, k] = tok_id

        ref_t = torch.cat([prompt_t, gen_t], dim=1)

        entries.append(
            {
                "reference_tokens": ref_t,
                "prompt_tokens": prompt_t,
                "generated_tokens": gen_t,
                "top5_tokens": top5,
                "tf_prompt_len": len(prompt_ids),
                "prompt": prompt_text,
                "decoded_generated_text": generated_text,
            }
        )

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert API JSON to .refpt")
    parser.add_argument("--input", required=True, help="Path to results JSON")
    parser.add_argument("--output", required=True, help="Output .refpt path")
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF model directory (for tokenizer)",
    )
    parser.add_argument(
        "--num-entries",
        type=int,
        default=512,
        help="Number of prompts to include (default 512)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max generated tokens per prompt (default 128)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    entries = _build_entries_from_api_json(
        results_json=Path(args.input),
        tokenizer=tokenizer,
        prompt_count=args.num_entries,
        max_new_tokens=args.max_new_tokens,
    )

    e0 = entries[0]
    payload = {
        "format_version": "multi_prompt_v1",
        "num_prompts": len(entries),
        "max_new_tokens": args.max_new_tokens,
        "token_ids_meta": {
            "eos_id": getattr(tokenizer, "eos_token_id", None),
            "bos_id": getattr(tokenizer, "bos_token_id", None),
            "pad_id": getattr(tokenizer, "pad_token_id", None),
        },
        "entries": entries,
        "reference_tokens": e0["reference_tokens"],
        "prompt_tokens": e0["prompt_tokens"],
        "generated_tokens": e0["generated_tokens"],
        "top5_tokens": e0["top5_tokens"],
        "tf_prompt_len": e0["tf_prompt_len"],
        "prompt": e0["prompt"],
        "decoded_generated_text": e0["decoded_generated_text"],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(entries)} entries to {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
