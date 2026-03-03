#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer


@dataclass
class CollapseHit:
    output_path: Path
    prompt_index: int
    prompt_text: str
    generation_text: str
    token_count: int
    collapse_start: int
    collapse_kind: str
    collapse_detail: str


def _find_collapse(
    tokens: list[int], run_threshold: int, window: int, window_ratio: float
) -> tuple[int | None, str, str]:
    # Run-length collapse.
    best_run_start = None
    best_run_len = 0
    run_start = 0
    for i in range(1, len(tokens) + 1):
        if i == len(tokens) or tokens[i] != tokens[i - 1]:
            run_len = i - run_start
            if run_len >= run_threshold and (best_run_start is None or run_start < best_run_start):
                best_run_start = run_start
                best_run_len = run_len
                break
            run_start = i
    if best_run_start is not None:
        return best_run_start, "run", f"len={best_run_len}"

    # Window collapse: most common token dominates the window.
    if len(tokens) >= window:
        for start in range(0, len(tokens) - window + 1):
            window_tokens = tokens[start : start + window]
            counts = Counter(window_tokens)
            top_token, top_count = counts.most_common(1)[0]
            if top_count >= int(window * window_ratio):
                return start, "window", f"top_token={top_token} count={top_count}"

    return None, "", ""


def _iter_output_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(f"**/{pattern}"))


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare collapse reproduction prompts.")
    p.add_argument("--outputs-root", type=Path, default=Path("/data/deepseek/outputs"))
    p.add_argument("--pattern", type=str, default="demo_aime24_full_output.json")
    p.add_argument("--model-path", type=Path, default=Path("/data/deepseek/DeepSeek-R1-0528"))
    p.add_argument("--out", type=Path, default=Path("/data/deepseek/prompts/demo_collapse_repro.json"))
    p.add_argument("--users", type=int, default=64)
    p.add_argument("--pre-collapse-tokens", type=int, default=128)
    p.add_argument("--run-threshold", type=int, default=64)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--window-ratio", type=float, default=0.9)
    args = p.parse_args()

    output_files = _iter_output_files(args.outputs_root, args.pattern)
    if not output_files:
        raise SystemExit(f"No output files found under {args.outputs_root} matching {args.pattern}")

    tokenizer = load_tokenizer(str(args.model_path))
    hits: list[CollapseHit] = []

    for output_path in output_files:
        try:
            data = json.loads(output_path.read_text())
        except Exception as exc:
            logger.warning(f"Failed to read {output_path}: {exc}")
            continue
        generations = data.get("generations", [])
        for idx, gen in enumerate(generations):
            prompt_text = gen.get("prompt") or ""
            gen_text = gen.get("text") or ""
            if not gen_text:
                continue
            tokens = tokenizer.encode(gen_text, add_special_tokens=False)
            collapse_start, kind, detail = _find_collapse(tokens, args.run_threshold, args.window, args.window_ratio)
            if collapse_start is None:
                continue
            hits.append(
                CollapseHit(
                    output_path=output_path,
                    prompt_index=idx,
                    prompt_text=prompt_text,
                    generation_text=gen_text,
                    token_count=len(tokens),
                    collapse_start=collapse_start,
                    collapse_kind=kind,
                    collapse_detail=detail,
                )
            )

    if not hits:
        raise SystemExit("No collapse candidates found with current thresholds.")

    hits.sort(key=lambda h: h.collapse_start)
    hit = hits[0]

    tokens = tokenizer.encode(hit.generation_text, add_special_tokens=False)
    cut_idx = max(0, hit.collapse_start - args.pre_collapse_tokens)
    prefix_tokens = tokens[:cut_idx]
    prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
    prompt_out = hit.prompt_text + prefix_text

    out_data = [{"prompt": prompt_out} for _ in range(args.users)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))

    # Emit a short report to stdout for review.
    snippet_start = max(0, hit.collapse_start - 40)
    snippet_end = min(len(tokens), hit.collapse_start + 40)
    snippet_text = tokenizer.decode(tokens[snippet_start:snippet_end], skip_special_tokens=True)

    print("=== Collapse Candidate ===")
    print(f"file: {hit.output_path}")
    print(f"prompt_index: {hit.prompt_index}")
    print(f"collapse_start_token: {hit.collapse_start}")
    print(f"collapse_kind: {hit.collapse_kind} ({hit.collapse_detail})")
    print(f"cut_idx (pre-collapse): {cut_idx}")
    print(f"prompt_out_tokens: {len(tokenizer.encode(prompt_out, add_special_tokens=False))}")
    print("--- snippet around collapse ---")
    print(snippet_text)
    print("--- output file ---")
    print(args.out)


if __name__ == "__main__":
    main()
