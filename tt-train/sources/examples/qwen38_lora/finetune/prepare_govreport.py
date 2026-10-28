#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tokenize GovReport (ccdv/govreport-summarization) for long-context LoRA SFT.

Each example is ``PROMPT.format(report) + summary + eos``. The loss is taken on
the summary and eos only, so the model has to read the report to lower it.
Examples longer than ``--max-tokens`` are dropped rather than truncated (a cut
report no longer supports its summary), as are short reports
(``--min-report-tokens``) so the set exercises long context.

Output: ``<out>/{train,validation}.npz`` with a flat ``tokens`` array plus
``offsets`` (start of each example) and ``prompt_lens``.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PROMPT = "Read the following government report and write a summary of it.\n\nReport:\n{report}\n\nSummary:\n"

_tok = None


def _init(model_path: str) -> None:
    global _tok
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def _encode(example: dict) -> tuple[list[int], int, int]:
    prompt = _tok(PROMPT.format(report=example["report"].strip()))["input_ids"]
    summary = _tok(example["summary"].strip())["input_ids"] + [_tok.eos_token_id]
    report_len = len(prompt)
    return prompt + summary, len(prompt), report_len


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=None, help="dataset saved with save_to_disk; downloaded from the Hub if omitted")
    p.add_argument("--model-path", required=True, help="Qwen3.8-27B checkpoint dir (for its tokenizer)")
    p.add_argument("--out", required=True, help="output dir for train.npz / validation.npz")
    p.add_argument("--max-tokens", type=int, default=8193, help="seq_len + 1 (inputs and shifted targets)")
    p.add_argument("--min-report-tokens", type=int, default=3000)
    args = p.parse_args()

    from datasets import load_dataset, load_from_disk

    ds = load_from_disk(args.raw) if args.raw else load_dataset("ccdv/govreport-summarization")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with Pool(32, initializer=_init, initargs=(args.model_path,)) as pool:
        for split in ("train", "validation"):
            encoded = pool.map(_encode, list(ds[split]), chunksize=64)
            lens = np.array([len(t) for t, _, _ in encoded])
            keep = [(t, pl) for t, pl, rl in encoded if len(t) <= args.max_tokens and rl >= args.min_report_tokens]
            print(
                f"{split}: {len(encoded)} examples, tokens p10/p50/p90 = "
                f"{np.percentile(lens, 10):.0f}/{np.percentile(lens, 50):.0f}/{np.percentile(lens, 90):.0f}; "
                f"kept {len(keep)} with {args.min_report_tokens} <= prompt and total <= {args.max_tokens}"
            )
            kl = np.array([len(t) for t, _ in keep])
            sl = np.array([len(t) - pl for t, pl in keep])
            print(
                f"  kept: total tokens mean {kl.mean():.0f} (min {kl.min()}, max {kl.max()}), "
                f"summary tokens mean {sl.mean():.0f}"
            )
            offsets = np.cumsum([0] + [len(t) for t, _ in keep])
            np.savez(
                out / f"{split}.npz",
                tokens=np.concatenate([np.asarray(t, dtype=np.uint32) for t, _ in keep]),
                offsets=offsets.astype(np.int64),
                prompt_lens=np.array([pl for _, pl in keep], dtype=np.int64),
            )


if __name__ == "__main__":
    main()
