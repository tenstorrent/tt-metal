# SPDX-License-Identifier: Apache-2.0
"""Per-document audit of an lm-eval --log_samples run.

The blocked Stage 11 run deleted its raw samples and reasoned from aggregates
only, which is how four cap-exhausted requests went unclassified. This prints
what those aggregates hide: per document, how long the answer ran, whether it
terminated, whether a boxed choice was extracted, and whether it was right.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True, help="samples_*.jsonl from lm_eval")
    ap.add_argument("--tokenizer", default="google/gemma-4-26B-A4B-it")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # lm-eval's writer leaves raw newlines inside strings, so decode as a stream
    # of concatenated JSON objects rather than one-object-per-line.
    text_blob = args.samples.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(strict=False)
    rows = []
    index = 0
    while index < len(text_blob):
        while index < len(text_blob) and text_blob[index] in " \n\r\t":
            index += 1
        if index >= len(text_blob):
            break
        obj, index = decoder.raw_decode(text_blob, index)
        rows.append(obj)
    print(f"{len(rows)} documents\n")
    header = f"{'doc':>3} {'gen_tok':>8} {'chars':>7} {'boxed':>6} {'target':>6} {'flex':>5} {'strict':>6}  tail"
    print(header)
    print("-" * len(header))
    total_correct = 0
    lengths = []
    for row in rows:
        doc_id = row.get("doc_id")
        resp = row.get("resps") or row.get("filtered_resps") or [[""]]
        text = resp[0][0] if isinstance(resp[0], list) else resp[0]
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        lengths.append(n_tok)
        boxed = BOXED.findall(text)
        target = row.get("target")
        flex = row.get("exact_match,flexible-extract", row.get("exact_match"))
        strict = row.get("exact_match,strict-match")
        total_correct += 1 if flex == 1 else 0
        tail = text[-60:].replace("\n", " ")
        print(
            f"{doc_id:>3} {n_tok:>8} {len(text):>7} {str(boxed[-1] if boxed else '-'):>6} "
            f"{str(target):>6} {str(flex):>5} {str(strict):>6}  {tail!r}"
        )
    print(f"\nflexible-extract correct: {total_correct}/{len(rows)}")
    if lengths:
        srt = sorted(lengths)
        print(f"generated tokens: min {srt[0]}  median {srt[len(srt)//2]}  max {srt[-1]}")
        print(f"documents at/over 32760 tokens (cap-exhausted): {sum(1 for n in lengths if n >= 32760)}")


if __name__ == "__main__":
    main()
