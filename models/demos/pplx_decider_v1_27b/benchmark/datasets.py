# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Frozen, seeded 200-example subsets of three model-card benchmarks, as app decisions.

Each example becomes one ``choice`` decision for ``TTDecider.predict(state, question)``:
``state`` = the context (sentence / passage), ``question.instructions`` = the task question,
``question.criteria`` = the answer options. The snapshot's own ``decision_messages`` renders it
(``"State:\\n..."``, ``"Question:\\n..."``, ``"Options:\\nA: <key>[: <description>]"``).

The model card's numbers were measured through the Perplexity API with converters from an external
package (``nimble.datasets.public_benchmarks``, referenced by ``autojev/data.py`` but not shipped in
the snapshot), so the card's prompt protocol is unknown. The templates below are one fixed template
per benchmark, chosen before any TT accuracy was observed and not tuned afterwards.

| benchmark | source (HF dataset @ revision) | config / split | population |
|---|---|---|---|
| WinoGrande | allenai/winogrande | winogrande_xl / validation (test labels are hidden) | 1267 |
| FinancialPhraseBank | takala/financial_phrasebank (zip in the repo) | Sentences_50Agree.txt (the only split) | 4846 |
| Belebele | facebook/belebele | eng_Latn / test (the only split) | 900 |

Selection: ``random.Random(SEED).sample(range(population), 200)``, sorted by source index.
Ids, a sha256 of the newline-joined id list and a sha256 of each subset file go to the manifest.

Usage (CPU only, no device)::

    python -m models.demos.pplx_decider_v1_27b.benchmark.datasets --out $STAGE11/subsets
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

SEED = 20260920  # the snapshot's autojev/data.py SEED; not tuned
SUBSET_SIZE = 200
MAX_TOKENS = 8192  # app max_length; longer prompts are rejected by tt/model.bucket_for, never truncated

SOURCES = {
    "winogrande": {
        "repo": "allenai/winogrande",
        "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "config": "winogrande_xl",
        "split": "validation",
    },
    "financial_phrasebank": {
        "repo": "takala/financial_phrasebank",
        "revision": "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964",
        "config": "sentences_50agree",
        "split": "train (only split)",
        "file": "data/FinancialPhraseBank-v1.0.zip",
        "member": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    },
    "belebele": {
        "repo": "facebook/belebele",
        "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
        "config": "eng_Latn",
        "split": "test",
    },
}

# Card figures (README of the snapshot): pplx-decider-v1-27b and its Qwen3.8-27B base, in %.
CARD = {
    "winogrande": {"pplx_decider_v1_27b": 83.30, "qwen3_8_27b_base": 73.10},
    "financial_phrasebank": {"pplx_decider_v1_27b": 84.18, "qwen3_8_27b_base": 75.68},
    "belebele": {"pplx_decider_v1_27b": 94.00, "qwen3_8_27b_base": 93.20},
}

WINOGRANDE_INSTRUCTIONS = "Which option correctly fills the blank (_) in the sentence?"
FPB_INSTRUCTIONS = "What is the sentiment of this financial news sentence from the point of view of an investor?"
FPB_CRITERIA = {
    "negative": "Bad news for investors; likely to have a negative effect on the stock price",
    "neutral": "Neither good nor bad news for investors",
    "positive": "Good news for investors; likely to have a positive effect on the stock price",
}


def winogrande_example(index: int, raw: dict) -> dict:
    options = [raw["option1"], raw["option2"]]
    if options[0] == options[1]:
        raise ValueError(f"winogrande {index}: identical options")
    return {
        "id": f"winogrande_xl/validation/{index}",
        "benchmark": "winogrande",
        "state": raw["sentence"],
        "question": {"type": "choice", "instructions": WINOGRANDE_INSTRUCTIONS, "criteria": {o: None for o in options}},
        "label": options[int(raw["answer"]) - 1],
    }


def fpb_example(index: int, sentence: str, label: str) -> dict:
    if label not in FPB_CRITERIA:
        raise ValueError(f"fpb {index}: label {label!r}")
    return {
        "id": f"sentences_50agree/{index}",
        "benchmark": "financial_phrasebank",
        "state": sentence,
        "question": {"type": "choice", "instructions": FPB_INSTRUCTIONS, "criteria": dict(FPB_CRITERIA)},
        "label": label,
    }


def belebele_example(index: int, raw: dict) -> dict:
    options = [raw[f"mc_answer{i}"] for i in range(1, 5)]
    # Option text is the key (rendered "A: <text>"). Duplicate texts within one item would collapse
    # keys; fall back to numbered keys "1".."4" with the text as description for that item only.
    if len(set(options)) == len(options):
        criteria, label, style = {o: None for o in options}, options[int(raw["correct_answer_num"]) - 1], "text"
    else:
        criteria, label, style = {str(i + 1): o for i, o in enumerate(options)}, raw["correct_answer_num"], "numbered"
    return {
        "id": f"belebele/eng_Latn/test/{index}",
        "benchmark": "belebele",
        "key_style": style,
        "upstream": {"link": raw["link"], "question_number": raw["question_number"]},
        "state": raw["flores_passage"],
        "question": {"type": "choice", "instructions": raw["question"], "criteria": criteria},
        "label": label,
    }


def load_population(name: str) -> list[dict]:
    src = SOURCES[name]
    if name == "financial_phrasebank":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(src["repo"], src["file"], repo_type="dataset", revision=src["revision"])
        with zipfile.ZipFile(path) as z:
            text = io.TextIOWrapper(z.open(src["member"]), encoding="iso-8859-1").read()
        rows = []
        for i, line in enumerate(l for l in text.splitlines() if l.strip()):
            sentence, label = line.rsplit("@", 1)
            rows.append(fpb_example(i, sentence.strip(), label.strip()))
        return rows
    from datasets import load_dataset

    ds = load_dataset(src["repo"], src["config"], split=src["split"], revision=src["revision"])
    make = winogrande_example if name == "winogrande" else belebele_example
    return [make(i, raw) for i, raw in enumerate(ds)]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build(out: Path) -> dict:
    from models.demos.pplx_decider_v1_27b.reference.decision_prompts import AppTokenizer
    from models.demos.pplx_decider_v1_27b.tt.model import BUCKETS

    tok = AppTokenizer()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "subset_size": SUBSET_SIZE, "max_tokens": MAX_TOKENS, "benchmarks": {}}
    for name in SOURCES:
        population = load_population(name)
        picked = sorted(random.Random(SEED).sample(range(len(population)), SUBSET_SIZE))
        rows = []
        for i in picked:
            ex = population[i]
            n = len(tok.input_ids({"state": ex["state"], "question": ex["question"]}))
            ex["seq_len"] = n
            ex["bucket"] = next((b for b in BUCKETS if n <= b), None)  # None: rejected (> 8192)
            ex["count"] = len(ex["question"]["criteria"])
            rows.append(ex)
        ids = [r["id"] for r in rows]
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (out / f"{name}.jsonl").write_text(body)
        lens = sorted(r["seq_len"] for r in rows)
        manifest["benchmarks"][name] = {
            **SOURCES[name],
            "population": len(population),
            "label_distribution": dict(Counter(str(r["label"]) for r in rows))
            if name == "financial_phrasebank"
            else None,
            "numbered_key_fallbacks": sum(r.get("key_style") == "numbered" for r in rows),
            "ids_sha256": sha256("\n".join(ids)),
            "subset_file_sha256": sha256(body),
            "ids": ids,
            "seq_len": {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]},
            "buckets": dict(Counter(str(r["bucket"]) for r in rows)),
            "rejected_over_8192": sum(r["bucket"] is None for r in rows),
            "card": CARD[name],
        }
        print(
            f"{name}: population {len(population)}, ids sha256 {manifest['benchmarks'][name]['ids_sha256'][:16]}, "
            f"seq_len {lens[0]}..{lens[-1]}, buckets {manifest['benchmarks'][name]['buckets']}"
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_subsets(subset_dir: Path) -> list[dict]:
    """All frozen examples, benchmark order winogrande, financial_phrasebank, belebele."""
    rows = []
    for name in SOURCES:
        rows += [json.loads(l) for l in (subset_dir / f"{name}.jsonl").read_text().splitlines() if l.strip()]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
