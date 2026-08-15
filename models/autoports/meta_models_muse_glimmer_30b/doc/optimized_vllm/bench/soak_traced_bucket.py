# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Soak the traced prefill bucket with prompts that actually land in it, and read the text.

The first attempt at this (``soak_1bucket/``) was worthless and the round-2 stage review
caught it: a prefill trace is keyed by the *exact* padded row count, the shipped bucket is
padded 128, and every prompt in that soak's qualitative rounds was 7-79 tokens -- padded 32,
64 or 96. All 84 completions took the eager path. The traced path's only exposure was the
benchmark, whose text nobody reads.

This builds chat prompts whose **rendered** length lands in the traced bucket, asserts that
before sending anything, then serves them for several rounds and reads the output. The
failure this is looking for is not subtle once you can see it -- served text decays into
U+FFFD replacement characters -- but it is invisible in a benchmark, which is exactly how it
survived the first sweep.

Two properties are checked per round:

* ``replacement_char_fraction`` per completion, against the same thresholds
  ``check_degenerate_output.py`` uses (critical 0.10, advisory 0.02);
* **round-to-round stability**: greedy completions of the same prompt must be identical
  across rounds. Corruption here is deterministic -- the 20-bucket arms produced
  byte-identical corrupt strings on two different servers -- so a *change* between rounds is
  the onset, and it is a sharper signal than any threshold.

Usage (against an already-running server)::

    python doc/optimized_vllm/bench/soak_traced_bucket.py \\
        --server-url http://localhost:8000 --rounds 8 \\
        --out doc/optimized_vllm/soak_traced_bucket.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[5]
MODEL_DOC = pathlib.Path(__file__).resolve().parents[2]

#: User messages chosen so the chat-rendered prompt lands in the traced bucket. Ordinary
#: questions, not filler: the point is to read the answers and see that they stay English.
QUESTIONS = [
    "I am writing a short introduction to machine learning for a general audience. "
    "Please explain, in plain language and without equations, what the difference is "
    "between supervised and unsupervised learning, and give one everyday example of each "
    "that a reader with no technical background would recognise immediately.",
    "Imagine you are helping a colleague who has never used version control. Explain what "
    "a commit is, what a branch is, and why someone would want to work on a branch instead "
    "of directly on the main line of development. Keep it concrete and use a small running "
    "example rather than abstract definitions throughout.",
    "A friend asks why the sky is blue during the day but red near sunset. Give a clear "
    "physical explanation that mentions scattering and the path light takes through the "
    "atmosphere, at the level of a curious adult who last studied physics in school and "
    "would like to actually understand it rather than memorise it.",
    "Write a short, practical checklist that a first-time traveller should work through in "
    "the week before an international flight. Cover documents, money, health, luggage and "
    "communications, and keep each item to a single actionable sentence so the whole thing "
    "fits comfortably on one printed page for reference.",
    "Explain the difference between weather and climate to someone who uses the two words "
    "interchangeably. Include why a single unusually cold week does not tell you much about "
    "long-term trends, and suggest one simple analogy that makes the distinction stick in "
    "the mind of a listener who is not scientifically trained at all.",
    "Describe, step by step, how you would go about debugging a program that produces the "
    "wrong answer only occasionally and only on large inputs. Focus on the reasoning and the "
    "order of operations rather than on any particular language or tool, and say what you "
    "would rule out first and why that ordering matters.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--hf-model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--bucket", type=int, default=128, help="the padded row count that must be hit")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    import openai
    from transformers import AutoTokenizer

    from models.autoports.meta_models_muse_glimmer_30b.tt.model import weights_snapshot_dir

    tokenizer = AutoTokenizer.from_pretrained(str(weights_snapshot_dir(args.hf_model)), local_files_only=True)
    lo, hi = args.bucket - 31, args.bucket

    prompts = []
    for i, question in enumerate(QUESTIONS):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True, tokenize=True
        )
        # Same unwrapping the full-model stage's renderer does: this tokenizer returns a
        # BatchEncoding, and a nested list inside it.
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if len(ids) and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        prompts.append({"id": f"s{i}", "token_ids": [int(t) for t in ids], "question": question})

    lengths = [len(p["token_ids"]) for p in prompts]
    padded = [((n + 31) // 32) * 32 for n in lengths]
    in_bucket = [p == args.bucket for p in padded]
    print(f"[soak] prompt lengths {lengths} -> padded {padded}; bucket {args.bucket} needs {lo}-{hi}", flush=True)
    if not all(in_bucket):
        # Refuse to produce evidence about a path the prompts do not take. This is the
        # exact failure this script exists to replace.
        print("[soak] FATAL: not every prompt lands in the traced bucket; this would soak the eager path", flush=True)
        args.out.write_text(
            json.dumps(
                {
                    "status": "prompts_not_in_bucket",
                    "bucket": args.bucket,
                    "prompt_tokens": lengths,
                    "padded": padded,
                },
                indent=2,
            )
            + "\n"
        )
        return 3

    client = openai.OpenAI(base_url=f"{args.server_url.rstrip('/')}/v1", api_key="x")
    report: dict = {
        "bucket": args.bucket,
        "prompt_tokens": lengths,
        "padded": padded,
        "all_prompts_in_traced_bucket": True,
        "rounds": [],
        "max_tokens": args.max_tokens,
    }
    baseline: dict[str, str] = {}
    worst = 0.0
    unstable: list[dict] = []

    for rnd in range(args.rounds):
        row: dict = {"round": rnd, "completions": []}
        for prompt in prompts:
            text = (
                client.completions.create(
                    model=args.hf_model,
                    prompt=prompt["token_ids"],
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                )
                .choices[0]
                .text
            )
            frac = text.count("�") / max(1, len(text))
            worst = max(worst, frac)
            stable = baseline.setdefault(prompt["id"], text) == text
            if not stable:
                unstable.append(
                    {"round": rnd, "id": prompt["id"], "first": baseline[prompt["id"]][:120], "now": text[:120]}
                )
            row["completions"].append(
                {
                    "id": prompt["id"],
                    "replacement_char_fraction": round(frac, 4),
                    "stable_vs_round0": stable,
                    "head": text[:100],
                }
            )
        row["worst_replacement_char_fraction"] = round(
            max(c["replacement_char_fraction"] for c in row["completions"]), 4
        )
        row["all_stable"] = all(c["stable_vs_round0"] for c in row["completions"])
        report["rounds"].append(row)
        print(
            f"[soak] round {rnd}: worst_replacement={row['worst_replacement_char_fraction']} "
            f"all_stable={row['all_stable']}",
            flush=True,
        )

    report["generations"] = args.rounds * len(prompts)
    report["worst_replacement_char_fraction"] = round(worst, 4)
    report["all_stable"] = not unstable
    report["instability"] = unstable[:10]
    report["status"] = "ok" if (worst <= 0.02 and not unstable) else "degraded"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[soak] {report['generations']} generations through the traced bucket; "
        f"worst_replacement={report['worst_replacement_char_fraction']} stable={report['all_stable']} "
        f"-> {report['status']}",
        flush=True,
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
