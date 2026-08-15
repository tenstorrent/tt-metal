# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Determinism and non-aligned-prompt evidence through the live vLLM server.

Three things the sampling suite does not pin, and one the stage contract asks for
by name:

1. **run-to-run** — the same prompt sent twice, greedily, must produce the same
   token ids. A traced decode whose token feedback or position advance is racy
   fails here first.
2. **cross-batch-position** — ``n`` copies of the same prompt sent *concurrently*
   must all produce that same sequence. The decode batch is 32 rows wide and each
   row indexes its own cache slot and page-table row, so a row-indexing or
   page-table bug shows as one row differing from the rest while each row is
   individually self-consistent.
3. **standalone baseline** — the same pinned prompts run through the standalone
   generator by the previous stage (``doc/datatype_sweep/qualitative``). Serving
   should reproduce it; where it diverges, *when* it diverges is the diagnostic.
4. **non-aligned prompt lengths** — prompt lengths that divide neither the 32-row
   tile, the 64-token page, nor the 8192-token prefill chunk must be ordinary
   requests. Sent as explicit token-id prompts so the length is exact rather than
   whatever a tokenizer happened to produce.

Usage::

    python doc/vllm_integration/bench/determinism_vllm.py --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

OUT = ROOT / "doc/vllm_integration"
SWEEP = ROOT / "doc/datatype_sweep/qualitative"
FULL_MODEL = ROOT / "doc/full_model/qualitative"

#: Lengths chosen so that none is a multiple of 32 (tile), 64 (page) or 8192
#: (prefill chunk), spanning single-token, sub-page, multi-page, multi-chunk and
#: a length just past a chunk boundary.
NON_ALIGNED_LENGTHS = (1, 37, 127, 129, 1023, 2049, 4097, 8193, 12345)

#: Control tokens the standalone harnesses keep (they decode with
#: ``skip_special_tokens=False``) and the OpenAI API drops. Removed from the
#: baseline before the text comparison so the two arms are compared like for like.
SPECIAL_TOKENS_STRIPPED_BY_THE_API = ("<|message|>",)


def say(*args) -> None:
    print(*args, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--hf-model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--out", type=pathlib.Path, default=OUT / "determinism_vllm.json")
    args = parser.parse_args()

    import openai
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "doc/full_model/bench"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_muse_glimmer_qualitative", ROOT / "doc/full_model/bench/qualitative.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    snapshot = harness.resolve_snapshot(args.hf_model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)

    client = openai.OpenAI(base_url=f"{args.server_url.rstrip('/')}/v1", api_key="dummy", timeout=1800)

    def greedy_ids(prompt_ids, max_tokens=None):
        text = (
            client.completions.create(
                model=args.hf_model,
                prompt=prompt_ids,
                max_tokens=max_tokens or args.max_tokens,
                temperature=0.0,
            )
            .choices[0]
            .text
        )
        return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)], text

    report: dict = {"server_url": args.server_url, "max_tokens": args.max_tokens}

    pinned = json.loads((FULL_MODEL / "qualitative_prompts.json").read_text())
    probe = pinned[0]

    # --- 1. run to run -------------------------------------------------------
    first_ids, first_text = greedy_ids(probe["token_ids"])
    second_ids, _ = greedy_ids(probe["token_ids"])
    report["run_to_run"] = {
        "prompt_id": probe["id"],
        "prompt_tokens": len(probe["token_ids"]),
        "identical": first_ids == second_ids,
        "tokens": first_ids,
        "second_tokens": second_ids,
        "text_head": first_text[:200],
    }
    say(f"RUN_TO_RUN identical={report['run_to_run']['identical']}")

    # --- 2. cross batch position --------------------------------------------
    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futures = [pool.submit(greedy_ids, probe["token_ids"]) for _ in range(args.concurrent)]
        concurrent = [future.result()[0] for future in futures]
    distinct = {tuple(ids) for ids in concurrent}
    report["cross_batch_position"] = {
        "concurrent_requests": args.concurrent,
        "all_identical": len(distinct) == 1,
        "distinct_outputs": len(distinct),
        "matches_single_request": all(ids == first_ids for ids in concurrent),
        "outputs": concurrent,
    }
    say(
        f"CROSS_BATCH all_identical={report['cross_batch_position']['all_identical']} "
        f"matches_single={report['cross_batch_position']['matches_single_request']}"
    )

    # --- 3. standalone baseline ---------------------------------------------
    baseline_path = SWEEP / "qualitative_tt_chat.json"
    if baseline_path.is_file():
        baseline = {item["id"]: item for item in json.loads(baseline_path.read_text())}
        entry = baseline.get(probe["id"])
        if entry is not None:
            # Compare TEXT, with the baseline's special tokens removed, rather than
            # re-encoded token ids.  The standalone harness decodes with
            # ``skip_special_tokens=False`` while the OpenAI API strips them, so the
            # served text is missing control tokens such as ``<|message|>`` that the
            # baseline text carries.  Re-encoding both and diffing ids therefore reports
            # a divergence at the first control token and every id after it — a property
            # of the two detokenizer settings, not of the model.  Removing the known
            # control tokens from the baseline makes this like-for-like.
            base_text = entry["completion"]
            for special in SPECIAL_TOKENS_STRIPPED_BY_THE_API:
                base_text = base_text.replace(special, "")
            width = min(len(first_text), len(base_text))
            diverge = next((i for i in range(width) if first_text[i] != base_text[i]), None)
            report["standalone_baseline"] = {
                "source": str(baseline_path.relative_to(REPO)),
                "comparison": "characters, baseline stripped of API-invisible special tokens",
                "special_tokens_stripped": list(SPECIAL_TOKENS_STRIPPED_BY_THE_API),
                "compared_chars": width,
                "identical_over_common_prefix": diverge is None,
                "first_char_divergence": -1 if diverge is None else diverge,
            }
            say(
                "BASELINE identical_over_common_prefix="
                f"{report['standalone_baseline']['identical_over_common_prefix']} "
                f"over {width} chars"
            )

    # --- 4. non-aligned prompt lengths --------------------------------------
    rows = []
    for length in NON_ALIGNED_LENGTHS:
        ids = [(1000 + i) % 200000 for i in range(length)]
        try:
            out_ids, text = greedy_ids(ids, max_tokens=8)
            rows.append(
                {
                    "prompt_len": length,
                    "divides_tile32": length % 32 == 0,
                    "divides_page64": length % 64 == 0,
                    "divides_chunk8192": length % 8192 == 0,
                    "ok": True,
                    "output_tokens": len(out_ids),
                    "text_head": text[:80],
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"prompt_len": length, "ok": False, "error": str(exc)[:400]})
        say(f"NON_ALIGNED len={length} ok={rows[-1]['ok']}")
    report["non_aligned_prompt_lengths"] = rows

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    say(f"DETERMINISM_OK -> {args.out}")
    failures = [
        not report["run_to_run"]["identical"],
        not report["cross_batch_position"]["all_identical"],
        any(not row["ok"] for row in rows),
    ]
    return 1 if any(failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
