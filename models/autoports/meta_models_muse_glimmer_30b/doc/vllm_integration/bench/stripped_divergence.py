# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Like-for-like token divergence, with the API-stripped control token removed.

`qualitative.py`'s `compare()` diffs raw token-id sequences, and its docstring makes
`first_divergence_from_hf` the wrapper-bug tripwire: "divergence at token 0-2 is a
wrapper bug, late divergence with both texts coherent is ordinary numerics".

The serving arm cannot be read on that scale directly. The OpenAI API strips special
tokens, so `<|message|>` (id 200023) is present in the HF and standalone arms and
absent from the served text; every served completion therefore diverges at position
1-2 no matter how well it matches. `compare()` reads as tripped on 6/6 for a reason
that has nothing to do with the model.

This removes that one token from both sides and recomputes the divergence, which is
the comparison the tripwire is actually asking for. Two pairs:

  * **served vs the datatype-sweep standalone arm** — the same weights, the same
    precision policy, the same greedy decode, so this one should be *exact*; and
  * **served vs the HF control** — greedy TT and greedy HF diverge eventually
    (bf16 against a different reduction order), so here the number to read is
    *where*.

Offline: reads committed JSON, opens no device and contacts no server.

    python doc/vllm_integration/bench/stripped_divergence.py
    python doc/vllm_integration/bench/stripped_divergence.py --check   # exit 1 on regression
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/vllm_integration/
MODEL_DOC = ROOT.parent  # doc/
QUAL = ROOT / "qualitative"
OUT = QUAL / "qualitative_stripped_divergence_chat.json"

# `<|message|>`. The only special token that appears mid-completion in this
# checkpoint's harmony-style channel format, and the only one the API strips.
MESSAGE_TOKEN_ID = 200023


def _load(path: pathlib.Path) -> dict:
    return {item["id"]: item for item in json.loads(path.read_text())}


def _strip(ids: list[int]) -> list[int]:
    return [i for i in ids if i != MESSAGE_TOKEN_ID]


def _first_divergence(a: list[int], b: list[int]):
    """Index of the first differing token over the common prefix, or None."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the stripped comparison regresses")
    args = ap.parse_args()

    served = _load(QUAL / "qualitative_tt_chat.json")
    hf = _load(QUAL / "qualitative_hf_chat.json")
    sweep = _load(MODEL_DOC / "datatype_sweep/qualitative/qualitative_tt_chat.json")
    sweep_vs_hf = {
        r["id"]: r
        for r in json.loads((MODEL_DOC / "datatype_sweep/qualitative/qualitative_comparison_chat.json").read_text())
    }

    rows = []
    for key in sorted(served):
        s = _strip(served[key]["token_ids"])
        h = _strip(hf[key]["token_ids"])
        w = _strip(sweep[key]["token_ids"])
        rows.append(
            {
                "id": key,
                "served_tokens": len(served[key]["token_ids"]),
                "message_token_in_served": MESSAGE_TOKEN_ID in served[key]["token_ids"],
                "message_token_in_standalone": MESSAGE_TOKEN_ID in sweep[key]["token_ids"],
                "raw_first_divergence_vs_standalone": _first_divergence(
                    served[key]["token_ids"], sweep[key]["token_ids"]
                ),
                "stripped_first_divergence_vs_standalone": _first_divergence(s, w),
                "stripped_identical_to_standalone_over_common_prefix": s == w[: len(s)],
                "raw_first_divergence_vs_hf": _first_divergence(served[key]["token_ids"], hf[key]["token_ids"]),
                "stripped_first_divergence_vs_hf": _first_divergence(s, h),
                # The datatype-sweep stage ran its own HF comparison with the control
                # token PRESENT on both sides, so this column is free of any stripping
                # and shows whether a divergence predates serving.
                "standalone_stage_first_divergence_vs_hf": sweep_vs_hf.get(key, {}).get("first_divergence_from_hf"),
            }
        )

    vs_sweep_exact = all(r["stripped_identical_to_standalone_over_common_prefix"] for r in rows)
    hf_div = {r["id"]: r["stripped_first_divergence_vs_hf"] for r in rows}
    early_vs_hf = {k: v for k, v in hf_div.items() if v is not None and v <= 2}
    # An early divergence is only a serving finding if the standalone stage did not
    # already record one for the same prompt.
    serving_introduced = {
        k: v for k, v in early_vs_hf.items() if (sweep_vs_hf.get(k, {}).get("first_divergence_from_hf") or 99) > 2
    }

    report = {
        "message_token_id": MESSAGE_TOKEN_ID,
        "note": (
            "compare()'s raw first_divergence is 1-2 on every prompt purely because the API strips "
            "<|message|>; these columns remove it from both sides and are the like-for-like numbers"
        ),
        "rows": rows,
        "verdict": {
            "served_matches_standalone_exactly_when_stripped": vs_sweep_exact,
            "stripped_first_divergence_vs_hf": hf_div,
            "prompts_diverging_from_hf_at_token_le_2": sorted(early_vs_hf),
            "serving_introduced_early_hf_divergences": sorted(serving_introduced),
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    for r in rows:
        print(
            f"  {r['id']}: vs standalone stripped={r['stripped_first_divergence_vs_standalone']} "
            f"exact={r['stripped_identical_to_standalone_over_common_prefix']} | "
            f"vs HF raw={r['raw_first_divergence_vs_hf']} stripped={r['stripped_first_divergence_vs_hf']} "
            f"(standalone stage vs HF: {r['standalone_stage_first_divergence_vs_hf']})"
        )
    print(f"\nserved == standalone with <|message|> removed, all prompts: {vs_sweep_exact}")
    print(f"stripped divergence vs HF: {hf_div}")
    print(f"early (<=2) vs HF: {sorted(early_vs_hf)}; of those, serving-introduced: {sorted(serving_introduced)}")
    print(f"-> {OUT}")

    ok = vs_sweep_exact and not serving_introduced
    return 1 if (args.check and not ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
