# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Line up a mesh long-context run with its HF CPU reference.

    python models/demos/glm53_flash/reference/compare_to_tt.py \
        --tt generated/glm53_flash_long_context/mesh8x4_N16380.json \
        --ref bringup/references/glm53_flash/NEEDLE_16384

``--tt`` is the JSON ``tests/test_mesh_long_context.py`` writes (keys ``isl``, ``generated``, ``text``,
``needle_found``); ``--ref`` is a ``NEEDLE_<N>`` directory (or its ``reference.json``) from
``hf_long_reference.py``. Reports the prompt length, the first-token match, how many leading greedy tokens agree
(the greedy chains diverge for good after the first mismatch, so the prefix length is the figure of merit), the
positional agreement over the common length, both texts and the needle flags. Exit status 0 when the first tokens
agree, 1 otherwise, 2 when the prompt lengths differ (the two sides did not run the same prompt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_ref(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "reference.json"
    return json.loads(path.read_text())


def compare(tt: dict, ref: dict, tok=None) -> dict:
    tt_gen = [int(t) for t in tt["generated"]]
    hf_gen = [int(t) for t in ref["generated"]]
    n = min(len(tt_gen), len(hf_gen))
    prefix = 0
    while prefix < n and tt_gen[prefix] == hf_gen[prefix]:
        prefix += 1
    positional = sum(1 for a, b in zip(tt_gen, hf_gen) if a == b)
    out = {
        "isl_tt": int(tt["isl"]),
        "isl_hf": int(ref["isl"]),
        "same_prompt_length": int(tt["isl"]) == int(ref["isl"]),
        "first_tt": tt_gen[0] if tt_gen else None,
        "first_hf": hf_gen[0] if hf_gen else None,
        "first_match": bool(tt_gen and hf_gen and tt_gen[0] == hf_gen[0]),
        "matching_prefix": prefix,
        "positional_matches": positional,
        "common_length": n,
        "len_tt": len(tt_gen),
        "len_hf": len(hf_gen),
        "text_tt": tt.get("text", ""),
        "text_hf": ref.get("text", ""),
        "needle_tt": bool(tt.get("needle_found")),
        "needle_hf": bool(ref.get("needle_found")),
        "generated_tt": tt_gen,
        "generated_hf": hf_gen,
    }
    if tok is not None and prefix < n:
        out["first_divergence"] = {
            "index": prefix,
            "tt": tok.decode([tt_gen[prefix]]),
            "hf": tok.decode([hf_gen[prefix]]),
            "hf_top5_at_step": (ref.get("steps") or [{}])[prefix].get("top5_ids")
            if prefix < len(ref.get("steps", []))
            else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--tt",
        type=Path,
        required=True,
        help="mesh test JSON (generated/glm53_flash_long_context/mesh<R>x<C>_N<S>.json)",
    )
    ap.add_argument("--ref", type=Path, required=True, help="NEEDLE_<N> directory or its reference.json")
    ap.add_argument("--snapshot", type=Path, default=None, help="tokenizer for decoding the divergence (optional)")
    ap.add_argument("--json", action="store_true", help="print the comparison as JSON")
    args = ap.parse_args()
    tt = json.loads(args.tt.read_text())
    ref = load_ref(args.ref)
    tok = None
    if args.snapshot is not None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(args.snapshot))
    r = compare(tt, ref, tok)
    if args.json:
        print(json.dumps(r, indent=1))
    else:
        print(
            f"prompt length: TT {r['isl_tt']} vs HF {r['isl_hf']} {'(same)' if r['same_prompt_length'] else '(DIFFERENT PROMPT)'}"
        )
        print(
            f"first token:   TT {r['first_tt']} vs HF {r['first_hf']} -> {'MATCH' if r['first_match'] else 'MISMATCH'}"
        )
        print(
            f"greedy tokens: {r['matching_prefix']} leading tokens agree, {r['positional_matches']}/{r['common_length']} "
            f"positions agree (TT {r['len_tt']} tokens, HF {r['len_hf']} tokens)"
        )
        if "first_divergence" in r:
            d = r["first_divergence"]
            print(
                f"first divergence at step {d['index']}: TT {d['tt']!r} vs HF {d['hf']!r} (HF top-5 ids {d['hf_top5_at_step']})"
            )
        print(f"TT ids: {r['generated_tt']}\nHF ids: {r['generated_hf']}")
        print(f"TT text: {r['text_tt']!r}\nHF text: {r['text_hf']!r}")
        print(f"needle:  TT {'found' if r['needle_tt'] else 'missing'}, HF {'found' if r['needle_hf'] else 'missing'}")
    sys.exit(0 if r["first_match"] else (2 if not r["same_prompt_length"] else 1))


if __name__ == "__main__":
    main()
