# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The shared qualitative suite, re-run on the selected precision config.

Same harness as the full-model and optimized-full-model stages --
``doc/full_model/bench/qualitative.py``, loaded by path -- with artifacts under
``doc/datatype_sweep/qualitative/``.  The ``tt`` arm calls
``build_generator(ROOT, mesh, ...)`` with no precision knobs, so it constructs
whatever ``doc/datatype_sweep/selected_precision_config.json`` currently selects:
running this after the selection is what proves the *selected* policy generates
the text, not a harness-specific build.

The HF control is the full-model stage's committed one (CPU bf16, 128 tokens per
prompt, same checkpoint, tokenizer, prompt set and generation parameters -- the
copied ``qualitative_prompt_format.json`` is the proof), which
``$qualitative-check`` accepts for a regression comparison.

``--reuse-hf-control`` also **pins the prompt token ids to the control's own**.
The suite's system message embeds the current date, so re-rendering it on a later
day produces a one-token-different prompt; a reused control plus a re-rendered TT
prompt would put the two arms on different inputs and make
``first_divergence_from_hf`` a measure of the calendar rather than of precision.

``--vs-optimized-full-model`` additionally diffs this stage's TT completions
against the optimized-full-model stage's, prompt by prompt, which is the
comparison that says whether the precision change is visible in the text.

Usage::

    python doc/datatype_sweep/bench/qualitative.py --arm tt --reuse-hf-control
    python doc/datatype_sweep/bench/qualitative.py --arm compare
    python doc/datatype_sweep/bench/qualitative.py --vs-optimized-full-model
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

PREV_FULL = ROOT / "doc/full_model/qualitative"
PREV_OPT = ROOT / "doc/optimized_full_model/qualitative"
OUT = ROOT / "doc/datatype_sweep/qualitative"


def diff_against_optimized_full_model(mode: str = "chat") -> int:
    """Prompt-by-prompt diff of this stage's TT text against the previous stage's."""
    new_path = OUT / f"qualitative_tt_{mode}.json"
    prev_path = PREV_OPT / f"qualitative_tt_{mode}.json"
    if not (new_path.is_file() and prev_path.is_file()):
        print(f"need both {new_path} and {prev_path}", flush=True)
        return 2
    new = {item["id"]: item for item in json.loads(new_path.read_text())}
    prev = {item["id"]: item for item in json.loads(prev_path.read_text())}
    rows = []
    for key in sorted(new):
        a, b = prev.get(key, {}), new[key]
        first_char = next((i for i, (x, y) in enumerate(zip(a.get("completion", ""), b["completion"])) if x != y), None)
        if first_char is None and len(a.get("completion", "")) != len(b["completion"]):
            first_char = min(len(a.get("completion", "")), len(b["completion"]))
        first_token = next((i for i, (x, y) in enumerate(zip(a.get("token_ids", []), b["token_ids"])) if x != y), None)
        rows.append(
            {
                "id": key,
                "identical_text": a.get("completion") == b["completion"],
                "identical_tokens": a.get("token_ids") == b["token_ids"],
                "first_char_divergence": first_char,
                "first_token_divergence": first_token,
                "prev_len": len(a.get("completion", "")),
                "new_len": len(b["completion"]),
                "prev_snippet": (a.get("completion", "") or "")[:160],
                "new_snippet": b["completion"][:160],
            }
        )
        print(
            f"VSPREV {key} identical_tokens={rows[-1]['identical_tokens']} " f"first_token_divergence={first_token}",
            flush=True,
        )
    (OUT / f"qualitative_tt_vs_optimized_full_model_{mode}.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"VSPREV_OK prompts={len(rows)}", flush=True)
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--vs-optimized-full-model" in sys.argv:
        sys.argv.remove("--vs-optimized-full-model")
        return diff_against_optimized_full_model()

    path = ROOT / "doc/full_model/bench/qualitative.py"
    spec = importlib.util.spec_from_file_location("muse_glimmer_full_model_qualitative", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = OUT

    if "--reuse-hf-control" in sys.argv:
        sys.argv.remove("--reuse-hf-control")
        for name in ("qualitative_hf_chat.json", "qualitative_prompt_format.json", "qualitative_prompts.json"):
            source = PREV_FULL / name
            if source.exists() and not (OUT / name).exists():
                shutil.copy2(source, OUT / name)
                module.say(f"QUAL reused the full-model stage control {name}")

        # **Pin the prompts to the control's own token ids.**  The suite's system
        # message embeds the current date, so ``apply_chat_template`` renders a
        # *different* prompt on a different day -- one token, at index 31.  The
        # parent harness re-renders unconditionally, so reusing a control
        # generated on an earlier day and re-rendering here would have the two
        # arms answering one-token-different prompts, which guarantees a
        # divergence regardless of precision and makes
        # ``first_divergence_from_hf`` meaningless. Round 2 of the stage review
        # caught exactly that. Reusing a control means reusing its prompt.
        pinned = json.loads((OUT / "qualitative_prompts.json").read_text())
        original_render = module.render

        def render(tokenizer, prompts, *, mode):
            rendered = original_render(tokenizer, prompts, mode=mode)
            if [item["id"] for item in rendered] != [item["id"] for item in pinned]:
                raise SystemExit(
                    "the reused control's prompt ids do not match this run's suite; the control "
                    "cannot be reused for a different prompt set"
                )
            changed = [item["id"] for item, keep in zip(rendered, pinned) if item["token_ids"] != keep["token_ids"]]
            if changed:
                module.say(
                    f"QUAL pinned {len(changed)} prompt(s) to the reused control's token ids "
                    f"(re-rendering today would have changed {changed}); the two arms run the "
                    "same tokens"
                )
            return pinned

        module.render = render
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
