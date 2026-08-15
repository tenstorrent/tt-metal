# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The shared qualitative suite through the **live vLLM server**, prompt-correct.

``$qualitative-check`` requires the prompt format the checkpoint declares, and
this tokenizer has a non-empty ``chat_template``: the model is chat/instruct, so
the verdict has to come from chat-rendered prompts.  The readiness runner's own
``qualitative`` stage sends the suite through ``/v1/completions`` as raw text,
which for this checkpoint is *continuation stress coverage* and is kept as such;
this script is the arm the verdict is read from.

Comparability is the point, so nothing here re-renders anything.  The prompts are
the **exact token ids** the full-model stage recorded in
``doc/full_model/qualitative/qualitative_prompts.json`` and posted to the HF
control, and they are sent to the server as token ids rather than as text --
``/v1/completions`` accepts a list of ints -- so the serving arm, the full-model
TT arm, the datatype-sweep TT arm and the HF control all ran the same input.  (The
rendered system message embeds the current date, so re-rendering today would move
one token and turn every divergence metric into a measure of the calendar.)

Artifacts under ``doc/vllm_integration/qualitative/``:

* ``qualitative_prompt_format.json`` -- the prompt-format decision plus the
  serving endpoint and generation parameters;
* ``qualitative_prompts.json`` -- the pinned rendered prompts and token ids;
* ``qualitative_vllm_chat.json`` -- greedy completions, in the shared schema;
* ``vllm_qualitative_outputs.json`` -- greedy and sampled completions in the
  schema ``models/common/readiness_check/check_degenerate_output.py`` scans;
* ``qualitative_comparison_chat.json`` -- mechanical HF-vs-serving comparison,
  computed by the full-model stage's own ``compare()``;
* ``qualitative_vllm_vs_datatype_sweep.json`` -- prompt-by-prompt diff against the
  immediately preceding stage's TT text, which is what says whether serving
  changed the model's output at all.

Usage::

    python doc/vllm_integration/bench/qualitative_vllm.py --server-url http://localhost:8000
    python doc/vllm_integration/bench/qualitative_vllm.py --compare
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

FULL_MODEL = ROOT / "doc/full_model/qualitative"
SWEEP = ROOT / "doc/datatype_sweep/qualitative"
OUT = ROOT / "doc/vllm_integration/qualitative"
HARNESS = ROOT / "doc/full_model/bench/qualitative.py"


def _full_model_harness():
    """The full-model stage's own comparison code, loaded by path.

    Re-implementing ``compare()`` here would mean the serving verdict was computed
    by different arithmetic from the verdict it is compared against.
    """
    spec = importlib.util.spec_from_file_location("_muse_glimmer_qualitative", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def say(*args) -> None:
    print(*args, flush=True)


def compare(mode: str = "chat") -> int:
    harness = _full_model_harness()
    harness.OUT = OUT
    return harness.compare(mode)


def diff_against_datatype_sweep(mode: str = "chat") -> int:
    new_path = OUT / f"qualitative_tt_{mode}.json"
    prev_path = SWEEP / f"qualitative_tt_{mode}.json"
    if not (new_path.is_file() and prev_path.is_file()):
        say(f"need both {new_path} and {prev_path}")
        return 2
    new = {item["id"]: item for item in json.loads(new_path.read_text())}
    prev = {item["id"]: item for item in json.loads(prev_path.read_text())}
    rows = []
    for key in sorted(new):
        a, b = new[key]["token_ids"], prev.get(key, {}).get("token_ids", [])
        first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        rows.append(
            {
                "id": key,
                "vllm_tokens": len(a),
                "datatype_sweep_tokens": len(b),
                "identical": a == b,
                "first_divergence": -1 if first_diff is None else first_diff,
                "vllm_head": new[key]["completion"][:160],
                "datatype_sweep_head": prev.get(key, {}).get("completion", "")[:160],
            }
        )
        say(f"VS_SWEEP {rows[-1]['id']} identical={rows[-1]['identical']} first_div={rows[-1]['first_divergence']}")
    (OUT / f"qualitative_vllm_vs_datatype_sweep_{mode}.json").write_text(json.dumps(rows, indent=2) + "\n")
    identical = sum(1 for row in rows if row["identical"])
    say(f"VS_SWEEP_OK prompts={len(rows)} identical={identical}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--hf-model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--mode", default="chat")
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Artifact directory (default doc/vllm_integration/qualitative).  A variant arm -- "
            "e.g. the --async-scheduling overlap validation -- must pass its own directory so it "
            "cannot overwrite the stage's committed non-overlap evidence."
        ),
    )
    args = parser.parse_args()

    global OUT
    if args.out_dir:
        OUT = pathlib.Path(args.out_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.compare:
        for name in (f"qualitative_hf_{args.mode}.json",):
            source = FULL_MODEL / name
            if source.is_file():
                shutil.copy2(source, OUT / name)
        rc = compare(args.mode)
        rc |= diff_against_datatype_sweep(args.mode)
        return rc

    import openai
    from transformers import AutoTokenizer

    harness = _full_model_harness()
    snapshot = harness.resolve_snapshot(args.hf_model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)

    pinned = FULL_MODEL / "qualitative_prompts.json"
    if not pinned.is_file():
        raise SystemExit(f"the pinned prompt set is missing: {pinned}")
    rendered = json.loads(pinned.read_text())
    shutil.copy2(pinned, OUT / "qualitative_prompts.json")

    format_meta = json.loads((FULL_MODEL / "qualitative_prompt_format.json").read_text())
    format_meta.update(
        {
            "stage": "vllm_integration",
            "arm": "vllm-serving",
            "endpoint": f"{args.server_url.rstrip('/')}/v1/completions",
            "endpoint_note": (
                "the pinned chat-rendered token ids are posted directly as `prompt`, so the server runs "
                "exactly the input the HF control and the previous TT stages ran; sending text would let "
                "the server re-render the date-dependent system message and move the comparison"
            ),
            "prompts_pinned_from": str(pinned.relative_to(REPO)),
            "generation": {
                "greedy": {"temperature": 0.0, "max_tokens": args.max_new_tokens},
                "sampled": {"temperature": 0.7, "top_p": 0.9, "max_tokens": args.max_new_tokens},
            },
        }
    )
    (OUT / "qualitative_prompt_format.json").write_text(json.dumps(format_meta, indent=2) + "\n")

    client = openai.OpenAI(base_url=f"{args.server_url.rstrip('/')}/v1", api_key="dummy")
    results = []
    degenerate_schema = []
    for item in rendered:
        started = time.perf_counter()
        greedy = client.completions.create(
            model=args.hf_model,
            prompt=item["token_ids"],
            max_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        greedy_text = greedy.choices[0].text
        sampled_text = (
            client.completions.create(
                model=args.hf_model,
                prompt=item["token_ids"],
                max_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            .choices[0]
            .text
        )
        greedy_ids = [int(i) for i in tokenizer.encode(greedy_text, add_special_tokens=False)]
        results.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "token_ids": greedy_ids,
                "completion": greedy_text,
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        degenerate_schema.append(
            {
                "prompt": item["prompt"],
                "greedy_completion": greedy_text,
                "sampled_completion": sampled_text,
            }
        )
        say(f"QUAL vllm {item['id']} {len(greedy_ids)} tokens in {results[-1]['seconds']}s")
        say(f"QUAL vllm {item['id']} greedy  :: {greedy_text[:300]!r}")
        say(f"QUAL vllm {item['id']} sampled :: {sampled_text[:300]!r}")

    (OUT / f"qualitative_tt_{args.mode}.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "vllm_qualitative_outputs.json").write_text(json.dumps(degenerate_schema, indent=2) + "\n")
    say(f"QUAL_OK arm=vllm mode={args.mode} prompts={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
