# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Does the on-device penalty stage agree with vLLM's own sampler, on the real model?

`penalty_shard_boundary_probe.py` checks the stage against a torch transcription
of `apply_penalties` at the tensor level. This checks the *whole served path*
against the **reference implementation actually installed in this environment**,
on the 48-layer model, through HTTP.

The trick is that the TT plugin gives us a reference sampler for free. A request
that sets `min_p` is routed to **host sampling** by
`vllm_tt_plugin/platform.py` — vLLM builds logits, and vLLM's own
`model_executor/layers/utils.py::apply_penalties` applies the penalties. A
request that does not set `min_p` takes this port's traced on-device path. So the
same prompt and the same penalty, sent twice, gives:

    device-sampled text   vs   vLLM-reference-sampled text

and at temperature 0 both are deterministic, so the comparison is byte equality
rather than a similarity score. `min_p=0.01` with greedy decoding cannot change
the winner (the nucleus always keeps the top token), and the zero-penalty control
rows check exactly that rather than assuming it.

**Why this probe exists.** Two of the six `test_tt_penalties` tests
(`TestPresencePenalty`) fail for this checkpoint, and the question is whether the
stage is wrong or the test's premise does not hold here. This settles it: if the
reference sampler produces the *same* unchanged output for the same request, the
test would fail against the reference too and the stage is not what is failing.

Needs a live server; does not open a device itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: A request setting ``min_p`` is routed to host sampling by the plugin, so this
#: is the switch between "our on-device stage" and "vLLM's reference sampler"
#: while everything else about the request stays identical.
HOST_SAMPLING_SWITCH = {"min_p": 0.01}

CASES = [
    # (label, prompt, max_tokens, penalty kwargs)
    ("control_no_penalty", "a b c a b c a b c", 40, {}),
    ("presence_neg2", "a b c a b c a b c", 40, {"presence_penalty": -2.0}),
    ("presence_pos2", "a b c a b c a b c", 40, {"presence_penalty": 2.0}),
    ("frequency_0p3", "a b c a b c a b c", 40, {"frequency_penalty": 0.3}),
    ("frequency_0p5", "a b c a b c a b c", 40, {"frequency_penalty": 0.5}),
    ("frequency_1p0", "a b c a b c a b c", 40, {"frequency_penalty": 1.0}),
    ("frequency_2p0_a", "a a a a a a a a a", 15, {"frequency_penalty": 2.0}),
    ("repetition_2p0", "a a a a a a a a a", 10, {"repetition_penalty": 2.0}),
    ("repetition_0p5", "a a a a a a a a a", 10, {"repetition_penalty": 0.5}),
    ("repetition_1p5_abc", "a b c a b c a b c", 20, {"repetition_penalty": 1.5}),
    (
        "mixed_all_three",
        "a b c a b c a b c",
        20,
        {"repetition_penalty": 1.2, "frequency_penalty": 0.5, "presence_penalty": 1.0},
    ),
]


def complete(url: str, model: str, prompt: str, max_tokens: int, extra: dict) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        **extra,
    }
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = json.load(response)
    if "choices" not in payload:
        raise RuntimeError(f"server rejected {body!r}: {payload!r}")
    return payload["choices"][0]["text"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--json", default=str(HERE / "penalty_serving_parity_probe.json"))
    args = parser.parse_args()

    rows = []
    for label, prompt, max_tokens, extra in CASES:
        device = complete(args.url, args.model, prompt, max_tokens, extra)
        reference = complete(args.url, args.model, prompt, max_tokens, {**extra, **HOST_SAMPLING_SWITCH})
        rows.append(
            {
                "case": label,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "penalties": extra,
                "device_sampled": device,
                "vllm_reference_sampled": reference,
                "identical": device == reference,
            }
        )
        print(f"[{'ok ' if device == reference else 'DIFF'}] {label}: {device!r}")

    baseline = next(r for r in rows if r["case"] == "control_no_penalty")["device_sampled"]
    results = {
        "url": args.url,
        "model": args.model,
        "host_sampling_switch": HOST_SAMPLING_SWITCH,
        "cases": rows,
        "all_identical_to_vllm_reference": all(r["identical"] for r in rows),
        # The presence question, stated as data: does the *reference* sampler
        # move off the unpenalised output when presence_penalty is at either end
        # of vLLM's permitted [-2, 2] range?
        "presence_reference_also_unchanged": all(
            r["vllm_reference_sampled"] == baseline for r in rows if r["case"].startswith("presence_")
        ),
        # ... while frequency, on the same prompt, does move it -- so this is not
        # "penalties do nothing", it is "presence is capped too low for this
        # checkpoint's logit gap on this prompt".
        "frequency_changes_output_on_same_prompt": any(
            r["device_sampled"] != baseline for r in rows if r["case"].startswith("frequency_")
        ),
        "repetition_changes_output": any(
            r["device_sampled"] != r["prompt"] and r["penalties"].get("repetition_penalty", 1.0) != 1.0 for r in rows
        ),
    }
    results["passed"] = bool(
        results["all_identical_to_vllm_reference"]
        and results["frequency_changes_output_on_same_prompt"]
        and results["presence_reference_also_unchanged"]
    )
    Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "cases"}, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
