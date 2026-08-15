# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which part of the sampling suite corrupts a server that has prefill traces resident?

The optimized-vLLM arm with 20 resident prefill traces served coherent English before
the plugin sampling suite and replacement characters after it; the same binary with
prefill tracing off was clean on both sides.  Warming every sampling *mode* at warmup
did not fix it, so the trigger is not the late trace capture.

ttnn states the rule this is testing: "Allocating device buffers is unsafe due to the
existence of an active trace ... buffers allocated when a trace is active must have a
lifetime that ends before the trace is executed" (``allocator.cpp:113-126``).  A
captured trace's intermediates are freed after capture but their addresses stay baked
into the replay, so a *long-lived* buffer allocated afterwards that lands in that range
is overwritten the next time the trace runs.  Twenty 52-layer prefill traces poison a
far larger address range than the decode and sampling traces alone, which is why the
same suite was survivable before.

This bisects which test file allocates such a buffer: run one file, then ask the model
a pinned prompt and compare against the reference answer taken before any of them ran.

Usage (against an already-running server)::

    python doc/optimized_vllm/bench/localize_corruption.py \
        --server-url http://localhost:8000 --out doc/optimized_vllm/corruption_localization.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[5]
MODEL_DOC = pathlib.Path(__file__).resolve().parents[2]

#: Run order.  Cheapest and most-suspected first: ``test_logprobs`` is the file that
#: turns on the log-probs calculator, ``test_seeding_and_variety`` is the one that
#: activates request seeds (which make the sampler run untraced), and
#: ``test_host_only_params`` forces the host-sampling route with its full-vocab gather.
TEST_FILES = (
    "test_config.py",
    "test_build_logprobs_from_topk.py",
    "test_logprobs.py",
    "test_seeding_and_variety.py",
    "test_tt_penalties.py",
    "test_host_only_params.py",
    "test_request_isolation.py",
    "test_structured_output_dp1.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--hf-model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--tests-dir", default="/home/ttuser/dev/vllm-tt-plugin/tests/tt")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import openai

    client = openai.OpenAI(base_url=f"{args.server_url.rstrip('/')}/v1", api_key="x")
    pinned = json.loads((MODEL_DOC / "full_model/qualitative/qualitative_prompts.json").read_text())
    probe_ids = [int(t) for t in pinned[0]["token_ids"]]

    def answer() -> str:
        return (
            client.completions.create(
                model=args.hf_model, prompt=probe_ids, max_tokens=args.max_tokens, temperature=0.0
            )
            .choices[0]
            .text
        )

    reference = answer()
    print(f"[localize] reference: {reference!r}", flush=True)
    report: dict = {"reference": reference, "steps": []}
    tests_dir = pathlib.Path(args.tests_dir)

    for name in TEST_FILES:
        target = tests_dir / name
        if not target.is_file():
            report["steps"].append({"file": name, "skipped": "missing"})
            continue
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(target),
            "-q",
            f"--tt-server-url={args.server_url}",
            f"--tt-model-name={args.hf_model}",
            f"--tt-max-num-seqs={args.max_num_seqs}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        after = answer()
        healthy = after == reference
        report["steps"].append(
            {
                "file": name,
                "pytest_rc": proc.returncode,
                "pytest_tail": proc.stdout.strip().splitlines()[-1:] if proc.stdout else [],
                "answer": after,
                "matches_reference": healthy,
            }
        )
        print(f"[localize] {name}: rc={proc.returncode} healthy={healthy} answer={after[:60]!r}", flush=True)
        if not healthy:
            report["first_corrupting_file"] = name
            break

    report.setdefault("first_corrupting_file", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[localize] first_corrupting_file={report['first_corrupting_file']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
