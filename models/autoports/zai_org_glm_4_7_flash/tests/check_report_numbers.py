# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail when a load-bearing figure in the stage report has no matching artifact.

Six review rounds found stale numbers in ``doc/full_model/README.md`` and
``work_log.md``, and each round they were fixed by hand, which is how the next
round found more. This closes the loop mechanically: every check below takes a
value out of a committed JSON artifact, formats it the way the documents quote
it, and requires that exact string to be present. A figure that moved in a
re-run and was not propagated fails here instead of in the next review.

It is deliberately not a parser of the prose. Anchoring on "this string must
appear" is what makes it useful: rewording a sentence is free, changing a
number without regenerating the sentence is not.

    python models/autoports/zai_org_glm_4_7_flash/tests/check_report_numbers.py

Exit code 0 means every checked figure matches its artifact. The sweep runs
this last, after every artifact it reads has been written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC = MODEL_DIR / "doc" / "full_model"
README = DOC / "README.md"
WORK_LOG = DOC / "work_log.md"
CONTRACT = MODEL_DIR / "doc" / "context_contract.json"


def _at(data, dotted):
    """``a.b.0.c`` -> ``data['a']['b'][0]['c']``."""
    node = data
    for part in dotted.split("."):
        node = node[int(part)] if part.isdigit() else node[part]
    return node


#: (artifact, key path, format spec, where it must appear).
#: ``fmt`` is applied to the artifact value; the result must occur verbatim.
CHECKS = [
    # ---- headline
    ("perf.json", "ttft_ms.prompt_128_warmed", "{:.1f} ms", "readme"),
    ("perf.json", "prefill_tokens_per_s.prompt_128", "{:.1f} tok/s prefill", "readme"),
    ("perf.json", "decode_ms_per_token.traced_model_only_no_sampling", "{:.3f} ms/token", "readme"),
    ("perf.json", "decode_tokens_per_s_per_user.traced_model_only_no_sampling", "{:.2f} t/s/u", "readme"),
    ("perf.json", "decode_ms_per_token.token_out_incl_readback", "{:.3f} ms/token", "readme"),
    ("perf.json", "decode_tokens_per_s_per_user.token_out_incl_readback", "{:.2f} t/s/u", "readme"),
    ("perf.json", "end_to_end.prompt_128_generate_128_s", "{:.3f} s", "readme"),
    ("perf.json", "end_to_end.tokens_per_s", "{:.2f} tok/s", "readme"),
    # ---- the accounting block adds up to the token-out figure
    ("perf.json", "layer_stack_lower_bound.sampling_ms", "{:.3f} ms", "readme"),
    ("perf.json", "layer_stack_lower_bound.token_readback_ms", "{:.3f} ms", "readme"),
    ("perf.json", "ttft_ms.request_boundary_reset_ms", "{:.1f} ms", "readme"),
    # ---- first-use cost
    ("first_use_ttft.json", "rows.1.first_request.trace_recapture_ms", "{:.1f} ms", "readme"),
    ("first_use_ttft.json", "rows.1.first_request.new_programs_compiled", "**{:.0f}** of them", "readme"),
    ("first_use_ttft.json", "rows.0.first_request.harness_style_ttft_ms", "{:.1f} ms", "readme"),
    ("first_use_ttft.json", "rows.0.second_request.harness_style_ttft_ms", "{:.1f} ms", "readme"),
    # ---- accuracy
    ("accuracy.json", "prefill.top1", "| {:.3f} |", "readme"),
    ("accuracy.json", "teacher_forcing.top1", "| {:.3f} |", "readme"),
    # ---- capacity
    ("capacity.json", "gib.total_resident", "{:.3f}", "readme"),
    ("capacity.json", "weights_bytes.total", "{:,}", "readme"),
    ("capacity.json", "kv_cache_bytes_batch1_full_context", "{:,}", "readme"),
    # ---- full context
    ("full_context.json", "prompt_len", "**{:.0f}-token** prefill", "readme"),
    ("full_context.json", "prefill_tokens_per_s", "{:.1f} tok/s", "readme"),
    ("full_context.json", "decode_ms_per_token", "{:.1f} ms/token", "readme"),
    ("full_context.json", "needle.reach_positions", "**{:.0f} positions**", "readme"),
    # ---- compile cost, both arms
    ("compile_cost.json", "build.prefill_programs_warmed_s", "| {:.1f} s |", "readme"),
    ("compile_cost.json", "prefill.prompt_3000.first_call_ms", "| {:.1f} ms |", "readme"),
    ("compile_cost.json", "prefill.prompt_3000.first_minus_repeat_mean_ms", "+{:.1f} ms", "readme"),
    ("compile_cost_warm.json", "prefill.prompt_3000.first_call_ms", "| {:.1f} ms |", "readme"),
    ("compile_cost_warm.json", "prefill.prompt_3000.repeat_mean_ms", "| {:.1f} ms |", "readme"),
    # ---- profiler
    ("perf_report_summary.json", "decode_model.device_us_per_step", "{:.1f} and", "readme"),
    ("perf_report_summary.json", "decode_tokenout.device_us_per_step", "{:.1f} us/step", "readme"),
    ("perf_report_summary.json", "prefill.bound.slow_rows", "**{:.0f} of 110 prefill rows", "readme"),
    ("perf_report_summary.json", "prefill.bound.slow_pct_of_window", "{:.1f}% of\n   that window", "readme"),
    ("perf_report_summary.json", "decode_tokenout.bound.slow_rows", "{:.0f} of 1600\n   token-out rows", "readme"),
    ("perf_reduced_decode.json", "traced_model_only_ms", "{:.3f} /", "readme"),
    ("perf_reduced_decode.json", "traced_token_out_ms", "{:.3f} ms wall clock", "readme"),
    # ---- decode position ladder
    ("decode_position_scaling.json", "rows.0.traced_ms", "{:.2f} ->", "readme"),
    ("decode_position_scaling.json", "rows.6.traced_ms", "-> {:.2f} ms per traced step", "readme"),
    # ---- degeneracy
    ("degenerate_check.json", "measured.0.trigram_loop_fraction", "trigram loop fraction {:.4f}", "readme"),
    # ---- trace allocation
    ("trace_alloc.json", "arms.2.unsafe_total", "{:.0f} (16 per trace)", "worklog"),
    ("trace_alloc.json", "arms.2.program_cache_entries", "{:.0f} entries against", "worklog"),
    ("trace_alloc.json", "arms.2.program_cache_entries_at_capture", "{:.0f} at capture", "worklog"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-dir", type=Path, default=DOC)
    args = ap.parse_args()

    docs = {
        "readme": (args.doc_dir / "README.md").read_text(),
        "worklog": (args.doc_dir / "work_log.md").read_text(),
        "contract": CONTRACT.read_text(),
    }
    cache: dict[str, object] = {}
    failures = []
    checked = 0
    for artifact, keys, fmt, where in CHECKS:
        path = args.doc_dir / artifact
        if artifact not in cache:
            cache[artifact] = json.loads(path.read_text())
        try:
            value = _at(cache[artifact], keys)
        except (KeyError, IndexError, TypeError) as exc:
            failures.append(f"{artifact}:{keys} not in the artifact ({type(exc).__name__})")
            continue
        expected = fmt.format(value)
        checked += 1
        if expected not in docs[where]:
            failures.append(f"{where} is missing {expected!r} (from {artifact}:{keys} = {value!r})")

    # The capability contract must agree with the artifacts it summarises.
    contract = json.loads(CONTRACT.read_text())
    fce = contract["full_model"]["full_context_evidence"]
    full = json.loads((args.doc_dir / "full_context.json").read_text())
    for key, artifact_key in (
        ("prompt_len", "prompt_len"),
        ("prefill_tokens_per_s", "prefill_tokens_per_s"),
        ("decode_ms_per_token_at_202751", "decode_ms_per_token"),
        ("last_decode_position", "last_decode_position"),
    ):
        checked += 1
        if fce[key] != full[artifact_key]:
            failures.append(
                f"context_contract full_context_evidence.{key} = {fce[key]!r} but "
                f"full_context.json {artifact_key} = {full[artifact_key]!r}"
            )
    cap = json.loads((args.doc_dir / "capacity.json").read_text())
    for key, artifact_value in (
        ("resident_total", cap["total_resident_bytes"]),
        ("kv_cache_batch1_202752_bf8", cap["kv_cache_bytes_batch1_full_context"]),
        ("weights_plus_persistent_scratch", cap["weights_bytes"]["total"]),
    ):
        checked += 1
        if contract["full_model"]["dram_budget_bytes"][key] != artifact_value:
            failures.append(f"context_contract dram_budget_bytes.{key} disagrees with capacity.json")

    print(f"checked {checked} figures against {len(set(a for a, *_ in CHECKS))} artifacts")
    for line in failures:
        print("MISMATCH:", line)
    if failures:
        print(f"{len(failures)} report figure(s) have no matching artifact value")
        return 1
    print("every checked figure matches its artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
