# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only transition checks for the unintegrated trace-reuse policy."""

import json
from dataclasses import replace
from pathlib import Path

from serving_trace_reuse_plan import CapturedContract, RequestContract, after_prefill, before_prefill


def main():
    request = RequestContract(
        prefill_signature=(32, 128, 128, "native_block8", (32, 4096), "int32"),
        active_slots=(0,),
        sampling_signature=("unseeded", "greedy", 1, 0.0),
        cache_bindings=(("conv", 100), ("recurrent", 200), ("key", 300), ("value", 400)),
        page_table_binding=("generator_owned", 500, (32, 4096)),
    )
    captured = CapturedContract(request, ("model_trace", 1, "sampler_trace", 2, "stable_io"), True)
    cases = []

    def check(name, decision, expected):
        assert decision.action == expected, (name, decision)
        cases.append({"case": name, "action": decision.action, "reason": decision.reason})

    check("cold request", before_prefill(None, request), "release_and_capture")
    check("same shape, slot and owners", before_prefill(captured, request), "preserve_until_prefill_finishes")
    for name, changed in [
        ("new logical/physical prompt shape", replace(request, prefill_signature=(32, 129, 129))),
        ("new active slot", replace(request, active_slots=(7,))),
        ("concurrent active requests", replace(request, active_slots=(0, 1))),
        ("new cache owner", replace(request, cache_bindings=(("key", 301),))),
        ("external page table", replace(request, page_table_binding=("external", 500, (32, 4096)))),
        ("sampling parameters changed", replace(request, sampling_signature=("unseeded", "sampled", 20, 0.9))),
        ("request seed", replace(request, seeded=True)),
        ("penalties", replace(request, penalties=True)),
        ("log probabilities", replace(request, logprobs=True)),
        ("slot remap", replace(request, nonidentity_remap=True)),
    ]:
        check(name, before_prefill(captured, changed), "release_and_capture")

    evidence = dict(
        current_trace_bindings=captured.trace_bindings,
        program_entries_before=700,
        program_entries_after=700,
        allocation_evidence_clear=True,
        inputs_reload_authorized=True,
        request_prefill_complete=True,
    )
    check("validated boundary reload", after_prefill(captured, request, **evidence), "refresh_and_replay")
    for name, change, expected in [
        ("new cached program after prefill", {"program_entries_after": 701}, "release_and_capture"),
        ("surviving buffer unverified", {"allocation_evidence_clear": False}, "release_and_capture"),
        ("sampler trace lost", {"current_trace_bindings": ("model_trace", 1)}, "release_and_capture"),
        ("scheduler inputs stale", {"inputs_reload_authorized": False}, "reject"),
        ("reset slot missing prefill", {"request_prefill_complete": False}, "reject"),
    ]:
        check(name, after_prefill(captured, request, **{**evidence, **change}), expected)

    result = {
        "scope": "Pure-host policy transitions; no TTNN import, device run or production integration.",
        "passed": len(cases),
        "cases": cases,
    }
    target = Path(__file__).parent / "artifacts/serving_trace_reuse_host_validation.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": len(cases), "artifact": str(target)}))


if __name__ == "__main__":
    main()
