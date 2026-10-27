# SPDX-License-Identifier: Apache-2.0
"""CPU comparison of completed split/grouped all64-layer evidence receipts."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def compare(baseline, candidate, require_states=True, policy="grouped"):
    assert baseline["status"] == candidate["status"] == "pass"
    assert policy in ("grouped", "masked_recurrence")
    assert baseline.get(policy, False) is False and candidate.get(policy, False) is True
    for key in ("grouped", "masked_recurrence"):
        if key != policy:
            assert baseline.get(key, False) == candidate.get(key, False), f"Unmatched policy: {key}"
    active_requests = baseline.get("active_requests", baseline["batch"])
    assert active_requests == candidate.get("active_requests", candidate["batch"])
    assert baseline["scope"].startswith("all64-layer")
    assert (
        baseline.get("layer_indices", list(range(64)))
        == candidate.get("layer_indices", list(range(64)))
        == list(range(64))
    )
    for key in (
        "scope",
        "batch",
        "tested_cache_context",
        "supported_model_context",
        "length",
        "outputs",
        "physical_prefill_chunk_size",
        "allocation_tracking",
        "program_cache_skip",
        "precision",
        "prompt_sha256",
    ):
        assert baseline[key] == candidate[key], f"Unmatched configuration: {key}"
    assert baseline["mixed_repeat_exact"] and candidate["mixed_repeat_exact"]
    assert baseline["mixed_outputs"] == candidate["mixed_outputs"], "Ragged token mismatch"
    bqual, cqual = baseline["qualitative"], candidate["qualitative"]
    for key in ("source_sha256", "prompt_mode", "max_output_tokens"):
        assert bqual[key] == cqual[key], f"Unmatched qualitative configuration: {key}"
    assert bqual["outputs"] == cqual["outputs"], "Qualitative tokens or prompts differ"
    assert len(baseline["records"]) == len(candidate["records"]) > 0

    def state_pair(left, right, label):
        assert len(left) == len(right) == 48 * 2 * 4, f"Incomplete state receipt: {label}"
        assert left == right, f"State mismatch: {label}"

    for left, right in zip(baseline["records"], candidate["records"]):
        assert left["repeat"] == right["repeat"]
        assert left["tokens"] == right["tokens"], f"Token mismatch: repeat{left['repeat']}"
        for record in (left, right):
            assert record["tokens_sha256"] == hashlib.sha256(json.dumps(record["tokens"]).encode()).hexdigest()
            assert len(record["tokens"]) == active_requests
            metrics = record["metrics"]
            assert not metrics["host_sampling"]
            assert metrics["steady_state_counters"]["captures"] == 0
            assert metrics["inner_loop_host_refreshes"] == metrics["inner_loop_readbacks"] == 0
        if require_states:
            state_pair(left["state_sha256"], right["state_sha256"], f"repeat{left['repeat']}")
    if require_states:
        assert baseline["continuation_prefix_length"] == candidate["continuation_prefix_length"]
        assert baseline["source_sha256"] == candidate["source_sha256"], "Source changed between arms"
        state_pair(baseline["mixed_state_sha256"], candidate["mixed_state_sha256"], "mixed reset")
        assert len(baseline["continuation"]) == len(candidate["continuation"]) == 3
        for phase, (left, right) in enumerate(zip(baseline["continuation"], candidate["continuation"])):
            state_pair(left["state_sha256"], right["state_sha256"], f"continuation{phase}")
            assert len(left["logits_sha256"]) == len(right["logits_sha256"]) == 4
            assert left == right, f"Continuation logits or protocol mismatch: {phase}"
    candidate_label = "grouped" if policy == "grouped" else "masked_commit"
    medians = {
        name: {
            key: statistics.median(r["metrics"][key] for r in arm["records"])
            for key in ("ttft_ms", "decode_seconds", "decode_tps_per_user")
        }
        for name, arm in (("baseline", baseline), (candidate_label, candidate))
    }
    return dict(
        status="pass",
        scope="Paired all64-layer fixed-work comparison; not SWE accuracy",
        batch=baseline["batch"],
        active_requests=active_requests,
        policy_compared=policy,
        states_compared=require_states,
        tokens_exact=True,
        continuation_logits_exact=True if require_states else None,
        medians=medians,
        decode_speedup=medians["baseline"]["decode_seconds"] / medians[candidate_label]["decode_seconds"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tokens-only", action="store_true", help="Explicitly omit full-state acceptance")
    parser.add_argument("--policy", choices=("grouped", "masked_recurrence"), default="grouped")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        require_states=not args.tokens_only,
        policy=args.policy,
    )
    result["inputs_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.baseline, args.candidate)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
