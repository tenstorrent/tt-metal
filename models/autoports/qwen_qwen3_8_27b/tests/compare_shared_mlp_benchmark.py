# SPDX-License-Identifier: Apache-2.0
"""Consume paired clean all64-layer B32 timing/quality artifacts without TTNN."""
import argparse
import json
import statistics
from pathlib import Path


def compare(control, candidate):
    assert control["status"] == candidate["status"] == "pass"
    assert (control["shared"], candidate["shared"]) == ("0", "1")
    assert control["layers"] == candidate["layers"] == "all64"
    assert control["batch"] == candidate["batch"] == 32
    for name in (
        "native_context",
        "physical_pool_tokens_per_row",
        "physical_prefill_chunk",
        "input_tokens",
        "output_tokens",
        "runtime",
        "source_sha256",
        "prompt_sha256",
        "effective_precision",
        "environment",
        "layer_types",
    ):
        assert control[name] == candidate[name], ("Unmatched clean arm identity", name)
    for arm in (control, candidate):
        assert arm["close_mesh_device_returned"] is True
        assert arm["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "0"
        assert len(arm["records"]) >= 3
        assert len(arm["tokens"]) == arm["batch"]
        assert all(len(tokens) == arm["output_tokens"] for tokens in arm["tokens"])
    assert len(control["records"]) == len(candidate["records"])
    assert control["tokens"] == candidate["tokens"], "Clean paired greedy tokens differ"
    if "qualitative" in control or "qualitative" in candidate:
        assert control["qualitative_prompt_format"] == candidate["qualitative_prompt_format"]
        left, right = control["qualitative"], candidate["qualitative"]
        assert [r["id"] for r in left] == [r["id"] for r in right] == ["explain", "structured", "code"]
        for x, y in zip(left, right):
            for field in ("messages", "prompt_token_ids", "generated_token_ids", "completion", "hf_control"):
                assert x[field] == y[field], ("Paired qualitative mismatch", x["id"], field)
    metrics = {}
    for arm_name, arm in (("control", control), ("shared", candidate)):
        metrics[arm_name] = dict(
            median_ttft_ms=statistics.median(r["metrics"]["ttft_ms"] for r in arm["records"]),
            median_decode_tps_per_user=statistics.median(r["metrics"]["decode_tps_per_user"] for r in arm["records"]),
        )
    return dict(
        status="pass",
        all64_B32_greedy_tokens_exact=True,
        shared_qualitative_tokens_exact=("qualitative" in control),
        metrics=metrics,
        ttft_change_percent=(metrics["shared"]["median_ttft_ms"] / metrics["control"]["median_ttft_ms"] - 1) * 100,
        decode_tps_change_percent=(
            metrics["shared"]["median_decode_tps_per_user"] / metrics["control"]["median_decode_tps_per_user"] - 1
        )
        * 100,
        limits=[
            "Same-native-harness comparison; not actual CI serving performance or prefix retention.",
            "Paired qualitative equality is separate from manual HF/control output-quality review.",
            "No performance tradeoff or stage promotion is accepted automatically.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.control.read_text()), json.loads(args.candidate.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
