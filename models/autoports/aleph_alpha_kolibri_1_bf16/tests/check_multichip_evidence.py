# SPDX-License-Identifier: Apache-2.0
"""Check current multichip correctness, capability, trace and profiler artifacts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"
SOURCE = ROOT / "tt/multichip_decoder.py"


def read(name, current=True):
    value = json.loads((DOC / name).read_text())
    if current:
        assert value["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest(), name
    return value


def pcc(value):
    rows = value["rows"]
    assert rows and min(r["pcc"] for r in rows) >= 0.995
    for row in rows:
        if "per_request_pcc" in row:
            assert min(row["per_request_pcc"]) >= 0.995
    return min(r["pcc"] for r in rows)


def main():
    plan = read("memory_capacity_plan.json", False)
    assert plan["status"] == "passed" and plan["supported_context"] == 1048576
    assert plan["capability_reduction"] is None
    assert plan["planned_bytes_per_device"] < plan["physical_dram_bytes_per_device"]
    contract = json.loads((ROOT / "doc/context_contract.json").read_text())["multichip_decoder_plan"]
    assert contract == plan
    result = []
    for layer in (0, 4):
        coverage = read(f"coverage_{layer}.json")
        assert set(coverage["lengths"]) == {1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 777, 4095, 4096, 4097, 8193}
        assert coverage["trace_variants"] == [1, 3, 17, 32] and coverage["repeated_replays"] == 96
        for key in ("changed_inputs", "changed_pages", "changed_positions", "runtime_audit", "tracking"):
            assert coverage[key]
        assert not coverage["skip_program_cache"]
        minimum = pcc(coverage)
        context = read(f"context_{layer}.json")
        assert context["capacity"] == 1048576 and context["largest_physical_chunk"] == 4096
        assert any(r["case"] == "prefill_1048559_17" for r in context["rows"])
        assert context["physical_cache_tokens"] == (4608 if layer == 0 else 1048576)
        assert (
            context["reservation_bytes"]
            >= plan["planned_bytes_per_device"] - plan["reserved_trace_activation_ccl_bytes"]
        )
        assert context["tracking"] and not context["skip_program_cache"]
        minimum = min(minimum, pcc(context))
        watcher = read(f"watcher_{layer}.json")
        assert watcher["watcher"] == "10" and watcher["watcher_disable_eth"] == "1"
        assert watcher["tracking"] and not watcher["skip_program_cache"] and watcher["repeated_replays"] == 96
        minimum = min(minimum, pcc(watcher))
        for tokens in (128, 8193):
            final = read(f"final_{layer}_{tokens}.json")
            baseline = read(f"final_baseline_{layer}_{tokens}.json", False)
            assert final["repetitions"] == baseline["repetitions"] == 100
            assert baseline["baseline"] and not final["baseline"]
            assert final["tracking"] == baseline["tracking"] == "0"
            assert min(final["pcc"].values()) >= 0.995
            assert final["policy"]["residual_layout"] == "replicated" and final["policy"]["grouped_prefill"]
            assert final["policy"]["ccl_mode"] == "direct" and final["policy"]["indexed_experts"]
            assert final["policy"]["prefix_matmul"]
            speedup = {mode: baseline["latency_ms"][mode] / final["latency_ms"][mode] for mode in ("prefill", "decode")}
            result.append(
                dict(
                    layer=layer,
                    tokens=tokens,
                    minimum_pcc=minimum,
                    baseline_ms=baseline["latency_ms"],
                    multichip_ms=final["latency_ms"],
                    speedup=speedup,
                    efficiency={k: v / 4 for k, v in speedup.items()},
                )
            )
        profile = read(f"profile_final_{layer}_128/provenance.json", False)
        read(f"profiled_final_{layer}_128.json")
        assert len(profile["runs"]) == 8
        for row in profile["runs"]:
            assert row["rows"] and row["kernel_us"] > 0
            for suffix in ("txt", "csv"):
                assert (
                    DOC / f"profile_final_{layer}_128" / f"{row['mode']}_device{row['device']}_report.{suffix}"
                ).is_file()
    long = read("profile_final_0_8193/provenance.json", False)
    read("profiled_final_0_8193.json")
    assert len(long["runs"]) == 8
    ring = read("ring_batch.json")
    assert ring["batch"] == 3 and len(set(ring["positions"])) == 3
    assert ring["physical_tokens_per_owner"] == 8192 and ring["logical_capacity"] == 16384
    assert ring["trace_replays"] == 6 and ring["tracking"] and ring["runtime_audit"]
    assert ring["skip_program_cache"] == "0" and ring["watcher"] == "10"
    assert sum(r["case"].startswith("cache_") for r in ring["rows"]) == 6
    pcc(ring)
    stack = read("stack.json", False)
    assert stack["layers"] == [0, 4] and stack["boundary_conversions"] == 0 and stack["shared_collective_workspace"]
    assert stack["tracking"] and stack["runtime_audit"] and stack["trace_replays"] == 10
    pcc(stack)
    output = dict(
        status="pass",
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        measurements=result,
        watcher_scope="worker watcher10; Ethernet instrumentation exceeds ACTIVE_ETH config buffer",
    )
    (DOC / "evidence_check.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
