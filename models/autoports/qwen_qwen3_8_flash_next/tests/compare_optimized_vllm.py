# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize matched HTTP serving arms without treating burst rates as B1 rates."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def benchmark(directory, filename, expected):
    path = directory / filename
    data = read(path)
    for key, value in expected.items():
        assert data["config"][key] == value, (path, key, data["config"][key], value)
    assert data["completed_requests"] == expected["num_requests"]
    assert data["missing_output_tokens"] == 0
    assert data["total_output_tokens"] == expected["num_requests"] * expected["output_len"]
    assert data["total_input_tokens"] == expected["num_requests"] * expected["prompt_len"]
    command = data["command"]
    assert command[command.index("--num-warmups") + 1] == "1"
    assert all(
        math.isfinite(data[name]["mean"]) and data[name]["mean"] > 0 for name in ("ttft_ms", "tpot_ms", "itl_ms")
    )
    assert math.isfinite(data["output_throughput_tok_per_s"]) and data["output_throughput_tok_per_s"] > 0
    return dict(
        artifact=str(path),
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        workload=data["config"],
        completed_requests=data["completed_requests"],
        ttft_ms=data["ttft_ms"],
        tpot_ms=data["tpot_ms"],
        itl_ms=data["itl_ms"],
        output_throughput_tok_per_s=data["output_throughput_tok_per_s"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the existing comparison before rerunning")
    provenance = {
        name: read(path / "provenance.json") for name, path in (("before", args.before_dir), ("after", args.after_dir))
    }
    contract = provenance["before"]["server_contract"]
    assert contract == provenance["after"]["server_contract"], "Server contracts differ"
    for key in ("shared_runtime", "loaded_binaries_sha256", "launch_environment"):
        assert provenance["before"][key] == provenance["after"][key], f"Shared runtime differs: {key}"
    assert (
        provenance["before"]["actual_launch_summary"]["server_command"]
        == provenance["after"]["actual_launch_summary"]["server_command"]
    ), "Actual launch commands differ"
    replaced = ("/tt/generator.py", "/tt/slot_state.py")
    shared_sources = {
        name: {path: value for path, value in row["source_sha256"].items() if not path.endswith(replaced)}
        for name, row in provenance.items()
    }
    assert shared_sources["before"] == shared_sources["after"], "Shared Python source differs"
    required = {
        "context": 262144,
        "max_num_seqs": 2,
        "mesh": [1, 4],
        "block_size": 32,
        "sample_on_device_mode": "all",
        "async_scheduling": True,
        "prefill_chunk_alignment": 8,
        "decode_interleave_prefill_steps": 1,
        "decode_interleave_decode_steps": 2,
        "trace_mode": "all",
        "tracking": False,
        "device_profiler": False,
        "shutdown_timeout": 0,
        "fabric_config": "FABRIC_1D_RING",
        "trace_region_size": 0,
    }
    assert all(contract[key] == value for key, value in required.items())
    assert contract["max_num_batched_tokens"] > 0 and contract["selected_config_sha256"]
    rows = {}
    for name, directory in (("before", args.before_dir), ("after", args.after_dir)):
        execution = read(directory / "execution.json")
        assert all(
            execution[key]
            for key in (
                "passed",
                "measured_source_unchanged",
                "source_restored",
                "shared_runtime_unchanged",
                "loaded_binaries_unchanged",
                "devices_free",
            )
        ), "Measurement or cleanup incomplete"
        assert execution["remaining_owned_pids"] == []
        decision = execution["cleanup_owner_decision"]
        assert decision["accepted"] and decision["plugin_unchanged"]
        assert read(Path(decision["path"]))["plugin_change_authorized"] is False
        assert hashlib.sha256(Path(decision["path"]).read_bytes()).hexdigest() == decision["sha256"]
        assert all(
            execution["shutdown_audit"][key] for key in ("no_teardown_error", "shutdown_recorded")
        ), "Unexpected teardown error or missing shutdown record"
        assert execution["idle_before_shutdown"] == dict(num_requests_running=0.0, num_requests_waiting=0.0)
        route = read(directory / "benchmark_route_audit.json")
        assert route["host_submissions"] == route["out_of_vocabulary"] == 0
        assert route["device_submissions"] > 0
        primary = benchmark(
            directory,
            "vllm_benchmark.json",
            dict(prompt_len=128, output_len=128, num_requests=1, concurrency=1, temperature=0.0, ignore_eos=True),
        )
        primary["decode_t_s_u"] = 1000 / primary["tpot_ms"]["mean"]
        ci = benchmark(
            directory,
            "vllm_ci_serving_benchmark.json",
            dict(prompt_len=100, output_len=100, num_requests=32, concurrency=None, temperature=0.0, ignore_eos=True),
        )
        burst = benchmark(
            directory,
            "vllm_prefill_burst_benchmark.json",
            dict(prompt_len=128, output_len=32, num_requests=32, concurrency=32, temperature=0.0, ignore_eos=True),
        )
        rows[name] = dict(primary=primary, ci_capacity=ci, serial_prefill_burst=burst)
    full = read(args.full_model)
    assert full["passed"] and full["capacity"] == contract["context"]
    perf = full["performance"]
    assert perf["source_provenance"]["unchanged"]
    assert perf["selected_default_snapshot"]["passed"]
    assert perf["selected_default_snapshot"]["selected_sha256"] == contract["selected_config_sha256"]
    resident = perf["resident_token_out"]
    assert perf["workload"] == dict(batch=1, prompt_tokens=128, generated_tokens=128, decode_steps=127)
    before, after = rows["before"]["primary"], rows["after"]["primary"]
    result = dict(
        passed=True,
        stage_acceptance_claim=False,
        scope="Matched warmed real-vLLM TT-plugin serving through tt/generator_vllm.py",
        server_contract=contract,
        provenance=provenance,
        measurements=rows,
        primary_decode_gain_percent=100 * (after["decode_t_s_u"] / before["decode_t_s_u"] - 1),
        primary_ttft_change_percent=100 * (after["ttft_ms"]["mean"] / before["ttft_ms"]["mean"] - 1),
        final_http_tpot_over_full_model_resident=after["tpot_ms"]["mean"] / resident["ms_per_step"],
        full_model=dict(
            artifact=str(args.full_model),
            workload=perf["workload"],
            ms_per_step=resident["ms_per_step"],
            decode_t_s_u=resident["t_s_u"],
            scope=resident["scope"],
        ),
        serial_prefill_burst_ttft_ratio_after_over_before={
            key: rows["after"]["serial_prefill_burst"]["ttft_ms"][key]
            / rows["before"]["serial_prefill_burst"]["ttft_ms"][key]
            for key in ("mean", "p50", "p99")
        },
        burst_scope="Capacity only. Both arms use serial prefill; this is not a batched-prefill adoption claim.",
        device_profile=None,
        profiler_artifacts=None,
        profiler_exclusion_reason="Serving-stage contract forbids live-server/adapter/device profiling.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key in ("passed", "primary_decode_gain_percent", "final_http_tpot_over_full_model_resident")
            }
        )
    )


if __name__ == "__main__":
    main()
