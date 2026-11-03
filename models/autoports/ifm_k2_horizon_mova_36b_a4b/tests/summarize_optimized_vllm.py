# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Assert comparable real runner configurations and retain raw serving metrics."""

import gzip
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "readiness_vllm/optimized"


def launch(directory):
    path = directory / "runner.log"
    contents = path.read_text() if path.exists() else gzip.decompress(path.with_suffix(".log.gz").read_bytes()).decode()
    lines = contents.splitlines()
    return {
        "mesh": next(s for s in lines if s.startswith("Launching vLLM server")),
        "command": shlex.split(next(s for s in lines if s.startswith("  cmd:"))[7:]),
    }


def benchmark_command(command):
    command = list(command)
    command[command.index("--result-filename") + 1] = "<artifact>"
    return command


results = []
for batch, filename, label in (
    (1, "vllm_benchmark.json", "primary_single_user"),
    (32, "vllm_benchmark.json", "single_user_at_capacity32"),
    (32, "vllm_ci_serving_benchmark.json", "secondary_ci_burst"),
):
    before_dir, after_dir = [EVIDENCE / f"{phase}_b{batch}" for phase in ("before", "after")]
    assert launch(before_dir) == launch(after_dir), "Server launch settings changed"
    before, after = [json.loads((p / "readiness_vllm" / filename).read_text()) for p in (before_dir, after_dir)]
    assert before["config"] == after["config"]
    assert benchmark_command(before["command"]) == benchmark_command(after["command"])
    assert after["command"][-2:] == ["--num-warmups", "1"]
    for result in (before, after):
        assert result["completed_requests"] == result["config"]["num_requests"]
        assert result["missing_output_tokens"] == 0
        assert result["config"]["temperature"] == 0
        assert abs(result["vllm_mean_tpot_decode_tps"] - 1000 / result["tpot_ms"]["mean"]) < 1e-9
    fields = (
        "ttft_ms",
        "tpot_ms",
        "itl_ms",
        "output_throughput_tok_per_s",
        "vllm_mean_tpot_decode_tps",
        "completed_requests",
        "total_output_tokens",
    )
    results.append(
        dict(
            profile=label,
            max_num_seqs=batch,
            workload=after["config"],
            warmup_requests=1,
            server=launch(after_dir),
            before={k: before[k] for k in fields},
            after={k: after[k] for k in fields},
            ttft_change_percent=100 * (after["ttft_ms"]["p50"] / before["ttft_ms"]["p50"] - 1),
            decode_change_percent=100 * (after["vllm_mean_tpot_decode_tps"] / before["vllm_mean_tpot_decode_tps"] - 1),
            source_before=str(before_dir / "readiness_vllm" / filename),
            source_after=str(after_dir / "readiness_vllm" / filename),
        )
    )
(ROOT / "doc/optimized_vllm/benchmark_comparison.json").write_text(
    json.dumps({"status": "pass", "profiles": results}, indent=2) + "\n"
)
print(json.dumps(results, indent=2))
