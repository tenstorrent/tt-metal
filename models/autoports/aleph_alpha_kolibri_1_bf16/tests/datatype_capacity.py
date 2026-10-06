# SPDX-License-Identifier: Apache-2.0
"""Classify the observed 8K gate/up allocation failure without device execution."""

import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
    low = root / "candidates/expert_gate_up_bfp8_lofi"
    failed_path = low / "attempt_result.json"
    if not failed_path.exists():
        failed_path.write_bytes((low / "result.json").read_bytes())
    failure = json.loads(failed_path.read_text())
    assert failure["status"] == "runtime-fail"
    pattern = r"allocate (\d+) B DRAM buffer across (\d+) banks.*allocated: (\d+) B, free: (\d+) B, largest free block: (\d+) B"
    required, banks, allocated, free, contiguous = map(int, re.search(pattern, failure["error"]).groups())
    assert required > banks * free
    log = (low / "run.log").read_text()
    assert "PREPARED_PREFILL_BUCKET 2048" in log and "PREPARED_PREFILL_BUCKET 8192" not in log
    assert "_grouped_prefill_moe" in log and "Closing devices in cluster completed" in log
    buffer_bytes = lambda tokens: ((6 * tokens + 31 * 384 + 31) // 32) * 32 * 2560 * 2
    assert required == buffer_bytes(8192)
    low_config = failure["precision_config"]
    high_config = json.loads((root / "configs/expert_gate_up_bfp8_hifi2.json").read_text())
    assert low_config["runtime"] == high_config["runtime"]
    assert low_config["layer_exceptions"] == high_config["layer_exceptions"]
    differing = [k for k in low_config["mesh_policy"] if low_config["mesh_policy"][k] != high_config["mesh_policy"][k]]
    assert differing == ["expert_fidelity"]
    evidence = dict(
        status="capacity-classified; smaller-chunk full-model validation pending",
        source_failure="candidates/expert_gate_up_bfp8_lofi/attempt_result.json",
        source_log="candidates/expert_gate_up_bfp8_lofi/run.log",
        scope="The inherited 8192-row prefill geometry exceeds available DRAM for this weight policy; not an impossibility claim for other chunking or layouts",
        allocation_site="MultichipDecoder._grouped_prefill_moe: BF16 output tensor (dispatch_capacity,2560)",
        required_bytes_per_device=required,
        allocated_bytes_per_device=banks * allocated,
        free_bytes_per_device=banks * free,
        largest_free_bytes_per_device=banks * contiguous,
        capacity_deficit_bytes_per_device=required - banks * free,
        prefill_2048_output_bytes_per_device=buffer_bytes(2048),
        prefill_2048_output_saving_bytes_per_device=required - buffer_bytes(2048),
        logical_context=1048576,
        context_reduction=None,
        recovery="No reset: allocator exception, clean mesh close. Test both fidelities with prefill_chunk_size=2048 and unchanged full KV capacity.",
        hifi2_inference="Only sparse expert_fidelity differs. Failure occurs in grouped prefill before decode preparation; grouped prefill uses the unchanged prefill_expert_fidelity. Thus the failing allocation and its live weights/cache/temporaries are identical. No HiFi2 hardware timing or accuracy is claimed.",
    )
    validation = {}
    for fidelity in ("lofi", "hifi2"):
        path = root / "candidates" / f"expert_gate_up_bfp8_{fidelity}_chunk2048" / "result.json"
        if path.exists():
            trial = json.loads(path.read_text())
            validation[fidelity] = dict(
                evidence=str(path.relative_to(root)),
                status=trial["status"],
                full_context=trial.get("runtime_summary", {}).get("capacity"),
                non_aligned_count=len(trial.get("capability", {}).get("non_aligned_checks", [])),
            )
    evidence["smaller_chunk_validation"] = validation
    if len(validation) == 2 and all(
        v["status"] in ("pass", "accuracy-fail") and v["full_context"] == 1048576 and v["non_aligned_count"] == 21
        for v in validation.values()
    ):
        evidence["status"] = "capacity resolved with 2048-row chunks at unchanged 1M context; 8K policies rejected"
    (root / "gate_up_capacity_rejection.json").write_text(json.dumps(evidence, indent=2) + "\n")
    classified = copy.deepcopy(failure)
    classified.update(status="capacity-rejected", original_status="runtime-fail", capacity_evidence=evidence)
    (low / "result.json").write_text(json.dumps(classified, indent=2) + "\n")
    high = root / "candidates/expert_gate_up_bfp8_hifi2"
    high.mkdir(parents=True, exist_ok=True)
    command = dict(
        command=[sys.executable, *sys.argv],
        start=time.time(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        environment={k: v for k, v in os.environ.items() if k.startswith(("TT_METAL_", "FULL_", "KOLIBRI_"))},
        exit_code=0,
        hardware_executed=False,
    )
    derived = {
        k: copy.deepcopy(failure[k]) for k in ("hardware", "mesh", "reference", "reference_sha256", "thresholds")
    }
    derived.update(
        config_id=high_config["config_id"],
        precision_config=high_config,
        status="capacity-rejected",
        measurement_regime="not timed; analytical allocation rejection from the identical grouped-prefill phase of the LoFi failure",
        provenance=dict(command=command["command"], git_head=command["git_head"], environment=command["environment"]),
        capacity_evidence=evidence,
    )
    (high / "result.json").write_text(json.dumps(derived, indent=2) + "\n")
    (high / "run.command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
