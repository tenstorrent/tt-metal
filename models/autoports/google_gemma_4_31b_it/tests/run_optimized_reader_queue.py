"""Collect isolated reader experiments and compute device-bandwidth evidence."""

import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

from .run_optimized_profile_queue import OUT, command
from .summarize_optimized_profile import extract, window


def main():
    for layer in (0, 5):
        path = OUT / "reader_microbench" / f"layer{layer}"
        path.mkdir(parents=True, exist_ok=True)
        args = [
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-p",
            "-v",
            "--no-web-server",
            "--op-support-count",
            "16384",
            "-o",
            str(path),
            "-m",
            "models.autoports.google_gemma_4_31b_it.tests.profile_optimized_readers",
            "--layer",
            str(layer),
            "--result",
            str(path / "correctness.json"),
        ]
        env = dict(os.environ, TT_METAL_DEVICE_PROFILER="1", TT_METAL_TRACE_ALLOC_TRACKING="0")
        env.pop("TT_METAL_WATCHER", None)
        source_files = [
            Path(__file__),
            Path(__file__).with_name("profile_optimized_readers.py"),
            Path(__file__).parents[1] / "tt/optimized_decoder.py",
        ]
        hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
        print("READER_START", layer, flush=True)
        command(args, path / "collection.log", env)
        original = max(path.rglob("ops_perf_results*.csv"), key=lambda p: p.stat().st_mtime_ns)
        shutil.copy2(original, path / "ops.csv")
        rows = list(csv.DictReader(original.open()))
        evidence = json.loads((path / "correctness.json").read_text())
        for record in evidence["records"]:
            times = []
            trials = []
            runtime = None
            for trial in range(6):
                label = f"READER_{record['role_id']}_{record['readers']}_{trial}"
                group = extract(rows, label, label + "_END")
                ops = [r for r in group if r.get("DEVICE FW START CYCLE")]
                assert len(ops) == 16 and all("Matmul" in r["OP CODE"] for r in ops)
                assert all(
                    r["INPUT_1_DATATYPE"] == "BFLOAT4_B"
                    and r["MATH FIDELITY"].lower() == "lofi"
                    and f"num_workers_per_dram_bank={record['readers']}" in r["ATTRIBUTES"]
                    for r in ops
                )
                values = [window([r])[0] for r in ops]
                times.extend(values)
                trials.append(statistics.median(values))
                runtime = ops[0]
            median = statistics.median(times)
            physical = record["k"] * record["n"] * 576 / 1024
            logical = record["k"] * record["logical_n"] / 2
            record.update(
                device_us=median,
                min_device_us=min(times),
                max_device_us=max(times),
                trial_medians_us=trials,
                relative_stdev=statistics.stdev(times) / statistics.mean(times),
                physical_weight_bytes=physical,
                logical_weight_payload_bytes=logical,
                physical_gbps=physical / median / 1000,
                logical_gbps=logical / median / 1000,
                physical_dram_peak_pct=100 * physical / median / 1000 / 512,
                runtime_attributes=runtime["ATTRIBUTES"],
                device_time_basis="complete isolated matmul FW window, including its internal gaps;96 samples in6 alternating-order trials",
                bandwidth_basis="one weight read; physical BFP4 includes exponent headers and zero padding, logical payload excludes both; activations/output are L1",
            )
        evidence["source_hashes"] = hashes
        evidence["command"] = args
        evidence["ops_source"] = str(original)
        (path / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
        fields = [
            "role",
            "readers",
            "k",
            "n",
            "input_cores",
            "in0_block_w",
            "bank_tiles",
            "per_reader_row_bytes",
            "device_us",
            "min_device_us",
            "max_device_us",
            "physical_gbps",
            "logical_gbps",
            "physical_dram_peak_pct",
            "pcc_vs_one_reader",
            "relative_stdev",
        ]
        with (path / "results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(evidence["records"])
        print("READER_END", layer, flush=True)


if __name__ == "__main__":
    main()
