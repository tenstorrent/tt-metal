"""Rebuild validated per-run matrix summaries without pooling repetitions."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from extract_timings import extract, OPS


def aggregate(root):
    records = []
    for run in sorted(root.glob("run-*")):
        exitcode = run / "exitcode"
        if not exitcode.exists() or exitcode.read_text().strip() != "0":
            continue
        worker = (
            (run / "worker.py").read_bytes()
            if (run / "worker.py").exists()
            else gzip.decompress((run / "worker.py.gz").read_bytes())
        )
        assert b"mesh_device.load_sub_device_manager(" not in worker, run
        paths = list((run / "profiler").rglob("ops_perf_results*.csv")) or list(
            (run / "profiler").rglob("ops_perf_results*.csv.gz")
        )
        assert len(paths) == 1, (run, paths)
        result = extract(paths[0], 10)
        case = json.loads((run / "case.json").read_text())
        assert (result["layer"], result["mode"]) == (case["layer"], case["mode"]), run
        (run / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
        for op in OPS:
            stats = result["operations"][op]
            device_means = list(stats["per_device_mean_ms"].values())
            records.append(
                dict(
                    run=run.name,
                    layer=result["layer"],
                    mode=result["mode"],
                    operation=op,
                    mean_device_max_ms=stats["mean_of_device_max_ms"],
                    median_device_max_ms=stats["median_of_device_max_ms"],
                    p90_device_max_ms=stats["p90_nearest_rank_device_max_ms"],
                    minimum_device_mean_ms=min(device_means),
                    maximum_device_mean_ms=max(device_means),
                    device_mean_spread_ms=max(device_means) - min(device_means),
                    retained_iterations=9,
                    worker_sha256=hashlib.sha256(worker).hexdigest(),
                    routing_case_sha256=hashlib.sha256((run / "case.json").read_bytes()).hexdigest(),
                )
            )
    assert records, "No completed captures"
    assert len({r["worker_sha256"] for r in records}) == 1, "Worker changed across successful captures"
    with (root / "RUN_TIMINGS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    groups = []
    for key in sorted({(r["layer"], r["mode"], r["operation"]) for r in records}):
        selected = [r for r in records if (r["layer"], r["mode"], r["operation"]) == key]
        assert len({r["routing_case_sha256"] for r in selected}) == 1, ("routing changed across repeats", key)
        vals = [r["mean_device_max_ms"] for r in selected]
        groups.append(
            dict(
                layer=key[0],
                mode=key[1],
                operation=key[2],
                independent_process_runs=len(vals),
                run_means_ms=vals,
                mean_of_run_means_ms=statistics.mean(vals),
                minimum_run_mean_ms=min(vals),
                maximum_run_mean_ms=max(vals),
            )
        )
    report = {
        "metric": "Within each process discard iteration0, retain9; kernel duration max across8devices per iteration, then mean. Summarize fresh-process run means separately; no pooled pseudo-replicates.",
        "worker_sha256": records[0]["worker_sha256"],
        "runs": records,
        "grouped_run_summaries": groups,
    }
    (root / "matrix_timings.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# B1 isolated dispatch/combine matrix",
        "",
        "Only successful manager-free captures with complete signposted coverage are included. Times are milliseconds; each row is one operation in one fresh process. Each process discards its initial iteration and retains nine. Device spread is max minus min of the eight device means.",
        "",
        "| Run | Layer | Routing | Operation | Mean device max | Median device max | Device mean spread |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['run']} | {r['layer']} | {r['mode']} | {r['operation'].replace('DeviceOperation','')} | {r['mean_device_max_ms']:.6f} | {r['median_device_max_ms']:.6f} | {r['device_mean_spread_ms']:.6f} |"
        )
    lines += [
        "",
        "## Fresh-process run summaries",
        "",
        "| Layer | Routing | Operation | Runs | Mean of run means | Run-mean range |",
        "|---:|---|---|---:|---:|---|",
    ]
    for g in groups:
        lines.append(
            f"| {g['layer']} | {g['mode']} | {g['operation'].replace('DeviceOperation','')} | {g['independent_process_runs']} | {g['mean_of_run_means_ms']:.6f} | {g['minimum_run_mean_ms']:.6f}–{g['maximum_run_mean_ms']:.6f} |"
        )
    lines += [
        "",
        "Placement changes destination imbalance, per-token destination fanout/locality, per-chip tile-rounded expert-region workload and expert ordering. Source shuffling preserves expert/destination totals and fanout histogram while changing source flow and per-expert ordering/batching. The host_comparison.json records tile-rounded load ratios alongside raw assignments. These controls constrain hypotheses; they do not prove imbalance alone determines time.",
        "",
        "These are isolated eager dispatch/combine kernels with an FFN surrogate, not full-model latency or throughput. Numerical output correctness was not checked; this is not a validated production optimization. The recovered routing came from a different execution than the historical PP timings. Manager-free success after hardware recovery does not isolate manager removal from recovery as the cause of the earlier stall.",
        "",
        "Reproduce: `python3 aggregate_timings.py --root .` from this directory.",
    ]
    (root / "MATRIX_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        f"Validated {len(records)//2} completed captures; wrote RUN_TIMINGS.csv, matrix_timings.json and MATRIX_RESULTS.md"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    aggregate(p.parse_args().root)
