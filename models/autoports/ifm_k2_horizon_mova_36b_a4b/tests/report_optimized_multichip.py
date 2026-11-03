# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only consolidation of stage-5 final-run and profiler evidence."""

import csv
import gzip
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_multichip_decoder"


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (DOC / name).write_text(json.dumps(value, indent=2) + "\n")


def table(name, rows):
    with (DOC / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*decoder.py")
    }
    candidates = []
    for path in sorted(DOC.glob("*.json")):
        data = read(path)
        if not isinstance(data, dict) or not all(k in data for k in ("decode_ms", "prefill_ms", "layer", "seq")):
            continue
        candidates.append(
            dict(
                artifact=path.name,
                layer=data["layer"],
                batch=data["batch"],
                seq=data["seq"],
                baseline=data["baseline"],
                tracking=data.get("tracking"),
                profile_samples=data.get("profile_samples", 0),
                prefill_ms=data["prefill_ms"],
                decode_ms=data["decode_ms"],
                prefill_pcc=data.get("pcc", {}).get("prefill"),
                decode_pcc=data.get("pcc", {}).get("decode"),
                min_cache_pcc=min(data.get("cache_pcc", [1])),
                source_sha256=data.get("source_sha256"),
                options=json.dumps(data.get("options", {}), sort_keys=True),
            )
        )
    table("candidate_measurements.csv", candidates)
    performance = []
    for layer in (0, 3):
        for seq in (32, 256, 2048, 2049, 4096):
            after_path = DOC / f"accepted_default_l{layer}_b1_s{seq}.json"
            after = read(after_path)
            assert after["source_sha256"] == sources["tt/multichip_decoder.py"]
            assert not after["baseline"] and after["options"] == {} and after["tracking"] == "0"
            before_path = DOC / f"before_l{layer}_b1_s{seq}.json"
            before = read(before_path) if before_path.exists() else None
            performance.append(
                dict(
                    layer=layer,
                    seq=seq,
                    context=after["context"],
                    before_prefill_ms=before["prefill_ms"] if before else None,
                    after_prefill_ms=after["prefill_ms"],
                    prefill_speedup=before["prefill_ms"] / after["prefill_ms"] if before else None,
                    before_decode_ms=before["decode_ms"] if before else None,
                    after_decode_ms=after["decode_ms"],
                    decode_speedup=before["decode_ms"] / after["decode_ms"] if before else None,
                    before_prefill_pcc=before["pcc"]["prefill"] if before else None,
                    after_prefill_pcc=after["pcc"]["prefill"],
                    before_decode_pcc=before["pcc"]["decode"] if before else None,
                    after_decode_pcc=after["pcc"]["decode"],
                    min_cache_pcc=min(after["cache_pcc"]),
                    before_artifact=before_path.name if before else None,
                    after_artifact=after_path.name,
                )
            )
    table("performance.csv", performance)
    write("performance.json", dict(sources=sources, rows=performance))
    prefill_traces = []
    for layer in (0, 3):
        for seq in (32, 33, 2048, 2049, 4095, 4096, 4097):
            path = DOC / f"accepted_prefill_trace_l{layer}_b1_s{seq}.json"
            record = read(path)
            assert record["source_sha256"] == sources["tt/multichip_decoder.py"]
            assert record["tracking"] == "1" and record["skip_program_cache"] == "0"
            assert record["options"] == {} and "five queued bursts" in record["prefill_trace_determinism"]
            prefill_traces.append(
                dict(
                    layer=layer,
                    seq=seq,
                    artifact=path.name,
                    artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_sha256=record["source_sha256"],
                    prefill_pcc=record["pcc"]["prefill"],
                    decode_pcc=record["pcc"]["decode"],
                    eager_ms_tracking_enabled=record["prefill_ms"],
                    traced_ms_tracking_enabled=record["traced_prefill_ms"],
                    capture_ms=record["prefill_trace_capture_ms"],
                    determinism=record["prefill_trace_determinism"],
                )
            )
    table("prefill_trace_summary.csv", prefill_traces)
    write(
        "prefill_trace_index.json",
        dict(
            status="pass",
            records=prefill_traces,
            blocking_checked_replays=1400,
            queued_burst_replays=1400,
            timed_queued_outputs_checked=14,
            exact_eager_repeats=140,
            note="Every check compares all4 output replicas. Tracking-enabled advice trials are separate from untracked headline timing.",
        ),
    )
    accounting, important, runtime_rows = [], [], []
    for tag in ("before", "accepted"):
        for layer, kind in ((0, "dense"), (3, "mova_moe")):
            host = read(DOC / f"profile_{tag}_l{layer}_b1_s2048.json")
            audit = read(DOC / f"profile_{tag}_l{layer}_audit.json")
            assert audit["status"] == "pass"
            for phase in ("decode", "prefill"):
                folder = DOC / "tracy" / tag / kind
                raw_path = folder / f"{phase}_ops.csv"
                with raw_path.open() as f:
                    raw = list(csv.DictReader(f))
                marker = "PERF_" + phase.upper()
                start = next(i for i, r in enumerate(raw) if r["OP CODE"] == marker)
                end = next(i for i, r in enumerate(raw) if r["OP CODE"] == marker + "_END")
                window = raw[start + 1 : end]
                if tag == "accepted":
                    for op in window:
                        if op["DEVICE ID"] != "0" or not any(
                            k in op["OP CODE"].lower()
                            for k in ("matmul", "norm", "sdpa", "scatter", "gather", "allreduce")
                        ):
                            continue
                        fields = [
                            "OP CODE",
                            "GLOBAL CALL COUNT",
                            "ATTRIBUTES",
                            "MATH FIDELITY",
                            "CORE COUNT",
                            "DEVICE KERNEL DURATION [ns]",
                            "COMPUTE KERNEL SOURCE",
                            "DATA MOVEMENT KERNEL SOURCE",
                        ]
                        fields += [
                            f"{prefix}_{key}"
                            for prefix in ("INPUT_0", "INPUT_1", "OUTPUT_0")
                            for key in (
                                "W_PAD[LOGICAL]",
                                "Z_PAD[LOGICAL]",
                                "Y_PAD[LOGICAL]",
                                "X_PAD[LOGICAL]",
                                "DATATYPE",
                                "MEMORY",
                                "LAYOUT",
                            )
                        ]
                        runtime_rows.append(dict(kind=kind, phase=phase, **{k: op.get(k) for k in fields}))
                ranks = []
                for rank in range(4):
                    rr = [r for r in window if r["DEVICE ID"] == str(rank) and r["DEVICE FW DURATION [ns]"]]
                    rates = [
                        (float(r["DEVICE FW END CYCLE"]) - float(r["DEVICE FW START CYCLE"]))
                        / float(r["DEVICE FW DURATION [ns]"])
                        for r in rr
                        if float(r["DEVICE FW DURATION [ns]"])
                    ]
                    rate = statistics.median(rates)
                    span = (
                        (
                            max(float(r["DEVICE FW END CYCLE"]) for r in rr)
                            - min(float(r["DEVICE FW START CYCLE"]) for r in rr)
                        )
                        / rate
                        / 1e6
                    )
                    ranks.append(
                        dict(
                            rank=rank,
                            operation_count=len(rr),
                            cycles_per_ns=rate,
                            elapsed_ms=span,
                            kernel_sum_ms=sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in rr) / 1e6,
                        )
                    )
                with (folder / f"{phase}_perf_report.csv").open() as f:
                    report = list(csv.DictReader(f))
                host_ms = host[phase + "_ms"]
                maxspan = max(r["elapsed_ms"] for r in ranks)
                weights = (15073280 if tag == "before" else 11796480) if layer == 0 else 14745600
                minimum_bytes = weights + 1131520 + 34816
                floor_ms = minimum_bytes / 512e6 if phase == "decode" else None
                accounting.append(
                    dict(
                        tag=tag,
                        kind=kind,
                        phase=phase,
                        host_ms=host_ms,
                        ranks=ranks,
                        max_rank_elapsed_ms=maxspan,
                        host_minus_max_rank_ms=host_ms - maxspan,
                        report_sum_max_rank_kernel_ms=sum(float(r["Device Time [μs]"] or 0) for r in report) / 1000,
                        minimum_decode_traffic_bytes_per_chip=minimum_bytes if phase == "decode" else None,
                        modeled_512GBs_bandwidth_floor_ms=floor_ms,
                        floor_over_same_run_host_fraction=floor_ms / host_ms if floor_ms else None,
                        floor_over_max_rank_elapsed_fraction=floor_ms / maxspan if floor_ms else None,
                        operations_rank0=dict(Counter(r["OP CODE"] for r in window if r["DEVICE ID"] == "0")),
                        raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    )
                )
                for index, row in enumerate(report):
                    if any(
                        word in row["OP Code"].lower()
                        for word in (
                            "matmul",
                            "norm",
                            "sdpa",
                            "shard",
                            "reshape",
                            "slice",
                            "allreduce",
                            "scatter",
                            "gather",
                            "binary",
                        )
                    ):
                        important.append(
                            dict(
                                tag=tag,
                                kind=kind,
                                phase=phase,
                                row=index,
                                **{
                                    key: row[key]
                                    for key in (
                                        "OP Code",
                                        "Device Time [μs]",
                                        "Op-to-Op Gap [μs]",
                                        "Math Fidelity",
                                        "Input 0 Datatype",
                                        "Input 1 Datatype",
                                        "Output Datatype",
                                        "Inner Dim Block Size",
                                        "Output Subblock H",
                                        "Output Subblock W",
                                        "Input 0 Memory",
                                        "Advice",
                                    )
                                },
                            )
                        )
    write(
        "profile_accounting.json",
        dict(
            rows=accounting,
            interpretation="Per-rank elapsed uses each chip own clock. Sum of per-op max-rank durations is not a critical-path elapsed time; rank choices switch and overlap sparse work/CCL waits.",
            bandwidth_model="512 GB/s per chip; stored BFP4/BFP8/BF16 tile bytes576/1088/2048. Decode minimum reads selected weights and2080 cache rows at logical2049 plus one cache tile read/write. Excludes activation/gamma traffic; floor ratio is not measured DRAM utilization. All4 chip traffic and bandwidth scale together.",
        ),
    )
    table("important_profile_rows.csv", important)
    write(
        "runtime_program_rows.json", dict(source_sha256=sources["tt/multichip_decoder.py"], rank=0, rows=runtime_rows)
    )
    records = []
    patterns = [
        "final_boundaries_*.json",
        "final_contract_*.json",
        "final_chunk_l*.json",
        "final_awkward_l*.json",
        "final_stack_l*_tp4.json",
        "capacity_tp4_*.json",
        "full_prefill_tp4_*.json",
        "watcher_l*_b*.json",
    ]
    for pattern in patterns:
        for path in sorted(DOC.glob(pattern)):
            data = read(path)
            if data.get("baseline"):
                continue
            source = data.get("source_sha256", data.get("decoder_sha256"))
            assert source == sources["tt/multichip_decoder.py"], path
            tracking = data.get("tracking", data.get("trace_tracking", data.get("trace_alloc_tracking")))
            assert tracking == "1", path
            if "cases" in data:
                assert data["stress_replays"] > 0 and data["capture_count"] == 1
                assert data["skip_program_cache"] == "0"
                assert min(r[k] for r in data["cases"] for k in ("prefill_pcc", "decode_pcc")) >= 0.995
            records.append(
                dict(
                    artifact=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_sha256=source,
                    tracking=tracking,
                    stress_replays=data.get("stress_replays"),
                    case_count=len(data.get("cases", data.get("rows", []))),
                )
            )
    assert len(records) == 42, len(records)
    write(
        "validation_index.json",
        dict(sources=sources, records=records, total_stress_replays=sum(r["stress_replays"] or 0 for r in records)),
    )

    watcher = []
    for layer in (0, 3):
        execution = read(DOC / f"watcher_l{layer}_execution.json")
        env = execution["environment"]
        assert execution["exit_code"] == 0 and env["TT_METAL_WATCHER"] == "10"
        assert env["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
        assert env["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
        assert not any("WATCHER_DISABLE" in k and v != "0" for k, v in env.items())
        assert env.get("TT_METAL_DEVICE_PROFILER", "0") == "0"
        folder = DOC / f"watcher/layer{layer}/generated/watcher"
        log = folder / "watcher.log"
        archive = log.with_suffix(log.suffix + ".gz")
        content = gzip.decompress(archive.read_bytes()).decode() if archive.exists() else log.read_text()
        dumps = re.findall(r"Dump #(\d+) completed at ([\d.]+)s", content)
        detaches = sorted(map(int, re.findall(r"detach device (\d+)", content)))
        assert dumps and detaches == [0, 1, 2, 3]
        kernel_count = len((folder / "kernel_names.txt").read_text().splitlines())
        assert kernel_count > 0
        result_path = next(DOC.glob(f"watcher_l{layer}_b*.json"))
        result = read(result_path)
        watcher.append(
            dict(
                layer=layer,
                execution=f"watcher_l{layer}_execution.json",
                result=result_path.name,
                watcher_log=str(log),
                log_sha256=hashlib.sha256(content.encode()).hexdigest(),
                completed_dumps=len(dumps),
                last_dump_seconds=float(dumps[-1][1]),
                kernel_map_entries=kernel_count,
                stress_replays=result["stress_replays"],
                detach_devices=detaches,
                disabled_watcher_features=[],
                min_prefill_pcc=min(r["prefill_pcc"] for r in result["cases"]),
                min_decode_pcc=min(r["decode_pcc"] for r in result["cases"]),
                source_sha256=result["source_sha256"],
            )
        )
    write(
        "watcher_audit.json",
        dict(
            status="pass",
            fabric_opt="Os",
            isolated_cache=True,
            profiler_enabled=False,
            records=watcher,
            coverage="Both layer kinds B2 lengths31/33/65 plus continuation and nonsequential positions; all asserts and12-second stress; normal O3 timing/profile runs are separate",
        ),
    )


if __name__ == "__main__":
    main()
