# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Index preserved stage5 runs and summarize final default evidence."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "doc/optimized_multichip_decoder"


def load(name):
    return json.loads((D / name).read_text())


def pcc_values(value, key=""):
    if isinstance(value, dict):
        return [v for k, child in value.items() for v in pcc_values(child, k)]
    if isinstance(value, list):
        return [v for child in value for v in pcc_values(child, key)]
    return [value] if "pcc" in key and isinstance(value, (int, float)) else []


def save(name, data):
    (D / name).write_text(json.dumps(data, indent=2) + "\n")


def watcher_summary(rows):
    evidence = []
    for row in rows:
        name = row["name"]
        data = load(name + ".json")
        env = data["environment"]
        assert env["TT_METAL_WATCHER"] == "10"
        assert env["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
        assert env["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
        assert env["TT_METAL_DEVICE_PROFILER"] == "0"
        log = (D / name / "generated/watcher/watcher.log").read_bytes()
        contents = log.decode()
        stdout = (D / (name + ".log")).read_text()
        assert "Watcher server initialized, disabled features: None" in stdout
        assert "ALL_PASS" in stdout and "TT_FATAL" not in stdout
        samples = re.findall(r"Dump #(\d+) completed at ([\d.]+)s", contents)
        detached = sorted(set(re.findall(r"detach device (\d+)", contents)))
        assert samples and detached == ["0", "1", "2", "3"]
        evidence.append(
            dict(
                name=name,
                status="pass",
                completed_samples=len(samples),
                last_sample_seconds=float(samples[-1][1]),
                detached_devices=detached,
                disabled_features=[],
                minimum_recorded_pcc=min(pcc_values(data)),
                watcher_log_sha256=hashlib.sha256(log).hexdigest(),
                archive=f"watcher_artifacts/{name}/watcher.log.gz",
            )
        )
    save("watcher_summary.json", dict(status="pass", rows=evidence))


def main():
    experiments = []
    for path in sorted(D.glob("*_sweep.json")):
        for row in json.loads(path.read_text()):
            if not isinstance(row, dict) or "name" not in row:
                continue
            entry = dict(row, sweep=path.name)
            result = D / (row["name"] + ".json")
            # Final names are intentionally retaken. Historical sweep metrics
            # remain exact, but must not inherit PCC from a newer result file.
            historical_final = row["name"].startswith(("final_", "watcher_")) and path.name not in (
                "final_sweep.json",
                "watcher_sweep.json",
            )
            if historical_final:
                entry[
                    "result_note"
                ] = "Historical invocation; final-named result may have been superseded. Use preserved sweep metrics and source/log archives."
            elif row["returncode"] == 0 and result.exists():
                data = json.loads(result.read_text())
                scores = pcc_values(data)
                entry["minimum_recorded_pcc"] = min(scores) if scores else None
            experiments.append(entry)
    save("candidate_summary.json", experiments)
    perf = []
    for layer in (0, 3):
        for length in (33, 1024, 4097):
            before = load(f"before_l{layer}_s{length}.json")
            final = load(f"final_l{layer}_s{length}.json")
            assert final["policy"] and final["mode"] == "replicated"
            row = dict(
                layer=layer,
                length=length,
                batch=1,
                before=f"before_l{layer}_s{length}.json",
                final=f"final_l{layer}_s{length}.json",
            )
            for phase in ("prefill", "decode"):
                b, f = before[phase + "_ms"], final[phase + "_ms"]
                row[phase] = dict(
                    before_ms=b,
                    final_ms=f,
                    speedup=b / f,
                    latency_reduction_percent=100 * (b - f) / b,
                    before_samples_ms=before[phase + "_samples_ms"],
                    final_samples_ms=final[phase + "_samples_ms"],
                )
            row["minimum_recorded_pcc"] = min(pcc_values(final))
            perf.append(row)
    save("performance_summary.json", dict(classification="measured warmed wall time; final default TP4", rows=perf))
    final = load("final_sweep.json")
    watcher = load("watcher_sweep.json")
    assert all(r["returncode"] == 0 for r in final + watcher)
    helper_hash = hashlib.sha256((ROOT / "tt/prepared_prefill.py").read_bytes()).hexdigest()
    current_sources = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "tt/multichip_decoder.py", ROOT / "tt/optimized_decoder.py")
    }
    checks = []
    for row in final + watcher:
        data = load(row["name"] + ".json")
        if isinstance(data, list):
            sources = load(row["name"] + ".provenance.json")["source_sha256"]
        elif "source_sha256" in data:
            sources = data["source_sha256"]
        else:
            sources = {
                "tt/multichip_decoder.py": data["decoder_sha256"],
                "tt/optimized_decoder.py": data["inherited_decoder_sha256"],
            }
        assert all(sources[path] == digest for path, digest in current_sources.items())
        helper_used = isinstance(data, dict) and (
            data.get("prefill_capture_count", 0) or data.get("prefill_trace_captures", 0)
        )
        if helper_used:
            assert data.get("prepared_prefill_sha256", data.get("prepared_prefill_source_sha256")) == helper_hash
            if "prefill_trace_ids" in data:
                assert len(data["prefill_trace_ids"]) == 2
                assert data["prefill_teardown_release_count"] == 2
                assert data["request_recapture_count"] == 0
        scores = pcc_values(data)
        checks.append(
            dict(
                name=row["name"],
                returncode=0,
                minimum_recorded_pcc=min(scores) if scores else None,
                evidence=row["name"] + ".json",
                prepared_prefill=bool(helper_used),
            )
        )
    assert min(r["minimum_recorded_pcc"] for r in checks if r["minimum_recorded_pcc"] is not None) >= 0.995
    save(
        "final_acceptance_summary.json",
        dict(
            status="pass",
            acceptance=0.995,
            checks=checks,
            minimum_recorded_pcc=min(
                r["minimum_recorded_pcc"] for r in checks if r["minimum_recorded_pcc"] is not None
            ),
            source_sha256=dict(current_sources, **{"tt/prepared_prefill.py": helper_hash}),
        ),
    )
    watcher_summary(watcher)
    print("FINAL_SUMMARY_PASS", len(checks), "checks", len(experiments), "experiments")


if __name__ == "__main__":
    main()
