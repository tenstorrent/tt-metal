#!/usr/bin/env python3
"""Linux runner: one same-process cold/hot capture, validation, and comparison.
No third-party Python dependencies. Exit 0 = report/pass; 1 = regression;
2 = invalid measurement/configuration/missing baseline in enforce mode.
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
BASELINES_DIR = HERE / "baselines"
PHASES = ("cold", "hot")

class MeasurementError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise MeasurementError(message)

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

REQUIRED_ZONE_COLUMNS = ("name", "ns_since_start", "exec_time_ns", "thread")

@dataclasses.dataclass(frozen=True)
class Zone:
    name: str
    start: int
    duration: int
    thread: str
    extra: dict = dataclasses.field(default_factory=dict)
    @property
    def end(self):
        return self.start + self.duration

def read_zones(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None, "Empty Tracy CSV")
        fields = {k.strip().lstrip("\ufeff") for k in reader.fieldnames}
        require(set(REQUIRED_ZONE_COLUMNS) <= fields, f"Unsupported CSV header {fields}; expected host csvexport -u")
        zones = []
        for raw in reader:
            row = {k.strip().lstrip("\ufeff"): v for k, v in raw.items() if k is not None}
            try:
                extra = {k: row[k] for k in row if k not in REQUIRED_ZONE_COLUMNS}
                z = Zone(row["name"], int(row["ns_since_start"]), int(row["exec_time_ns"]), row["thread"], extra)
            except (KeyError, TypeError, ValueError) as e:
                raise MeasurementError(f"Malformed Tracy row: {row}") from e
            require(z.duration >= 0, f"Unfinished/negative-duration zone: {z.name}")
            zones.append(z)
    return zones

def contains(parent, child):
    return parent.thread == child.thread and parent.start <= child.start and child.end <= parent.end

def validate_zone_specs(specs):
    """Each entry is {key, name, parent?}. Children of one parent run in list order."""
    require(isinstance(specs, list) and specs, "Zone list is empty")
    keys = []
    names = []
    for spec in specs:
        require(isinstance(spec, dict) and spec.get("key") and spec.get("name"), f"Zone entry needs key and name: {spec}")
        parent = spec.get("parent")
        require(parent is None or (isinstance(parent, str) and parent), f"Zone parent must be a key: {spec}")
        require(parent != spec["key"], f"Zone {spec['key']} cannot be its own parent")
        keys.append(spec["key"])
        names.append(spec["name"])
    require(len(set(keys)) == len(keys), "Zone keys must be distinct")
    require(len(set(names)) == len(names), "Zone names must be distinct")
    by_key = {spec["key"]: spec for spec in specs}
    for spec in specs:
        seen = set()
        parent = spec.get("parent")
        while parent is not None:
            require(parent in by_key, f"Unknown parent {parent} for {spec['key']}")
            require(parent not in seen, f"Zone parent cycle at {spec['key']}")
            seen.add(parent)
            parent = by_key[parent].get("parent")
    return specs

def extract(zones, specs):
    specs = validate_zone_specs(specs)
    markers = {}
    for phase in PHASES:
        found = [z for z in zones if z.name == f"FabricInitBenchmark::{phase}"]
        require(len(found) == 1, f"Expected exactly one {phase} phase marker, got {len(found)}")
        markers[phase] = found[0]
    require(markers["cold"].thread == markers["hot"].thread, "Phases ran on different host threads")
    require(markers["cold"].end <= markers["hot"].start, "Cold/hot markers overlap or are reversed")
    output = {}
    for phase, marker in markers.items():
        selected = {}
        for spec in specs:
            found = [z for z in zones if z.name == spec["name"] and contains(marker, z)]
            require(len(found) == 1, f"{phase}: expected one complete same-thread zone {spec['name']}, got {len(found)}")
            require(found[0].duration > 0, f"{phase}/{spec['key']}: zero duration")
            selected[spec["key"]] = found[0]
        children = {}
        for spec in specs:
            parent = spec.get("parent")
            if parent is None:
                continue
            require(contains(selected[parent], selected[spec["key"]]), f"{phase}: {spec['key']} outside {parent}")
            children.setdefault(parent, []).append(spec["key"])
        for siblings in children.values():
            for earlier_key, later_key in zip(siblings, siblings[1:]):
                require(selected[earlier_key].end <= selected[later_key].start,
                        f"{phase}: {earlier_key} and {later_key} overlap or are out of order")
        output[phase] = {key: z.duration / 1_000_000 for key, z in selected.items()}
    return output

def validate_metadata(meta, hw):
    require(meta.get("schema_version") == 1 and meta.get("completed") is True, "Incomplete benchmark metadata")
    require(meta.get("build_type") == "Release", "Benchmark requires a Release build; do not compare debug/sanitizer builds")
    require(meta.get("fabric_mode") == hw["fabric_mode"], "Wrong fabric mode")
    require(isinstance(meta.get("pid"), int), "Missing process ID")
    shapes = []
    device_ids = []
    for phase in PHASES:
        p = meta.get("phases", {}).get(phase, {})
        require(p.get("teardown_complete") is True, f"{phase}: teardown did not complete")
        require(p.get("arch") == hw["arch"] and p.get("devices") == hw["devices"], f"{phase}: hardware mismatch")
        shape = p.get("shape", [])
        require(len(shape) == 2 and math.prod(shape) == hw["devices"], f"{phase}: not a full 2D Galaxy mesh")
        require(p.get("artifacts_after", 0) > 0, f"{phase}: no compiled cache artifacts")
        ids = p.get("device_ids", [])
        require(len(ids) == hw["devices"] and len(set(ids)) == len(ids), f"{phase}: invalid physical device IDs")
        device_ids.append(sorted(ids))
        shapes.append(shape)
    require(shapes[0] == shapes[1], "Mesh shape changed between cold and hot")
    require(device_ids[0] == device_ids[1], "Physical device selection changed between cold and hot")
    require(meta["phases"]["cold"].get("artifacts_before") == 0, "Cold phase started with compiled artifacts")
    require(meta["phases"]["hot"].get("artifacts_before", 0) > 0, "Hot started with an empty cache")
    return shapes[0]

def finite_number(v, label, positive=False):
    require(type(v) in (float, int) and math.isfinite(v) and (v > 0 if positive else v >= 0),
            f"Invalid {label}: {v}")
    return float(v)

def compare(samples, baseline, identity, keys, mode):
    require(bool(samples), "No samples")
    require(baseline.get("schema_version") == 1, "Unsupported baseline schema")
    require(baseline.get("hardware") == identity["hardware"], "Wrong baseline hardware")
    stored_identity = baseline.get("identity")
    if stored_identity is not None:
        require(stored_identity == identity, "Baseline environment/measurement identity mismatch; review before re-baselining")
    elif mode == "enforce":
        raise MeasurementError("No baseline identity; collect/report first")
    if mode == "enforce":
        require(bool(baseline.get("provenance", {}).get("reference_commit")), "Baseline lacks reference provenance")
        require(bool(baseline.get("provenance", {}).get("approved_by")), "Baseline is not approved")
    rows = []
    for phase in PHASES:
        for key in keys:
            values = [s[phase][key] for s in samples]
            for v in values:
                finite_number(v, "sample duration", positive=True)
            actual = statistics.median(values)
            entry = baseline.get("phases", {}).get(phase, {}).get(key, {})
            ref = entry.get("median_ms")
            row = {"phase": phase, "zone": key, "median_ms": actual,
                   "min_ms": min(values), "max_ms": max(values), "samples_ms": values,
                   "baseline_ms": ref, "delta_ms": None, "delta_percent": None,
                   "limit_ms": None, "status": "UNBASELINED"}
            if ref is not None:
                ref = finite_number(ref, "baseline duration", positive=True)
                percent = finite_number(entry.get("max_regression_percent"), "relative tolerance")
                absolute = finite_number(entry.get("min_regression_ms"), "absolute allowance")
                allowance = max(ref * percent / 100, absolute)
                row.update(delta_ms=actual-ref, delta_percent=(actual/ref-1)*100, limit_ms=ref+allowance,
                           status="REGRESSION" if actual > ref+allowance else "PASS")
            elif mode == "enforce":
                raise MeasurementError(f"Missing baseline: {phase}/{key}")
            rows.append(row)
    return rows

def report(out, rows, mode, identity):
    lines = [f"# Fabric Init — {identity['hardware']} / FABRIC_2D", "",
             f"Mode: **{mode}**. Inclusive elapsed host-zone durations; milliseconds.", "",
             "| Phase | Zone | Median | Baseline | Delta % | Limit | Result |",
             "|---|---|---:|---:|---:|---:|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    suite = ET.Element("testsuite", name="fabric-init", tests=str(len(rows)))
    failures = skips = 0
    for r in rows:
        lines.append("| " + " | ".join([r["phase"], r["zone"], fmt(r["median_ms"]), fmt(r["baseline_ms"]),
                                       fmt(r["delta_percent"]), fmt(r["limit_ms"]), r["status"]]) + " |")
        case = ET.SubElement(suite, "testcase", classname=identity["hardware"] + "." + r["phase"],
                             name=r["zone"], time=str(r["median_ms"]/1000))
        if r["status"] == "REGRESSION" and mode == "enforce":
            ET.SubElement(case, "failure", message="Initialization latency regression").text = json.dumps(r)
            failures += 1
        elif r["status"] == "UNBASELINED":
            ET.SubElement(case, "skipped", message="No approved performance baseline; measurement was valid")
            skips += 1
    suite.set("failures", str(failures)); suite.set("skipped", str(skips))
    ET.ElementTree(suite).write(out / "junit.xml", encoding="utf-8", xml_declaration=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    dump(out / "comparison.json", {"identity": identity, "mode": mode, "rows": rows})
    with (out / "timings.csv").open("w", newline="") as f:
        fields = [k for k in rows[0] if k != "samples_ms"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k in fields} for r in rows)

def stop_group(proc, sig=signal.SIGTERM, timeout=10):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
        proc.wait(timeout=timeout)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)

def run_command(command, env, log, timeout):
    with Path(log).open("w") as f:
        proc = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        finally:
            stop_group(proc)
    require(rc == 0, f"Command failed ({rc}): {command}; see {log}")

def controlled_env(cache, build_dir, repo):
    env = os.environ.copy()
    for key in ("TT_METAL_CCACHE_KERNEL_SUPPORT", "CCACHE_REMOTE_STORAGE", "CCACHE_PREFIX", "TRACY_NO_EXIT"):
        env.pop(key, None)
    forbidden = [k for k in env if (k.startswith("TT_METAL_WATCHER") or k.startswith("TT_METAL_DPRINT"))]
    forbidden += [k for k in ("TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_MOCK_CLUSTER_DESC_PATH",
                              "TT_METAL_EMULE_MODE", "TT_METAL_SIMULATOR", "TT_METAL_NULL_KERNELS",
                              "TT_METAL_KERNELS_EARLY_RETURN", "TT_METAL_VISIBLE_DEVICES", "TT_VISIBLE_DEVICES",
                              "TT_MESH_ID", "TT_MESH_GRAPH_DESC_PATH", "TT_METAL_FABRIC_CONFIG") if k in env]
    require(not forbidden, "Remove conflicting benchmark environment: " + ", ".join(forbidden))

    # Update the environment
    env.update(CCACHE_DISABLE="1", TT_METAL_CACHE=str(cache), TT_METAL_DEVICE_PROFILER="0",
               TT_METAL_STREAMING_PROFILER="0", TT_METAL_HOME=str(repo), TT_METAL_RUNTIME_ROOT=str(repo))
    env["LD_LIBRARY_PATH"] = str(build_dir / "lib") + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env

def measure_cold_hot(args, hw, directory):
    """One process: cold open/close, then hot open/close, under a Tracy capture."""
    # Create the cache directory and set up the environment
    cache = directory / "cache"; cache.mkdir()
    env = controlled_env(cache, args.build_dir, args.repo)
    metadata_path = directory / "metadata.json"
    command = [str(args.binary), "--output", str(metadata_path), "--arch", hw["arch"], "--devices", str(hw["devices"])]
    trace = directory / "capture.tracy"

    # Bind to the capture tool
    with socket.socket() as sock:
        try: sock.bind(("127.0.0.1", 8086))
        except OSError as e: raise MeasurementError("Tracy port 8086 already occupied; benchmark requires exclusivity") from e
    collector = None
    with (directory / "capture.log").open("w") as log:
        try:
            collector = subprocess.Popen([str(args.capture), "-a", "127.0.0.1", "-o", str(trace)],
                                         env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            # Run the test
            # Binary waits for TracyIsConnected before any measured initialization.
            run_command(command, env, directory / "workload.log", args.timeout)
            try:
                collector.wait(timeout=30)
            except subprocess.TimeoutExpired:
                stop_group(collector, signal.SIGINT, timeout=60)
            require(collector.returncode == 0, f"Capture failed ({collector.returncode}); see {directory}/capture.log")
        finally:
            stop_group(collector, signal.SIGINT, timeout=30)
    require(trace.is_file() and trace.stat().st_size > 0, "Missing/empty Tracy capture")
    with (directory / "zones.csv").open("w") as csv_file, (directory / "export.log").open("w") as log:
        subprocess.run([str(args.exporter), "-u", str(trace)], env=env, stdout=csv_file,
                       stderr=log, timeout=120, check=True)
    metadata = json.loads(metadata_path.read_text())
    mesh_shape = validate_metadata(metadata, hw)
    return metadata, mesh_shape

def main(argv=None):
    # Arg parsing
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=["report", "enforce"], default="report")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)
    args.repo = Path.cwd().resolve()
    args.build_dir = (args.repo / "build").resolve()
    args.output = args.output.resolve()
    require(args.timeout > 0, "Positive --timeout required")
    args.output.mkdir(parents=True, exist_ok=True)
    require(not any(args.output.iterdir()), "Output directory must be empty, refusing to overwrite prior results")
    
    try:
        # Load config and baseline
        cfg = json.loads(CONFIG_PATH.read_text()); hw = cfg["hardware"][args.hardware]
        specs = validate_zone_specs(cfg["zones"])
        keys = [spec["key"] for spec in specs]
        baseline = json.loads((BASELINES_DIR / f"{args.hardware}.json").read_text())
        args.binary = args.build_dir / "test/tt_metal/tt_fabric/fabric_init_benchmark"
        args.capture = args.build_dir / "tools/profiler/bin/tracy-capture"
        args.exporter = args.build_dir / "tools/profiler/bin/tracy-csvexport"
        for p in (args.binary, args.capture, args.exporter):
            require(p.is_file() and os.access(p, os.X_OK), f"Missing executable: {p}")
        
        # Get commit sha for the provenance of this run, and write the provenance file
        commit_sha = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(args.repo), "diff", "--name-only"], text=True).strip())
        provenance = {"commit": commit_sha, "tracked_changes": dirty, "runner": os.environ.get("RUNNER_NAME", socket.gethostname()),
                      "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                      "build_artifact": os.environ.get("FABRIC_BUILDER_BUILD_ARTIFACT"),
                      "image": os.environ.get("FABRIC_BUILDER_DOCKER_IMAGE"),
                      "cache_policy": "fresh-job-cache; ccache disabled; same-process hot; preserve Metal cache",
                      "measurement_telemetry": "disabled"}
        dump(args.output / "provenance.json", provenance)

        # Run the fabric builder perf profiling test and extract the profiled zones
        _, shape = measure_cold_hot(args, hw, args.output)
        samples = [extract(read_zones(args.output / "zones.csv"), specs)]

        # CPU model and visible CPU budget guard comparisons across heterogeneous hosts.
        cpu_model = "unknown"
        cpu_info = Path("/proc/cpuinfo")
        if cpu_info.exists():
            cpu_model = next((s.split(":", 1)[1].strip() for s in cpu_info.read_text().splitlines()
                              if s.startswith("model name")), "unknown")
        identity = {"hardware": args.hardware, "arch": hw["arch"], "devices": hw["devices"],
                    "shape": shape, "fabric_mode": hw["fabric_mode"], "build_type": "Release",
                    "cpu_model": cpu_model, "cpu_affinity_count": len(os.sched_getaffinity(0)),
                    "zones": specs,
                    "cache_policy": provenance["cache_policy"], "measurement_telemetry": "disabled"}
        dump(args.output / "measurements.json", {"identity": identity, "samples": samples, "provenance": provenance})
        rows = compare(samples, baseline, identity, keys, args.mode)
        report(args.output, rows, args.mode, identity)
        # Candidate is never installed automatically; approval and tolerances are still required.
        candidate = {"schema_version": 1, "hardware": args.hardware, "identity": identity,
                     "provenance": {"reference_commit": commit_sha, "approved_by": None},
                     "phases": {p: {} for p in PHASES}}
        for r in rows:
            candidate["phases"][r["phase"]][r["zone"]] = {
                "median_ms": r["median_ms"], "max_regression_percent": None, "min_regression_ms": None}
        dump(args.output / "baseline-candidate.json", candidate)
        return 1 if args.mode == "enforce" and any(r["status"] == "REGRESSION" for r in rows) else 0
    except Exception as e:
        dump(args.output / "error.json", {"status": "MEASUREMENT_ERROR", "error": str(e)})
        (args.output / "summary.md").write_text(f"# Fabric Init measurement error\n\n{e}\n")
        suite = ET.Element("testsuite", name="fabric-init", tests="1", errors="1")
        case = ET.SubElement(suite, "testcase", name="measurement-integrity")
        ET.SubElement(case, "error", message=str(e))
        ET.ElementTree(suite).write(args.output / "junit.xml", encoding="utf-8", xml_declaration=True)
        print(str(e), file=sys.stderr)
        return 2

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (MeasurementError, KeyboardInterrupt) as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
