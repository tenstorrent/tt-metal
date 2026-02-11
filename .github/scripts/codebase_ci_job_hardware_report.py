#!/usr/bin/env python3
"""
Generate a JSON report mapping CI jobs to the hardware they run on and the
tests they execute.

Focuses on *-impl.yaml workflows (the ones that actually run tests on hardware)
plus a handful of non-impl files that run directly on hardware.

Usage:
    python codebase_ci_job_hardware_report.py [--output FILE] [--repo-root PATH]
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Machine-type resolution
# ---------------------------------------------------------------------------

# runner-label → canonical machine
_LABEL_MACHINE_PATTERNS: list[tuple[str, str]] = [
    (r"^P100a?(?:-viommu)?$", "P100"),
    (r"^P150b?(?:-viommu)?$", "P150"),
    (r"^P300(?:-viommu)?$", "P300"),
    (r"^N150(?:-viommu)?$", "N150"),
    (r"^N300(?:-viommu|-llmbox)?$", "N300"),
    (r"^BH-LLMBox$", "Galaxy (Blackhole)"),
    (r"^BH-DeskBox$", "Galaxy (Blackhole)"),
    (r"^BH-LoudBox$", "Galaxy (Blackhole)"),
    (r"^BH-QB-GE$", "Galaxy (Blackhole)"),
    (r"^bh-llmbox$", "Galaxy (Blackhole)"),
    (r"^bh-loudbox$", "Galaxy (Blackhole)"),
    (r"^bh-deskbox$", "Galaxy (Blackhole)"),
    (r"^bh-qb-ge$", "Galaxy (Blackhole)"),
]

# SKU names from tests/pipeline_reorg YAML → machine type
SKU_TO_MACHINE: dict[str, str] = {
    "wh_galaxy": "Galaxy (Wormhole)",
    "bh_galaxy": "Galaxy (Blackhole)",
    "wh_llmbox": "T3K (Wormhole)",
    "N300": "N300",
    "N150": "N150",
}


def _label_to_machine(label: str) -> str | None:
    """Map a single runner label string to a machine type."""
    label = label.strip()
    for pat, mach in _LABEL_MACHINE_PATTERNS:
        if re.match(pat, label, re.IGNORECASE):
            return mach
    # tt-ubuntu-2204-<inner>-stable
    m = re.match(r"^tt-ubuntu-2204-(.+?)(?:-stable)?$", label, re.IGNORECASE)
    if m:
        return _label_to_machine(m.group(1))
    return None


def machines_from_labels(labels: list[str]) -> list[str]:
    """Derive machine types from a runs-on label array."""
    machines: set[str] = set()
    low = {str(l).lower() for l in labels}
    for lbl in labels:
        m = _label_to_machine(str(lbl))
        if m:
            machines.add(m)
    if "config-t3000" in low:
        machines.add("T3K (Wormhole)" if "arch-wormhole_b0" in low else "T3K (Blackhole)")
    if "topology-6u" in low:
        if "arch-wormhole_b0" in low:
            machines.add("Galaxy (Wormhole)")
        if "arch-blackhole" in low:
            machines.add("Galaxy (Blackhole)")
    if not machines:
        if "arch-blackhole" in low:
            machines.add("Blackhole (generic)")
        if "arch-wormhole_b0" in low:
            machines.add("Wormhole (generic)")
    return sorted(machines)


def machines_from_runs_on(runs_on: Any) -> list[str]:
    """Best-effort machine resolution from a runs-on value."""
    if isinstance(runs_on, list):
        flat = [str(x) for x in runs_on if not str(x).startswith("${{")]
        return machines_from_labels(flat)
    s = str(runs_on).strip()
    if "ubuntu" in s.lower():
        return []
    m = _label_to_machine(s)
    if m:
        return [m]
    return machines_from_labels([s])


# ---------------------------------------------------------------------------
# Test extraction helpers
# ---------------------------------------------------------------------------

_NON_TEST_PATTERNS = re.compile(
    r"^(source |set |mkdir |cd |wget |pip |echo |ls |cat |tar |cp |mv |"
    r"python3 \.github/|sudo |\.github/|export |if |then |else |fi$|"
    r"\.\/build\/tools\/scaleout\/run_cluster_validation|"
    r"\.\/build\/tools\/scaleout\/run_fabric_manager|"
    r"python3 models/perf/|"
    r"for |done$|do |sleep |\{|^\})",
    re.IGNORECASE,
)

# pytest flags that take a value argument (the next token is their value, not a test path)
_PYTEST_VALUE_FLAGS = {
    "-k",
    "-m",
    "-x",
    "-p",
    "--timeout",
    "--tb",
    "--co",
    "-W",
    "--rootdir",
    "--override-ini",
    "-c",
    "-o",
    "--import-mode",
    "--basetemp",
    "--confcutdir",
    "-n",
    "--dist",
    "--data_parallel",
    "--max_seq_len",
    "--max_generated_tokens",
    "--use_prefetcher",
    "--durations",
    "--durations-min",
}


def _parse_cmd(cmd_str: str) -> list[str]:
    """
    Parse a test command string into human-readable test descriptions.

    Handles:
      - ENV=val ENV2=val2 pytest path -k filter
      - Multiple commands chained with &&, newlines, or semicolons
      - Shell function names (run_t3000_mixtral_tests)
      - gtest binaries (./build/test/...)
      - for loops (collapsed to single lines)
    """
    tests: list[str] = []
    # First collapse for loops to avoid splitting them
    collapsed = re.sub(r"for\s+\w+\s+in\s+[^;]+;\s*do\s+(.+?)\s*;\s*done", r"\1", cmd_str)
    # Split on &&, newlines, and standalone semicolons
    for line in re.split(r"\s*&&\s*|\n|(?<=[^{])\s*;\s*", collapsed):
        line = line.strip().rstrip(";").strip()
        if not line:
            continue
        # Strip leading env var assignments  (KEY=value ...)
        cleaned = re.sub(r"^(\s*\w+=\S+\s+)+", "", line).strip()
        if not cleaned:
            continue
        if _NON_TEST_PATTERNS.search(cleaned):
            continue
        if "pytest" in cleaned:
            tests.extend(_parse_pytest(cleaned))
        elif "pytest" in line and "pytest" not in cleaned:
            # env vars hid pytest; parse full line
            tests.extend(_parse_pytest(line))
        elif re.search(r"\.?/build/|^build/|gtest|ctest|^tt-run ", cleaned):
            parsed = _parse_binary(cleaned)
            if parsed:
                tests.append(parsed)
        elif re.match(r"^run_\w+", cleaned):
            tests.append(cleaned.split()[0])
        elif re.match(r"^\.?/?tests/|^models/", cleaned):
            # bare test path (e.g. "tests/ttnn/unit_tests/.../nightly")
            tests.append(cleaned.split()[0])
    return list(dict.fromkeys(t for t in tests if t))


def _parse_pytest(cmd: str) -> list[str]:
    """Extract test paths from a pytest command, handling flags before path."""
    results = []
    # Strip env vars from the front
    stripped = re.sub(r"^(\s*\w+=\S+\s+)+", "", cmd).strip()
    parts = stripped.split()
    i = 0
    while i < len(parts):
        if parts[i] == "pytest":
            # Scan forward past flags to find test path(s)
            path = None
            k_filter = None
            j = i + 1
            while j < len(parts):
                token = parts[j]
                if token == "-k" and j + 1 < len(parts):
                    k_filter = parts[j + 1].strip("\"'")
                    j += 2
                    continue
                # Check if this is a flag with a value
                flag_base = token.split("=")[0] if "=" in token else token
                if flag_base in _PYTEST_VALUE_FLAGS and "=" not in token:
                    j += 2  # skip flag and its value
                    continue
                if token.startswith("-"):
                    j += 1  # skip boolean flag or --flag=value
                    continue
                # First non-flag token is the test path
                if path is None:
                    path = token.strip("\"'")
                j += 1
            if path:
                desc = path
                if k_filter:
                    desc = f"{path} -k {k_filter}"
                results.append(desc)
            i = j
        else:
            i += 1
    return results


def _parse_binary(cmd: str) -> str:
    """Summarize a gtest/binary command."""
    # Strip env vars
    cleaned = re.sub(r"^(\s*\w+=\S+\s+)+", "", cmd).strip()
    parts = cleaned.split()
    if not parts:
        return ""
    binary = parts[0].rstrip(";")
    gtest_filter = None
    for k, p in enumerate(parts):
        p = p.rstrip(";").strip('"')
        if p.startswith("--gtest_filter="):
            gtest_filter = p.split("=", 1)[1].strip("\"';")
        elif p == "--gtest_filter" and k + 1 < len(parts):
            gtest_filter = parts[k + 1].strip("\"';")
    if gtest_filter:
        return f"{binary} --gtest_filter={gtest_filter}"
    # Return the binary + key args, drop --timeout values
    result = re.sub(r"\s*--timeout[= ]\d+", "", cleaned).strip()
    return result[:200]


def extract_tests_from_matrix(job: dict) -> list[str]:
    """Extract tests from strategy.matrix.test-group[].cmd (or similar)."""
    tests: list[str] = []
    strategy = job.get("strategy") or {}
    matrix = strategy.get("matrix")
    if not isinstance(matrix, dict):
        return tests
    # Look for test-group, test-info, test-config — different workflows use different names
    for key in ("test-group", "test-info", "test-config"):
        groups = matrix.get(key)
        if not isinstance(groups, list):
            continue
        for g in groups:
            if not isinstance(g, dict):
                continue
            cmd = g.get("cmd") or g.get("run-args") or g.get("commands") or ""
            if cmd:
                tests.extend(_parse_cmd(str(cmd)))
    return list(dict.fromkeys(tests))


_STEP_SKIP_NAMES = (
    "checkout",
    "setup",
    "cleanup",
    "upload",
    "report",
    "download",
    "install",
    "workaround",
    "check for",
    "check if",
    "extract",
    "print tt-smi",
    "save environment",
    "disable performance",
    "enable performance",
    "generate gtest annotations",
    "download",
    "benchmark",
    "check data",
    "upload latency",
    "upload benchmark",
)


def extract_tests_from_steps(job: dict) -> list[str]:
    """Extract tests from step run: blocks."""
    tests: list[str] = []
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        run_block = step.get("run")
        if not run_block or not isinstance(run_block, str):
            continue
        name = (step.get("name") or "").lower()
        if any(skip in name for skip in _STEP_SKIP_NAMES):
            continue
        # Also skip if the run block is just ${{ matrix.test-group.cmd }} — tests come from matrix
        if re.match(r"^\s*\$\{\{.*\}\}\s*$", run_block.strip()):
            continue
        tests.extend(_parse_cmd(run_block))
    return list(dict.fromkeys(tests))


def extract_tests_from_external_yaml(repo_root: Path, workflow_text: str) -> list[str]:
    """
    When a workflow uses TESTS_YAML_PATH, load that external YAML
    and extract name + cmd from every entry.
    """
    m = re.search(r"TESTS_YAML_PATH:\s*(\S+)", workflow_text)
    if not m:
        return []
    rel_path = m.group(1).strip("\"'").lstrip("./")
    yaml_path = repo_root / rel_path
    if not yaml_path.exists():
        return []
    data = _load_yaml(yaml_path)
    if not isinstance(data, list):
        return []
    tests: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        cmd = entry.get("cmd", "")
        if cmd:
            parsed = _parse_cmd(str(cmd))
            if parsed:
                tests.extend(parsed)
            else:
                # Shell function name (run_t3000_mixtral_tests) or bare name
                tests.append(str(cmd).split()[0] if cmd else name)
    return list(dict.fromkeys(tests))


def sku_from_external_yaml(repo_root: Path, workflow_text: str) -> str | None:
    """Read the sku field from the first entry in the external tests YAML."""
    m = re.search(r"TESTS_YAML_PATH:\s*(\S+)", workflow_text)
    if not m:
        return None
    rel_path = m.group(1).strip("\"'").lstrip("./")
    yaml_path = repo_root / rel_path
    if not yaml_path.exists():
        return None
    data = _load_yaml(yaml_path)
    if not isinstance(data, list) or not data:
        return None
    return data[0].get("sku") if isinstance(data[0], dict) else None


# ---------------------------------------------------------------------------
# YAML / utility
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict | list | None:
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _to_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return json.dumps(val)
    return str(val)


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start


# ---------------------------------------------------------------------------
# Caller resolution (which top-level workflows trigger each impl)
# ---------------------------------------------------------------------------


def build_caller_map(repo_root: Path) -> dict[str, list[dict]]:
    """
    Build a map of impl_filename → list of {workflow, job_id, runner_values}.
    Computed once for all workflows.
    """
    wf_dir = repo_root / ".github" / "workflows"
    caller_map: dict[str, list[dict]] = {}
    for wf_path in sorted(wf_dir.glob("*.yaml")) + sorted(wf_dir.glob("*.yml")):
        wf = _load_yaml(wf_path)
        if not wf or not isinstance(wf, dict):
            continue
        for job_id, job_def in (wf.get("jobs") or {}).items():
            if not isinstance(job_def, dict):
                continue
            uses = str(job_def.get("uses", ""))
            if not uses:
                continue
            # Extract the target filename from the uses path
            target = uses.split("@")[0].strip()
            if "/" in target:
                target = target.rsplit("/", 1)[-1]
            target = target.strip("./")
            if not target:
                continue
            runner_vals = _get_runner_values_from_caller(job_def)
            caller_map.setdefault(target, []).append(
                {
                    "workflow": wf_path.name,
                    "job_id": job_id,
                    "runner_values": runner_vals,
                }
            )
    return caller_map


def _get_runner_values_from_caller(caller_job: dict) -> list[str]:
    """Extract runner-label or runner values from a caller job's with + matrix."""
    with_block = caller_job.get("with") or {}
    strategy = caller_job.get("strategy") or {}
    matrix = strategy.get("matrix") or {}
    for inp_name in ("runner-label", "runner"):
        val = with_block.get(inp_name)
        if val is None:
            continue
        raw = _to_str(val)
        # matrix.XXX.YYY
        m = re.search(r"matrix\.([\w.-]+)\.([\w.-]+)", raw)
        if m and isinstance(matrix, dict):
            mk, field = m.group(1), m.group(2)
            inner = matrix.get(mk)
            if isinstance(inner, list):
                return [str(e[field]) for e in inner if isinstance(e, dict) and field in e]
        # matrix.XXX (flat list)
        m2 = re.search(r"matrix\.([\w.-]+)", raw)
        if m2 and not m and isinstance(matrix, dict):
            inner = matrix.get(m2.group(1))
            if isinstance(inner, list):
                return [str(x) for x in inner]
        # Static value
        cleaned = re.sub(r"\$\{\{.+?\}\}", "", raw).strip()
        if cleaned and not re.match(r"^(inputs\.|needs\.|github\.|matrix\.)", cleaned):
            return [cleaned]
    # Fallback: look for card_type / platform in matrix
    for key in ("card_type", "platform", "runner-label"):
        if isinstance(matrix, dict) and isinstance(matrix.get(key), list):
            return [str(x) for x in matrix[key]]
    return []


def apply_job_if_filter(job: dict, runner_values: list[str]) -> list[str]:
    """Apply job's if: to filter which runner values actually apply."""
    if_block = _to_str(job.get("if", ""))
    if not if_block:
        return runner_values
    matches = re.findall(r"inputs\.(?:runner-label|runner)\s*==\s*['\"]([^'\"]+)['\"]", if_block)
    if matches:
        return [v for v in runner_values if any(m.lower() == v.lower() for m in matches)]
    return runner_values


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_impl_workflows(repo_root: Path) -> list[dict]:
    """Process all impl workflows and return job entries."""
    wf_dir = repo_root / ".github" / "workflows"
    results: list[dict] = []
    caller_map = build_caller_map(repo_root)

    for wf_path in sorted(wf_dir.glob("*-impl.yaml")) + sorted(wf_dir.glob("*-impl.yml")):
        wf = _load_yaml(wf_path)
        if not wf or not isinstance(wf, dict):
            continue
        wf_text = wf_path.read_text(encoding="utf-8")
        callers = caller_map.get(wf_path.name, [])

        # External YAML tests (loaded once per workflow)
        ext_tests = extract_tests_from_external_yaml(repo_root, wf_text) if "TESTS_YAML_PATH" in wf_text else []
        ext_sku = sku_from_external_yaml(repo_root, wf_text) if "TESTS_YAML_PATH" in wf_text else None

        for job_id, job_def in (wf.get("jobs") or {}).items():
            if not isinstance(job_def, dict):
                continue
            runs_on = job_def.get("runs-on")
            if runs_on is None:
                continue
            ro_str = _to_str(runs_on)
            if "ubuntu-latest" in ro_str or "ubuntu-22.04" in ro_str:
                continue
            # Skip jobs that call other workflows (they're orchestrators, not test runners)
            if job_def.get("uses"):
                continue

            # --- Machines ---
            machines = machines_from_runs_on(runs_on)

            # If runs-on references inputs.runner-label, resolve from callers
            if not machines or "inputs.runner-label" in ro_str or "inputs.runner" in ro_str:
                all_runner_vals: list[str] = []
                for c in callers:
                    filtered = apply_job_if_filter(job_def, c["runner_values"])
                    all_runner_vals.extend(filtered)
                if all_runner_vals:
                    for v in all_runner_vals:
                        m = _label_to_machine(v)
                        if m and m not in machines:
                            machines.append(m)
                    if not machines:
                        # Try substituting into the label array
                        if isinstance(runs_on, list):
                            resolved = []
                            for lbl in runs_on:
                                s = str(lbl)
                                if "inputs.runner-label" in s or "inputs.runner" in s:
                                    resolved.extend(all_runner_vals)
                                else:
                                    resolved.append(s)
                            machines = machines_from_labels(resolved)

            # If external YAML has a sku, use that for machine mapping
            if ext_sku and not machines:
                m = SKU_TO_MACHINE.get(ext_sku)
                if m:
                    machines = [m]

            if not machines:
                continue

            # --- Tests ---
            tests = extract_tests_from_matrix(job_def)
            if not tests:
                tests = extract_tests_from_steps(job_def)
            if not tests and ext_tests:
                tests = list(ext_tests)  # copy

            # --- triggered_from ---
            triggered_from = [f"{c['workflow']}:{c['job_id']}" for c in callers]
            if not triggered_from:
                triggered_from = [f"{wf_path.name}:{job_id}"]

            results.append(
                {
                    "job_name": job_id,
                    "workflow_file": wf_path.name,
                    "machines": sorted(set(machines)),
                    "tests": tests,
                    "triggered_from": triggered_from,
                }
            )

    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="CI job → hardware + tests report")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root or _find_repo_root(Path(__file__).resolve().parent)
    if not (repo_root / ".github" / "workflows").exists():
        print("::error::.github/workflows not found", file=sys.stderr)
        return 1

    all_jobs = process_impl_workflows(repo_root)

    # Deduplicate by (job_name, workflow_file), merging machines + triggered_from
    by_key: dict[tuple[str, str], dict] = {}
    for j in all_jobs:
        key = (j["job_name"], j["workflow_file"])
        if key not in by_key:
            by_key[key] = {**j}
        else:
            entry = by_key[key]
            for m in j["machines"]:
                if m not in entry["machines"]:
                    entry["machines"].append(m)
            for t in j.get("triggered_from", []):
                if t not in entry["triggered_from"]:
                    entry["triggered_from"].append(t)
    for v in by_key.values():
        v["machines"] = sorted(set(v["machines"]))

    output = {
        "jobs": list(by_key.values()),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    text = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(by_key)} jobs to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
