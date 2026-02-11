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
    if "config-tg" in low:
        machines.add("Galaxy (Wormhole)" if "arch-wormhole_b0" in low else "Galaxy (Blackhole)")
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


def _get_input_defaults(wf: dict) -> dict[str, str]:
    """Get default values for workflow_call inputs."""
    on_block = wf.get("on") or wf.get(True) or {}
    wc = on_block.get("workflow_call") or {}
    inputs_def = wc.get("inputs") or {}
    defaults: dict[str, str] = {}
    for name, defn in inputs_def.items():
        if isinstance(defn, dict) and "default" in defn:
            defaults[str(name)] = str(defn["default"])
    return defaults


def _resolve_label(
    label: str,
    input_defaults: dict[str, str],
    caller_inputs: dict[str, list[str]],
    job_matrix: dict,
    wf: dict | None = None,
    wf_text: str = "",
) -> list[str]:
    """Resolve a single ${{ ... }} expression in a runs-on label to concrete values."""
    if "${{" not in label:
        return [label]

    # ${{ inputs.X }}
    m = re.match(r"^\$\{\{\s*inputs\.([\w-]+)\s*\}\}$", label)
    if m:
        name = m.group(1)
        vals = caller_inputs.get(name, [])
        if vals:
            return vals
        if name in input_defaults:
            return [input_defaults[name]]
        return []

    # ${{ matrix.X.Y }}
    m = re.match(r"^\$\{\{\s*matrix\.([\w-]+)\.([\w-]+)\s*\}\}$", label)
    if m:
        mk, field = m.group(1), m.group(2)
        inner = job_matrix.get(mk)
        if isinstance(inner, list):
            vals: list[str] = []
            for entry in inner:
                if isinstance(entry, dict) and field in entry:
                    v = entry[field]
                    if isinstance(v, list):
                        vals.extend(str(x) for x in v)
                    else:
                        vals.append(str(v))
            return list(dict.fromkeys(vals))
        # Dynamic matrix (fromJSON/fromJson) - try to find runner labels
        if isinstance(inner, str) and "fromjson" in inner.lower():
            if field == "runner-label" and wf:
                # Try to find the specific generator job referenced in fromJSON
                gen_match = re.search(r"needs\.([\w-]+)\.", inner)
                found: list[str] = []
                if gen_match:
                    gen_job_id = gen_match.group(1)
                    gen_job = (wf.get("jobs") or {}).get(gen_job_id, {})
                    if isinstance(gen_job, dict):
                        # Search the generator job's step texts for runner labels
                        for step in gen_job.get("steps") or []:
                            if isinstance(step, dict):
                                run_text = step.get("run", "")
                                if run_text:
                                    found.extend(_extract_runner_labels_from_text(str(run_text)))
                        # Also check the generator job's matrix
                        gen_matrix = (gen_job.get("strategy") or {}).get("matrix")
                        if isinstance(gen_matrix, dict):
                            found.extend(_find_runner_labels_in_matrix(gen_matrix))
                if not found and wf_text:
                    # Broad text fallback (less precise)
                    found = _extract_runner_labels_from_text(wf_text)
                if found:
                    return list(dict.fromkeys(found))
        return []

    # ${{ matrix.X }} (flat list)
    m = re.match(r"^\$\{\{\s*matrix\.([\w-]+)\s*\}\}$", label)
    if m:
        mk = m.group(1)
        inner = job_matrix.get(mk)
        if isinstance(inner, list):
            return [str(x) for x in inner if not isinstance(x, (dict, list))]
        return []

    return []


def _find_runner_labels_in_matrix(matrix: dict) -> list[str]:
    """Find runner-label values in a matrix definition."""
    vals: list[str] = []
    for key in ("test-group", "test-info", "test-config", "default-demo-tests"):
        inner = matrix.get(key)
        if isinstance(inner, list):
            for entry in inner:
                if isinstance(entry, dict) and "runner-label" in entry:
                    vals.append(str(entry["runner-label"]))
                # Nested lists like [[{...}, {...}]]
                if isinstance(entry, list):
                    for sub in entry:
                        if isinstance(sub, dict) and "runner-label" in sub:
                            vals.append(str(sub["runner-label"]))
    return list(dict.fromkeys(vals))


def _find_runner_labels_in_workflow(wf: dict) -> list[str]:
    """Search ALL jobs in a workflow for runner-label values in their matrices."""
    vals: list[str] = []
    for job_id, job_def in (wf.get("jobs") or {}).items():
        if not isinstance(job_def, dict):
            continue
        strategy = job_def.get("strategy") or {}
        matrix = strategy.get("matrix")
        if isinstance(matrix, dict):
            vals.extend(_find_runner_labels_in_matrix(matrix))
    return list(dict.fromkeys(vals))


def resolve_machines_for_job(
    runs_on: Any,
    wf: dict,
    job_def: dict,
    callers: list[dict],
    wf_text: str = "",
) -> list[str]:
    """
    Comprehensive machine resolution for a job.

    Handles:
    - Static runs-on labels
    - ${{ inputs.X }} resolved from input defaults + caller values
    - ${{ matrix.X }} / ${{ matrix.X.Y }} resolved from the job's own matrix
    - ${{ matrix.X.runs-on }} where the matrix entry's runs-on is itself an array
    - Complex conditional expressions referencing matrix.test-group.runner-label
    - format() expressions
    """
    input_defaults = _get_input_defaults(wf)

    # Merge caller input values
    caller_inputs: dict[str, list[str]] = {}
    for c in callers:
        for name, vals in c.get("input_values", {}).items():
            caller_inputs.setdefault(name, []).extend(vals)
    caller_inputs = {k: list(dict.fromkeys(v)) for k, v in caller_inputs.items()}

    # Get job's own matrix
    strategy = job_def.get("strategy") or {}
    raw_matrix = strategy.get("matrix")
    job_matrix = raw_matrix if isinstance(raw_matrix, dict) else {}

    all_machines: set[str] = set()

    if isinstance(runs_on, list):
        # Resolve each label in the array, flattening multi-valued expansions
        resolved_labels: list[str] = []
        for lbl in runs_on:
            s = str(lbl)
            if "${{" not in s:
                resolved_labels.append(s)
            else:
                vals = _resolve_label(s, input_defaults, caller_inputs, job_matrix, wf=wf, wf_text=wf_text)
                resolved_labels.extend(vals)
        all_machines.update(machines_from_labels(resolved_labels))

    elif isinstance(runs_on, str):
        s = runs_on.strip()
        # Only skip plain ubuntu runners, not complex expressions that happen to mention ubuntu
        if "${{" not in s and "ubuntu" in s.lower():
            return []
        if "${{" not in s:
            return machines_from_labels([s])

        # Case 1: ${{ matrix.X.runs-on }} — matrix entry has a runs-on array
        m = re.search(r"matrix\.([\w-]+)\.(runs-on)", s)
        if m:
            mk, field = m.group(1), m.group(2)
            inner = job_matrix.get(mk)
            if isinstance(inner, list):
                for entry in inner:
                    if isinstance(entry, dict) and field in entry:
                        v = entry[field]
                        if isinstance(v, list):
                            nested_resolved: list[str] = []
                            for nested_lbl in v:
                                ns = str(nested_lbl)
                                if "${{" not in ns:
                                    nested_resolved.append(ns)
                                else:
                                    nested_resolved.extend(
                                        _resolve_label(
                                            ns,
                                            input_defaults,
                                            caller_inputs,
                                            {},
                                            wf=wf,
                                            wf_text=wf_text,
                                        )
                                    )
                            all_machines.update(machines_from_labels(nested_resolved))
            elif (isinstance(inner, str) and "fromjson" in inner.lower()) or (
                inner is None and isinstance(raw_matrix, str) and "fromjson" in raw_matrix.lower()
            ):
                # Dynamic matrix — search the generator job's text for runs-on arrays
                from_json_str = inner if isinstance(inner, str) else raw_matrix
                gen_match = re.search(r"needs\.([\w-]+)\.", from_json_str)
                if gen_match:
                    gen_job = (wf.get("jobs") or {}).get(gen_match.group(1), {})
                    if isinstance(gen_job, dict):
                        for step in gen_job.get("steps") or []:
                            if isinstance(step, dict):
                                run_text = step.get("run", "")
                                if not run_text:
                                    continue
                                # Find JSON-like runs-on arrays in the text
                                for ro_m in re.finditer(r'"runs-on"\s*:\s*\[([^\]]+)\]', str(run_text)):
                                    labels_str = ro_m.group(1)
                                    labels = [lbl.strip().strip("\"'") for lbl in labels_str.split(",")]
                                    resolved: list[str] = []
                                    for lbl in labels:
                                        if "${{" not in lbl:
                                            resolved.append(lbl)
                                        else:
                                            resolved.extend(
                                                _resolve_label(
                                                    lbl,
                                                    input_defaults,
                                                    caller_inputs,
                                                    {},
                                                    wf=wf,
                                                    wf_text=wf_text,
                                                )
                                            )
                                    all_machines.update(machines_from_labels(resolved))

        # Case 2: Complex expression referencing matrix.X.runner-label
        if not all_machines and "runner-label" in s:
            runner_vals: list[str] = []
            if "matrix" in s:
                runner_vals = _find_runner_labels_in_matrix(job_matrix)
                if not runner_vals:
                    runner_vals = _find_runner_labels_in_workflow(wf)
            if not runner_vals:
                runner_vals = caller_inputs.get("runner-label", []) or caller_inputs.get("runner", [])

            for v in runner_vals:
                m2 = _label_to_machine(v)
                if m2:
                    all_machines.add(m2)
                # Also try format('tt-ubuntu-2204-{0}-stable', val)
                if "format(" in s:
                    fmt_m = re.search(r"format\(\s*'([^']+)'", s)
                    if fmt_m:
                        label = fmt_m.group(1).replace("{0}", v)
                        m3 = _label_to_machine(label)
                        if m3:
                            all_machines.add(m3)

        # Case 3: Expression references inputs.runner-label / inputs.runner
        if not all_machines and ("inputs.runner-label" in s or "inputs.runner" in s):
            inp_key = "runner-label" if "inputs.runner-label" in s else "runner"
            runner_vals2 = caller_inputs.get(inp_key, [])
            if not runner_vals2 and inp_key in input_defaults:
                runner_vals2 = [input_defaults[inp_key]]
            for v in runner_vals2:
                m4 = _label_to_machine(v)
                if m4:
                    all_machines.add(m4)

    # Backward-compat: try the old caller runner_values path with if-filter
    if not all_machines:
        ro_str = _to_str(runs_on)
        if "inputs.runner-label" in ro_str or "inputs.runner" in ro_str:
            for c in callers:
                filtered = apply_job_if_filter(job_def, c.get("runner_values", []))
                for v in filtered:
                    m5 = _label_to_machine(v)
                    if m5:
                        all_machines.add(m5)
                # Also try substituting into the label array
                if not all_machines and isinstance(runs_on, list):
                    resolved: list[str] = []
                    for lbl in runs_on:
                        lbl_s = str(lbl)
                        if "inputs.runner-label" in lbl_s or "inputs.runner" in lbl_s:
                            resolved.extend(filtered)
                        elif "${{" not in lbl_s:
                            resolved.append(lbl_s)
                    if resolved:
                        all_machines.update(machines_from_labels(resolved))

    return sorted(all_machines)


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


def extract_tests_from_workflow_matrices(wf: dict) -> list[str]:
    """
    Fallback: extract tests from ALL matrix definitions in the workflow.
    Used when a job's own matrix is dynamic (fromJSON referencing another job's output).
    """
    tests: list[str] = []
    for _jid, jdef in (wf.get("jobs") or {}).items():
        if not isinstance(jdef, dict):
            continue
        strategy = jdef.get("strategy") or {}
        matrix = strategy.get("matrix")
        if not isinstance(matrix, dict):
            continue
        for _key, entries in matrix.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                # Handle nested lists [[{...}, ...]]
                if isinstance(entry, list):
                    for sub in entry:
                        if isinstance(sub, dict):
                            cmd = sub.get("cmd", "")
                            if cmd:
                                tests.extend(_parse_cmd(str(cmd)))
                elif isinstance(entry, dict):
                    cmd = entry.get("cmd") or entry.get("run-args") or entry.get("commands") or ""
                    if cmd:
                        tests.extend(_parse_cmd(str(cmd)))
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


def _get_all_input_values_from_caller(caller_job: dict) -> dict[str, list[str]]:
    """Extract ALL input values from a caller job's with: block, expanding matrix refs."""
    with_block = caller_job.get("with") or {}
    strategy = caller_job.get("strategy") or {}
    matrix = strategy.get("matrix") or {}
    if not isinstance(matrix, dict):
        matrix = {}

    result: dict[str, list[str]] = {}
    for inp_name, raw_val in with_block.items():
        raw = _to_str(raw_val)
        vals: list[str] = []

        # matrix.X.Y reference
        m = re.search(r"matrix\.([\w.-]+)\.([\w.-]+)", raw)
        if m:
            mk, field = m.group(1), m.group(2)
            inner = matrix.get(mk)
            if isinstance(inner, list):
                vals = [str(e[field]) for e in inner if isinstance(e, dict) and field in e]

        # matrix.X reference (flat list)
        if not vals:
            m2 = re.search(r"matrix\.([\w.-]+)", raw)
            if m2:
                inner = matrix.get(m2.group(1))
                if isinstance(inner, list):
                    vals = [str(x) for x in inner if not isinstance(x, (dict, list))]

        # Static value
        if not vals:
            cleaned = re.sub(r"\$\{\{.+?\}\}", "", raw).strip()
            if cleaned and not re.match(r"^(inputs\.|needs\.|github\.|matrix\.)", cleaned):
                vals = [cleaned]

        if vals:
            result[str(inp_name)] = vals

    return result


# Regex to find quoted machine labels in workflow text
_MACHINE_LABEL_TEXT_RE = re.compile(r"""(?:"|')([NP]\d{3}[ab]?(?:-viommu)?|BH-[\w-]+)(?:"|')""")


def _extract_runner_labels_from_text(text: str) -> list[str]:
    """Find machine label values from workflow text (for dynamic matrix fallback)."""
    vals: list[str] = []
    for m in _MACHINE_LABEL_TEXT_RE.finditer(text):
        v = m.group(1)
        if _label_to_machine(v):
            vals.append(v)
    return list(dict.fromkeys(vals))


def build_caller_map(repo_root: Path) -> dict[str, list[dict]]:
    """
    Build a map of impl_filename → list of {workflow, job_id, runner_values, input_values}.
    Computed once for all workflows.
    """
    wf_dir = repo_root / ".github" / "workflows"
    caller_map: dict[str, list[dict]] = {}
    for wf_path in sorted(wf_dir.glob("*.yaml")) + sorted(wf_dir.glob("*.yml")):
        wf = _load_yaml(wf_path)
        if not wf or not isinstance(wf, dict):
            continue
        wf_text: str | None = None  # Lazy-load
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
            input_vals = _get_all_input_values_from_caller(job_def)

            # If runner-label couldn't be resolved (dynamic matrix), try text-based extraction
            with_block = job_def.get("with") or {}
            rl_raw = _to_str(with_block.get("runner-label", ""))
            if "runner-label" not in input_vals and not runner_vals and rl_raw and "matrix" in rl_raw:
                # First try structured: search all matrices in the caller workflow
                found = _find_runner_labels_in_workflow(wf)
                if not found:
                    # Text-based fallback: search for known machine labels in the workflow text
                    if wf_text is None:
                        wf_text = wf_path.read_text(encoding="utf-8")
                    found = _extract_runner_labels_from_text(wf_text)
                if found:
                    input_vals["runner-label"] = found
                    runner_vals = found

            caller_map.setdefault(target, []).append(
                {
                    "workflow": wf_path.name,
                    "job_id": job_id,
                    "runner_values": runner_vals,
                    "input_values": input_vals,
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
            machines = resolve_machines_for_job(runs_on, wf, job_def, callers, wf_text=wf_text)

            # Apply if-filter to restrict machines from callers
            if_block = _to_str(job_def.get("if", ""))
            if if_block and ("inputs.runner-label" in if_block or "inputs.runner" in if_block):
                # This job has a conditional on runner-label; rebuild machines per-caller
                filtered_machines: set[str] = set()
                for c in callers:
                    filtered_vals = apply_job_if_filter(job_def, c.get("runner_values", []))
                    for v in filtered_vals:
                        m = _label_to_machine(v)
                        if m:
                            filtered_machines.add(m)
                    # Also resolve with the caller's specific inputs
                    c_inputs: dict[str, list[str]] = {}
                    for n2, v2 in c.get("input_values", {}).items():
                        c_inputs[n2] = v2
                    if filtered_vals:
                        for v in filtered_vals:
                            m = _label_to_machine(v)
                            if m:
                                filtered_machines.add(m)
                            # Try substituting into labels
                            if isinstance(runs_on, list):
                                resolved_here: list[str] = []
                                for lbl in runs_on:
                                    lbl_s = str(lbl)
                                    if "inputs.runner-label" in lbl_s or "inputs.runner" in lbl_s:
                                        resolved_here.append(v)
                                    elif "${{" not in lbl_s:
                                        resolved_here.append(lbl_s)
                                    else:
                                        r = _resolve_label(lbl_s, _get_input_defaults(wf), c_inputs, {})
                                        resolved_here.extend(r)
                                filtered_machines.update(machines_from_labels(resolved_here))
                if filtered_machines:
                    machines = sorted(filtered_machines)

            # If external YAML has a sku, use that for machine mapping
            if ext_sku and not machines:
                m6 = SKU_TO_MACHINE.get(ext_sku)
                if m6:
                    machines = [m6]

            if not machines:
                continue

            # --- Tests ---
            tests = extract_tests_from_matrix(job_def)
            if not tests:
                tests = extract_tests_from_steps(job_def)
            # If matrix is dynamic (fromJSON), try extracting from source matrices
            if not tests:
                raw_mat = (job_def.get("strategy") or {}).get("matrix")
                if isinstance(raw_mat, dict):
                    for _mk, mv in raw_mat.items():
                        if isinstance(mv, str) and "fromjson" in mv.lower():
                            tests = extract_tests_from_workflow_matrices(wf)
                            break
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


def deduplicate_jobs(all_jobs: list[dict]) -> list[dict]:
    """Deduplicate by (job_name, workflow_file), merging machines + triggered_from."""
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
    return list(by_key.values())


# ---------------------------------------------------------------------------
# CSV matrix output
# ---------------------------------------------------------------------------

# Canonical order of machine columns in the spreadsheet
MACHINE_COLUMNS = [
    "N150",
    "N300",
    "P100",
    "P150",
    "P300",
    "T3K (Wormhole)",
    "Galaxy (Wormhole)",
    "Galaxy (Blackhole)",
    "Blackhole (generic)",
    "Wormhole (generic)",
]


def write_csv(jobs: list[dict], path: Path) -> None:
    """
    Write a CSV matrix: rows = tests (grouped by job), columns = machine types.

    Two sheets in one file:
      1. Test-level matrix (one row per test)
      2. Job-level summary matrix (one row per job)
      3. Coverage summary row at the bottom
    """
    import csv

    # Collect all machine types that actually appear
    all_machines: set[str] = set()
    for j in jobs:
        all_machines.update(j["machines"])
    # Use canonical order, then append any extras
    columns = [m for m in MACHINE_COLUMNS if m in all_machines]
    extras = sorted(all_machines - set(columns))
    columns.extend(extras)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # ── Section 1: Test-level matrix ──
        w.writerow(["TEST-LEVEL MATRIX"])
        w.writerow(["test", "job", "workflow"] + columns)

        test_counts = {m: 0 for m in columns}
        test_rows = 0

        for j in sorted(jobs, key=lambda x: (x["workflow_file"], x["job_name"])):
            machine_set = set(j["machines"])
            tests = j["tests"] or [f"(job: {j['job_name']})"]
            for test in tests:
                row = [test, j["job_name"], j["workflow_file"]]
                for m in columns:
                    if m in machine_set:
                        row.append("X")
                        test_counts[m] += 1
                    else:
                        row.append("")
                w.writerow(row)
                test_rows += 1

        # Summary row
        w.writerow([])
        w.writerow([f"TOTAL ({test_rows} tests)", "", ""] + [str(test_counts[m]) for m in columns])

        # ── Section 2: Job-level matrix ──
        w.writerow([])
        w.writerow([])
        w.writerow(["JOB-LEVEL MATRIX"])
        w.writerow(["job", "workflow", "num_tests"] + columns)

        job_counts = {m: 0 for m in columns}

        for j in sorted(jobs, key=lambda x: (x["workflow_file"], x["job_name"])):
            machine_set = set(j["machines"])
            row = [j["job_name"], j["workflow_file"], str(len(j["tests"]))]
            for m in columns:
                if m in machine_set:
                    row.append("X")
                    job_counts[m] += 1
                else:
                    row.append("")
            w.writerow(row)

        w.writerow([])
        w.writerow([f"TOTAL ({len(jobs)} jobs)", "", ""] + [str(job_counts[m]) for m in columns])


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="CI job → hardware + tests report")
    parser.add_argument("--output", "-o", type=Path, default=None, help="JSON output path")
    parser.add_argument("--csv", type=Path, default=None, help="CSV matrix output path")
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root or _find_repo_root(Path(__file__).resolve().parent)
    if not (repo_root / ".github" / "workflows").exists():
        print("::error::.github/workflows not found", file=sys.stderr)
        return 1

    all_jobs = process_impl_workflows(repo_root)
    jobs = deduplicate_jobs(all_jobs)

    # JSON output
    output = {
        "jobs": jobs,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    text = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(jobs)} jobs to {args.output}", file=sys.stderr)
    else:
        print(text)

    # CSV matrix output
    if args.csv:
        write_csv(jobs, args.csv)
        print(f"Wrote CSV matrix to {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
