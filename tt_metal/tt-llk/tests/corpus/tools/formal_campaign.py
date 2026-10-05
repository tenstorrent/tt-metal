#!/usr/bin/env python3
"""Run current-tuple SFPU translation validation.

The campaign input is the LLK corpus, optionally paired with a tuning search
JSON.  Each operation is compiled in three arms: selected flags on the
semantic source, frozen-baseline flags on that same source, and frozen-baseline
flags on the handwritten source.  The first pair is the exact compiler gate;
the second pair is the semantic-uplift gate.  Historical boards and recorded
overlays are deliberately outside this runner: every verdict in the output is
produced by this invocation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import csv
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent
DEFAULT_CASES = CORPUS / "sweep_2x2_ops.tsv"
DEFAULT_DOMAINS = HERE / "formal_domains.json"
FORMAL_ENGINE = HERE / "formal_equiv.py"
ELF_TEXT_SHA = HERE / "elf_text_sha.py"
TRACE_SCHEMA = "SFPUJO I"

RESULT_STATUS = {
    "PROVEN-EQUIV-ALL-INPUTS": "PROVEN_EQUIVALENT",
    "PROVEN-EQUIV-ON-DOCUMENTED-DOMAIN": "PROVEN_EQUIVALENT_ON_DOMAIN",
    "DIVERGENT": "DIVERGENT",
    "SEMANTICS-UNVALIDATED": "TRACE_VALIDATION_FAILED",
    "SCOPE-REFUSED": "UNSUPPORTED",
    "UNDECIDED": "TIMEOUT",
}
OPERATIONAL_FAILURES = {
    "BUILD_FAILED",
    "PROVER_FAILED",
    "TRACE_CAPTURE_FAILED",
    "TRACE_VALIDATION_FAILED",
}
COMPILER_ADMITTED = {
    "PROVEN_EQUIVALENT",
    "NOT_APPLICABLE_IDENTICAL_CONFIGURATION",
}
SEMANTIC_ADMITTED = {"PROVEN_EQUIVALENT", "PROVEN_EQUIVALENT_ON_DOMAIN"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict:
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"root": root, "head": head, "tracked_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"root": None, "head": None, "tracked_dirty": None}


def read_tsv(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    if not lines:
        raise ValueError(f"empty TSV: {path}")
    return list(csv.DictReader(lines, delimiter="\t"))


def load_cases(path: Path) -> dict[str, dict[str, str]]:
    cases = {}
    for row in read_tsv(path):
        op = row.get("op", "")
        if not op:
            raise ValueError(f"case without op in {path}")
        if op in cases:
            raise ValueError(f"duplicate case {op} in {path}")
        cases[op] = row
    return cases


def _selection_from_json(data: dict) -> dict[str, str]:
    operations = data.get("operations")
    if not isinstance(operations, dict):
        raise ValueError("selection JSON needs an operations object")
    selected = {}
    for op, record in operations.items():
        choice = record.get("selection") if isinstance(record, dict) else None
        if choice is None and isinstance(record, dict):
            proposal = record.get("proposal")
            choice = proposal.get("selection") if isinstance(proposal, dict) else None
        if not isinstance(choice, dict):
            continue
        flags = choice.get("flags")
        if flags is None:
            raise ValueError(f"selected operation {op} has no flags")
        selected[op] = str(flags)
    if not selected:
        raise ValueError("selection JSON contains no selected operations")
    return selected


def _profiles_from_json(
    data: dict, fallback_baseline: str | None = None
) -> dict[str, dict[str, str]]:
    """Return selected and frozen flags, rejecting a split baseline authority."""
    selected = _selection_from_json(data)
    settings = data.get("settings")
    global_baseline = settings.get("baseline_flags") if isinstance(settings, dict) else None
    profiles = {}
    for op, selected_flags in selected.items():
        record = data["operations"][op]
        proposal = record.get("proposal") if isinstance(record, dict) else None
        proposal_baseline = (
            proposal.get("frozen_baseline_flags") if isinstance(proposal, dict) else None
        )
        if global_baseline is not None and proposal_baseline is not None:
            if str(global_baseline) != str(proposal_baseline):
                raise ValueError(f"selected operation {op} disagrees with the frozen baseline")
        embedded_baseline = (
            proposal_baseline if proposal_baseline is not None else global_baseline
        )
        if fallback_baseline is not None and embedded_baseline is not None:
            if str(fallback_baseline) != str(embedded_baseline):
                raise ValueError(
                    f"selected operation {op} disagrees with --baseline-flags"
                )
        baseline = (
            proposal_baseline
            if proposal_baseline is not None
            else global_baseline
            if global_baseline is not None
            else fallback_baseline
        )
        if baseline is None:
            raise ValueError(f"selected operation {op} has no frozen baseline flags")
        profiles[op] = {
            "selected_flags": selected_flags,
            "baseline_flags": str(baseline),
        }
    return profiles


def load_selection(path: Path) -> dict[str, str]:
    if path.suffix == ".json":
        return _selection_from_json(json.loads(path.read_text()))
    rows = read_tsv(path)
    if not rows or "flags" not in rows[0]:
        raise ValueError("selection TSV needs op and flags columns; selected.tsv toggles alone are insufficient")
    selected = {row["op"]: row["flags"] for row in rows if row.get("op")}
    if not selected:
        raise ValueError("selection TSV contains no operations")
    return selected


def load_profiles(path: Path, fallback_baseline: str | None = None) -> dict[str, dict[str, str]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return _profiles_from_json(data, fallback_baseline)
    rows = read_tsv(path)
    if not rows or "flags" not in rows[0]:
        raise ValueError(
            "selection TSV needs op and flags columns; selected.tsv toggles alone are insufficient"
        )
    profiles = {}
    for row in rows:
        if not row.get("op"):
            continue
        baseline = row.get("baseline_flags") or fallback_baseline
        if baseline is None:
            raise ValueError(
                "selection TSV needs baseline_flags or --baseline-flags for the compiler gate"
            )
        profiles[row["op"]] = {
            "selected_flags": row["flags"],
            "baseline_flags": baseline,
        }
    if not profiles:
        raise ValueError("selection TSV contains no operations")
    return profiles


def load_domains(path: Path | None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("domain map must be a JSON object keyed by operation")
    for op, domain in data.items():
        if not isinstance(domain, list) or not all(isinstance(entry, dict) for entry in domain):
            raise ValueError(f"domain for {op} must be a list of objects")
    return data


def selected_ops(cases: dict, selection: dict | None, patterns: str | None) -> list[str]:
    ops = sorted(selection if selection is not None else cases)
    if patterns:
        globs = [item for item in patterns.split(",") if item]
        ops = [op for op in ops if any(fnmatch.fnmatch(op, pattern) for pattern in globs)]
    return ops


def operation_slug(op: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", op) is not None


def case_nodes(case: dict[str, str]) -> tuple[str, str]:
    return case.get("sem_corr", "").strip(), case.get("hand_corr", "").strip()


def clean_environment(overrides: dict[str, str]) -> dict[str, str]:
    keep = (
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USER",
        "VIRTUAL_ENV",
        "XDG_CACHE_HOME",
    )
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env.update(overrides)
    return env


def locate_math_elf(runtime: Path) -> Path:
    elfs = sorted(runtime.glob("tt-llk-build/sources/**/elf/math.elf"))
    if len(elfs) != 1:
        raise RuntimeError(f"expected one math.elf, found {len(elfs)}")
    return elfs[0]


def text_identity(python: Path, elf: Path) -> str:
    run = subprocess.run(
        [str(python), str(ELF_TEXT_SHA), str(elf)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    identity = run.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise RuntimeError("invalid ELF text identity")
    return identity


def run_leg(
    *,
    op: str,
    leg: str,
    node: str,
    flags: str,
    tests: Path,
    python: Path,
    sim: Path,
    out: Path,
    timeout: int,
) -> dict:
    runtime = Path(tempfile.mkdtemp(prefix=f"runtime-{leg}-", dir=out))
    trace = out / f"trace-{leg}.log"
    log = out / f"pytest-{leg}.log"
    env = clean_environment(
        {
            "CHIP_ARCH": "blackhole",
            "SHORT_ARCH": "bh",
            "LLK_HOME": str(tests.parent),
            "RUNNER_TEMP": str(runtime),
            "TT_LLK_EXTRA_COMPILER_OPTIONS": flags,
            "TT_METAL_SIMULATOR": str(sim),
            "TTSIM_TRACE_SFPU_FILE": str(trace),
            "TTSIM_TRACE_SFPU_STREAM": "1",
        }
    )
    with log.open("w") as stream:
        run = subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                "-s",
                "-o",
                "addopts=",
                "--run-simulator",
                f"python_tests/{node}",
            ],
            cwd=tests,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    if run.returncode:
        raise RuntimeError(f"{leg} pytest exited {run.returncode}; see {log.name}")
    if not trace.is_file() or TRACE_SCHEMA not in trace.read_text(errors="ignore"):
        raise RuntimeError(f"{leg} produced no SFPU trace")
    elf = locate_math_elf(runtime)
    result = {
        "node": node,
        "trace": trace.name,
        "trace_sha256": sha256(trace),
        "elf": str(elf.relative_to(runtime)),
        "elf_text_sha256": text_identity(python, elf),
    }
    shutil.rmtree(runtime)
    return result


def invoke_prover(
    *,
    op: str,
    trace_sem: Path,
    trace_hand: Path,
    domain: list[dict] | None,
    python: Path,
    out: Path,
    isa_json: Path,
    timeout: int,
) -> tuple[dict | None, subprocess.CompletedProcess]:
    command = [
        str(python),
        str(FORMAL_ENGINE),
        "--row",
        op,
        "--trace-sem",
        str(trace_sem),
        "--trace-hand",
        str(trace_hand),
        "--out",
        str(out),
        "--isa-json",
        str(isa_json),
        "--timeout",
        str(timeout),
    ]
    if domain:
        command.extend(("--domain-json", json.dumps(domain, sort_keys=True)))
    run = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 120)
    verdict_path = out / f"{op}-verdict.json"
    try:
        verdict = json.loads(verdict_path.read_text())
    except (OSError, json.JSONDecodeError):
        verdict = None
    return verdict, run


def verdict_status(verdict: dict) -> str:
    return RESULT_STATUS.get(verdict.get("verdict"), "PROVER_FAILED")


def admitted_status(all_inputs: str, domain: str | None) -> str:
    if all_inputs == "PROVEN_EQUIVALENT":
        return all_inputs
    if all_inputs == "DIVERGENT" and domain == "PROVEN_EQUIVALENT_ON_DOMAIN":
        return domain
    return all_inputs


def prove_pair(
    *,
    row: str,
    trace_sem: Path,
    trace_hand: Path,
    domain: list[dict] | None,
    python: Path,
    sim: Path,
    out: Path,
    timeout: int,
) -> dict:
    """Prove one ordered trace pair; domain fallback is explicitly opt-in."""
    try:
        verdict, run = invoke_prover(
            op=row,
            trace_sem=trace_sem,
            trace_hand=trace_hand,
            domain=None,
            python=python,
            out=out,
            isa_json=sim.parent / "tensix_isa.json",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "detail": f"prover exceeded {timeout + 120}s"}
    if not isinstance(verdict, dict) or not verdict.get("verdict"):
        detail = (run.stderr or run.stdout or "prover wrote no verdict")[-500:]
        return {"status": "PROVER_FAILED", "detail": detail}
    all_input_status = verdict_status(verdict)
    result = {
        "status": all_input_status,
        "formal_verdict": verdict["verdict"],
        "validation": verdict.get("validation"),
        "details": verdict.get("details"),
        "witness": (verdict.get("details") or {}).get("witness"),
        "verdict_file": f"{out.name}/{row}-verdict.json",
    }
    if all_input_status != "DIVERGENT" or not domain:
        return result
    try:
        domain_verdict, domain_run = invoke_prover(
            op=f"{row}-domain",
            trace_sem=trace_sem,
            trace_hand=trace_hand,
            domain=domain,
            python=python,
            out=out,
            isa_json=sim.parent / "tensix_isa.json",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result.update(status="TIMEOUT", domain_status="TIMEOUT")
        return result
    if not isinstance(domain_verdict, dict) or not domain_verdict.get("verdict"):
        detail = (
            domain_run.stderr or domain_run.stdout or "domain prover wrote no verdict"
        )[-500:]
        result.update(
            status="PROVER_FAILED", domain_status="PROVER_FAILED", detail=detail
        )
        return result
    domain_status = verdict_status(domain_verdict)
    result.update(
        status=admitted_status(all_input_status, domain_status),
        domain_status=domain_status,
        domain_formal_verdict=domain_verdict["verdict"],
        domain_validation=domain_verdict.get("validation"),
        domain_details=domain_verdict.get("details"),
        domain_verdict_file=f"{out.name}/{row}-domain-verdict.json",
    )
    return result


def run_case(
    *,
    op: str,
    case: dict[str, str] | None,
    selected_flags: str,
    baseline_flags: str,
    domain: list[dict] | None,
    tests: Path,
    python: Path,
    sim: Path,
    root: Path,
    timeout: int,
) -> dict:
    started = time.monotonic()
    result = {
        "op": op,
        # `flags` and the legacy semantic fields below remain for readers of
        # schema v1. New readers must use the independently named gates.
        "flags": selected_flags,
        "selected_flags": selected_flags,
        "baseline_flags": baseline_flags,
        "domain": domain,
        "status": None,
    }
    if case is None:
        result.update(
            status="NO_CASE",
            compiler_status="NO_CASE",
            semantic_status="NO_CASE",
            detail="operation is absent from the corpus",
        )
        return result
    sem_node, hand_node = case_nodes(case)
    result.update(kind=case.get("kind"), semantic_node=sem_node, reference_node=hand_node)
    if not sem_node:
        result.update(
            status="NO_CASE",
            compiler_status="NO_CASE",
            semantic_status="NO_CASE",
            detail="corpus row has no semantic correctness node",
        )
        return result

    out = root / op
    out.mkdir(parents=True)
    try:
        selected_sem = run_leg(
            op=op,
            leg="selected-sem",
            node=sem_node,
            flags=selected_flags,
            tests=tests,
            python=python,
            sim=sim,
            out=out,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result.update(
            status="BUILD_FAILED",
            compiler_status="BUILD_FAILED",
            semantic_status="NOT_RUN",
            detail=f"selected semantic pytest exceeded {timeout}s",
        )
        return result
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        result.update(
            status="BUILD_FAILED",
            compiler_status="BUILD_FAILED",
            semantic_status="NOT_RUN",
            detail=str(error),
        )
        return result

    if selected_flags == baseline_flags:
        baseline_sem = selected_sem
        baseline_sem_trace = out / "trace-selected-sem.log"
        compiler = {
            "status": "NOT_APPLICABLE_IDENTICAL_CONFIGURATION",
            "detail": "selected and frozen compiler flags are identical",
        }
    else:
        try:
            baseline_sem = run_leg(
                op=op,
                leg="baseline-sem",
                node=sem_node,
                flags=baseline_flags,
                tests=tests,
                python=python,
                sim=sim,
                out=out,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            result.update(
                status="BUILD_FAILED",
                compiler_status="BUILD_FAILED",
                semantic_status="NOT_RUN",
                detail=f"baseline semantic pytest exceeded {timeout}s",
            )
            return result
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            result.update(
                status="BUILD_FAILED",
                compiler_status="BUILD_FAILED",
                semantic_status="NOT_RUN",
                detail=str(error),
            )
            return result
        baseline_sem_trace = out / "trace-baseline-sem.log"
        compiler = prove_pair(
            row=f"{op}-compiler",
            trace_sem=out / "trace-selected-sem.log",
            trace_hand=out / "trace-baseline-sem.log",
            domain=None,
            python=python,
            sim=sim,
            out=out,
            timeout=timeout,
        )
    result["compiler"] = compiler
    result["compiler_status"] = compiler["status"]
    result["legs"] = {
        "selected_semantic": selected_sem,
        "baseline_semantic": baseline_sem,
    }

    if not hand_node or sem_node == hand_node:
        semantic = {
            "status": "NO_REFERENCE",
            "detail": (
                "corpus row has no handwritten correctness node"
                if not hand_node
                else "semantic and reference nodes are identical"
            ),
        }
    else:
        try:
            baseline_hand = run_leg(
                op=op,
                leg="baseline-hand",
                node=hand_node,
                flags=baseline_flags,
                tests=tests,
                python=python,
                sim=sim,
                out=out,
                timeout=timeout,
            )
            result["legs"]["baseline_handwritten"] = baseline_hand
        except subprocess.TimeoutExpired:
            semantic = {
                "status": "BUILD_FAILED",
                "detail": f"baseline handwritten pytest exceeded {timeout}s",
            }
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            semantic = {"status": "BUILD_FAILED", "detail": str(error)}
        else:
            semantic = prove_pair(
                row=f"{op}-semantic",
                trace_sem=baseline_sem_trace,
                trace_hand=out / "trace-baseline-hand.log",
                domain=domain,
                python=python,
                sim=sim,
                out=out,
                timeout=timeout,
            )
    result["semantic"] = semantic
    result["semantic_status"] = semantic["status"]
    result["compiler_gate"] = (
        "PASS" if compiler["status"] in COMPILER_ADMITTED else "FOLLOWUP_REQUIRED"
    )
    result["semantic_gate"] = (
        "PASS" if semantic["status"] in SEMANTIC_ADMITTED else "FOLLOWUP_REQUIRED"
    )
    result["deployment_gate"] = (
        "PASS"
        if result["compiler_gate"] == result["semantic_gate"] == "PASS"
        else "FOLLOWUP_REQUIRED"
    )
    # Schema-v1 compatibility: top-level proof fields describe the semantic
    # comparison, exactly as they did before the compiler gate was separated.
    result["status"] = semantic["status"]
    for key in (
        "formal_verdict",
        "validation",
        "details",
        "witness",
        "verdict_file",
        "domain_status",
        "domain_formal_verdict",
        "domain_validation",
        "domain_details",
        "domain_verdict_file",
        "detail",
    ):
        if key in semantic:
            result[key] = semantic[key]
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    return result


def preflight(tests: Path, python: Path, sim: Path) -> dict:
    if not tests.is_dir():
        raise ValueError(f"tests root does not exist: {tests}")
    if not python.is_file():
        raise ValueError(f"harness Python does not exist: {python}")
    if not sim.is_file():
        raise ValueError(f"simulator does not exist: {sim}")
    descriptor = sim.parent / "soc_descriptor.yaml"
    isa_json = sim.parent / "tensix_isa.json"
    if not descriptor.is_file():
        raise ValueError(f"soc_descriptor.yaml is missing beside simulator: {descriptor}")
    if not isa_json.is_file():
        raise ValueError(f"tensix_isa.json is missing beside simulator: {isa_json}")
    strings = subprocess.run(
        ["strings", "-a", str(sim)], check=True, capture_output=True, text=True
    ).stdout
    if TRACE_SCHEMA not in strings:
        raise ValueError("simulator does not contain the SFPUJO trace instrument")
    dependency = subprocess.run(
        [str(python), "-c", "import elftools, z3"], capture_output=True, text=True
    )
    if dependency.returncode:
        raise ValueError("harness Python needs pyelftools and z3")
    cc1plus = sorted(
        path
        for path in (tests / "sfpi/compiler/libexec/gcc/riscv-tt-elf").glob("*/cc1plus")
        if ".pin-backup" not in str(path)
    )
    build_manifest_path = sim.parent / "formal-instrument.json"
    build_manifest = None
    if build_manifest_path.is_file():
        try:
            recorded = json.loads(build_manifest_path.read_text())
            expected = recorded.get("artifacts", {}).get("libttsim.so", {}).get("sha256")
            build_manifest = {
                "path": str(build_manifest_path),
                "sha256": sha256(build_manifest_path),
                "source": recorded.get("source"),
                "build_command": recorded.get("build_command"),
                "artifact_matches_manifest": expected == sha256(sim),
            }
        except (OSError, json.JSONDecodeError, AttributeError):
            build_manifest = {"path": str(build_manifest_path), "readable": False}
    return {
        "simulator": {"path": str(sim), "sha256": sha256(sim)},
        "descriptor": {"path": str(descriptor), "sha256": sha256(descriptor)},
        "isa": {"path": str(isa_json), "sha256": sha256(isa_json)},
        "compiler": (
            {"path": str(cc1plus[0]), "sha256": sha256(cc1plus[0])}
            if len(cc1plus) == 1
            else {"path": None, "sha256": None, "candidates": len(cc1plus)}
        ),
        "tt_metal": git_state(tests),
        "formal_engine_sha256": sha256(FORMAL_ENGINE),
        "trace_schema": "SFPUJO-v1",
        "build_manifest": build_manifest,
    }


def atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete text file without exposing a truncated checkpoint."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
        os.chmod(temporary, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_results(
    root: Path,
    records: list[dict],
    metadata: dict,
    started: float,
    expected_total: int,
) -> None:
    completed = len(records)
    if completed > expected_total:
        raise ValueError("completed records exceed expected total")
    counts = Counter(record["status"] for record in records)
    compiler_counts = Counter(
        record.get("compiler_status", record["status"]) for record in records
    )
    semantic_counts = Counter(
        record.get("semantic_status", record["status"]) for record in records
    )
    operational_failures = sum(
        1
        for record in records
        if record.get("compiler_status", record["status"]) in OPERATIONAL_FAILURES
        or record.get("semantic_status", record["status"]) in OPERATIONAL_FAILURES
    )
    compiler_admitted = sum(compiler_counts[status] for status in COMPILER_ADMITTED)
    semantic_admitted = sum(semantic_counts[status] for status in SEMANTIC_ADMITTED)
    deployed = sum(
        (
            record.get("deployment_gate") == "PASS"
            if "deployment_gate" in record
            else record["status"] in SEMANTIC_ADMITTED
        )
        for record in records
    )
    if completed < expected_total:
        campaign_status = "RUNNING"
    else:
        campaign_status = "INCOMPLETE" if operational_failures else "COMPLETE"
    summary = {
        "schema_version": 2,
        "status": campaign_status,
        "selected": completed,
        "expected_total": expected_total,
        "counts": dict(sorted(counts.items())),
        "compiler_counts": dict(sorted(compiler_counts.items())),
        "semantic_counts": dict(sorted(semantic_counts.items())),
        "operational_failures": operational_failures,
        "compiler_admission": (
            "ALL_PROVEN"
            if completed == expected_total and compiler_admitted == expected_total
            else "FOLLOWUP_REQUIRED"
        ),
        "compiler_admitted": compiler_admitted,
        "semantic_admission": (
            "ALL_PROVEN"
            if completed == expected_total and semantic_admitted == expected_total
            else "FOLLOWUP_REQUIRED"
        ),
        "semantic_admitted": semantic_admitted,
        "deployment_admission": (
            "ALL_PROVEN"
            if completed == expected_total and deployed == expected_total
            else "FOLLOWUP_REQUIRED"
        ),
        "deployment_admitted": deployed,
        "deployment_followup_required": expected_total - deployed,
        # Schema-v1 compatibility: formal admission is the semantic gate.
        "formal_admission": (
            "ALL_PROVEN"
            if completed == expected_total and semantic_admitted == expected_total
            else "FOLLOWUP_REQUIRED"
        ),
        "formally_admitted": semantic_admitted,
        "followup_required": expected_total - semantic_admitted,
        "wall_seconds": round(time.monotonic() - started, 3),
        "metadata": metadata,
        "results": records,
    }
    json_content = json.dumps(summary, indent=2) + "\n"
    columns = (
        "op",
        "status",
        "compiler_status",
        "semantic_status",
        "compiler_gate",
        "semantic_gate",
        "deployment_gate",
        "formal_verdict",
        "wall_seconds",
        "flags",
        "baseline_flags",
        "detail",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    for record in records:
        writer.writerow({name: record.get(name, "") for name in columns})
    # Publish the detail first and the JSON summary last so consumers that use
    # the summary as a checkpoint marker cannot observe a newer summary paired
    # with an older TSV.
    atomic_write_text(root / "formal-results.tsv", stream.getvalue())
    atomic_write_text(root / "formal-results.json", json_content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-root", type=Path, default=CORPUS.parent)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--flags", default="")
    parser.add_argument(
        "--baseline-flags",
        help="explicit frozen compiler profile; required for a selection file that "
        "does not embed frozen_baseline_flags (uniform --flags defaults to itself)",
    )
    parser.add_argument("--domains", type=Path, default=DEFAULT_DOMAINS)
    parser.add_argument("--ops", help="comma-separated operation globs")
    parser.add_argument("--sem-node", help="semantic pytest node for a single --ops row")
    parser.add_argument("--hand-node", help="reference pytest node for a single --ops row")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.jobs <= 0:
        parser.error("jobs must be positive")
    try:
        cases = load_cases(args.cases.resolve())
        profiles = (
            load_profiles(args.selection.resolve(), args.baseline_flags)
            if args.selection
            else None
        )
        domains = load_domains(args.domains.resolve() if args.domains else None)
        ops = selected_ops(cases, profiles, args.ops)
        if not ops:
            raise ValueError("no operations selected")
        invalid = [op for op in ops if not operation_slug(op)]
        if invalid:
            raise ValueError(f"operations are not safe artifact names: {invalid}")
        if args.sem_node or args.hand_node:
            if len(ops) != 1 or not args.sem_node or not args.hand_node:
                raise ValueError("--sem-node and --hand-node require one exact --ops row")
            cases[ops[0]] = dict(cases.get(ops[0], {}))
            cases[ops[0]].update(sem_corr=args.sem_node, hand_corr=args.hand_node)
        tests = args.tests_root.resolve()
        python = tests / ".venv/bin/python"
        sim = args.sim.resolve()
        metadata = preflight(tests, python, sim)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    started = time.monotonic()
    args.out.mkdir(parents=True)
    metadata.update(
        cases={"path": str(args.cases.resolve()), "sha256": sha256(args.cases.resolve())},
        selection=(
            {"path": str(args.selection.resolve()), "sha256": sha256(args.selection.resolve())}
            if args.selection
            else None
        ),
        domains=(
            {"path": str(args.domains.resolve()), "sha256": sha256(args.domains.resolve())}
            if args.domains
            else None
        ),
    )
    def run_one(index: int, op: str) -> tuple[int, dict]:
        if profiles is not None:
            selected_flags = profiles[op]["selected_flags"]
            baseline_flags = profiles[op]["baseline_flags"]
        else:
            selected_flags = args.flags
            baseline_flags = (
                args.baseline_flags if args.baseline_flags is not None else args.flags
            )
        record = run_case(
            op=op,
            case=cases.get(op),
            selected_flags=selected_flags,
            baseline_flags=baseline_flags,
            domain=domains.get(op),
            tests=tests,
            python=python,
            sim=sim,
            root=args.out,
            timeout=args.timeout,
        )
        return index, record

    records_by_index = {}
    if args.jobs == 1:
        for index, op in enumerate(ops):
            print(f"[{index + 1}/{len(ops)}] {op}", flush=True)
            _, record = run_one(index, op)
            records_by_index[index] = record
            print(f"  {record['status']}", flush=True)
            write_results(
                args.out,
                [records_by_index[i] for i in sorted(records_by_index)],
                metadata,
                started,
                len(ops),
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_one, index, op): (index, op)
                for index, op in enumerate(ops)
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), 1
            ):
                index, op = futures[future]
                _, record = future.result()
                records_by_index[index] = record
                print(
                    f"[{completed}/{len(ops)}] {op}: {record['status']}",
                    flush=True,
                )
                write_results(
                    args.out,
                    [records_by_index[i] for i in sorted(records_by_index)],
                    metadata,
                    started,
                    len(ops),
                )
    records = [records_by_index[i] for i in range(len(ops))]
    return 2 if any(
        record.get("compiler_status", record["status"]) in OPERATIONAL_FAILURES
        or record.get("semantic_status", record["status"]) in OPERATIONAL_FAILURES
        for record in records
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
