#!/usr/bin/env python3
"""Run current-tuple SFPU translation validation.

The campaign input is the LLK corpus, optionally paired with a tuning search
JSON.  Each operation is compiled with its own selected flags.  Historical
boards and recorded overlays are deliberately outside this runner: every
verdict in the output is produced by this invocation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import fnmatch
import hashlib
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
        if not isinstance(choice, dict):
            continue
        flags = choice.get("flags")
        if flags is None:
            raise ValueError(f"selected operation {op} has no flags")
        selected[op] = str(flags)
    if not selected:
        raise ValueError("selection JSON contains no selected operations")
    return selected


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
    domain: list[dict] | None,
    python: Path,
    out: Path,
    timeout: int,
) -> tuple[dict | None, subprocess.CompletedProcess]:
    command = [
        str(python),
        str(FORMAL_ENGINE),
        "--row",
        op,
        "--trace-sem",
        str(out / "trace-sem.log"),
        "--trace-hand",
        str(out / "trace-hand.log"),
        "--out",
        str(out),
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


def run_case(
    *,
    op: str,
    case: dict[str, str] | None,
    flags: str,
    domain: list[dict] | None,
    tests: Path,
    python: Path,
    sim: Path,
    root: Path,
    timeout: int,
) -> dict:
    started = time.monotonic()
    result = {"op": op, "flags": flags, "domain": domain, "status": None}
    if case is None:
        result.update(status="NO_CASE", detail="operation is absent from the corpus")
        return result
    sem_node, hand_node = case_nodes(case)
    result.update(kind=case.get("kind"), semantic_node=sem_node, reference_node=hand_node)
    if not sem_node:
        result.update(status="NO_CASE", detail="corpus row has no semantic correctness node")
        return result
    if not hand_node:
        result.update(status="NO_REFERENCE", detail="corpus row has no handwritten correctness node")
        return result
    if sem_node == hand_node:
        result.update(status="NO_REFERENCE", detail="semantic and reference nodes are identical")
        return result

    out = root / op
    out.mkdir(parents=True)
    try:
        sem = run_leg(
            op=op,
            leg="sem",
            node=sem_node,
            flags=flags,
            tests=tests,
            python=python,
            sim=sim,
            out=out,
            timeout=timeout,
        )
        hand = run_leg(
            op=op,
            leg="hand",
            node=hand_node,
            flags=flags,
            tests=tests,
            python=python,
            sim=sim,
            out=out,
            timeout=timeout,
        )
        result["legs"] = {"semantic": sem, "reference": hand}
    except subprocess.TimeoutExpired:
        result.update(status="BUILD_FAILED", detail=f"pytest exceeded {timeout}s")
        return result
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        result.update(status="BUILD_FAILED", detail=str(error))
        return result

    try:
        verdict, run = invoke_prover(
            op=op,
            domain=domain,
            python=python,
            out=out,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result.update(status="TIMEOUT", detail=f"prover exceeded {timeout + 120}s")
        return result
    if not isinstance(verdict, dict) or not verdict.get("verdict"):
        detail = (run.stderr or run.stdout or "prover wrote no verdict")[-500:]
        result.update(status="PROVER_FAILED", detail=detail)
        return result
    result.update(
        status=RESULT_STATUS.get(verdict["verdict"], "PROVER_FAILED"),
        formal_verdict=verdict["verdict"],
        validation=verdict.get("validation"),
        details=verdict.get("details"),
        witness=(verdict.get("details") or {}).get("witness"),
        verdict_file=f"{op}/{op}-verdict.json",
        wall_seconds=round(time.monotonic() - started, 3),
    )
    return result


def preflight(tests: Path, python: Path, sim: Path) -> dict:
    if not tests.is_dir():
        raise ValueError(f"tests root does not exist: {tests}")
    if not python.is_file():
        raise ValueError(f"harness Python does not exist: {python}")
    if not sim.is_file():
        raise ValueError(f"simulator does not exist: {sim}")
    descriptor = sim.parent / "soc_descriptor.yaml"
    if not descriptor.is_file():
        raise ValueError(f"soc_descriptor.yaml is missing beside simulator: {descriptor}")
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


def write_results(root: Path, records: list[dict], metadata: dict, started: float) -> None:
    counts = Counter(record["status"] for record in records)
    operational_failures = sum(counts[status] for status in OPERATIONAL_FAILURES)
    summary = {
        "schema_version": 1,
        "status": "INCOMPLETE" if operational_failures else "COMPLETE",
        "selected": len(records),
        "counts": dict(sorted(counts.items())),
        "operational_failures": operational_failures,
        "wall_seconds": round(time.monotonic() - started, 3),
        "metadata": metadata,
        "results": records,
    }
    (root / "formal-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    columns = ("op", "status", "formal_verdict", "wall_seconds", "flags", "detail")
    with (root / "formal-results.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in columns})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-root", type=Path, default=CORPUS.parent)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--flags", default="")
    parser.add_argument("--domains", type=Path)
    parser.add_argument("--ops", help="comma-separated operation globs")
    parser.add_argument("--sem-node", help="semantic pytest node for a single --ops row")
    parser.add_argument("--hand-node", help="reference pytest node for a single --ops row")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    try:
        cases = load_cases(args.cases.resolve())
        selection = load_selection(args.selection.resolve()) if args.selection else None
        domains = load_domains(args.domains.resolve() if args.domains else None)
        ops = selected_ops(cases, selection, args.ops)
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
    records = []
    for index, op in enumerate(ops, 1):
        print(f"[{index}/{len(ops)}] {op}", flush=True)
        flags = selection[op] if selection is not None else args.flags
        record = run_case(
            op=op,
            case=cases.get(op),
            flags=flags,
            domain=domains.get(op),
            tests=tests,
            python=python,
            sim=sim,
            root=args.out,
            timeout=args.timeout,
        )
        records.append(record)
        print(f"  {record['status']}", flush=True)
        write_results(args.out, records, metadata, started)
    return 2 if any(record["status"] in OPERATIONAL_FAILURES for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
