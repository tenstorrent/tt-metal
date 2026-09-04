#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# prove_all.py — the reusable, re-runnable "prove-every-op" driver (laneMH).
#
# Runs BOTH proof engines across ALL 134 kernel-decided board ops at the
# current pin and emits the master coverage ledger, in one command.
#
#   * formal_equiv.py  (laneJO)  — z3 QF_BV translation validation on the
#     FINAL emitted SFPU stream (sem vs hand, all inputs, Dst-state), on the
#     PINNED instrumented craq-sim (value-observation trace patch).
#   * bitexact_sweep.py (laneJN) — exhaustive 2^16 single-input sweep on the
#     PINNED craq-sim (sim-only probe/sweep/verdict; the device-anchor
#     validation gate is a separate stage this driver does not run).
#
# The two engines are REUSED as libraries/subprocesses — this driver never
# re-implements any proof logic.  Two provenance-pinned overlays that are NOT
# re-run here (they need a device campaign / are recorded z3 proofs on the
# identical instrument) are joined in with strict precedence:
#   * silicon overlay  — laneKC device-exhaustive 2^16 bit-exact sweeps.
#   * domain overlay   — laneJO documented-deliverable-domain z3 re-proofs
#                        (upgrade clamp-fresh / mulint32-fresh base divergence).
#
# Routing is DATA-DRIVEN by the checked-in prove_all_manifest.tsv (one row per
# op: arity/space + chosen engine + reason).  Every routing decision and every
# verdict lands in the ledger so a human can audit why each op went where.
#
# Honesty is absolute: an EQUAL class is emitted only for a verdict the engine
# actually returned (z3 UNSAT / exhaustive-16 0-diff, VALIDATED by the engine's
# own gate) or the silicon overlay.  UNDECIDED / timeout / infeasible /
# scope-refused stay labeled with reasons.  Never fake or overclaim a proof.

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent  # .../corpus/tools
CORPUS = HERE.parent  # .../corpus
TESTS = CORPUS.parent  # .../tests
LLK = TESTS.parent  # .../tt-llk
HOME = Path(os.path.expanduser("~/sfpi-uplift"))

# ---- pinned inputs (the provenance gate verifies these) --------------------
PIN = "pin-59"
EXPECT_CC1PLUS_PREFIX = "b013967fffaa"  # pin-59 cc1plus
JO_INSTRUMENT = HOME / "laneKS-evidence-20260901/simstage-trace-old/libttsim.so"
EXPECT_JO_SHA = "ba23c3f169126425998b53b0202a10a81e35fba0692ed4eca5964f073ec31113"
BITEXACT_SIM = HOME / "laneJN-simstage/libttsim.so"  # plain pinned craq-sim 1c47e9cd
EXPECT_BITEXACT_SHA_PREFIX = "1d162f0adf67"
SFPI = TESTS / "sfpi"  # symlink -> pinned toolchain

BOARD = HOME / "laneFM-evidence-20260822/FINAL-BOARD.tsv"
MANIFEST = HERE / "prove_all_manifest.tsv"
SILICON_OVERLAY = HERE / "prove_all_silicon_overlay.tsv"
DOMAIN_OVERLAY = HERE / "prove_all_domain_overlay.tsv"
FAST_OPS = HERE / "prove_all_fast_ops.tsv"
FORMAL_ENGINE = HERE / "formal_equiv.py"
BITEXACT_ENGINE = HERE / "bitexact_sweep.py"
OPS_TSV = CORPUS / "sweep_2x2_ops.tsv"
VENV_PY = TESTS / ".venv/bin/python"

# ---- provability precedence (identical to laneMD master-coverage join) -----
CLASS_ORDER = [
    "SILICON-EXHAUSTIVE",  # device 2^16 bit-exact          (CERTIFIED EQUAL)
    "SMT-PROVEN-ALL-INPUTS",  # z3 UNSAT over all inputs        (CERTIFIED EQUAL)
    "SMT-PROVEN-DOMAIN",  # z3 UNSAT on documented domain
    "DIVERGENCE-CERTIFIED",  # proven/witnessed UNEQUAL (licensed/accuracy-gated)
    "SIM-BIT-EXACT-16",  # sim 2^16 exhaustive EQUAL (no silicon)
    "UNDECIDED-Z3-TIMEOUT",  # z3 resource cap
    "INFEASIBLE-2^32",  # single-input fp32/int32, not brute-forced
    "NOT-EXHAUSTIBLE",  # multi-operand / cross-lane
    "SCOPE-REFUSED",  # cross-lane / semantics-unvalidated
    "UNSWEPT",
]
RANK = {c: i for i, c in enumerate(CLASS_ORDER)}
MACHINE_CERTIFIED = {"SILICON-EXHAUSTIVE", "SMT-PROVEN-ALL-INPUTS"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tsv(path):
    rows, hdr = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split("\t")
        if hdr is None:
            hdr = p
            continue
        rows.append(dict(zip(hdr, p)))
    return rows


# ---------------------------------------------------------------------------
# ON flag set (pin-59 ON-39) — imported from the canonical sweep_2x2 module.
# ---------------------------------------------------------------------------
def on_flags():
    sys.path.insert(0, str(CORPUS))
    import sweep_2x2  # noqa: E402

    return sweep_2x2.ON_FLAGS


# ---------------------------------------------------------------------------
# Board op set + manifest.  Assert manifest op set == board 1:1.
# ---------------------------------------------------------------------------
def load_board():
    board = {}
    for r in read_tsv(BOARD):
        if r.get("class") in ("WIN", "PARITY", "LOSS"):
            board[r["op"]] = r["class"]
    if len(board) != 134:
        sys.exit(f"FATAL: board op count {len(board)} != 134")
    return board


def load_manifest(board):
    man = OrderedDict()
    for r in read_tsv(MANIFEST):
        man[r["op"]] = r
    mset, bset = set(man), set(board)
    if mset != bset:
        sys.exit(
            "FATAL: manifest op set != board op set 1:1\n"
            f"  manifest-only: {sorted(mset - bset)}\n"
            f"  board-only:    {sorted(bset - mset)}"
        )
    return man


# ---------------------------------------------------------------------------
# Provenance gate — verify every required instrument by sha; fail LOUDLY.
# ---------------------------------------------------------------------------
def active_cc1plus():
    cands = [
        p
        for p in SFPI.glob("compiler/libexec/gcc/riscv-tt-elf/*/cc1plus")
        if ".pin-backup" not in str(p)
    ]
    return cands[0] if cands else None


def provenance_gate(strict=True):
    prov = {"pin": PIN, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "shas": {}}
    problems = []

    cc = active_cc1plus()
    if cc is None:
        problems.append("active cc1plus not found under tests/sfpi")
    else:
        s = sha256(cc)
        prov["shas"]["cc1plus"] = s
        prov["cc1plus_path"] = str(cc)
        if not s.startswith(EXPECT_CC1PLUS_PREFIX):
            problems.append(
                f"cc1plus sha {s[:12]} != expected {PIN} {EXPECT_CC1PLUS_PREFIX}"
            )

    if not JO_INSTRUMENT.exists():
        problems.append(f"JO instrumented sim missing: {JO_INSTRUMENT}")
    else:
        s = sha256(JO_INSTRUMENT)
        prov["shas"]["jo_instrument_sim"] = s
        prov["jo_instrument_path"] = str(JO_INSTRUMENT)
        if s != EXPECT_JO_SHA:
            problems.append(
                f"JO instrument sha {s[:12]} != expected {EXPECT_JO_SHA[:12]}"
            )
        if not (JO_INSTRUMENT.parent / "soc_descriptor.yaml").exists():
            problems.append("soc_descriptor.yaml missing next to JO instrument")

    if not BITEXACT_SIM.exists():
        problems.append(f"bitexact pinned sim missing: {BITEXACT_SIM}")
    else:
        s = sha256(BITEXACT_SIM)
        prov["shas"]["bitexact_sim"] = s
        prov["bitexact_sim_path"] = str(BITEXACT_SIM)
        if not s.startswith(EXPECT_BITEXACT_SHA_PREFIX):
            problems.append(
                f"bitexact sim sha {s[:12]} != expected {EXPECT_BITEXACT_SHA_PREFIX}"
            )

    for req in (
        FORMAL_ENGINE,
        BITEXACT_ENGINE,
        BOARD,
        MANIFEST,
        SILICON_OVERLAY,
        DOMAIN_OVERLAY,
        FAST_OPS,
        OPS_TSV,
    ):
        if not Path(req).exists():
            problems.append(f"required file missing: {req}")
    if not VENV_PY.exists():
        problems.append(f"harness venv python missing: {VENV_PY}")

    # provenance of the overlays (recorded, not re-run)
    prov["shas"]["silicon_overlay"] = (
        sha256(SILICON_OVERLAY) if SILICON_OVERLAY.exists() else "-"
    )
    prov["shas"]["domain_overlay"] = (
        sha256(DOMAIN_OVERLAY) if DOMAIN_OVERLAY.exists() else "-"
    )
    prov["shas"]["manifest"] = sha256(MANIFEST) if MANIFEST.exists() else "-"
    prov["shas"]["formal_equiv.py"] = (
        sha256(FORMAL_ENGINE) if FORMAL_ENGINE.exists() else "-"
    )
    prov["shas"]["bitexact_sweep.py"] = (
        sha256(BITEXACT_ENGINE) if BITEXACT_ENGINE.exists() else "-"
    )

    prov["problems"] = problems
    if problems and strict:
        sys.stderr.write("PROVENANCE GATE FAILED:\n")
        for p in problems:
            sys.stderr.write(f"  - {p}\n")
        sys.exit(3)
    return prov


# ---------------------------------------------------------------------------
# Engine: formal_equiv  (run both legs on the JO instrument, then z3).
# ---------------------------------------------------------------------------
def _run_leg(op, leg, node, out_dir, flags, timeout):
    """Run one pytest sim leg on the JO instrument, emit a trace file."""
    rt = out_dir / f"rt-{op}-{leg}"
    if rt.exists():
        subprocess.run(["rm", "-rf", str(rt)], check=False)
    rt.mkdir(parents=True, exist_ok=True)
    trace = out_dir / f"trace-{op}-{leg}.log"
    trace.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(
        RUNNER_TEMP=str(rt),
        CHIP_ARCH="blackhole",
        SHORT_ARCH="bh",
        TT_METAL_SIMULATOR=str(JO_INSTRUMENT),
        TT_LLK_EXTRA_COMPILER_OPTIONS=flags,
        TTSIM_TRACE_SFPU_STREAM="1",
        TTSIM_TRACE_SFPU_FILE=str(trace),
        LLK_HOME=str(LLK),
    )
    log = out_dir / f"pytest-{op}-{leg}.log"
    with log.open("w") as fh:
        r = subprocess.run(
            [
                str(VENV_PY),
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-q",
                "-s",
                "--run-simulator",
                f"python_tests/{node}",
            ],
            cwd=str(TESTS),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    if r.returncode != 0:
        return None, f"{leg}-leg pytest rc={r.returncode}"
    if "SFPUJO I" not in trace.read_text(errors="ignore"):
        return None, f"{leg}-leg produced no SFPU stream"
    return trace, None


def run_formal(op, man_row, out_dir, flags, timeout):
    sem, hand = man_row["sem_node"], man_row["hand_node"]
    if sem in ("-", "") or hand in ("-", ""):
        return {
            "op": op,
            "engine": "formal_equiv",
            "class": "UNSWEPT",
            "verdict": "NO-NODE-IDS",
            "reason": "no sem/hand node in manifest",
        }
    try:
        t0 = time.time()
        tsem, e1 = _run_leg(op, "sem", sem, out_dir, flags, timeout)
        if e1:
            return {
                "op": op,
                "engine": "formal_equiv",
                "class": "UNSWEPT",
                "verdict": "LEG-FAILED",
                "reason": e1,
                "wall_s": round(time.time() - t0, 1),
            }
        thand, e2 = _run_leg(op, "hand", hand, out_dir, flags, timeout)
        if e2:
            return {
                "op": op,
                "engine": "formal_equiv",
                "class": "UNSWEPT",
                "verdict": "LEG-FAILED",
                "reason": e2,
                "wall_s": round(time.time() - t0, 1),
            }
        r = subprocess.run(
            [
                "python3",
                str(FORMAL_ENGINE),
                "--row",
                op,
                "--trace-sem",
                str(tsem),
                "--trace-hand",
                str(thand),
                "--out",
                str(out_dir),
                "--timeout",
                str(timeout),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 120,
        )
        vj = out_dir / f"{op}-verdict.json"
        if not vj.exists():
            return {
                "op": op,
                "engine": "formal_equiv",
                "class": "UNSWEPT",
                "verdict": "PROVER-FAILED",
                "reason": (r.stderr or r.stdout)[-300:],
                "wall_s": round(time.time() - t0, 1),
            }
        d = json.loads(vj.read_text())
        v = d["verdict"]
        cls = {
            "PROVEN-EQUIV-ALL-INPUTS": "SMT-PROVEN-ALL-INPUTS",
            "PROVEN-EQUIV-ON-DOCUMENTED-DOMAIN": "SMT-PROVEN-DOMAIN",
            "DIVERGENT": "DIVERGENCE-CERTIFIED",
            "UNDECIDED": "UNDECIDED-Z3-TIMEOUT",
            "SEMANTICS-UNVALIDATED": "SCOPE-REFUSED",
        }.get(v, "UNSWEPT")
        det = d.get("details", {})
        return {
            "op": op,
            "engine": "formal_equiv",
            "class": cls,
            "verdict": v,
            "cells": det.get("cells"),
            "unique_queries": det.get("unique_queries"),
            "witness": det.get("witness"),
            "validation": d.get("validation", {}).get("status"),
            "reason": man_row["reason"],
            "wall_s": round(time.time() - t0, 1),
        }
    except subprocess.TimeoutExpired:
        return {
            "op": op,
            "engine": "formal_equiv",
            "class": "UNDECIDED-Z3-TIMEOUT",
            "verdict": "UNDECIDED",
            "reason": f"per-op wall timeout {timeout}s expired",
            "wall_s": timeout,
        }


# ---------------------------------------------------------------------------
# Engine: bitexact_sweep  (sim-only probe/sweep/verdict for a batch of rows).
# ---------------------------------------------------------------------------
def run_bitexact_batch(rows, out_dir, flags, jobs, timeout):
    """Drive bitexact_sweep.py on `rows` (sim-only) and parse its ledger."""
    bdir = out_dir / "bitexact"
    bdir.mkdir(parents=True, exist_ok=True)
    log = bdir / "bitexact.log"
    with log.open("w") as fh:
        subprocess.run(
            [
                str(VENV_PY),
                str(BITEXACT_ENGINE),
                "--out",
                str(bdir),
                "--rows",
                ",".join(rows),
                "--stage",
                "all",
                "--jobs",
                str(jobs),
                "--timeout",
                str(timeout),
                "--sim",
                str(BITEXACT_SIM),
                "--flags",
                flags,
            ],
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    ledger = bdir / "BIT-EXACT-LEDGER.tsv"
    out = {}
    if ledger.exists():
        for r in read_tsv(ledger):
            out[r["row"]] = r
    jn_map = {
        "BIT-EXACT-ALL-INPUTS": "SIM-BIT-EXACT-16",
        "DIVERGENT": "DIVERGENCE-CERTIFIED",
        "INFEASIBLE-2^32-THIS-EXECUTOR": "INFEASIBLE-2^32",
        "NOT-EXHAUSTIBLE": "NOT-EXHAUSTIBLE",
        "EXECUTOR-UNVALIDATED": "SCOPE-REFUSED",
        "NOT-RUN-UNHOOKED-HARNESS": "UNSWEPT",
    }
    res = {}
    for row in rows:
        rr = out.get(row)
        if rr is None:
            res[row] = {
                "op": row,
                "engine": "bitexact",
                "class": "UNSWEPT",
                "verdict": "NO-LEDGER-ROW",
                "reason": "bitexact emitted no row",
            }
        else:
            v = rr["verdict"]
            res[row] = {
                "op": row,
                "engine": "bitexact",
                "class": jn_map.get(v, "UNSWEPT"),
                "verdict": v,
                "input_space": rr.get("input-space"),
                "diverging": rr.get("diverging-inputs"),
                "validation": rr.get("validation"),
                "reason": rr.get("note", ""),
            }
    return res


# ---------------------------------------------------------------------------
# Engine: classify  (no run — recorded infeasibility / cross-lane scope).
# ---------------------------------------------------------------------------
def run_classify(op, man_row):
    arity = man_row["arity_space"]
    if arity == "single-2^32":
        cls = "INFEASIBLE-2^32"
    elif arity.startswith("cross-lane") or "2^64" in arity:
        cls = "NOT-EXHAUSTIBLE"
    else:
        cls = "NOT-EXHAUSTIBLE"
    return {
        "op": op,
        "engine": "classify",
        "class": cls,
        "verdict": cls,
        "reason": man_row["reason"],
    }


# ---------------------------------------------------------------------------
# Overlays (recorded, provenance-pinned).
# ---------------------------------------------------------------------------
def load_silicon_overlay():
    ov = {}
    for r in read_tsv(SILICON_OVERLAY):
        ov[r["op"]] = r
    return ov


def load_domain_overlay():
    ov = {}
    for r in read_tsv(DOMAIN_OVERLAY):
        ov[r["op"]] = r
    return ov


# ---------------------------------------------------------------------------
# Per-op driver with resume + verdict cache.
# ---------------------------------------------------------------------------
def verdict_path(out_dir, op):
    return out_dir / "verdicts" / op / "prove_all_verdict.json"


def valid_cached(out_dir, op):
    p = verdict_path(out_dir, op)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        if d.get("class") in RANK and d.get("engine"):
            return d
    except Exception:
        return None
    return None


def save_verdict(out_dir, op, rec):
    p = verdict_path(out_dir, op)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))


# ---------------------------------------------------------------------------
# The join: engine verdict + silicon overlay + domain overlay -> final class.
# ---------------------------------------------------------------------------
def join_op(op, board_cls, engine_rec, sil, dom):
    contribs = []  # (class, source, detail)
    contribs.append(
        (engine_rec["class"], engine_rec["engine"], engine_rec.get("verdict", ""))
    )
    if op in sil:
        contribs.append(
            (
                sil[op]["silicon_class"],
                "KC-silicon(recorded)",
                sil[op]["silicon_verdict"],
            )
        )
    if op in dom:
        contribs.append(
            (dom[op]["domain_class"], "JO-domain(recorded)", dom[op]["jo_verdict"])
        )
    contribs = [c for c in contribs if c[0] in RANK]
    best = (
        min(contribs, key=lambda c: RANK[c[0]]) if contribs else ("UNSWEPT", "-", "-")
    )
    cls = best[0]
    equal_ev = [
        c
        for c in contribs
        if c[0]
        in (
            "SILICON-EXHAUSTIVE",
            "SMT-PROVEN-ALL-INPUTS",
            "SMT-PROVEN-DOMAIN",
            "SIM-BIT-EXACT-16",
        )
    ]
    div_ev = [c for c in contribs if c[0] == "DIVERGENCE-CERTIFIED"]
    return {
        "op": op,
        "board_class": board_cls,
        "provability_class": cls,
        "machine_certified_equal": "YES" if cls in MACHINE_CERTIFIED else "no",
        "engine": engine_rec["engine"],
        "engine_verdict": engine_rec.get("verdict", "-"),
        "arity_space": engine_rec.get("arity_space", "-"),
        "equal_evidence": ";".join("%s/%s" % (c[1], c[0]) for c in equal_ev) or "-",
        "divergence_evidence": ";".join(c[1] for c in div_ev) or "-",
        "evidence_ptr": engine_rec.get("evidence_ptr", "-"),
        "reason": engine_rec.get("reason", "-"),
    }


# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------
LEDGER_COLS = [
    "op",
    "board_class",
    "provability_class",
    "machine_certified_equal",
    "engine",
    "engine_verdict",
    "arity_space",
    "equal_evidence",
    "divergence_evidence",
    "evidence_ptr",
    "reason",
]


def write_ledger(out_dir, rows, prov):
    lp = out_dir / "MASTER-COVERAGE-LEDGER.tsv"
    with lp.open("w", newline="") as fh:
        fh.write(
            f"# prove_all MASTER-COVERAGE-LEDGER ({PIN}; cc1plus "
            f"{prov['shas'].get('cc1plus','?')[:12]}; JO-sim "
            f"{prov['shas'].get('jo_instrument_sim','?')[:12]}; bitexact-sim "
            f"{prov['shas'].get('bitexact_sim','?')[:12]}; {prov['timestamp']})\n"
        )
        fh.write(
            "# engines: formal_equiv (z3 TV, laneJO) + bitexact_sweep (2^16 sim, laneJN); "
            "overlays: KC-silicon(recorded device) + JO-domain(recorded z3).\n"
        )
        fh.write("# precedence: " + " > ".join(CLASS_ORDER) + "\n")
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS, delimiter="\t")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["op"]):
            w.writerow({k: r.get(k, "-") for k in LEDGER_COLS})
    return lp


def census(rows):
    return Counter(r["provability_class"] for r in rows)


def write_summary(out_dir, rows, prov, wall, fast_ops):
    cen = census(rows)
    mc = sum(1 for r in rows if r["machine_certified_equal"] == "YES")
    dom = cen.get("SMT-PROVEN-DOMAIN", 0)
    dv = cen.get("DIVERGENCE-CERTIFIED", 0)
    unreached = sum(
        cen.get(c, 0)
        for c in (
            "UNDECIDED-Z3-TIMEOUT",
            "INFEASIBLE-2^32",
            "NOT-EXHAUSTIBLE",
            "SCOPE-REFUSED",
            "UNSWEPT",
            "SIM-BIT-EXACT-16",
        )
    )
    by_op = {r["op"]: r for r in rows}
    fast_cen = Counter(by_op[o]["provability_class"] for o in fast_ops if o in by_op)
    fast_div = fast_cen.get("DIVERGENCE-CERTIFIED", 0)
    fast_cert = (
        len(fast_ops)
        - fast_div
        - fast_cen.get("UNDECIDED-Z3-TIMEOUT", 0)
        - fast_cen.get("NOT-EXHAUSTIBLE", 0)
    )
    L = []
    L.append("=" * 78)
    L.append(
        f"prove_all MASTER PROOF-COVERAGE SUMMARY  ({PIN}, cc1plus "
        f"{prov['shas'].get('cc1plus','?')[:12]})"
    )
    L.append("134 board-decided ops; one provability class each (strict precedence).")
    L.append(
        f"engines RE-RUN live: formal_equiv (z3 TV) + bitexact_sweep (2^16 sim). "
        f"overlays: KC-silicon + JO-domain (recorded, provenance-pinned)."
    )
    L.append("=" * 78)
    L.append("PROVABILITY CLASS CENSUS (all 134):")
    for c in CLASS_ORDER:
        if cen.get(c):
            tag = "  <== MACHINE-CERTIFIED-EQUAL" if c in MACHINE_CERTIFIED else ""
            L.append("   %-24s %3d%s" % (c, cen[c], tag))
    L.append("")
    L.append("HEADLINE:")
    L.append(
        "   MACHINE-CERTIFIED-EQUAL (silicon-exhaustive U SMT-proven-all-inputs) : %d"
        % mc
    )
    L.append(
        "   + SMT-PROVEN on documented deliverable domain (clamp, mulint32)      : %d"
        % dom
    )
    L.append(
        "   DIVERGENCE-CERTIFIED (proven-unequal, licensed/accuracy-gated)       : %d"
        % dv
    )
    L.append(
        "   UNREACHED (undecided/infeasible-2^32/not-exhaustible/scope/sim-only) : %d"
        % unreached
    )
    L.append("")
    L.append(
        "RECONCILIATION vs paper '36 fast ops -> 25 certified / 11 accuracy-gated':"
    )
    L.append(
        "   Fast set = %d laneJO-covered distinct-hand-leg (optimized sem!=hand) ops. Of these:"
        % len(fast_ops)
    )
    for c in CLASS_ORDER:
        if fast_cen.get(c):
            L.append("       %-24s %d" % (c, fast_cen[c]))
    L.append(
        "   => DIVERGENCE-CERTIFIED in fast set = %d  (paper's 11 accuracy-gated)"
        % fast_div
    )
    L.append("   => certified-or-domain in fast set  = %d" % fast_cert)
    L.append("=" * 78)
    L.append(f"total wall: {wall}")
    (out_dir / "SUMMARY.txt").write_text("\n".join(L) + "\n")
    return "\n".join(L)


def write_run_manifest(out_dir, prov, args, wall, rows):
    rec = {
        "pin": PIN,
        "timestamp": prov["timestamp"],
        "wall": wall,
        "shas": prov["shas"],
        "cc1plus_path": prov.get("cc1plus_path"),
        "jo_instrument_path": prov.get("jo_instrument_path"),
        "bitexact_sim_path": prov.get("bitexact_sim_path"),
        "args": vars(args),
        "census": dict(census(rows)),
        "provenance_problems": prov.get("problems", []),
    }
    (out_dir / "RUN-MANIFEST.json").write_text(json.dumps(rec, indent=2))


def write_sha256sums(out_dir):
    lines = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{sha256(p)}  {p.relative_to(out_dir)}")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--all", action="store_true", help="prove every board op")
    ap.add_argument(
        "--only", default=None, help="comma-separated glob(s) filtering op names"
    )
    ap.add_argument(
        "--engine",
        default=None,
        choices=["formal_equiv", "bitexact", "classify"],
        help="restrict to one engine's ops",
    )
    ap.add_argument("--out", default=str(HOME / "laneMH-evidence-20260903/run"))
    ap.add_argument("--jobs", type=int, default=8, help="bitexact sim parallelism")
    ap.add_argument("--timeout", type=int, default=1800, help="per-op wall seconds")
    ap.add_argument("--force", action="store_true", help="ignore cached verdicts")
    ap.add_argument(
        "--no-gate", action="store_true", help="record but do not enforce provenance"
    )
    args = ap.parse_args()

    board = load_board()
    man = load_manifest(board)
    prov = provenance_gate(strict=not args.no_gate)
    flags = on_flags()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdicts").mkdir(exist_ok=True)
    sil = load_silicon_overlay()
    dom = load_domain_overlay()
    fast_ops = [r["op"] for r in read_tsv(FAST_OPS)]

    # select ops
    ops = list(man)
    if args.only:
        pats = [p for p in args.only.split(",") if p]
        ops = [o for o in ops if any(fnmatch.fnmatch(o, p) for p in pats)]
    if args.engine:
        ops = [o for o in ops if man[o]["engine"] == args.engine]
    if not ops:
        sys.exit("no ops selected")
    print(
        f"prove_all {PIN}: {len(ops)} ops selected "
        f"(cc1plus {prov['shas'].get('cc1plus','?')[:12]}, "
        f"JO-sim {prov['shas'].get('jo_instrument_sim','?')[:12]}); out={out_dir}"
    )

    t_start = time.time()
    engine_recs = {}

    # cached / to-run split
    to_run = []
    for op in ops:
        if not args.force:
            c = valid_cached(out_dir, op)
            if c is not None:
                engine_recs[op] = c
                continue
        to_run.append(op)

    # classify (instant) + formal (serial, shares the pinned instrument)
    formal_ops = [o for o in to_run if man[o]["engine"] == "formal_equiv"]
    bitexact_ops = [o for o in to_run if man[o]["engine"] == "bitexact"]
    classify_ops = [o for o in to_run if man[o]["engine"] == "classify"]

    for op in classify_ops:
        rec = run_classify(op, man[op])
        rec["arity_space"] = man[op]["arity_space"]
        rec["evidence_ptr"] = "manifest-classification"
        engine_recs[op] = rec
        save_verdict(out_dir, op, rec)
        print(f"  [classify] {op:28s} -> {rec['class']}")

    # bitexact batch (sim-only, parallel inside the engine)
    if bitexact_ops:
        print(
            f"  [bitexact] running {len(bitexact_ops)} rows (2^16 sim, jobs={args.jobs}) ..."
        )
        res = run_bitexact_batch(bitexact_ops, out_dir, flags, args.jobs, args.timeout)
        for op in bitexact_ops:
            rec = res[op]
            rec["arity_space"] = man[op]["arity_space"]
            rec["evidence_ptr"] = f"bitexact/BIT-EXACT-LEDGER.tsv:{op}"
            engine_recs[op] = rec
            save_verdict(out_dir, op, rec)
            print(f"  [bitexact] {op:28s} -> {rec['class']} ({rec.get('verdict')})")

    # formal (serial: the instrumented sim + z3 are shared, deep queries are CPU-bound)
    fdir = out_dir / "formal"
    fdir.mkdir(exist_ok=True)
    for i, op in enumerate(formal_ops, 1):
        print(f"  [formal {i}/{len(formal_ops)}] {op} ...", flush=True)
        odir = fdir / op
        odir.mkdir(parents=True, exist_ok=True)
        rec = run_formal(op, man[op], odir, flags, args.timeout)
        rec["arity_space"] = man[op]["arity_space"]
        rec["evidence_ptr"] = f"formal/{op}/{op}-verdict.json"
        engine_recs[op] = rec
        save_verdict(out_dir, op, rec)
        print(f"      -> {rec['class']} ({rec.get('verdict')}, {rec.get('wall_s')}s)")

    # join
    rows = []
    for op in ops:
        rows.append(join_op(op, board[op], engine_recs[op], sil, dom))

    wall = time.strftime("%H:%M:%S", time.gmtime(time.time() - t_start))
    lp = write_ledger(out_dir, rows, prov)
    summ = write_summary(out_dir, rows, prov, wall, fast_ops)
    write_run_manifest(out_dir, prov, args, wall, rows)
    write_sha256sums(out_dir)
    print("\n" + summ)
    print(f"\nledger  -> {lp}")
    print(f"summary -> {out_dir/'SUMMARY.txt'}")


if __name__ == "__main__":
    main()
