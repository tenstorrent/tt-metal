#!/usr/bin/env python3
"""One-command {semantic, hand} x {passes OFF, ON} silicon sweep (HANDOFF §1/§3).

Encodes the silicon protocol as executable policy:
  1. changed-binary classification BEFORE any device job — a byte-identical
     OFF/ON pair is a recorded refusal, never a device run;
  2. paired CRAQ correctness (generic-path libttsim) before silicon;
  3. every device job serialized under BOTH exclusive flocks
     (/tmp/tt-device.lock outer, /tmp/tt-llk-sfpu-silicon.lock inner);
  4. per selector: correctness OFF+ON, then 3 fresh profiler processes per
     leg, alternating OFF/ON, each with a unique RUNNER_TEMP;
  5. raw+post perf CSVs copied while the lock is still held (they are
     overwritten per run); results are parsed only after the lock is released;
  6. hand OFF==ON byte-identity fills both hand cells with one physical run;
  7. per-op evidence: ELFs, .text hashes, build.h, logs, CSVs, compiler
     sha256, SHA256SUMS manifest.

Rows/markers/nodes live in sweep_2x2_ops.tsv, never in this file.  Absent
rows are machine-readable SKIPs.  Metric: post CSV mean(<metric>) at the
row's marker (KERNEL for fire-and-forget replay-launch shapes, TILE_LOOP for
eltwise suites) divided by tile_cnt = cycles/tile (per_tile=0 rows keep the
absolute scoped reading, e.g. the Reduce-SDPA REDUCE_SDPA_BODY pair).

Post-review hardening (PULL_ANALYSIS-20260817 §4):
  * the toolchain pin is the CC1PLUS binary (resolved via g++
    -print-prog-name=cc1plus): the g++ driver is byte-identical across
    cc1plus-only changes, so the driver sha alone is structurally blind and
    is kept only as a secondary check (D6);
  * resume is HASH-MATCHED: a cached device job is reused only when its
    archived .text hash set equals what THIS run's compiler produces for the
    same node/flags (stale-compiler cells re-measure); classify/CRAQ verdicts
    are keyed to the cc1plus (and simulator) sha and re-run on mismatch;
  * weekly per-knob silicon legs run the identical classify -> paired CRAQ ->
    correctness-then-perf pipeline as the main legs (D3);
  * report() is class-aware: baseline rows carry an expected class
    (win/parity/loss/refusal); a prior win row that becomes a byte-identical
    refusal is RED, refusal->changed is a flagged notice (D4;
    selftest_sweep_2x2_report.py proves win->refusal = RED);
  * rows with issue_slot_lb get the HANDOFF §1 issue-slot sanity check:
    a BODY-family reading on a macro-launch shape below the payload's
    issue-slot lower bound is INVALID_MARKER (KERNEL marker required);
  * kind=pinpair rows (Reduce-SDPA) run a paired gen-vs-hand A/B at the
    row's pinned flag set (default profitability gate), keeping the checked
    -in baseline pair and the compiler pin coherent.

Enforcement layer (ledger item 10, 2026-08-17 — the wave-6 violations were
"all one missing enforcement layer, repeated"; these convert the by-memory
rules into mechanical gates):
  * REVIEW RECORD REQUIRED (makes HANDOFF §1(4) unbypassable): a sweep whose
    phases include silicon and that authorizes hardware REFUSES in preflight
    unless <evidence-root>/../REVIEW_RECORD-<cc1plus-12hex>.md exists for the
    CURRENT cc1plus pin, names a reviewer, lists the reviewed commits and the
    gates checked, and quotes the full cc1plus sha256 (pin-match).  The
    record's sha256 is written into preflight.json and MANIFEST.txt so the
    evidence carries which review authorized it.  Template:
    corpus/REVIEW_RECORD_TEMPLATE.md.
  * SIM SHA PINNED (closes ledger 8(e): "any env-supplied libttsim.so
    satisfies the D3 gate"): --sim-bh-sha/--sim-wh-sha carry the reviewed
    libttsim sha256 pins from sweep_2x2.conf; preflight and EVERY phase
    entry (verify_toolchain) re-hash the simulators against them and refuse
    on mismatch, exactly like the compiler pins.
  * MACRO-LAUNCH ROWS REQUIRE issue_slot_lb (closes ledger 8(i)/V3: the
    check was "opt-in per row, not structural" and a headline rode an empty
    lb): the classify phase disassembles each leg's math.elf (objdump is a
    preflight-verified tool) and records a macro_scan verdict; a row whose
    ON binary contains SFPLOADMACRO launches, or fire-and-forget replay
    launches absent from the OFF leg, with an EMPTY issue_slot_lb is RED —
    named in the report with the §1 caveat — never a silent no-op.
  * issue_slot_lb units: the bound is compared against the row's RECORDED
    cell values.  For marker=TILE_LOOP rows the post CSV mean(...) is
    already per-tile (helpers/perf/core.py postprocess_tile_loop divides by
    loop_factor*tile_cnt) and _perf_value divides by tile_cnt again — a
    historical units convention kept for baseline continuity (uniform
    across cells, every booked ratio unaffected).  A TILE_LOOP row's lb
    must therefore be the true per-tile issue-slot bound divided by the
    fixture tile_cnt; each row's note records the raw arithmetic.

Sweep-hardening round 2 (adversarial review, 2026-08-16):
  * the silicon phase trusts NOTHING unkeyed: the BH CRAQ gate re-validates
    every verdict against THIS run's cc1plus+simulator+tt-metal keys, and a
    row without classify evidence keyed to this run is withheld RED (a
    `--phases silicon` resume on an old evidence root can no longer reuse a
    stale-toolchain green or skip the byte-identical refusal logic);
  * cached device jobs re-run when the classify hash reference is absent
    (expected_texts=None never reuses) and are additionally keyed on the
    pytest node id + flags + extra_env (jobkey.json);
  * tt_metal_head keys carry a +dirty.<sha> suffix when tracked tt-llk files
    are modified, so an edited kernel/TSV re-derives evidence;
  * every perf selector requires its own correctness selector (ops-load
    validation, loud failure) — no device perf cell without a correctness
    gate on the same leg;
  * report() acceptance is class- AND magnitude-aware: per-cell ABSOLUTE
    cycle drift vs baseline (uniform slowdowns, hand legs on refusal rows),
    INVALID_METRIC (unparsable metric on a row with baseline history = RED),
    WIN→PARITY = RED (unless --allow-win-to-parity), loss growth beyond
    --red-loss-growth-pct = RED; YELLOW rows show as 'YELLOW', never 'ok';
  * the toolchain the pytest HARNESS uses (tests/sfpi, an untracked
    repointable symlink — test_config.py hardcodes it) is the pinned
    subject: preflight records its realpath, refuses a divergent
    --compiler, and the harness-resolved cc1plus is re-verified against the
    pin at every phase entry.

Typical one-command full sweep:
  python3 tt_metal/tt-llk/tests/corpus/sweep_2x2.py \
    --evidence-root ~/sfpi-uplift/sweep-2x2/evidence-$(date +%Y%m%d) \
    --sim-bh <libttsim-bh> --sim-wh <libttsim-wh> --allow-hardware \
    --baseline tt_metal/tt-llk/tests/corpus/sfpu_device_baseline_p150_v1.tsv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[4]
LLK = ROOT / "tt_metal/tt-llk"
TESTS = LLK / "tests"
PYDIR = TESTS / "python_tests"
DEFAULT_CONFIG = HERE / "sweep_2x2_ops.tsv"

# Post-WP8 flag sets (HANDOFF §10 as amended: the -m{no-,}tt-tensix-{analyze,
# emit}-loadmacro flags were REMOVED with the quarantined exact-calendar pass
# and now error on use; the planner ON leg is -mtt-tensix-macro-planner).
OFF_FLAGS = (
    "-mno-tt-tensix-optimize-latency-schedule "
    "-mno-tt-tensix-optimize-dst-iteration-fusion "
    "-mno-tt-tensix-optimize-replay-hoist "
    "-mno-tt-tensix-optimize-invariant-loadi "
    "-mno-tt-tensix-optimize-dst-autoincr "
    "-mno-tt-tensix-optimize-dst-ownership "
    "-mno-tt-tensix-optimize-lut-select "
    "-mno-tt-tensix-optimize-setexp-fold "
    "-mno-tt-tensix-optimize-prgm-const "
    "-mno-tt-tensix-optimize-capture-rotation"
)
ON_FLAGS = (
    "-mtt-tensix-optimize-latency-schedule "
    "-mtt-tensix-optimize-dst-iteration-fusion "
    "-mtt-tensix-optimize-replay-hoist "
    "-mtt-tensix-optimize-invariant-loadi "
    "-mtt-tensix-optimize-dst-autoincr "
    "-mtt-tensix-optimize-dst-ownership "
    "-mtt-tensix-optimize-lut-select "
    "-mtt-tensix-optimize-setexp-fold "
    "-mtt-tensix-macro-planner "
    "-mtt-tensix-macro-planner-replay "
    "-mtt-tensix-optimize-mop-form "
    "-mtt-tensix-optimize-capture-rotation "
    # M3 fire (pin 9): prgm-const needs the tt-metal ttregion markers compiled
    # in, and markers without the fire are pure ghost-scheduling ripple (AQ),
    # so the flag and the define travel together or not at all.
    "-mtt-tensix-optimize-prgm-const -DLLK_ENABLE_TTREGION_MARKERS"
)
REMOVED_FLAGS = ("-mtt-tensix-emit-loadmacro", "-mtt-tensix-analyze-loadmacro")
# Weekly per-knob attribution: OFF set plus exactly one positive knob.
KNOBS = {
    "latency-schedule": "-mtt-tensix-optimize-latency-schedule",
    "dst-iteration-fusion": "-mtt-tensix-optimize-dst-iteration-fusion",
    "replay-hoist": "-mtt-tensix-optimize-replay-hoist",
    "invariant-loadi": "-mtt-tensix-optimize-invariant-loadi",
    "dst-autoincr": "-mtt-tensix-optimize-dst-autoincr",
    "macro-planner": "-mtt-tensix-macro-planner",
    "dst-ownership": "-mtt-tensix-optimize-dst-ownership",
    "lut-select": "-mtt-tensix-optimize-lut-select",
    "setexp-fold": "-mtt-tensix-optimize-setexp-fold",
    "planner-replay": "-mtt-tensix-macro-planner-replay",
    "mop-form": "-mtt-tensix-optimize-mop-form",
    "capture-rotation": "-mtt-tensix-optimize-capture-rotation",
    # The knob leg must carry the marker define for the same reason as the ON
    # set: prgm-const without ttregion markers cannot fire.
    "prgm-const": "-mtt-tensix-optimize-prgm-const -DLLK_ENABLE_TTREGION_MARKERS",
}
HARNESS_TOOLCHAIN = TESTS / "sfpi"  # untracked symlink the harness hardcodes
DEVICE_LOCK = "/tmp/tt-device.lock"
SILICON_LOCK = "/tmp/tt-llk-sfpu-silicon.lock"
CHIP = {"bh": "blackhole", "wh": "wormhole"}
SELECTORS = ("sem-corr", "sem-perf", "hand-corr", "hand-perf")
PERF_RUNS = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- enforcement-layer gate primitives (selftest-covered) ----
# Kept as module-level pure(ish) functions so selftest_enforcement_gates.py
# can drive the REAL gate logic without a toolchain or device.


def check_review_record(record_path, cc1plus_sha256):
    """REVIEW_RECORD gate (HANDOFF §1(4) as code).

    Returns (ok, detail).  A silicon-authorized sweep refuses unless the
    record at <evidence-root>/../REVIEW_RECORD-<cc1plus-12hex>.md exists,
    quotes the FULL cc1plus sha256 (pin-match — a record for another build
    never authorizes this one), names a reviewer, and carries the reviewed
    commits and gates-checked sections.  Content honesty stays human; the
    gate makes the record's EXISTENCE and pin-binding mechanical.
    """
    record_path = pathlib.Path(record_path)
    if not record_path.is_file():
        return False, f"missing review record {record_path}"
    text = record_path.read_text(errors="replace")
    if cc1plus_sha256 not in text:
        return False, (
            f"review record {record_path} does not quote the full pinned "
            f"cc1plus sha256 {cc1plus_sha256} — pin-match failed (a record "
            "minted for another build does not authorize this one)"
        )
    checks = (
        (r"(?mi)^\s*(?:[-*]\s*)?Reviewer\s*:\s*\S", "a non-empty 'Reviewer:' line"),
        (r"(?mi)^##\s*Reviewed\b", "a '## Reviewed ...' commits/branches section"),
        (r"(?mi)^##\s*Gates\b", "a '## Gates checked' section"),
    )
    missing = [what for pat, what in checks if not re.search(pat, text)]
    if missing:
        return False, (
            f"review record {record_path} is malformed — missing "
            + "; ".join(missing)
            + " (see corpus/REVIEW_RECORD_TEMPLATE.md)"
        )
    return True, "ok"


# Fire-and-forget replay launch = TT_OP_REPLAY with execute_while_loading=0
# and load_mode=0 (a pure launch of previously recorded slots).  Record-form
# replays (…,1,1 / …,0,1) are the drain-synchronous record path and are NOT
# launches.
_SCAN_PATTERNS = {
    "sfploadmacro": re.compile(r"\bsfploadmacro\b"),
    "replay_launch": re.compile(r"\bttreplay\s+\d+,\s*\d+,\s*0,\s*0\b"),
}


def scan_disasm_text(text):
    """Mnemonic census of one disassembly relevant to macro-launch detection."""
    return {name: len(pat.findall(text)) for name, pat in _SCAN_PATTERNS.items()}


def classify_macro_launch(on_counts, off_counts=None):
    """Macro-launch classification of an ON leg census vs its OFF leg.

    Returns 'MACRO_LAUNCH' (SFPLOADMACRO launches present),
    'REPLAY_LAUNCH' (fire-and-forget replay launches beyond the OFF leg's —
    source-level lltt::record replays appear in both legs and do not count),
    or None.  With no OFF census (single-leg pinpair rows) only the
    SFPLOADMACRO criterion applies; replay counts stay informational.
    """
    if on_counts.get("sfploadmacro", 0) > 0:
        return "MACRO_LAUNCH"
    if off_counts is not None and on_counts.get("replay_launch", 0) > off_counts.get(
        "replay_launch", 0
    ):
        return "REPLAY_LAUNCH"
    return None


def macro_lb_red(op, marker, issue_slot_lb, macro_scan):
    """RED message for a macro-launch row with an EMPTY issue_slot_lb, else
    None.  Ledger 8(i)/wave-6 V3: an empty lb silently no-opped the §1
    issue-slot sanity check under exactly the shapes that need it most."""
    if not macro_scan or issue_slot_lb is not None:
        return None
    cls = macro_scan.get("classification")
    if not cls:
        return None
    return (
        f"{op}: {cls} row measured at marker {marker} with EMPTY "
        f"issue_slot_lb (ON binary: {macro_scan.get('sfploadmacro_on', 0)} "
        f"SFPLOADMACRO launches, {macro_scan.get('replay_launch_on', 0)} "
        "fire-and-forget replay launches) — HANDOFF §1 metric caveat: a "
        "BODY-family reading on a fire-and-forget shape is INVALID below the "
        "payload's issue-slot lower bound, and without issue_slot_lb that "
        "check cannot run; populate issue_slot_lb (units: the row's recorded "
        "cell units, see sweep_2x2_ops.tsv header) or move the row to the "
        "drain-inclusive KERNEL marker"
    )


def load_config(path):
    with path.open() as f:
        rows = list(
            csv.DictReader((x for x in f if not x.startswith("#")), delimiter="\t")
        )
    for row in rows:
        row["nodes"] = {
            sel: (row.get(sel.replace("-", "_")) or "").strip() for sel in SELECTORS
        }
        # Optional columns (sweep-2x2-ops-version 2); absent = v1 defaults.
        row["metric"] = (row.get("metric") or "").strip() or "MATH_ISOLATE"
        row["per_tile"] = (row.get("per_tile") or "1").strip() != "0"
        lb = (row.get("issue_slot_lb") or "").strip()
        row["issue_slot_lb"] = float(lb) if lb else None
        row["pin_flags"] = (row.get("pin_flags") or "").strip()
        env = (row.get("extra_env") or "").strip()
        row["extra_env"] = dict(kv.split("=", 1) for kv in env.split(";") if kv)
        if row["kind"] == "pinpair" and not row["pin_flags"]:
            sys.exit(f"config row {row['op']}: kind=pinpair requires pin_flags")
        # Sweep-hardening 2: a perf leg without its own correctness leg
        # produces device cycles from a kernel nothing verified — a broken
        # hand kernel would silently keep feeding vs_hand_pct GREEN.  Loud
        # failure at ops-load; withhold the perf node until a corr node lands.
        if row["kind"] != "skip":
            for perf_sel, corr_sel in (
                ("sem-perf", "sem-corr"),
                ("hand-perf", "hand-corr"),
            ):
                if row["nodes"][perf_sel] and not row["nodes"][corr_sel]:
                    sys.exit(
                        f"config row {row['op']}: {perf_sel} has a node but "
                        f"{corr_sel} is empty — every device perf leg requires "
                        "its own correctness node (perf cycles from an "
                        "unverified kernel are not evidence); add the corr "
                        "node or withhold the perf leg in the row note"
                    )
    return rows


def row_scope(row):
    """Baseline/scoreboard scope string for a config row (or stored result)."""
    if row["kind"] == "pinpair":
        return row["marker"]
    return f"{row['marker']}_{row.get('metric', 'MATH_ISOLATE')}_PER_TILE"


# Perf-cell naming per row kind.  pinpair rows keep the checked-in baseline's
# native selectors (e.g. Reduce-SDPA 'generated'/'handwritten_replay').
PINPAIR_CELLS = {"sem-perf": "generated", "hand-perf": "handwritten_replay"}


def cell_selector(r, cell):
    """Baseline/scoreboard selector for a result's cell."""
    if r["kind"] == "pinpair":
        return cell
    return f"{r['op']}:{cell}"


class Sweep:
    def __init__(self, args):
        self.a = args
        self.ev = args.evidence_root.resolve()
        self.compiler = (
            args.compiler or TESTS / "sfpi/compiler/bin/riscv-tt-elf-g++"
        ).resolve()
        self.objcopy = self.compiler.with_name("riscv-tt-elf-objcopy")
        self.objdump = self.compiler.with_name("riscv-tt-elf-objdump")
        self.python = self._find_python(args.venv)
        self.rows = [
            r for r in load_config(args.config) if not args.ops or r["op"] in args.ops
        ]
        if args.ops:
            missing = set(args.ops) - {r["op"] for r in self.rows}
            if missing:
                sys.exit(f"unknown ops in --ops: {','.join(sorted(missing))}")
        self.reds = []

    @staticmethod
    def _find_python(venv):
        candidates = (
            [venv]
            if venv
            else [TESTS / ".venv-laneE", TESTS / ".venv", PYDIR / ".venv"]
        )
        for c in candidates:
            if c and (pathlib.Path(c) / "bin/python").is_file():
                return pathlib.Path(c) / "bin/python"
        sys.exit(
            "no tt-llk virtualenv found (looked for tests/.venv-laneE, tests/.venv, "
            "tests/python_tests/.venv); pass --venv"
        )

    # ---------------- preflight ----------------
    @staticmethod
    def _pin_value(pin, what):
        """A pin must be a FULL 64-hex sha256.  The previous prefix
        acceptance (startswith) meant a 1-char env leak 'pinned' essentially
        nothing (adversarial finding sweep_2x2.conf:31)."""
        pin = pin.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", pin):
            sys.exit(
                f"{what} pin '{pin}' is not a full 64-hex sha256 — refusing "
                "to sweep (prefix pins accept almost anything; pass the "
                "complete sha256 from sweep_2x2.conf)"
            )
        return pin

    def _resolve_cc1plus(self):
        """cc1plus resolved through the driver (the binary that compiles)."""
        cc1 = subprocess.run(
            [str(self.compiler), "-print-prog-name=cc1plus"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not cc1 or not pathlib.Path(cc1).is_file():
            sys.exit(f"cannot resolve cc1plus via {self.compiler} (got '{cc1}')")
        return cc1, sha256(pathlib.Path(cc1))

    def verify_toolchain(self, phase):
        """Re-verify the harness toolchain identity at a phase entry.

        tests/sfpi is an UNTRACKED, repointable symlink and the pytest
        harness hardcodes it (test_config.py setup_paths) — a mid-run
        repoint would silently measure with an unpinned compiler while the
        manifest swears otherwise (adversarial finding sweep_2x2.py:160).
        """
        real = str(HARNESS_TOOLCHAIN.resolve())
        cc1, cc1_sha = self._resolve_cc1plus()
        if (
            real != self.info["harness_toolchain_realpath"]
            or cc1_sha != self.info["cc1plus_sha256"]
        ):
            sys.exit(
                f"TOOLCHAIN CHANGED MID-RUN (phase '{phase}'): tests/sfpi now "
                f"resolves to {real} with cc1plus {cc1_sha} at {cc1}; "
                f"preflight recorded {self.info['harness_toolchain_realpath']} "
                f"with cc1plus {self.info['cc1plus_sha256']} — refusing to "
                "continue (evidence already produced is keyed to the "
                "preflight identity)"
            )
        # Sim sha pin: re-verified at EVERY phase entry, not just preflight —
        # a mid-run libttsim swap must never let an unpinned oracle open the
        # CRAQ gate (ledger 8(e)).
        self._verify_sim_pins(phase)

    def _verify_sim_pins(self, phase):
        """Hash each provided simulator against its reviewed pin (and the
        preflight-recorded identity); refuse on any mismatch."""
        for arch in ("bh", "wh"):
            sim = getattr(self.a, f"sim_{arch}")
            pin = getattr(self.a, f"sim_{arch}_sha", None)
            if not pin:
                continue
            pin = self._pin_value(pin, f"libttsim {arch} (--sim-{arch}-sha)")
            if not sim or not sim.is_file():
                sys.exit(
                    f"SIM PIN {arch} set ({pin}) but no simulator file at "
                    f"'{sim}' (phase '{phase}') — a pinned CRAQ oracle that "
                    "silently degrades to SKIP_NO_SIMULATOR would withhold "
                    "silicon rows one by one instead of failing loudly; "
                    "build/point the pinned libttsim or drop the pin through "
                    "review"
                )
            found = sha256(sim)
            if found != pin:
                sys.exit(
                    f"SIM SHA MISMATCH ({arch}, phase '{phase}'): pinned "
                    f"{pin}, found {found} at {sim} — refusing (the CRAQ "
                    "oracle is pinned like the compiler: any env-supplied "
                    "libttsim.so must hash to the reviewed value; re-pin "
                    "through review, never through the environment)"
                )
            recorded = (
                self.info.get(f"sim_{arch}_sha256") if hasattr(self, "info") else None
            )
            if recorded and recorded != found:
                sys.exit(
                    f"SIMULATOR CHANGED MID-RUN ({arch}, phase '{phase}'): "
                    f"preflight recorded {recorded}, now {found} at {sim} — "
                    "refusing (evidence already produced is keyed to the "
                    "preflight identity)"
                )

    def preflight(self):
        self.ev.mkdir(parents=True, exist_ok=True)
        info = {
            "compiler": str(self.compiler),
            "off_flags": OFF_FLAGS,
            "on_flags": ON_FLAGS,
            "config": str(self.a.config),
            "evidence_root": str(self.ev),
        }
        if not self.compiler.is_file():
            sys.exit(f"missing compiler {self.compiler}")
        # The pytest harness HARDCODES its toolchain to the tests/sfpi
        # symlink (test_config.py setup_paths: TOOL_PATH = LLK_ROOT /
        # 'tests/sfpi/compiler/bin') and the sweep passes only flags/env —
        # never a compiler path.  --compiler therefore controls what
        # preflight HASHES, not what BUILDS: a divergent --compiler would
        # verify one binary and measure with another (adversarial finding
        # sweep_2x2.py:160).  Enforce that the pinned subject IS the harness
        # toolchain, and record the symlink's realpath as evidence.
        harness_gxx = (HARNESS_TOOLCHAIN / "compiler/bin/riscv-tt-elf-g++").resolve()
        info["harness_toolchain_symlink"] = str(HARNESS_TOOLCHAIN)
        info["harness_toolchain_realpath"] = (
            str(HARNESS_TOOLCHAIN.resolve()) if HARNESS_TOOLCHAIN.exists() else ""
        )
        if self.compiler != harness_gxx:
            sys.exit(
                f"--compiler {self.compiler} is NOT the harness toolchain "
                f"{harness_gxx} (tests/sfpi resolves to "
                f"{info['harness_toolchain_realpath'] or 'MISSING'}): the "
                "pytest harness hardcodes tests/sfpi/compiler/bin "
                "(test_config.py), so every build would use the harness "
                "toolchain while preflight verified a different binary — "
                "repoint tests/sfpi at the pinned build or drop --compiler"
            )
        # SECONDARY pin: the g++ driver.  Historically byte-identical across
        # cc1plus-only rebuilds (structurally blind, D6) — it can catch a
        # wrong toolchain layout but never a compiler-proper change.  Full
        # sha equality required (no prefixes).
        info["compiler_sha256"] = sha256(self.compiler)
        if self.a.compiler_sha and (
            self._pin_value(self.a.compiler_sha, "driver (--compiler-sha)")
            != info["compiler_sha256"]
        ):
            sys.exit(
                f"DRIVER SHA MISMATCH: pinned {self.a.compiler_sha}, "
                f"found {info['compiler_sha256']} — refusing to sweep"
            )
        # PRIMARY pin: cc1plus (the compiler proper), resolved through the
        # driver itself so the pin follows whatever binary actually compiles.
        cc1, cc1_sha = self._resolve_cc1plus()
        info["cc1plus"] = cc1
        info["cc1plus_sha256"] = cc1_sha
        if self.a.cc1plus_sha and (
            self._pin_value(self.a.cc1plus_sha, "cc1plus (--cc1plus-sha)")
            != info["cc1plus_sha256"]
        ):
            sys.exit(
                "CC1PLUS SHA MISMATCH (primary toolchain pin): pinned "
                f"{self.a.cc1plus_sha}, found {info['cc1plus_sha256']} at {cc1} "
                "— refusing to sweep (the g++ driver sha alone cannot detect "
                "cc1plus-only changes; rebuild/point the pinned toolchain or "
                "update the pin through review)"
            )
        ver = subprocess.run(
            [str(self.compiler), "--version"], capture_output=True, text=True
        )
        info["compiler_version"] = (
            (ver.stdout or "").splitlines()[0] if ver.stdout else ""
        )
        # The removed exact-calendar flags MUST error on use (post-WP8 pin proof).
        for flag in REMOVED_FLAGS:
            probe = subprocess.run(
                [
                    str(self.compiler),
                    "-mcpu=tt-bh-tensix",
                    flag,
                    "-fsyntax-only",
                    "-x",
                    "c++",
                    "-",
                ],
                input="int main(){return 0;}",
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                sys.exit(
                    f"pin check failed: removed flag {flag} was ACCEPTED — wrong toolchain"
                )
        info["removed_flags_error_on_use"] = True
        # Both flag sets must be accepted.
        for label, flags in (("off", OFF_FLAGS), ("on", ON_FLAGS)):
            probe = subprocess.run(
                [
                    str(self.compiler),
                    "-mcpu=tt-bh-tensix",
                    *flags.split(),
                    "-fsyntax-only",
                    "-x",
                    "c++",
                    "-",
                ],
                input="int main(){return 0;}",
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0:
                sys.exit(
                    f"{label} flag set rejected by compiler:\n{probe.stdout}{probe.stderr}"
                )
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        # Sweep-hardening 2: `rev-parse HEAD` is blind to an UNCOMMITTED
        # working tree — an edited kernel header or ops TSV would resume
        # every classify/CRAQ verdict and device cell stale.  Key on the
        # tracked tt-llk diff as well (untracked files are excluded: the
        # tests/sfpi symlink and pytest __pycache__ churn would otherwise
        # invalidate every resume).
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "diff", "HEAD", "--", "tt_metal/tt-llk"],
            capture_output=True,
        ).stdout
        if dirty.strip():
            head += "+dirty." + hashlib.sha256(dirty).hexdigest()[:16]
        info["tt_metal_head"] = head
        for arch in ("bh", "wh"):
            sim = getattr(self.a, f"sim_{arch}")
            info[f"sim_{arch}"] = str(sim) if sim else ""
            info[f"sim_{arch}_sha256"] = sha256(sim) if sim and sim.is_file() else ""
            info[f"sim_{arch}_sha_pin"] = getattr(self.a, f"sim_{arch}_sha", "") or ""
        # objdump is a GATE TOOL (macro-launch classification): a missing
        # objdump would silently disable the issue_slot_lb requirement — the
        # exact silent-no-op class this enforcement layer exists to kill.
        if not self.objdump.is_file():
            sys.exit(
                f"missing objdump {self.objdump} — the classify phase "
                "disassembles every leg to detect macro-launch shapes "
                "(issue_slot_lb enforcement); a toolchain without objdump "
                "cannot run a gated sweep"
            )
        self.info = info  # phase-entry checks below read the recorded identity
        self._verify_sim_pins("preflight")
        # REVIEW RECORD gate (HANDOFF §1(4) as code): silicon-authorized
        # sweeps refuse without a pin-matched review record beside the
        # evidence root.  --phases without silicon (classify/craq-only
        # runs) and non-hardware runs stay ungated: they produce no device
        # evidence.
        info["review_record"] = ""
        info["review_record_sha256"] = ""
        if "silicon" in self.a.phases and self.a.allow_hardware:
            record = self.ev.parent / f"REVIEW_RECORD-{info['cc1plus_sha256'][:12]}.md"
            ok, detail = check_review_record(record, info["cc1plus_sha256"])
            if not ok:
                sys.exit(
                    "REVIEW RECORD REQUIRED (silicon phases authorized): "
                    + detail
                    + " — HANDOFF §1(4): independent review of compiler "
                    "mutations BEFORE silicon.  Write the record from "
                    "corpus/REVIEW_RECORD_TEMPLATE.md for the CURRENT "
                    f"cc1plus pin ({info['cc1plus_sha256']}), place it at "
                    f"{record}, then re-run.  No record, no silicon."
                )
            info["review_record"] = str(record)
            info["review_record_sha256"] = sha256(record)
        (self.ev / "preflight.json").write_text(json.dumps(info, indent=2) + "\n")
        man = [
            f"Lane sweep-2x2 evidence — {self.ev.name}",
            f"compiler driver: {self.compiler}",
            f"compiler driver sha256 (secondary pin): {info['compiler_sha256']}",
            f"cc1plus: {info['cc1plus']}",
            f"cc1plus sha256 (PRIMARY pin): {info['cc1plus_sha256']}",
            f"harness toolchain symlink: {info['harness_toolchain_symlink']}",
            "harness toolchain realpath (readlink -f, re-verified at every "
            f"phase entry): {info['harness_toolchain_realpath']}",
            f"compiler version: {info['compiler_version']}",
            f"tt-metal: {info['tt_metal_head']}",
            f"libttsim bh sha256: {info['sim_bh_sha256']}"
            + (
                f" (VERIFIED against reviewed pin {info['sim_bh_sha_pin']})"
                if info["sim_bh_sha_pin"]
                else " (UNPINNED — no --sim-bh-sha)"
            ),
            f"libttsim wh sha256: {info['sim_wh_sha256']}"
            + (
                f" (VERIFIED against reviewed pin {info['sim_wh_sha_pin']})"
                if info["sim_wh_sha_pin"]
                else " (UNPINNED — no --sim-wh-sha)"
            ),
            (
                f"review record: {info['review_record']} sha256 "
                f"{info['review_record_sha256']}"
                if info["review_record"]
                else "review record: not required (no silicon authorization this run)"
            ),
            f"OFF flags: {OFF_FLAGS}",
            f"ON flags: {ON_FLAGS}",
            "loadmacro flags: CONFIRMED error on use (removed with quarantined exact-calendar pass)",
        ]
        (self.ev / "MANIFEST.txt").write_text("\n".join(man) + "\n")
        self.info = info

    # ---------------- process helpers ----------------
    def _env(self, arch, runner_temp, flags, sim=None, extra=None):
        env = os.environ.copy()
        env.update(
            CHIP_ARCH=CHIP[arch],
            LLK_HOME=str(LLK),
            RUNNER_TEMP=str(runner_temp),
            TT_LLK_EXTRA_COMPILER_OPTIONS=flags,
        )
        if sim:
            env["TT_METAL_SIMULATOR"] = str(sim)
        if extra:
            env.update(extra)
        return env

    def _pytest(self, node, extra, env, log, timeout=1800):
        with open(log, "w") as f:
            rc = subprocess.run(
                [str(self.python), "-m", "pytest", "-q", *extra, node],
                cwd=PYDIR,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            ).returncode
        return rc

    @staticmethod
    def _passed(log):
        log = pathlib.Path(log)
        if not log.is_file():
            return False
        text = log.read_text(errors="replace")
        return bool(re.search(r"\b[1-9]\d* passed\b", text))

    def _hash_build(self, rt, out_file):
        """Hash .text and full bytes of every kernel ELF under one RUNNER_TEMP."""
        entries = []
        for elf in sorted((rt / "tt-llk-build").rglob("*.elf")):
            if "shared" in elf.parts:
                continue  # brisc bootrom is flag-independent scaffolding
            rel = elf.relative_to(rt / "tt-llk-build")
            text = subprocess.run(
                [
                    str(self.objcopy),
                    "-O",
                    "binary",
                    "--only-section=.text",
                    str(elf),
                    "/dev/stdout",
                ],
                capture_output=True,
            ).stdout
            entries.append((str(rel), hashlib.sha256(text).hexdigest(), sha256(elf)))
        with open(out_file, "w") as f:
            for rel, t, e in entries:
                f.write(f"{rel}\ttext:{t}\telf:{e}\n")
        return entries

    def _archive_build(self, rt, dest):
        """Keep ELFs and build.h from a RUNNER_TEMP; drop the rest."""
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted((rt / "tt-llk-build").rglob("*")):
            if (
                path.is_file()
                and (path.suffix == ".elf" or path.name == "build.h")
                and "shared" not in path.parts
            ):
                out = dest / path.relative_to(rt / "tt-llk-build")
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)

    # ---------------- phase: classify ----------------
    def classify(
        self, row, sel, legs=(("off", OFF_FLAGS), ("on", ON_FLAGS)), tag="classify"
    ):
        node = row["nodes"][sel]
        work = self.ev / row["op"] / tag / sel
        verdict_file = work / "verdict.json"
        if verdict_file.is_file() and not self.a.force:
            verdict = json.loads(verdict_file.read_text())
            # Hash-matched resume: a cached classification is only valid for
            # the compiler AND source tree that produced it.  Verdicts from
            # another cc1plus or tt-metal head (or the pre-keying schema)
            # are recompiled — kernel-source changes must re-derive hashes.
            if (
                verdict.get("cc1plus_sha256") == self.info["cc1plus_sha256"]
                and verdict.get("tt_metal_head") == self.info["tt_metal_head"]
                and (
                    verdict.get("status") != "OK" or "macro_scan" in verdict
                )  # pre-enforcement-layer verdicts lack the scan: re-derive
            ):
                return verdict
        work.mkdir(parents=True, exist_ok=True)
        (work / "node.txt").write_text(node + "\n")
        hashes = {}
        for leg, flags in legs:
            rt = work / f"rt-{leg}"
            shutil.rmtree(rt, ignore_errors=True)
            rt.mkdir(parents=True)
            (work / f"flags-{leg}.txt").write_text(flags + "\n")
            rc = self._pytest(
                node,
                ["--compile-producer"],
                self._env("bh", rt, flags, extra=row["extra_env"]),
                work / f"compile-{leg}.log",
            )
            if rc != 0 or not self._passed(work / f"compile-{leg}.log"):
                verdict = {
                    "selector": sel,
                    "status": "COMPILE_FAIL",
                    "leg": leg,
                    "cc1plus_sha256": self.info["cc1plus_sha256"],
                    "tt_metal_head": self.info["tt_metal_head"],
                }
                verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
                self.reds.append(f"{row['op']}/{sel}: compile {leg} failed")
                return verdict
            hashes[leg] = self._hash_build(rt, work / f"hashes-{leg}.txt")
            self._archive_build(rt, work / f"elf-{leg}")
            shutil.rmtree(rt, ignore_errors=True)
        legnames = [leg for leg, _ in legs]
        if len(legnames) == 1:
            verdict = {
                "selector": sel,
                "status": "OK",
                "all": "SINGLE_LEG",
                "math": "SINGLE_LEG",
            }
        else:
            a_set = sorted(h[1] for h in hashes[legnames[0]])
            b_set = sorted(h[1] for h in hashes[legnames[1]])
            math_a = sorted(
                h[1] for h in hashes[legnames[0]] if h[0].endswith("math.elf")
            )
            math_b = sorted(
                h[1] for h in hashes[legnames[1]] if h[0].endswith("math.elf")
            )
            verdict = {
                "selector": sel,
                "status": "OK",
                "all": "IDENTICAL" if a_set == b_set else "CHANGED",
                "math": "IDENTICAL" if math_a == math_b else "CHANGED",
            }
        verdict["macro_scan"] = self._macro_scan(work, legnames)
        verdict["cc1plus_sha256"] = self.info["cc1plus_sha256"]
        verdict["tt_metal_head"] = self.info["tt_metal_head"]
        verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
        return verdict

    def _scan_leg_disasm(self, work, leg):
        """Sum the macro-launch census over every archived math.elf of a
        classify leg (objdump -d; the classify phase already archives the
        ELFs, so the scan adds no compile work)."""
        counts = {name: 0 for name in _SCAN_PATTERNS}
        elfs = sorted((work / f"elf-{leg}").rglob("math.elf"))
        for elf in elfs:
            dis = subprocess.run(
                [str(self.objdump), "-d", str(elf)], capture_output=True, text=True
            )
            if dis.returncode != 0:
                sys.exit(
                    f"objdump failed on {elf} (macro-launch classification "
                    "is a gate; it must not silently degrade): "
                    f"{dis.stderr.strip()[:400]}"
                )
            for name, n in scan_disasm_text(dis.stdout).items():
                counts[name] += n
        counts["math_elfs"] = len(elfs)
        return counts

    def _macro_scan(self, work, legnames):
        """Macro-launch verdict for a classify evidence dir.

        Two-leg rows scan ON vs OFF (replay-launch = ON-only launches);
        single-leg rows (pinpair) scan their only leg with the
        SFPLOADMACRO criterion.  Returns the dict stored in the classify
        verdict as 'macro_scan'."""
        if len(legnames) == 1:
            on = self._scan_leg_disasm(work, legnames[0])
            off = None
        else:
            off = self._scan_leg_disasm(work, legnames[0])
            on = self._scan_leg_disasm(work, legnames[1])
        cls = classify_macro_launch(on, off)
        return {
            "classification": cls,
            "sfploadmacro_on": on.get("sfploadmacro", 0),
            "replay_launch_on": on.get("replay_launch", 0),
            "replay_launch_off": (off or {}).get("replay_launch", 0),
            "math_elfs_on": on.get("math_elfs", 0),
        }

    def _classify_texts(self, row, sel, leg, tag="classify"):
        """This run's .text hash set for (row, sel, leg) from the classify
        evidence — the reference a cached device job must hash-match."""
        path = self.ev / row["op"] / tag / sel / f"hashes-{leg}.txt"
        if not path.is_file():
            return None
        return self._texts_of(path)

    @staticmethod
    def _texts_of(hash_file):
        texts = []
        for line in pathlib.Path(hash_file).read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].startswith("text:"):
                texts.append(parts[1][len("text:") :])
        return sorted(texts)

    # ---------------- phase: craq ----------------
    SOC_DESCRIPTORS = {
        "bh": "tt_metal/soc_descriptors/blackhole_140_arch.yaml",
        "wh": "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml",
    }

    def _staged_sim(self, arch):
        """ttexalens needs soc_descriptor.yaml BESIDE libttsim.so; stage it.

        The craq-sim build tree ships only the .so, so a bare --sim-* path
        would fail with 'bad file: .../soc_descriptor.yaml'.  Stage the .so
        together with tt-metal's arch descriptor under the evidence root.
        """
        sim = getattr(self.a, f"sim_{arch}")
        if not sim or not sim.is_file():
            return None
        if (sim.parent / "soc_descriptor.yaml").is_file():
            return sim
        stage = self.ev / "simstage" / arch
        stage.mkdir(parents=True, exist_ok=True)
        if not (stage / "libttsim.so").is_file():
            shutil.copy2(sim, stage / "libttsim.so")
        shutil.copy2(ROOT / self.SOC_DESCRIPTORS[arch], stage / "soc_descriptor.yaml")
        return stage / "libttsim.so"

    def craq(
        self,
        row,
        sel,
        arch,
        legs_spec=(("off", OFF_FLAGS), ("on", ON_FLAGS)),
        tag="craq",
    ):
        node = row["nodes"][sel]
        sim = self._staged_sim(arch)
        work = self.ev / row["op"] / tag / f"{sel}-{arch}"
        verdict_file = work / "verdict.json"
        sim_sha = sha256(sim) if sim and sim.is_file() else ""
        if verdict_file.is_file() and not self.a.force:
            verdict = json.loads(verdict_file.read_text())
            # Hash-matched resume: verdicts are keyed to cc1plus + simulator
            # + tt-metal head (kernel-source changes re-run the pair).
            if (
                verdict.get("cc1plus_sha256") == self.info["cc1plus_sha256"]
                and verdict.get("sim_sha256") == sim_sha
                and verdict.get("tt_metal_head") == self.info["tt_metal_head"]
                and verdict.get("status") != "SKIP_NO_SIMULATOR"
            ):
                return verdict
        if not sim or not sim.is_file():
            verdict = {"selector": sel, "arch": arch, "status": "SKIP_NO_SIMULATOR"}
            work.mkdir(parents=True, exist_ok=True)
            verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
            return verdict
        work.mkdir(parents=True, exist_ok=True)
        (work / "node.txt").write_text(node + "\n")
        legs = {}
        for leg, flags in legs_spec:
            rt = work / f"rt-{leg}"
            shutil.rmtree(rt, ignore_errors=True)
            rt.mkdir(parents=True)
            log = work / f"craq-{leg}.log"
            rc = self._pytest(
                node,
                ["--run-simulator"],
                self._env(arch, rt, flags, sim=sim, extra=row["extra_env"]),
                log,
                timeout=2400,
            )
            text = log.read_text(errors="replace")
            if self._passed(log):
                legs[leg] = "PASS"
            elif "UnsupportedFunctionality" in text:
                legs[leg] = "UNSUPPORTED"
            elif re.search(r"\b[1-9]\d* skipped\b", text) and " failed" not in text:
                legs[leg] = "SKIPPED"
            else:
                legs[leg] = f"FAIL(rc={rc})"
            shutil.rmtree(rt, ignore_errors=True)
        verdict = {
            "selector": sel,
            "arch": arch,
            "status": "OK",
            "legs": legs,
            "cc1plus_sha256": self.info["cc1plus_sha256"],
            "sim_sha256": sim_sha,
            "tt_metal_head": self.info["tt_metal_head"],
        }
        verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
        if arch == "bh" and any(v != "PASS" for v in legs.values()):
            self.reds.append(f"{row['op']}/{sel}: CRAQ {arch} ({tag}) {legs}")
        return verdict

    # ---------------- phase: silicon ----------------
    def _bh_craq_gate(self, row):
        """Keyed silicon gate (adversarial finding sweep_2x2.py:1341).

        The gate trusts only BH CRAQ verdicts whose cc1plus + simulator +
        tt-metal keys match THIS run — legs==PASS alone would let a
        `--phases silicon` resume open the gate with greens earned by an
        older compiler/simulator/tree.  A SKIP_NO_SIMULATOR verdict (no
        legs) never opens it.
        """
        craq_dir = self.ev / row["op"] / "craq"
        if not craq_dir.is_dir():
            return False
        sim = self._staged_sim("bh")
        sim_sha = sha256(sim) if sim and sim.is_file() else ""
        verdicts = [
            json.loads(p.read_text())
            for p in sorted(craq_dir.glob("*-bh/verdict.json"))
        ]
        if not verdicts:
            return False
        for v in verdicts:
            if not (v.get("legs") and all(x == "PASS" for x in v["legs"].values())):
                return False
            if (
                v.get("cc1plus_sha256") != self.info["cc1plus_sha256"]
                or v.get("sim_sha256") != sim_sha
                or v.get("tt_metal_head") != self.info["tt_metal_head"]
            ):
                return False
        return True

    def _load_keyed_classification(self, row, sel):
        """Classification evidence for (row, sel) valid for THIS run's keys,
        or None.  Used when the classify phase was skipped: silicon must
        never run on unkeyed/stale classify evidence (the byte-identical
        refusal logic and the hash-matched device resume both depend on it)."""
        vf = self.ev / row["op"] / "classify" / sel / "verdict.json"
        if not vf.is_file():
            return None
        try:
            v = json.loads(vf.read_text())
        except ValueError:
            return None
        if (
            v.get("cc1plus_sha256") == self.info["cc1plus_sha256"]
            and v.get("tt_metal_head") == self.info["tt_metal_head"]
        ):
            return v
        return None

    def _device_job(
        self, row, sel, label, leg, flags, tag="silicon", expected_texts=None
    ):
        """One serialized device job under both flocks; CSVs copied in-lock."""
        node = row["nodes"][sel]
        work = self.ev / row["op"] / tag / sel / f"{label}-{leg}"
        # The full identity a cached cell must match before reuse: kernel
        # .text alone cannot see test parameters (node id: input ranges,
        # tolerances), flags, or extra_env (adversarial finding
        # sweep_2x2.py:572).
        jobkey = {
            "node": node,
            "flags": flags,
            "extra_env": row["extra_env"] or {},
            "tag": tag,
        }
        # Resume skips only GREEN jobs whose (node, flags, extra_env) jobkey
        # matches AND whose archived .text hash set equals what THIS run's
        # compiler produces for the same node/flags (from the classify
        # evidence).  ABSENT classify hashes (expected_texts=None: --phases
        # silicon without classify, or a leg whose classify stopped before
        # writing hashes) mean the cache cannot be validated: re-run, never
        # reuse (finding sweep_2x2.py:575).  A failed job, or a cell measured
        # from a stale binary, is re-run — never cached as done.
        if (work / "rc.txt").is_file() and not self.a.force:
            prior_rc = int((work / "rc.txt").read_text().strip() or 99)
            if prior_rc == 0 and self._passed(work / "log.txt"):
                cached_key = None
                if (work / "jobkey.json").is_file():
                    try:
                        cached_key = json.loads((work / "jobkey.json").read_text())
                    except (ValueError, OSError):
                        cached_key = None
                archived = (
                    self._texts_of(work / "TEXT_HASHES.txt")
                    if (work / "TEXT_HASHES.txt").is_file()
                    else None
                )
                if expected_texts is None:
                    print(
                        f"resume: {row['op']}/{sel} {label}-{leg} has no "
                        "classify hash reference for this run — cached cell "
                        "not trusted, re-measuring"
                    )
                elif cached_key != jobkey:
                    print(
                        f"resume: {row['op']}/{sel} {label}-{leg} job key "
                        "(node/flags/extra_env) changed or unrecorded — "
                        "re-measuring"
                    )
                elif archived == expected_texts:
                    return prior_rc  # keyed, hash-matched reuse
                else:
                    print(
                        f"resume: {row['op']}/{sel} {label}-{leg} .text hashes "
                        "changed — re-measuring"
                    )
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        rt = work / "rt"
        rt.mkdir()
        (work / "node.txt").write_text(node + "\n")
        (work / "flags.txt").write_text(flags + "\n")
        (work / "jobkey.json").write_text(json.dumps(jobkey, indent=2) + "\n")
        env_prefix = " ".join(f'{k}="{v}"' for k, v in (row["extra_env"] or {}).items())
        inner = work / "inner.sh"
        # Single-quoted node id survives the sh -c layers because pytest node
        # ids never contain single quotes.  Explicit check, not an assert:
        # asserts are compiled out under `python3 -O` (adversarial missed
        # item, sweep_2x2.py:598).
        if "'" in node:
            sys.exit(
                f"pytest node id contains a single quote (breaks the sh -c "
                f"quoting layers): {node}"
            )
        inner.write_text(
            f"""#!/usr/bin/env bash
rm -rf "{LLK}/perf_data"
cd "{PYDIR}" || exit 97
env {env_prefix} CHIP_ARCH=blackhole LLK_HOME="{LLK}" RUNNER_TEMP="{rt}" \\
TT_LLK_EXTRA_COMPILER_OPTIONS="{flags}" \\
timeout 1500 "{self.python}" -m pytest -q -v '{node}' > "{work}/log.txt" 2>&1
RC=$?
echo $RC > "{work}/rc.txt"
# copy raw+post perf CSVs IN-LOCK immediately (they are overwritten per run)
if [ -d "{LLK}/perf_data" ]; then cp -r "{LLK}/perf_data" "{work}/perf_data"; fi
if [ -d "{rt}/tt-llk-build/temp_perf_data" ]; then cp -r "{rt}/tt-llk-build/temp_perf_data" "{work}/raw_perf_data"; fi
exit $RC
"""
        )
        inner.chmod(0o755)
        if self.a.dry_run:
            print(f"DRY-RUN device job: {row['op']}/{sel} {label}-{leg}")
            return 0
        subprocess.run(
            ["flock", "-x", DEVICE_LOCK, "-c", f"flock -x {SILICON_LOCK} -c '{inner}'"],
            check=False,
        )
        rc = (
            int((work / "rc.txt").read_text().strip())
            if (work / "rc.txt").is_file()
            else 99
        )
        # Post-lock archival: ELFs/.text hashes/build.h live in this job's own
        # RUNNER_TEMP, so no other process can overwrite them.
        self._hash_build(rt, work / "TEXT_HASHES.txt")
        self._archive_build(rt, work / "elf")
        shutil.rmtree(rt, ignore_errors=True)
        if rc != 0 or not self._passed(work / "log.txt"):
            self.reds.append(
                f"{row['op']}/{sel} {label}-{leg}: device job FAIL rc={rc}"
            )
        elif (
            expected_texts is not None
            and self._texts_of(work / "TEXT_HASHES.txt") != expected_texts
        ):
            self.reds.append(
                f"{row['op']}/{sel} {label}-{leg}: device job .text differs "
                "from this run's classify build (non-deterministic build?)"
            )
        return rc

    def _perf_value(self, row, sel, label, leg, tag="silicon"):
        """Parse the row's scoped metric from the copied post CSV (lock long
        released).  per_tile rows divide by tile_cnt (cycles/tile); absolute
        rows (e.g. Reduce-SDPA REDUCE_SDPA_BODY) sum the marker's rows."""
        work = self.ev / row["op"] / tag / sel / f"{label}-{leg}"
        col = f"mean({row['metric']})"
        for post in sorted(work.glob("perf_data/*/*.post.csv")):
            total, tiles, seen = 0.0, 1.0, False
            with post.open() as f:
                for rec in csv.DictReader(f):
                    if rec.get("marker") != row["marker"] or col not in rec:
                        continue
                    total += float(rec[col])
                    if not seen:
                        try:
                            tiles = float(rec.get("tile_cnt", 1) or 1)
                        except ValueError:
                            tiles = 1.0
                    seen = True
            if seen:
                return total / (tiles or 1.0) if row["per_tile"] else total
        return None

    def _result_skeleton(self, row, classifications):
        return {
            "op": row["op"],
            "corpus_id": row["corpus_id"],
            "kind": row["kind"],
            "marker": row["marker"],
            "scope": row_scope(row),
            "classify": classifications,
            "cells": {},
            "runs": {},
            "notes": [],
        }

    def _macro_lb_gate(self, row, classifications, result):
        """Structural issue_slot_lb requirement (enforcement layer): a
        macro-launch row without a bound is RED, named in the report with
        the §1 caveat — never a silent no-op.  Uses the measured perf leg's
        classification when present, else the correctness leg's."""
        scan = None
        for sel in ("sem-perf", "sem-corr"):
            v = classifications.get(sel) or {}
            if v.get("macro_scan"):
                scan = v["macro_scan"]
                break
        msg = macro_lb_red(row["op"], row["marker"], row["issue_slot_lb"], scan)
        if msg:
            self.reds.append(msg)
            result["notes"].append(msg)

    def _issue_slot_check(self, row, result):
        """HANDOFF §1 metric caveat as code: a BODY-family reading on a
        macro-launch shape must be >= the payload's issue-slot lower bound
        (issue_slot_lb, cycles/tile), else the marker reading is INVALID and
        the KERNEL marker is required.  The check result is recorded either
        way so every measured cell carries its validity evidence."""
        lb = row["issue_slot_lb"]
        if lb is None:
            return
        cells = result["cells"]
        checked, invalid = [], []
        for cell, val in list(cells.items()):
            if not isinstance(val, (int, float)):
                continue
            if val < lb:
                cells[cell] = "INVALID_MARKER"
                invalid.append(f"{cell}={val:.2f}")
                self.reds.append(
                    f"{row['op']}/{cell}: INVALID_MARKER — {row['marker']} "
                    f"reading {val:.2f} < issue-slot lower bound {lb:g}; "
                    "KERNEL marker required"
                )
            else:
                checked.append(f"{cell}={val:.2f}")
        if invalid:
            result["notes"].append(
                f"issue-slot check FAIL ({', '.join(invalid)} < {lb:g}): "
                f"{row['marker']} is not a valid metric zone for this "
                "macro-launch shape — re-measure with the KERNEL marker"
            )
        elif checked:
            result["notes"].append(
                f"issue-slot check PASS: {', '.join(checked)} all >= payload "
                f"issue-slot lower bound {lb:g} cycles/tile "
                f"({row['marker']} reading valid for this macro-launch shape)"
            )

    def silicon_pinpair(self, row, classifications):
        """kind=pinpair: paired gen-vs-hand A/B at the row's pinned flag set
        (e.g. Reduce-SDPA at the default profitability gate).  Same pipeline
        discipline as the 2x2: correctness first, then 3 fresh processes per
        selector alternating gen/hand, hash-matched resume per job."""
        result = self._result_skeleton(row, classifications)
        self._macro_lb_gate(row, classifications, result)
        if self.a.dry_run:
            result["notes"].append(
                "DRY-RUN: device jobs printed, not executed; no cells expected"
            )
        flags = row["pin_flags"]
        result["notes"].append(f"pinpair leg flags: {flags}")
        for sel in ("sem-corr", "hand-corr"):
            if not row["nodes"][sel]:
                continue
            rc = self._device_job(
                row,
                sel,
                "corr",
                "default",
                flags,
                expected_texts=self._classify_texts(row, sel, "default"),
            )
            result["runs"][f"{sel}/corr-default"] = (
                "PASS" if rc == 0 else f"FAIL(rc={rc})"
            )
            if rc != 0:
                result["notes"].append(f"STOP: {sel} correctness failed; perf withheld")
                return result
        samples = {sel: [] for sel in ("sem-perf", "hand-perf")}
        for r in range(1, PERF_RUNS + 1):
            for sel in ("sem-perf", "hand-perf"):  # alternating gen/hand
                if not row["nodes"][sel]:
                    continue
                self._device_job(
                    row,
                    sel,
                    f"r{r}",
                    "default",
                    flags,
                    expected_texts=self._classify_texts(row, sel, "default"),
                )
                val = self._perf_value(row, sel, f"r{r}", "default")
                if val is not None:
                    samples[sel].append(val)
        for sel, cell in PINPAIR_CELLS.items():
            src = samples[sel]
            result["runs"][f"{sel}/{cell}_samples"] = src
            result["cells"][cell] = (sum(src) / len(src)) if src else None
        self._issue_slot_check(row, result)
        c = result["cells"]
        gen, hand = c.get("generated"), c.get("handwritten_replay")
        if isinstance(gen, (int, float)) and isinstance(hand, (int, float)) and hand:
            result["vs_hand_pct"] = 100.0 * (gen - hand) / hand
        return result

    def silicon(self, row, classifications):
        if row["kind"] == "pinpair":
            return self.silicon_pinpair(row, classifications)
        result = self._result_skeleton(row, classifications)
        self._macro_lb_gate(row, classifications, result)
        if self.a.dry_run:
            # A dry run proves gate wiring, never metrics: mark the row so
            # report() treats its empty cells as blocked-by-design instead
            # of INVALID_METRIC RED.
            result["notes"].append(
                "DRY-RUN: device jobs printed, not executed; no cells expected"
            )
        # correctness first, OFF then ON; byte-identical pair => one run fills both
        for sel in ("sem-corr", "hand-corr"):
            if not row["nodes"][sel]:
                continue
            cls = classifications.get(sel, {})
            legs = ["off"] if cls.get("all") == "IDENTICAL" else ["off", "on"]
            if len(legs) == 1:
                result["notes"].append(
                    f"{sel}: OFF==ON byte-identical — one correctness run fills both legs"
                )
            for leg in legs:
                rc = self._device_job(
                    row,
                    sel,
                    "corr",
                    leg,
                    OFF_FLAGS if leg == "off" else ON_FLAGS,
                    expected_texts=self._classify_texts(row, sel, leg),
                )
                result["runs"][f"{sel}/corr-{leg}"] = (
                    "PASS" if rc == 0 else f"FAIL(rc={rc})"
                )
                if rc != 0:
                    result["notes"].append(
                        f"STOP: {sel} correctness {leg} failed; perf withheld"
                    )
                    return result
        # perf: 3 fresh processes per leg, alternating OFF/ON
        for sel, cells in (
            ("sem-perf", ("sem_off", "sem_on")),
            ("hand-perf", ("hand_off", "hand_on")),
        ):
            if not row["nodes"][sel]:
                continue
            cls = classifications.get(sel, {})
            if cls.get("status") == "COMPILE_FAIL":
                result["cells"][cells[0]] = result["cells"][cells[1]] = None
                result["notes"].append(f"{sel}: COMPILE_FAIL — perf blocked")
                continue
            identical = cls.get("all") == "IDENTICAL"
            if identical and sel == "sem-perf":
                result["notes"].append(
                    "sem-perf OFF/ON byte-identical: recorded refusal, no device run"
                )
                result["cells"]["sem_off"] = result["cells"]["sem_on"] = (
                    "REFUSAL_BYTE_IDENTICAL"
                )
                continue
            legs = ["off"] if identical else ["off", "on"]
            if identical:
                result["notes"].append(
                    f"{sel}: OFF==ON byte-identical — one physical leg fills both cells"
                )
            samples = {leg: [] for leg in legs}
            for r in range(1, PERF_RUNS + 1):
                for leg in legs:  # alternating OFF/ON inside each round
                    self._device_job(
                        row,
                        sel,
                        f"r{r}",
                        leg,
                        OFF_FLAGS if leg == "off" else ON_FLAGS,
                        expected_texts=self._classify_texts(row, sel, leg),
                    )
                    val = self._perf_value(row, sel, f"r{r}", leg)
                    if val is not None:
                        samples[leg].append(val)
            for leg, cell in zip(("off", "on"), cells):
                src = samples[leg] if leg in samples else samples["off"]
                result["runs"][f"{sel}/{cell}_samples"] = src
                result["cells"][cell] = (sum(src) / len(src)) if src else None
        # marker validity first: an INVALID_MARKER cell must not feed a ratio
        self._issue_slot_check(row, result)
        # derived ratios
        c = result["cells"]
        num = lambda x: isinstance(x, (int, float))
        if num(c.get("sem_off")) and num(c.get("sem_on")) and c["sem_off"]:
            result["causal_pct"] = 100.0 * (c["sem_on"] - c["sem_off"]) / c["sem_off"]
        if num(c.get("sem_on")) and num(c.get("hand_on")) and c["hand_on"]:
            result["vs_hand_pct"] = 100.0 * (c["sem_on"] - c["hand_on"]) / c["hand_on"]
        return result

    # ---------------- weekly: per-knob attribution ----------------
    def attribute_knobs(self, row, classifications):
        if row["kind"] == "pinpair":
            return {"op": row["op"], "status": "SKIP_PINPAIR"}
        sel = "sem-perf" if row["nodes"]["sem-perf"] else "sem-corr"
        if (
            not row["nodes"][sel]
            or classifications.get(sel, {}).get("all") != "CHANGED"
        ):
            return {"op": row["op"], "status": "SKIP_NOT_CHANGED"}
        firing = []
        for knob, flag in KNOBS.items():
            verdict = self.classify(
                row,
                sel,
                legs=(("off", OFF_FLAGS), ("knob", f"{OFF_FLAGS} {flag}")),
                tag=f"knobs/{knob}",
            )
            if verdict.get("all") == "CHANGED":
                firing.append(knob)
        out = {
            "op": row["op"],
            "selector": sel,
            "status": "OK",
            "firing_knobs": firing,
            "single_knob_attribution": firing[0] if len(firing) == 1 else None,
        }
        (self.ev / row["op"] / "knob-attribution.json").write_text(
            json.dumps(out, indent=2) + "\n"
        )
        return out

    def knob_silicon(self, row, attribution):
        """Per-knob silicon legs (weekly, headline rows only): OFF vs OFF+knob.

        D3 fix (PULL_ANALYSIS-20260817): these legs run the IDENTICAL
        classify -> paired CRAQ -> correctness-then-perf pipeline as the main
        legs.  Per firing knob: the perf selector's OFF-vs-knob classification
        must be CHANGED (byte-identical => recorded refusal, no device run);
        the correctness selector is classified and paired-CRAQ'd with the same
        OFF-vs-knob legs and the BH gate must be green; device correctness
        runs OFF then knob BEFORE any perf leg; only then 3 fresh perf
        processes per leg, alternating, hash-matched like every device job.
        Callers must invoke this only for rows whose MAIN BH CRAQ gate is
        already green (enforced in run())."""
        sel = attribution.get("selector")
        if attribution.get("status") != "OK" or not sel:
            return
        corr_sel = "sem-corr" if row["nodes"]["sem-corr"] else None
        out = {}
        for knob in attribution.get("firing_knobs", []):
            knob_flags = f"{OFF_FLAGS} {KNOBS[knob]}"
            legs_spec = (("off", OFF_FLAGS), ("knob", knob_flags))
            entry = {"selector": sel, "flags": knob_flags}
            out[knob] = entry
            # 1. classification (perf selector; already produced by
            #    attribute_knobs — classify() resumes hash-matched).
            cls = self.classify(row, sel, legs=legs_spec, tag=f"knobs/{knob}")
            entry["classify"] = cls
            if cls.get("status") != "OK":
                entry["status"] = "CLASSIFY_FAIL"
                continue
            if cls.get("all") == "IDENTICAL":
                entry["status"] = "REFUSAL_BYTE_IDENTICAL"  # no device run
                continue
            if not corr_sel:
                entry["status"] = "WITHHELD_NO_CORR_NODE"
                self.reds.append(
                    f"{row['op']}/{knob}: knob silicon withheld — no correctness node"
                )
                continue
            # 2. correctness-selector classification (for its own byte-identity
            #    handling and the hash-matched device resume below).
            corr_cls = self.classify(
                row, corr_sel, legs=legs_spec, tag=f"knobs/{knob}-corr"
            )
            entry["classify_corr"] = corr_cls
            if corr_cls.get("status") != "OK":
                entry["status"] = "CLASSIFY_FAIL"
                continue
            # 3. paired CRAQ on the correctness node, same OFF-vs-knob legs;
            #    the BH gate must be green (SKIP_NO_SIMULATOR never opens it).
            bh_verdict = None
            for arch in row["craq_archs"].split(","):
                arch = arch.strip()
                v = self.craq(
                    row, corr_sel, arch, legs_spec=legs_spec, tag=f"knobs-craq/{knob}"
                )
                if arch == "bh":
                    bh_verdict = v
            gate = bool(
                bh_verdict
                and bh_verdict.get("legs")
                and all(x == "PASS" for x in bh_verdict["legs"].values())
            )
            entry["craq_bh"] = bh_verdict
            if not gate and not self.a.skip_craq_gate:
                entry["status"] = "WITHHELD_CRAQ_NOT_GREEN"
                self.reds.append(
                    f"{row['op']}/{knob}: knob silicon withheld — paired BH CRAQ not green"
                )
                continue
            # 4. device correctness FIRST (OFF then knob; byte-identical corr
            #    pair => one run fills both legs, like the main pipeline).
            tag = f"knobs-silicon/{knob}"
            corr_legs = (
                [("off", OFF_FLAGS)]
                if corr_cls.get("all") == "IDENTICAL"
                else list(legs_spec)
            )
            corr_fail = False
            for leg, flags in corr_legs:
                rc = self._device_job(
                    row,
                    corr_sel,
                    "corr",
                    leg,
                    flags,
                    tag=tag,
                    expected_texts=self._classify_texts(
                        row, corr_sel, leg, tag=f"knobs/{knob}-corr"
                    ),
                )
                entry[f"corr_{leg}"] = "PASS" if rc == 0 else f"FAIL(rc={rc})"
                if rc != 0:
                    corr_fail = True
                    break
            if corr_fail:
                entry["status"] = "STOP_CORRECTNESS_FAILED"
                self.reds.append(
                    f"{row['op']}/{knob}: knob correctness failed; perf withheld"
                )
                continue
            # 5. perf: 3 fresh processes per leg, alternating OFF/knob.
            samples = {"off": [], "knob": []}
            for r in range(1, PERF_RUNS + 1):
                for leg, flags in legs_spec:
                    self._device_job(
                        row,
                        sel,
                        f"r{r}",
                        leg,
                        flags,
                        tag=tag,
                        expected_texts=self._classify_texts(
                            row, sel, leg, tag=f"knobs/{knob}"
                        ),
                    )
                    val = self._perf_value(row, sel, f"r{r}", leg, tag=tag)
                    if val is not None:
                        samples[leg].append(val)
            cell = {leg: (sum(v) / len(v)) if v else None for leg, v in samples.items()}
            if cell["off"] and cell["knob"]:
                cell["delta_pct"] = 100.0 * (cell["knob"] - cell["off"]) / cell["off"]
            entry["cells"] = cell
            entry["status"] = "OK"
        (self.ev / row["op"] / "knob-silicon.json").write_text(
            json.dumps(out, indent=2) + "\n"
        )

    # ---------------- scoreboard / manifest ----------------
    def emit_scoreboard(self, results, skips):
        payload = {
            "provenance": self.info,
            "results": results,
            "skips": skips,
            "reds": self.reds,
        }
        (self.ev / "scoreboard.json").write_text(json.dumps(payload, indent=2) + "\n")
        cc1 = self.info["cc1plus_sha256"]
        sim_sha = self.info.get("sim_bh_sha256", "")
        with (self.ev / "scoreboard.tsv").open("w") as f:
            f.write(
                "# schema=2; chip-class silicon cells from sweep_2x2.py; "
                "compiler_sha = cc1plus binary sha256 (PRIMARY toolchain pin), "
                "craq_sim_sha = BH libttsim sha256\n"
            )
            f.write(
                "id\tarch\tmetric\tscope\tselector\tcycles\tstatus\t"
                "compiler_sha\tcraq_sim_sha\tprovenance\n"
            )
            for r in results:
                for cell, val in r.get("cells", {}).items():
                    status = (
                        "measured"
                        if isinstance(val, (int, float))
                        else (val or "missing")
                    )
                    cyc = f"{val}" if isinstance(val, (int, float)) else ""
                    f.write(
                        f"{r['corpus_id']}\tbh\tdevice_cycles\t{r['scope']}\t"
                        f"{cell_selector(r, cell)}\t{cyc}\t{status}\t"
                        f"{cc1}\t{sim_sha}\t{self.ev.name}\n"
                    )
        lines = [
            "# 2x2 sweep scoreboard",
            "",
            f"- evidence: `{self.ev}`",
            f"- cc1plus sha256 (primary pin): `{self.info['cc1plus_sha256']}`",
            f"- driver sha256 (secondary): `{self.info['compiler_sha256']}`",
            "",
            "| op | marker | sem OFF | sem ON | causal | hand | vs hand | notes |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else (v or "—")
        for r in results:
            c = r.get("cells", {})
            if r["kind"] == "pinpair":
                # gen-vs-hand pair at the row's pinned flag set: the generated
                # cell rides the "sem ON" column, hand is hand.
                so, sn = "—", fmt(c.get("generated"))
                h = fmt(c.get("handwritten_replay"))
            else:
                so, sn = fmt(c.get("sem_off")), fmt(c.get("sem_on"))
                h = fmt(c.get("hand_on", c.get("hand_off")))
            lines.append(
                "| {op} | {m} | {so} | {sn} | {cz} | {h} | {vh} | {n} |".format(
                    op=r["op"],
                    m=r["marker"],
                    so=so,
                    sn=sn,
                    cz=f"{r['causal_pct']:+.2f}%" if "causal_pct" in r else "—",
                    h=h,
                    vh=f"{r['vs_hand_pct']:+.2f}%" if "vs_hand_pct" in r else "—",
                    n="; ".join(r.get("notes", [])),
                )
            )
        for s in skips:
            lines.append(f"| {s['op']} | — | — | — | — | — | — | {s['reason']} |")
        (self.ev / "SCOREBOARD.md").write_text("\n".join(lines) + "\n")

    def emit_sha256sums(self):
        out = self.ev / "SHA256SUMS"
        entries = []
        for path in sorted(self.ev.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                entries.append(f"{sha256(path)}  {path.relative_to(self.ev)}")
        out.write_text("\n".join(entries) + "\n")

    # ---------------- report ----------------
    @staticmethod
    def _load_baseline(path):
        """Baseline TSV -> (cycles map, expected-class map).

        cycles:  (id, scope, selector) -> [floats]  (min = the established
                 three-process convention when aggregating repeats)
        classes: (id, scope) -> expected class from the schema-2
                 expected_class column (win/parity/loss/refusal); rows with
                 status 'refusal'/'expected_refusal' also declare refusal.
        Falls back to deriving win/parity/loss from the measured sem cells
        for schema-1 baselines without the column.
        """
        cycles, classes = {}, {}
        if not (path and path.is_file()):
            return cycles, classes
        with path.open() as f:
            for rec in csv.DictReader(
                (x for x in f if not x.startswith("#")), delimiter="\t"
            ):
                key2 = (rec["id"], rec["scope"])
                cls = (rec.get("expected_class") or "").strip()
                if cls:
                    classes.setdefault(key2, cls)
                if (rec.get("status") or "").strip() in (
                    "refusal",
                    "expected_refusal",
                    "refusal_byte_identical",
                ):
                    classes[key2] = "refusal"
                try:
                    cyc = float(rec.get("cycles", ""))
                except (TypeError, ValueError):
                    continue
                cycles.setdefault(
                    (rec["id"], rec["scope"], rec["selector"]), []
                ).append(cyc)
        return cycles, classes

    @staticmethod
    def _derived_class(baseline, r):
        """win/parity/loss from the baseline's measured sem cells (schema-1
        fallback when no expected_class column exists)."""
        off = baseline.get((r["corpus_id"], r["scope"], cell_selector(r, "sem_off")))
        on = baseline.get((r["corpus_id"], r["scope"], cell_selector(r, "sem_on")))
        if not (off and on and min(off)):
            return None
        pct = 100.0 * (min(on) - min(off)) / min(off)
        if pct < -0.5:
            return "win"
        if pct <= 0.5:
            return "parity"
        return "loss"

    def report(self, results, skips):
        baseline, base_classes = self._load_baseline(self.a.baseline)
        prev = {}
        if self.a.prev_run and (self.a.prev_run / "scoreboard.json").is_file():
            for r in json.loads((self.a.prev_run / "scoreboard.json").read_text()).get(
                "results", []
            ):
                prev[r["op"]] = r
        lines = [
            "# 2x2 sweep report",
            "",
            f"- run: `{self.ev}`",
            f"- baseline: `{self.a.baseline or 'none'}`",
            f"- previous run: `{self.a.prev_run or 'none'}`",
            "",
            "| op | verdict | detail |",
            "|---|---|---|",
        ]
        rag = "GREEN"
        for r in results:
            verdicts = []
            c = r.get("cells", {})
            scope = r.get("scope") or f"{r['marker']}_MATH_ISOLATE_PER_TILE"
            r = dict(r, scope=scope)
            expected = base_classes.get((r["corpus_id"], scope)) or self._derived_class(
                baseline, r
            )
            # acceptance 1 (class-aware, D4): a refusal is GREEN only when the
            # baseline class is refusal (or the row has no baseline history).
            # A row whose baseline carries a measured WIN that now collapses
            # to a byte-identical refusal is a total-refusal regression: RED.
            if c.get("sem_off") == "REFUSAL_BYTE_IDENTICAL":
                if expected == "win":
                    verdicts.append(
                        "WIN→REFUSAL FLIP (baseline class win, now "
                        "byte-identical refusal — planner stopped firing): RED"
                    )
                    rag = "RED"
                elif expected in ("parity", "loss"):
                    verdicts.append(
                        f"{expected.upper()}→REFUSAL: flagged notice "
                        "(measured baseline row now refuses): YELLOW"
                    )
                    if rag == "GREEN":
                        rag = "YELLOW"
                elif expected == "refusal":
                    verdicts.append(
                        "refusal byte-identical (baseline class refusal): GREEN"
                    )
                else:
                    verdicts.append(
                        "refusal byte-identical (no baseline history): GREEN"
                    )
            elif expected == "refusal" and any(
                isinstance(v, (int, float)) for v in c.values()
            ):
                # refusal -> changed: not a regression by itself, but a class
                # transition that must be surfaced, never silently blessed.
                verdicts.append(
                    "REFUSAL→CHANGED: flagged notice (baseline expects a "
                    "byte-identical refusal, OFF/ON now differs and was "
                    "measured): YELLOW"
                )
                if rag == "GREEN":
                    rag = "YELLOW"
            # acceptance 1b: INVALID_MARKER cells can never gate GREEN.
            if any(v == "INVALID_MARKER" for v in c.values()):
                verdicts.append(
                    "INVALID_MARKER cell(s) — reading below the payload "
                    "issue-slot lower bound (KERNEL marker required): RED"
                )
                rag = "RED"
            # acceptance 1d (enforcement layer): a macro-launch row with an
            # EMPTY issue_slot_lb is RED — the §1 issue-slot check silently
            # no-ops on empty lb, under exactly the fire-and-forget shapes
            # that need it (wave-6 V3).
            if any("EMPTY issue_slot_lb" in n for n in r.get("notes", [])):
                verdicts.append(
                    "MACRO-LAUNCH ROW WITHOUT issue_slot_lb — HANDOFF §1 "
                    "metric caveat unenforceable on this row's cells: RED"
                )
                rag = "RED"
            # acceptance 1c (finding sweep_2x2.py:1276): a cell with baseline
            # history that produced NO parsable metric this run is
            # INVALID_METRIC RED — a profiler/post-CSV or marker rename must
            # never turn the nightly permanently GREEN while measuring
            # nothing.  Withheld/blocked rows already carry their own RED.
            blocked = any(
                "STOP" in n or "COMPILE_FAIL" in n or "withheld" in n or "DRY-RUN" in n
                for n in r.get("notes", [])
            )
            if not blocked:
                dead = sorted(
                    cell
                    for cell, v in c.items()
                    if v is None
                    and baseline.get((r["corpus_id"], scope, cell_selector(r, cell)))
                )
                numeric_any = any(isinstance(v, (int, float)) for v in c.values())
                refused = any(v == "REFUSAL_BYTE_IDENTICAL" for v in c.values())
                if dead:
                    verdicts.append(
                        f"INVALID_METRIC — cell(s) {', '.join(dead)} have "
                        "baseline history but produced no parsable metric "
                        "(marker/post-CSV drift?): RED"
                    )
                    rag = "RED"
                elif expected and c and not numeric_any and not refused:
                    verdicts.append(
                        "INVALID_METRIC — row has baseline class history but "
                        "every cell is unparsable/None: RED"
                    )
                    rag = "RED"
            # acceptance 2a (findings sweep_2x2.py:1222/:1181): per-cell
            # ABSOLUTE cycle drift vs the baseline's min-aggregated cycles.
            # Ratio-only acceptance is blind to uniform slowdowns (both legs
            # +50% keeps every ratio) and never checks the hand leg on
            # refusal rows.  Slowdowns beyond --max-abs-drift-pct are RED;
            # improvements beyond it are YELLOW (stale baseline — reviewed
            # update needed), never silently blessed.
            for cell in sorted(c):
                val = c[cell]
                if not isinstance(val, (int, float)):
                    continue
                base = baseline.get((r["corpus_id"], scope, cell_selector(r, cell)))
                if not base or not min(base):
                    continue
                abs_pct = 100.0 * (val - min(base)) / min(base)
                if abs_pct > self.a.max_abs_drift_pct:
                    verdicts.append(
                        f"{cell} ABS CYCLES {min(base):g}→{val:g} "
                        f"({abs_pct:+.2f}% > {self.a.max_abs_drift_pct:g}%): RED"
                    )
                    rag = "RED"
                elif abs_pct < -self.a.max_abs_drift_pct:
                    verdicts.append(
                        f"{cell} abs cycles improved {min(base):g}→{val:g} "
                        f"({abs_pct:+.2f}%; baseline stale — reviewed update "
                        "needed): YELLOW"
                    )
                    if rag == "GREEN":
                        rag = "YELLOW"
            # acceptance 2: win-sign preservation vs baseline
            for name, key in (("causal", "causal_pct"), ("vs_hand", "vs_hand_pct")):
                if key not in r:
                    continue
                base_pair = None
                if r["kind"] == "pinpair":
                    if name == "causal":
                        continue
                    on = baseline.get((r["corpus_id"], scope, "generated"))
                    hand = baseline.get((r["corpus_id"], scope, "handwritten_replay"))
                    base_pair = (min(hand), min(on)) if on and hand else None
                elif name == "causal":
                    off = baseline.get(
                        (r["corpus_id"], scope, cell_selector(r, "sem_off"))
                    )
                    on = baseline.get(
                        (r["corpus_id"], scope, cell_selector(r, "sem_on"))
                    )
                    base_pair = (min(off), min(on)) if off and on else None
                else:
                    on = baseline.get(
                        (r["corpus_id"], scope, cell_selector(r, "sem_on"))
                    )
                    hand = baseline.get(
                        (r["corpus_id"], scope, cell_selector(r, "hand_on"))
                    )
                    base_pair = (min(hand), min(on)) if on and hand else None
                if not base_pair or not base_pair[0]:
                    verdicts.append(f"{name} {r[key]:+.2f}% (no baseline row)")
                    continue
                base_pct = 100.0 * (base_pair[1] - base_pair[0]) / base_pair[0]
                drift = abs(r[key] - base_pct)
                if base_pct < 0 <= r[key]:  # win-sign flip (causal or vs-hand)
                    verdicts.append(
                        f"{name} WIN→LOSS FLIP {base_pct:+.2f}%→{r[key]:+.2f}%: RED"
                    )
                    rag = "RED"
                elif base_pct <= -0.5 and r[key] > -0.5:
                    # Finding sweep_2x2.py:1259 (fixture C): a real win
                    # (class band <= -0.5%) eroding into the parity band is a
                    # regression, not drift — RED by default; a full flip to
                    # >= 0 is caught above.
                    tag = "YELLOW" if self.a.allow_win_to_parity else "RED"
                    verdicts.append(
                        f"{name} WIN→PARITY {base_pct:+.2f}%→{r[key]:+.2f}%: {tag}"
                    )
                    if tag == "RED":
                        rag = "RED"
                    elif rag == "GREEN":
                        rag = "YELLOW"
                elif (
                    base_pct > 0.5 and (r[key] - base_pct) > self.a.red_loss_growth_pct
                ):
                    # Finding sweep_2x2.py:1259 (fixture D): an existing loss
                    # growing beyond --red-loss-growth-pct percentage points
                    # is RED (exit 1), not an unalertable YELLOW.
                    verdicts.append(
                        f"{name} LOSS GREW {base_pct:+.2f}%→{r[key]:+.2f}% "
                        f"(+{r[key] - base_pct:.2f}pp > "
                        f"{self.a.red_loss_growth_pct:g}pp): RED"
                    )
                    rag = "RED"
                elif drift > self.a.max_drift_pct:
                    verdicts.append(
                        f"{name} drift {base_pct:+.2f}%→{r[key]:+.2f}%: YELLOW"
                    )
                    if rag == "GREEN":
                        rag = "YELLOW"
                else:
                    verdicts.append(
                        f"{name} {r[key]:+.2f}% vs baseline {base_pct:+.2f}%: GREEN"
                    )
            if r["op"] in prev and "causal_pct" in r and "causal_pct" in prev[r["op"]]:
                verdicts.append(
                    f"prev-run causal {prev[r['op']]['causal_pct']:+.2f}%→{r['causal_pct']:+.2f}%"
                )
            if any("STOP" in n or "COMPILE_FAIL" in n for n in r.get("notes", [])):
                verdicts.append("correctness/compile failure: RED")
                rag = "RED"
            # Verdict column carries YELLOW too (adversarial missed item:
            # a YELLOW row displaying 'ok' hid the one channel YELLOW has).
            col = (
                "RED"
                if any("RED" in v for v in verdicts)
                else ("YELLOW" if any("YELLOW" in v for v in verdicts) else "ok")
            )
            lines.append(
                f"| {r['op']} | {col} | "
                f"{'; '.join(verdicts) or 'no silicon cells this run'} |"
            )
        for s in skips:
            lines.append(f"| {s['op']} | SKIP | {s['reason']} |")
        if self.reds:
            rag = "RED"
            lines += ["", "## RED events", ""] + [f"- {x}" for x in self.reds]
        lines += ["", f"## Overall: {rag}"]
        (self.ev / "REPORT.md").write_text("\n".join(lines) + "\n")
        print(f"REPORT: {rag} -> {self.ev / 'REPORT.md'}")
        return rag

    # ---------------- main flow ----------------
    def run(self):
        phases = self.a.phases
        self.preflight()
        results, skips = [], []
        for row in self.rows:
            if row["kind"] == "skip":
                skips.append(
                    {
                        "op": row["op"],
                        "corpus_id": row["corpus_id"],
                        "status": "SKIP_ABSENT_NODE",
                        "reason": row["note"],
                    }
                )
                continue
            # pinpair rows classify/CRAQ a single pinned-flag leg per selector.
            pin_legs = (
                (("default", row["pin_flags"]),) if row["kind"] == "pinpair" else None
            )
            classifications = {}
            if "classify" in phases:
                self.verify_toolchain("classify")
                for sel in SELECTORS:
                    if row["nodes"][sel]:
                        classifications[sel] = (
                            self.classify(row, sel, legs=pin_legs)
                            if pin_legs
                            else self.classify(row, sel)
                        )
            if "craq" in phases:
                self.verify_toolchain("craq")
                for arch in row["craq_archs"].split(","):
                    for sel in ("sem-corr", "hand-corr"):
                        if row["nodes"][sel]:
                            if pin_legs:
                                self.craq(row, sel, arch.strip(), legs_spec=pin_legs)
                            else:
                                self.craq(row, sel, arch.strip())
            attribution = None
            if self.a.knob_attribution and "classify" in phases:
                attribution = self.attribute_knobs(row, classifications)
            if "silicon" in phases:
                if not self.a.allow_hardware:
                    skips.append(
                        {
                            "op": row["op"],
                            "corpus_id": row["corpus_id"],
                            "status": "SKIP_HARDWARE_NOT_AUTHORIZED",
                            "reason": "silicon phase requires --allow-hardware",
                        }
                    )
                    continue
                self.verify_toolchain("silicon")
                # Silicon runs only on classify evidence KEYED to this run
                # (finding sweep_2x2.py:1341: with classify skipped,
                # classifications={} disabled the byte-identical refusal
                # logic and every hash-match).  A resumed evidence root
                # supplies verdicts only if their cc1plus/tt-metal keys
                # match; otherwise the row is withheld RED.
                missing_cls = []
                for sel in SELECTORS:
                    if row["nodes"][sel] and sel not in classifications:
                        keyed = self._load_keyed_classification(row, sel)
                        if keyed is None:
                            missing_cls.append(sel)
                        else:
                            classifications[sel] = keyed
                if missing_cls:
                    self.reds.append(
                        f"{row['op']}: silicon withheld — no classify evidence "
                        f"keyed to this toolchain/tree for "
                        f"{','.join(missing_cls)} (run the classify phase)"
                    )
                    results.append(
                        dict(
                            self._result_skeleton(row, classifications),
                            notes=[
                                "silicon withheld: classify evidence missing or "
                                "keyed to another toolchain/tree"
                            ],
                        )
                    )
                    continue
                # Keyed BH CRAQ gate: stale-toolchain greens never open it.
                gate = self._bh_craq_gate(row)
                if not gate and not self.a.skip_craq_gate:
                    self.reds.append(
                        f"{row['op']}: silicon withheld — paired BH CRAQ not green"
                    )
                    if attribution and row["op"] in (self.a.knob_silicon_rows or []):
                        self.reds.append(
                            f"{row['op']}: knob silicon withheld — main BH CRAQ gate not green"
                        )
                    results.append(
                        dict(
                            self._result_skeleton(row, classifications),
                            notes=["silicon withheld: BH CRAQ gate not green"],
                        )
                    )
                    continue
                results.append(self.silicon(row, classifications))
                # Weekly per-knob silicon legs run BEHIND the main BH CRAQ
                # gate (D3) and add their own per-knob classify/CRAQ/
                # correctness pipeline inside knob_silicon().
                if attribution and row["op"] in (self.a.knob_silicon_rows or []):
                    self.knob_silicon(row, attribution)
            else:
                results.append(self._result_skeleton(row, classifications))
        self.emit_scoreboard(results, skips)
        rag = "GREEN"
        if "report" in phases:
            rag = self.report(results, skips)
        self.emit_sha256sums()
        if self.reds:
            print("RED events:\n  " + "\n  ".join(self.reds))
        return 1 if (self.reds or rag == "RED") else 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--evidence-root", type=pathlib.Path, required=True)
    ap.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--ops",
        type=lambda s: s.split(","),
        default=None,
        help="comma list of op rows (default: all config rows)",
    )
    ap.add_argument(
        "--phases",
        type=lambda s: s.split(","),
        default=["classify", "craq", "silicon", "report"],
        help="subset of classify,craq,silicon,report",
    )
    ap.add_argument(
        "--sim-bh", type=pathlib.Path, help="BH libttsim.so (generic-path CRAQ oracle)"
    )
    ap.add_argument("--sim-wh", type=pathlib.Path, help="WH libttsim.so")
    ap.add_argument(
        "--sim-bh-sha",
        help="required FULL sha256 of the BH libttsim.so — the reviewed CRAQ "
        "oracle pin (sweep_2x2.conf PINNED_SIM_BH_SHA256); verified at "
        "preflight and every phase entry; prefixes are rejected",
    )
    ap.add_argument(
        "--sim-wh-sha",
        help="required FULL sha256 of the WH libttsim.so — the reviewed CRAQ "
        "oracle pin (sweep_2x2.conf PINNED_SIM_WH_SHA256); verified at "
        "preflight and every phase entry; prefixes are rejected",
    )
    ap.add_argument("--venv", type=pathlib.Path, help="tt-llk virtualenv root")
    ap.add_argument(
        "--compiler",
        type=pathlib.Path,
        help="riscv-tt-elf-g++ (default: tests/sfpi symlink)",
    )
    ap.add_argument(
        "--compiler-sha",
        help="required FULL sha256 of the g++ DRIVER — secondary pin only "
        "(historically byte-identical across cc1plus-only changes); "
        "prefixes are rejected",
    )
    ap.add_argument(
        "--cc1plus-sha",
        help="required FULL sha256 of cc1plus, resolved via "
        "g++ -print-prog-name=cc1plus — the PRIMARY toolchain pin; "
        "prefixes are rejected",
    )
    ap.add_argument(
        "--baseline",
        type=pathlib.Path,
        help="chip-class device baseline TSV for --phases report",
    )
    ap.add_argument(
        "--prev-run",
        type=pathlib.Path,
        help="previous evidence root for drift comparison",
    )
    ap.add_argument("--max-drift-pct", type=float, default=5.0)
    ap.add_argument(
        "--max-abs-drift-pct",
        type=float,
        default=10.0,
        help="per-cell ABSOLUTE cycle drift vs baseline: slowdowns beyond "
        "this are RED (uniform slowdowns preserve every ratio); "
        "improvements beyond it are YELLOW (stale baseline)",
    )
    ap.add_argument(
        "--red-loss-growth-pct",
        type=float,
        default=5.0,
        help="a baseline loss growing by more than this many percentage "
        "points is RED (exit 1), not YELLOW",
    )
    ap.add_argument(
        "--allow-win-to-parity",
        action="store_true",
        help="downgrade WIN→PARITY erosion from RED (default) to YELLOW",
    )
    ap.add_argument(
        "--allow-hardware",
        action="store_true",
        help="authorize serialized device jobs (both flocks)",
    )
    ap.add_argument(
        "--knob-attribution",
        action="store_true",
        help="weekly: classify each changed row against each single optimization knob",
    )
    ap.add_argument(
        "--knob-silicon-rows",
        type=lambda s: s.split(","),
        default=None,
        help="weekly: comma list of headline rows that also get per-knob silicon legs",
    )
    ap.add_argument(
        "--skip-craq-gate",
        action="store_true",
        help="documented override: run silicon without a green CRAQ gate (control experiments only)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-run steps whose evidence already exists",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print device jobs instead of running them",
    )
    args = ap.parse_args()
    return Sweep(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
