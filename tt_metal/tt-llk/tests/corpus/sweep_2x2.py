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
    "-mno-tt-tensix-optimize-dst-autoincr"
)
ON_FLAGS = (
    "-mtt-tensix-optimize-latency-schedule "
    "-mtt-tensix-optimize-dst-iteration-fusion "
    "-mtt-tensix-optimize-replay-hoist "
    "-mtt-tensix-optimize-invariant-loadi "
    "-mtt-tensix-optimize-dst-autoincr "
    "-mtt-tensix-macro-planner"
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
}
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
        # SECONDARY pin: the g++ driver.  The driver binary is byte-identical
        # across cc1plus-only changes (structurally blind, D6) — it can catch
        # a wrong toolchain layout but never a compiler-proper change.
        info["compiler_sha256"] = sha256(self.compiler)
        if self.a.compiler_sha and not info["compiler_sha256"].startswith(
            self.a.compiler_sha
        ):
            sys.exit(
                f"DRIVER SHA MISMATCH: pinned {self.a.compiler_sha}, "
                f"found {info['compiler_sha256']} — refusing to sweep"
            )
        # PRIMARY pin: cc1plus (the compiler proper), resolved through the
        # driver itself so the pin follows whatever binary actually compiles.
        cc1 = subprocess.run(
            [str(self.compiler), "-print-prog-name=cc1plus"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not cc1 or not pathlib.Path(cc1).is_file():
            sys.exit(f"cannot resolve cc1plus via {self.compiler} (got '{cc1}')")
        info["cc1plus"] = cc1
        info["cc1plus_sha256"] = sha256(pathlib.Path(cc1))
        if self.a.cc1plus_sha and not info["cc1plus_sha256"].startswith(
            self.a.cc1plus_sha
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
        info["tt_metal_head"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        for arch in ("bh", "wh"):
            sim = getattr(self.a, f"sim_{arch}")
            info[f"sim_{arch}"] = str(sim) if sim else ""
            info[f"sim_{arch}_sha256"] = sha256(sim) if sim and sim.is_file() else ""
        (self.ev / "preflight.json").write_text(json.dumps(info, indent=2) + "\n")
        man = [
            f"Lane sweep-2x2 evidence — {self.ev.name}",
            f"compiler driver: {self.compiler}",
            f"compiler driver sha256 (secondary pin): {info['compiler_sha256']}",
            f"cc1plus: {info['cc1plus']}",
            f"cc1plus sha256 (PRIMARY pin): {info['cc1plus_sha256']}",
            f"compiler version: {info['compiler_version']}",
            f"tt-metal: {info['tt_metal_head']}",
            f"libttsim bh sha256: {info['sim_bh_sha256']}",
            f"libttsim wh sha256: {info['sim_wh_sha256']}",
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
            # the compiler that produced it.  Verdicts from another cc1plus
            # (or from the pre-keying schema) are recompiled.
            if verdict.get("cc1plus_sha256") == self.info["cc1plus_sha256"]:
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
        verdict["cc1plus_sha256"] = self.info["cc1plus_sha256"]
        verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
        return verdict

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
            # Hash-matched resume: verdicts are keyed to cc1plus + simulator.
            if (
                verdict.get("cc1plus_sha256") == self.info["cc1plus_sha256"]
                and verdict.get("sim_sha256") == sim_sha
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
        }
        verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
        if arch == "bh" and any(v != "PASS" for v in legs.values()):
            self.reds.append(f"{row['op']}/{sel}: CRAQ {arch} ({tag}) {legs}")
        return verdict

    # ---------------- phase: silicon ----------------
    def _device_job(
        self, row, sel, label, leg, flags, tag="silicon", expected_texts=None
    ):
        """One serialized device job under both flocks; CSVs copied in-lock."""
        node = row["nodes"][sel]
        work = self.ev / row["op"] / tag / sel / f"{label}-{leg}"
        # Resume skips only GREEN jobs whose archived .text hash set matches
        # what THIS run's compiler produces for the same node/flags (from the
        # classify evidence).  A failed job, or a cell measured from a stale
        # binary (different compiler), is re-run — never cached as done.
        if (work / "rc.txt").is_file() and not self.a.force:
            prior_rc = int((work / "rc.txt").read_text().strip() or 99)
            if prior_rc == 0 and self._passed(work / "log.txt"):
                if expected_texts is None:
                    return prior_rc
                archived = (
                    self._texts_of(work / "TEXT_HASHES.txt")
                    if (work / "TEXT_HASHES.txt").is_file()
                    else None
                )
                if archived == expected_texts:
                    return prior_rc  # hash-matched reuse
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
        env_prefix = " ".join(f'{k}="{v}"' for k, v in (row["extra_env"] or {}).items())
        inner = work / "inner.sh"
        # Single-quoted node id survives the sh -c layers because pytest node
        # ids never contain single quotes.
        assert "'" not in node
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
            lines.append(
                f"| {r['op']} | {'RED' if any('RED' in v for v in verdicts) else 'ok'} | "
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
                for sel in SELECTORS:
                    if row["nodes"][sel]:
                        classifications[sel] = (
                            self.classify(row, sel, legs=pin_legs)
                            if pin_legs
                            else self.classify(row, sel)
                        )
            if "craq" in phases:
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
                bh_craq = (
                    [
                        json.loads(p.read_text())
                        for p in (self.ev / row["op"] / "craq").glob(
                            "*-bh/verdict.json"
                        )
                    ]
                    if (self.ev / row["op"] / "craq").is_dir()
                    else []
                )
                # Gate requires at least one BH verdict WITH legs, and every
                # leg PASS; a SKIP_NO_SIMULATOR verdict never opens the gate.
                gate = bool(bh_craq) and all(
                    c.get("legs") and all(v == "PASS" for v in c["legs"].values())
                    for c in bh_craq
                )
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
    ap.add_argument("--venv", type=pathlib.Path, help="tt-llk virtualenv root")
    ap.add_argument(
        "--compiler",
        type=pathlib.Path,
        help="riscv-tt-elf-g++ (default: tests/sfpi symlink)",
    )
    ap.add_argument(
        "--compiler-sha",
        help="required sha256 (prefix ok) of the g++ DRIVER — secondary pin "
        "only (the driver is byte-identical across cc1plus-only changes)",
    )
    ap.add_argument(
        "--cc1plus-sha",
        help="required sha256 (prefix ok) of cc1plus, resolved via "
        "g++ -print-prog-name=cc1plus — the PRIMARY toolchain pin",
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
