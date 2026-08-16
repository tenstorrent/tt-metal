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
rows are machine-readable SKIPs.  Metric: post CSV mean(MATH_ISOLATE) at the
row's marker (KERNEL for fire-and-forget replay-launch shapes, TILE_LOOP for
eltwise suites) divided by tile_cnt = cycles/tile.

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
            sel: row.get(sel.replace("-", "_"), "").strip() for sel in SELECTORS
        }
    return rows


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
        info["compiler_sha256"] = sha256(self.compiler)
        if self.a.compiler_sha and not info["compiler_sha256"].startswith(
            self.a.compiler_sha
        ):
            sys.exit(
                f"COMPILER SHA MISMATCH: pinned {self.a.compiler_sha}, "
                f"found {info['compiler_sha256']} — refusing to sweep"
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
            f"compiler: {self.compiler}",
            f"compiler sha256: {info['compiler_sha256']}",
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
    def _env(self, arch, runner_temp, flags, sim=None):
        env = os.environ.copy()
        env.update(
            CHIP_ARCH=CHIP[arch],
            LLK_HOME=str(LLK),
            RUNNER_TEMP=str(runner_temp),
            TT_LLK_EXTRA_COMPILER_OPTIONS=flags,
        )
        if sim:
            env["TT_METAL_SIMULATOR"] = str(sim)
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
        text = pathlib.Path(log).read_text(errors="replace")
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
            return json.loads(verdict_file.read_text())
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
                self._env("bh", rt, flags),
                work / f"compile-{leg}.log",
            )
            if rc != 0 or not self._passed(work / f"compile-{leg}.log"):
                verdict = {"selector": sel, "status": "COMPILE_FAIL", "leg": leg}
                verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
                self.reds.append(f"{row['op']}/{sel}: compile {leg} failed")
                return verdict
            hashes[leg] = self._hash_build(rt, work / f"hashes-{leg}.txt")
            self._archive_build(rt, work / f"elf-{leg}")
            shutil.rmtree(rt, ignore_errors=True)
        legnames = [leg for leg, _ in legs]
        a_set = sorted(h[1] for h in hashes[legnames[0]])
        b_set = sorted(h[1] for h in hashes[legnames[1]])
        math_a = sorted(h[1] for h in hashes[legnames[0]] if h[0].endswith("math.elf"))
        math_b = sorted(h[1] for h in hashes[legnames[1]] if h[0].endswith("math.elf"))
        verdict = {
            "selector": sel,
            "status": "OK",
            "all": "IDENTICAL" if a_set == b_set else "CHANGED",
            "math": "IDENTICAL" if math_a == math_b else "CHANGED",
        }
        verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
        return verdict

    # ---------------- phase: craq ----------------
    def craq(self, row, sel, arch):
        node = row["nodes"][sel]
        sim = getattr(self.a, f"sim_{arch}")
        work = self.ev / row["op"] / "craq" / f"{sel}-{arch}"
        verdict_file = work / "verdict.json"
        if verdict_file.is_file() and not self.a.force:
            return json.loads(verdict_file.read_text())
        if not sim or not sim.is_file():
            verdict = {"selector": sel, "arch": arch, "status": "SKIP_NO_SIMULATOR"}
            work.mkdir(parents=True, exist_ok=True)
            verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
            return verdict
        work.mkdir(parents=True, exist_ok=True)
        (work / "node.txt").write_text(node + "\n")
        legs = {}
        for leg, flags in (("off", OFF_FLAGS), ("on", ON_FLAGS)):
            rt = work / f"rt-{leg}"
            shutil.rmtree(rt, ignore_errors=True)
            rt.mkdir(parents=True)
            log = work / f"craq-{leg}.log"
            rc = self._pytest(
                node,
                ["--run-simulator"],
                self._env(arch, rt, flags, sim=sim),
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
        verdict = {"selector": sel, "arch": arch, "status": "OK", "legs": legs}
        verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
        if arch == "bh" and any(v != "PASS" for v in legs.values()):
            self.reds.append(f"{row['op']}/{sel}: CRAQ {arch} {legs}")
        return verdict

    # ---------------- phase: silicon ----------------
    def _device_job(self, row, sel, label, leg, flags, tag="silicon"):
        """One serialized device job under both flocks; CSVs copied in-lock."""
        node = row["nodes"][sel]
        work = self.ev / row["op"] / tag / sel / f"{label}-{leg}"
        if (work / "rc.txt").is_file() and not self.a.force:
            return int((work / "rc.txt").read_text().strip() or 99)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        rt = work / "rt"
        rt.mkdir()
        (work / "node.txt").write_text(node + "\n")
        (work / "flags.txt").write_text(flags + "\n")
        inner = work / "inner.sh"
        # Single-quoted node id survives the sh -c layers because pytest node
        # ids never contain single quotes.
        assert "'" not in node
        inner.write_text(
            f"""#!/usr/bin/env bash
rm -rf "{LLK}/perf_data"
cd "{PYDIR}" || exit 97
env CHIP_ARCH=blackhole LLK_HOME="{LLK}" RUNNER_TEMP="{rt}" \\
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
        return rc

    def _perf_value(self, row, sel, label, leg, tag="silicon"):
        """Parse cycles/tile from the copied post CSV (lock long released)."""
        work = self.ev / row["op"] / tag / sel / f"{label}-{leg}"
        for post in sorted(work.glob("perf_data/*/*.post.csv")):
            with post.open() as f:
                for rec in csv.DictReader(f):
                    if rec.get("marker") != row["marker"]:
                        continue
                    try:
                        tiles = float(rec.get("tile_cnt", 1) or 1)
                    except ValueError:
                        tiles = 1.0
                    return float(rec["mean(MATH_ISOLATE)"]) / (tiles or 1.0)
        return None

    def silicon(self, row, classifications):
        op = row["op"]
        result = {
            "op": op,
            "corpus_id": row["corpus_id"],
            "kind": row["kind"],
            "marker": row["marker"],
            "classify": classifications,
            "cells": {},
            "runs": {},
            "notes": [],
        }
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
                    row, sel, "corr", leg, OFF_FLAGS if leg == "off" else ON_FLAGS
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
                        row, sel, f"r{r}", leg, OFF_FLAGS if leg == "off" else ON_FLAGS
                    )
                    val = self._perf_value(row, sel, f"r{r}", leg)
                    if val is not None:
                        samples[leg].append(val)
            for leg, cell in zip(("off", "on"), cells):
                src = samples[leg] if leg in samples else samples["off"]
                result["runs"][f"{sel}/{cell}_samples"] = src
                result["cells"][cell] = (sum(src) / len(src)) if src else None
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
        """Per-knob silicon legs (weekly, headline rows only): OFF vs OFF+knob."""
        sel = attribution.get("selector")
        if attribution.get("status") != "OK" or not sel:
            return
        out = {}
        for knob in attribution.get("firing_knobs", []):
            knob_flags = f"{OFF_FLAGS} {KNOBS[knob]}"
            samples = {"off": [], "knob": []}
            for r in range(1, PERF_RUNS + 1):
                for leg, flags in (("off", OFF_FLAGS), ("knob", knob_flags)):
                    tag = f"knobs-silicon/{knob}"
                    self._device_job(row, sel, f"r{r}", leg, flags, tag=tag)
                    val = self._perf_value(row, sel, f"r{r}", leg, tag=tag)
                    if val is not None:
                        samples[leg].append(val)
            cell = {leg: (sum(v) / len(v)) if v else None for leg, v in samples.items()}
            if cell["off"] and cell["knob"]:
                cell["delta_pct"] = 100.0 * (cell["knob"] - cell["off"]) / cell["off"]
            out[knob] = cell
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
        scope = lambda r: f"{r['marker']}_MATH_ISOLATE_PER_TILE"
        with (self.ev / "scoreboard.tsv").open("w") as f:
            f.write("# schema=1; chip-class silicon cells from sweep_2x2.py\n")
            f.write("id\tarch\tmetric\tscope\tselector\tcycles\tstatus\tprovenance\n")
            for r in results:
                for cell, val in r.get("cells", {}).items():
                    status = (
                        "measured"
                        if isinstance(val, (int, float))
                        else (val or "missing")
                    )
                    cyc = f"{val}" if isinstance(val, (int, float)) else ""
                    f.write(
                        f"{r['corpus_id']}\tbh\tdevice_cycles\t{scope(r)}\t"
                        f"{r['op']}:{cell}\t{cyc}\t{status}\t{self.ev.name}\n"
                    )
        lines = [
            "# 2x2 sweep scoreboard",
            "",
            f"- evidence: `{self.ev}`",
            f"- compiler sha256: `{self.info['compiler_sha256']}`",
            "",
            "| op | marker | sem OFF | sem ON | causal | hand | vs hand | notes |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else (v or "—")
        for r in results:
            c = r.get("cells", {})
            lines.append(
                "| {op} | {m} | {so} | {sn} | {cz} | {h} | {vh} | {n} |".format(
                    op=r["op"],
                    m=r["marker"],
                    so=fmt(c.get("sem_off")),
                    sn=fmt(c.get("sem_on")),
                    cz=f"{r['causal_pct']:+.2f}%" if "causal_pct" in r else "—",
                    h=fmt(c.get("hand_on", c.get("hand_off"))),
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
    def report(self, results, skips):
        baseline = {}
        if self.a.baseline and self.a.baseline.is_file():
            with self.a.baseline.open() as f:
                for rec in csv.DictReader(
                    (x for x in f if not x.startswith("#")), delimiter="\t"
                ):
                    try:
                        cyc = float(rec.get("cycles", ""))
                    except (TypeError, ValueError):
                        continue
                    baseline.setdefault(
                        (rec["id"], rec["scope"], rec["selector"]), []
                    ).append(cyc)
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
            scope = f"{r['marker']}_MATH_ISOLATE_PER_TILE"
            num = lambda x: isinstance(x, (int, float))
            # acceptance 1: refusal rows must stay byte-identical refusals
            if c.get("sem_off") == "REFUSAL_BYTE_IDENTICAL":
                verdicts.append("refusal byte-identical: GREEN")
            # acceptance 2: win-sign preservation vs baseline
            for name, key in (("causal", "causal_pct"), ("vs_hand", "vs_hand_pct")):
                if key not in r:
                    continue
                base_pair = None
                if name == "causal":
                    off = baseline.get((r["corpus_id"], scope, f"{r['op']}:sem_off"))
                    on = baseline.get((r["corpus_id"], scope, f"{r['op']}:sem_on"))
                    base_pair = (min(off), min(on)) if off and on else None
                else:
                    on = baseline.get((r["corpus_id"], scope, f"{r['op']}:sem_on"))
                    hand = baseline.get((r["corpus_id"], scope, f"{r['op']}:hand_on"))
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
            classifications = {}
            if "classify" in phases:
                for sel in SELECTORS:
                    if row["nodes"][sel]:
                        classifications[sel] = self.classify(row, sel)
            if "craq" in phases:
                for arch in row["craq_archs"].split(","):
                    for sel in ("sem-corr", "hand-corr"):
                        if row["nodes"][sel]:
                            self.craq(row, sel, arch.strip())
            if self.a.knob_attribution and "classify" in phases:
                attribution = self.attribute_knobs(row, classifications)
                if (
                    row["op"] in (self.a.knob_silicon_rows or [])
                    and "silicon" in phases
                    and self.a.allow_hardware
                ):
                    self.knob_silicon(row, attribution)
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
                    results.append(
                        {
                            "op": row["op"],
                            "corpus_id": row["corpus_id"],
                            "kind": row["kind"],
                            "marker": row["marker"],
                            "classify": classifications,
                            "cells": {},
                            "notes": ["silicon withheld: BH CRAQ gate not green"],
                        }
                    )
                    continue
                results.append(self.silicon(row, classifications))
            else:
                results.append(
                    {
                        "op": row["op"],
                        "corpus_id": row["corpus_id"],
                        "kind": row["kind"],
                        "marker": row["marker"],
                        "classify": classifications,
                        "cells": {},
                        "notes": [],
                    }
                )
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
        "--compiler-sha", help="required sha256 (prefix ok) of the pinned compiler"
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
