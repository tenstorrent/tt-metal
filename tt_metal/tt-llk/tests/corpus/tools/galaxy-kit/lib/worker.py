#!/usr/bin/env python3
"""galaxy-kit per-chip benchmark worker — work-stealing, execute-only.

Runs on a galaxy compute node.  ELFs were compiled on quietbox (stage.sh)
and shipped in builds/<group>/; every pytest session runs with
--compile-consumer and needs no riscv toolchain (only the shipped
riscv-tt-elf-size/objdump for the TEXT_SIZE metric).

Honesty rules baked in:
  - CORR GATE: an arm's perf reps run only after that arm's correctness
    node PASSED on this same chip (<arm>-CORR-FAIL.txt otherwise).
  - SAME-CHIP PAIRS: both arms of an (op, leg) item run back-to-back on
    the claiming worker's chip; the chip id is recorded per cell.
  - REPS: LK_REPS solo perf sessions per arm (default 5); reps are
    expected cycle-identical (the collector reports spreads).
  - NO DEVICE RESETS ever in a worker (EXABOX.md §7 wall 2): recovery is
    one retry after a kill-tree session timeout, nothing else.

Work-stealing: queue/<op>__<leg>__c<k> items, claimed by atomic mkdir in
claims/.  A worker skips copies of an (op,leg) it already ran, so K copies
land on K distinct chips.  Resume-safe: session dirs with rc.txt are kept.

OPT-IN batched mode (LK_BATCH=1, default OFF — the default path above is
byte-unchanged): amortizes pytest startup by claiming up to LK_BATCH_OPS
queue items at once and putting MANY DISTINCT nodes in ONE pytest session
(one corr session for the whole batch, then one session PER REP INDEX over
all gated perf nodes — pytest dedups identical node ids, so a node's five
reps stay in five sessions, exactly like the solo path).  Per-node perf
attribution comes from lk_batch_plugin.py, which replays the harness's own
dump/post/combine code on each node's frames as it finishes; any cell the
batch cannot PROVE (failed node, missing demux, boundary interference)
falls back to a solo default session with the usual one-retry rule.  All
honesty rules above still hold: same-chip pairs, corr gate before perf,
LK_REPS reps, no resets.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get("LK_BASE", "/data/nkapre/craq-laneLK"))
NREPS = int(os.environ.get("LK_REPS", "5"))
SESSION_TIMEOUT = int(os.environ.get("LK_SESSION_TIMEOUT", "900"))
BATCH = os.environ.get("LK_BATCH", "") == "1"
BATCH_OPS = int(os.environ.get("LK_BATCH_OPS", "8"))
BATCH_NODE_SECS = int(os.environ.get("LK_BATCH_NODE_SECS", "60"))
BATCH_AUDIT = os.environ.get("LK_BATCH_AUDIT", "1") not in ("0", "")
# solo sessions themselves jitter by a cycle or two (observed 0-2 on the
# quietbox p150 and across laneLK's galaxy chips), so the audit flags only
# beyond this tolerance; the raw values are always recorded either way.
BATCH_AUDIT_TOL = float(os.environ.get("LK_BATCH_AUDIT_TOL", "2"))


def read_arms():
    arms = {}
    with (BASE / "ARMS.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            arms.setdefault((r["op"], r["leg"]), []).append(r)
    return arms


def group_of(arm):
    g = arm["flagskey"]
    if arm["env"]:
        g += "+" + re.sub(r"[^A-Za-z0-9]+", "_", arm["env"])[:30]
    return g


class Worker:
    def __init__(self, chip, device=None):
        self.chip = chip
        self.device = str(chip) if device is None else device
        self.tag = f"chip{chip:02d}"
        self.wroot = BASE / "workers" / self.tag
        self.farm = self.wroot / "farm/tt_metal/tt-llk"
        self.python = str(BASE / "venv/bin/python")
        self.rt_cache = {}
        self.done_ops = set()
        self.wroot.mkdir(parents=True, exist_ok=True)
        self._setup_farm()
        d = self.wroot / "done-ops.txt"
        if d.is_file():
            for line in d.read_text().splitlines():
                if line.strip():
                    self.done_ops.add(tuple(line.split("\t")[:2]))

    def _setup_farm(self):
        # private farm copy: perf_data isolation between 32 workers
        if (self.farm / "tests/python_tests/conftest.py").is_file():
            return
        self.farm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(BASE / "farm/tt_metal/tt-llk", self.farm, symlinks=True)

    def _runner_temp(self, group):
        # private hardlink clone of the shipped build: consumer sessions only
        # ADD files (temp_perf_data, order_records) — never edit in place
        if group in self.rt_cache:
            return self.rt_cache[group]
        rt = self.wroot / "rt" / group
        if not (rt / "tt-llk-build").is_dir():
            rt.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "cp",
                    "-al",
                    str(BASE / "builds" / group / "tt-llk-build"),
                    str(rt / "tt-llk-build"),
                ],
                check=True,
            )
        self.rt_cache[group] = rt
        return rt

    def _session(self, sdir, node, flags, env_extra, timeout=SESSION_TIMEOUT):
        rcf = sdir / "rc.txt"
        if rcf.is_file():
            return int(rcf.read_text().strip() or 99)
        sdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for k in ("TT_METAL_SIMULATOR", "TT_UMD_SIMULATOR_PATH"):
            env.pop(k, None)
        group = env_extra.pop("__group")
        rt = self._runner_temp(group)
        env.update(
            CHIP_ARCH="blackhole",
            SHORT_ARCH="bh",
            LLK_HOME=str(self.farm),
            RUNNER_TEMP=str(rt),
            TT_LLK_EXTRA_COMPILER_OPTIONS=flags,
            TT_VISIBLE_DEVICES=self.device,
            PYTHONUNBUFFERED="1",
        )
        env.update(env_extra)
        shutil.rmtree(self.farm / "perf_data", ignore_errors=True)
        shutil.rmtree(rt / "tt-llk-build/temp_perf_data", ignore_errors=True)
        (sdir / "node.txt").write_text(node + "\n")
        (sdir / "flags.txt").write_text(flags + "\n")
        cmd = [
            self.python,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "--compile-consumer",
            node,
        ]
        t0 = time.time()
        try:
            r = subprocess.run(
                cmd,
                cwd=self.farm / "tests/python_tests",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            out, rc = r.stdout, r.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"") + b"\nLK-SESSION-TIMEOUT\n"
            rc = 98
        dt = time.time() - t0
        (sdir / "log.txt").write_bytes(out)
        if b" passed" not in out and rc == 0:
            rc = 97  # green rc with no pass line = silently-empty session
        pd = self.farm / "perf_data"
        if pd.is_dir():
            shutil.copytree(pd, sdir / "perf_data", dirs_exist_ok=True)
            shutil.rmtree(pd, ignore_errors=True)
        (sdir / "secs.txt").write_text(f"{dt:.1f}\n")
        rcf.write_text(f"{rc}\n")
        return rc

    def run_item(self, op, leg, arms):
        res = BASE / "results" / f"{op}__{leg}" / self.tag
        res.mkdir(parents=True, exist_ok=True)
        (res / "chip.txt").write_text(
            f"{self.chip}\tdevice={self.device}\thost={os.uname().nodename}\n"
        )
        fails = 0
        for arm in arms:
            flags = (BASE / "flags" / (arm["flagskey"] + ".txt")).read_text().strip()
            env_extra = {"__group": group_of(arm)}
            for kv in filter(None, arm["env"].split(";")):
                k, _, v = kv.partition("=")
                env_extra[k] = v
            aname = arm["arm"]
            if arm["corr_node"]:
                rc = self._session(
                    res / f"{aname}-corr", arm["corr_node"], flags, dict(env_extra)
                )
                if rc != 0:
                    (res / f"{aname}-CORR-FAIL.txt").write_text(f"rc={rc}\n")
                    print(
                        f"[{self.tag}] {op}/{leg} {aname}: CORR FAIL rc={rc}",
                        flush=True,
                    )
                    fails += 1
                    continue
            for k in range(1, NREPS + 1):
                sdir = res / f"{aname}-perf-r{k}"
                rc = self._session(sdir, arm["perf_node"], flags, dict(env_extra))
                if rc != 0 and not (sdir / "rc.first.txt").is_file():
                    # one retry (transient hiccup); kill-tree only, no reset
                    (sdir / "rc.txt").rename(sdir / "rc.first.txt")
                    if (sdir / "log.txt").is_file():
                        (sdir / "log.txt").rename(sdir / "log.first.txt")
                    shutil.rmtree(sdir / "perf_data", ignore_errors=True)
                    rc = self._session(sdir, arm["perf_node"], flags, dict(env_extra))
                if rc != 0:
                    fails += 1
                print(f"[{self.tag}] {op}/{leg} {aname} r{k}: rc={rc}", flush=True)
        with (self.wroot / "done-ops.txt").open("a") as f:
            f.write(f"{op}\t{leg}\n")
        self.done_ops.add((op, leg))
        return fails

    # ---- batched mode (LK_BATCH=1) — everything below is opt-in ----

    def _batch_session(self, sdir, nodes, flags, env_extra, timeout):
        """One pytest session over many DISTINCT nodes, with the kit demux
        plugin recording per-node outcomes and per-node perf_data trees.
        Resume-safe via rc.txt like _session.  Returns (rc, demux_dir)."""
        demux = sdir / "demux"
        rcf = sdir / "rc.txt"
        if rcf.is_file():
            return int(rcf.read_text().strip() or 99), demux
        sdir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(demux, ignore_errors=True)
        demux.mkdir(parents=True)
        env = dict(os.environ)
        for k in ("TT_METAL_SIMULATOR", "TT_UMD_SIMULATOR_PATH"):
            env.pop(k, None)
        group = env_extra.pop("__group")
        rt = self._runner_temp(group)
        plugdir = str(Path(__file__).resolve().parent)
        pyp = env.get("PYTHONPATH", "")
        env.update(
            CHIP_ARCH="blackhole",
            SHORT_ARCH="bh",
            LLK_HOME=str(self.farm),
            RUNNER_TEMP=str(rt),
            TT_LLK_EXTRA_COMPILER_OPTIONS=flags,
            TT_VISIBLE_DEVICES=self.device,
            PYTHONUNBUFFERED="1",
            LK_BATCH_DEMUX=str(demux),
            PYTHONPATH=plugdir + (os.pathsep + pyp if pyp else ""),
        )
        env.update(env_extra)
        shutil.rmtree(self.farm / "perf_data", ignore_errors=True)
        shutil.rmtree(rt / "tt-llk-build/temp_perf_data", ignore_errors=True)
        # an empty node list would silently run the ENTIRE suite — refuse
        assert nodes, "refuse: empty node list would run the full suite"
        (sdir / "nodes.txt").write_text("\n".join(nodes) + "\n")
        (sdir / "flags.txt").write_text(flags + "\n")
        cmd = [
            self.python,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "-p",
            "lk_batch_plugin",
            "--compile-consumer",
            *nodes,
        ]
        t0 = time.time()
        try:
            r = subprocess.run(
                cmd,
                cwd=self.farm / "tests/python_tests",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            out, rc = r.stdout, r.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"") + b"\nLK-SESSION-TIMEOUT\n"
            rc = 98
        dt = time.time() - t0
        (sdir / "log.txt").write_bytes(out)
        # the harness's own combined (multi-op, NOT per-cell) perf_data is
        # kept for audit only; cells use the plugin's per-node demux trees
        pd = self.farm / "perf_data"
        if pd.is_dir():
            shutil.copytree(pd, sdir / "perf_data-combined", dirs_exist_ok=True)
            shutil.rmtree(pd, ignore_errors=True)
        (sdir / "secs.txt").write_text(f"{dt:.1f}\n")
        rcf.write_text(f"{rc}\n")
        return rc, demux

    @staticmethod
    def _demux_outcomes(demux):
        """node -> True iff every recorded phase so far passed (call included)."""
        ok = {}
        f = demux / "outcomes.jsonl"
        if not f.is_file():
            return ok
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            node, when, outc = rec["node"], rec["when"], rec["outcome"]
            if when == "call":
                ok[node] = ok.get(node, True) and outc == "passed"
            elif outc != "passed" and when == "setup":
                ok[node] = False
        return ok

    @staticmethod
    def _demux_perfdata(demux):
        """node -> per-node perf_data dir (indexed only after a successful
        per-node combine — presence is the demux-OK contract)."""
        m = {}
        f = demux / "index.tsv"
        if not f.is_file():
            return m
        for line in f.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                pdd = demux / parts[1] / "out/perf_data"
                if pdd.is_dir():
                    m[parts[0]] = pdd
        return m

    def _emit_cell(self, cell, node, flags, bdir, secs, perfdata=None):
        """Synthesize one result cell (same file schema as a solo session)
        from a batched session.  rc.txt is written LAST (resume marker)."""
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "node.txt").write_text(node + "\n")
        (cell / "flags.txt").write_text(flags + "\n")
        (cell / "log.txt").write_text(f"LK-BATCHED: session log at {bdir}/log.txt\n")
        (cell / "batch.txt").write_text(str(bdir) + "\n")
        (cell / "secs.txt").write_text(secs)
        if perfdata is not None:
            shutil.copytree(perfdata, cell / "perf_data", dirs_exist_ok=True)
        (cell / "rc.txt").write_text("0\n")

    @staticmethod
    def _kernel_cycles(cell, metric):
        """Ledger-identical kernel_cycles of one cell: sum of mean(<metric>)
        over KERNEL-marker rows of the cell's (single) post CSV."""
        col = f"mean({metric})"
        for post in sorted(cell.glob("perf_data/*/*.post.csv")):
            tot, seen = 0.0, False
            with post.open() as f:
                for rec in csv.DictReader(f):
                    if rec.get("marker") == "KERNEL" and col in rec:
                        tot += float(rec[col])
                        seen = True
            if seen:
                return tot
        return None

    def _audit_op(self, op, leg, arms, res, flags_by_arm, env_by_arm):
        """Honesty audit for batched measurements: ONE extra SOLO perf session
        per corr-clean arm of one op per batch group.  Reps are expected
        cycle-identical, so the solo kernel_cycles must equal every batched
        rep's — a difference is an in-session state-leak (reconfig escape)
        signal, flagged as AUDIT-DIVERGE for adjudication, never silently
        accepted."""
        metric = None
        rows = BASE / "ROWS.tsv"
        if rows.is_file():
            for r in csv.DictReader(rows.open(), delimiter="\t"):
                if (r["op"], r["leg"]) == (op, leg):
                    metric = r["metric"]
                    break
        for arm in arms:
            a = arm["arm"]
            crc = res / f"{a}-corr/rc.txt"
            if arm["corr_node"] and (
                not crc.is_file() or crc.read_text().strip() != "0"
            ):
                continue
            batched = [
                res / f"{a}-perf-r{k}"
                for k in range(1, NREPS + 1)
                if (res / f"{a}-perf-r{k}/batch.txt").is_file()
            ]
            if not batched:
                continue  # every rep already ran solo; nothing to audit
            sdir = res / f"{a}-perf-audit"
            rc = self._session(
                sdir, arm["perf_node"], flags_by_arm[a], dict(env_by_arm[a])
            )
            print(f"[{self.tag}] {op}/{leg} {a} audit: rc={rc}", flush=True)
            if rc != 0 or metric is None:
                continue
            solo_kc = self._kernel_cycles(sdir, metric)
            vals = {b.name: self._kernel_cycles(b, metric) for b in batched}
            (res / f"AUDIT-{a}.txt").write_text(
                f"metric={metric}\tsolo_audit={solo_kc}\ttol={BATCH_AUDIT_TOL}\n"
                + "".join(f"{n}\t{v}\n" for n, v in sorted(vals.items()))
            )
            bad = [
                n
                for n, v in vals.items()
                if v is None or solo_kc is None or abs(v - solo_kc) > BATCH_AUDIT_TOL
            ]
            if bad:
                (res / f"AUDIT-DIVERGE-{a}.txt").write_text(
                    f"solo audit kernel_cycles={solo_kc} (metric {metric}) "
                    f"differs beyond tol={BATCH_AUDIT_TOL} from batched "
                    f"rep(s): {','.join(sorted(bad))}\n"
                    "suspected in-session state leak; adjudicate before "
                    "booking this op's batched cells.\n"
                )
                print(
                    f"[{self.tag}] {op}/{leg} {a}: AUDIT-DIVERGE "
                    f"(solo {solo_kc} vs batched {sorted(bad)})",
                    flush=True,
                )

    def _solo_perf_cell(self, sdir, node, flags, env_extra):
        """Default-path solo perf session for one cell, with the standard
        one-retry-after-kill-tree rule (mirrors run_item's perf loop)."""
        rc = self._session(sdir, node, flags, dict(env_extra))
        if rc != 0 and not (sdir / "rc.first.txt").is_file():
            (sdir / "rc.txt").rename(sdir / "rc.first.txt")
            if (sdir / "log.txt").is_file():
                (sdir / "log.txt").rename(sdir / "log.first.txt")
            shutil.rmtree(sdir / "perf_data", ignore_errors=True)
            rc = self._session(sdir, node, flags, dict(env_extra))
        return rc

    def run_batch(self, batch, arms):
        """Run a claimed batch of (op, leg) items.  Arms are grouped by
        (flagskey, env) — a session's flags/env are session-wide, so only
        same-group arms may share one.  Per group: ONE corr session gating
        each arm, then ONE session per rep index over all gated perf nodes.
        Any cell the batch cannot prove runs as a solo default session."""
        fails = 0
        entries = []
        for op, leg in batch:
            res = BASE / "results" / f"{op}__{leg}" / self.tag
            res.mkdir(parents=True, exist_ok=True)
            (res / "chip.txt").write_text(
                f"{self.chip}\tdevice={self.device}\thost={os.uname().nodename}\n"
            )
            for arm in arms[(op, leg)]:
                entries.append((op, leg, arm, res))
        groups = {}
        for e in entries:
            groups.setdefault((e[2]["flagskey"], e[2]["env"]), []).append(e)
        bid = hashlib.sha1(
            ("|".join(sorted(f"{op}__{leg}" for op, leg in batch))).encode()
        ).hexdigest()[:12]
        for gi, ((fk, envs), ents) in enumerate(sorted(groups.items())):
            flags = (BASE / "flags" / (fk + ".txt")).read_text().strip()
            env_extra = {"__group": group_of(ents[0][2])}
            for kv in filter(None, envs.split(";")):
                k, _, v = kv.partition("=")
                env_extra[k] = v
            bdir = self.wroot / "batch" / f"{bid}-g{gi}"

            # ---- corr phase: one batched session; only a batched PASS is
            # booked directly.  A batched corr FAIL could be a reconfig
            # escape from a PRECEDING node in the session (tt-llk known
            # issue), so it is re-checked SOLO before anything is booked;
            # a solo pass after a batch fail is recorded as suspect.
            batch_corr_fail = set()
            need = [
                e
                for e in ents
                if e[2]["corr_node"]
                and not (e[3] / f"{e[2]['arm']}-corr/rc.txt").is_file()
            ]
            if need:
                nodes = list(dict.fromkeys(e[2]["corr_node"] for e in need))
                to = SESSION_TIMEOUT + BATCH_NODE_SECS * len(nodes)
                rc, demux = self._batch_session(
                    bdir / "corr", nodes, flags, dict(env_extra), to
                )
                ok = self._demux_outcomes(demux)
                secs = (
                    (bdir / "corr/secs.txt").read_text()
                    if (bdir / "corr/secs.txt").is_file()
                    else "0.0\n"
                )
                for op, leg, arm, res in need:
                    node = arm["corr_node"]
                    if ok.get(node):
                        self._emit_cell(
                            res / f"{arm['arm']}-corr", node, flags, bdir / "corr", secs
                        )
                    elif node in ok:
                        batch_corr_fail.add((op, leg, arm["arm"]))
            for op, leg, arm, res in ents:
                if not arm["corr_node"]:
                    continue
                cd = res / f"{arm['arm']}-corr"
                if not (cd / "rc.txt").is_file():  # batch fail/unknown -> solo
                    src = self._session(cd, arm["corr_node"], flags, dict(env_extra))
                    if src == 0 and (op, leg, arm["arm"]) in batch_corr_fail:
                        (res / f"BATCH-ESCAPE-SUSPECT-{arm['arm']}.txt").write_text(
                            "corr FAILED inside a batched session but PASSED "
                            "solo on the same chip: suspected reconfig escape "
                            "from a preceding node in the batch.\n"
                        )
                        print(
                            f"[{self.tag}] {op}/{leg} {arm['arm']}: "
                            "BATCH-ESCAPE-SUSPECT (batch corr fail, solo "
                            "corr pass)",
                            flush=True,
                        )
            gated = []
            for op, leg, arm, res in ents:
                if arm["corr_node"]:
                    crc = (res / f"{arm['arm']}-corr/rc.txt").read_text().strip()
                    if crc != "0":
                        cf = res / f"{arm['arm']}-CORR-FAIL.txt"
                        if not cf.is_file():
                            cf.write_text(f"rc={crc}\n")
                        print(
                            f"[{self.tag}] {op}/{leg} {arm['arm']}: "
                            f"CORR FAIL rc={crc} (batched)",
                            flush=True,
                        )
                        fails += 1
                        continue
                gated.append((op, leg, arm, res))

            # ---- perf phase: one batched session per rep index.  A device
            # hang mid-session fails every node after it (fast timeouts), so
            # a rep with missing cells gets ONE batched retry over just the
            # missing nodes (one startup instead of N solo sessions); the
            # solo fallback below remains the authority of last resort. ----
            for k in range(1, NREPS + 1):
                for attempt in ("", "-retry"):
                    need = [
                        e
                        for e in gated
                        if not (e[3] / f"{e[2]['arm']}-perf-r{k}/rc.txt").is_file()
                    ]
                    if not need:
                        break
                    sname = f"perf-r{k}{attempt}"
                    nodes = list(dict.fromkeys(e[2]["perf_node"] for e in need))
                    to = SESSION_TIMEOUT + BATCH_NODE_SECS * len(nodes)
                    rc, demux = self._batch_session(
                        bdir / sname, nodes, flags, dict(env_extra), to
                    )
                    ok = self._demux_outcomes(demux)
                    pdm = self._demux_perfdata(demux)
                    secs = (
                        (bdir / sname / "secs.txt").read_text()
                        if (bdir / sname / "secs.txt").is_file()
                        else "0.0\n"
                    )
                    for op, leg, arm, res in need:
                        node = arm["perf_node"]
                        if ok.get(node) and node in pdm:
                            self._emit_cell(
                                res / f"{arm['arm']}-perf-r{k}",
                                node,
                                flags,
                                bdir / sname,
                                secs,
                                perfdata=pdm[node],
                            )
                            print(
                                f"[{self.tag}] {op}/{leg} {arm['arm']} "
                                f"r{k}: rc=0 (batched{attempt})",
                                flush=True,
                            )
                    if rc == 0:
                        break
            # ---- solo fallback for any cell the batch could not prove ----
            for op, leg, arm, res in gated:
                for k in range(1, NREPS + 1):
                    sdir = res / f"{arm['arm']}-perf-r{k}"
                    if (sdir / "rc.txt").is_file():
                        continue
                    rc = self._solo_perf_cell(sdir, arm["perf_node"], flags, env_extra)
                    if rc != 0:
                        fails += 1
                    print(
                        f"[{self.tag}] {op}/{leg} {arm['arm']} r{k}: "
                        f"rc={rc} (solo-fallback)",
                        flush=True,
                    )

            # ---- audit: one op per group also measured SOLO per arm and
            # compared (reps are cycle-identical, so equality is the bar) ----
            if BATCH_AUDIT and gated:
                aop, aleg = gated[0][0], gated[0][1]
                aarms = [e[2] for e in gated if (e[0], e[1]) == (aop, aleg)]
                self._audit_op(
                    aop,
                    aleg,
                    aarms,
                    gated[0][3],
                    {a["arm"]: flags for a in aarms},
                    {a["arm"]: env_extra for a in aarms},
                )
        with (self.wroot / "done-ops.txt").open("a") as f:
            for op, leg in batch:
                f.write(f"{op}\t{leg}\n")
                self.done_ops.add((op, leg))
        return fails

    def steal_loop_batched(self):
        arms = read_arms()
        qdir = BASE / "queue"
        cdir = BASE / "claims"
        cdir.mkdir(exist_ok=True)
        idle_passes = 0
        while True:
            claimed = []
            in_batch = set()
            for item in sorted(p.name for p in qdir.iterdir()):
                op, leg, _copy = item.rsplit("__", 2)
                if (op, leg) in self.done_ops or (op, leg) in in_batch:
                    continue
                try:
                    (cdir / item).mkdir()
                except FileExistsError:
                    continue
                (cdir / item / "owner.txt").write_text(self.tag + "\n")
                claimed.append((item, op, leg))
                in_batch.add((op, leg))
                if len(claimed) >= BATCH_OPS:
                    break
            if claimed:
                print(
                    f"[{self.tag}] claimed batch: "
                    f"{','.join(i for i, _, _ in claimed)}",
                    flush=True,
                )
                self.run_batch([(op, leg) for _, op, leg in claimed], arms)
                for item, _, _ in claimed:
                    (cdir / item / "done.txt").write_text("done\n")
                idle_passes = 0
                continue
            idle_passes += 1
            if idle_passes >= 2:
                break
            time.sleep(10)
        print(f"LK-WORKER-DONE {self.tag}", flush=True)

    def steal_loop(self):
        arms = read_arms()
        qdir = BASE / "queue"
        cdir = BASE / "claims"
        cdir.mkdir(exist_ok=True)
        idle_passes = 0
        while True:
            progressed = False
            for item in sorted(p.name for p in qdir.iterdir()):
                op, leg, _copy = item.rsplit("__", 2)
                if (op, leg) in self.done_ops:
                    continue
                try:
                    (cdir / item).mkdir()
                except FileExistsError:
                    continue
                (cdir / item / "owner.txt").write_text(self.tag + "\n")
                print(f"[{self.tag}] claimed {item}", flush=True)
                self.run_item(op, leg, arms[(op, leg)])
                (cdir / item / "done.txt").write_text("done\n")
                progressed = True
            if not progressed:
                idle_passes += 1
                if idle_passes >= 2:
                    break
                time.sleep(10)
        print(f"LK-WORKER-DONE {self.tag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--device", help="TT_VISIBLE_DEVICES override (pilot/local)")
    ap.add_argument("--item", help="run one queue item directly (pilot mode)")
    args = ap.parse_args()
    w = Worker(args.chip, device=args.device)
    if args.item:
        op, leg = args.item.split("__")[:2]
        if BATCH:
            rc = w.run_batch([(op, leg)], read_arms())
        else:
            rc = w.run_item(op, leg, read_arms()[(op, leg)])
        sys.exit(0 if rc == 0 else 2)
    if BATCH:
        w.steal_loop_batched()
    else:
        w.steal_loop()


if __name__ == "__main__":
    main()
