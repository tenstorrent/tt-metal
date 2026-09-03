#!/usr/bin/env python3
"""galaxy-kit per-chip benchmark worker — work-stealing, execute-only.

Runs on a galaxy compute node.  ELFs were compiled on quietbox (stage.sh)
and shipped in builds/<group>/; every pytest session runs with
--compile-consumer and needs no riscv toolchain (only the shipped
riscv-tt-elf-size/objdump for the TEXT_SIZE metric).

Honesty rules baked in:
  - CORR GATE: an arm's perf reps run only after that arm's correctness
    node PASSED on this same chip (<arm>-CORR-FAIL.txt otherwise).  In a
    batched session the gate is enforced IN-SESSION by the kit's pytest
    plugin (lk_batch_plugin.py): a failed corr node makes the plugin skip
    that arm's perf nodes before they touch the device.
  - SAME-CHIP PAIRS: both arms of an (op, leg) item run back-to-back on
    the claiming worker's chip; the chip id is recorded per cell.
  - REPS: LK_REPS perf runs per arm (default 5); reps are expected
    cycle-identical (the collector reports spreads).
  - NO DEVICE RESETS ever in a worker (EXABOX.md §7 wall 2): recovery is
    one retry after a kill-tree session timeout, nothing else.

SESSION BATCHING (LK_BATCH_OPS, default 8): instead of one pytest session
per (op, chip, arm, rep) measurement — which pays a few seconds of harness
startup ~14k times per campaign — the worker claims up to LK_BATCH_OPS ops
at once and runs ONE pytest session per (chip, chunk), nodes ordered so
each op's block stays intact and adjacent:

    [op1 corr sem, op1 corr hand, op1 perf sem x reps, op1 perf hand x
     reps], [op2 ...], ...

Batching must not change what a measurement means, so:
  - only ops whose arms share one (flags, env) group share a session (the
    process environment is per-session); mixed-group ops (e.g. the trig
    licensed pair) fall back to one session per arm block, still adjacent;
  - the plugin writes ONE post CSV per node OCCURRENCE (per rep), through
    the harness's own postprocess/collapse code path, so per-rep values
    are preserved exactly as a solo session would record them (the
    harness's own module CSV would silently AVERAGE same-key reps);
  - the plugin refuses to run at all if pytest's collected order differs
    from the requested order (order is an invariant, not a hope);
  - RECONFIG-ESCAPE guard (tt-llk known issue: HW state can leak between
    kernel reconfigurations inside one session, so a later test can see
    state a solo run would not): (a) each op block leads with its corr
    nodes — a semantic escape fails the gate and the op is re-run SOLO
    (a solo corr pass after an in-batch corr fail is recorded as
    BATCH-ESCAPE-SUSPECT, never silently retried); (b) with LK_BATCH_AUDIT
    (default on) the first op of every chunk also gets one SOLO perf
    session per arm (<arm>-perf-audit) — reps are expected
    cycle-identical, so any batch-vs-solo difference is a pollution
    signal the ledger flags as AUDIT-DIVERGE; (c) every batched rep
    carries provenance (batch id, position, predecessor node) in
    batch.txt so an anomaly is attributable after the fact.
  - anything anomalous (order violation, session crash/timeout, failed or
    empty perf item) degrades to the proven SOLO path for that op — the
    batch is an optimization, solo is the authority of last resort.

LK_BATCH_OPS=0 disables batching entirely (legacy solo sessions).

Work-stealing: queue/<op>__<leg>__c<k> items, claimed by atomic mkdir in
claims/.  A worker skips copies of an (op,leg) it already ran, so K copies
land on K distinct chips.  Resume-safe: session dirs with rc.txt are kept.
"""
import argparse
import csv
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
BATCH_OPS = int(os.environ.get("LK_BATCH_OPS", "8"))
BATCH_AUDIT = os.environ.get("LK_BATCH_AUDIT", "1") not in ("0", "")
BATCH_NODE_SECS = int(os.environ.get("LK_BATCH_NODE_SECS", "40"))


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


def env_extra_of(arm):
    env_extra = {"__group": group_of(arm)}
    for kv in filter(None, arm["env"].split(";")):
        k, _, v = kv.partition("=")
        env_extra[k] = v
    return env_extra


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
        self.batch_seq = 0
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
                ["cp", "-al", str(BASE / "builds" / group / "tt-llk-build"),
                 str(rt / "tt-llk-build")],
                check=True,
            )
        self.rt_cache[group] = rt
        return rt

    def _pytest_env(self, flags, env_extra):
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
        return env

    def _run_pytest(self, sdir, nodes, env, timeout, extra_args=()):
        # ONE pytest invocation.  An empty node list would silently run the
        # ENTIRE suite — refuse (this is a known foot-gun, see project notes).
        assert nodes, "refuse: empty node list would run the full suite"
        cmd = [
            self.python, "-m", "pytest", "-o", "addopts=", "-q",
            "--compile-consumer", *extra_args, *nodes,
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
        (sdir / "secs.txt").write_text(f"{dt:.1f}\n")
        return rc

    def _session(self, sdir, node, flags, env_extra, timeout=SESSION_TIMEOUT):
        """One SOLO pytest session for one node (legacy grain; also the
        fallback/audit path in batch mode)."""
        rcf = sdir / "rc.txt"
        if rcf.is_file():
            return int(rcf.read_text().strip() or 99)
        sdir.mkdir(parents=True, exist_ok=True)
        env = self._pytest_env(flags, env_extra)
        (sdir / "node.txt").write_text(node + "\n")
        (sdir / "flags.txt").write_text(flags + "\n")
        rc = self._run_pytest(sdir, [node], env, timeout)
        pd = self.farm / "perf_data"
        if pd.is_dir():
            shutil.copytree(pd, sdir / "perf_data", dirs_exist_ok=True)
            shutil.rmtree(pd, ignore_errors=True)
        rcf.write_text(f"{rc}\n")
        return rc

    # ------------------------------------------------------------------
    # legacy solo item runner (LK_BATCH_OPS=0, and the batch fallback)
    # ------------------------------------------------------------------
    def run_item(self, op, leg, arms):
        res = BASE / "results" / f"{op}__{leg}" / self.tag
        res.mkdir(parents=True, exist_ok=True)
        (res / "chip.txt").write_text(
            f"{self.chip}\tdevice={self.device}\thost={os.uname().nodename}\n"
        )
        fails = 0
        for arm in arms:
            flags = (BASE / "flags" / (arm["flagskey"] + ".txt")).read_text().strip()
            env_extra = env_extra_of(arm)
            aname = arm["arm"]
            if arm["corr_node"]:
                rc = self._session(
                    res / f"{aname}-corr", arm["corr_node"], flags, dict(env_extra)
                )
                if rc != 0:
                    (res / f"{aname}-CORR-FAIL.txt").write_text(f"rc={rc}\n")
                    print(f"[{self.tag}] {op}/{leg} {aname}: CORR FAIL rc={rc}",
                          flush=True)
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
                    rc = self._session(sdir, arm["perf_node"], flags,
                                       dict(env_extra))
                if rc != 0:
                    fails += 1
                print(f"[{self.tag}] {op}/{leg} {aname} r{k}: rc={rc}", flush=True)
        with (self.wroot / "done-ops.txt").open("a") as f:
            f.write(f"{op}\t{leg}\n")
        self.done_ops.add((op, leg))
        return fails

    # ------------------------------------------------------------------
    # batched chunk runner
    # ------------------------------------------------------------------
    def _res_dir(self, op, leg):
        res = BASE / "results" / f"{op}__{leg}" / self.tag
        res.mkdir(parents=True, exist_ok=True)
        cf = res / "chip.txt"
        if not cf.is_file():
            cf.write_text(
                f"{self.chip}\tdevice={self.device}\thost={os.uname().nodename}\n"
            )
        return res

    def _op_needs_work(self, op, leg, arms):
        """True unless every session dir this op needs already has rc.txt=0
        (resume: a fully-booked op is only marked done, never re-run)."""
        res = BASE / "results" / f"{op}__{leg}" / self.tag
        for arm in arms:
            a = arm["arm"]
            if arm["corr_node"] and not (res / f"{a}-corr" / "rc.txt").is_file():
                return True
            if (res / f"{a}-CORR-FAIL.txt").is_file():
                continue
            for k in range(1, NREPS + 1):
                if not (res / f"{a}-perf-r{k}" / "rc.txt").is_file():
                    return True
        return False

    def _build_batch_spec(self, ops, arms_map):
        """Ordered node occurrences for one batched session.
        Charter order per op: corr(sem), corr(hand), perf(sem) x reps,
        perf(hand) x reps.  Occurrences whose session dir already holds an
        rc.txt (resume) are not re-requested."""
        occ = []
        for op, leg in ops:
            res = BASE / "results" / f"{op}__{leg}" / self.tag
            arms = arms_map[(op, leg)]
            gate = {}
            for arm in arms:
                a = arm["arm"]
                if arm["corr_node"]:
                    if (res / f"{a}-corr" / "rc.txt").is_file():
                        # corr already booked on this chip (resume): gate
                        # already decided; only re-run if it PASSED
                        rc = (res / f"{a}-corr" / "rc.txt").read_text().strip()
                        gate[a] = None if rc == "0" else "FAILED-PRIOR"
                    else:
                        gate[a] = len(occ)
                        occ.append(dict(seq=len(occ), nodeid=arm["corr_node"],
                                        role="corr", op=op, leg=leg, arm=a,
                                        rep=0, gate_seq=None))
                else:
                    gate[a] = None
            for arm in arms:
                a = arm["arm"]
                if gate.get(a) == "FAILED-PRIOR":
                    continue
                for k in range(1, NREPS + 1):
                    if (res / f"{a}-perf-r{k}" / "rc.txt").is_file():
                        continue
                    occ.append(dict(seq=len(occ), nodeid=arm["perf_node"],
                                    role="perf", op=op, leg=leg, arm=a,
                                    rep=k, gate_seq=gate.get(a)))
        return occ

    def _materialize(self, occ_by_seq, manifest, bdir, bid, flags):
        """Turn the plugin's per-occurrence outputs into the solo results
        layout the ledger already reads.  Returns (ops_needing_solo,
        corr_failed_in_batch)."""
        need_solo, corr_failed = set(), set()
        prev_node = {"v": "<session-start>"}
        for row in manifest:
            o = occ_by_seq.get(int(row["seq"]))
            if o is None:
                continue
            res = self._res_dir(o["op"], o["leg"])
            prov = (f"batch={bid}\tseq={row['seq']}\tprev={prev_node['v']}\t"
                    f"outcome={row['outcome']}\n")
            prev_node["v"] = row["nodeid"]
            if o["role"] == "corr":
                if row["outcome"] == "passed":
                    d = res / f"{o['arm']}-corr"
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "node.txt").write_text(o["nodeid"] + "\n")
                    (d / "batch.txt").write_text(prov)
                    (d / "rc.txt").write_text("0\n")
                else:
                    # do NOT book CORR-FAIL from inside a batch: a reconfig
                    # escape from a preceding op could be the real culprit.
                    # The op is re-run solo; a solo pass exposes the escape.
                    # (a "bailed" corr never ran — no escape evidence)
                    if row["outcome"] in ("failed", "setup-error"):
                        corr_failed.add((o["op"], o["leg"], o["arm"]))
                    need_solo.add((o["op"], o["leg"]))
            else:
                csvf = bdir / "items" / f"{int(row['seq']):04d}.post.csv"
                if row["outcome"] == "passed" and csvf.is_file():
                    d = res / f"{o['arm']}-perf-r{o['rep']}"
                    pd_dir = d / "perf_data" / "batch"
                    pd_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(csvf, pd_dir / f"item{int(row['seq']):04d}.post.csv")
                    rawf = bdir / "items" / f"{int(row['seq']):04d}.raw.csv"
                    if rawf.is_file():
                        shutil.copy2(rawf, d / "raw.csv")
                    (d / "node.txt").write_text(o["nodeid"] + "\n")
                    (d / "flags.txt").write_text(flags + "\n")
                    (d / "batch.txt").write_text(prov)
                    (d / "secs.txt").write_text(row.get("secs", "") + "\n")
                    (d / "rc.txt").write_text("0\n")
                elif row["outcome"] == "gate-skipped":
                    pass  # its arm's corr failed; op is already in need_solo
                else:
                    need_solo.add((o["op"], o["leg"]))
        return need_solo, corr_failed

    def run_chunk_bucket(self, ops, arms_map, flags, env_extra):
        """ONE pytest session for a list of (op,leg) whose arms all share
        one (flags,env) group; then audit + solo fallback as needed."""
        occ = self._build_batch_spec(ops, arms_map)
        if not occ:
            return set(), set()
        self.batch_seq += 1
        bid = f"{self.tag}-b{self.batch_seq:04d}"
        bdir = BASE / "results" / "_batches" / self.tag / f"b{self.batch_seq:04d}"
        (bdir / "items").mkdir(parents=True, exist_ok=True)
        spec = dict(batch_id=bid, occurrences=occ)
        (bdir / "spec.json").write_text(json.dumps(spec, indent=1))
        (bdir / "flags.txt").write_text(flags + "\n")
        env = self._pytest_env(flags, dict(env_extra))
        env["LK_BATCH_SPEC"] = str(bdir / "spec.json")
        env["LK_BATCH_OUT"] = str(bdir / "items")
        # the plugin sits next to worker.py (BASE in the shipped layout,
        # the kit's lib/ when run from a checkout)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(Path(__file__).resolve().parent), str(BASE),
                        env.get("PYTHONPATH", "")) if p)
        timeout = SESSION_TIMEOUT + BATCH_NODE_SECS * len(occ)
        print(f"[{self.tag}] batch {bid}: {len(ops)} ops, {len(occ)} node "
              f"occurrences, timeout {timeout}s", flush=True)
        # each node once (pytest dedupes duplicate node-id args anyway);
        # the plugin expands them back into the spec's occurrence order
        nodes, seen = [], set()
        for o in occ:
            if o["nodeid"] not in seen:
                seen.add(o["nodeid"])
                nodes.append(o["nodeid"])
        rc = self._run_pytest(
            bdir, nodes, env, timeout,
            extra_args=["-p", "lk_batch_plugin"],
        )
        (bdir / "rc.txt").write_text(f"{rc}\n")
        # keep the harness's own (collapsed) session CSVs for adjudication
        pd = self.farm / "perf_data"
        if pd.is_dir():
            shutil.copytree(pd, bdir / "harness_perf_data", dirs_exist_ok=True)
            shutil.rmtree(pd, ignore_errors=True)
        manifest = []
        mf = bdir / "items" / "manifest.tsv"
        order_violation = False
        if mf.is_file():
            with mf.open() as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r.get("outcome") == "ORDER-VIOLATION":
                        order_violation = True
                    manifest.append(r)
        if order_violation or not manifest:
            # collection refused / plugin never ran: nothing is booked from
            # this session; every op goes solo
            print(f"[{self.tag}] batch {bid}: rc={rc} "
                  f"{'ORDER-VIOLATION' if order_violation else 'NO-MANIFEST'}"
                  " -> all ops solo", flush=True)
            return set(ops), set()
        occ_by_seq = {o["seq"]: o for o in occ}
        need_solo, corr_failed = self._materialize(
            occ_by_seq, manifest, bdir, bid, flags)
        # occurrences the session never reached (timeout/crash) need solo too
        seen = {int(r["seq"]) for r in manifest if r["seq"].isdigit()}
        for o in occ:
            if o["seq"] not in seen:
                need_solo.add((o["op"], o["leg"]))
        print(f"[{self.tag}] batch {bid}: rc={rc} manifest={len(manifest)} "
              f"solo-fallback={sorted(op for op, _ in need_solo)}", flush=True)
        return need_solo, corr_failed

    def _audit(self, op, leg, arms_map):
        """One SOLO perf session per arm for one batched op: reps are
        expected cycle-identical, so any batch-vs-solo difference is a
        reconfig-escape signal (the ledger compares and flags)."""
        res = BASE / "results" / f"{op}__{leg}" / self.tag
        for arm in arms_map[(op, leg)]:
            a = arm["arm"]
            corr_rc = res / f"{a}-corr" / "rc.txt"
            if arm["corr_node"] and (not corr_rc.is_file()
                                     or corr_rc.read_text().strip() != "0"):
                continue  # audit only a corr-clean arm
            flags = (BASE / "flags" / (arm["flagskey"] + ".txt")).read_text().strip()
            sdir = res / f"{a}-perf-audit"
            rc = self._session(sdir, arm["perf_node"], flags, env_extra_of(arm))
            print(f"[{self.tag}] {op}/{leg} {a} audit: rc={rc}", flush=True)

    def run_chunk(self, chunk, arms_map):
        """chunk = list of (op, leg).  Partition into same-group buckets
        (one batched session each, ops in claim order) and mixed-group ops
        (one batched session PER ARM BLOCK, the two blocks adjacent)."""
        buckets, mixed = {}, []
        for op, leg in chunk:
            arms = arms_map[(op, leg)]
            groups = {group_of(a) for a in arms}
            if len(groups) == 1:
                buckets.setdefault(groups.pop(), []).append((op, leg))
            else:
                mixed.append((op, leg))
        need_solo_all, corr_failed_all = set(), set()
        audit_ops = []
        for g, ops in buckets.items():
            arm0 = arms_map[ops[0]][0]
            flags = (BASE / "flags" / (arm0["flagskey"] + ".txt")).read_text().strip()
            ns, cf = self.run_chunk_bucket(ops, arms_map, flags,
                                           env_extra_of(arm0))
            need_solo_all |= ns
            corr_failed_all |= cf
            if BATCH_AUDIT and ops[0] not in ns:
                audit_ops.append(ops[0])
        for op, leg in mixed:
            # per-arm blocks, back-to-back on this chip (A/B adjacency)
            for arm in arms_map[(op, leg)]:
                sub = {(op, leg): [arm]}
                flags = (BASE / "flags" /
                         (arm["flagskey"] + ".txt")).read_text().strip()
                ns, cf = self.run_chunk_bucket([(op, leg)], sub, flags,
                                               env_extra_of(arm))
                need_solo_all |= ns
                corr_failed_all |= cf
            if BATCH_AUDIT and (op, leg) not in need_solo_all:
                audit_ops.append((op, leg))
        # solo fallback: the resume-safe solo runner re-runs exactly the
        # sessions that lack rc.txt (booked batch sessions are kept)
        for op, leg in sorted(need_solo_all):
            fails = self.run_item(op, leg, arms_map[(op, leg)])
            for a in [x for (o2, l2, x) in corr_failed_all
                      if (o2, l2) == (op, leg)]:
                res = BASE / "results" / f"{op}__{leg}" / self.tag
                solo_rc = res / f"{a}-corr" / "rc.txt"
                if solo_rc.is_file() and solo_rc.read_text().strip() == "0":
                    (res / f"BATCH-ESCAPE-SUSPECT-{a}.txt").write_text(
                        "corr FAILED inside a batched session but PASSED "
                        "solo on the same chip: suspected reconfig escape "
                        "from a preceding op in the batch.\n")
                    print(f"[{self.tag}] {op}/{leg} {a}: "
                          "BATCH-ESCAPE-SUSPECT (batch corr fail, solo corr "
                          "pass)", flush=True)
        for op, leg in audit_ops:
            self._audit(op, leg, arms_map)
        for op, leg in chunk:
            if (op, leg) not in self.done_ops:
                with (self.wroot / "done-ops.txt").open("a") as f:
                    f.write(f"{op}\t{leg}\n")
                self.done_ops.add((op, leg))

    # ------------------------------------------------------------------
    # work stealing
    # ------------------------------------------------------------------
    def steal_loop(self):
        arms = read_arms()
        qdir = BASE / "queue"
        cdir = BASE / "claims"
        cdir.mkdir(exist_ok=True)
        idle_passes = 0
        while True:
            claimed = []  # [(op, leg, item-name)]
            for item in sorted(p.name for p in qdir.iterdir()):
                op, leg, _copy = item.rsplit("__", 2)
                if (op, leg) in self.done_ops:
                    continue
                if any(c[0] == op and c[1] == leg for c in claimed):
                    continue
                try:
                    (cdir / item).mkdir()
                except FileExistsError:
                    continue
                (cdir / item / "owner.txt").write_text(self.tag + "\n")
                print(f"[{self.tag}] claimed {item}", flush=True)
                claimed.append((op, leg, item))
                if BATCH_OPS and len(claimed) >= BATCH_OPS:
                    break
                if not BATCH_OPS:
                    break
            if claimed:
                if BATCH_OPS:
                    self.run_chunk([(o, l) for o, l, _ in claimed], arms)
                else:
                    for op, leg, _ in claimed:
                        self.run_item(op, leg, arms[(op, leg)])
                for _, _, item in claimed:
                    (cdir / item / "done.txt").write_text("done\n")
                idle_passes = 0
                continue
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
        arms = read_arms()
        if BATCH_OPS:
            w.run_chunk([(op, leg)], arms)
            res = BASE / "results" / f"{op}__{leg}" / w.tag
            bad = (w._op_needs_work(op, leg, arms[(op, leg)])
                   or any(res.glob("*-CORR-FAIL.txt"))
                   or any(res.glob("BATCH-ESCAPE-SUSPECT-*.txt")))
            rc = 1 if bad else 0
        else:
            rc = w.run_item(op, leg, arms[(op, leg)])
        sys.exit(0 if rc == 0 else 2)
    w.steal_loop()


if __name__ == "__main__":
    main()
