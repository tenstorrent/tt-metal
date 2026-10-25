# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""In-trace cross-sub-device ordering: ttnn.experimental.subdevice_signal / subdevice_wait (GLM pipeline E2).

Layout: persistent manager {SD0 = (0,0)-(10,9) (110 cores), SD1 = column x=11 (10 cores)}.
Per replay (one "round"), with the stream ops below issued in the order given by a variant string:
  producer stream on SD0:  D = spin-delay kernel (1 core), A = x += 1 (multi-core eltwise, in place)
  consumer stream on SD1:  K = viol += (x - e)^2 (3 multi-core eltwise ops), E = e += 1
  hand-off:                S = subdevice_signal on SD0 (1 core), W = subdevice_wait on SD1 (1 core),
                           F = a 1-core no-op program on SD0 (apply_device_delay(0)): a pure "fence"
State: x = 0, e = 1, viol = 0. After replay k (1-based) with correct ordering x = k, e = k + 1 and viol stays 0;
every element the consumer reads before the producer wrote it adds 1 to viol (x = k - 1 vs e = k).

Ping-pong variants ("pp"): R rounds per replay with a forward hand-off (SD0 -> SD1, data ready) and a backward
hand-off (SD1 -> SD0, data consumed) so the same two semaphores are reused R times per replay.

Run as pytest (gate tests) or as a script for variant batches / timing:
  python test_subdevice_signal_wait.py --mesh 1x1 --replays 1000 --variants "fence:D A F K E,sw:W D A S K E"
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback

import pytest
import torch
import ttnn

SD0 = ttnn.SubDeviceId(0)
SD1 = ttnn.SubDeviceId(1)
SD0_CORES = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
SD1_CORES = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(11, 0), ttnn.CoreCoord(11, 9))})
# Waiter cores (where each semaphore lives) and signaller cores. Chosen away from (0,0), where apply_device_delay
# (the D/F programs) runs, so S and D never share a core (they are on the same sub-device anyway).
FWD_WAIT_CORE = ttnn.CoreCoord(11, 0)  # on SD1: consumer waits here for "x ready"
FWD_SIGNAL_CORE = ttnn.CoreCoord(10, 9)  # on SD0
BWD_WAIT_CORE = ttnn.CoreCoord(10, 9)  # on SD0: producer waits here for "x consumed"
BWD_SIGNAL_CORE = ttnn.CoreCoord(11, 9)  # on SD1

HAS_OPS = hasattr(ttnn, "experimental") and hasattr(ttnn.experimental, "subdevice_signal")


def log(msg):
    print(f"[e2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Watchdog:
    """Exit with 124 (devrun treats it as a hang -> NEEDS_RESET) if a phase does not finish in time."""

    def __init__(self, seconds, what):
        self.t = threading.Timer(seconds, self._fire, args=(what,))
        self.t.daemon = True

    def _fire(self, what):
        print(f"[e2] WATCHDOG: '{what}' did not finish; dumping stacks and exiting 124", flush=True)
        import faulthandler

        faulthandler.dump_traceback(all_threads=True)
        os._exit(124)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.t.cancel()


class Harness:
    def __init__(self, mesh, numel_tiles=256, delay_cycles=100_000, timeout_cycles=0):
        self.mesh = mesh
        self.rows, self.cols = mesh.shape[0], mesh.shape[1]
        self.ndev = mesh.get_num_devices()
        self.delay_cycles = delay_cycles
        self.timeout_cycles = timeout_cycles
        self.shape = [1, 1, 32 * 8, 32 * (numel_tiles // 8)]
        self.numel = self.shape[2] * self.shape[3]
        self.mgr = mesh.create_sub_device_manager([ttnn.SubDevice([SD0_CORES]), ttnn.SubDevice([SD1_CORES])], 0)
        mesh.load_sub_device_manager(self.mgr)
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def dev(v):
            return ttnn.from_torch(
                torch.full(self.shape, float(v)),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.x, self.e, self.d, self.viol = dev(0), dev(1), dev(0), dev(0)
        self.sem_fwd = self.sem_bwd = self.sem_dummy = None
        if HAS_OPS:
            self.sem_fwd = ttnn.create_global_semaphore(mesh, ttnn.CoreRangeSet({ttnn.CoreRange(FWD_WAIT_CORE, FWD_WAIT_CORE)}), 0)
            self.sem_bwd = ttnn.create_global_semaphore(mesh, ttnn.CoreRangeSet({ttnn.CoreRange(BWD_WAIT_CORE, BWD_WAIT_CORE)}), 0)
            self.sem_dummy = ttnn.create_global_semaphore(mesh, ttnn.CoreRangeSet({ttnn.CoreRange(BWD_SIGNAL_CORE, BWD_SIGNAL_CORE)}), 0)
        ttnn.synchronize_device(mesh)

    def close(self):
        ttnn.synchronize_device(self.mesh)
        for t in (self.x, self.e, self.d, self.viol):
            ttnn.deallocate(t)
        self.sem_fwd = self.sem_bwd = self.sem_dummy = None
        self.mesh.clear_loaded_sub_device_manager()
        self.mesh.remove_sub_device_manager(self.mgr)

    def reset_state(self, bwd_credit=0):
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for t, v in ((self.x, 0), (self.e, 1), (self.d, 0), (self.viol, 0)):
            h = ttnn.from_torch(torch.full(self.shape, float(v)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
            ttnn.copy_host_to_device_tensor(h, t)
        if self.sem_fwd is not None:
            ttnn.reset_global_semaphore_value(self.sem_fwd, 0)
            ttnn.reset_global_semaphore_value(self.sem_bwd, bwd_credit)
            ttnn.reset_global_semaphore_value(self.sem_dummy, 0)
        ttnn.synchronize_device(self.mesh)

    def read(self, t):
        return [ttnn.to_torch(h) for h in ttnn.get_device_tensors(t.cpu())]

    # ---- stream ops ----
    def op(self, tok, rnd=0, nrounds=1, credit=False):
        m = self.mesh
        if tok == "D":
            ttnn.apply_device_delay(m, [[self.delay_cycles] * self.cols for _ in range(self.rows)], subdevice_id=SD0)
        elif tok == "F":
            ttnn.apply_device_delay(m, [[0] * self.cols for _ in range(self.rows)], subdevice_id=SD0)
        elif tok == "A":
            ttnn.add(self.x, 1.0, output_tensor=self.x, sub_device_id=SD0)
        elif tok == "K":
            ttnn.subtract(self.x, self.e, output_tensor=self.d, sub_device_id=SD1)
            ttnn.multiply(self.d, self.d, output_tensor=self.d, sub_device_id=SD1)
            ttnn.add(self.viol, self.d, output_tensor=self.viol, sub_device_id=SD1)
        elif tok == "E":
            ttnn.add(self.e, 1.0, output_tensor=self.e, sub_device_id=SD1)
        elif tok == "k":  # K on SD0 (same-sub-device baseline)
            ttnn.subtract(self.x, self.e, output_tensor=self.d, sub_device_id=SD0)
            ttnn.multiply(self.d, self.d, output_tensor=self.d, sub_device_id=SD0)
            ttnn.add(self.viol, self.d, output_tensor=self.viol, sub_device_id=SD0)
        elif tok == "e":
            ttnn.add(self.e, 1.0, output_tensor=self.e, sub_device_id=SD0)
        elif tok == "S":  # forward: x ready (SD0 -> waiter on SD1)
            ttnn.experimental.subdevice_signal(
                self.sem_fwd, subdevice_id=SD0, core=FWD_SIGNAL_CORE, consumer_core=FWD_WAIT_CORE
            )
        elif tok in ("W", "Wt"):  # Wt: with the debug timeout
            ttnn.experimental.subdevice_wait(
                self.sem_fwd,
                subdevice_id=SD1,
                core=FWD_WAIT_CORE,
                timeout_cycles=self.timeout_cycles if tok == "Wt" else 0,
            )
        elif tok == "s":  # backward: x consumed (SD1 -> waiter on SD0); skipped on the last round
            if credit or rnd < nrounds - 1:
                ttnn.experimental.subdevice_signal(
                    self.sem_bwd, subdevice_id=SD1, core=BWD_SIGNAL_CORE, consumer_core=BWD_WAIT_CORE
                )
        elif tok == "w":  # skipped on the first round
            if credit or rnd > 0:
                ttnn.experimental.subdevice_wait(self.sem_bwd, subdevice_id=SD0, core=BWD_WAIT_CORE)
        elif tok == "G":  # 1-core no-op program on SD1 (fence on the consumer side)
            ttnn.experimental.subdevice_signal(
                self.sem_dummy, subdevice_id=SD1, core=BWD_SIGNAL_CORE, consumer_core=BWD_SIGNAL_CORE, increment=1
            )
        else:
            raise ValueError(tok)

    def body(self, order, nrounds=1):
        for r in range(nrounds):
            for tok in order.split():
                self.op(tok, r, nrounds)

    def run_variant(self, name, order, replays, nrounds=1, eager_check=True, timeout_s=120):
        """Returns a dict with violation counts per device and state checks."""
        res = {"variant": name, "order": order, "rounds": nrounds, "replays": replays, "ndev": self.ndev}
        with Watchdog(timeout_s, f"{name} eager"):
            self.reset_state()
            self.body(order, nrounds)  # eager pass: compiles everything, and is itself a test
            ttnn.synchronize_device(self.mesh)
            if eager_check:
                v = self.read(self.viol)
                res["eager_viol"] = [int(t.to(torch.int64).sum().item()) for t in v]
            self.reset_state()
        with Watchdog(timeout_s, f"{name} capture"):
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.body(order, nrounds)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.synchronize_device(self.mesh)
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
            ttnn.ReadDeviceProfiler(self.mesh)  # drain eager/capture-time entries
        with Watchdog(timeout_s, f"{name} replay x{replays}"):
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)  # first replay (warm), counted
            ttnn.synchronize_device(self.mesh)
            t0 = time.perf_counter()
            for _ in range(replays - 1):
                ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            dt = time.perf_counter() - t0
        res["us_per_replay"] = dt / max(replays - 1, 1) * 1e6
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
            gaps, counts = summarize_gaps(profile_replays(self.mesh, [program_labels(order, nrounds)]))
            res["profile_us"] = gaps
            res["profile_n"] = max(counts.values()) if counts else 0
        ttnn.release_trace(self.mesh, tid)
        x, e, v = self.read(self.x), self.read(self.e), self.read(self.viol)
        n = replays * nrounds
        res["viol"] = [int(t.to(torch.int64).sum().item()) for t in v]
        res["viol_elems_max"] = self.numel * n
        res["x_ok"] = all(bool(torch.all(t == n)) for t in x)
        res["e_ok"] = all(bool(torch.all(t == n + 1)) for t in e)
        res["x_minmax"] = [float(x[0].min()), float(x[0].max())]
        res["e_minmax"] = [float(e[0].min()), float(e[0].max())]
        res["total_viol"] = sum(res["viol"])
        log(json.dumps(res))
        return res


    def run_2cq(self, name, prod, cons, replays, nrounds=1, timeout_s=120):
        """Producer stream on CQ0 (SD0 programs only), consumer stream on CQ1 (SD1 programs only), one trace per CQ.
        No dispatch coupling between the streams, so every cross-SD edge needs signal/wait. The backward semaphore
        starts with one credit: the producer waits for it ("w") before overwriting x, the consumer returns it ("s")."""
        res = {"variant": name, "order": f"cq0[{prod}] cq1[{cons}]", "rounds": nrounds, "replays": replays, "ndev": self.ndev}

        def body(order, cq):
            with ttnn.command_queue(cq):
                for r in range(nrounds):
                    for tok in order.split():
                        self.op(tok, r, nrounds, credit=True)

        with Watchdog(timeout_s, f"{name} eager"):
            self.reset_state(bwd_credit=1)
            body(prod, 0)
            body(cons, 1)
            ttnn.synchronize_device(self.mesh)
            res["eager_viol"] = [int(t.to(torch.int64).sum().item()) for t in self.read(self.viol)]
            self.reset_state(bwd_credit=1)
        with Watchdog(timeout_s, f"{name} capture"):
            tids = []
            for order, cq in ((prod, 0), (cons, 1)):
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=cq)
                body(order, cq)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=cq)
                tids.append(tid)
            ttnn.synchronize_device(self.mesh)
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
            ttnn.ReadDeviceProfiler(self.mesh)
        with Watchdog(timeout_s, f"{name} replay x{replays}"):
            t0 = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(self.mesh, tids[0], cq_id=0, blocking=False)
                ttnn.execute_trace(self.mesh, tids[1], cq_id=1, blocking=False)
            ttnn.synchronize_device(self.mesh)
            dt = time.perf_counter() - t0
        res["us_per_replay"] = dt / replays * 1e6
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
            gaps, counts = summarize_gaps(
                profile_replays(self.mesh, [program_labels(prod, nrounds, True), program_labels(cons, nrounds, True)])
            )
            res["profile_us"] = gaps
            res["profile_n"] = max(counts.values()) if counts else 0
        for tid in tids:
            ttnn.release_trace(self.mesh, tid)
        x, e, v = self.read(self.x), self.read(self.e), self.read(self.viol)
        n = replays * nrounds
        res["viol"] = [int(t.to(torch.int64).sum().item()) for t in v]
        res["viol_elems_max"] = self.numel * n
        res["x_ok"] = all(bool(torch.all(t == n)) for t in x)
        res["e_ok"] = all(bool(torch.all(t == n + 1)) for t in e)
        res["x_minmax"] = [float(x[0].min()), float(x[0].max())]
        res["e_minmax"] = [float(e[0].min()), float(e[0].max())]
        res["total_viol"] = sum(res["viol"])
        log(json.dumps(res))
        return res


# ------------------------------------------------------------------------------------------------------------------
# device-profiler analysis (TT_METAL_DEVICE_PROFILER=1): per-replay program FW start/end -> hand-off gaps
# ------------------------------------------------------------------------------------------------------------------
def program_labels(order, nrounds, credit=False):
    labels = []
    for r in range(nrounds):
        for tok in order.split():
            if tok in ("K", "k"):
                labels += [tok + "1", tok + "2", tok + "3"]
            elif tok == "s" and not (credit or r < nrounds - 1):
                continue
            elif tok == "w" and not (credit or r > 0):
                continue
            else:
                labels.append(tok if tok != "Wt" else "W")
    return labels


def profile_replays(mesh, label_lists):
    """label_lists: one label list per captured trace, in capture order. Returns per-replay dicts label -> (start, end)
    in cycles (DEVICE FW span), for the replays present in the profiler buffer, all chips (keyed by chip)."""
    ttnn.ReadDeviceProfiler(mesh)
    data = ttnn.get_latest_programs_perf_data()
    out = {}
    for chip, progs in data.items():
        by_trace = {}
        for p in progs:
            uid = p.program_execution_uid
            if uid.trace_id is None or uid.trace_id >= (1 << 63):
                continue
            fw = p.program_analyses_results.get("DEVICE FW DURATION [ns]")
            if fw is None:
                continue
            by_trace.setdefault(uid.trace_id, {}).setdefault(uid.trace_id_counter, []).append(
                (uid.runtime_id, fw.start_timestamp, fw.end_timestamp)
            )
        if not by_trace:
            log(f"profile: chip {chip}: {len(progs)} programs, none traced; sample uid {progs[0].program_execution_uid if progs else None}")
        traces = sorted(by_trace.items(), key=lambda kv: min(r[0] for c in kv[1].values() for r in c))
        traces = traces[-len(label_lists):]
        replays = []
        if len(traces) == len(label_lists):
            counters = [sorted(t[1].keys()) for t in traces]
            n = min(len(c) for c in counters)
            for i in range(n):
                rec = {}
                ok = True
                for (tid, per_counter), cs, labels in zip(traces, counters, label_lists):
                    rows = sorted(per_counter[cs[i]])
                    if len(rows) != len(labels):
                        if i == 0:
                            log(f"profile: chip {chip} trace {tid}: {len(rows)} programs vs {len(labels)} labels")
                        ok = False
                        break
                    for lab, (_, st, en) in zip(labels, rows):
                        rec.setdefault(lab, (st, en))  # first occurrence (round 0) per label
                        rec.setdefault("_all", []).append((lab, st, en))
                if ok:
                    replays.append(rec)
        out[chip] = replays
    return out


def summarize_gaps(replays_by_chip, mhz=1350.0):
    """Medians (us) over replays and chips of the hand-off path A -> K1 (or k1)."""
    import statistics

    vals = {}
    for chip, replays in replays_by_chip.items():
        for rec in replays:
            if "A" not in rec:
                continue
            a_end = rec["A"][1]
            for lab, (st, en) in rec.items():
                if lab == "_all":
                    continue
                vals.setdefault(f"start({lab})-end(A)", []).append((st - a_end) / mhz)
                vals.setdefault(f"dur({lab})", []).append((en - st) / mhz)
            # round-to-round period for multi-round chains: start of the same label in consecutive rounds
            seq = {}
            for lab, st, en in rec.get("_all", []):
                seq.setdefault(lab, []).append(st)
            for lab, sts in seq.items():
                if len(sts) > 2:
                    d = [(b - a) / mhz for a, b in zip(sts, sts[1:])]
                    vals.setdefault(f"period({lab})", []).append(statistics.median(d))
    return {k: round(statistics.median(v), 3) for k, v in sorted(vals.items())}, {k: len(v) for k, v in vals.items()}


# ------------------------------------------------------------------------------------------------------------------
# pytest gate tests
# ------------------------------------------------------------------------------------------------------------------
TRACE_REGION = 64 << 20


def _open(shape, num_cqs=1):
    return ttnn.open_mesh_device(ttnn.MeshShape(*shape), trace_region_size=TRACE_REGION, num_command_queues=num_cqs)


@pytest.fixture(scope="module", params=[(1, 1)], ids=["1x1"])
def harness(request):
    mesh = _open(request.param)
    h = Harness(mesh)
    yield h
    h.close()
    ttnn.close_mesh_device(mesh)


@pytest.mark.skipif(not HAS_OPS, reason="ttnn.experimental.subdevice_signal not built")
def test_signal_wait_1000_replays(harness):
    r = harness.run_variant("sw", "W D A S K E", 1000)
    assert r["total_viol"] == 0 and r["x_ok"] and r["e_ok"], r


@pytest.mark.skipif(not HAS_OPS, reason="ttnn.experimental.subdevice_signal not built")
def test_ping_pong_4_rounds(harness):
    r = harness.run_variant("pp4", "W w D A S K E s", 250, nrounds=4)
    assert r["total_viol"] == 0 and r["x_ok"] and r["e_ok"], r


def test_negative_control_detects_violations(harness):
    # consumer issued before the producer, no hand-off: it reads x while D spins -> every element violates
    r = harness.run_variant("neg", "D K E A", 100)
    assert r["total_viol"] == r["viol_elems_max"] * r["ndev"], r


# ------------------------------------------------------------------------------------------------------------------
# script mode
# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="1x1")
    ap.add_argument("--replays", type=int, default=1000)
    ap.add_argument("--delay", type=int, default=100_000)
    ap.add_argument("--tiles", type=int, default=256)
    ap.add_argument("--timeout-cycles", type=int, default=0)
    ap.add_argument("--variants", required=True, help="comma list of name:order[:rounds]")
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--cqs", type=int, default=1, help="2: also accepts 2-CQ variants name:2cq:<cq0 order>|<cq1 order>[:rounds]")
    a = ap.parse_args()
    shape = tuple(int(v) for v in a.mesh.split("x"))
    with Watchdog(a.timeout_s, "open"):
        mesh = _open(shape, a.cqs)
    log(f"mesh {shape} grid={mesh.compute_with_storage_grid_size()} has_ops={HAS_OPS}")
    h = Harness(mesh, numel_tiles=a.tiles, delay_cycles=a.delay, timeout_cycles=a.timeout_cycles)
    out = []
    rc = 0
    try:
        for spec in a.variants.split(","):
            parts = spec.split(":")
            try:
                if parts[1] == "2cq":
                    prod, cons = parts[2].split("|")
                    nr = int(parts[3]) if len(parts) > 3 else 1
                    out.append(h.run_2cq(parts[0], prod, cons, a.replays, nrounds=nr, timeout_s=a.timeout_s))
                    continue
                name, order = parts[0], parts[1]
                nr = int(parts[2]) if len(parts) > 2 else 1
                out.append(h.run_variant(name, order, a.replays, nrounds=nr, timeout_s=a.timeout_s))
            except Exception:
                traceback.print_exc()
                rc = 1
    finally:
        h.close()
        ttnn.close_mesh_device(mesh)
    log("SUMMARY " + json.dumps([{k: r[k] for k in ("variant", "order", "rounds", "total_viol", "x_ok", "e_ok", "us_per_replay")} for r in out]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
