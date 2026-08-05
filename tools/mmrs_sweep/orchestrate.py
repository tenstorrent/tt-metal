#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Resumable orchestrator for the MinimalMatmulStridedReduceScatterAsync sweep.

    python tools/mmrs_sweep/orchestrate.py <shape> <stage> [batch_size]

Runs test_mmrs_sweep in small batches under the on-device profiler, reads per-config device time from
the raw profiler CSV, and appends to a persistent JSONL. Interrupt and rerun at any time: completed
config indices are remembered per (shape, stage).

Deliberately does NOT use tracy. The capture-release daemon wedges on large traces and can orphan
holding the device; the raw CSV carries everything needed.

A fused CCL op can hang the device outright, which no in-process handler can catch. The defence is
layered: TT_METAL_OPERATION_TIMEOUT_SECONDS turns a hung dispatch into an exception the test records
before aborting; batches stay small so a hang costs little; the completed prefix of a dead batch is
salvaged from the manifest; and a batch that dies leaves its child output on disk to explain why.

This orchestrator never resets the device. It runs as one tenant among others on a shared galaxy, and
device health belongs to the broker, which gates after every job and can refuse work while degraded. A
bare `tt-smi -r` from here would reset all 32 chips behind the broker's back and out from under
whatever else is queued. On a batch that will not run, stop and report; let a human decide.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.mmrs_sweep.space import cfg_key, enumerate_configs  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.expanduser("~/.tt-buddy/mmrs_sweep")
TEST = "tests/nightly/tg/ccl/test_mmrs_sweep.py"
RAW_CSV = os.path.join(REPO, "generated/profiler/.logs/profile_log_device.csv")

REPS, WARMUP = 5, 2
# Per-op dispatch timeout. The ops here are microseconds, so 20s is a wide margin; the check is on
# lack of dispatch progress rather than wall time, so a slow-but-progressing op will not trip it.
OP_TIMEOUT = "20"
WALL_TIMEOUT = 900  # backstop for hangs the dispatch timeout misses (host-side compile/link)
MAX_ATTEMPTS = 2  # a hang here is often intermittent, so retry once before giving up on a config


def device_durations_us(csv_path):
    """Ordered per-op device durations in microseconds.

    A `run host ID` in the raw CSV identifies one (op, device) pair, not one op: an op dispatched to
    a 32-device mesh appears under 32 distinct ids. Each device also timestamps against its own
    counter, so cross-device timestamps cannot be compared directly. Hence: duration per (device, op)
    = max(FW ZONE_END) - min(FW ZONE_START) over that device's cores; ops are then matched across
    devices by their rank within each device, and the op's wall clock is the slowest device.

    Cycles come from the CHIP_FREQ in the CSV header -- Blackhole is 1350 MHz, so treating a cycle as
    a nanosecond would overstate every number by a third.
    """
    lines = open(csv_path).read().splitlines()
    if len(lines) < 3:
        return []
    freq_mhz = 1000.0
    for tok in lines[0].split(","):
        if "CHIP_FREQ" in tok:
            freq_mhz = float(tok.split(":")[1])
    hdr = [h.strip() for h in lines[1].split(",")]
    ix = {h: i for i, h in enumerate(hdr)}
    zi, ti, ri, ty, di = (
        ix["zone name"],
        ix["time[cycles since reset]"],
        ix["run host ID"],
        ix["type"],
        ix["PCIe slot"],
    )
    starts, ends = {}, {}
    for ln in lines[2:]:
        f = ln.split(",")
        if len(f) <= max(zi, ti, ri, ty, di) or not f[zi].strip().endswith("-FW"):
            continue
        rid, t = f[ri].strip(), f[ti].strip()
        if not rid or not t:
            continue
        key = (f[di].strip(), int(rid))
        t = int(t)
        if f[ty].strip() == "ZONE_START":
            starts[key] = min(starts.get(key, t), t)
        elif f[ty].strip() == "ZONE_END":
            ends[key] = max(ends.get(key, t), t)
    per_device = {}
    for key in sorted(k for k in starts if k in ends):
        dev, rid = key
        per_device.setdefault(dev, []).append((rid, (ends[key] - starts[key]) / freq_mhz))
    if not per_device:
        return []
    counts = {len(v) for v in per_device.values()}
    if len(counts) != 1:
        # Devices disagree on how many ops they saw; ranks are not comparable, so report nothing
        # rather than align the wrong ops.
        return []
    return [max(v[k][1] for v in per_device.values()) for k in range(counts.pop())]


def main():
    shape = sys.argv[1] if len(sys.argv) > 1 else "small"
    stage = sys.argv[2] if len(sys.argv) > 2 else "structural"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    os.makedirs(OUT, exist_ok=True)
    results_path = os.path.join(OUT, f"results_{shape}_{stage}.jsonl")
    done_path = os.path.join(OUT, f"done_{shape}_{stage}.json")
    stuck_path = os.path.join(OUT, f"stuck_{shape}_{stage}.json")

    configs = enumerate_configs(shape, stage)
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    stuck = json.load(open(stuck_path)) if os.path.exists(stuck_path) else {}

    todo = [(i, c) for i, c in enumerate(configs) if cfg_key(c) not in done]
    # Batch within a single packet size: it is applied through device_params at mesh open, so a batch
    # that mixed payloads would need to reopen the mesh mid-run.
    queue = deque()
    cur = []
    for item in todo:
        if cur and (len(cur) >= batch_size or cur[0][1]["packet"] != item[1]["packet"]):
            queue.append((cur, 0))
            cur = []
        cur.append(item)
    if cur:
        queue.append((cur, 0))

    print(
        f"{shape}/{stage}: {len(configs)} configs, {len(done)} done, {len(todo)} todo "
        f"in {len(queue)} batches (<= {batch_size} each), op_timeout={OP_TIMEOUT}s",
        flush=True,
    )

    t0, ncfg, batch_no, barren = time.time(), 0, 0, 0
    while queue:
        batch, attempt = queue.popleft()
        batch_no += 1
        packet = batch[0][1]["packet"]
        tag = os.path.join(OUT, f"batch_{shape}_{stage}_{batch_no}")
        cfg_path, man_path = tag + "_cfgs.json", tag + "_man.json"
        json.dump([c for _i, c in batch], open(cfg_path, "w"))
        for p in (man_path, RAW_CSV):
            if os.path.exists(p):
                os.remove(p)

        env = dict(
            os.environ,
            MMRS_CONFIGS=cfg_path,
            MMRS_MANIFEST=man_path,
            MMRS_REPS=str(REPS),
            MMRS_WARMUP=str(WARMUP),
            MMRS_PACKET=str(packet),
            TT_METAL_DEVICE_PROFILER="1",
            TT_METAL_OPERATION_TIMEOUT_SECONDS=OP_TIMEOUT,
            TT_METAL_HOME=REPO,
            PYTHONPATH=REPO,
        )
        clean, rc, child_out = False, None, ""
        try:
            r = subprocess.run(
                ["python", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", TEST],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=WALL_TIMEOUT,
                start_new_session=True,
            )
            rc, child_out = r.returncode, (r.stdout or "") + (r.stderr or "")
            clean = rc == 0 and "SWEEP_DONE" in child_out
        except subprocess.TimeoutExpired as e:
            rc = "wall-timeout"
            child_out = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""
        if not clean:
            # Always keep the child's output for a batch that did not finish: the exception that killed
            # it is the only explanation of why, and discarding it turns a diagnosable failure into a
            # config that silently lands in `stuck`.
            log_path = tag + "_fail.log"
            with open(log_path, "w") as f:
                f.write(f"rc={rc}\n\n{child_out}")
            print(f"  !! batch did not finish (rc={rc}); child output -> {log_path}", flush=True)
            for line in [ln for ln in child_out.splitlines() if "Error" in ln or "FAILED" in ln][:3]:
                print(f"     {line[:160]}", flush=True)

        man = json.load(open(man_path)) if os.path.exists(man_path) else []

        # A clean finish means every recorded config is complete. Otherwise the last one is the config
        # that hung: keep the ones before it and requeue the rest.
        recorded = man if clean else man[:-1]
        ok_recs = [r for r in recorded if r["ok"]]
        durs = device_durations_us(RAW_CSV) if os.path.exists(RAW_CSV) else []
        # Anchor at the END of the op stream. Building the input tensors emits a handful of device ops
        # first, and their count is not worth predicting; the configs are the trailing ops, in order.
        # A config that failed before dispatch contributes none, so only ok records are counted.
        want = sum(m["nops"] * (1 + WARMUP + REPS) for m in ok_recs)
        if want and len(durs) >= want:
            tail = durs[len(durs) - want :]
            i = 0
            for m in ok_recs:
                chunk = m["nops"] * (1 + WARMUP + REPS)
                reps = tail[i + m["nops"] * (1 + WARMUP) : i + chunk]
                i += chunk
                # Sum the ops of one invocation (the unfused path is matmul + RS), then median.
                per_invoke = [sum(reps[j : j + m["nops"]]) for j in range(0, len(reps), m["nops"])]
                m["us"] = round(statistics.median(per_invoke), 3)
        elif ok_recs:
            # Refuse to guess an alignment. Configs stay done -- rerunning hits the same mismatch --
            # but carry no timing, and the discrepancy is logged rather than papered over.
            print(f"  !! timing unusable: {len(durs)} ops in CSV, need >= {want}", flush=True)

        done_now = {cfg_key(m) for m in recorded}
        requeue = [(i, c) for i, c in batch if cfg_key(c) not in done_now]
        if requeue and attempt + 1 < MAX_ATTEMPTS:
            queue.append((requeue, attempt + 1))
        elif requeue:
            for _i, c in requeue:
                stuck[cfg_key(c)] = c
                done.add(cfg_key(c))
            json.dump(stuck, open(stuck_path, "w"))

        if recorded:
            with open(results_path, "a") as f:
                for m in recorded:
                    f.write(json.dumps(m) + "\n")
                    ncfg += 1
            done |= done_now
            json.dump(sorted(done), open(done_path, "w"))

        print(
            f"batch {batch_no}: {len(batch)} cfgs packet={packet}{'' if clean else ' INCOMPLETE'} | "
            f"recorded={len(recorded)} requeue={len(requeue) if attempt + 1 < MAX_ATTEMPTS else 0} "
            f"stuck={len(stuck)} | done={len(done)}/{len(configs)} queue={len(queue)} "
            f"elapsed={(time.time() - t0) / 60:.1f}min",
            flush=True,
        )

        # A batch recording nothing at all failed before its first config even reported -- that is the
        # mesh fixture or the process itself, not any one config. Two in a row means the device or the
        # environment is not fit, and continuing would grind the whole space into `stuck` while
        # learning nothing. Stop and leave the queue intact so a later rerun resumes.
        barren = 0 if recorded else barren + 1
        if barren >= 2:
            print(
                f"ORCH_ABORT {shape}/{stage}: {barren} consecutive batches recorded no configs. "
                f"See {tag}_fail.log. Device or environment is unfit; not continuing.",
                flush=True,
            )
            return

    print(
        f"ORCH_DONE {shape}/{stage} done={len(done)}/{len(configs)} recorded={ncfg} "
        f"stuck={len(stuck)} elapsed={(time.time() - t0) / 60:.1f}min",
        flush=True,
    )


if __name__ == "__main__":
    main()
