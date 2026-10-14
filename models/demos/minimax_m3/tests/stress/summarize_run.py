# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Summarize a stress run directory (or a cold-start cycle directory) into <run>/summary.txt and stdout.

  python3 summarize_run.py <run_dir>
"""

import glob
import json
import os
import re
import sys

CHUNK_START = re.compile(r"\[pp rank (\d+)\] CHUNK_START")
END = re.compile(r"\[stress\] END (\{.*\})")
DRAM = re.compile(r"\[pp rank (\d+)\] DRAM per chip after KV alloc: (.*)")


def _lines(path):
    try:
        with open(path, errors="replace") as f:
            yield from f
    except OSError:
        return


def one_run(d, stages):
    chunks = [0] * stages
    dram = {}
    for line in _lines(os.path.join(d, "runner.log")):
        m = CHUNK_START.search(line)
        if m and int(m.group(1)) < stages:
            chunks[int(m.group(1))] += 1
        m = DRAM.search(line)
        if m:
            dram[int(m.group(1))] = m.group(2).strip()
    pushes, end = 0, None
    for line in _lines(os.path.join(d, "producer.log")):
        pushes += "[producer] push slot" in line
        m = END.search(line)
        if m:
            end = json.loads(m.group(1))
    out = [f"chunks started per rank: {'/'.join(map(str, chunks))}; producer pushes {pushes}"]
    if pushes and any(c != pushes for c in chunks):
        out.append(f"  ranks behind the producer: {[r for r, c in enumerate(chunks) if c != pushes]}")
    for r in sorted(dram):
        out.append(f"  rank {r} DRAM: {dram[r]}")
    if end:
        p = max(end.get("pushes", 0), 1)
        out.append(
            "producer END: " + " ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in end.items())
        )
        if "mid_start" in end:
            out.append(
                f"  mid-slab starts {end['mid_start'] / p:.0%}, ragged ends {end['ragged_end'] / p:.0%} of pushes"
            )
        if end.get("wall_s") and "idle_s" in end:
            out.append(f"  idle {end['idle_s'] / end['wall_s']:.0%} of the load window, {end['gaps']} gaps, "
                       f"{end['long_idles']} long idles, {end['tiny']} tiny requests")  # fmt: skip
    else:
        out.append("producer END: none (producer did not finish its schedule)")
    for hd in sorted(glob.glob(os.path.join(d, "hang_*"))):
        done = os.path.exists(os.path.join(hd, "CAPTURE_DONE"))
        why = next(_lines(os.path.join(hd, "why.txt")), "").strip()
        out.append(f"HANG capture {hd} ({'complete' if done else 'incomplete'}): {why}")
    return out


def main():
    d = os.path.abspath(sys.argv[1])
    info_path = next(p for p in (os.path.join(d, "run.json"), os.path.join(d, "c001", "run.json")) if os.path.exists(p))
    with open(info_path) as f:
        info = json.load(f)
    out = [f"run {d}", f"profile {info['profile']}, {info['stages']} stages, layers {info['layers']}"]
    cycles = os.path.join(d, "cycles.txt")
    if os.path.exists(cycles):
        rows = [line.split() for line in _lines(cycles) if line.strip()]
        bad = [" ".join(r) for r in rows if len(r) < 2 or r[1] != "ok"]
        out.append(f"cold-start cycles: {len(rows)}, ok {len(rows) - len(bad)}")
        out += [f"  {b}" for b in bad]
        for cd in sorted(glob.glob(os.path.join(d, "c[0-9][0-9][0-9]"))):
            if glob.glob(os.path.join(cd, "hang_*")):
                out += [f"[{os.path.basename(cd)}] {x}" for x in one_run(cd, info["stages"])]
    else:
        out += one_run(d, info["stages"])
    results = [line.strip() for line in _lines(os.path.join(d, "events.log")) if "RESULT" in line]
    out += results[-3:]
    text = "\n".join(out) + "\n"
    with open(os.path.join(d, "summary.txt"), "w") as f:
        f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
