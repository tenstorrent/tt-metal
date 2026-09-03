# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cut one producer pass out of a long-lived runner log, for `plot_pipeline_trace`.

The pipeline runner is a persistent server: one log accumulates every pass a producer ever pushed
at it, separated by however long the operator took between them. `plot_pipeline_trace` plots what it
is given, so on such a log the compute compresses into unreadable slivers between minutes of idle —
mechanically correct and analytically useless.

`PREFILL_SEND_SHUTDOWN=0` also means the chunk index never resets, so passes are not delimited by
`c=0`; they are delimited by the gaps. This groups on the chunk index instead (passes are a known,
fixed number of chunks) and emits only the lines of the pass asked for, which is all the plotter
reads.

    python -m models.demos.deepseek_v3_d_p.scripts.slice_pipeline_run run.log --chunks 11 -o one.log
"""

import argparse
import re
import sys

_CHUNK_START = re.compile(r"\[pp rank (\d+)\] CHUNK_START c=(\d+) compute_start=([\d.]+)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The three line kinds plot_pipeline_trace parses; everything else is noise to it.
_KEEP = re.compile(r"CHUNK_START c=|CHUNK_COMPUTE c=|E2E_CLOCK ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="runner log covering one or more producer passes")
    ap.add_argument("--chunks", type=int, default=11, help="chunks per pass (default 11)")
    ap.add_argument("--pass-index", type=int, default=-1, help="which pass; -1 = last (default)")
    ap.add_argument("-o", "--out", default="one_pass.log")
    args = ap.parse_args()

    lines = []
    with open(args.log, "rb") as handle:
        for raw in handle:
            text = _ANSI.sub("", raw.decode("utf8", "ignore"))
            if _KEEP.search(text):
                lines.append(text)

    starts = [(int(m.group(1)), int(m.group(2))) for line in lines for m in [_CHUNK_START.search(line)] if m]
    if not starts:
        sys.exit(f"no CHUNK_START lines in {args.log}")

    # Chunk indices are global and monotonic across passes, so pass p is [p*chunks, (p+1)*chunks).
    highest = max(c for _, c in starts)
    num_passes = (highest + 1) // args.chunks
    if num_passes == 0:
        sys.exit(f"only {highest + 1} chunks in {args.log}; fewer than one pass of {args.chunks}")
    index = args.pass_index if args.pass_index >= 0 else num_passes + args.pass_index
    lo, hi = index * args.chunks, (index + 1) * args.chunks

    kept = []
    for line in lines:
        m = re.search(r"c=(\d+)", line)
        # E2E_CLOCK carries no chunk index; it summarises the whole server lifetime, so dropping it
        # is what keeps the plot's x-axis the pass rather than the session.
        if m and lo <= int(m.group(1)) < hi:
            kept.append(line)

    with open(args.out, "w") as handle:
        handle.writelines(kept)
    print(f"wrote {args.out}: pass {index} of {num_passes} (chunks {lo}..{hi - 1}), {len(kept)} lines")


if __name__ == "__main__":
    main()
