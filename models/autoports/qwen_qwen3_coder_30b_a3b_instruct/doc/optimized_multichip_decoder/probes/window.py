# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Re-derive a layer window from an op profile, check its invariants, print it.

Every published op-level figure in ``README.md`` and ``work_log.md`` is a line
of this script's output. It exists because the defect that has cost this model a
review in all three previous stages is a number in prose that no artifact
produces -- most sharply in stage 03, where the published decode window was one
full layer plus a two-op tail and every derived figure moved when it was fixed
(``../multichip_decoder/work_log.md`` section 13).

So the window is *computed*, not transcribed, and two structural invariants are
asserted before anything is printed:

* a decode layer contains exactly **two** ``ReduceScatterMinimalAsync`` and
  **two** ``AllGatherAsync`` -- the layer's two RS+AG all-reduces;
* it starts at the ``InterleavedToSharded`` that feeds ``input_layernorm`` and
  ends at its own residual ``BinaryNg``.

Usage::

    python window.py <ops_perf_results.csv[.gz]> [decode|prefill] [--device N]

Timings are ``DEVICE KERNEL DURATION [ns]``, which is the column whose sum over
stage 03's published window reproduces its 414.661 us; ``DEVICE FW DURATION``
over the same rows reads 490.339 and is not what any published figure uses.
"""
import csv
import gzip
import sys

KERNEL = "DEVICE KERNEL DURATION [ns]"


def load(path, device):
    op = gzip.open if str(path).endswith(".gz") else open
    rows = [r for r in csv.DictReader(op(path, "rt")) if r["DEVICE ID"] == str(device)]
    rows.sort(key=lambda r: int(r["HOST START TS"]))
    return rows


def shape(r, i):
    try:
        return "x".join(r[f"INPUT_{i}_{d}_PAD[LOGICAL]"] for d in "WZYX")
    except KeyError:
        return "-"


def windows(rows, mode):
    """Return ``[(start, end_inclusive)]`` for the last two layer iterations.

    Derived from the op stream, never from a transcribed row number:

    * a layer **ends** at the first ``BinaryNg`` at or after its second
      ``AllGatherAsync`` -- at or after, not exactly one past, because the
      all-reduce's persistent output buffer is cloned out before the residual
      add and a ``CloneOperation`` sits between them;
    * a layer **starts** at the first ``InterleavedToSharded`` after the
      *previous* layer's end -- the reshard that feeds ``input_layernorm``.

    Walking the start forwards from the previous end rather than backing a fixed
    op count off this one matters: ``profile_layer.py decode`` emits one-off
    setup ops between the priming prefill and the first decode iteration.
    """
    ags = [i for i, r in enumerate(rows) if r["OP CODE"].startswith("AllGatherAsync")]
    assert len(ags) % 2 == 0 and ags, f"expected an even, non-zero number of all-gathers; got {len(ags)}"

    def first(prefix, i):
        for j in range(i, len(rows)):
            if rows[j]["OP CODE"].startswith(prefix):
                return j
        raise AssertionError(f"no {prefix} at or after row {i}")

    ends = [first("BinaryNg", ags[2 * j + 1] + 1) for j in range(len(ags) // 2)]
    # ``decode`` primes the program cache with a prefill pass, which also carries
    # two all-gathers, so the two iterations of interest are the last two -- and
    # the pass before each of them is what fixes its start.
    assert len(ends) >= 3, "need a priming pass plus two iterations"
    return [(first("InterleavedToSharded", ends[j - 1] + 1), ends[j]) for j in (len(ends) - 2, len(ends) - 1)]


def check(rows, lo, hi):
    win = rows[lo : hi + 1]
    rs = sum(1 for r in win if r["OP CODE"].startswith("ReduceScatterMinimalAsync"))
    ag = sum(1 for r in win if r["OP CODE"].startswith("AllGatherAsync"))
    assert rs == 2 and ag == 2, f"rows {lo}-{hi}: {rs} reduce-scatter and {ag} all-gather, expected 2 and 2"
    assert win[-1]["OP CODE"].startswith("BinaryNg"), f"row {hi} is {win[-1]['OP CODE']}, expected the residual add"
    assert win[0]["OP CODE"].startswith(
        ("InterleavedToSharded", "LayerNorm")
    ), f"row {lo} is {win[0]['OP CODE']}, expected the layer's first norm op"


def main():
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else "decode"
    device = 0
    if "--device" in sys.argv:
        device = int(sys.argv[sys.argv.index("--device") + 1])

    rows = load(path, device)
    wins = windows(rows, mode)
    for lo, hi in wins:
        check(rows, lo, hi)
    for lo, hi in wins:
        tot = sum(int(r[KERNEL]) for r in rows[lo : hi + 1])
        print(f"P|device {device}  rows {lo}-{hi}  {hi - lo + 1} ops  {tot / 1000:.3f} us")

    lo, hi = wins[-1]
    print(f"P|--- published window: device {device}, rows {lo}-{hi} (last iteration) ---")
    for i in range(lo, hi + 1):
        r = rows[i]
        print(
            f"P|{i:5d} {int(r[KERNEL]) / 1000:8.3f} c{r['CORE COUNT']:>3} {r['OP CODE'][:40]:42s}"
            f" in0={shape(r, 0)} in1={shape(r, 1)} {r['INPUT_0_MEMORY'][:22]}"
        )
    print(f"P|total {sum(int(rows[i][KERNEL]) for i in range(lo, hi + 1)) / 1000:.3f} us over {hi - lo + 1} ops")

    # per-device totals for the same window size, so "slowest die" is a measurement
    for d in range(4):
        try:
            rd = load(path, d)
        except Exception:
            break
        if not rd:
            break
        w = windows(rd, mode)[-1]
        print(
            f"P|device {d} last-iteration window rows {w[0]}-{w[1]}: {sum(int(rd[i][KERNEL]) for i in range(w[0], w[1] + 1)) / 1000:.3f} us"
        )


if __name__ == "__main__":
    main()
