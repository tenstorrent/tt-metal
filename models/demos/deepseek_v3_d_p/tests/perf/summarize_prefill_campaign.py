"""One summary table for a whole campaign: throughput, latency, and the per-layer budgets.

Exists because deriving these by hand invites two mistakes that have both actually been made:

1. **Warm-up.** `analyze_prefill_throughput.py` discards 4 intervals by default; every published table here uses 8.
   It is not cosmetic. A multi-chunk request's chunk-to-chunk interval GROWS as the KV cache deepens,
   so the median moves with how many early intervals you drop -- 1rank@102,400 reads 226.7 ms at 4
   and 304.8 ms at 8. Comparing a 4 against a published 8 looks like a 35% win that is not there.

2. **`analyze_prefill_latency.py` cannot be trusted alone.** Its number comes from E2E_CLOCK, whose
   `last_compute_end` is stamped at DRAIN, so it can absorb a multi-second shutdown stall. Measured:
   agrees with the interval reconstruction on 5 of 6 cells, then reads 9.122 s against a
   reconstructed 4.102 s on pp4_102400_ttft -- a 20-chunk request, so this is NOT the single-chunk
   artifact the docs originally described. This script reports BOTH and flags any disagreement.

Usage: summarize_prefill_campaign.py <results_dir> [profile_dir]
  results_dir  the per-host matrix output, e.g. mistral4_perf_$(hostname)
  profile_dir  optional, e.g. mistral4_perf_profile_$(hostname), for the layer budgets
"""
import glob
import os
import re
import statistics
import subprocess
import sys

ISLS = [5120, 25600, 102400, 261120]
CFGS = ["1rank", "pp4"]
CHUNK = 5120
WARMUP = 8
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

CHUNK_START = re.compile(r"\[pp rank (\d)\] CHUNK_START c=(\d+) compute_start=([0-9.]+)")


def _starts(log):
    """rank -> [compute_start] in chunk order. Empty dict if the log has none."""
    out = {}
    try:
        fh = open(log, errors="replace")
    except OSError:
        return out
    with fh:
        for line in fh:
            m = CHUNK_START.search(line)
            if m:
                out.setdefault(int(m.group(1)), []).append((int(m.group(2)), float(m.group(3))))
    return {r: [t for _, t in sorted(v)] for r, v in out.items()}


def throughput(log):
    """Steady-state tok/s from the LAST rank's chunk-to-chunk interval, warmup discarded."""
    st = _starts(log)
    if not st:
        return None, None
    xs = st[max(st)]
    d = [b - a for a, b in zip(xs, xs[1:])]
    if len(d) <= WARMUP:
        return None, None
    med = statistics.median(d[WARMUP:])
    return med * 1000.0, CHUNK / med


def latency_reconstructed(log):
    """Rebuilt from the timeline: last rank's span + one trailing chunk + pipeline fill.

    Independent of E2E_CLOCK, which is what makes it a usable cross-check.
    """
    st = _starts(log)
    if not st:
        return None
    xs = st[max(st)]
    if len(xs) < 2:
        return None
    fill = xs[0] - min(st[min(st)])
    return (xs[-1] - xs[0]) + (xs[-1] - xs[-2]) + fill


def latency_reported(log):
    """Whatever analyze_prefill_latency.py says, for comparison only."""
    try:
        out = subprocess.run(
            [PY, os.path.join(HERE, "analyze_prefill_latency.py"), log], capture_output=True, text=True, timeout=180
        ).stdout
    except Exception:
        return None
    m = re.search(r"in ([0-9]+\.[0-9]+)s", out)
    return float(m.group(1)) if m else None


def main():
    res = sys.argv[1]
    prof = sys.argv[2] if len(sys.argv) > 2 else None

    def pick(cfg, isl, mode):
        """The per-host results directory is the only source read.

        There used to be a fallback to the driver's staging directory. It is deliberately gone: that
        copy is overwritten by the next campaign, so reading it silently mixes runs -- and this script
        no longer sits next to the drivers, so the fallback path was wrong as well as unwise.
        """
        return os.path.join(res, f"{cfg}_{isl}_{mode}", "runner.log")

    print(f"\n=== campaign summary: {res} ===")
    print(f"(throughput: last rank median chunk interval, first {WARMUP} discarded)\n")
    print(f"{'ISL':>8} {'1rank ms':>10} {'1rank tok/s':>12} {'PP=4 ms':>10} {'PP=4 tok/s':>12} {'ratio':>7}")
    for isl in ISLS:
        row, tks = [], []
        for cfg in CFGS:
            ms, tk = throughput(pick(cfg, isl, "thru"))
            row += [f"{ms:.1f}" if ms else "-", f"{tk:,.0f}" if tk else "-"]
            tks.append(tk)
        ratio = f"{tks[1]/tks[0]:.2f}x" if all(tks) else "-"
        print(f"{isl:>8} {row[0]:>10} {row[1]:>12} {row[2]:>10} {row[3]:>12} {ratio:>7}")

    print(f"\n{'ISL':>8} {'1rank s':>10} {'PP=4 s':>10} {'speedup':>8}   note")
    for isl in ISLS:
        vals, flags = [], []
        for cfg in CFGS:
            log = pick(cfg, isl, "ttft")
            rec, rep = latency_reconstructed(log), latency_reported(log)
            vals.append(rec)
            # A gap here means E2E_CLOCK absorbed drain time; the reconstruction is the honest one.
            if rec and rep and abs(rep - rec) > max(0.5, 0.1 * rec):
                flags.append(f"{cfg}: analyze_ttft says {rep:.3f}s vs {rec:.3f}s reconstructed")
        sp = f"{vals[0]/vals[1]:.2f}x" if all(vals) and vals[1] else "-"
        a = f"{vals[0]:.3f}" if vals[0] else "-"
        b = f"{vals[1]:.3f}" if vals[1] else "-"
        print(f"{isl:>8} {a:>10} {b:>10} {sp:>8}   {'; '.join(flags) if flags else ''}")
    print("\n  (a flagged row means E2E_CLOCK absorbed drain time -- trust the reconstruction)")

    if prof and os.path.isdir(prof):
        print(f"\n=== per-layer budget: {prof} ===")
        for name, ranks in (("1rank_deep", [0]), ("pp4_deep", [0, 1, 2, 3])):
            for r in ranks:
                pat = os.path.join(prof, name, f"rank{r}", "reports", "*", "*", "ops_perf_results*.csv")
                hits = sorted(glob.glob(pat))
                if not hits:
                    continue
                out = subprocess.run(
                    [PY, os.path.join(HERE, "analyze_prefill_layer_budget.py"), hits[-1], f"{name} rank{r}"],
                    capture_output=True, text=True,
                ).stdout
                for line in out.splitlines():
                    if "COMPUTE per layer" in line:
                        print(f"  {name} rank{r}: {line.split(':')[-1].strip()}")


if __name__ == "__main__":
    main()
