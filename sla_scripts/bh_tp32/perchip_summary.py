# Per-chip telemetry summary from tests/test_op_energy.py's OE_PERCHIP_LOG (JSON lines: [t, [[aiclk, W, temp] x 32]]).
# Averages the samples taken under load (mean chip power above --min-w) and prints each chip, each group of 8 (tt-smi
# order; a Galaxy tray is 8 chips), and the spread. If a few chips are systematically slower, they set the pace of
# every column collective and of the whole step.
#   python sla_scripts/bh_tp32/perchip_summary.py runs/perchip.jsonl [--min-w 100]
import argparse
import json
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--min-w", type=float, default=100.0)
    a = ap.parse_args()
    loaded = []
    for line in open(a.log):
        t, chips = json.loads(line)
        if len(chips) == 32 and statistics.mean(c[1] for c in chips) > a.min_w:
            loaded.append(chips)
    if not loaded:
        print("no loaded samples")
        return
    n = len(loaded)
    clk = [statistics.mean(s[i][0] for s in loaded) for i in range(32)]
    pw = [statistics.mean(s[i][1] for s in loaded) for i in range(32)]
    tmp = [statistics.mean(s[i][2] for s in loaded) for i in range(32)]
    print(f"{n} loaded samples")
    for i in range(32):
        end = "\n" if i % 4 == 3 else " | "
        print(f"{i:2d}: {clk[i]:6.0f} MHz {pw[i]:5.1f} W {tmp[i]:4.1f} C", end=end)
    for g in range(4):
        sl = slice(8 * g, 8 * g + 8)
        print(
            f"chips {8 * g}-{8 * g + 7}: {statistics.mean(clk[sl]):.0f} MHz, {statistics.mean(pw[sl]):.1f} W, "
            f"{statistics.mean(tmp[sl]):.1f} C"
        )
    lo = sorted(range(32), key=lambda i: clk[i])[:4]
    print(f"clock spread {min(clk):.0f}-{max(clk):.0f} MHz (mean {statistics.mean(clk):.0f}); slowest chips {lo}")
    # how often each chip is the slowest in a sample
    slow = [0] * 32
    for s in loaded:
        slow[min(range(32), key=lambda i: s[i][0])] += 1
    top = sorted(range(32), key=lambda i: -slow[i])[:4]
    print("slowest-in-sample counts: " + ", ".join(f"chip {i}: {slow[i]}" for i in top))


if __name__ == "__main__":
    main()
