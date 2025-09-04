#!/usr/bin/env python3
import argparse, csv, os
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="CSV file with columns: bytes,ms,gbps")
    ap.add_argument("--title", default="Unicast E2E – N300")
    args = ap.parse_args()

    sizes_B, ms, gbps = [], [], []
    with open(args.csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sizes_B.append(float(row["bytes"]))
            ms.append(float(row["ms"]))
            gbps.append(float(row["gbps"]))

    sizes_MB = [b / 1e6 for b in sizes_B]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    ln1 = ax1.plot(sizes_MB, gbps, marker="o", color="tab:blue", label="Throughput (GB/s)")
    ln2 = ax2.plot(sizes_MB, ms, marker="s", color="tab:red", label="Latency (ms)")

    ax1.set_xlabel("Tensor size (MB)")
    ax1.set_ylabel("Throughput (GB/s)")
    ax2.set_ylabel("Latency (ms)")

    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_title(args.title)

    # One combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")

    out_png = os.path.splitext(args.csv_path)[0] + ".png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
