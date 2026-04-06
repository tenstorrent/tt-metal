# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
import argparse, csv, os, math
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    xs, ys, ms, GB_s = [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["x"]))
            ys.append(int(row["y"]))
            ms.append(float(row["ms"]))
            GB_s.append(float(row["GB_s"]))
    return xs, ys, ms, GB_s


def to_grid(xs, ys, vals):
    maxx = max(xs)
    maxy = max(ys)
    grid = np.full((maxy + 1, maxx + 1), np.nan, dtype=float)
    for x, y, v in zip(xs, ys, vals):
        grid[y, x] = v
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="CSV with columns: x,y,ms,GB_s")
    ap.add_argument("--title", default="Unicast E2E Heatmaps – N300")
    args = ap.parse_args()

    xs, ys, ms, GB_s = load_csv(args.csv_path)
    ms_grid = to_grid(xs, ys, ms)
    GB_s_grid = to_grid(xs, ys, GB_s)

    # Latency heatmap
    plt.figure(figsize=(7, 5))
    im1 = plt.imshow(ms_grid, origin="lower", aspect="equal")
    plt.colorbar(im1, label="Latency (ms)")
    plt.title(args.title + " — Latency")
    plt.xlabel("Core X")
    plt.ylabel("Core Y")
    out1 = os.path.splitext(args.csv_path)[0] + "_latency.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=160)
    print(f"Wrote {out1}")

    # Throughput heatmap
    plt.figure(figsize=(7, 5))
    im2 = plt.imshow(GB_s_grid, origin="lower", aspect="equal")
    plt.colorbar(im2, label="Throughput (GB/s)")
    plt.title(args.title + " — Throughput")
    plt.xlabel("Core X")
    plt.ylabel("Core Y")
    out2 = os.path.splitext(args.csv_path)[0] + "_throughput.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
