# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# Standalone Chronos-2 forecast CLI on a Tenstorrent device: CSV in, quantile
# CSV out.
#
# Input CSV (wide): one column per univariate series, one row per time step,
# oldest first. Header row = series ids. Empty / NaN cells are treated as
# missing (observed mask 0). An optional leading timestamp column can be
# skipped with --index-column.
#
# Output CSV (long): series_id, step (1..H), then one column per model quantile
# (config order, e.g. q0.01 ... q0.99).
#
# Example (from the tt-metal root):
#   python -m models.experimental.chronos2.demo.demo --checkpoint /path/to/chronos-2 \
#       --input series.csv --output forecast.csv --prediction-length 64

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

from models.experimental.chronos2.tt import (
    DEFAULT_PRECISION,
    DEVICE_OPTIONS,
    MAX_PREDICTION_LENGTH,
    SUPPORTED_PRECISIONS,
    create_backend,
)


def read_wide_csv(path: str, index_column: str | None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (series ids, values [B,T] float32, observed mask [B,T] float32)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"{path}: need a header row and at least one data row")
    header = [h.strip() for h in rows[0]]
    keep = [i for i, h in enumerate(header) if h != index_column]
    if not keep:
        raise ValueError(f"{path}: no series columns")
    ids = [header[i] for i in keep]
    vals = np.full((len(keep), len(rows) - 1), np.nan, dtype=np.float32)
    for t, row in enumerate(rows[1:]):
        for b, i in enumerate(keep):
            cell = row[i].strip() if i < len(row) else ""
            if cell:
                vals[b, t] = np.float32(float(cell))
    mask = np.isfinite(vals).astype(np.float32)
    return ids, np.where(mask > 0, vals, np.float32(0.0)).astype(np.float32), mask


def write_quantile_csv(path: str, ids: list[str], quantiles: tuple, q: np.ndarray) -> None:
    """q: [B, H, Q] -> long CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series_id", "step"] + [f"q{qv:g}" for qv in quantiles])
        for b, sid in enumerate(ids):
            for h in range(q.shape[1]):
                w.writerow([sid, h + 1] + [f"{float(x):.7g}" for x in q[b, h]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "Chronos-2 TTNN forecast demo")
    ap.add_argument("--checkpoint", required=True, help="directory with config.json + model.safetensors")
    ap.add_argument("--config", default=None, help="config.json path (default: <checkpoint>/config.json)")
    ap.add_argument("--input", required=True, help="wide CSV, one column per series")
    ap.add_argument("--output", required=True, help="long quantile CSV to write")
    ap.add_argument("--prediction-length", type=int, default=MAX_PREDICTION_LENGTH)
    ap.add_argument("--precision", choices=SUPPORTED_PRECISIONS, default=DEFAULT_PRECISION)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument(
        "--trace-region-size",
        type=int,
        default=DEVICE_OPTIONS["trace_region_size"],
        help="bytes; 0 disables metal trace (eager execution)",
    )
    ap.add_argument("--index-column", default=None, help="name of a timestamp column to ignore")
    args = ap.parse_args(argv)

    import ttnn

    ids, values, mask = read_wide_csv(args.input, args.index_column)
    options = {"trace_region_size": int(args.trace_region_size)}
    device = ttnn.open_device(device_id=args.device_id, **options)
    try:
        backend = create_backend(args.checkpoint, args.config, device, precision=args.precision, device_options=options)
        t0 = time.perf_counter()
        out = backend.forecast(values, mask, args.prediction_length)["quantiles"]
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        backend.release()
    finally:
        ttnn.close_device(device)
    write_quantile_csv(args.output, ids, backend.cfg.quantiles, out)
    print(
        f"wrote {args.output}: {len(ids)} series x {out.shape[1]} steps x {out.shape[2]} quantiles "
        f"({args.precision}, first call incl. compile {dt:.2f} s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
