# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Convert the harness's Parquet output to CSV so we can read it.

The harness writes Parquet, which isn't human-readable. Run this to get a CSV
we can open as text or in a spreadsheet:

    python -m accuracy.to_csv _csv_output/wh/exp.parquet   # one file
    python -m accuracy.to_csv _csv_output/wh               # whole dir

Each <name>.parquet becomes <name>.csv next to it. The CSV is just a view;
the Parquet stays the source of truth.
"""

import sys
from pathlib import Path

import pandas as pd


def convert(path: Path) -> Path:
    """Write <path>.csv next to a single .parquet file and return the csv path."""
    csv_path = path.with_suffix(".csv")
    df = pd.read_parquet(path)
    # Write bools as T/F, not pandas' True/False.
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].map({True: "T", False: "F"})
    # %.9g keeps full float32 precision without the extra float64 noise digits.
    df.to_csv(csv_path, index=False, float_format="%.9g")
    return csv_path


def main(args: list[str]) -> None:
    if not args:
        print(__doc__)
        raise SystemExit(2)

    for arg in args:
        target = Path(arg)
        files = sorted(target.glob("*.parquet")) if target.is_dir() else [target]
        for f in files:
            print(convert(f))


if __name__ == "__main__":
    main(sys.argv[1:])
