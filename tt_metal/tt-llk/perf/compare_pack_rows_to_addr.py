# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare raw LLK row-pack reports from Quasar and Blackhole in cycles per call."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FORMATS = {"Float16_b", "Float32", "Fp8_e4m3"}
ROWS = {1, 8, 15, 16, 17, 32, 33, 50, 64}
MEAN = "mean(PACK_ISOLATE)"
STD = "std(PACK_ISOLATE)"
KEYS = ["formats.output", "num_rows_to_pack"]
CONFIG = [
    "dest_acc",
    "dest_sync",
    "speed_of_light",
    "loop_factor",
    "tile_cnt",
    "formats.input_A",
    "formats.register_A",
    "formats.sfpu_src",
    "unpack_to_dest",
]


def load_report(path):
    if path.name.endswith(".post.csv"):
        raise ValueError(f"{path}: pass the raw .csv; .post.csv is already normalized")
    frame = pd.read_csv(path)
    missing = set(KEYS + CONFIG + ["marker", MEAN, STD]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.loc[frame.marker == "TILE_LOOP"].copy()
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate format/row-count measurements")
    expected = {(fmt, rows) for fmt in FORMATS for rows in ROWS}
    actual = set(frame[KEYS].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"{path}: expected the complete 27-case row-pack sweep")
    for column in [MEAN, STD, "loop_factor", "tile_cnt"]:
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f"{path}: invalid {column}")
    calls = frame.loop_factor * frame.tile_cnt
    if (calls == 0).any() or (frame[MEAN] == 0).any():
        raise ValueError(f"{path}: cycle counts and pack-call counts must be positive")
    # The profiler times the whole TILE_LOOP. Each iteration packs TILE_CNT
    # outputs; normalize both statistics once, before comparing architectures.
    frame[[MEAN, STD]] = frame[[MEAN, STD]].div(calls, axis=0)
    return frame


def compare(quasar, blackhole):
    joined = quasar.merge(
        blackhole, on=KEYS, suffixes=("_quasar", "_blackhole"), validate="1:1"
    )
    for column in CONFIG:
        if not (joined[f"{column}_quasar"] == joined[f"{column}_blackhole"]).all():
            raise ValueError(f"different {column} configurations across targets")
    result = joined[KEYS].rename(
        columns={"formats.output": "format", "num_rows_to_pack": "rows"}
    )
    for arch in ["quasar", "blackhole"]:
        result[f"{arch}_cycles_per_call"] = joined[f"{MEAN}_{arch}"]
        result[f"{arch}_std_cycles_per_call"] = joined[f"{STD}_{arch}"]
    result["quasar_speedup_vs_blackhole"] = (
        result.blackhole_cycles_per_call / result.quasar_cycles_per_call
    )
    return result.sort_values(["format", "rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quasar", required=True, type=Path, help="Raw Quasar CSV")
    parser.add_argument(
        "--blackhole", required=True, type=Path, help="Raw Blackhole CSV"
    )
    parser.add_argument("--output", type=Path, help="Also save the comparison as CSV")
    args = parser.parse_args()
    try:
        result = compare(load_report(args.quasar), load_report(args.blackhole))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.output:
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nSpeedup > 1 means Quasar uses fewer cycles.")


if __name__ == "__main__":
    main()
