# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""L1 contention and per-stream L1 demand per TILE_LOOP row, from the two Wormhole L1 mux group counter dumps.

usage: python helpers/perf/l1_contention.py <group 0 run dir> [<group 1 run dir>] [-o out.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Plain pandas on the CSVs a run wrote, so it runs without the device stack (schema.py names, repeated here).
MARKER = "marker"
LOOP_FACTOR_COLUMN = "loop_factor"
TILE_CNT_COLUMN = "tile_cnt"

# Wormhole TDMA client port -> (request counter, grant counter). Mux group 0: unpacker 0 requester 0, port 1 (unpacker 1
# requester 0 and packer 1), packers 2 and 3 (shared with the RISCs); group 1: packer 0, unpacker requesters 1-3.
PORTS = {
    0: ("L1.L1_0_UNPACKER_0", "L1.L1_0_UNPACKER_0_GRANT"),
    1: ("L1.L1_0_UNPACKER_1_ECC_PACK1", "L1.L1_0_PORT1_GRANT"),
    2: ("L1.L1_0_TDMA_BUNDLE_0_RISC", "L1.L1_0_TDMA_BUNDLE_0_GRANT"),
    3: ("L1.L1_0_TDMA_BUNDLE_1_TRISC", "L1.L1_0_TDMA_BUNDLE_1_GRANT"),
    8: ("L1.L1_1_TDMA_PACKER_2", "L1.L1_1_PORT8_GRANT"),
    9: ("L1.L1_1_EXT_UNPACKER_1", "L1.L1_1_EXT_UNPACKER_1_GRANT"),
    10: ("L1.L1_1_EXT_UNPACKER_2", "L1.L1_1_EXT_UNPACKER_2_GRANT"),
    11: ("L1.L1_1_EXT_UNPACKER_3", "L1.L1_1_EXT_UNPACKER_3_GRANT"),
}
UNPACK_PORTS = (0, 1, 9, 10, 11)
PACK_PORTS = (8, 1, 2, 3)
BANKS = 16
RUN_TYPES = ("L1_TO_L1", "UNPACK_ISOLATE", "PACK_ISOLATE")


def _value(frame, run_type, counter):
    for col in (f"{run_type}_{counter}", f"{run_type}_mean({counter})"):
        if col in frame.columns:
            return frame[col]
    return None


def _module_frame(counters_csv: Path, timing_csv: Path) -> pd.DataFrame:
    """Per TILE_LOOP row: words granted and cycles refused per tile on each captured port, and the run type times."""
    raw = pd.read_csv(counters_csv, low_memory=False)
    config = [
        c
        for c in raw.columns[: raw.columns.get_loc(MARKER)]
        if not c.startswith("TEXT_SIZE(")
    ]
    raw = raw[raw[MARKER] == "TILE_LOOP"]
    out = raw[config].copy()
    tiles = raw.get(LOOP_FACTOR_COLUMN, 1) * raw.get(TILE_CNT_COLUMN, 1)
    out["tiles"] = tiles
    for run_type in RUN_TYPES:
        for port, (req, gnt) in PORTS.items():
            r, g = _value(raw, run_type, req), _value(raw, run_type, gnt)
            if r is None or g is None:
                continue
            out[f"{run_type}:words{port}"] = g / tiles
            out[f"{run_type}:wait{port}"] = (r - g) / tiles
    timing = pd.read_csv(timing_csv, low_memory=False)
    timing = timing[timing[MARKER] == "TILE_LOOP"]
    for run_type in RUN_TYPES:
        col = f"mean({run_type})"
        if col in timing.columns:
            out = out.merge(
                timing[config + [col]].rename(columns={col: run_type}),
                on=config,
                how="left",
            )
    return out


def _merge_groups(frames):
    """The group 0 and group 1 captures of one module side by side (same configs, disjoint ports)."""
    base = frames[0]
    config = [
        c for c in base.columns if ":" not in c and c not in RUN_TYPES + ("tiles",)
    ]
    for other in frames[1:]:
        extra = [c for c in other.columns if ":" in c and c not in base.columns]
        base = base.merge(other[config + extra], on=config, how="left")
    return base


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Contention in L1_TO_L1 (refused cycles per tile, and the part the isolates do not have on their own) and the
    words per tile of each stream, with the load both would put on port 1 and the banks at their isolate pace.
    """

    def present(run_type, kind, ports):
        return [
            f"{run_type}:{kind}{p}" for p in ports if f"{run_type}:{kind}{p}" in frame
        ]

    out = frame.copy()
    l2l_wait = out[present("L1_TO_L1", "wait", PORTS)].sum(axis=1, min_count=1)
    iso_wait = (
        out[present("UNPACK_ISOLATE", "wait", PORTS)]
        .sum(axis=1)
        .add(out[present("PACK_ISOLATE", "wait", PORTS)].sum(axis=1), fill_value=0)
    )
    out["l2l_wait_per_tile"] = l2l_wait
    out["l2l_cross_wait_per_tile"] = l2l_wait - iso_wait
    out["unpack_words_per_tile"] = out[
        present("UNPACK_ISOLATE", "words", UNPACK_PORTS)
    ].sum(axis=1, min_count=1)
    out["pack_words_per_tile"] = out[present("PACK_ISOLATE", "words", PACK_PORTS)].sum(
        axis=1, min_count=1
    )
    if {"UNPACK_ISOLATE", "PACK_ISOLATE"} <= set(out.columns):

        def words_per_cycle(run_type, ports):
            words = out[present(run_type, "words", ports)].sum(axis=1)
            return words * out["tiles"] / out[run_type]

        def load(ports):
            return words_per_cycle("UNPACK_ISOLATE", ports) + words_per_cycle(
                "PACK_ISOLATE", ports
            )

        # above 1: the two streams cannot both run at their isolate pace (port 1 takes one word a cycle)
        out["saturation_port1_load"] = load((1,))
        out["saturation_bank_load"] = load(PORTS) / BANKS
    return out


def build(run_dirs) -> pd.DataFrame:
    run_dirs = [Path(d) for d in run_dirs]
    rows = []
    for counters_csv in sorted(run_dirs[0].glob("*/*.counters.csv")):
        module = counters_csv.parent.name
        frames = []
        for d in run_dirs:
            c, t = d / module / f"{module}.counters.csv", d / module / f"{module}.csv"
            if c.is_file() and t.is_file():
                frames.append(_module_frame(c, t))
        if not frames:
            continue
        merged = summarize(_merge_groups(frames))
        merged.insert(0, "test_module", module)
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "runs",
        nargs="+",
        help="run dirs of the group 0 and (optionally) group 1 counter captures",
    )
    ap.add_argument("-o", "--output", default="l1_contention.csv")
    a = ap.parse_args(argv)
    out = build(a.runs)
    out.to_csv(a.output, index=False)
    print(f"{len(out)} rows -> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
