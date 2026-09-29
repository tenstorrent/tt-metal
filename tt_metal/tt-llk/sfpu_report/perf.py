# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Before/after perf of the selected ops, on this machine's device.

Each side runs the family's perf module narrowed to the selected ops, speed of
light, ``L1_TO_L1`` and ``MATH_ISOLATE`` only. Iterations alternate base, head,
base, head, ... so drift hits both sides alike. The raw CSVs are compared with
the LLK perf gate's own comparer (``tt-llk/perf/regression_compare.py``), so this
report and the gate cannot disagree on what a regression is.
"""

import importlib.util
import shutil
from pathlib import Path

import runner
from detect import MODULES

RUN_TYPES = ("L1_TO_L1", "MATH_ISOLATE")

#: The two SFPLOADMACRO schedules: the default build, and the plain-loop
#: fallback that ``-DDISABLE_SFPLOADMACRO`` selects.
SCHEDULES = {"loadmacro": [], "no-loadmacro": ["--disable-sfploadmacro"]}


def _load_comparer():
    path = runner.TOOL_LLK / "perf" / "regression_compare.py"
    spec = importlib.util.spec_from_file_location("regression_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _latest_csv(side, module):
    stem = Path(module).stem
    csv = side.llk / "perf_data" / "latest" / stem / f"{stem}.csv"
    return csv if csv.exists() else None


def _args(family, ops, schedule):
    op_args = [a for op in ops for a in ("--op", op)]
    return [
        "-m",
        "perf",
        "--speed-of-light",
        *SCHEDULES[schedule],
        *op_args,
        MODULES[family],
    ]


_ENV = {"LLK_PERF_RUN_TYPES": ",".join(RUN_TYPES)}


def build(side, arch, family, ops, schedule, log, jobs=8):
    """Compile the side's variants once; every iteration reuses the ELFs."""
    runner.pytest(
        side,
        arch,
        ["--compile-producer", "-n", str(jobs), *_args(family, ops, schedule)],
        env=_ENV,
        log=log,
    )


def measure(side, arch, family, ops, schedule, out_csv, log):
    """One device run of the compiled variants; the raw CSV is copied to ``out_csv``."""
    shutil.rmtree(side.llk / "perf_data", ignore_errors=True)
    runner.pytest(
        side,
        arch,
        ["--compile-consumer", "-n", "1", *_args(family, ops, schedule)],
        env=_ENV,
        log=log,
    )
    csv = _latest_csv(side, MODULES[family])
    if csv is None:
        raise RuntimeError(
            f"no perf CSV from the {side.name} side for {family} {ops}; see {log}"
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(csv, out_csv)
    return out_csv


def sweep(
    base,
    head,
    arch,
    family,
    ops,
    out_dir,
    log,
    iterations=3,
    schedules=tuple(SCHEDULES),
    jobs=8,
):
    """Interleaved runs. Returns ``{schedule: {"base": [csv...], "head": [csv...]}}``."""
    runs = {}
    for schedule in schedules:
        for side in (base, head):
            build(side, arch, family, ops, schedule, log, jobs)
        runs[schedule] = {"base": [], "head": []}
        for i in range(1, iterations + 1):
            for side in (base, head):
                csv = out_dir / family / schedule / side.name / f"run_{i}.csv"
                runs[schedule][side.name].append(
                    measure(side, arch, family, ops, schedule, csv, log)
                )
    return runs


def compare(runs, threshold, min_cycles):
    """Per schedule: the gate comparer's verdict over the raw (whole-loop) CSVs."""
    comparer = _load_comparer()
    return {
        schedule: comparer.compare_runs(
            [str(p) for p in sides["head"]],
            [str(p) for p in sides["base"]],
            threshold=threshold,
            min_cycles=min_cycles,
            run_types=",".join(RUN_TYPES),
        )
        for schedule, sides in runs.items()
    }


#: Datums per SFPU iteration ("row" in bounty issues): one 32-lane vector.
ROWS_PER_TILE = 32


def _config(rec):
    return dict(rec["config"])


def _text_sizes(csvs):
    """{(op, fmt, dest_acc, approx, fast_mode): TEXT_SIZE(MATH_ISOLATE)} from raw CSVs."""
    import pandas as pd

    out = {}
    for path in csvs:
        df = pd.read_csv(path)
        if "TEXT_SIZE(MATH_ISOLATE)" not in df.columns:
            continue
        for _, r in df[df["marker"] == "TILE_LOOP"].iterrows():
            out[_row_key(r)] = int(r["TEXT_SIZE(MATH_ISOLATE)"])
    return out


def _row_key(cfg):
    get = cfg.get if hasattr(cfg, "get") else cfg.__getitem__
    return (
        str(get("mathop")).split(".")[-1],
        f'{get("formats.input_A")}->{get("formats.output")}',
        str(get("dest_acc")).split(".")[-1],
        str(get("approx_mode")).split(".")[-1],
        str(get("fast_mode")).split(".")[-1],
    )


def rows(runs, verdicts):
    """Report rows per schedule: one per variant, TILE_LOOP, cycles per tile."""
    out = {}
    for schedule, verdict in verdicts.items():
        base_sizes = _text_sizes(runs[schedule]["base"])
        head_sizes = _text_sizes(runs[schedule]["head"])
        table = {}
        for rec in verdict["records"]:
            if rec["marker"] != "TILE_LOOP":
                continue
            cfg = _config(rec)
            tiles = float(cfg.get("tile_cnt", 1)) * float(cfg.get("loop_factor", 1))
            key = _row_key(cfg)
            row = table.setdefault(
                key,
                {
                    "op": key[0],
                    "formats": key[1],
                    "dest_acc": key[2],
                    "approx": key[3],
                    "fast_mode": key[4],
                    "text_base": base_sizes.get(key),
                    "text_head": head_sizes.get(key),
                },
            )
            rt = rec["run_type"]
            row[rt] = {
                "base": rec["baseline"] / tiles,
                "head": rec["current"] / tiles,
                "delta": rec["delta"],
                "regression": rec["regression"],
                "improvement": rec["improvement"],
            }
        out[schedule] = sorted(
            table.values(),
            key=lambda r: (r["op"], r["formats"], r["dest_acc"], r["approx"]),
        )
    return out
