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
import re
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


#: Perf tests without a ``mathop`` parameter that measure one op: op -> test.
TEST_OF_OP = {"SfpuElwdiv": "test_perf_eltwise_binary_sfpu_div"}

#: Binary ops whose second operand the test feeds as an input, while ttnn computes it
#: on the device before the kernel runs: op -> (the operand, what ttnn runs for it).
#: Main's cycles leave that work out, so a kernel that takes it over measures slower
#: here than it runs in ttnn.
DEVICE_COMPUTED_OPERAND = {"SfpuLogsigmoid": ("exp(-x)", "a negative and a fast exp")}


def _selection(family, ops, formats=()):
    """The ``-k`` expression that keeps ``ops``' variants, in ``formats`` when given.

    Not ``--op``: it keeps only tests with a MathOperation parameter, and
    test_perf_eltwise_binary_sfpu_div and the typecast module have none. Test ids read
    ``formats:<in>-><out>-...-mathop:<op>-...``.
    """
    terms = []
    if family != "typecast":  # the typecast module measures nothing else
        terms.append(" or ".join(TEST_OF_OP.get(op) or f"mathop:{op}- or mathop:{op}]" for op in ops))
    if formats:
        # Tests without a formats parameter (test_perf_eltwise_binary_sfpu_int_uniform,
        # typecast) run in their only format.
        terms.append(" or ".join(f"formats:{f}-" for f in formats) + " or not formats:")
    return " and ".join(f"({t})" for t in terms)


def _args(family, ops, schedule, formats=()):
    selection = _selection(family, ops, formats)
    return [
        "-m",
        "perf",
        "--speed-of-light",
        *SCHEDULES[schedule],
        *(["-k", selection] if selection else []),
        MODULES[family],
    ]


_ENV = {"LLK_PERF_RUN_TYPES": ",".join(RUN_TYPES)}


def build(side, arch, family, ops, schedule, log, jobs=8, formats=()):
    """Compile the side's variants once; every iteration reuses the ELFs.

    Returns the pytest exit code: ``runner.NO_TESTS_COLLECTED`` means no variant
    matched ``ops``.
    """
    return runner.pytest(
        side,
        arch,
        ["--compile-producer", "-n", str(jobs), *_args(family, ops, schedule, formats)],
        env=_ENV,
        log=log,
        check=False,  # a variant that does not build is simply not measured
    )


def measure(side, arch, family, ops, schedule, out_dir, log, formats=()):
    """One device run of the compiled variants; its raw CSV is copied into ``out_dir``.

    Returns the copy, or None when the run left no CSV. The CSV keeps its module's
    name: the gate comparer stamps each row's ``test_module`` from the file name and
    keys points by it, so runs of one module pool into one median only under one name.
    """
    shutil.rmtree(side.llk / "perf_data", ignore_errors=True)
    runner.pytest(
        side,
        arch,
        ["--compile-consumer", "-n", "1", *_args(family, ops, schedule, formats)],
        env=_ENV,
        log=log,
        check=False,  # the variants that ran are in the CSV; the rest are reported missing
    )
    csv = _latest_csv(side, MODULES[family])
    if csv is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy(csv, out_dir / csv.name))


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
    formats=(),
):
    """Interleaved runs. Returns ``{schedule: {"base": [csv...], "head": [csv...]}}``.

    A run that left no CSV is left out: its side's variants come out missing.
    """
    runs = {}
    for schedule in schedules:
        for side in (base, head):
            if build(side, arch, family, ops, schedule, log, jobs, formats) == runner.NO_TESTS_COLLECTED:
                return None  # no perf test covers these ops
        runs[schedule] = {"base": [], "head": []}
        for i in range(1, iterations + 1):
            for side in (base, head):
                run_dir = out_dir / family / schedule / side.name / f"run_{i}"
                csv = measure(side, arch, family, ops, schedule, run_dir, log, formats)
                if csv is not None:
                    runs[schedule][side.name].append(csv)
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


#: Config columns that are not part of what a variant is: the comparer's own stamp.
_NOT_IDENTITY = {"test_module"}

#: Shown in their own table columns, so never in a row's op label.
_SHOWN = ("mathop", "formats.input_A", "formats.output", "dest_acc", "approx_mode")

#: Sweep switches an op label names when a variant has one on: column -> label name.
_SWITCHES = {
    "fast_mode": "fast_mode",
    "broadcast_type": "unpack bcast",
    "sfpu_bcast_dim": "Dest bcast",
    "stable_sort": "stable_sort",
    "fused_sort": "fused_sort",
}

#: Values that mean "off".
_OFF = {"None_", "None", "No", "False"}


def _value(v):
    """One spelling per config value: a CSV cell and the comparer's group key for it
    differ in type (16 vs 16.0, numpy vs Python), and enums read ``FastMode.Yes``."""
    v = v.item() if hasattr(v, "item") else v
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    text = str(v)
    return text.rsplit(".", 1)[-1] if re.fullmatch(r"[A-Za-z_]\w*\.\w+", text) else text


def _identity(pairs):
    """What a variant is: every non-null sweep column, the way the comparer keys a point."""
    import pandas as pd

    return tuple(sorted((c, _value(v)) for c, v in pairs if c not in _NOT_IDENTITY and pd.notna(v)))


def _points(csvs):
    """{identity: TEXT_SIZE(MATH_ISOLATE) or None} of every TILE_LOOP row in ``csvs``."""
    import pandas as pd

    out = {}
    for path in csvs:
        df = pd.read_csv(path)
        for _, r in df[df["marker"] == "TILE_LOOP"].iterrows():
            size = r.get("TEXT_SIZE(MATH_ISOLATE)")
            pairs = [
                (c, v) for c, v in r.items() if c != "marker" and not c.startswith(("mean(", "std(", "TEXT_SIZE("))
            ]
            out[_identity(pairs)] = int(size) if pd.notna(size) else None
    return out


def _labels(identities):
    """{identity: op label}: the op and its switches that are on, ``SfpuElwadd (unpack
    bcast Row)``, ``Exp (fast_mode Yes)``. Variants that still read alike -- same label,
    formats, dest_acc and approx -- are told apart by every column they differ in.
    """
    labels = {}
    for ident in identities:
        cfg = dict(ident)
        on = [f"{name} {cfg[c]}" for c, name in _SWITCHES.items() if cfg.get(c, "None") not in _OFF]
        labels[ident] = cfg.get("mathop", "?") + (f" ({', '.join(on)})" if on else "")
    alike = {}
    for ident, label in labels.items():
        alike.setdefault((label, *(dict(ident).get(c) for c in _SHOWN)), []).append(ident)
    for group in (g for g in alike.values() if len(g) > 1):
        cfgs = [dict(i) for i in group]
        differ = sorted({c for cfg in cfgs for c in cfg if len({x.get(c) for x in cfgs}) > 1})
        for ident, cfg in zip(group, cfgs):
            extra = ", ".join(f"{c} {cfg.get(c, '–')}" for c in differ)
            label = labels[ident]
            labels[ident] = f"{label[:-1]}, {extra})" if label.endswith(")") else f"{label} ({extra})"
    return labels


def rows(runs, verdicts):
    """Report rows per schedule: one per variant, TILE_LOOP, cycles per tile.

    A variant that ran on one side only has a row too, with ``missing`` naming the side
    without a result. The comparer leaves those out (it walks the head side's points),
    so a variant the PR broke would otherwise drop out of the report.
    """
    points = {schedule: {side: _points(csvs) for side, csvs in sides.items()} for schedule, sides in runs.items()}
    measured = {
        _identity(rec["config"])
        for verdict in verdicts.values()
        for rec in verdict["records"]
        if rec["marker"] == "TILE_LOOP"
    }
    labels = _labels(measured | {i for sides in points.values() for side in sides.values() for i in side})
    out = {}
    for schedule, verdict in verdicts.items():
        base, head = points[schedule]["base"], points[schedule]["head"]
        table = {}

        def row(ident):
            cfg = dict(ident)
            return table.setdefault(
                ident,
                {
                    "op": labels[ident],
                    "formats": f'{cfg.get("formats.input_A")}->{cfg.get("formats.output")}',
                    "dest_acc": cfg.get("dest_acc"),
                    "approx": cfg.get("approx_mode"),
                    "fast_mode": cfg.get("fast_mode", "No"),
                    "text_base": base.get(ident),
                    "text_head": head.get(ident),
                },
            )

        for rec in verdict["records"]:
            if rec["marker"] != "TILE_LOOP":
                continue
            cfg = dict(rec["config"])
            tiles = float(cfg.get("tile_cnt", 1)) * float(cfg.get("loop_factor", 1))
            row(_identity(rec["config"]))[rec["run_type"]] = {
                "base": rec["baseline"] / tiles,
                "head": rec["current"] / tiles,
                "delta": rec["delta"],
                "regression": rec["regression"],
                "improvement": rec["improvement"],
            }
        for ident in base.keys() - head.keys():
            row(ident)["missing"] = "head"
        for ident in head.keys() - base.keys():
            row(ident)["missing"] = "base"
        out[schedule] = sorted(
            table.values(),
            key=lambda r: (r["op"], r["formats"], r["dest_acc"], r["approx"]),
        )
    return out
