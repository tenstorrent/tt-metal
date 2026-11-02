#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Headroom verdict: can this LLK be faster, and what is in the way?

Combines three inputs for ONE perf-test variant and ONE run type:

1. the wave breakdown JSON from the private ``llk_wave_debug.py perf`` command
   (per thread: issue / stall-by-reason / idle cycles over the TILE_LOOP window,
   plus issued instructions by opcode);
2. the perf test's ``.post.csv`` (per-tile cycles per run type);
3. one or more issue-rate tables (every entry cites its ``source``).

and reports:

* ``reconcile`` - wave cycles per tile vs the CSV, |delta| <= 2 % or the
  verdict is not trusted (exit 2);
* ``bottleneck`` - the thread with the most issue+stall cycles per tile;
* ``stall_headroom_pct`` - how far the measured time is above the bottleneck's
  no-stall floor (sum of issued instructions x issue cost). High = scheduling /
  synchronisation problem;
* ``instruction_headroom`` - bottleneck instructions per tile vs the best known
  count for the op (only with ``--best-count``). High = rewrite the math;
* ``causes`` - the bottleneck's stall reasons and idle, per tile, largest first.

Per-tile normalisation is the post.csv one: cycles / (loop_factor * tile_cnt).
Stdlib only.

Exit codes: 0 verdict produced; 2 bad/ambiguous input or not reconciled.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

RECONCILE_TOLERANCE_PCT = 2.0
_DEFAULT_ASSEMBLY = (
    Path(__file__).resolve().parents[2]
    / "tt_llk_quasar"
    / "instructions"
    / "assembly.yaml"
)
_DEFAULT_RATES = (
    Path(__file__).resolve().parents[1] / "references" / "quasar_issue_rates.json"
)


# --- rates -------------------------------------------------------------------


def _check_entry(name: str, entry: Any) -> None:
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("cycles"), (int, float))
        or entry["cycles"] <= 0
    ):
        raise ValueError(f"rate entry {name!r} needs a positive numeric 'cycles'")
    if not str(entry.get("source", "")).strip():
        raise ValueError(
            f"rate entry {name!r} has no 'source'; unsourced rates are not allowed"
        )


def load_rates(paths: list[Path]) -> dict:
    """Merge rate tables in order (later overrides earlier); every entry must cite a source."""

    merged: dict[str, Any] = {"default": None, "opcodes": {}}
    for path in paths:
        data = json.loads(Path(path).read_text())
        if "default" in data:
            _check_entry("default", data["default"])
            merged["default"] = data["default"]
        for name, entry in data.get("opcodes", {}).items():
            _check_entry(name, entry)
            merged["opcodes"][name] = entry
    if merged["default"] is None:
        raise ValueError("no rate table provides a 'default' entry")
    return merged


# --- opcodes -----------------------------------------------------------------


def load_opcode_names(assembly_yaml: Path) -> dict[str, str]:
    """Map "0x84" -> "SFPMAD" from an assembly.yaml (top-level NAME: + op_binary)."""

    table: dict[str, str] = {}
    current: str | None = None
    for line in Path(assembly_yaml).read_text().splitlines():
        top = re.match(r"^([A-Z][A-Z0-9_]*):\s*$", line)
        if top:
            current = top.group(1)
            continue
        op = re.match(r"^\s+op_binary:\s*(0x[0-9a-fA-F]+)", line)
        if op and current:
            key = f"0x{int(op.group(1), 16):02x}"
            table[key] = current if key not in table else f"{table[key]}|{current}"
            current = None
    return table


def name_opcodes(by_opcode: dict[str, int], table: dict[str, str]) -> dict[str, int]:
    """Rename hex opcode keys to mnemonics where known; unknown keys stay hex."""

    named: dict[str, int] = {}
    for key, count in by_opcode.items():
        name = table.get(key, key)
        named[name] = named.get(name, 0) + count
    return named


# --- arithmetic --------------------------------------------------------------


def per_tile(cycles: float, loop_factor: int, tile_cnt: int) -> float:
    """post.csv normalisation: cycles / (loop_factor * tile_cnt)."""

    return cycles / (loop_factor * tile_cnt)


def unit_floor_cycles(thread: dict, rates: dict) -> tuple[float, list[str]]:
    """No-stall floor for one thread: sum(count x issue cost); also the opcodes that used the default."""

    default = rates["default"]["cycles"]
    by_opcode = thread.get("instructions_by_opcode") or {}
    if not by_opcode:
        return thread["instructions_issued"] * default, ["<all>"]
    total = 0
    defaulted: list[str] = []
    for name, count in by_opcode.items():
        entry = rates["opcodes"].get(name)
        if entry is None:
            defaulted.append(name)
            total += count * default
        else:
            total += count * entry["cycles"]
    return total, sorted(defaulted)


def _name(opcode: str, table: dict[str, str] | None) -> str:
    return (table or {}).get(opcode, opcode)


def _round(value: float) -> float:
    return round(value, 3)


def headroom_verdict(
    report: dict,
    csv_row: dict,
    rates: dict,
    *,
    run_type: str,
    loop_factor: int | None = None,
    tile_cnt: int | None = None,
    best_count: int | None = None,
    best_count_source: str | None = None,
    opcode_names: dict[str, str] | None = None,
) -> dict:
    status = report.get("status")
    if status not in ("complete", "missing_signals"):
        raise ValueError(
            f"wave report status is {status!r}; refusing to judge an incomplete capture"
        )
    core = report.get("missing_core_signals") or []
    if core:
        raise ValueError(
            f"wave report is missing core signals {core}; stall time would read as idle"
        )
    loop_factor = int(
        loop_factor if loop_factor is not None else csv_row["loop_factor"]
    )
    tile_cnt = int(tile_cnt if tile_cnt is not None else csv_row["tile_cnt"])
    column = f"mean({run_type})"
    if column not in csv_row:
        raise ValueError(f"CSV row has no {column!r} column")
    csv_per_tile = float(csv_row[column])
    wave_per_tile = per_tile(report["window"]["cycles"], loop_factor, tile_cnt)
    delta_pct = 100.0 * (wave_per_tile - csv_per_tile) / csv_per_tile

    units: dict[str, Any] = {}
    for name, thread in report["threads"].items():
        if opcode_names:
            thread = dict(
                thread,
                instructions_by_opcode=name_opcodes(
                    thread.get("instructions_by_opcode") or {}, opcode_names
                ),
            )
        floor, defaulted = unit_floor_cycles(thread, rates)
        units[name] = {
            "floor_per_tile": _round(per_tile(floor, loop_factor, tile_cnt)),
            "issue_per_tile": _round(per_tile(thread["issue"], loop_factor, tile_cnt)),
            "stall_per_tile": {
                r: _round(per_tile(c, loop_factor, tile_cnt))
                for r, c in thread["stall"].items()
            },
            "idle_per_tile": _round(per_tile(thread["idle"], loop_factor, tile_cnt)),
            "bubbles_per_tile": {
                f"{_name(b['after'], opcode_names)}->{_name(b['before'], opcode_names)}": _round(
                    per_tile(b["cycles"], loop_factor, tile_cnt)
                )
                for b in thread.get("bubbles") or []
            },
            "instructions_per_tile": _round(
                per_tile(thread["instructions_issued"], loop_factor, tile_cnt)
            ),
            "instructions_by_opcode": thread.get("instructions_by_opcode") or {},
            "defaulted_opcodes": defaulted,
        }
    if not units:
        raise ValueError("wave report has no threads")

    # The bottleneck is the busiest *issuer*: a thread that is mostly stalled is
    # usually waiting on another thread (e.g. unpack stalled on instissue while
    # math owns Dest), so stall time alone must not elect it.
    bottleneck = max(units, key=lambda name: (units[name]["issue_per_tile"], name))
    waiting = {
        name: {r: v for r, v in unit["stall_per_tile"].items() if v > 0}
        for name, unit in units.items()
        if name != bottleneck
        and sum(unit["stall_per_tile"].values()) > unit["issue_per_tile"]
    }
    b = units[bottleneck]
    causes = [[f"{bottleneck}.{r}", v] for r, v in b["stall_per_tile"].items()]
    causes += [
        [f"{bottleneck}.bubble.{k}", v] for k, v in b["bubbles_per_tile"].items()
    ]
    residual_idle = b["idle_per_tile"] - sum(b["bubbles_per_tile"].values())
    causes.append([f"{bottleneck}.idle", _round(residual_idle)])
    causes = sorted((c for c in causes if c[1] > 0), key=lambda c: -c[1])

    instruction = None
    if best_count is not None:
        current = b["instructions_per_tile"]
        instruction = {
            "current": current,
            "best": best_count,
            "source": best_count_source or "",
            "pct": _round(100.0 * (current - best_count) / current) if current else 0.0,
        }
    # A floor above the measured time means some issue cost is overstated (e.g. an opcode
    # costed at 2 whose second cycle is already paid by an explicit SFPNOP or a stall).
    multi = sorted(
        name
        for name in b["instructions_by_opcode"]
        if rates["opcodes"].get(name, rates["default"])["cycles"] > 1
    )
    floor_check = {
        "ok": b["floor_per_tile"] <= csv_per_tile * 1.005,
        "multi_cycle_opcodes": multi,
    }
    return {
        "floor_check": floor_check,
        "run_type": run_type,
        "wave_status": status,
        "missing_signals": report.get("missing_signals", []),
        "reconcile": {
            "wave_per_tile": _round(wave_per_tile),
            "csv_per_tile": _round(csv_per_tile),
            "delta_pct": _round(delta_pct),
            "ok": abs(delta_pct) <= RECONCILE_TOLERANCE_PCT,
        },
        "units": units,
        "bottleneck": bottleneck,
        "waiting_threads": waiting,
        "stall_headroom_pct": _round(
            100.0 * (csv_per_tile - b["floor_per_tile"]) / csv_per_tile
        ),
        "instruction_headroom": instruction,
        "causes": causes,
    }


# --- CLI ---------------------------------------------------------------------


def _select_row(csv_path: Path, selectors: list[str]) -> dict:
    with open(csv_path, newline="") as handle:
        rows = [
            r
            for r in csv.DictReader(handle)
            if r.get("marker", "TILE_LOOP") == "TILE_LOOP"
        ]
    wanted = []
    for selector in selectors:
        if "=" not in selector:
            raise ValueError(f"--select expects key=value, got {selector!r}")
        key, value = selector.split("=", 1)
        wanted.append((key, value))
    matches = [r for r in rows if all(r.get(k) == v for k, v in wanted)]
    if len(matches) != 1:
        raise ValueError(
            f"--select {selectors} matched {len(matches)} TILE_LOOP rows; need exactly 1"
        )
    return matches[0]


def _summary(v: dict) -> str:
    r = v["reconcile"]
    lines = [
        f"run type        : {v['run_type']}  (wave status: {v['wave_status']})",
        f"reconcile       : wave {r['wave_per_tile']} vs csv {r['csv_per_tile']} cyc/tile "
        f"({r['delta_pct']:+.2f} %) -> {'OK' if r['ok'] else 'NOT RECONCILED'}",
        f"bottleneck      : {v['bottleneck']}",
        f"stall headroom  : {v['stall_headroom_pct']} % (measured vs no-stall floor "
        f"{v['units'][v['bottleneck']]['floor_per_tile']} cyc/tile)",
    ]
    fc = v.get("floor_check") or {"ok": True}
    if not fc["ok"]:
        lines.append(
            "WARNING         : floor above measured: issue costs overstated for "
            f"{', '.join(fc['multi_cycle_opcodes']) or 'unknown opcodes'}; stall headroom is not meaningful"
        )
    ih = v["instruction_headroom"]
    if ih:
        lines.append(
            f"instr headroom  : {ih['pct']} % ({ih['current']} -> {ih['best']} instr/tile; {ih['source']})"
        )
    for name, reasons in v.get("waiting_threads", {}).items():
        lines.append(
            f"waiting thread  : {name} stalled on {reasons} (waits on the bottleneck, not a cause)"
        )
    lines.append("causes (cyc/tile):")
    lines += [f"  {name:<28} {value}" for name, value in v["causes"]]
    if v["missing_signals"]:
        lines.append(f"missing signals : {', '.join(v['missing_signals'])}")
    for name, unit in v["units"].items():
        if unit["defaulted_opcodes"]:
            lines.append(
                f"{name}: default issue cost used for {', '.join(unit['defaulted_opcodes'])}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--wave-report",
        required=True,
        type=Path,
        help="JSON from llk_wave_debug.py perf",
    )
    p.add_argument("--perf-csv", required=True, type=Path, help="perf test .post.csv")
    p.add_argument(
        "--select",
        action="append",
        default=[],
        help="key=value to pick one TILE_LOOP row (repeatable)",
    )
    p.add_argument(
        "--run-type",
        required=True,
        help="run type the wave was captured for, e.g. MATH_ISOLATE",
    )
    p.add_argument(
        "--rates",
        action="append",
        type=Path,
        default=None,
        help=f"issue-rate table(s), later override earlier (default: {_DEFAULT_RATES.name})",
    )
    p.add_argument(
        "--assembly-yaml",
        type=Path,
        default=_DEFAULT_ASSEMBLY,
        help="opcode -> mnemonic table",
    )
    p.add_argument(
        "--loop-factor",
        type=int,
        default=None,
        help="override the CSV row's loop_factor",
    )
    p.add_argument(
        "--tile-cnt", type=int, default=None, help="override the CSV row's tile_cnt"
    )
    p.add_argument(
        "--best-count",
        type=int,
        default=None,
        help="best known bottleneck instructions per tile",
    )
    p.add_argument(
        "--best-count-source", default=None, help="where --best-count comes from"
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    try:
        rates = load_rates(args.rates or [_DEFAULT_RATES])
        report = json.loads(args.wave_report.read_text())
        row = _select_row(args.perf_csv, args.select)
        names = (
            load_opcode_names(args.assembly_yaml)
            if args.assembly_yaml.exists()
            else None
        )
        verdict = headroom_verdict(
            report,
            row,
            rates,
            run_type=args.run_type,
            loop_factor=args.loop_factor,
            tile_cnt=args.tile_cnt,
            best_count=args.best_count,
            best_count_source=args.best_count_source,
            opcode_names=names,
        )
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"perf_headroom: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(verdict, indent=2) if args.json else _summary(verdict))
    return 0 if verdict["reconcile"]["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
