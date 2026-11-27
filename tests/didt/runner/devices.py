# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Failing devices of one attempt, in status.json's devices[] shape: identity joined from the chip snapshot on `dev`,
then what the log or triage CSV says about that device. Data the source does not give is left out."""

import csv
import io
import re

from .chips import TRAY_OF_BUS_HIGH, Chip

MAX_CORES = 8
_BDF = r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"


def identity(dev: int, chips: dict[int, Chip]) -> dict:
    return chips[dev].as_dict() if dev in chips else {"dev": dev}


def _identity_by_bdf(dev: int, bdf: str, chips: dict[int, Chip]) -> dict:
    hit = next((c for c in chips.values() if c.bdf == bdf), None)
    if hit is not None:
        return {**hit.as_dict(), "dev": dev}
    bus = int(bdf.split(":")[1], 16)
    tray = TRAY_OF_BUS_HIGH.get(bus >> 4)
    return {"dev": dev, "bdf": bdf, **({"tray": tray} if tray else {}), "asic": bus & 0xF}


def _top(cores: list[dict], key: str) -> list[dict]:
    return sorted(cores, key=lambda c: (-c[key], c["y"], c["x"]))[:MAX_CORES]


def nd_report_devices(
    text: str, chips: dict[int, Chip], dev_override: int | None = None
) -> tuple[list[dict], list[str]]:
    """From op_test_base's end-of-run `Chip map device N`, `ND summary device N [...]` and per-core summary lines.
    dev_override: the parent's chip id for a child process that sees itself as device 0."""
    log_bdf = {
        int(d): m.group(0)
        for d, where in re.findall(r"Chip map device (\d+): (.*)$", text, re.MULTILINE)
        if (m := re.search(_BDF, where))
    }
    summary = re.findall(
        r"ND summary device (\d+) \[[^\]]*\]: (\d+)/(\d+) checks; pattern (.*?); failing checks per core:",
        text,
    )
    core_lines = re.findall(
        r"ND summary device (\d+) core \((\d+),(\d+)\)(?: translated=\((\d+),(\d+)\))?: "
        r"failed (\d+)/\d+ checks; (.*)$",
        text,
        re.MULTILINE,
    )
    devices, warnings = {}, []
    for logged, failed, checks, pattern in summary:
        logged = int(logged)
        dev = logged if dev_override is None else dev_override
        ident = identity(dev, chips)
        bdf = log_bdf.get(logged)
        if bdf and bdf != ident.get("bdf"):
            if "bdf" in ident:
                warnings.append(f"device {dev}: log says bdf {bdf}, chip snapshot says {ident['bdf']}; kept the log's")
            ident = _identity_by_bdf(dev, bdf, chips)
        cores = []
        for d, x, y, tx, ty, n, signature in core_lines:
            if int(d) != logged:
                continue
            core = {"x": int(x), "y": int(y)}
            if tx:
                core["translated"] = [int(tx), int(ty)]
            cores.append({**core, "failed_checks": int(n), "signature": signature.strip()})
        devices[dev] = {
            **ident,
            "failed_checks": int(failed),
            "checks": int(checks),
            "pattern": pattern,
            "hung": False,
            **({"cores": _top(cores, "failed_checks")} if cores else {}),
        }
    return [devices[d] for d in sorted(devices)], warnings


def run_to_run_devices(text: str, chips: dict[int, Chip]) -> list[dict]:
    """tests/didt/test_matmul_run_to_run_determinism.py: differing elements per device and core, summed over windows."""
    per: dict[int, dict[tuple, int]] = {}
    for dev, x, y, px, py, n in re.findall(
        r"\(device id (\d+)\), logical core \((\d+), (\d+)\) -> physical \((\d+),(\d+)\): "
        r"run-to-run NOT deterministic, (\d+) elements",
        text,
    ):
        cores = per.setdefault(int(dev), {})
        key = (int(x), int(y), int(px), int(py))
        cores[key] = cores.get(key, 0) + int(n)
    return [
        {
            **identity(dev, chips),
            "hung": False,
            "cores": _top(
                [{"x": x, "y": y, "translated": [px, py], "elements": n} for (x, y, px, py), n in per[dev].items()],
                "elements",
            ),
        }
        for dev in sorted(per)
    ]


def core_sweep_devices(text: str, chips: dict[int, Chip]) -> list[dict]:
    """tests/didt/scratch_core_sweep.py: differing elements per device; a 1x1 window is a core."""
    total: dict[int, int] = {}
    cores: dict[int, list[dict]] = {}
    for gx, gy, ox, oy, dev, n in re.findall(
        r"window (\d+)x(\d+) at \((\d+),(\d+)\): device (\d+) NOT deterministic, (\d+) elements", text
    ):
        dev = int(dev)
        total[dev] = total.get(dev, 0) + int(n)
        if (gx, gy) == ("1", "1"):
            cores.setdefault(dev, []).append({"x": int(ox), "y": int(oy), "elements": int(n)})
    return [
        {
            **identity(dev, chips),
            "hung": False,
            "elements": total[dev],
            **({"cores": _top(cores[dev], "elements")} if dev in cores else {}),
        }
        for dev in sorted(total)
    ]


def _sections(csv_text: str) -> dict[str, str]:
    """tt-triage --llm-output-path CSV: one `<script>.py:` header line per script, then that script's table."""
    parts = re.split(r"^(\w+\.py):$", csv_text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _rows(section: str) -> list[dict[str, str]]:
    # Cells are quoted where they hold commas or newlines; tt-triage writes line breaks inside a cell as a literal \n.
    reader = csv.DictReader(io.StringIO(section.strip("\n")))
    return [{k: (v or "").replace("\\n", "\n") for k, v in row.items() if k} for row in reader]


def triage_hang_devices(
    csv_text: str, chips: dict[int, Chip], dev_override: int | None = None
) -> tuple[list[dict], list[str]]:
    """Stalled devices from dump_running_operations.py, cross-checked with dump_op_mesh.py's T<tray>:N<asic> cells.
    The workload is still up when the dispatch-timeout hook runs triage, so this names the op and its cores.
    dev_override: the parent's chip id for a child process that sees itself as device 0."""
    sections = _sections(csv_text)
    mesh = {
        int(dev): (int(tray), int(asic))
        for tray, asic, dev in re.findall(r"T(\d+):N(\d+) \(Device (\d+)\)", sections.get("dump_op_mesh.py", ""))
    }
    devices, warnings = {}, []
    for row in _rows(sections.get("dump_running_operations.py", "")):
        for logged in (int(d) for d in re.findall(r"\d+", row.get("Devices", ""))):
            dev = logged if dev_override is None else dev_override
            if dev in devices:
                continue
            ident = identity(dev, chips)
            if logged in mesh and "tray" in ident and mesh[logged] != (ident["tray"], ident["asic"]):
                warnings.append(
                    f"device {dev}: triage says T{mesh[logged][0]}:N{mesh[logged][1]}, chip snapshot says "
                    f"T{ident['tray']}:N{ident['asic']} ({ident['bdf']})"
                )
            cores = [
                {"x": int(x), "y": int(y), "translated": [int(tx), int(ty)]}
                for d, tx, ty, x, y in re.findall(r"^(\d+):(\d+)-(\d+) \((\d+),(\d+)\)$", row.get("Cores", ""), re.M)
                if int(d) == logged
            ]
            count = row.get("Core Cnt", "").strip()
            devices[dev] = {
                **ident,
                "hung": True,
                **({"op": row["Op Name"]} if row.get("Op Name") else {}),
                **({"core_count": int(count)} if count.isdigit() else {}),
                **({"cores": cores} if cores else {}),
            }
    return [devices[d] for d in sorted(devices)], warnings
