# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.NC (OWNER 2026-11-27): results/MB.NC_layer.md, ONE report for the batching branch's C++ ports brought to
model_bringup, written by the six op-level gate tests (tests/woa/test_gmm_native_layer.py, tests/native/test_native_*).

Each test hands its programs' rows (program, host us before / after a call, calls a 5120 chunk) and its detail table;
the sections are kept under generated/bringup/MB.NC_layer/ (one file a test) and the report is rebuilt from every
section present, so a gate that re-runs only some units (and deletes the report first, F292) still writes them all."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
REPORT = REPO / "models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/results/MB.NC_layer.md"
SECTIONS = REPO / "generated" / "bringup" / "MB.NC_layer"
ORDER = ("gmm", "reduce_scatter", "sends", "lowbit", "block_select", "chunk_programs")


def write(name: str, rows: list[dict], detail: str) -> None:
    """Section ``name`` (one of ORDER): ``rows`` [{program, before_us, after_us, calls, note}] (host us a warm call on
    the Python prepared route / the native op, calls a single 5120 chunk), ``detail`` its markdown detail table."""
    SECTIONS.mkdir(parents=True, exist_ok=True)
    (SECTIONS / f"{name}.json").write_text(json.dumps({"rows": rows, "detail": detail}, indent=1))
    _render()


def _render() -> None:
    secs = {}
    for name in ORDER:
        p = SECTIONS / f"{name}.json"
        if p.exists():
            secs[name] = json.loads(p.read_text())
    lines = [
        "# MB.NC layer check: the batching branch's C++ ports on model_bringup (5120 chunk, 160 rows a chip)",
        "",
        "Written by the gate's op-level tests (no model load; the resident runner's held mesh, BATCH_LAYERS none): "
        "every program native vs the Python prepared route it replaces, max abs diff on every output and chip, "
        "repeats on fresh buffers, program-cache growth after warm-up. One row a program; host us a warm call "
        "(perf_counter, the median of rounds alternating the routes); calls a 5120 chunk estimated from the call "
        "sites (layers 0-20, DECODER_SKIP=decode). Model-wide check: task MB.NCL (the 5120 ladder).",
        "",
        "| program | host us before (Python prepared) | host us after (native) | calls a 5120 chunk | host ms a 5120 chunk "
        "saved | note |",
        "|---|---|---|---|---|---|",
    ]
    total = 0.0
    for name, sec in secs.items():
        for r in sec["rows"]:
            saved = (float(r["before_us"]) - float(r["after_us"])) * float(r["calls"]) / 1e3
            total += saved
            lines.append(
                f"| {r['program']} | {r['before_us']} | {r['after_us']} | {r['calls']} | {saved:.3f} "
                f"| {r.get('note', '')} |"
            )
    lines += [f"| total | | | | {total:.3f} | sections present: {', '.join(secs) or '-'} |", ""]
    for name, sec in secs.items():
        lines += [sec["detail"].rstrip(), ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
