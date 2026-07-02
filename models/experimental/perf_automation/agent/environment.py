"""M3 · environment_check (PLAN section 7.1).

Capture HW facts so sweeps use the REAL device grid. A `probe` callable returns
the raw snapshot text (mock fixture in M3; real `tt-smi` snapshot on hardware —
TBD(env-script) for the exact command + schema). The arch string maps to the
fixed grid / DRAM-bandwidth facts (PLAN section 4.3 arch peaks) so the schema is
stable across the mock->real swap.

Out (-> manifest.json): {card, arch, grid_x, grid_y, worker_cores, dram_bw_gbps}.
"""

from __future__ import annotations

import json
from typing import Any, Callable

# Arch peak facts (PLAN section 4.3): WH 288 GB/s, BH 512 GB/s; worker grids.
ARCH_FACTS: dict[str, dict[str, Any]] = {
    "wormhole": {"arch": "wormhole", "grid_x": 8, "grid_y": 8, "dram_bw_gbps": 288.0},
    "wormhole_b0": {"arch": "wormhole", "grid_x": 8, "grid_y": 8, "dram_bw_gbps": 288.0},
    "blackhole": {"arch": "blackhole", "grid_x": 13, "grid_y": 10, "dram_bw_gbps": 512.0},
}


class EnvironmentError_(Exception):
    """Raised when the snapshot is unparseable or the arch is unknown."""


def parse_env_snapshot(text: str) -> dict[str, Any]:
    """Parse a tt-smi-style JSON snapshot into the manifest HW-facts dict.

    Accepts {"card"/"board_type", "arch"} either at top level or inside the
    first entry of a "device_info" list. Derives grid + DRAM bw from ARCH_FACTS.
    """
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise EnvironmentError_(f"unparseable environment snapshot: {exc}") from exc

    node: dict[str, Any] = data
    if isinstance(data, dict) and data.get("device_info"):
        node = data["device_info"][0]

    arch_raw = str(node.get("arch", "")).strip().lower()
    if arch_raw not in ARCH_FACTS:
        raise EnvironmentError_(f"unknown arch in snapshot: {arch_raw!r}")
    facts = dict(ARCH_FACTS[arch_raw])
    card = node.get("card") or node.get("board_type") or node.get("board")
    facts["card"] = card
    facts["worker_cores"] = facts["grid_x"] * facts["grid_y"]
    return facts


def environment_check(probe: Callable[[], str] | None = None) -> dict[str, Any]:
    """Return HW facts for the manifest. `probe` yields the raw snapshot text.

    In M3 `probe` is a mock returning fixture text; on real HW it runs the
    tt-smi snapshot command (TBD(env-script)). The parsed schema is identical.
    """
    if probe is None:
        raise ValueError("probe (snapshot source) required until M8 wires real tt-smi")
    return parse_env_snapshot(probe())
