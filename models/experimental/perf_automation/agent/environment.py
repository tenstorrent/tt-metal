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
# peak_tflops_per_core = matrix-engine peak per core per MATH_FIDELITY (tech_reports/
# matrix_engine + GEMM_FLOPS): the ROOFLINE compute ceiling. Per-core × full grid =
# chip peak (BH ~580 TFLOPS measured @ LoFi). clock_ghz from GEMM_FLOPS (BH 1.35, WH 1.0).
ARCH_FACTS: dict[str, dict[str, Any]] = {
    "wormhole": {
        "arch": "wormhole",
        "grid_x": 8,
        "grid_y": 8,
        "dram_bw_gbps": 288.0,
        "clock_ghz": 1.0,
        "peak_tflops_per_core": {"lofi": 4.0, "hifi2": 2.0, "hifi3": 1.33, "hifi4": 1.0},
    },
    "wormhole_b0": {
        "arch": "wormhole",
        "grid_x": 8,
        "grid_y": 8,
        "dram_bw_gbps": 288.0,
        "clock_ghz": 1.0,
        "peak_tflops_per_core": {"lofi": 4.0, "hifi2": 2.0, "hifi3": 1.33, "hifi4": 1.0},
    },
    "blackhole": {
        "arch": "blackhole",
        "grid_x": 13,
        "grid_y": 10,
        "dram_bw_gbps": 512.0,
        "clock_ghz": 1.35,
        "peak_tflops_per_core": {"lofi": 5.4, "hifi2": 2.7, "hifi3": 1.8, "hifi4": 1.35},
    },
}


# Per-BOX compute grid override (tt-hw-planner's registry carries chips/arch/mesh but
# NOT grid_x/grid_y). Blackhole SKUs differ: p300c=13x10=130 (ARCH_FACTS default),
# QB2's p150c=11x10=110. Keyed by box name; falls back to ARCH_FACTS[arch] grid.
BOX_COMPUTE_GRID: dict[str, tuple[int, int]] = {
    "QB2": (11, 10),
}


def box_facts(box_name: str, mesh: tuple[int, int] | None = None) -> dict[str, Any]:
    """Roofline facts for a DECLARED box+mesh (the --box/--mesh flag), REUSING
    tt-hw-planner's hardware registry (no duplicate board table). Returns the arch
    peak facts with worker_cores = (mesh chips) × (per-chip grid) and the mesh/eth
    facts for the CCL roofline. Raises EnvironmentError_ on an unknown box or a mesh
    that isn't canonical for it. Guarded import: if tt-hw-planner is absent, the
    caller keeps the single-chip ARCH_FACTS path."""
    try:
        from scripts.tt_hw_planner.hardware import HARDWARE
    except Exception as exc:  # tt-hw-planner not importable -> caller falls back
        raise EnvironmentError_(f"tt-hw-planner hardware registry unavailable: {exc}") from exc

    box = next((b for b in HARDWARE if b.name.lower() == (box_name or "").lower()), None)
    if box is None:
        raise EnvironmentError_(f"unknown box {box_name!r}; known: {[b.name for b in HARDWARE]}")
    mesh = mesh or box.default_mesh or (box.mesh_shapes[0] if box.mesh_shapes else (1, 1))
    if box.mesh_shapes and tuple(mesh) not in [tuple(m) for m in box.mesh_shapes]:
        raise EnvironmentError_(f"mesh {tuple(mesh)} not canonical for {box.name}; valid: {box.mesh_shapes}")

    arch = box.arch.lower()
    base = dict(ARCH_FACTS.get(arch, ARCH_FACTS["blackhole"]))  # peak_tflops/dram_bw/clock by arch
    gx, gy = BOX_COMPUTE_GRID.get(box.name, (base.get("grid_x", 13), base.get("grid_y", 10)))
    mesh_chips = int(mesh[0]) * int(mesh[1])
    per_chip_dram_bw = base.get("dram_bw_gbps", 0.0)
    base.update(
        {
            "card": box.name,
            "arch": arch,
            "grid_x": gx,
            "grid_y": gy,
            "mesh_shape": list(mesh),
            "mesh_chips": mesh_chips,
            "worker_cores": gx * gy * mesh_chips,  # roofline COMPUTE floor = full mesh, not one chip
            # AGGREGATE DRAM bandwidth across the mesh for the MEMORY floor. Valid when data is
            # sharded across chips (tensor/data-parallel) — the common multi-chip case. For a
            # fully REPLICATED model this overstates effective bw (a noted simplification).
            "dram_bw_gbps": per_chip_dram_bw * mesh_chips,
            "dram_bw_per_chip_gbps": per_chip_dram_bw,
            "eth_link_gbps": box.eth_link_gbps,
            "hbm_total_gb": box.total_hbm_gb,
        }
    )
    return base


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
