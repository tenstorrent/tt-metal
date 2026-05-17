# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Calibration — load measured per-chip memory from successful runs and use
them to correct the analytical overhead constants.

Phase 1 (this file): defines the on-disk schema and a no-op loader.  The
file format is finalized so calibration data can be added without changing
the consumer code.

Phase 2 (next): integrate with `tt-smi` telemetry capture during demo runs
and write the implied overhead back to `data/calibration.yaml`.

Schema (YAML or JSON):

    schema_version: "1.0"
    runs:
      - model: meta-llama/Llama-3.1-8B
        commit: <repo SHA at time of measurement>
        box: N150
        mesh: [1, 1]
        batch: 1
        seq: 8192
        dtype: bf16
        kv_dtype: bf16
        measured_per_chip_gb: 8.4
        predicted_per_chip_gb: 9.2
        implied_overhead_gb: 0.8
        source: "demo CI, tt-smi telem 2026-01-15"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class CalibrationRun:
    model: str
    box: str
    mesh: tuple
    batch: int
    seq: int
    dtype: str
    measured_per_chip_gb: float
    predicted_per_chip_gb: float
    implied_overhead_gb: float
    source: str = ""
    commit: str = ""
    kv_dtype: str = "bf16"


@dataclass
class CalibrationDB:
    schema_version: str = "1.0"
    runs: List[CalibrationRun] = field(default_factory=list)

    def implied_overhead_for(self, box_name: str) -> Optional[float]:
        """
        Return the median implied overhead (GB/chip) across runs on this box.
        Returns None if no calibration data is available for this box.
        """
        matching = [r.implied_overhead_gb for r in self.runs if r.box == box_name]
        if not matching:
            return None
        matching.sort()
        n = len(matching)
        return matching[n // 2] if n % 2 else (matching[n // 2 - 1] + matching[n // 2]) / 2


def load(path: Path) -> CalibrationDB:
    """Load calibration data from a YAML or JSON file. Returns an empty DB if missing."""
    if not path.exists():
        return CalibrationDB()

    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            raise RuntimeError("PyYAML required to load YAML calibration data")
    else:
        data = json.loads(text)

    db = CalibrationDB(schema_version=data.get("schema_version", "1.0"))
    for r in data.get("runs", []):
        db.runs.append(
            CalibrationRun(
                model=r["model"],
                box=r["box"],
                mesh=tuple(r["mesh"]),
                batch=r["batch"],
                seq=r["seq"],
                dtype=r["dtype"],
                kv_dtype=r.get("kv_dtype", "bf16"),
                measured_per_chip_gb=r["measured_per_chip_gb"],
                predicted_per_chip_gb=r["predicted_per_chip_gb"],
                implied_overhead_gb=r["implied_overhead_gb"],
                source=r.get("source", ""),
                commit=r.get("commit", ""),
            )
        )
    return db


DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "data" / "calibration.yaml"


def save(db: CalibrationDB, path: Path) -> None:
    """Persist calibration DB to a YAML or JSON file."""
    data = {
        "schema_version": db.schema_version,
        "runs": [
            {
                "model": r.model,
                "commit": r.commit,
                "box": r.box,
                "mesh": list(r.mesh),
                "batch": r.batch,
                "seq": r.seq,
                "dtype": r.dtype,
                "kv_dtype": r.kv_dtype,
                "measured_per_chip_gb": r.measured_per_chip_gb,
                "predicted_per_chip_gb": r.predicted_per_chip_gb,
                "implied_overhead_gb": r.implied_overhead_gb,
                "source": r.source,
            }
            for r in db.runs
        ],
    }
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            text = yaml.safe_dump(data, sort_keys=False)
        except ImportError:
            raise RuntimeError("PyYAML required to write YAML calibration data")
    else:
        text = json.dumps(data, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def calibrate_box(
    box_name: str, mesh_shape: tuple, source_label: str = "manual calibrate run", db_path: Optional[Path] = None
) -> CalibrationRun:
    """
    Open the requested mesh on real hardware, measure the largest single
    allocation that succeeds, and back out the implied per-chip overhead.

    Writes a new entry to the calibration DB.

    Requires `ttnn` to be importable (i.e. tt-metal python env active).
    """
    from .device import open_mesh, usable_bytes_per_chip
    from .hardware import find_box

    box = find_box(box_name)
    rows, cols = mesh_shape
    if (rows, cols) not in box.mesh_shapes:
        raise ValueError(
            f"mesh {mesh_shape} is not a canonical shape for {box_name}; " f"valid shapes: {box.mesh_shapes}"
        )

    print(f"  Opening mesh {mesh_shape} on {box_name} ({box.arch}) ...")
    with open_mesh(mesh_shape) as mesh:
        print(f"  Measuring largest single allocation ...")
        measured_gb = usable_bytes_per_chip(
            mesh,
            min_gb=0.5,
            max_gb=box.hbm_per_chip_gb * 1.0,
            tolerance_gb=0.25,
        )

    # Analytical prediction with current (uncalibrated) constants.
    tp = rows * cols
    predicted_gb = box.usable_per_chip_gb(tp)

    # Implied overhead = hbm - measured.  The analytical model says:
    #   usable = hbm - dispatch - ccl - frag*hbm
    # We invert: implied_overhead = hbm - measured.
    implied_overhead_gb = box.hbm_per_chip_gb - measured_gb

    run = CalibrationRun(
        model="(empty-mesh probe)",
        box=box_name,
        mesh=tuple(mesh_shape),
        batch=0,
        seq=0,
        dtype="bf16",
        kv_dtype="bf16",
        measured_per_chip_gb=round(measured_gb, 3),
        predicted_per_chip_gb=round(predicted_gb, 3),
        implied_overhead_gb=round(implied_overhead_gb, 3),
        source=source_label,
    )

    path = db_path or DEFAULT_CALIBRATION_PATH
    db = load(path)
    db.runs.append(run)
    save(db, path)

    return run
