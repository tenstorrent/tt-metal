# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Split exported vectors into homogeneous DEVICE-KEY batches.

Today a job is "3 modules", so one job's vectors span several mesh shapes and dispatch axes.
The device is opened once and every vector inherits whatever shape auto-detect picked -- a
Galaxy job resolved (4, 8) and the 16 vectors traced at (8, 4) silently fell back to
ReplicateTensorToMesh instead of their traced placement. Modules then try to correct for this
at runtime by deriving an axis per vector and asking for a different device, which is where
reopens (and the Galaxy dispatch-wedge risk) come from.

This inverts that: group vectors OFFLINE by the device they need -- (mesh shape, dispatch
axis, fabric) -- and let each CI job declare that device up front, so it opens exactly one
device and never switches. The axis stops being inferred at runtime by a heuristic and becomes
a property of the batch.

Measured on the current Galaxy set (1990 vectors): 5 mesh shapes, 11 batches, versus ~35 jobs
today. Every batch is reopen-free by construction.

Axis policy is STRICT: a vector whose grids force ROW goes to a row batch, one that forces COL
goes to a col batch, and one that forces neither goes to col (the documented Galaxy default --
see vector_axis() in split_vectors_by_axis.py). Vectors that force neither COULD ride along in
a row batch and save ~4 jobs, but that relies on the grid scanner being right, and it has a
known gap (linear's gather_in0 path keys off output/hop grids, not the nominal compute width).
Paying 4 extra jobs to avoid a wrong-grid failure is the better trade.

Writes one directory per batch plus a batch manifest for the CI matrix to consume.
"""

import ast
import json
import math
from pathlib import Path

from split_vectors_by_axis import vector_dispatch_axis_hint

# Time model for splitting an oversized batch across jobs. Deliberately conservative: the
# 3s/vector figure comes from slice (a light op) -- conv2d/matmul/CCL are slower, so a batch
# near the cap may still need manual attention. Tune from measured per-module timings.
DEVICE_OPEN_MINUTES = 4
SECONDS_PER_VECTOR = 3.0
USABLE_BUDGET_MINUTES = 45


def _max_vectors_per_job() -> int:
    return int((USABLE_BUDGET_MINUTES - DEVICE_OPEN_MINUTES) * 60 / SECONDS_PER_VECTOR)


def _parse_pair(value):
    """Parse a mesh shape that may be a list, tuple, or the string '[4, 8]'."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return (int(parsed[0]), int(parsed[1]))
    return None


def vector_mesh_shape(vec):
    """The (rows, cols) mesh a vector was traced on, or None.

    Placement metadata first (that is what create_tensor_on_mesh compares against), then the
    per-vector traced_machine_info, which every vector records.
    """
    for value in vec.values():
        if isinstance(value, dict) and value.get("mesh_device_shape"):
            pair = _parse_pair(value["mesh_device_shape"])
            if pair:
                return pair
    machine_info = vec.get("traced_machine_info")
    for entry in machine_info if isinstance(machine_info, list) else [machine_info]:
        if isinstance(entry, dict) and entry.get("mesh_device_shape"):
            pair = _parse_pair(entry["mesh_device_shape"])
            if pair:
                return pair
    return None


def vector_device_key(vec):
    """(mesh_shape, axis, fabric) -- the device this vector needs.

    fabric follows the mesh: a unit axis means a 1D line/ring, otherwise 2D. It is derived
    rather than declared so it cannot drift out of sync with the shape.
    """
    mesh = vector_mesh_shape(vec)
    axis = vector_dispatch_axis_hint(vec) or "col"
    fabric = "1d" if (mesh and (mesh[0] == 1 or mesh[1] == 1)) else "2d"
    return mesh, axis, fabric


def batch_name(mesh, axis, fabric, part=None, total_parts=1):
    mesh_str = f"{mesh[0]}x{mesh[1]}" if mesh else "unknown"
    name = f"mesh{mesh_str}_{axis}_{fabric}"
    if total_parts > 1:
        name = f"{name}_p{part}"
    return name


def group_vectors(src_dir: Path):
    """Return {device_key: {suite: {vector_id: vector}}} plus the source file each came from."""
    grouped = {}
    for path in sorted(src_dir.glob("*.json")):
        if path.name == "generation_manifest.json":
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for suite, vectors in data.items():
            if not isinstance(vectors, dict):
                continue
            for vector_id, vec in vectors.items():
                if not isinstance(vec, dict):
                    continue
                key = vector_device_key(vec)
                grouped.setdefault(key, {}).setdefault(path.name, {}).setdefault(suite, {})[vector_id] = vec
    return grouped


def write_batches(grouped, dst_root: Path):
    """Write one directory per batch (splitting oversized ones) and return the manifest."""
    cap = _max_vectors_per_job()
    batches = []
    for (mesh, axis, fabric), files in sorted(grouped.items(), key=lambda kv: -_count(kv[1])):
        total = _count(files)
        parts = max(1, math.ceil(total / cap))
        # Split by FILE (a file is one module x one hw/mesh variant), never mid-file, so a
        # module's vectors stay together and the per-module device/program cache still helps.
        chunks = _chunk_files(files, parts)
        for index, chunk in enumerate(chunks, start=1):
            name = batch_name(mesh, axis, fabric, index, len(chunks))
            out_dir = dst_root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            written = []
            for file_name, suites in chunk.items():
                (out_dir / file_name).write_text(json.dumps(suites, indent=2))
                written.append(file_name)
            (out_dir / "generation_manifest.json").write_text(
                json.dumps({"vector_files": sorted(written), "vector_grouping_mode": "device_key"}, indent=2)
            )
            batches.append(
                {
                    "name": name,
                    "mesh_shape": list(mesh) if mesh else None,
                    "dispatch_axis": axis,
                    "fabric": fabric,
                    "vectors": _count(chunk),
                    "modules": len(written),
                    "vectors_dir": str(out_dir.relative_to(dst_root.parent)),
                }
            )
    return batches


def _count(files):
    return sum(len(v) for suites in files.values() for v in suites.values())


def _chunk_files(files, parts):
    """Greedily pack whole files into `parts` chunks, largest first (least-full bin)."""
    if parts <= 1:
        return [files]
    ordered = sorted(files.items(), key=lambda kv: -sum(len(v) for v in kv[1].values()))
    chunks = [{} for _ in range(parts)]
    sizes = [0] * parts
    for file_name, suites in ordered:
        target = sizes.index(min(sizes))
        chunks[target][file_name] = suites
        sizes[target] += sum(len(v) for v in suites.values())
    return [c for c in chunks if c]


_SWEEP_FRAMEWORK = Path(__file__).resolve().parent
_SRC_DIR = _SWEEP_FRAMEWORK / "vectors_export"
_DST_ROOT = _SWEEP_FRAMEWORK / "vectors_export_by_device"
_MANIFEST = _DST_ROOT / "batch_manifest.json"
# Same content as the JSON, one batch per line, tab separated: name, mesh ("4x8"), axis, dir.
# The CI shell loop reads this directly -- embedding a multi-line python one-liner inside a
# YAML block scalar is fragile (unindented lines silently break the workflow).
_MANIFEST_TSV = _DST_ROOT / "batch_manifest.tsv"


def main():
    if not _SRC_DIR.is_dir():
        raise SystemExit(f"No vectors at {_SRC_DIR}")
    grouped = group_vectors(_SRC_DIR)
    _DST_ROOT.mkdir(parents=True, exist_ok=True)
    batches = write_batches(grouped, _DST_ROOT)
    _MANIFEST.write_text(json.dumps({"batches": batches}, indent=2))
    _MANIFEST_TSV.write_text(
        "".join(
            "\t".join(
                [
                    b["name"],
                    f"{b['mesh_shape'][0]}x{b['mesh_shape'][1]}" if b["mesh_shape"] else "",
                    b["dispatch_axis"],
                    "tests/sweep_framework/" + b["vectors_dir"],
                ]
            )
            + "\n"
            for b in batches
        )
    )

    total = sum(b["vectors"] for b in batches)
    print(f"{total} vectors -> {len(batches)} batches (max {_max_vectors_per_job()} vectors/job)")
    print(f"{'batch':<26}{'mesh':>8}{'axis':>6}{'fab':>5}{'vectors':>9}{'modules':>9}")
    for b in batches:
        mesh = f"{b['mesh_shape'][0]}x{b['mesh_shape'][1]}" if b["mesh_shape"] else "?"
        print(f"{b['name']:<26}{mesh:>8}{b['dispatch_axis']:>6}{b['fabric']:>5}{b['vectors']:>9}{b['modules']:>9}")
    print(f"\nmanifest: {_MANIFEST}")


if __name__ == "__main__":
    main()
