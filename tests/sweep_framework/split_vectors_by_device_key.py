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

Modules sharing a device key share the job, EXCEPT CCL collectives, which get their own batch:
they open their own device per vector (a submesh-traced CCL vector opens the full galaxy), and
the device profiler is disabled for a whole job whenever any of its modules is CCL. A batch is
bounded by VECTOR COUNT, not module count, which is what actually caps a job's runtime.

Writes one directory per batch plus a batch manifest for the CI matrix to consume.
"""

import json
import math
import re
import sys
from pathlib import Path

from split_vectors_by_axis import vector_dispatch_axis_hint

sys.path.insert(0, str(Path(__file__).resolve().parent / "framework"))
from constants import CCL_OP_TOKENS, strip_grouping_suffix  # noqa: E402

# Time model for splitting an oversized batch across jobs. Deliberately conservative: the
# 3s/vector figure comes from slice (a light op) -- conv2d/matmul/CCL are slower, so a batch
# near the cap may still need manual attention. Tune from measured per-module timings.
DEVICE_OPEN_MINUTES = 4
SECONDS_PER_VECTOR = 3.0
USABLE_BUDGET_MINUTES = 45

_SWEEP_FRAMEWORK = Path(__file__).resolve().parent
_SRC_DIR = _SWEEP_FRAMEWORK / "vectors_export"
_DST_ROOT = _SWEEP_FRAMEWORK / "vectors_export_by_device"
_MANIFEST = _DST_ROOT / "batch_manifest.json"
# Same content as the JSON, one batch per line, tab separated: name, mesh ("4x8"), axis, dir.
# Kept for local/ad-hoc use and as the workflow's fallback path -- embedding a multi-line
# python one-liner inside a YAML block scalar is fragile (unindented lines silently break
# the workflow).
_MANIFEST_TSV = _DST_ROOT / "batch_manifest.tsv"
# Path a batch directory is reported under, relative to the repo. Both the CI matrix and the
# run job build this string, so it lives here rather than being spelled out in either.
VECTORS_BATCH_ROOT = "tests/sweep_framework/vectors_export_by_device"


def _max_vectors_per_job() -> int:
    return int((USABLE_BUDGET_MINUTES - DEVICE_OPEN_MINUTES) * 60 / SECONDS_PER_VECTOR)


# A serialized mesh shape: '[4, 8]', '(4, 8)', '4x8' or bare '4, 8'. Matched with an explicit
# pattern rather than handed to ast.literal_eval -- these strings come out of vector JSON
# produced by the tracer, and a literal evaluator accepts arbitrarily large/nested literals
# from that input (CWE-400). Two bounded integers is the entire grammar we need.
_PAIR_RE = re.compile(r"^\s*[\[(]?\s*(\d{1,4})\s*[,x]\s*(\d{1,4})\s*[\])]?\s*$")


def _parse_pair(value):
    """Parse a mesh shape that may be a list, tuple, or the string '[4, 8]'."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = _PAIR_RE.match(value)
        if match:
            return (int(match.group(1)), int(match.group(2)))
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


def batch_name(mesh, axis, fabric, part=None, total_parts=1, solo=None):
    """Deterministic batch/directory name. ``solo`` tags a batch pinned to one module (CCL --
    see plan_batches), keeping its directory distinct from the shared batch's."""
    mesh_str = f"{mesh[0]}x{mesh[1]}" if mesh else "unknown"
    name = f"mesh{mesh_str}_{axis}_{fabric}"
    if solo:
        name = f"{name}_{solo}"
    if total_parts > 1:
        name = f"{name}_p{part}"
    return name


def _base_module(file_name):
    """'model_traced.conv2d_model_traced.mesh_4x8.json' -> 'model_traced.conv2d_model_traced'."""
    return strip_grouping_suffix(Path(file_name).stem)


def _is_ccl(file_name):
    """Whether this file's module is a CCL collective, which must not share a job."""
    return any(token in _base_module(file_name) for token in CCL_OP_TOKENS)


def _solo_token(file_name):
    """Short, filename-safe tag for an isolated module's batch ('...all_gather_async_model_traced'
    -> 'all_gather_async')."""
    return _base_module(file_name).split(".")[-1].replace("_model_traced", "") or "solo"


def source_manifest(src_dir: Path):
    """The export's generation_manifest.json, or {} if unreadable."""
    path = src_dir / "generation_manifest.json"
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def group_vectors(src_dir: Path, manifest=None):
    """Return {device_key: {suite: {vector_id: vector}}} plus the source file each came from.

    The manifest's vector_files list is the authoritative index of what THIS run generated
    (vector_source treats it that way too). Honour it: an export directory can also hold
    files left by an earlier run, and partitioning those would schedule jobs for vectors the
    run never asked for -- and bill their time against the lane's budget.
    """
    declared = (manifest or {}).get("vector_files")
    declared = set(declared) if isinstance(declared, list) and declared else None
    grouped = {}
    for path in sorted(src_dir.glob("*.json")):
        if path.name == "generation_manifest.json":
            continue
        if declared is not None and path.name not in declared:
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            # A file the manifest DECLARED is not optional. Skipping it would drop vectors the
            # generation step asked for, and because the matrix planner and the writer both
            # call this function they would agree on the reduced set -- so the loss would show
            # up nowhere and the run would report success over a silently smaller suite. Only
            # directory-scan mode (no manifest) stays best-effort, where a stray unparseable
            # file is genuinely not ours to run.
            if declared is not None:
                raise RuntimeError(f"generation manifest declares {path.name} but it could not be read: {e}") from e
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


def plan_batches(grouped, dst_root: Path = _DST_ROOT):
    """Decide the batches WITHOUT writing anything.

    Kept separate from write_batches so the CI matrix can derive the exact same batch set
    (names included) at matrix-build time, where only the plan is needed and writing 17k
    vectors would be wasted work. Both callers therefore agree by construction rather than
    by two implementations that have to be kept in step.
    """
    cap = _max_vectors_per_job()
    batches = []
    for (mesh, axis, fabric), files in sorted(grouped.items(), key=lambda kv: -_count(kv[1])):
        # Solo modules keep their own job inside the device key. Grouping by device rather
        # than by module must not quietly undo that isolation: conv2d's heavy convs deadlock
        # the dispatch hang-detector once device state has accumulated from OTHER modules in
        # the same process, and run clean alone (see LEAD_MODELS_BATCH_POLICY.solo_modules).
        # CCL modules get their own batch inside the device key; everything else shares one.
        #
        # conv2d used to be isolated too, for cross-module device-state accumulation across a
        # job that opened the device repeatedly. A device-key job opens it once, and merging
        # was validated on lead-models run 30699228172 (12 conv2d vectors among 29 modules,
        # 582/582, no hang), so conv2d now shares.
        #
        # CCL cannot share, for two independent reasons:
        #  1. all_gather_async is the only module whose mesh_device_fixture() yields None -- it
        #     opens its own device per vector via _full_galaxy_mesh_for(), which for a 2D
        #     SUBMESH of the host opens the FULL galaxy and carves the submesh out (opening a
        #     submesh directly fails fabric router sync on its boundary links). So in a submesh
        #     batch the job opens the declared mesh for every other module and then a different,
        #     larger one for CCL -- the in-job mesh switch this whole design exists to remove.
        #     Observed in run 30699228172 mesh4x4_col_2d: 57 vectors at 4x4, then
        #     MeshShape([8,4]). Two vectors with IDENTICAL metadata, so no grouping rule can
        #     separate them -- the device opened is a property of the MODULE, not the vector.
        #     On a healthy 32-card host this silently succeeds, which is why it needs isolating
        #     rather than detecting.
        #  2. The device profiler is a process-global toggle and _should_skip_device_profiler
        #     disables it for the WHOLE job when ANY module in the selector is CCL. Merged, that
        #     cost device-perf for all 41 modules of mesh8x4_col_2d; solo, it costs only CCL's
        #     own batch.
        ccl_groups = {}
        shared = {}
        for name, suites in files.items():
            if _is_ccl(name):
                # Group by MODULE, not per file: one module can contribute several files to the
                # same device key (the key comes from each vector's recorded shape, not the
                # filename), and one batch per file would mint duplicate batch names whose
                # directories overwrite each other.
                ccl_groups.setdefault(_solo_token(name), {})[name] = suites
            else:
                shared[name] = suites

        for solo, subset in sorted(ccl_groups.items(), key=lambda kv: kv[0]):
            _plan_subset(batches, subset, mesh, axis, fabric, solo, cap, dst_root)
        if shared:
            _plan_subset(batches, shared, mesh, axis, fabric, None, cap, dst_root)

    duplicates = {b["name"] for b in batches if sum(1 for o in batches if o["name"] == b["name"]) > 1}
    if duplicates:
        # A duplicate name means two batches share one directory, and the later write wins.
        # Fail loudly rather than dispatch a run that quietly skips vectors.
        raise ValueError(f"device-key batch names are not unique: {sorted(duplicates)}")
    return batches


def _is_solo(file_name):
    return _base_module(file_name) in SOLO_MODULES


def _plan_subset(batches, files, mesh, axis, fabric, solo, cap, dst_root):
    """Chunk one subset of a device key's files into cap-sized batches."""
    total = _count(files)
    parts = max(1, math.ceil(total / cap))
    # Split by FILE (a file is one module x one hw/mesh variant), never mid-file, so a
    # module's vectors stay together and the per-module device/program cache still helps.
    chunks = _chunk_files(files, parts)
    for index, chunk in enumerate(chunks, start=1):
        name = batch_name(mesh, axis, fabric, index, len(chunks), solo)
        batches.append(
            {
                "name": name,
                "mesh_shape": list(mesh) if mesh else None,
                "dispatch_axis": axis,
                "fabric": fabric,
                "vectors": _count(chunk),
                "modules": len(chunk),
                "vector_files": sorted(chunk),
                "file_vectors": {f: sum(len(v) for v in suites.values()) for f, suites in chunk.items()},
                "vectors_dir": f"{dst_root.name}/{name}",
                "_files": chunk,
            }
        )
    return batches


def write_batches(grouped, dst_root: Path, src_manifest=None):
    """Write one directory per batch (splitting oversized ones) and return the manifest."""
    batches = plan_batches(grouped, dst_root)
    for batch in batches:
        out_dir = dst_root / batch["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for file_name, suites in batch["_files"].items():
            (out_dir / file_name).write_text(json.dumps(suites, indent=2))
        # Inherit the SOURCE export's vector_grouping_mode verbatim. It must stay "mesh" or
        # "hw": vector_source rejects anything else outright, and it selects the mesh- vs
        # hardware-capability load filter that keeps a lane from running another lane's
        # vectors. A batch is a re-partition of the same vectors, not a new grouping scheme,
        # so the mode is copied rather than restated.
        manifest = {
            "vector_grouping_mode": (src_manifest or {}).get("vector_grouping_mode", "mesh"),
            "vector_files": batch["vector_files"],
            "device_key_batch": {
                "name": batch["name"],
                "mesh_shape": batch["mesh_shape"],
                "dispatch_axis": batch["dispatch_axis"],
                "fabric": batch["fabric"],
            },
        }
        (out_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))
    for batch in batches:
        del batch["_files"]
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


def main():
    if not _SRC_DIR.is_dir():
        raise SystemExit(f"No vectors at {_SRC_DIR}")
    src_manifest = source_manifest(_SRC_DIR)
    grouped = group_vectors(_SRC_DIR, src_manifest)
    _DST_ROOT.mkdir(parents=True, exist_ok=True)
    batches = write_batches(grouped, _DST_ROOT, src_manifest)
    _MANIFEST.write_text(json.dumps({"batches": batches}, indent=2))
    _MANIFEST_TSV.write_text(
        "".join(
            "\t".join(
                [
                    b["name"],
                    f"{b['mesh_shape'][0]}x{b['mesh_shape'][1]}" if b["mesh_shape"] else "",
                    b["dispatch_axis"],
                    f"{VECTORS_BATCH_ROOT}/{b['name']}",
                ]
            )
            + "\n"
            for b in batches
        )
    )

    total = sum(b["vectors"] for b in batches)
    print(f"{total} vectors -> {len(batches)} batches (max {_max_vectors_per_job()} vectors/job)")
    print(f"{'batch':<26}{'mesh':>8}{'axis':>6}{'fab':>5}{'vectors':>9}{'modules':>9}")
    cap = _max_vectors_per_job()
    for b in batches:
        mesh = f"{b['mesh_shape'][0]}x{b['mesh_shape'][1]}" if b["mesh_shape"] else "?"
        # A batch can exceed the cap only when ONE file does: files are never split, so the
        # module's vectors stay in one job (as they did under module batching). Flagged, not
        # silently accepted, because it is the one case the time model cannot bound.
        over = "  <-- over cap, single file" if b["vectors"] > cap else ""
        print(
            f"{b['name']:<34}{mesh:>8}{b['dispatch_axis']:>6}{b['fabric']:>5}"
            f"{b['vectors']:>9}{b['modules']:>9}{over}"
        )
    print(f"\nmanifest: {_MANIFEST}")


if __name__ == "__main__":
    main()
