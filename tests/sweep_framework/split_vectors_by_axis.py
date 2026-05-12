# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Updated splitter that also regenerates generation_manifest.json per axis."""
import json, re
from pathlib import Path

_Y9 = re.compile(r"""['"]y['"]\s*:\s*9(?!\d)""")
_X7 = re.compile(r"""['"]x['"]\s*:\s*7(?!\d)""")
_GRID_X = re.compile(r"x\s*=\s*(\d+)")
_GRID_Y = re.compile(r"y\s*=\s*(\d+)")
_GRID_XY_DASH = re.compile(r"compute_with_storage_grid_size=(\d+)-(\d+)")


def _scan_mc(obj, state):
    if obj is None or obj == "__ABSENT__":
        return
    if isinstance(obj, dict):
        data = obj.get("data") if "type" in obj and "data" in obj else obj
        if not isinstance(data, dict):
            return
        bt = str(data.get("buffer_type", ""))
        if "DRAM" in bt:
            return
        ss = data.get("shard_spec")
        if not ss or ss == "None":
            return
        if isinstance(ss, dict):
            grid = ss.get("grid", [])
            for cr in grid:
                for key in ("start", "end"):
                    p = cr.get(key, {}) if isinstance(cr, dict) else {}
                    if p.get("y") == 9:
                        state["needs_col"] = True
                    if p.get("x") == 7:
                        state["needs_row"] = True
            return
        rep = repr(ss)
        if _Y9.search(rep):
            state["needs_col"] = True
        if _X7.search(rep):
            state["needs_row"] = True


def _scan_pc(obj, state):
    if obj is None or obj == "__ABSENT__":
        return
    # Handle dict format: {'compute_with_storage_grid_size': {'x': 8, 'y': 8}, ...}
    if isinstance(obj, dict):
        grid = obj.get("compute_with_storage_grid_size")
        if isinstance(grid, dict):
            gx = grid.get("x", 0)
            gy = grid.get("y", 0)
            if isinstance(gx, (int, float)) and gx >= 8:
                state["needs_row"] = True
            if isinstance(gy, (int, float)) and gy >= 10:
                state["needs_col"] = True
            return
    # Handle string/repr format: "compute_with_storage_grid_size=8-8"
    pc_text = ""
    if isinstance(obj, dict):
        pc_text = str(obj.get("value", "")) or str(obj.get("repr", ""))
    else:
        pc_text = repr(obj)
    if "compute_with_storage_grid_size" not in pc_text:
        return
    # Try X-Y dash format first (e.g. "=8-8")
    dash_m = _GRID_XY_DASH.search(pc_text)
    if dash_m:
        gx, gy = int(dash_m.group(1)), int(dash_m.group(2))
        if gx >= 8:
            state["needs_row"] = True
        if gy >= 10:
            state["needs_col"] = True
        return
    # Fallback to x=N, y=N format
    idx = pc_text.find("compute_with_storage_grid_size")
    section = pc_text[idx : idx + 80]
    xm = _GRID_X.search(section)
    ym = _GRID_Y.search(section)
    if xm and int(xm.group(1)) >= 8:
        state["needs_row"] = True
    if ym and int(ym.group(1)) >= 10:
        state["needs_col"] = True


def vector_axis(vec):
    state = {"needs_col": False, "needs_row": False}
    for k, v in vec.items():
        if "memory_config" in k:
            _scan_mc(v, state)
        elif k.startswith("arg") and isinstance(v, dict) and "shard_spec" in v:
            _scan_mc(v, state)
        if k == "program_config":
            _scan_pc(v, state)
    if state["needs_col"]:
        return "col"
    if state["needs_row"]:
        return "row"
    return "col"


def split_file(in_path: Path, out_col: Path, out_row: Path):
    with in_path.open() as f:
        data = json.load(f)
    col_out = {}
    row_out = {}
    for suite, vectors in data.items():
        col_vec = {}
        row_vec = {}
        if not isinstance(vectors, dict):
            col_out[suite] = vectors
            continue
        for cid, vec in vectors.items():
            axis = vector_axis(vec) if isinstance(vec, dict) else "col"
            if axis == "row":
                row_vec[cid] = vec
            else:
                col_vec[cid] = vec
        if col_vec:
            col_out[suite] = col_vec
        if row_vec:
            row_out[suite] = row_vec
    out_col.parent.mkdir(parents=True, exist_ok=True)
    out_row.parent.mkdir(parents=True, exist_ok=True)
    wrote_col = wrote_row = False
    if col_out:
        with out_col.open("w") as f:
            json.dump(col_out, f, indent=2)
        wrote_col = True
    if row_out:
        with out_row.open("w") as f:
            json.dump(row_out, f, indent=2)
        wrote_row = True
    n_col = sum(len(v) for v in col_out.values() if isinstance(v, dict))
    n_row = sum(len(v) for v in row_out.values() if isinstance(v, dict))
    return n_col, n_row, wrote_col, wrote_row


# Canonical paths under the sweep framework. Hardcoded (not argv-driven) so
# this utility can't be pointed at arbitrary file paths — both for security
# (unsanitized user input in file paths flagged by Cycode SAST) and because
# the CI workflow always operates on the standard vectors_export tree.
_SWEEP_FRAMEWORK = Path(__file__).resolve().parent
_SRC_DIR = _SWEEP_FRAMEWORK / "vectors_export"
_DST_COL = _SWEEP_FRAMEWORK / "vectors_export_col"
_DST_ROW = _SWEEP_FRAMEWORK / "vectors_export_row"


def _assert_under_sweep_framework(p: Path) -> Path:
    """Resolve `p` and assert it stays under the sweep framework directory.

    All paths in this script are derived from `Path(__file__).resolve().parent`
    (no argv, no env), so this should always succeed; the explicit check is a
    defense-in-depth guard against future refactors that might accidentally
    point this utility at an attacker-controlled path. Returns the resolved
    path so callers can use it directly.
    """
    base = _SWEEP_FRAMEWORK.resolve()
    resolved = p.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"Refusing to operate on path outside sweep framework: {resolved}")
    return resolved


def main():
    src = _assert_under_sweep_framework(_SRC_DIR)
    dst_col = _assert_under_sweep_framework(_DST_COL)
    dst_row = _assert_under_sweep_framework(_DST_ROW)

    # Clean dst dirs to avoid stale files from prior runs.
    for d in (dst_col, dst_row):
        if d.exists():
            for f in d.iterdir():
                _assert_under_sweep_framework(f).unlink()
            d.rmdir()

    files_in_col = []
    files_in_row = []
    total_col = total_row = 0
    for f in sorted(src.glob("*.json")):
        if f.name == "generation_manifest.json":
            continue
        n_col, n_row, wc, wr = split_file(f, dst_col / f.name, dst_row / f.name)
        total_col += n_col
        total_row += n_row
        if wc:
            files_in_col.append(f.name)
        if wr:
            files_in_row.append(f.name)

    # Regenerate per-axis generation_manifest.json so the loader only references
    # files that actually exist in that axis directory.
    src_manifest_path = _assert_under_sweep_framework(src / "generation_manifest.json")
    if src_manifest_path.exists():
        with src_manifest_path.open() as f:
            base_manifest = json.load(f)
    else:
        base_manifest = {}
    for dst, files in [(dst_col, files_in_col), (dst_row, files_in_row)]:
        m = dict(base_manifest)
        m["vector_files"] = sorted(files)
        out_manifest = _assert_under_sweep_framework(dst / "generation_manifest.json")
        with out_manifest.open("w") as f:
            json.dump(m, f, indent=2)

    print(f"col: {len(files_in_col)} files, {total_col} vectors")
    print(f"row: {len(files_in_row)} files, {total_row} vectors")


if __name__ == "__main__":
    main()
