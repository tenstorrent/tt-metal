# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape-matrix loader and row-helper utilities.

The Phase 0 capture writes dedup JSON files under ``shape_matrix/``. This
module loads those, filters rows by op name / phase, builds stable
pytest ``-k``-filterable IDs, and exposes per-row tags so the test files
can build kwargs/sharding decisions consistently.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SHAPE_MATRIX_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "shape_matrix"))


DEFAULT_DEDUP_SOURCES = [
    os.path.join(_SHAPE_MATRIX_DIR, "text_ops_dedup.json"),
    os.path.join(_SHAPE_MATRIX_DIR, "vision_ops_dedup.json"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_one(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shape matrix file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at {path}, got {type(data).__name__}")
    return data


def load_op_matrix(
    op_name: str,
    sources: Optional[List[str]] = None,
    *,
    include_vision: bool = True,
    include_text: bool = True,
) -> List[Dict[str, Any]]:
    """Load deduped op records matching ``op_name``.

    ``sources`` defaults to the text+vision dedup files under
    ``shape_matrix/``. The ``include_vision`` / ``include_text`` flags
    operate on each record's ``phase`` field.
    """
    if sources is None:
        sources = []
        if include_text:
            sources.append(DEFAULT_DEDUP_SOURCES[0])
        if include_vision:
            sources.append(DEFAULT_DEDUP_SOURCES[1])

    rows: List[Dict[str, Any]] = []
    for path in sources:
        for r in _load_one(path):
            if r.get("op") != op_name:
                continue
            phase = r.get("phase", "")
            if "text" in phase and not include_text:
                continue
            if "vision" in phase and not include_vision:
                continue
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Row metadata helpers
# ---------------------------------------------------------------------------


def _short_dtype(dt: str) -> str:
    """``'DataType.BFLOAT16'`` -> ``'bf16'``; ``'DataType.BFLOAT8_B'`` -> ``'bfp8'`` etc."""
    if not isinstance(dt, str):
        return "?"
    s = dt.upper()
    if "BFLOAT16" in s:
        return "bf16"
    if "BFLOAT8" in s:
        return "bfp8"
    if "BFLOAT4" in s:
        return "bfp4"
    if "FLOAT32" in s:
        return "fp32"
    if "UINT32" in s:
        return "u32"
    if "INT32" in s:
        return "i32"
    # Fall back to the token after the dot.
    return s.split(".")[-1].lower()


def _short_fidelity(fid: str) -> str:
    if not isinstance(fid, str):
        return "?"
    return fid.split(".")[-1]  # 'MathFidelity.HiFi2' -> 'HiFi2'


def _short_module(module_path: str) -> str:
    """Pick the last component of ``module_path`` and strip indexing."""
    if not isinstance(module_path, str) or not module_path:
        return "anon"
    last = module_path.split("/")[-1]
    return last.split("[", 1)[0]


def make_row_id(row: Dict[str, Any]) -> str:
    """Build a stable, pytest -k filterable id.

    Example: ``linear_text_b1_M14_K1536_N2048_bf16xbfp8_HiFi2``.
    """
    op = row.get("op", "op").replace("ttnn.", "").replace(".", "_")
    phase = row.get("phase", "?")
    inputs = row.get("inputs", []) or []
    if not inputs:
        return f"{op}_{phase}_call{row.get('call_id', '?')}"

    in0 = inputs[0]
    shape = in0.get("shape", [])
    dim_str = "x".join(str(s) for s in shape)
    in0_dt = _short_dtype(in0.get("dtype", ""))
    other_dts = "x".join(_short_dtype(i.get("dtype", "")) for i in inputs[1:]) or "."

    fid = ""
    kwargs = row.get("kwargs", {}) or {}
    ckc = kwargs.get("compute_kernel_config")
    if isinstance(ckc, dict):
        fid = _short_fidelity(ckc.get("fields", {}).get("math_fidelity", ""))

    mod = _short_module(row.get("module_path", ""))

    parts = [op, phase, mod, f"in{dim_str}", in0_dt]
    if other_dts != ".":
        parts.append(f"w{other_dts}")
    if fid:
        parts.append(fid)
    parts.append(f"cid{row.get('call_id', '?')}")

    # Make the id pytest-safe: alnum, underscore, hyphen, dot.
    safe = "_".join(parts)
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in safe)
    return safe


def make_row_tags(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a flat dict of high-level tags useful for assertions / filtering.

    Keys:
        phase, transpose_b, num_devices, mesh_shape, sharding_axes,
        math_fidelity, program_config_kind, op, module_class.
    """
    kwargs = row.get("kwargs", {}) or {}
    inputs = row.get("inputs", []) or []

    # transpose_b can appear directly in kwargs (vision_patch_embed) — see
    # phase-0 finding #2.
    transpose_b = bool(kwargs.get("transpose_b", False))

    # num_devices / mesh_shape: read from any tensor's device_shape if present.
    num_devices = 1
    mesh_shape = (1, 1)
    for t in inputs + (row.get("output") or []):
        ds = t.get("device_shape") if isinstance(t, dict) else None
        if isinstance(ds, str) and ds.startswith("MeshShape"):
            from .mesh_gather import _parse_mesh_shape  # local import to dodge cycle

            ms = _parse_mesh_shape(ds)
            if ms is not None:
                mesh_shape = ms
                num_devices = ms[0] * ms[1]
                break

    # math_fidelity / program_config_kind
    math_fidelity = ""
    program_config_kind = ""
    ckc = kwargs.get("compute_kernel_config")
    if isinstance(ckc, dict):
        math_fidelity = _short_fidelity(ckc.get("fields", {}).get("math_fidelity", ""))
    pc = kwargs.get("program_config")
    if isinstance(pc, dict):
        program_config_kind = pc.get("kind", "")

    return {
        "phase": row.get("phase", ""),
        "op": row.get("op", ""),
        "module_class": _short_module(row.get("module_path", "")),
        "transpose_b": transpose_b,
        "num_devices": num_devices,
        "mesh_shape": mesh_shape,
        "sharding_axes": None,  # No explicit info captured in Phase 0; Phase 2 may infer.
        "math_fidelity": math_fidelity,
        "program_config_kind": program_config_kind,
    }
