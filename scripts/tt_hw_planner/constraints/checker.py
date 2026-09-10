"""Run the catalog against a component's on-disk metadata.

The checker is pure: given the captures directory + opplan path, it
loads the lightweight manifest.json + opplan.json, builds an
``OpCallContext``, and runs every constraint in the catalog. No torch
or HF needed — manifests already carry the captured shape/dtype info.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .catalog import Catalog, OpCallContext, Violation, default_catalog


def _extract_shapes_from_manifest(manifest: dict) -> tuple[list, list]:
    """Walk the args block in a capture manifest and pull every tensor's
    shape + dtype. Tuples/lists of tensors are flattened in DFS order so
    multi-input components get every input represented."""
    shapes: List[List[int]] = []
    dtypes: List[str] = []

    def _walk(node: object) -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("kind")
        if kind == "tensor":
            sh = node.get("shape")
            if isinstance(sh, list):
                shapes.append([int(x) for x in sh])
                dtypes.append(str(node.get("dtype") or ""))
            return
        if kind in ("tuple", "list"):
            for item in node.get("items", []) or []:
                _walk(item)
            return
        if kind == "dict":
            items = node.get("items") or {}
            for v in items.values():
                _walk(v)
            return

    args = manifest.get("args") or {}
    kwargs = manifest.get("kwargs") or {}
    _walk(args)
    _walk(kwargs)
    return shapes, dtypes


def _extract_output(manifest: dict) -> tuple[Optional[List[int]], Optional[str]]:
    out = manifest.get("output") or {}
    if not isinstance(out, dict):
        return None, None
    if out.get("kind") == "tensor":
        sh = out.get("shape")
        if isinstance(sh, list):
            return [int(x) for x in sh], str(out.get("dtype") or "")
    return None, None


def build_context(
    *,
    component_name: str,
    hf_class_name: str,
    manifest_path: Optional[Path] = None,
    opplan_path: Optional[Path] = None,
) -> OpCallContext:
    """Load the on-disk manifest + opplan and build an OpCallContext.

    Missing files are tolerated — the returned context just has empty
    shape lists, and constraints that need that info won't fire.
    """
    input_shapes: List[List[int]] = []
    input_dtypes: List[str] = []
    output_shape: Optional[List[int]] = None
    output_dtype: Optional[str] = None
    opplan: dict = {}

    if manifest_path is not None and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            input_shapes, input_dtypes = _extract_shapes_from_manifest(manifest)
            output_shape, output_dtype = _extract_output(manifest)
        except Exception:
            pass

    if opplan_path is not None and opplan_path.is_file():
        try:
            opplan = json.loads(opplan_path.read_text())
        except Exception:
            opplan = {}

    return OpCallContext(
        component_name=component_name,
        hf_class_name=hf_class_name,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shape=output_shape,
        output_dtype=output_dtype,
        opplan=opplan,
    )


def check_component(
    *,
    component_name: str,
    hf_class_name: str,
    manifest_path: Optional[Path] = None,
    opplan_path: Optional[Path] = None,
    catalog: Optional[Catalog] = None,
) -> List[Violation]:
    """High-level entry point: load context + run catalog. Returns the
    list of violations that fired (possibly empty)."""
    cat = catalog or default_catalog()
    ctx = build_context(
        component_name=component_name,
        hf_class_name=hf_class_name,
        manifest_path=manifest_path,
        opplan_path=opplan_path,
    )
    return cat.run(ctx)
