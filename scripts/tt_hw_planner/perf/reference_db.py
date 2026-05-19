# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load + query the best-known reference configurations.

This module hosts TWO reference databases that complement each other:

1. **Cluster-level references** (legacy, ``perf_reference.yaml``): keyed
   by ``(op, shape_sig, box)``. Used by ``program_config_tuner`` and the
   per-cluster red-diamond in the static report. Stays untouched.

2. **Module-class references** (new, ``references/*.yaml``): keyed by
   ``(arch_family, role_path, module_class, mesh_shape, dtype)``. Used
   by the suggestion engine and the cytoscape inspector to compare
   *your* model's submodules against the equivalent submodule of a
   well-optimized peer model.

The module-class layer is what powers the user-visible workflow: "this
attention block is 80% slower than Llama 3.1's; here is the optimizer
block (matmul_tuner@in1_block_w=4) that closed the gap on that model."

To add a reference, drop a YAML file in ``references/`` that conforms
to ``references/schema.yaml`` — see ``references/README.md`` for the
how-to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


REFERENCE_PATH = Path(__file__).parent / "perf_reference.yaml"


@dataclass
class ReferenceEntry:
    op: str
    box: str
    arch: str
    shape_sig: str
    fidelity: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    observed_device_us: float = 0.0
    source_run: str = ""


_CACHE: Optional[List[ReferenceEntry]] = None


def load_reference(path: Optional[Path] = None) -> List[ReferenceEntry]:
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE
    p = path or REFERENCE_PATH
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text()) or {}
    out: List[ReferenceEntry] = []
    for entry in raw.get("entries", []):
        out.append(
            ReferenceEntry(
                op=entry.get("op", ""),
                box=entry.get("box", ""),
                arch=entry.get("arch", ""),
                shape_sig=entry.get("shape_sig", ""),
                fidelity=entry.get("fidelity", "HiFi2"),
                kwargs=dict(entry.get("kwargs") or {}),
                observed_device_us=float(entry.get("observed_device_us", 0.0)),
                source_run=entry.get("source_run", ""),
            )
        )
    if path is None:
        _CACHE = out
    return out


def find_best(op: str, shape_sig: str, box: str) -> Optional[ReferenceEntry]:
    """Exact match on (op, shape_sig, box). Returns None if not present."""
    for e in load_reference():
        if e.op == op and e.box == box and e.shape_sig == shape_sig:
            return e
    return None


def find_by_op_box(op: str, box: str) -> List[ReferenceEntry]:
    return [e for e in load_reference() if e.op == op and e.box == box]


def list_op_boxes() -> List[Tuple[str, str]]:
    return sorted({(e.op, e.box) for e in load_reference()})


# ---------------------------------------------------------------------------
# Module-class reference layer (new in stage 1 of the cytoscape work)
# ---------------------------------------------------------------------------


REFERENCES_DIR = Path(__file__).parent / "references"


@dataclass
class ModuleReference:
    """One row of a module-class reference YAML.

    Captures: what a "well-optimized" version of this submodule looks
    like (perf metrics + the optimizer blocks that got it there) so a
    user-run can compare against it.
    """

    # Provenance (the run this reference was distilled from)
    reference_id: str
    model_id: str
    arch_family: str
    box: str
    mesh_shape: Tuple[int, int]
    dtype: str
    source_run_id: Optional[str] = None
    curator: Optional[str] = None

    # Module identity (what the matcher keys on)
    role_path: str = ""
    module_class: str = ""
    role: str = ""  # short tag like "attention_qkv_projection", "mlp_gate", "sdpa"

    # Performance fingerprint
    runtime_ms_p50: float = 0.0
    fpu_util_pct: float = 0.0
    dram_bw_util_pct: float = 0.0
    noc_util_pct: float = 0.0

    # Configuration that produced these metrics
    config_summary: Dict[str, Any] = field(default_factory=dict)
    optimizer_blocks_applied: List[str] = field(default_factory=list)
    expected_speedup_from_baseline: Optional[str] = None

    # Free-form
    notes: str = ""

    @property
    def quality(self) -> str:
        curator = (self.curator or "").strip().lower()
        if curator == "synthetic":
            return "synthetic"
        if self.source_run_id:
            return "curated"
        return "uncurated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "model_id": self.model_id,
            "arch_family": self.arch_family,
            "box": self.box,
            "mesh_shape": list(self.mesh_shape),
            "dtype": self.dtype,
            "source_run_id": self.source_run_id,
            "curator": self.curator,
            "quality": self.quality,
            "role_path": self.role_path,
            "module_class": self.module_class,
            "role": self.role,
            "runtime_ms_p50": self.runtime_ms_p50,
            "fpu_util_pct": self.fpu_util_pct,
            "dram_bw_util_pct": self.dram_bw_util_pct,
            "noc_util_pct": self.noc_util_pct,
            "config_summary": dict(self.config_summary),
            "optimizer_blocks_applied": list(self.optimizer_blocks_applied),
            "expected_speedup_from_baseline": self.expected_speedup_from_baseline,
            "notes": self.notes,
        }


_MODULE_REF_CACHE: Optional[List[ModuleReference]] = None


def _coerce_mesh(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str) and "," in value:
        a, b = value.split(",", 1)
        return int(a), int(b)
    return (0, 0)


def load_module_references(refresh: bool = False) -> List[ModuleReference]:
    """Load every YAML file in ``references/`` that matches the schema.

    The directory layout is one YAML per reference run. Files starting
    with ``_`` (e.g. ``_template.yaml``) and the ``schema.yaml`` and
    ``README.md`` files are skipped. Malformed YAMLs are skipped with a
    warning rather than crashing the whole DB.
    """
    global _MODULE_REF_CACHE
    if _MODULE_REF_CACHE is not None and not refresh:
        return _MODULE_REF_CACHE
    out: List[ModuleReference] = []
    if not REFERENCES_DIR.exists():
        _MODULE_REF_CACHE = out
        return out
    for yaml_path in sorted(REFERENCES_DIR.glob("*.yaml")):
        if yaml_path.name.startswith("_") or yaml_path.name == "schema.yaml":
            continue
        try:
            data = yaml.safe_load(yaml_path.read_text()) or {}
        except yaml.YAMLError as exc:
            print(f"reference_db: skipping malformed reference {yaml_path.name}: {exc}")
            continue
        prov = data.get("provenance") or {}
        ref_id = yaml_path.stem
        common = dict(
            reference_id=ref_id,
            model_id=str(prov.get("model_id", "")),
            arch_family=str(prov.get("arch_family", "")),
            box=str(prov.get("box", "")),
            mesh_shape=_coerce_mesh(prov.get("mesh_shape")),
            dtype=str(prov.get("dtype", "")),
            source_run_id=prov.get("source_run_id"),
            curator=prov.get("curator"),
        )
        for entry in data.get("modules") or []:
            metrics = entry.get("metrics") or {}
            ref = ModuleReference(
                **common,
                role_path=str(entry.get("role_path", "")),
                module_class=str(entry.get("module_class", "")),
                role=str(entry.get("role", "")),
                runtime_ms_p50=float(metrics.get("runtime_ms_p50") or 0.0),
                fpu_util_pct=float(metrics.get("fpu_util_pct") or 0.0),
                dram_bw_util_pct=float(metrics.get("dram_bw_util_pct") or 0.0),
                noc_util_pct=float(metrics.get("noc_util_pct") or 0.0),
                config_summary=dict(entry.get("config_summary") or {}),
                optimizer_blocks_applied=list(entry.get("optimizer_blocks_applied") or []),
                expected_speedup_from_baseline=entry.get("expected_speedup_from_baseline"),
                notes=str(entry.get("notes", "")),
            )
            out.append(ref)
    _MODULE_REF_CACHE = out
    return out


# Architecturally-similar families: when an exact-family match fails we
# fall back to these aliases. Conservative on purpose — only families
# whose transformer block shape is genuinely close enough that an
# optimizer block from one is likely to apply to the other.
_FAMILY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "qwen": ("qwen", "llama"),
    "qwen2": ("qwen2", "qwen", "llama"),
    "llama": ("llama", "qwen"),
    "mistral": ("mistral", "llama"),
    "phi": ("phi", "llama"),
    "gemma": ("gemma", "llama"),
    "mamba": ("mamba",),
    "gpt2": ("gpt2",),
    "gptj": ("gptj", "gpt2"),
}


def _family_candidates(arch_family: str) -> Tuple[str, ...]:
    """Return the ordered list of architecture families to try when
    matching a user-module against the reference DB."""
    if not arch_family:
        return ("",)
    return _FAMILY_ALIASES.get(arch_family.lower(), (arch_family.lower(),))


def find_module_reference(
    *,
    attribute_path: str,
    module_class: str,
    arch_family: str,
    mesh_shape: Tuple[int, int],
    dtype: Optional[str] = None,
    box: Optional[str] = None,
) -> Optional[ModuleReference]:
    """Find the best-matching reference for a user-module.

    Matching is graded:

      1. Try the user's exact arch_family first; then aliased families.
      2. Within each family, prefer matches with the same role_path
         (after wildcarding "*") AND same mesh_shape AND same dtype.
      3. Fall back to role_path match with different mesh/dtype if no
         exact-mesh match exists (the inspector will note the mismatch).
      4. Fall back to ``module_class`` match (broadest; used when the
         user's module isn't in the role_path catalog).

    Returns None if nothing matches at any level.
    """
    from .module_graph import matches_role_path

    refs = load_module_references()
    if not refs:
        return None

    families = _family_candidates(arch_family)
    norm_dtype = (dtype or "").lower()
    target_mesh = tuple(mesh_shape)

    best: Optional[Tuple[int, ModuleReference]] = None

    for fam in families:
        for ref in refs:
            if fam and ref.arch_family.lower() != fam:
                continue
            score = 0
            # Role-path match is the strongest signal.
            if ref.role_path and matches_role_path(attribute_path, ref.role_path):
                score += 100
            # Module-class match is the safety net.
            if ref.module_class and module_class and ref.module_class == module_class:
                score += 30
            if score == 0:
                continue
            # Bonuses for matching mesh and dtype.
            if ref.mesh_shape == target_mesh:
                score += 20
            if norm_dtype and ref.dtype.lower() == norm_dtype:
                score += 10
            if box and ref.box == box:
                score += 5
            # Family-priority bonus: exact-family match wins over aliased.
            if ref.arch_family.lower() == (arch_family or "").lower():
                score += 50
            if best is None or score > best[0]:
                best = (score, ref)
        if best is not None and best[0] >= 100:
            # Strong match found — don't keep trying aliased families.
            return best[1]

    return best[1] if best is not None else None


def list_module_references() -> List[Tuple[str, str, str, Tuple[int, int], str]]:
    """Return (reference_id, model_id, arch_family, mesh, dtype) tuples
    for the UI dropdown."""
    seen: Dict[str, Tuple[str, str, str, Tuple[int, int], str]] = {}
    for r in load_module_references():
        if r.reference_id not in seen:
            seen[r.reference_id] = (
                r.reference_id,
                r.model_id,
                r.arch_family,
                r.mesh_shape,
                r.dtype,
            )
    return sorted(seen.values())
