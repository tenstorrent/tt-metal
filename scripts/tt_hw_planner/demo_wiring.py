"""Multi-component demo wiring.

After the auto-loop converges, MANY components may have graduated
(native ttnn forward, PCC >= 0.99). The end-to-end demo should exercise
ALL of them on device — not pick a single primary. This module produces
the wiring spec the demo template iterates over.

Antichain selection: when a parent and child both graduate, the parent's
TT forward already contains the child's logic end-to-end. Wiring both
would be redundant. We pick the largest non-overlapping units (outermost
paths win).

Components in COLD or KERNEL_MISSING bucket do NOT appear here; they
remain on CPU as HF reference automatically (no wiring needed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GraduatedComponent:
    """A component that graduated (native ttnn forward, PCC validated) AND
    has on-disk capture artifacts + a submodule_path resolved."""

    name: str
    safe: str
    submodule_path: str
    op_count: int = 0


def collect_graduated_components(
    demo_dir: Path,
    components: List[Dict[str, Any]],
    *,
    model_id: Optional[str] = None,
) -> List[GraduatedComponent]:
    """Walk ``components`` and return every entry that:

    * has ``status == "NEW"``,
    * has a graduated stub on disk (native ttnn forward, not autofill fallback),
    * has the four capture artifacts (``args.pt``/``kwargs.pt``/``manifest.json``)
      under ``_captured/<safe>/``,
    * has a non-empty ``submodule_path`` recorded in the capture manifest,
    * is NOT marked COLD by the cold-evidence record (if ``model_id`` is
      provided AND a hot_cold.json exists for it). Stage 4 empirical
      bench can demote a graduated stub to COLD (e.g. CPU_WINS because
      the kernel exists but transfer cost dominates). Demo wiring
      should respect that verdict — putting a CPU_WINS component on
      device makes the demo slower than the pure-CPU reference.

    Components missing any of these criteria are silently skipped — they
    cannot be wired into the demo without ambiguity.
    """
    from .bringup_loop import (
        _component_op_count,
        _safe_id,
        _stub_has_graduated_from_autofill,
    )

    # Pre-load cold-evidence (if available) to skip CPU_WINS components.
    cold_kinds: Dict[str, str] = {}
    if model_id is not None:
        try:
            from .overlay_manager import load_hot_cold

            cold_kinds = load_hot_cold(model_id)
        except Exception:
            cold_kinds = {}

    out: List[GraduatedComponent] = []
    for comp in components:
        if comp.get("status") != "NEW":
            continue
        name = comp.get("name") or ""
        if not name:
            continue
        # Bench-aware filter: if the cold-evidence says this component
        # is COLD (incl. bench CPU_WINS demotion), don't wire it.
        if cold_kinds.get(name, "").upper() == "COLD":
            continue
        safe = _safe_id(name)
        stub = demo_dir / "_stubs" / f"{safe}.py"
        if not _stub_has_graduated_from_autofill(stub):
            continue
        cap_dir = demo_dir / "_captured" / safe
        required = (
            cap_dir / "args.pt",
            cap_dir / "kwargs.pt",
            cap_dir / "manifest.json",
        )
        if not all(p.is_file() for p in required):
            continue
        try:
            mani = json.loads((cap_dir / "manifest.json").read_text())
        except Exception:
            continue
        path = (mani.get("submodule_path") or "").strip()
        if not path:
            continue
        out.append(
            GraduatedComponent(
                name=name,
                safe=safe,
                submodule_path=path,
                op_count=_component_op_count(demo_dir, safe),
            )
        )
    return out


def _is_ancestor_or_equal(a: str, b: str) -> bool:
    """True iff ``a == b`` OR ``a`` is a strict ancestor of ``b`` in the
    dotted/indexed HF submodule-path notation.

    Examples::

      ("vision_encoder", "vision_encoder.neck")       -> True
      ("vision_encoder", "vision_encoder")            -> True
      ("vision_encoder.neck", "vision_encoder")       -> False
      ("vision_encoder", "vision_encoder_2.neck")     -> False  (word boundary)
      ("encoder.layers[0]", "encoder.layers[0].mlp")  -> True
    """
    if not a or not b:
        return False
    if a == b:
        return True
    if not b.startswith(a):
        return False
    rest = b[len(a) :]
    return rest.startswith(".") or rest.startswith("[")


def select_maximal_antichain(
    components: List[GraduatedComponent],
) -> List[GraduatedComponent]:
    """Pick the largest non-overlapping subset of graduated components.

    Outermost paths win: when both ``vision_encoder`` and
    ``vision_encoder.neck`` are graduated, only ``vision_encoder`` is
    selected (its TT stub subsumes the child's logic). The child stub
    keeps its isolated PCC test as standalone validation.

    Sort key: ``(path-length asc, path asc, name asc)``. Ensures parents
    appear before children with deterministic tiebreak.
    """
    items = sorted(
        components,
        key=lambda c: (len(c.submodule_path), c.submodule_path, c.name),
    )
    claimed: List[str] = []
    selected: List[GraduatedComponent] = []
    for c in items:
        if any(_is_ancestor_or_equal(p, c.submodule_path) for p in claimed):
            continue
        claimed.append(c.submodule_path)
        selected.append(c)
    return selected


def build_wiring_specs(
    *,
    demo_dir: Path,
    components: List[Dict[str, Any]],
    repo_root: Path,
    model_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """End-to-end pipeline: collect graduated components, prune via
    antichain selection, resolve each one's stub import path.

    Returns a list of dicts ready to be embedded as a Python literal in
    the demo template::

      [
        {"path": "<submodule_path>",
         "import": "models.demos.<model>.demo._stubs.<safe_id>",
         "name": "<component_name>",
         "safe": "<safe_id>"},
        ...
      ]

    Empty list iff no graduated components are wirable (caller treats
    this as "no demo to emit").
    """
    from .bringup_loop import _stub_import_path

    graduated = collect_graduated_components(demo_dir, components, model_id=model_id)
    selected = select_maximal_antichain(graduated)
    return [
        {
            "path": c.submodule_path,
            "import": _stub_import_path(demo_dir, c.safe, repo_root),
            "name": c.name,
            "safe": c.safe,
        }
        for c in selected
    ]


def format_wiring_literal(specs: List[Dict[str, str]]) -> str:
    """Format ``specs`` as a Python list-of-tuples literal suitable for
    embedding via ``.format()`` substitution into the demo template.

    Output layout (one tuple per line for readability)::

      [
          ("<submodule_path>", "<stub_import_path>", "<component_name>"),
          ("<submodule_path>", "<stub_import_path>", "<component_name>"),
      ]
    """
    if not specs:
        return "[]"
    lines = ["["]
    for s in specs:
        lines.append(f"    ({s['path']!r}, {s['import']!r}, {s['name']!r}),")
    lines.append("]")
    return "\n".join(lines)
