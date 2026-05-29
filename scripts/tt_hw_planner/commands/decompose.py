"""`tt_hw_planner decompose <model> <component>` — surface the
non-trivial children of a stuck large component so they can be brought
up independently.

This complements `failure_classifier.classify_failure` + `_skip_component
_to_fallback`: when the auto-loop's verdict is AGENT_STUCK,
KERNEL_VERIFIED_MISSING, CONSTRAINT_MISMATCH or ITERATION_BUDGET, the
loop emits a one-line CTA pointing here. Running this command loads the
HF reference, resolves the component's submodule, and prints the list of
non-trivial children with their submodule_paths, leaf counts and class
names.

With ``--write-plan``, the children are persisted to
``<demo_dir>/decomposition_plan.json`` as a PLANNING ARTIFACT. There is
no automatic consumer in ``up`` today — a human must read the JSON and
either (a) manually update ``bringup_status.json`` to add the children
as new components, or (b) feed the list to a follow-up scaffold step.
The plan file is structured so the auto-consumer is a future-feature
drop-in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..bringup_loop import find_demo_dir
from ..component_decomposer import decompose_component


def _load_status(demo_dir: Path) -> Dict[str, Any]:
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"no bringup_status.json at {status_path}")
    return json.loads(status_path.read_text())


def _resolve_torch(model, dotted: str):
    cur = model
    for tok in dotted.replace("[", ".").replace("]", "").split("."):
        if not tok:
            continue
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            cur = getattr(cur, tok)
    return cur


def _find_component_path(status: Dict[str, Any], comp_name: str) -> str:
    for c in status.get("components", []) or []:
        if c.get("name") == comp_name:
            return (c.get("submodule_path") or "").strip()
    raise KeyError(f"component `{comp_name}` not found in bringup_status.json")


def cmd_decompose(args) -> int:
    model_id = args.model_id
    comp_name = args.component
    min_leaf_count = int(getattr(args, "min_leaf_count", 2) or 2)
    write_plan = bool(getattr(args, "write_plan", False))

    demo_dir = find_demo_dir(model_id)
    if demo_dir is None:
        print(f"error: no scaffolded demo dir found for `{model_id}`. " f"Run `up {model_id}` first.", file=sys.stderr)
        return 2

    try:
        status = _load_status(demo_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        submodule_path = _find_component_path(status, comp_name)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not submodule_path:
        # Fall back to capture-time manifest (might differ from
        # status when the bringup loop discovered the path later).
        from ..bringup_loop import _safe_id

        safe = _safe_id(comp_name)
        manifest_path = demo_dir / "_captured" / safe / "manifest.json"
        if manifest_path.is_file():
            try:
                submodule_path = (json.loads(manifest_path.read_text()).get("submodule_path") or "").strip()
            except Exception:
                pass
    if not submodule_path:
        print(
            f"error: no submodule_path recorded for `{comp_name}` — can't "
            f"locate it on the HF model. Run capture-inputs first.",
            file=sys.stderr,
        )
        return 2

    print(f"decompose: loading HF model `{model_id}` (may take a moment)…")
    try:
        import transformers
    except ImportError:
        print("error: `transformers` not installed", file=sys.stderr)
        return 2

    try:
        hf_model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"error: HF load failed: {e}", file=sys.stderr)
        return 2

    try:
        sub = _resolve_torch(hf_model, submodule_path)
    except Exception as e:
        print(
            f"error: could not resolve `{submodule_path}` on the HF model: {e}",
            file=sys.stderr,
        )
        return 2

    children = decompose_component(
        parent_path=submodule_path,
        parent_module=sub,
        min_leaf_count=min_leaf_count,
    )
    if not children:
        print(f"decompose: `{comp_name}` has NO non-trivial children at " f"min_leaf_count={min_leaf_count}.")
        print(f"  This component is primitive — decomposition exhausted.")
        print(f"  If the auto-loop pushed it to CPU with verdict ")
        print(f"  KERNEL_VERIFIED_MISSING, the verdict is correct: TTNN " f"genuinely lacks the needed op.")
        return 1

    print(f"decompose: `{comp_name}` → {len(children)} non-trivial children:")
    print(f"  {'name':<24} {'class':<24} {'leaves':>6}  submodule_path")
    print(f"  {'-'*24} {'-'*24} {'-'*6}  {'-'*40}")
    for c in children:
        print(f"  {c.name:<24} {c.class_name:<24} {c.leaf_count:>6}  {c.submodule_path}")

    if write_plan:
        plan_path = demo_dir / "decomposition_plan.json"
        existing: List[Dict[str, Any]] = []
        if plan_path.is_file():
            try:
                existing = json.loads(plan_path.read_text()) or []
            except Exception:
                existing = []
        # Replace any prior plan entry for this same parent path.
        existing = [e for e in existing if e.get("parent_path") != submodule_path]
        existing.append(
            {
                "parent_name": comp_name,
                "parent_path": submodule_path,
                "children": [
                    {
                        "name": c.name,
                        "submodule_path": c.submodule_path,
                        "class_name": c.class_name,
                        "leaf_count": c.leaf_count,
                    }
                    for c in children
                ],
            }
        )
        plan_path.write_text(json.dumps(existing, indent=2))
        print(f"\ndecompose: wrote plan to {plan_path}")
        print(
            f"  NOTE: no auto-consumer in `up` today. To onboard these "
            f"children, manually update bringup_status.json or feed the "
            f"plan JSON to a follow-up scaffold step."
        )
    return 0
