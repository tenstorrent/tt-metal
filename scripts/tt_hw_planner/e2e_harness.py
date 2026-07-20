"""End-to-end TT-substitution harness for brought-up models.

This is the runtime engine that powers the auto-generated e2e tests.
Given a HF model and a bring-up plan, it:

  1. Resolves each on-device component to its HF submodule path
     (via `capture_inputs._resolve_submodule` — already exists).
  2. Parses the per-component PCC test to extract the TT module's
     instantiation recipe (a working `TtClass(mesh_device=..., ...)`
     expression that the bring-up loop already produced).
  3. Builds each TT module by evaluating that recipe in a sandbox with
     the right context (mesh_device, state_dict, args, dtype).
  4. Registers a forward hook on each HF submodule that REPLACES the
     HF output with the TT-computed output.
  5. Runs the HF model forward with hooks installed AND once without,
     then compares the final top-level outputs via comp_pcc.

This avoids the per-component constructor-synthesis problem (the
existing PCC tests already contain the right invocations) AND the
dataflow-inference problem (HF's own forward graph IS the dataflow).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class TtSubstitution:
    """One TT-for-HF substitution to install."""

    component_name: str
    hf_submodule_path: str
    tt_construction_expr: str
    construction_imports: List[str]


@dataclass
class E2EResult:
    """Outcome of `run_e2e_pipeline`."""

    passed: bool
    pcc: float
    per_component_status: Dict[str, str]
    notes: List[str]


def _extract_tt_construction_from_pcc_test(
    test_path: Path,
) -> Optional[Tuple[str, List[str]]]:
    """AST-scan a per-component PCC test file and return the FIRST
    statement that constructs the TT module, plus the import lines
    needed for it.

    Returns (construction_expr, imports) or None if no construction
    is found. The construction is identified by:
      * a top-level Assign / AnnAssign whose target name starts with
        ``tt_`` (the bring-up loop's naming convention), OR
      * a top-level Assign whose value is a Call whose func is a Name
        that matches a class imported via ``from models.... import``.

    Imports are returned verbatim from the test file so the caller can
    re-emit them in the harness context.
    """
    if not test_path.exists():
        return None
    try:
        src = test_path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        return None

    imports: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))

    def _find_construction(body: List[ast.stmt]) -> Optional[str]:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = _find_construction(stmt.body)
                if inner is not None:
                    return inner
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.startswith("tt_"):
                        if stmt.value is not None:
                            return ast.unparse(stmt.value)
        return None

    expr = _find_construction(tree.body)
    if expr is None:
        return None
    return expr, imports


def _resolve_hf_submodule_for_component(
    hf_model: Any,
    component_name: str,
    demo_dir: Path,
) -> Optional[Tuple[Any, str]]:
    """Thin wrapper around capture_inputs._resolve_submodule so we
    don't duplicate the candidate-path heuristics."""
    try:
        from .capture_inputs import _resolve_submodule

        return _resolve_submodule(hf_model, component_name, demo_dir=demo_dir)
    except Exception:
        return None


def _collect_substitutions(
    hf_model: Any,
    plan: Any,
    demo_dir: Path,
) -> Tuple[List[TtSubstitution], List[str]]:
    """For each on-device component in the plan, gather the data
    needed to install a TT-substitution hook.

    Returns (list_of_substitutions, list_of_skip_notes). Notes are
    surfaced so the caller can include them in the test output —
    they explain which components were left as plain HF (because we
    couldn't resolve the submodule, or couldn't parse a TT
    construction from their PCC test)."""
    subs: List[TtSubstitution] = []
    notes: List[str] = []
    for c in getattr(plan, "components", []):
        status = getattr(c, "status", "NEW")
        # ADAPT removed 2026-05-31 — accept REUSE / NEW only. Legacy
        # bringup_status.json files with "ADAPT" status drop through
        # the continue (treat as unknown -> skip extraction).
        if status not in ("REUSE", "NEW"):
            continue
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in c.name)
        pcc_test = demo_dir / "tests" / "pcc" / f"test_{safe}.py"
        extracted = _extract_tt_construction_from_pcc_test(pcc_test)
        if extracted is None:
            notes.append(
                f"  skip {c.name}: no construction recipe found in {pcc_test} "
                f"(PCC test missing or doesn't follow the `tt_xxx = ...` pattern)"
            )
            continue
        construction_expr, construction_imports = extracted

        resolved = _resolve_hf_submodule_for_component(hf_model, c.name, demo_dir)
        if resolved is None:
            notes.append(
                f"  skip {c.name}: couldn't resolve HF submodule path "
                f"(checked _CANDIDATE_SUBMODULE_PATHS in the PCC test)"
            )
            continue
        _, hf_path = resolved
        subs.append(
            TtSubstitution(
                component_name=c.name,
                hf_submodule_path=hf_path,
                tt_construction_expr=construction_expr,
                construction_imports=construction_imports,
            )
        )
    return subs, notes


def _build_tt_module_in_sandbox(
    construction_expr: str,
    construction_imports: List[str],
    *,
    device: Any,
    sandbox_extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """Evaluate the TT-construction expression in a sandbox dict that
    includes the per-test imports plus ``device``.

    The sandbox lets us reuse the EXACT instantiation the per-component
    PCC test uses — including any helper-function calls the test
    defined — without trying to synthesize the call from scratch.
    Errors during construction are surfaced to the caller as None
    plus a note (no silent skip)."""
    sandbox: Dict[str, Any] = {"device": device}
    if sandbox_extra:
        sandbox.update(sandbox_extra)

    for imp in construction_imports:
        try:
            exec(imp, sandbox)
        except Exception:
            continue
    try:
        return eval(construction_expr, sandbox)
    except Exception as exc:
        raise RuntimeError(f"construction `{construction_expr}` failed: " f"{type(exc).__name__}: {exc}") from exc


def _make_tt_swap_hook(
    tt_module: Any,
    device: Any,
    to_tt: Callable[[Any, Any], Any],
    to_torch: Callable[[Any, Any], Any],
) -> Callable[..., Any]:
    """Build a `register_forward_hook(with_kwargs=True)` callable that
    runs the TT module on each captured (args, kwargs) and returns the
    TT-computed output as a torch tensor. HF's own output is ignored.

    HF passes positional and keyword tensors straight through; we
    convert each tensor to ttnn before the TT call and back to torch
    after, so HF's downstream code sees a normal torch tensor."""

    def hook(_module, args, kwargs, _hf_output):
        tt_args = tuple(to_tt(a, device) for a in args)
        tt_kwargs = {k: (to_tt(v, device) if hasattr(v, "shape") else v) for k, v in (kwargs or {}).items()}
        tt_out = tt_module.forward(*tt_args, **tt_kwargs)
        return to_torch(tt_out, device)

    return hook


def run_e2e_pipeline(
    *,
    hf_model: Any,
    plan: Any,
    demo_dir: Path,
    device: Any,
    sample_input: Any,
    pcc_target: float = 0.95,
    sample_kwargs: Optional[Dict[str, Any]] = None,
) -> E2EResult:
    """Run HF.forward() once with TT-substitution hooks installed and
    once without. PCC-compare the final outputs.

    Args:
        hf_model: an instance of the HF model. The caller owns it.
        plan: a ``bringup_plan.BringUpPlan``.
        demo_dir: ``models/demos/.../<model>`` directory. The
            ``tests/pcc/`` subdir is read for construction recipes.
        device: the TT device (mesh or single).
        sample_input: the top-level input torch tensor.
        pcc_target: PCC threshold for the final assert.
        sample_kwargs: optional kwargs passed to ``hf_model(sample, **kwargs)``.

    Returns an ``E2EResult`` summarising the outcome.
    """
    import torch
    from copy import deepcopy

    from .e2e_emitter import _resolve_tt_class_from_file

    sample_kwargs = sample_kwargs or {}
    notes: List[str] = []

    import ttnn

    def _to_tt(x, dev):
        if isinstance(x, ttnn.Tensor):
            return x
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            t = x.to(torch.bfloat16)
            try:
                if isinstance(dev, ttnn.MeshDevice):
                    return ttnn.from_torch(
                        t,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
                    )
            except Exception:
                pass
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        return x

    def _to_torch(t, dev):
        if not isinstance(t, ttnn.Tensor):
            return t
        try:
            from models.common.auto_compose import to_torch_auto_compose

            return to_torch_auto_compose(t)
        except Exception:
            return ttnn.to_torch(t)

    with torch.no_grad():
        hf_ref_output = hf_model(sample_input, **sample_kwargs)

    subs, skip_notes = _collect_substitutions(hf_model, plan, demo_dir)
    notes.extend(skip_notes)
    handles: List[Any] = []
    per_component: Dict[str, str] = {}
    for sub in subs:
        try:
            tt_mod = _build_tt_module_in_sandbox(
                sub.tt_construction_expr,
                sub.construction_imports,
                device=device,
            )
        except Exception as exc:
            per_component[sub.component_name] = f"construction failed: {exc}"
            continue

        resolved = _resolve_hf_submodule_for_component(hf_model, sub.component_name, demo_dir)
        if resolved is None:
            per_component[sub.component_name] = "submodule resolution failed at hook-install"
            continue
        submod, _path = resolved
        try:
            handle = submod.register_forward_hook(
                _make_tt_swap_hook(tt_mod, device, _to_tt, _to_torch),
                with_kwargs=True,
            )
            handles.append(handle)
            per_component[sub.component_name] = "substituted"
        except TypeError:
            per_component[sub.component_name] = "register_forward_hook lacks with_kwargs (old torch)"

    try:
        with torch.no_grad():
            hf_with_tt_output = hf_model(sample_input, **sample_kwargs)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    def _final_tensor(o):
        if isinstance(o, torch.Tensor):
            return o
        if hasattr(o, "last_hidden_state"):
            return o.last_hidden_state
        if isinstance(o, (tuple, list)) and len(o) > 0 and isinstance(o[0], torch.Tensor):
            return o[0]
        if isinstance(o, dict):
            for v in o.values():
                if isinstance(v, torch.Tensor):
                    return v
        return o

    a = _final_tensor(hf_ref_output)
    b = _final_tensor(hf_with_tt_output)
    try:
        from models.common.utility_functions import comp_pcc

        passed, pcc_value = comp_pcc(a, b, pcc=pcc_target)
        pcc_float = float(pcc_value) if isinstance(pcc_value, (int, float)) else 0.0
    except Exception as exc:
        passed, pcc_float = False, 0.0
        notes.append(f"comp_pcc failed: {type(exc).__name__}: {exc}")

    return E2EResult(
        passed=bool(passed),
        pcc=pcc_float,
        per_component_status=per_component,
        notes=notes,
    )


__all__ = [
    "TtSubstitution",
    "E2EResult",
    "run_e2e_pipeline",
]
