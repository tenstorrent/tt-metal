from __future__ import annotations
from ..discovery import safe_relative_to_root

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def cmd_op_synth(args) -> int:
    """Per-component op-level classifier + deterministic emitter.

    Walks the HF reference submodule for each NEW (or selected) component,
    classifies every leaf op as op-REUSE / op-ADAPT / op-NEW against the
    known ttnn primitive set, prints the op-plan, and (with --emit-stub)
    writes a partial native TTNN stub to `<demo>/_synth_responses/<safe>.py`
    where weight loading and deterministic op helpers are pre-bound. The
    LLM's only remaining task is to rewrite `__call__` using those helpers.
    """
    from ..cli import (
        REPO_ROOT,
        _load_bringup_status,
        _resolve_torch_submodule_for_component,
        _safe_id,
        _select_op_synth_targets,
        classify_ops_in_component,
        emit_partial_stub,
        format_op_plan,
        json,
        summarize_ops,
    )

    try:
        demo_dir, status = _load_bringup_status(args.model_id)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    targets = _select_op_synth_targets(
        status,
        component=args.component,
        include_adapt=args.include_adapt,
    )
    if not targets:
        if args.component:
            print(f"ERROR: component {args.component!r} not found in bringup_status.json", file=sys.stderr)
            return 2
        print("No NEW components to op-synth (model already fully ported).")
        return 0

    out_dir = demo_dir / "_synth_responses"
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_counts = {"op-REUSE": 0, "op-ADAPT": 0, "op-NEW": 0, "total": 0}
    written: List[Tuple[str, Path]] = []
    no_resolve: List[str] = []

    print(f"\nOp-level bring-up plan for {args.model_id}")
    print(f"  demo dir: {safe_relative_to_root(demo_dir)}")
    print(f"  targets:  {len(targets)} component(s)")
    print()

    for comp in targets:
        name = comp.get("name", "?")
        sub = _resolve_torch_submodule_for_component(args.model_id, demo_dir, name)
        if sub is None:
            print(f"  {name}: could not resolve HF submodule — skipping op-synth")
            no_resolve.append(name)
            continue

        ops = classify_ops_in_component(sub)
        print(format_op_plan(name, ops))
        summary = summarize_ops(ops)
        counts = summary["counts"]
        overall_counts["total"] += summary["total"]
        for k in ("op-REUSE", "op-ADAPT", "op-NEW"):
            overall_counts[k] += counts.get(k, 0)

        if args.emit_stub:
            safe = _safe_id(name)
            candidate_paths: List[str] = []
            test_path = demo_dir / "tests" / "pcc" / f"test_{safe}.py"
            if test_path.is_file():
                try:
                    import ast as _ast

                    tree = _ast.parse(test_path.read_text(errors="ignore"))
                    for node in _ast.walk(tree):
                        if isinstance(node, _ast.Assign):
                            for tgt in node.targets:
                                if isinstance(tgt, _ast.Name) and tgt.id == "_CANDIDATE_SUBMODULE_PATHS":
                                    try:
                                        val = _ast.literal_eval(node.value)
                                        if isinstance(val, (list, tuple)):
                                            candidate_paths = [str(x) for x in val]
                                    except Exception:
                                        pass
                except Exception:
                    candidate_paths = []
            source, manifest = emit_partial_stub(
                component_name=name,
                model_id=args.model_id,
                hf_reference=comp.get("hf_reference", "") or "",
                submodule_candidates=candidate_paths or ["<UNKNOWN>"],
                ops=ops,
            )
            out_path = out_dir / f"{safe}.py"
            out_path.write_text(source)
            (out_dir / f"{safe}.opplan.json").write_text(json.dumps(manifest, indent=2))
            written.append((name, out_path))

    print("=" * 72)
    total = overall_counts["total"]
    reusable = overall_counts["op-REUSE"] + overall_counts["op-ADAPT"]
    pct = (reusable / total * 100) if total else 0.0
    print(
        f"Overall: {total} leaf ops across {len(targets)} component(s) — "
        f"op-REUSE={overall_counts['op-REUSE']}, "
        f"op-ADAPT={overall_counts['op-ADAPT']}, "
        f"op-NEW={overall_counts['op-NEW']}  ({pct:.0f}% deterministic)"
    )
    if no_resolve:
        print(f"  (skipped {len(no_resolve)}: could not resolve HF submodule " f"-> {', '.join(no_resolve)})")
    if args.emit_stub:
        print(f"\nWrote {len(written)} partial-stub file(s):")
        for name, path in written:
            print(f"  {name:30s}  {safe_relative_to_root(path)}")
        if written:
            print(
                "\nThese stubs have all op-REUSE/op-ADAPT weights pre-loaded "
                "and helpers pre-bound. `__call__` falls back to HF torch so "
                "the smoke test still passes. To install: copy / move from "
                "`_synth_responses/` into `_stubs/` (or let `bringup "
                "--apply-all-responses` do it). The LLM only needs to "
                "rewrite `__call__` to use the `_apply_*` helpers."
            )
    else:
        print("\n(Use --emit-stub to write the partial TTNN ports to " "`_synth_responses/`.)")
    return 0


from ..commands.emit_e2e import cmd_emit_e2e  # noqa: F401


from ..commands.auto_onboard import cmd_auto_onboard  # noqa: F401
