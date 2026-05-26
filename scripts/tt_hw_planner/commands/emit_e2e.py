from __future__ import annotations

from pathlib import Path
from typing import Optional


def cmd_emit_e2e(args) -> int:
    from ..e2e_emitter import (
        emit_e2e_pipeline_test,
        emit_e2e_pipeline_test_harness,
        emit_e2e_pipeline_test_wired,
    )
    from ..bringup_plan import build_bringup_plan
    from ..family_backends import pick_backend
    from ..probe import probe_model

    sep = "=" * 78
    print(sep)
    print(f"  EMIT-E2E  emitting chained-pipeline test for {args.model_id}")
    print(sep)

    probe = probe_model(args.model_id)
    cfg = probe.raw_config if probe is not None else None
    if cfg is None:
        print(f"  probe failed for {args.model_id}; cannot resolve backend")
        return 2
    category = getattr(probe, "category", None) or "Other"
    model_type = (cfg or {}).get("model_type")
    pipeline_tag = getattr(probe, "pipeline_tag", None)
    backend = pick_backend(category=category, model_type=model_type, pipeline_tag=pipeline_tag)
    if backend is None:
        print(
            f"  no FamilyBackend resolved for {args.model_id}. "
            f"Run `auto-onboard {args.model_id}` first, or use --output "
            f"to write the skeleton to an explicit path."
        )
        return 2

    plan = build_bringup_plan(
        new_model_id=args.model_id,
        new_cfg=cfg,
        backend=backend,
        repo_root=Path.cwd(),
    )

    def _resolve_demo_dir(raw: Optional[str]) -> Path:
        if not raw:
            return Path(f"models/demos/_emitted/{args.model_id.split('/')[-1].replace('-', '_').lower()}")
        p = Path(raw)
        return p.parent if p.suffix == ".py" else p

    if args.output:
        output_path = Path(args.output)
    else:
        slug = args.model_id.split("/")[-1].replace("-", "_").lower()
        demo_dir = _resolve_demo_dir(backend.demo_path)

        canonical_dir = Path(f"models/demos/{demo_dir.parent.name}/{slug}/tests")
        if canonical_dir.parent.exists():
            output_path = canonical_dir / "test_e2e.py"
        else:
            output_path = demo_dir / "tests" / f"test_e2e_{slug}.py"

    captured_dir = (_resolve_demo_dir(backend.demo_path) / "_captured") if backend.demo_path else None
    written: Optional[Path] = None
    used_wired = False
    if captured_dir and captured_dir.exists():
        written = emit_e2e_pipeline_test_wired(
            model_id=args.model_id,
            components=plan.components,
            captured_dir=captured_dir,
            repo_root=Path.cwd(),
            output_path=output_path,
            pcc_target=args.pcc_target,
            overwrite=args.overwrite,
        )
        used_wired = written is not None
    used_harness = False
    if written is None:
        written = emit_e2e_pipeline_test_harness(
            model_id=args.model_id,
            components=plan.components,
            output_path=output_path,
            pcc_target=args.pcc_target,
            overwrite=args.overwrite,
        )
        used_harness = written is not None
    if written is None:
        written = emit_e2e_pipeline_test(
            model_id=args.model_id,
            components=plan.components,
            output_path=output_path,
            pcc_target=args.pcc_target,
            overwrite=args.overwrite,
        )
    if written is None:
        print(f"  test_e2e file already exists at {output_path}; use " f"--overwrite to replace it.")
        return 1
    variant = "wired" if used_wired else ("harness" if used_harness else "skeleton")
    print(f"  emitted ({variant}): {written}")
    print(
        f"  components in plan: {len(plan.components)} "
        f"(on-device: {sum(1 for c in plan.components if c.status in ('REUSE','ADAPT','NEW'))})"
    )
    if used_wired:
        print(
            f"  wired variant: every block validates a TT component against "
            f"its captured HF I/O. Run the test directly — failing components "
            f"are named in the pytest output."
        )
    elif used_harness:
        print(
            f"  harness variant: pytest runs HF.forward() twice (once pure, "
            f"once with TT-substitution hooks installed at each component's "
            f"HF submodule). No per-component TODO wiring required — the "
            f"harness extracts each TT-construction recipe from the existing "
            f"per-component PCC tests at runtime."
        )
        if captured_dir is not None and not captured_dir.exists():
            print(
                f"  TIP: running `capture-inputs {args.model_id}` first will "
                f"unlock the WIRED variant on the next emit, which adds "
                f"per-component PCC checks alongside the end-to-end one."
            )
    else:
        print(f"  next: open the file, fill in each TODO[e2e] marker, then run pytest.")
    return 0
