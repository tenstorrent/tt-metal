from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from .discovery import BRINGUP_ROOT, safe_relative_to_root
from .bringup import (
    MODEL_CONFIG_PATH,
    REPO_ROOT,
    TRACE_REGION_PATH,
    derive_base_model_name,
)
from .bringup_plan import (
    NEW,
    BringUpPlan,
    build_bringup_plan,
    collect_bringup_plan_files,
)
from .compatibility import Status, check_compatibility
from .family_backends import pick_backend
from .probe import probe_model
from .scaffold_demo_folder import collect_demo_folder_changes


def _model_params_dir() -> Path:
    return BRINGUP_ROOT() / "models" / "tt_transformers" / "model_params"


def _find_sibling_params_dir(sibling_tail: str, sibling_base: str) -> Optional[Path]:
    base = _model_params_dir()
    if not base.is_dir():
        return None
    candidate = base / sibling_tail
    if candidate.is_dir():
        return candidate
    matches = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(sibling_base)]
    return matches[0] if len(matches) == 1 else None


class ScaffoldError(RuntimeError):
    pass


class ColdStartScaffoldError(ScaffoldError):
    """2026-05-23 cold-start signal: scaffold cannot produce a per-model
    `tt/` folder (no closest sibling, or the architecture doesn't fit the
    standard ``tt_transformers`` template), but the caller CAN still attempt
    to run the model "from scratch" via a generic, architecture-portable
    demo such as ``models/tt_transformers/demo/simple_text_demo.py``.

    Catchers in ``cli.cmd_up`` are expected to:
      1. Print a clear warning that this is a Hail-Mary cold-start
         (no per-model TTNN tuning).
      2. Re-dispatch to ``prepare --execute`` so the user gets one
         automated attempt to run the model on tt-metal.
      3. If ``prepare`` itself fails, surface a coherent message
         pointing to ``compatibility.py`` + ``family_backends.py``
         as the spots where someone could add proper support.

    Subclass of ``ScaffoldError`` so existing call sites that only
    catch ``ScaffoldError`` keep working (the message includes a
    "(cold-start: try `...`)" hint)."""

    def __init__(self, model_id: str, reason: str, suggested_cmd: Optional[str] = None):
        self.model_id = model_id
        self.reason = reason
        self.suggested_cmd = suggested_cmd
        suffix = f" (cold-start: try `{suggested_cmd}`)" if suggested_cmd else " (cold-start path available)"
        super().__init__(reason + suffix)


@dataclass
class ScaffoldChange:
    kind: str
    path: str
    diff: Optional[str] = None
    new_content: Optional[bytes] = None
    source: Optional[str] = None
    added_lines: int = 0
    preserve_if_exists: bool = False


@dataclass
class ScaffoldPlan:
    new_model_id: str
    new_base_name: str
    new_tail: str
    sibling_model_id: str
    sibling_base_name: str
    sibling_tail: str
    compat_overall: str
    compat_summary: str
    changes: List[ScaffoldChange] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    new_demo_dir: Optional[str] = None
    bringup_plan: Optional[BringUpPlan] = None


def _insert_after_sibling_in_table(src: str, sibling_key: str, new_key: str) -> Optional[str]:
    lines = src.splitlines(keepends=True)
    pattern = re.compile(rf'^(\s*)"{re.escape(sibling_key)}"\s*:\s*(\{{.*)$')
    for i, line in enumerate(lines):
        m = pattern.match(line.rstrip("\n"))
        if m:
            indent, rest = m.group(1), m.group(2)
            new_line = f'{indent}"{new_key}": {rest}\n'
            lines.insert(i + 1, new_line)
            return "".join(lines)
    return None


def _table_key_present(src: str, key: str) -> bool:
    return bool(re.search(rf'^\s*"{re.escape(key)}"\s*:\s*\{{', src, re.MULTILINE))


def _build_table_insert(file_path: Path, sibling_key: str, new_key: str) -> Optional[ScaffoldChange]:
    src = file_path.read_text()
    if _table_key_present(src, new_key):
        return None
    new_src = _insert_after_sibling_in_table(src, sibling_key, new_key)
    if new_src is None:
        return None
    diff_iter = difflib.unified_diff(
        src.splitlines(keepends=True),
        new_src.splitlines(keepends=True),
        fromfile=f"a/{safe_relative_to_root(file_path)}",
        tofile=f"b/{safe_relative_to_root(file_path)}",
        n=3,
    )
    diff_text = "".join(diff_iter)
    added = sum(1 for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    return ScaffoldChange(
        kind="edit",
        path=str(safe_relative_to_root(file_path)),
        diff=diff_text,
        new_content=new_src.encode(),
        added_lines=added,
    )


def plan_scaffold(new_model_id: str, *, force_already_supported: bool = False) -> ScaffoldPlan:
    """Plan a scaffold for ``new_model_id``.

    ``force_already_supported`` lets the escalation hook
    (``_maybe_escalate_pcc_fail`` in cli.py) bypass the
    "ALREADY SUPPORTED — no scaffolding needed" gate. The gate exists
    to stop users scaffolding twice; the escalation explicitly wants
    re-scaffolding because the ALREADY-SUPPORTED routing produced
    output the PCC gate rejected, and we need Path 1 (scaffold +
    per-component iterate) to take over.
    """
    probe = probe_model(new_model_id)
    if not probe.raw_config:
        raise ScaffoldError(f"could not load config.json for {new_model_id} — set HF_TOKEN for gated repos")

    if probe.category not in {"LLM", "VLM"}:
        return _plan_demo_folder_scaffold(new_model_id=new_model_id, probe=probe)

    compat = check_compatibility(new_model_id, probe.raw_config)

    if compat.in_external_demo and compat.primary_demo is not None:
        raise ScaffoldError(
            f"{new_model_id} is supported via an external demo at "
            f"`{compat.primary_demo.as_posix()}`, not via `tt_transformers/`. "
            "Scaffold only adds rows to `tt_transformers`'s tuning tables, so it "
            "does not apply here. "
            f"Run: python -m scripts.tt_hw_planner prepare {new_model_id} --box <BOX> --execute"
        )
    if compat.overall.startswith("ALREADY SUPPORTED") and not force_already_supported:
        raise ScaffoldError(
            f"{new_model_id} is already supported — no scaffolding needed. "
            f"Run: python -m scripts.tt_hw_planner prepare {new_model_id}"
        )
    missing = [r for r in compat.results if r.needed and r.status == Status.MISSING]
    if compat.overall == "BLOCKED" or missing:
        blockers = [r.block.name for r in missing] or ["(see compat output)"]
        raise ScaffoldError(
            f"{new_model_id} is BLOCKED — missing TT building block(s): {blockers}. "
            "Scaffolding can't help; new TTNN kernel work is required first."
        )

    _allowed_verdicts = ("READY", "FEASIBLE WITH WORK")
    if compat.overall not in _allowed_verdicts:
        # When the escalation hook forces re-scaffolding of an already-
        # supported model (because Path 2 PCC-gate rejected its output),
        # the verdict is "ALREADY SUPPORTED" and we must accept it.
        if not (force_already_supported and compat.overall.startswith("ALREADY SUPPORTED")):
            raise ScaffoldError(f"unexpected compat verdict {compat.overall!r}; refusing to scaffold")

    # WIRING #13 (2026-05-31): escalation fast-path. When
    # force_already_supported=True and the model is genuinely
    # ALREADY SUPPORTED, the sibling-copy logic below is a no-op
    # (the demo files already exist at the backend's demo_path).
    # All we need is a bringup_status.json that lists every
    # component as ADAPT (since the global PCC gate failed, every
    # REUSE label is suspect). The existing per-component iterate
    # loop in auto_iterate.py reads bringup_status.json and
    # PCC-tests each non-REUSE component — that's how Path 1 gets
    # something to iterate on for an already-supported-but-broken
    # model. Pairs with bringup_plan.build_bringup_plan's
    # ``force_adapt_all`` kwarg.
    if force_already_supported and compat.overall.startswith("ALREADY SUPPORTED"):
        # NOTE: the routing-mode block further down does
        # ``from .family_backends import pick_backend`` which makes
        # Python treat the name as function-local for the whole
        # ``plan_scaffold`` body — so reaching the module-level
        # ``pick_backend`` from here would UnboundLocalError. Import
        # under an alias to sidestep the shadowing.
        from .family_backends import pick_backend as _pick_backend_esc

        _be_esc = _pick_backend_esc(
            category=probe.category,
            model_type=(probe.raw_config or {}).get("model_type"),
            pipeline_tag=getattr(probe, "pipeline_tag", None),
        )
        if _be_esc is None:
            raise ScaffoldError(
                f"force_already_supported=True but no backend mapped for "
                f"{new_model_id}; cannot enumerate components for ADAPT demotion."
            )
        bplan_esc = build_bringup_plan(
            new_model_id=new_model_id,
            new_cfg=probe.raw_config or {},
            backend=_be_esc,
            repo_root=BRINGUP_ROOT(),
            force_adapt_all=True,
        )
        # Derive a SIBLING demo directory (not the backend's demo
        # file itself). Without the slug, this used to be
        # `models/tt_transformers/demo/simple_text_demo.py` — a
        # regular file — which then failed `mkdir` with
        # `[Errno 17] File exists` when scaffold tried to put
        # BRING_UP_PLAN.md inside it. Mirrors the non-escalation
        # path at line ~460 below.
        from .scaffold_demo_folder import _slug as _scaffold_slug

        # 2026-06-04 Fix 6 (escalation path): prefer existing demo dir
        # for this model_id over a freshly-computed default — see the
        # matching Fix 6 comment further down in this function.
        from .bringup_loop import find_demo_dir as _find_demo_dir

        _be_esc_parent = Path(_be_esc.demo_path).parent
        _esc_bringup_root = BRINGUP_ROOT()
        _esc_existing_dir = _find_demo_dir(new_model_id, repo_root=_esc_bringup_root)
        if _esc_existing_dir is not None:
            try:
                demo_dir_esc_rel = _esc_existing_dir.relative_to(_esc_bringup_root)
            except Exception:
                demo_dir_esc_rel = _esc_existing_dir
        else:
            demo_dir_esc_rel = _be_esc_parent / _scaffold_slug(new_model_id.split("/")[-1])
        changes_esc: List[ScaffoldChange] = []
        for target_rel, content, label in collect_bringup_plan_files(
            plan=bplan_esc,
            new_demo_dir_rel=demo_dir_esc_rel,
        ):
            # bringup_status.json/BRING_UP_PLAN.md overwrite; _stubs/
            # only created when missing.
            preserve = "NEW-stub" in label
            changes_esc.append(
                ScaffoldChange(
                    kind="create",
                    path=str(target_rel),
                    new_content=content,
                    source=None,
                    added_lines=content.count(b"\n"),
                    preserve_if_exists=preserve,
                )
            )
        c = bplan_esc.counts
        return ScaffoldPlan(
            new_model_id=new_model_id,
            new_base_name=derive_base_model_name(new_model_id),
            new_tail=new_model_id.split("/")[-1],
            sibling_model_id=_be_esc.canonical_hf_id or _be_esc.name,
            sibling_base_name=Path(_be_esc.demo_path).name,
            sibling_tail=(
                _be_esc.canonical_hf_id.split("/")[-1] if _be_esc.canonical_hf_id else Path(_be_esc.demo_path).name
            ),
            compat_overall=compat.overall,
            compat_summary=(
                f"force_already_supported: PCC-gate failure promoted REUSE "
                f"-> ADAPT (canonical wrapper exists) or NEW (no wrapper) "
                f"for per-component iteration. "
                f"Plan: {c.get('REUSE', 0)} REUSE / "
                f"{c.get('ADAPT', 0)} ADAPT / "
                f"{c.get('NEW', 0)} NEW."
            ),
            changes=changes_esc,
            skipped=[],
            warnings=[],
            new_demo_dir=str(demo_dir_esc_rel),
            bringup_plan=bplan_esc,
        )

    try:
        from .family_backends import pick_backend

        _be = pick_backend(
            category=probe.category,
            model_type=(probe.raw_config or {}).get("model_type"),
            pipeline_tag=getattr(probe, "pipeline_tag", None),
        )
        if _be is not None and getattr(_be, "routing_mode", "") == "generic":
            raise ColdStartScaffoldError(
                new_model_id,
                f"{new_model_id} uses a GENERIC LLM/VLM backend "
                f"(`{_be.name}` / `{_be.demo_path}`). Scaffolding "
                f"copies a per-model `tt/` folder from a sibling; "
                f"generic backends don't have one (the demo is "
                f"architecture-portable and reads HF_MODEL from the "
                f"env)",
                suggested_cmd=(f"python -m scripts.tt_hw_planner prepare " f"{new_model_id} --execute"),
            )
    except ScaffoldError:
        raise
    except Exception:
        pass

    sibling_id = compat.similar_supported_model
    if not sibling_id:
        family = compat.architecture_family

        raise ColdStartScaffoldError(
            new_model_id,
            f"no closest already-ported sibling found for architecture "
            f"family '{family}'. This model is the first of its kind in "
            f"tt-metal; there's nothing to copy from",
            suggested_cmd=(f"python -m scripts.tt_hw_planner prepare {new_model_id} --execute"),
        )

    new_base = derive_base_model_name(new_model_id)
    sibling_base = derive_base_model_name(sibling_id)
    new_tail = new_model_id.split("/")[-1]
    sibling_tail = sibling_id.split("/")[-1]

    if new_base == sibling_base:
        raise ScaffoldError(
            f"new model derives the same base_model_name '{new_base}' as the "
            f"sibling — they would collide in the tuning tables."
        )

    changes: List[ScaffoldChange] = []
    skipped: List[str] = []
    warnings: List[str] = []

    edit1 = _build_table_insert(MODEL_CONFIG_PATH, sibling_base, new_base)
    if edit1:
        changes.append(edit1)
    elif _table_key_present(MODEL_CONFIG_PATH.read_text(), new_base):
        skipped.append(f"MAX_PREFILL_CHUNK_SIZES_DIV1024 already contains '{new_base}'")
    else:
        skipped.append(
            f"sibling '{sibling_base}' has no entry in MAX_PREFILL_CHUNK_SIZES_DIV1024 — "
            "demo will fall back to MAX_PREFILL_CHUNK_SIZE=4 (×1024)"
        )

    edit2 = _build_table_insert(TRACE_REGION_PATH, sibling_base, new_base)
    if edit2:
        changes.append(edit2)
    elif _table_key_present(TRACE_REGION_PATH.read_text(), new_base):
        skipped.append(f"trace_region_size_dict already contains '{new_base}'")
    else:
        skipped.append(
            f"sibling '{sibling_base}' has no entry in trace_region_size_dict — "
            "demo will use the parametrize default"
        )

    sibling_params_dir = _find_sibling_params_dir(sibling_tail, sibling_base)
    new_params_dir = _model_params_dir() / new_tail
    if sibling_params_dir is not None:
        if new_params_dir.exists():
            skipped.append(f"model_params/{new_tail}/ already exists — leaving untouched")
        else:
            json_files = sorted(p for p in sibling_params_dir.iterdir() if p.is_file() and p.suffix == ".json")
            if not json_files:
                skipped.append(f"sibling dir {safe_relative_to_root(sibling_params_dir)} contains no JSON files")
            for src_file in json_files:
                content = src_file.read_bytes()
                rel_target = safe_relative_to_root(new_params_dir / src_file.name)
                changes.append(
                    ScaffoldChange(
                        kind="create",
                        path=str(rel_target),
                        new_content=content,
                        source=str(safe_relative_to_root(src_file)),
                        added_lines=content.count(b"\n"),
                    )
                )
    else:
        skipped.append(f"no model_params/ dir found for sibling '{sibling_tail}' — using built-in defaults")

    partial_blocks = [r for r in compat.results if r.needed and r.status == Status.PARTIAL]
    for r in partial_blocks:
        warnings.append(f"{r.block.name} is PARTIAL — {r.notes or r.block.notes or 'see compatibility.py'}")

    if probe.total_params and edit1:
        sibling_probe = None
        try:
            sibling_probe = probe_model(sibling_id)
        except Exception:
            sibling_probe = None
        if sibling_probe and sibling_probe.total_params:
            ratio = probe.total_params / sibling_probe.total_params
            if ratio < 0.5 or ratio > 2.0:
                warnings.append(
                    f"sibling is {sibling_probe.total_params / 1e9:.1f}B params, "
                    f"new model is {probe.total_params / 1e9:.1f}B — the copied "
                    "MAX_PREFILL_CHUNK_SIZES_DIV1024 row was tuned for a different "
                    "size; verify it fits your KV budget."
                )

    if not changes:
        raise ScaffoldError(
            "nothing to scaffold — sibling had no entries to copy, and no new "
            "model_params files to create. This typically means the sibling lives "
            "outside `tt_transformers/` (e.g. a vision/audio demo). Run "
            "`tt_hw_planner prepare <model>` to see the routed family backend "
            "(closest demo) you can adapt manually."
        )

    return ScaffoldPlan(
        new_model_id=new_model_id,
        new_base_name=new_base,
        new_tail=new_tail,
        sibling_model_id=sibling_id,
        sibling_base_name=sibling_base,
        sibling_tail=sibling_tail,
        compat_overall=compat.overall,
        compat_summary=compat.effort_summary,
        changes=changes,
        skipped=skipped,
        warnings=warnings,
    )


def _plan_demo_folder_scaffold(*, new_model_id: str, probe: Any) -> ScaffoldPlan:
    model_type = (probe.raw_config or {}).get("model_type")
    pipeline_tag = getattr(probe, "pipeline_tag", None)
    backend = pick_backend(category=probe.category, model_type=model_type, pipeline_tag=pipeline_tag)
    if backend is None:
        raise ColdStartScaffoldError(
            new_model_id,
            f"no tt-metal family backend registered for " f"category={probe.category!r}",
            suggested_cmd=(f"python -m scripts.tt_hw_planner auto-onboard " f"{new_model_id} --apply"),
        )

    creates, skipped, warnings = collect_demo_folder_changes(
        backend=backend,
        new_model_id=new_model_id,
        repo_root=BRINGUP_ROOT(),
    )

    if not creates and not skipped:
        raise ScaffoldError(
            f"backend `{backend.name}` produced no files to scaffold — the source "
            f"demo folder `{backend.demo_path}` may be missing or empty."
        )

    changes: List[ScaffoldChange] = []
    for target_rel, new_content, source_rel in creates:
        changes.append(
            ScaffoldChange(
                kind="create",
                path=str(target_rel),
                new_content=new_content,
                source=str(source_rel),
                added_lines=new_content.count(b"\n"),
            )
        )

    sibling_id = backend.canonical_hf_id or backend.name
    new_tail = new_model_id.split("/")[-1]
    sibling_tail = backend.canonical_hf_id.split("/")[-1] if backend.canonical_hf_id else Path(backend.demo_path).name

    backend_parent = Path(backend.demo_path).parent
    from .scaffold_demo_folder import _slug as _scaffold_slug

    # 2026-06-04 Fix 6: prefer the existing demo dir for this model_id
    # (if any) over a freshly-computed default. Without this, a backend
    # swap mid-run (e.g., torch-less subprocess picking the generic
    # hf_eager backend instead of the architecture-specific one) creates
    # a DIFFERENT demo dir, leaving two competing `bringup_status.json`
    # files for the same model — exactly the seamless-m4t scenario where
    # `hf_eager/.../bringup_status.json` was corrupted while the orchestrator
    # kept reading from the original `hf_seamless_m4t_medium/...`.
    # One model → one demo dir → one manifest.
    from .bringup_loop import find_demo_dir as _find_demo_dir

    _bringup_root = BRINGUP_ROOT()
    _existing_dir = _find_demo_dir(new_model_id, repo_root=_bringup_root)
    if _existing_dir is not None:
        try:
            new_demo_dir = str(_existing_dir.relative_to(_bringup_root))
        except Exception:
            new_demo_dir = str(_existing_dir)
    else:
        new_demo_dir = str(backend_parent / _scaffold_slug(new_tail))

    bplan: Optional[BringUpPlan] = None
    try:
        bplan = build_bringup_plan(
            new_model_id=new_model_id,
            new_cfg=probe.raw_config or {},
            backend=backend,
            repo_root=BRINGUP_ROOT(),
        )
        for target_rel, content, label in collect_bringup_plan_files(
            plan=bplan,
            new_demo_dir_rel=Path(new_demo_dir),
        ):
            preserve = "NEW-stub" in label
            changes.append(
                ScaffoldChange(
                    kind="create",
                    path=str(target_rel),
                    new_content=content,
                    source=None,
                    added_lines=content.count(b"\n"),
                    preserve_if_exists=preserve,
                )
            )
    except Exception as exc:
        warnings.append(f"bring-up plan generation failed: {exc}")

    summary_extra = ""
    if bplan is not None:
        c = bplan.counts
        summary_extra = (
            f" Component plan: {c.get('REUSE', 0)} REUSE / " f"{c.get('NEW', 0)} NEW — see BRING_UP_PLAN.md."
        )

    return ScaffoldPlan(
        new_model_id=new_model_id,
        new_base_name=derive_base_model_name(new_model_id),
        new_tail=new_tail,
        sibling_model_id=sibling_id,
        sibling_base_name=Path(backend.demo_path).name,
        sibling_tail=sibling_tail,
        compat_overall=f"FAMILY TEMPLATE ({probe.category})",
        compat_summary=(f"Scaffolded from `{backend.name}` ({backend.demo_path})." + summary_extra),
        changes=changes,
        skipped=skipped,
        warnings=warnings,
        new_demo_dir=new_demo_dir,
        bringup_plan=bplan,
    )


def apply_scaffold(plan: ScaffoldPlan) -> List[str]:
    from .discovery import BRINGUP_ROOT

    applied: List[str] = []
    write_root = BRINGUP_ROOT()
    for ch in plan.changes:
        target = write_root / ch.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if ch.new_content is None:
            continue
        if ch.preserve_if_exists and target.exists():
            applied.append(f"-  {ch.path}  (preserved; already exists)")
            continue
        # 2026-06-04 Fix 3: refuse to overwrite a valid bringup_status.json
        # (N>0 components) with an empty plan (0 components). This is the
        # smoking-gun corruption pattern from the seamless-m4t promote
        # run: a torch-less subprocess produces an empty plan, scaffold
        # overwrites the canonical manifest, and the iter loop's apply
        # step silently rejects every LLM solution because the components
        # dict is empty. The right behavior is to leave the old manifest
        # in place and let the caller see the failure.
        if target.name == "bringup_status.json" and target.exists():
            try:
                _existing = json.loads(target.read_text())
                _existing_n = len(_existing.get("components", []) or [])
                _new = json.loads(ch.new_content.decode("utf-8"))
                _new_n = len(_new.get("components", []) or [])
                if _existing_n > 0 and _new_n == 0:
                    import sys as _sys

                    print(
                        f"  [scaffold] REFUSED to overwrite {ch.path}: "
                        f"existing has {_existing_n} components, new plan "
                        f"has 0. Likely a model-walk failure (e.g., torch-"
                        f"less Python). Existing manifest preserved.",
                        file=_sys.stderr,
                    )
                    applied.append(f"-  {ch.path}  (REFUSED overwrite: " f"existing={_existing_n} -> new=0 components)")
                    continue
            except Exception:
                # If we can't parse either side, fall through to the
                # normal write — better to write than to silently skip
                # on a transient JSON-decode error.
                pass
        if ch.kind == "edit":
            target.write_bytes(ch.new_content)
            applied.append(f"M  {ch.path}  (+{ch.added_lines} line)")
        else:
            target.write_bytes(ch.new_content)
            applied.append(f"A  {ch.path}")
    return applied


def _render_bringup_summary(bp: BringUpPlan) -> str:
    counts = bp.counts
    lines: List[str] = []
    lines.append("  Component status (REUSE/ADAPT/NEW):")
    lines.append(
        f"    Summary: {counts.get('REUSE', 0)} REUSE, " f"{counts.get('ADAPT', 0)} ADAPT, {counts.get('NEW', 0)} NEW"
    )
    lines.append("")
    name_w = max((len(c.name) for c in bp.components), default=10)
    name_w = min(max(name_w, 10), 26)
    for c in bp.components:
        target = c.tt_reuse_target or c.hf_reference or "—"
        lines.append(f"    [{c.status:5s}]  {c.name.ljust(name_w)}  {target}")
    new_only = [c for c in bp.components if c.status == NEW]
    if new_only:
        lines.append("")
        lines.append(
            f"  Missing components — {len(new_only)} stub(s) generated under `_stubs/` "
            "(replace `NotImplementedError` with a real TTNN port):"
        )
        for c in new_only:
            ref = c.hf_reference or "(no HF reference resolved)"
            lines.append(f"    !  {c.name}  ->  ref: {ref}")
    lines.append("")
    return "\n".join(lines)


def render_text(plan: ScaffoldPlan, *, show_diff: bool = True) -> str:
    sep = "=" * 78
    out: List[str] = [sep]
    out.append(f"  SCAFFOLDING {plan.new_model_id}  (compat={plan.compat_overall})")
    out.append(sep)
    out.append("")
    out.append(f"  Sibling:        {plan.sibling_model_id}")
    out.append(f"  Sibling base:   {plan.sibling_base_name}  ({plan.sibling_tail})")
    out.append(f"  New base:       {plan.new_base_name}  ({plan.new_tail})")
    if plan.compat_summary:
        out.append(f"  Compat note:    {plan.compat_summary}")
    out.append("")

    out.append("  Proposed changes:")
    for ch in plan.changes:
        if ch.kind == "edit":
            out.append(f"    M  {ch.path}  (+{ch.added_lines} line)")
        else:
            out.append(f"    A  {ch.path}")
            if ch.source:
                out.append(f"          copied from {ch.source}")
    out.append("")

    if plan.bringup_plan is not None:
        out.append(_render_bringup_summary(plan.bringup_plan))

    if plan.warnings:
        out.append("  Warnings (review before applying):")
        for w in plan.warnings:
            out.append(f"    [warn] {w}")
        out.append("")

    if plan.skipped:
        out.append("  Skipped:")
        for s in plan.skipped:
            out.append(f"    -  {s}")
        out.append("")

    if show_diff:
        any_diff = any(ch.diff for ch in plan.changes)
        if any_diff:
            out.append("  Diff (edits only; new files listed above):")
            out.append("")
            for ch in plan.changes:
                if ch.diff:
                    for line in ch.diff.splitlines():
                        out.append("    " + line)
                    out.append("")

    out.append("  Next steps:")
    out.append(f"    python -m scripts.tt_hw_planner scaffold {plan.new_model_id} --apply")
    out.append(f"    python -m scripts.tt_hw_planner prepare  {plan.new_model_id} --execute")
    out.append("")
    out.append(sep)
    return "\n".join(out)


def render_apply(plan: ScaffoldPlan, applied: List[str]) -> str:
    sep = "=" * 78
    out: List[str] = [sep]
    out.append(f"  APPLIED scaffold for {plan.new_model_id}")
    out.append(sep)
    out.append("")
    for line in applied:
        out.append(f"    {line}")
    out.append("")
    out.append("  Now run:")
    out.append(f"    python -m scripts.tt_hw_planner prepare {plan.new_model_id} --execute")
    out.append("")
    out.append("  To undo:")
    if plan.compat_overall.startswith("FAMILY TEMPLATE"):
        if plan.new_demo_dir:
            out.append(f"    rm -rf {plan.new_demo_dir}")
        out.append("")
        if plan.bringup_plan is not None and plan.new_demo_dir:
            out.append("  Component-level bring-up plan written to:")
            out.append(f"    {plan.new_demo_dir}/BRING_UP_PLAN.md")
            out.append(f"    {plan.new_demo_dir}/bringup_status.json")
            out.append("")
            out.append(_render_bringup_summary(plan.bringup_plan))
        out.append("  Workflow:")
        out.append("    1. Read BRING_UP_PLAN.md.  REUSE rows = import the sibling tt-module unchanged.")
        out.append("    2. NEW rows = replace the matching `_stubs/*.py` with a real TTNN port.")
        out.append("    3. Once every NEW stub is implemented, re-run `prepare --execute`.")
    else:
        out.append("    git restore models/tt_transformers/tt/model_config.py")
        out.append("    git restore models/tt_transformers/demo/trace_region_config.py")
        out.append(f"    rm -rf models/tt_transformers/model_params/{plan.new_tail}/")
    out.append("")
    out.append(sep)
    return "\n".join(out)


def render_json(plan: ScaffoldPlan, applied: Optional[List[str]] = None) -> str:
    import json

    payload = {
        "new_model_id": plan.new_model_id,
        "sibling_model_id": plan.sibling_model_id,
        "compat_overall": plan.compat_overall,
        "changes": [
            {
                "kind": ch.kind,
                "path": ch.path,
                "source": ch.source,
                "added_lines": ch.added_lines,
            }
            for ch in plan.changes
        ],
        "warnings": plan.warnings,
        "skipped": plan.skipped,
        "applied": applied,
    }
    return json.dumps(payload, indent=2)


def render_patch(plan: ScaffoldPlan) -> str:
    parts: List[str] = []
    creates = [ch for ch in plan.changes if ch.kind == "create"]
    if creates:
        parts.append(f"# scaffold for {plan.new_model_id} — also creates:")
        for ch in creates:
            parts.append(f"#   {ch.path}  (cp {ch.source} {ch.path})")
        parts.append("")
    for ch in plan.changes:
        if ch.diff:
            parts.append(ch.diff)
    return "\n".join(parts)
