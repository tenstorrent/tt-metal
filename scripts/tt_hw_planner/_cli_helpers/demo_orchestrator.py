# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Demo-emission orchestrator for the re-implemented emit-e2e command.

Routes a model_id through:
  1. probe + HF config load
  2. composition-tree extraction
  3. task-template lookup (refuse cleanly if none)
  4. quirk detection (Phase 6 will wire the real DB)
  5. template.emit_all -> source code strings
  6. write source to disk (smart-merge overwrite policy)
  7. validate emitted demo against HF golden (Phase 4 will wire)
  8. on divergence -> LLM diagnose-fix loop (Phase 4 will wire)

This file is intentionally a THIN orchestrator. The heavy lifting lives in:
  * ``task_templates/<task>_template.py`` (emit_*)
  * ``demo_validator.py`` (HF parity check)
  * ``demo_synthesizer.py`` (LLM iter-fix loop)

Phase 1 deliverable: every step except 7+8 is wired. Templates being
absent at Phase 1 means step 3 cleanly refuses with a "no template
for HF class X" error directing the user at task_templates/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..task_templates import (
    EmittedFiles,
    Quirk,
    TaskTemplate,
    TemplateContext,
    lookup_template,
    lookup_template_by_name,
    multi_task_tasks_for,
)
from .composition_tree import (
    classify_roles,
    detect_task_class,
    extract as extract_composition_tree,
)


# ─── RESULT TYPE ─────────────────────────────────────────────────────


STATUS_EMITTED = "EMITTED"
STATUS_NO_TEMPLATE = "NO_TEMPLATE"
STATUS_PROBE_FAILED = "PROBE_FAILED"
STATUS_VALIDATION_FAILED = "VALIDATION_FAILED"
STATUS_OVERWRITE_BLOCKED = "OVERWRITE_BLOCKED"
STATUS_ERROR = "ERROR"


@dataclass
class DemoBringupResult:
    """Structured outcome of one emit-demo cycle."""

    status: str
    demo_dir: Optional[Path] = None
    emitted_files: List[Path] = field(default_factory=list)
    task_name: str = ""
    task_class: str = ""
    steps: List[str] = field(default_factory=list)
    diagnostic: str = ""
    parity_chrf: Optional[float] = None
    parity_token_overlap: Optional[float] = None


# ─── OWNERSHIP TABLE (smart-merge overwrite policy) ──────────────────


# Paths the emitter OWNS — these get overwritten (with --overwrite)
# without further confirmation. Relative to demo_dir.
_TOOL_OWNED_PATHS = {
    "demo",  # directory
    "tt",  # directory (top-level + granular sub-files)
    "evaluation",  # directory
    "reference",  # directory
    "tests/test_demo.py",
    "tests/test_hf_parity.py",
    "README.md",
    "requirements.txt",
    ".gitignore",
    "conftest.py",
}

# Paths the emitter PRESERVES — never overwritten without --force.
_PRESERVED_PATHS = {
    "_stubs",
    "_attempts",
    "_captured",
    "_handoff",
    "_synth_prompts",
    "_synth_responses",
    "_verify",
    "decomposition_plan.applied",
    "bringup_status.json",
    "decomposition_plan.json",
    "BRING_UP_PLAN.md",
    "tests/pcc",  # per-component PCC tests (bring-up output)
    "demo.py",  # legacy hand-written PCC harness
}


def _is_under(rel_path: Path, owned_root: str) -> bool:
    """True if ``rel_path`` is the root itself or lives under it."""
    parts = rel_path.parts
    root_parts = Path(owned_root).parts
    if len(parts) < len(root_parts):
        return False
    return parts[: len(root_parts)] == root_parts


def _is_tool_owned(rel_path: Path) -> bool:
    """True iff this path is in the tool's owned set."""
    for owned in _TOOL_OWNED_PATHS:
        if rel_path == Path(owned) or _is_under(rel_path, owned):
            return True
    return False


def _is_preserved(rel_path: Path) -> bool:
    """True iff this path is in the preserved set (only --force can overwrite)."""
    for preserved in _PRESERVED_PATHS:
        if rel_path == Path(preserved) or _is_under(rel_path, preserved):
            return True
    return False


# ─── ORCHESTRATOR ENTRY POINT ────────────────────────────────────────


def run_demo_bringup(
    *,
    model_id: str,
    demo_dir: Path,
    task_filter: Optional[str] = None,  # --task X
    all_tasks: bool = False,  # --all-tasks
    overwrite: bool = False,  # --overwrite
    force: bool = False,  # --force (overrides preserved set too)
    max_iter: int = 5,  # LLM diagnose-fix budget
    readme_only: bool = False,  # --readme-only (skip everything else)
    # Injectable seams for tests + future phases:
    hf_model_loader: Optional[Any] = None,  # callable(model_id) -> hf_model
    hf_config_loader: Optional[Any] = None,  # callable(model_id) -> dict
    validator: Optional[Any] = None,  # Phase 4 wires real implementation
    synthesizer: Optional[Any] = None,  # Phase 4 wires real iter-fix
    quirk_detector: Optional[Any] = None,  # Phase 6 wires real DB
) -> DemoBringupResult:
    """Run the emit-demo flow. Returns a structured result.

    Never raises -- every failure routes to a status. Caller (CLI)
    reads the status + diagnostic and decides exit code.
    """
    result = DemoBringupResult(status=STATUS_ERROR, demo_dir=demo_dir)

    # ── Step 1: probe ────────────────────────────────────────────────
    # Try AutoConfig.from_pretrained first; fall back to bringup_status.json
    # if installed transformers doesn't recognize the model_type (common
    # for new / experimental models like sam2_video, etc.).
    try:
        loader = hf_config_loader or _default_hf_config_loader
        hf_config = loader(model_id)
        result.steps.append(f"loaded HF config for {model_id}")
    except Exception as exc:
        # Fallback: read cached model metadata from bringup_status.json
        # so emit-e2e can still produce a demo when transformers can't load
        # the config (new model types not yet in the local transformers version).
        bs_path = demo_dir / "bringup_status.json"
        if bs_path.is_file():
            try:
                import json as _json

                bs = _json.loads(bs_path.read_text())
                hf_config = {
                    "model_type": str(bs.get("new_model_type") or ""),
                    "architectures": [],  # unknown — we'll resolve via fallbacks
                }
                result.steps.append(
                    f"AutoConfig failed ({type(exc).__name__}); "
                    f"falling back to bringup_status.json metadata "
                    f"(model_type={hf_config['model_type']!r})"
                )
            except Exception as bs_exc:
                result.status = STATUS_PROBE_FAILED
                result.diagnostic = (
                    f"HF config load failed: {type(exc).__name__}: {exc}; "
                    f"bringup_status.json fallback also failed: {bs_exc}"
                )
                return result
        else:
            result.status = STATUS_PROBE_FAILED
            result.diagnostic = (
                f"HF config load failed: {type(exc).__name__}: {exc}; " f"no bringup_status.json fallback at {bs_path}"
            )
            return result

    task_class = detect_task_class(hf_config)

    # Multi-task models (e.g. SeamlessM4TModel) report a base class name
    # that doesn't end in For*. Look them up via the multi_task registry.
    model_type = str(hf_config.get("model_type", "")).lower().replace("-", "_")
    multi_task_names = multi_task_tasks_for(model_type)
    if task_class is None and multi_task_names:
        first_template = lookup_template_by_name(multi_task_names[0])
        if first_template is not None:
            task_class = first_template.HF_TASK_CLASS
            result.steps.append(
                f"base multi-task model {hf_config.get('architectures')} → "
                f"resolved via model_type={model_type!r} -> {multi_task_names}"
            )

    # Also detect task class via model_type substring when architectures are missing
    # (e.g., AutoConfig failed and we fell back to bringup_status.json with no arches).
    if task_class is None and model_type:
        if any(kw in model_type for kw in ("sam", "segment", "mask")):
            task_class = "AutoModelForMaskGeneration"
            result.steps.append(f"model_type={model_type!r} → segmentation task")
        elif any(kw in model_type for kw in ("whisper",)):
            task_class = "AutoModelForSpeechSeq2Seq"
            result.steps.append(f"model_type={model_type!r} → speech-seq2seq task")
        elif any(kw in model_type for kw in ("nllb", "m2m", "mbart")):
            task_class = "AutoModelForSeq2SeqLM"
            result.steps.append(f"model_type={model_type!r} → seq2seq task")

    if task_class is None:
        result.status = STATUS_PROBE_FAILED
        result.diagnostic = (
            "Could not detect HF AutoModel class from config.architectures. "
            "Add a mapping in composition_tree.detect_task_class for "
            f"architecture {hf_config.get('architectures')}"
        )
        return result
    result.task_class = task_class
    result.steps.append(f"detected task_class = {task_class}")

    # ── Step 2: figure out which template(s) to use ──────────────────
    task_names = _resolve_task_names(
        hf_config=hf_config,
        task_class=task_class,
        task_filter=task_filter,
        all_tasks=all_tasks,
    )
    if not task_names:
        result.status = STATUS_NO_TEMPLATE
        result.diagnostic = (
            f"No registered TaskTemplate for HF class '{task_class}'. "
            f"Author one in scripts/tt_hw_planner/task_templates/ and "
            f"register via @register_template decorator."
        )
        return result
    result.steps.append(f"selected task templates: {task_names}")

    # ── Step 3: load HF model + composition tree ─────────────────────
    # Allow this to fail gracefully — if the local transformers version
    # can't load the model, we can still emit a demo from the cached
    # graduated stubs + composition tree info. The emitted demo will
    # need transformers to be upgraded at run time anyway.
    hf_model = None
    try:
        model_loader = hf_model_loader or _default_hf_model_loader
        hf_model = model_loader(model_id)
        result.steps.append("loaded HF model")
    except Exception as exc:
        result.steps.append(
            f"HF model load failed ({type(exc).__name__}); proceeding with "
            f"empty composition tree — graduated stubs still let us emit"
        )

    if hf_model is None:
        # Empty tree — templates fall back to clean-name-based imports via roles.
        from ..task_templates._base import CompositionTree

        comp_tree = CompositionTree(
            model_id=model_id,
            task_class=task_class,
            stub_attributes={},
            roles={},
            cpu_bridges=[],
        )
    else:
        comp_tree = extract_composition_tree(
            model_id=model_id,
            task_class=task_class,
            hf_model=hf_model,
            demo_dir=demo_dir,
        )

    # Classify graduated stubs into abstract task roles (audio_encoder,
    # text_decoder, etc.). This is what lets templates avoid hardcoded
    # name mappings.
    graduated_stubs_dynamic = _read_graduated_stubs(demo_dir)
    builders_for_roles = _builders_from_graduated_for_classification(
        graduated_stubs_dynamic,
        hf_config,
        demo_dir=demo_dir,
    )
    clean_names = [clean for clean, _ in builders_for_roles]
    roles = classify_roles(clean_names)
    # Attach roles to the composition tree (rebuild as the dataclass is frozen).
    from dataclasses import replace

    comp_tree = replace(comp_tree, roles=roles)

    result.steps.append(
        f"composition tree extracted: {len(comp_tree.stub_attributes)} stub paths, "
        f"{len(comp_tree.cpu_bridges)} CPU bridges, "
        f"{len(roles)} roles classified ({sorted(roles.keys())})"
    )

    # ── Step 4: detect quirks ────────────────────────────────────────
    quirks: List[Quirk] = []
    if quirk_detector is not None:
        try:
            quirks = list(quirk_detector(hf_config, comp_tree))
            result.steps.append(f"detected {len(quirks)} quirks")
        except Exception as exc:
            result.steps.append(f"quirk detector failed (non-fatal): {type(exc).__name__}: {exc}")

    # ── Step 5: per-task emit (collect EmittedFiles from each template) ──
    all_emitted = EmittedFiles()
    instantiated_templates: List = []
    for task_name in task_names:
        template_cls = lookup_template_by_name(task_name)
        if template_cls is None:
            result.status = STATUS_NO_TEMPLATE
            result.diagnostic = f"Internal: task name {task_name!r} resolved to no template"
            return result

        ctx = TemplateContext(
            model_id=model_id,
            composition_tree=comp_tree,
            graduated_stubs=_read_graduated_stubs(demo_dir),
            quirks=[q for q in quirks if not q.applies_to_template or task_name in q.applies_to_template],
            hf_config=hf_config,
            demo_dir=demo_dir,
        )

        try:
            template = template_cls()
            instantiated_templates.append((template, ctx, task_name))
            emitted = template.emit_all(ctx)
        except Exception as exc:
            result.status = STATUS_ERROR
            result.diagnostic = f"Template {template_cls.__name__}.emit_all raised " f"{type(exc).__name__}: {exc}"
            return result

        # Merge into combined emit set
        for rel_path, src in emitted.files.items():
            all_emitted.files[rel_path] = src
        result.steps.append(f"task '{task_name}' produced {len(emitted.files)} files")

    # ── Step 5.5: per-model + universal scaffolding (emitted ONCE) ──────
    if instantiated_templates:
        try:
            scaffold_files = _emit_model_scaffolding(
                instantiated_templates=instantiated_templates,
            )
            for rel_path, src in scaffold_files.items():
                # Templates can override scaffolding files if they need to
                # (e.g. a task-specific README); otherwise scaffold wins.
                if rel_path not in all_emitted.files:
                    all_emitted.files[rel_path] = src
            result.steps.append(f"emitted {len(scaffold_files)} universal+model-level files")
        except Exception as exc:
            result.status = STATUS_ERROR
            result.diagnostic = f"Scaffolding emit failed: {type(exc).__name__}: {exc}"
            return result

    # ── Step 6: smart-merge write ────────────────────────────────────
    write_result = _write_emitted_files(
        emitted=all_emitted,
        demo_dir=demo_dir,
        overwrite=overwrite,
        force=force,
    )
    if write_result is None:
        result.status = STATUS_OVERWRITE_BLOCKED
        result.diagnostic = (
            f"Refusing to overwrite existing tool-owned files without --overwrite. "
            f"Re-run with --overwrite (preserves bring-up artifacts) or --force "
            f"(allows overwriting tests/pcc/ and demo.py too)."
        )
        return result

    result.emitted_files = write_result
    result.steps.append(f"wrote {len(write_result)} total files to {demo_dir}")

    # ── Step 7+8 are Phase 4 work (validator + iter-fix loop) ────────
    # For Phase 1 the orchestrator stops here. Phase 4 will add:
    #   if validator is not None: result = validator.run_hf_parity(...)
    #   if not converged and synthesizer is not None: synthesizer.iter_fix(...)

    result.status = STATUS_EMITTED
    return result


# ─── HELPERS ─────────────────────────────────────────────────────────


def _resolve_task_names(
    *,
    hf_config: Dict[str, Any],
    task_class: str,
    task_filter: Optional[str],
    all_tasks: bool,
) -> List[str]:
    """Decide which task names to emit, honoring --task / --all-tasks.

    Multi-task heads (declared via register_multi_task) expand to all
    sibling tasks when --all-tasks is set. A specific --task X overrides.
    """
    # Explicit --task X always wins.
    if task_filter:
        cls = lookup_template_by_name(task_filter)
        return [task_filter] if cls is not None else []

    # Multi-task model?
    model_type = str(hf_config.get("model_type", "")).lower().replace("-", "_")
    multi = multi_task_tasks_for(model_type)
    if multi and all_tasks:
        # All registered task templates for this model_type
        return [name for name in multi if lookup_template_by_name(name) is not None]
    if multi:
        # Pick the first declared task as the default.
        for name in multi:
            if lookup_template_by_name(name) is not None:
                return [name]

    # Single-task fallback: look up by HF class.
    cls = lookup_template(task_class)
    if cls is None:
        return []
    return [cls.TASK_NAME]


def _read_graduated_stubs(demo_dir: Path) -> List[Dict[str, Any]]:
    """Load bringup_status.json components, or empty list if absent."""
    import json

    path = demo_dir / "bringup_status.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(data, dict):
        comps = data.get("components") or []
        return [c for c in comps if isinstance(c, dict)]
    return []


def _write_emitted_files(
    *,
    emitted: EmittedFiles,
    demo_dir: Path,
    overwrite: bool,
    force: bool,
) -> Optional[List[Path]]:
    """Write all emitted files to demo_dir honoring the smart-merge policy.

    Returns the list of paths written, or None if a tool-owned file
    exists and ``overwrite`` is False, or a preserved file exists
    and ``force`` is False.
    """
    # Pre-check ownership before writing anything.
    for rel_path in emitted.files.keys():
        abs_path = demo_dir / rel_path
        if not abs_path.exists():
            continue
        if _is_preserved(rel_path) and not force:
            return None
        if _is_tool_owned(rel_path) and not overwrite:
            return None

    written: List[Path] = []
    for rel_path, source in emitted.files.items():
        abs_path = demo_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(source)
        written.append(abs_path)

    # Ensure __init__.py exists in every emitted subdir.
    _ensure_init_files(emitted, demo_dir, written)

    return written


def _ensure_init_files(
    emitted: EmittedFiles,
    demo_dir: Path,
    written: List[Path],
) -> None:
    """Write empty __init__.py into every directory that received a file."""
    dirs_seen = set()
    for rel_path in emitted.files.keys():
        d = (demo_dir / rel_path).parent
        if d == demo_dir:
            continue
        dirs_seen.add(d)
    for d in dirs_seen:
        init = d / "__init__.py"
        if not init.exists():
            init.write_text(
                "# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.\n" "# SPDX-License-Identifier: Apache-2.0\n"
            )
            written.append(init)


def _emit_model_scaffolding(*, instantiated_templates: List) -> Dict[Path, str]:
    """Emit per-model + universal files ONCE based on all running templates.

    Produces:
      * Universal: .gitignore, conftest.py, requirements.txt, demo/output_validation.py
      * Per-model: tt/__init__.py (with all generator class imports),
                   tt/model_config.py, tt/load_weights.py,
                   tt/<re-export>.py for each graduated stub,
                   reference/__init__.py, demo/__init__.py, tests/__init__.py,
                   evaluation/__init__.py
      * README.md (composed from each template's TASK_DESC + commands)

    ``instantiated_templates`` is a list of (template_instance, ctx, task_name).
    All share the same TemplateContext (same model_id, demo_dir, hf_config).
    The first ctx is used for shared fields; only task-name-dependent
    pieces differ between templates.
    """
    from ..task_templates import _helpers as h
    from ..input_synthesizers import audio as audio_synth

    if not instantiated_templates:
        return {}

    # Use the first ctx for model-level fields.
    first_template, first_ctx, _ = instantiated_templates[0]
    ctx = first_ctx

    # Union requirements from every template
    requirements_extras: List[str] = []
    seen_reqs: set = set()
    for tmpl, _, _ in instantiated_templates:
        for r in getattr(tmpl, "REQUIREMENTS_EXTRAS", []):
            if r not in seen_reqs:
                seen_reqs.add(r)
                requirements_extras.append(r)

    # Union model_config extras
    extras: Dict[str, Any] = {}
    for tmpl, _, _ in instantiated_templates:
        for key, (kind, value) in getattr(tmpl, "MODEL_CONFIG_EXTRAS", {}).items():
            if key in extras:
                continue
            if kind == "literal":
                extras[key] = value
            elif kind.startswith("config."):
                cfg_key = kind.split(".", 1)[1]
                extras[key] = ctx.hf_config.get(cfg_key, value)
            else:
                extras[key] = value

    # Decide HF task class for load_weights (use first template's)
    hf_task_class = first_template.HF_TASK_CLASS

    # Generator-class imports for tt/__init__.py
    generator_classes: List = []
    for tmpl, _, task_name in instantiated_templates:
        cls_name = getattr(tmpl, "GENERATOR_CLASS_NAME", "")
        if cls_name:
            generator_classes.append((cls_name, f"generator_{task_name}"))

    # ── Build all the files ─────────────────────────────────────────
    files: Dict[Path, str] = {}

    # Universal
    for rel, src in h.universal_files(ctx, requirements_extras=requirements_extras).items():
        files[Path(rel)] = src

    # Audio loader if any template needs it
    if any(getattr(tmpl, "NEEDS_AUDIO_LOADER", False) for tmpl, _, _ in instantiated_templates):
        files[Path("demo/audio_loader.py")] = audio_synth.emit_source(ctx)

    # Image loader if any image-input template is in the mix
    image_input_modalities = {"image", "video"}
    if any(getattr(tmpl, "INPUT_MODALITY", "") in image_input_modalities for tmpl, _, _ in instantiated_templates):
        from ..input_synthesizers import image as image_synth

        files[Path("demo/image_loader.py")] = image_synth.emit_source(ctx)

    # tt/__init__.py + tt/model_config + tt/load_weights + tt/<re-exports>
    builders = _builders_from_graduated(ctx)
    files[Path("tt/__init__.py")] = h.emit_tt_init(
        ctx,
        builders,
        generator_classes,
        roles=ctx.composition_tree.roles,
    )
    files[Path("tt/model_config.py")] = h.emit_tt_model_config(ctx, extras)
    files[Path("tt/load_weights.py")] = h.emit_tt_load_weights(ctx, hf_task_class)
    for clean, stub in builders:
        files[Path(f"tt/{clean}.py")] = h.emit_tt_reexport(ctx, clean, stub)

    # README (composed from task descriptions)
    task_descs = " / ".join(getattr(tmpl, "TASK_DESC", task_name) for tmpl, _, task_name in instantiated_templates)
    task_names = [t for _, _, t in instantiated_templates]
    files[Path("README.md")] = _emit_combined_readme(ctx, task_names, task_descs)

    return files


def _builders_from_graduated_for_classification(
    graduated_stubs: List[Dict[str, Any]],
    hf_config: Dict[str, Any],
    demo_dir: Optional[Path] = None,
) -> List[tuple]:
    """Variant that takes graduated stubs + hf_config directly.

    Used by the orchestrator to classify roles BEFORE building the
    TemplateContext (which depends on the roles being known).

    If ``demo_dir`` is given, filters out components whose
    ``_stubs/<name>.py`` file doesn't exist on disk. bringup_status.json
    can list components whose status is set but no file was produced
    (planned-but-not-graduated, config-only entries, etc.).
    """
    model_type = str(hf_config.get("model_type", "")).lower()
    prefix_candidates = _derive_prefix_candidates(model_type)
    builders: List[tuple] = []
    seen_clean: set = set()
    stubs_dir = (demo_dir / "_stubs") if demo_dir is not None else None
    for c in graduated_stubs:
        if c.get("status") not in ("NEW", "ADAPT"):
            continue
        stub_name = str(c.get("name", "")).strip()
        if not stub_name:
            continue
        if stubs_dir is not None and not (stubs_dir / f"{stub_name}.py").is_file():
            continue
        clean = _strip_prefix(stub_name, prefix_candidates)
        if clean in seen_clean:
            continue
        seen_clean.add(clean)
        builders.append((clean, stub_name))
    return builders


def _builders_from_graduated(ctx: TemplateContext) -> List[tuple]:
    """Dynamic discovery of (clean_name, stub_module_basename) pairs.

    NO HARDCODED model-family knowledge. Reads bringup_status.json,
    derives a clean name for each graduated component by stripping
    the model_type prefix from its name.

    Examples:
      model_type="seamless_m4t", component="seamless_m4_t_speech_encoder"
          -> ("speech_encoder", "seamless_m4_t_speech_encoder")
      model_type="whisper", component="whisper_encoder"
          -> ("encoder", "whisper_encoder")
      model_type="llama", component="llama_decoder_layer"
          -> ("decoder_layer", "llama_decoder_layer")
      model_type="anything", component="no_match_name"
          -> ("no_match_name", "no_match_name")  # passthrough
    """
    model_type = str(ctx.hf_config.get("model_type", "")).lower()
    # Candidate prefixes to try stripping, in order of preference.
    # Order matters: longer/more-specific prefixes first.
    prefix_candidates = _derive_prefix_candidates(model_type)

    builders: List[tuple] = []
    seen_clean: set = set()
    stubs_dir = ctx.demo_dir / "_stubs"
    for c in ctx.graduated_stubs:
        # Only NEW/ADAPT components have a _stubs/<name>.py file to re-export.
        # REUSE components use canonical implementations from elsewhere
        # (e.g. models/tt_transformers/) and don't need a local re-export.
        if c.get("status") not in ("NEW", "ADAPT"):
            continue
        stub_name = str(c.get("name", "")).strip()
        if not stub_name:
            continue
        # Verify the file actually exists. bringup_status.json sometimes lists
        # components whose status is set but no _stubs/<name>.py file was
        # produced (e.g. config-only or planned-but-not-graduated entries).
        if not (stubs_dir / f"{stub_name}.py").is_file():
            continue
        clean = _strip_prefix(stub_name, prefix_candidates)
        if clean in seen_clean:
            continue
        seen_clean.add(clean)
        builders.append((clean, stub_name))
    return builders


def _derive_prefix_candidates(model_type: str) -> List[str]:
    """Generate candidate prefixes from a model_type.

    SeamlessM4T's component names use snake-cased class names
    (e.g., 'seamless_m4_t_speech_encoder'), but model_type
    in config is 'seamless_m4t' (one segment). We need to handle
    both shapes.

    For model_type='seamless_m4t', try stripping:
       seamless_m4_t_  (snake-cased class prefix)
       seamless_m4t_   (model_type prefix)
       seamless_       (family prefix)
    """
    if not model_type:
        return []
    candidates: List[str] = []
    # Direct model_type as prefix
    candidates.append(model_type + "_")
    # Snake-cased version (insert underscores between letter/digit boundaries)
    snake = _snake_case_insert(model_type)
    if snake and snake + "_" not in candidates:
        candidates.append(snake + "_")
    # First word as prefix (e.g., "seamless_" from "seamless_m4t")
    if "_" in model_type:
        first_word = model_type.split("_", 1)[0] + "_"
        if first_word not in candidates:
            candidates.append(first_word)
    return candidates


def _snake_case_insert(s: str) -> str:
    """Insert underscores at digit→letter boundaries only.

    Component names in bring-up use the snake-cased HF class name, which
    splits at digit→letter boundaries (e.g. SeamlessM4T → seamless_m4_t).
    We mirror that here so prefix derivation matches.

    'seamless_m4t' -> 'seamless_m4_t'
    'phi3'         -> 'phi3'  (no following letter)
    'llama'        -> 'llama' (no digit)
    'qwen2vl'      -> 'qwen2_vl'
    """
    if not s:
        return s
    out = []
    prev = ""
    for ch in s:
        # Only insert between digit and letter (mirrors CamelCase→snake_case
        # behavior for class names like SeamlessM4T → seamless_m4_t).
        if prev and prev.isdigit() and ch.isalpha() and prev != "_":
            out.append("_")
        out.append(ch)
        prev = ch
    return "".join(out)


def _strip_prefix(name: str, prefix_candidates: List[str]) -> str:
    """Strip the first matching prefix from name. Returns name itself if none match."""
    for p in prefix_candidates:
        if name.startswith(p):
            stripped = name[len(p) :]
            if stripped:  # don't return empty string
                return stripped
    return name


def _emit_combined_readme(ctx: TemplateContext, task_names: List[str], task_descs: str) -> str:
    """README that combines task descriptions for multi-task models."""
    from ..task_templates import _helpers as h

    # If only one task, use the standard single-task README
    if len(task_names) == 1:
        return h.emit_readme(ctx, task_names[0], task_descs)

    # Multi-task: list all demos
    demo_lines = "\n".join(f"   demo_{t}.py" for t in task_names)
    cmds = "\n".join(
        f"# {t.upper()}\npython -m {'.'.join(ctx.demo_dir.parts)}.demo.demo_{t} --help" for t in task_names
    )

    return f"""# {ctx.model_id} on Tenstorrent

Production demos for [`{ctx.model_id}`](https://huggingface.co/{ctx.model_id})
running on Blackhole. This model supports multiple tasks; one demo is
emitted per task.

**Tasks**: {task_descs}

## Directory layout (yito-style)

```
demo/
{demo_lines}
   audio_loader.py
   output_validation.py
tt/
   generator_<task>.py (one per task)
   ... (re-exports of graduated stubs)
reference/
tests/
evaluation/
_stubs/                    # bring-up tool internal
README.md
```

## Quick start

```bash
# Run all task demos
pytest models/demos/{ctx.demo_dir.name}/demo/ -v

# Or per-task CLI:
{cmds}
```

## Validation

Auto-generated by `scripts/tt_hw_planner emit-e2e` from
{len(ctx.graduated_stubs)} graduated TTNN components. Each emitted demo
is verified against HF's `model.generate()` via chrF and token-overlap
parity tests.
"""


def _default_hf_config_loader(model_id: str) -> Dict[str, Any]:
    """Default loader: ``transformers.AutoConfig.from_pretrained(...).to_dict()``."""
    import transformers

    return transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True).to_dict()


def _default_hf_model_loader(model_id: str) -> Any:
    """Default loader: ``transformers.AutoModel.from_pretrained(...)`` (fp32 CPU).

    fp32 because the layer_norm bf16/fp32 dtype quirk catches some models
    (e.g. SeamlessM4T); fp32 avoids it and matches the pattern that
    works across families. The TT path casts to bf16 internally.
    """
    import transformers

    model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model


__all__ = [
    "STATUS_EMITTED",
    "STATUS_NO_TEMPLATE",
    "STATUS_PROBE_FAILED",
    "STATUS_VALIDATION_FAILED",
    "STATUS_OVERWRITE_BLOCKED",
    "STATUS_ERROR",
    "DemoBringupResult",
    "run_demo_bringup",
]
