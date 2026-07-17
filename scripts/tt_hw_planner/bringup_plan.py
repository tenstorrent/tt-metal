from __future__ import annotations

from .discovery import safe_relative_to_root

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .family_backends import FamilyBackend
from .probe import probe_model
from .reuse_registry import lookup as _reuse_lookup
from .reuse_registry import lookup_by_concept as _reuse_lookup_by_concept


REUSE = "REUSE"
ADAPT = "ADAPT"
NEW = "NEW"
# ADAPT restored 2026-06-01 with iterate-loop integration. Semantics:
#   REUSE = tt-module works as-is, trust it (no test, no LLM)
#   ADAPT = tt-module exists, may need wrapper/config refinement
#          (working canonical-wrapper stub, iter-0 PCC test,
#          LLM refines on PCC<0.99 — never rewrites the class)
#   NEW   = no tt-module exists, LLM writes from scratch


@dataclass
class Component:
    name: str
    kind: str
    status: str
    new_shape: Dict[str, Any] = field(default_factory=dict)
    sibling_shape: Dict[str, Any] = field(default_factory=dict)
    tt_reuse_target: Optional[str] = None
    hf_reference: Optional[str] = None
    notes: str = ""

    submodule_path: Optional[str] = None
    class_name: Optional[str] = None


@dataclass
class BringUpPlan:
    new_model_id: str
    new_model_type: Optional[str]
    sibling_hf_id: Optional[str]
    sibling_model_type: Optional[str]
    backend_name: str
    backend_demo_path: str
    components: List[Component] = field(default_factory=list)
    common_reuse: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # Serialised KernelFinding entries (WARN+BLOCKER) deduped across TPs.
    # Populated by build_bringup_plan via kernel_constraints.evaluate_kernels.
    # Empty list means kernel_constraints raised or the config was unusable
    # — consumers must tolerate that.
    kernel_findings: List[Dict[str, Any]] = field(default_factory=list)
    ranked_siblings: List[Tuple[str, int, str]] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {REUSE: 0, ADAPT: 0, NEW: 0}
        for c in self.components:
            out[c.status] = out.get(c.status, 0) + 1
        return out


_COMMON_REUSE_TABLE: List[Tuple[str, str, str]] = [
    ("LayerNorm / RMSNorm", "models/common/rmsnorm.py", "shared norm primitive"),
    ("LightweightModule base", "models/common/lightweightmodule.py", "TT-side nn.Module analog"),
    ("Tensor helpers", "models/common/tensor_utils.py", "host<->device move + sharding helpers"),
    ("Generic utility funcs", "models/common/utility_functions.py", "padding, layout, dtype helpers"),
]


_NEST_KEYS: List[Tuple[str, str]] = [
    ("vision_config", "vision_encoder"),
    ("text_config", "text_encoder"),
    ("prompt_encoder_config", "prompt_encoder"),
    ("mask_decoder_config", "mask_decoder"),
    ("decoder_config", "decoder"),
    ("audio_config", "audio_encoder"),
    ("encoder_config", "encoder"),
]


def _flat(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return {}
    if "hidden_size" in cfg or "num_hidden_layers" in cfg:
        return cfg
    for k in ("text_config", "vision_config", "encoder_config"):
        sub = cfg.get(k)
        if isinstance(sub, dict) and ("hidden_size" in sub or "num_hidden_layers" in sub):
            return sub
    return cfg


def _shape(cfg: dict) -> Dict[str, Any]:
    c = _flat(cfg) if isinstance(cfg, dict) else {}
    if not isinstance(c, dict):
        return {}
    keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "hidden_act",
        "layer_norm_eps",
        "patch_size",
        "image_size",
        "window_size",
        "depths",
        "num_channels",
        "vocab_size",
        "max_position_embeddings",
    )
    return {k: c.get(k) for k in keys if c.get(k) is not None}


def _diff_status(new_shape: Dict[str, Any], sibling_shape: Dict[str, Any]) -> str:
    # A sibling tt-module is never trusted blind: whether shapes match or
    # differ, it is ADAPT (canonical-wrapper stub + per-component PCC test +
    # LLM refinement, graduate on native pass). Only a missing sibling is NEW.
    if not sibling_shape:
        return NEW
    shared_keys = [k for k in new_shape if k in sibling_shape]
    if not shared_keys:
        return NEW
    return ADAPT


def _sibling_tt_file(
    backend: FamilyBackend,
    repo_root: Path,
    hint: str,
    extra_backends: Optional[List[FamilyBackend]] = None,
) -> Optional[str]:
    """Reuse target for one component, primary backend first then falling
    through the ranked runner-up siblings (fixes-plan Point 2b) so a component's
    reuse target comes from whichever sibling actually provides it — not only
    the single locked family."""
    for be in [backend, *(extra_backends or [])]:
        hit = _sibling_tt_file_one(be, repo_root, hint)
        if hit:
            return hit
    return None


def _sibling_tt_file_one(backend: FamilyBackend, repo_root: Path, hint: str) -> Optional[str]:
    base = (repo_root / backend.demo_path).resolve()
    if not base.is_dir():
        return None
    needle = hint.lower()
    tt_dirs = {"tt", "ttnn"}
    matches: List[Path] = []
    for p in base.rglob("*.py"):
        parts = {q.lower() for q in p.parts}
        if not (parts & tt_dirs):
            continue
        if needle in p.name.lower():
            matches.append(p)
    if matches:
        matches.sort(key=lambda q: (len(q.parts), q.name))
        return str(safe_relative_to_root(matches[0]))

    demo_slug = Path(backend.demo_path).name.lower()
    slug_aliases = {demo_slug, demo_slug.replace("_", "")}
    if not demo_slug:
        return None
    for p in base.rglob("*.py"):
        parts = {q.lower() for q in p.parts}
        if not (parts & tt_dirs):
            continue
        low = p.name.lower()
        if any(s and s in low for s in slug_aliases):
            return str(safe_relative_to_root(p))
    return None


def _hf_reference(new_model_type: Optional[str], suffix: str) -> Optional[str]:
    if not new_model_type:
        return None
    return f"transformers/src/transformers/models/{new_model_type}/{suffix}"


def _extract_components_vision(
    *,
    new_cfg: dict,
    sibling_cfg: dict,
    backend: FamilyBackend,
    repo_root: Path,
    new_model_type: Optional[str],
    extra_backends: Optional[List[FamilyBackend]] = None,
) -> List[Component]:
    out: List[Component] = []

    new_nest_keys = {k for k, _ in _NEST_KEYS if isinstance(new_cfg.get(k), dict)}
    sib_nest_keys = {k for k, _ in _NEST_KEYS if isinstance(sibling_cfg.get(k), dict)}

    for cfg_key, kind in _NEST_KEYS:
        if cfg_key not in new_nest_keys:
            continue
        new_sub = new_cfg.get(cfg_key, {}) or {}
        sib_sub = sibling_cfg.get(cfg_key, {}) or {}
        status = _diff_status(_shape(new_sub), _shape(sib_sub))
        if cfg_key not in sib_nest_keys:
            status = NEW
        tt_target = _sibling_tt_file(backend, repo_root, kind.split("_")[0], extra_backends=extra_backends)
        out.append(
            Component(
                name=cfg_key,
                kind=kind,
                status=status,
                new_shape=_shape(new_sub),
                sibling_shape=_shape(sib_sub),
                tt_reuse_target=tt_target if status != NEW else None,
                hf_reference=_hf_reference(new_model_type, "modeling_*.py") if status == NEW else None,
                notes=(
                    "Sibling has no analog of this nest — fully new TTNN port required."
                    if status == NEW
                    else "Same nest key present in sibling; verify field-by-field below."
                ),
            )
        )

    new_flat = _shape(new_cfg)
    sib_flat = _shape(sibling_cfg)
    primitives = [
        ("patch_embed", "patch_embed"),
        ("self_attention", "attention"),
        ("mlp", "mlp"),
        ("layer", "layer"),
        ("encoder_stack", "encoder"),
        ("decoder_head", "decode_head"),
    ]
    for comp_name, hint in primitives:
        registry_hit = _reuse_lookup_by_concept(new_model_type, hint or comp_name)
        if registry_hit is not None:
            out.append(
                Component(
                    name=comp_name,
                    kind=comp_name,
                    status=registry_hit.status,
                    new_shape=new_flat,
                    sibling_shape=sib_flat,
                    tt_reuse_target=registry_hit.tt_path,
                    notes=(
                        f"reuse_registry: {registry_hit.concept} -> "
                        f"{registry_hit.tt_path}::{registry_hit.tt_class} "
                        f"({registry_hit.status}). {registry_hit.notes}"
                    ),
                )
            )
            continue
        tt_target = _sibling_tt_file(backend, repo_root, hint, extra_backends=extra_backends)
        if tt_target is None:
            continue
        status = _diff_status(new_flat, sib_flat)
        if new_model_type and sibling_cfg.get("model_type") and new_model_type != sibling_cfg.get("model_type"):
            status = NEW if comp_name in {"self_attention", "encoder_stack", "decoder_head"} else status
        out.append(
            Component(
                name=comp_name,
                kind=comp_name,
                status=status,
                new_shape=new_flat,
                sibling_shape=sib_flat,
                tt_reuse_target=tt_target if status != NEW else None,
                hf_reference=(
                    _hf_reference(new_model_type, f"modeling_{new_model_type}.py") if status == NEW else None
                ),
                notes=(
                    "Sibling has a file with the same role — port the math; "
                    "shapes match so weight loader can clone the sibling's."
                    if status == REUSE
                    else (
                        # NEW covers both "architecturally distinct"
                        # and "same role, different shapes" since the
                        # ADAPT intermediate status was removed
                        # 2026-05-31. The agent reads the existing
                        # sibling file (if any) and decides at
                        # iteration time whether to tweak constants or
                        # rewrite the forward pass.
                        "Either architecturally distinct from sibling, or same role "
                        "with different shapes — write/adapt the TTNN module against "
                        "the HF reference impl above. If a sibling tt-file with the "
                        "same role exists, reuse its layout and update the shape "
                        "constants (hidden_size / num_heads / intermediate_size); "
                        "otherwise write from scratch."
                    )
                ),
            )
        )

    return out


def _extract_components_generic(
    *,
    new_cfg: dict,
    sibling_cfg: dict,
    backend: FamilyBackend,
    repo_root: Path,
    new_model_type: Optional[str],
    extra_backends: Optional[List[FamilyBackend]] = None,
) -> List[Component]:
    new_flat = _shape(new_cfg)
    sib_flat = _shape(sibling_cfg)
    primitives = [
        ("token_embed", "embed"),
        ("attention", "attention"),
        ("mlp", "mlp"),
        ("layer", "layer"),
        ("encoder_stack", "encoder"),
        ("decoder_head", "head"),
    ]
    out: List[Component] = []
    for comp_name, hint in primitives:
        registry_hit = _reuse_lookup_by_concept(new_model_type, hint or comp_name)
        if registry_hit is not None:
            out.append(
                Component(
                    name=comp_name,
                    kind=comp_name,
                    status=registry_hit.status,
                    new_shape=new_flat,
                    sibling_shape=sib_flat,
                    tt_reuse_target=registry_hit.tt_path,
                    notes=(
                        f"reuse_registry: {registry_hit.concept} -> "
                        f"{registry_hit.tt_path}::{registry_hit.tt_class} "
                        f"({registry_hit.status}). {registry_hit.notes}"
                    ),
                )
            )
            continue
        tt_target = _sibling_tt_file(backend, repo_root, hint, extra_backends=extra_backends)
        if tt_target is None:
            continue
        status = _diff_status(new_flat, sib_flat)
        if new_model_type and sibling_cfg.get("model_type") and new_model_type != sibling_cfg.get("model_type"):
            if comp_name in {"attention", "encoder_stack"}:
                status = NEW
        out.append(
            Component(
                name=comp_name,
                kind=comp_name,
                status=status,
                new_shape=new_flat,
                sibling_shape=sib_flat,
                tt_reuse_target=tt_target if status != NEW else None,
                hf_reference=(
                    _hf_reference(new_model_type, f"modeling_{new_model_type}.py") if status == NEW else None
                ),
            )
        )
    return out


def _extract_components_from_module_tree(
    *,
    new_model_id: str,
    new_model_type: Optional[str],
    demo_dir=None,
) -> List[Component]:
    """Module-tree-based component decomposition (2026-05-23 audit
    defect 2). Loads the new model's HF `nn.Module` via AutoModel,
    walks `named_modules()`, clusters by class name, and emits one
    `Component` per architectural cluster. Unlike the legacy sibling-
    config + filename-grep path, this:

      1. Reflects the ACTUAL structure of the new model, not the
         structure of some template the backend picked.
      2. Records the real `named_modules()` path on each component so
         autofill / op-synth / PCC capture don't need to consult the
         hand-maintained `COMPONENT_SUBMODULE_HINTS` dict.
      3. Works for brand-new architectures with no template demo.

    Failure modes:
      - AutoModel.from_pretrained fails (network, gated repo, oom)
        -> log a warning and return [] so the caller falls back to
        the legacy decomposition (back-compat).
      - Module tree produces no qualifying clusters (e.g. a model
        consisting only of a single nn.Linear) -> return a single
        catch-all root component so the LLM loop has something to
        target."""
    try:
        from .module_tree import discover_components_from_hf_id

        discovered = discover_components_from_hf_id(new_model_id, demo_dir=demo_dir)
    except Exception as exc:
        import sys as _sys

        bar = "=" * 72
        banner = (
            f"\n{bar}\n"
            f"  MODEL FAILED TO LOAD — cannot inspect {new_model_id!r}\n"
            f"{bar}\n"
            f"  reason: {type(exc).__name__}: {exc}\n"
            f"  This is NOT 'the model has no components' — the tool could not\n"
            f"  even construct the model on CPU. If this is a missing accelerator\n"
            f"  package (e.g. mamba-ssm / causal-conv1d), add a pure-CPU stand-in\n"
            f"  in scripts/tt_hw_planner/cpu_compat.py so the model opens on CPU.\n"
            f"{bar}"
        )
        for _stream in (_sys.stderr, _sys.stdout):
            print(banner, file=_stream)
        return []
    if not discovered:
        return [
            Component(
                name="model_root",
                kind="model_root",
                status=NEW,
                hf_reference=_hf_reference(new_model_type, f"modeling_{new_model_type}.py") if new_model_type else None,
                notes="auto-onboard: no module-tree clusters; treating model as one component",
            )
        ]
    comps: List[Component] = []
    for d in discovered:
        registry_hit = _reuse_lookup(new_model_type, d.class_name)
        if registry_hit is not None:
            status = registry_hit.status
            tt_reuse_target = registry_hit.tt_path
            hf_reference = None
            registry_note = (
                f"reuse_registry: {registry_hit.concept} -> "
                f"{registry_hit.tt_path}::{registry_hit.tt_class} "
                f"({registry_hit.status}). {registry_hit.notes}"
            )
            notes = (
                f"{registry_note} | module-tree: occ={d.occurrences} "
                f"leaves={d.leaf_op_count} sample_paths={d.sample_paths[:2]}"
            )
        else:
            status = d.status_hint
            tt_reuse_target = None
            hf_reference = (
                _hf_reference(new_model_type, f"modeling_{new_model_type}.py")
                if d.status_hint == NEW and new_model_type
                else None
            )
            notes = f"module-tree: occ={d.occurrences} leaves={d.leaf_op_count} " f"sample_paths={d.sample_paths[:2]}"
        comps.append(
            Component(
                name=d.name,
                kind=d.class_name or d.name,
                status=status,
                tt_reuse_target=tt_reuse_target,
                hf_reference=hf_reference,
                submodule_path=d.submodule_path or None,
                class_name=d.class_name,
                notes=notes,
            )
        )
    return comps


def build_bringup_plan(
    *,
    new_model_id: str,
    new_cfg: dict,
    backend: FamilyBackend,
    repo_root: Path,
    force_adapt_all: bool = False,
) -> BringUpPlan:
    """Build the per-component bring-up plan.

    ``force_adapt_all=True`` is used by the escalation hook
    (``_maybe_escalate_pcc_fail`` -> Path 1 re-entry) when the
    global PCC gate has already FAILED on an ALREADY-SUPPORTED model.
    In that case every component that the registry tags as REUSE is
    actually broken (the global failure proves the mappings are
    wrong) — promote them all to NEW so the per-component iterate
    loop has stubs + PCC tests to work on. Without this, an
    already-supported model that produces garbage output has nothing
    for Path 1 to iterate on.

    (The name ``force_adapt_all`` is historical — when this kwarg was
    introduced 2026-05-31 the demotion target was ADAPT. ADAPT was
    removed the same day; the demotion target is now NEW. Kwarg name
    kept for callsite stability.)
    """
    new_model_type = (new_cfg or {}).get("model_type")
    sibling_cfg: Dict[str, Any] = {}
    sibling_model_type: Optional[str] = None
    if backend.canonical_hf_id:
        try:
            sib_probe = probe_model(backend.canonical_hf_id)
            sibling_cfg = sib_probe.raw_config or {}
            sibling_model_type = (sibling_cfg or {}).get("model_type")
        except Exception:
            sibling_cfg = {}

    ranked: List[Tuple[FamilyBackend, int, str]] = []
    try:
        from .family_backends import rank_backends

        ranked = rank_backends(category=backend.category, model_type=new_model_type)
    except Exception:
        ranked = []
    extra_backends = [b for b, _, _ in ranked if b.name != backend.name]

    use_module_tree = bool(getattr(backend, "use_module_tree", False))
    if use_module_tree:
        comps = _extract_components_from_module_tree(
            new_model_id=new_model_id,
            new_model_type=new_model_type,
            demo_dir=repo_root / backend.demo_path,
        )
    elif backend.category in {"CNN", "Image", "Video"}:
        comps = _extract_components_vision(
            new_cfg=new_cfg or {},
            sibling_cfg=sibling_cfg,
            backend=backend,
            repo_root=repo_root,
            new_model_type=new_model_type,
            extra_backends=extra_backends,
        )
    else:
        comps = _extract_components_generic(
            new_cfg=new_cfg or {},
            sibling_cfg=sibling_cfg,
            backend=backend,
            repo_root=repo_root,
            new_model_type=new_model_type,
            extra_backends=extra_backends,
        )

    if not use_module_tree:
        try:
            supplemental = _extract_components_from_module_tree(
                new_model_id=new_model_id,
                new_model_type=new_model_type,
                demo_dir=repo_root / backend.demo_path,
            )
            primary_names = {c.name for c in comps}
            primary_classes = {c.class_name for c in comps if c.class_name}
            added: List[str] = []
            for s in supplemental:
                if s.name in primary_names:
                    continue
                if s.class_name and s.class_name in primary_classes:
                    continue
                s.notes = (
                    f"[supplemental module-tree pass] {s.notes} "
                    f"(primary extractor's template did not cover this "
                    f"class — falling back to module-tree discovery + "
                    f"op_classifier classification)."
                ).strip()
                comps.append(s)
                added.append(s.name)
            if added:
                try:
                    from loguru import logger

                    logger.info(
                        f"[bringup_plan] supplemental module-tree pass "
                        f"added {len(added)} component(s) not covered by "
                        f"the primary extractor: {added}"
                    )
                except Exception:
                    print(f"  [bringup_plan] supplemental module-tree " f"added {len(added)} component(s): {added}")
        except Exception as exc:
            try:
                from loguru import logger

                logger.warning(f"[bringup_plan] supplemental module-tree pass " f"failed: {type(exc).__name__}: {exc}")
            except Exception:
                pass

    if force_adapt_all:
        # Demote to ADAPT (2026-06-01). ADAPT semantics: the canonical
        # tt-module EXISTS and is wrapped as the per-component stub's
        # starting point. Iter 0 tests the wrapper as-is and graduates
        # if PCC ≥ 0.99; only on PCC < 0.99 does the LLM enter, and
        # then ONLY to refine config/wrapping — never to rewrite the
        # canonical class. This preserves the tt-module's correctness
        # for sibling models (Llama, Qwen3) while letting Qwen2-specific
        # quirks get adapted via small edits.
        promoted = 0
        for _c in comps:
            if _c.status == REUSE:
                _c.status = ADAPT
                _c.notes = (
                    f"[force_adapt_all] demoted REUSE -> ADAPT "
                    f"because global PCC gate failed on this "
                    f"already-supported model. Iter 0 will run "
                    f"canonical-wrapper PCC test; LLM refines only "
                    f"if PCC < 0.99. {_c.notes}"
                ).strip()
                promoted += 1
        if promoted:
            try:
                from loguru import logger

                logger.info(
                    f"[bringup_plan] force_adapt_all promoted "
                    f"{promoted} REUSE -> ADAPT component(s); per-component "
                    f"PCC iterate will run on canonical wrappers, "
                    f"LLM refines only on PCC < 0.99"
                )
            except Exception:
                print(f"  [bringup_plan] force_adapt_all promoted " f"{promoted} REUSE -> ADAPT component(s)")

    kernel_findings_serialized: List[Dict[str, Any]] = []
    try:
        from .kernel_constraints import collect_actionable_findings, evaluate_kernels

        _report = evaluate_kernels(new_cfg or {})
        kernel_findings_serialized = [f.to_dict() for f in collect_actionable_findings(_report)]
    except Exception:
        kernel_findings_serialized = []

    plan = BringUpPlan(
        new_model_id=new_model_id,
        new_model_type=new_model_type,
        sibling_hf_id=backend.canonical_hf_id,
        sibling_model_type=sibling_model_type,
        backend_name=backend.name,
        backend_demo_path=backend.demo_path,
        components=comps,
        common_reuse=[(label, path) for label, path, _ in _COMMON_REUSE_TABLE],
        kernel_findings=kernel_findings_serialized,
    )

    if not sibling_cfg:
        plan.notes.append(
            "Sibling config could not be fetched; classification falls back to NEW "
            "for components without a clear file match. Set HF_TOKEN or pre-download "
            f"`{backend.canonical_hf_id}` and re-run for a sharper diff."
        )
    if new_model_type and sibling_model_type and new_model_type != sibling_model_type:
        plan.notes.append(
            f"new model_type=`{new_model_type}` differs from sibling model_type=`{sibling_model_type}` — "
            "expect attention + encoder stacks to be NEW even if other shapes line up."
        )

    plan.ranked_siblings = [(b.name, s, r) for b, s, r in ranked]
    if len(ranked) > 1:
        alts = "; ".join(f"{b.name} (score {s}: {r})" for b, s, r in ranked)
        plan.notes.append(
            "Top sibling candidates (per-component reuse targets are pulled from whichever "
            f"sibling provides them, not only the first): {alts}"
        )

    return plan


def render_markdown(plan: BringUpPlan) -> str:
    counts = plan.counts
    parts: List[str] = []
    parts.append(f"# Bring-up plan: `{plan.new_model_id}`")
    parts.append("")
    parts.append(
        f"Backend template: **{plan.backend_name}** at `{plan.backend_demo_path}` "
        f"(canonical HF id: `{plan.sibling_hf_id}`)."
    )
    if plan.new_model_type or plan.sibling_model_type:
        parts.append(
            f"New `model_type` = `{plan.new_model_type}`; " f"sibling `model_type` = `{plan.sibling_model_type}`."
        )
    parts.append("")
    parts.append(f"**Summary:** {counts.get(REUSE, 0)} REUSE · " f"{counts.get(NEW, 0)} NEW component(s).")
    parts.append("")
    if plan.notes:
        parts.append("> **Notes:**")
        for n in plan.notes:
            parts.append(f"> - {n}")
        parts.append("")

    if plan.ranked_siblings:
        parts.append("## Sibling candidates (ranked)")
        parts.append("")
        parts.append(
            "Top backends by match score — components pull their reuse target from whichever "
            "of these provides it, not only rank 1."
        )
        parts.append("")
        parts.append("| Rank | Backend | Score | Match reason |")
        parts.append("|---|---|---|---|")
        for i, (name, score, reason) in enumerate(plan.ranked_siblings, 1):
            selected = " (selected)" if name == plan.backend_name else ""
            parts.append(f"| {i} | `{name}`{selected} | {score} | {reason} |")
        parts.append("")

    parts.append("## Components")
    parts.append("")
    parts.append("| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |")
    parts.append("|---|---|---|---|")
    for c in plan.components:
        target = c.tt_reuse_target or "—"
        href = c.hf_reference or "—"
        parts.append(f"| **{c.status}** | `{c.name}` | `{target}` | `{href}` |")
    parts.append("")

    parts.append("## Shared modules (always reusable, no copy needed)")
    parts.append("")
    parts.append("| Purpose | tt-metal path |")
    parts.append("|---|---|")
    for label, path in plan.common_reuse:
        parts.append(f"| {label} | `{path}` |")
    parts.append("")

    parts.append("## Action by status")
    parts.append("")
    parts.append(
        "- **REUSE**: import / call the sibling's tt-module unchanged. Weight names match. "
        "The global PCC gate enforces this — if it fails, `force_adapt_all` demotes the "
        "REUSE component to NEW and the brain iterates per-component."
    )
    parts.append(
        "- **NEW**: write/adapt the TTNN port. A stub file is generated under `_stubs/` "
        "(torch fallback by default), then progressively rewritten to native ttnn through "
        "per-component PCC iteration. If a sibling tt-file with the same role exists, the "
        "agent reuses its layout and updates shape constants (hidden_size, num_heads, "
        "intermediate_size, eps); otherwise it writes from scratch against the HF reference."
    )
    parts.append("")

    parts.append("## Per-component shape diff")
    parts.append("")
    for c in plan.components:
        parts.append(f"### `{c.name}` — {c.status}")
        if c.notes:
            parts.append(f"_{c.notes}_")
        parts.append("")
        parts.append("| field | new model | sibling |")
        parts.append("|---|---|---|")
        keys = sorted(set(c.new_shape) | set(c.sibling_shape))
        for k in keys:
            parts.append(f"| {k} | {c.new_shape.get(k, '—')} | {c.sibling_shape.get(k, '—')} |")
        parts.append("")

    parts.append("## Bring-up checklist")
    parts.append("")
    parts.append(
        "1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. "
        "The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`."
    )
    parts.append(
        "2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port "
        "driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants."
    )
    parts.append(
        "4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end."
    )
    parts.append("")
    return "\n".join(parts)


def render_json(plan: BringUpPlan) -> str:
    payload = {
        "new_model_id": plan.new_model_id,
        "new_model_type": plan.new_model_type,
        "sibling_hf_id": plan.sibling_hf_id,
        "sibling_model_type": plan.sibling_model_type,
        "backend": {"name": plan.backend_name, "demo_path": plan.backend_demo_path},
        "counts": plan.counts,
        "common_reuse": [{"purpose": p, "path": pth} for p, pth in plan.common_reuse],
        "components": [asdict(c) for c in plan.components],
        "notes": plan.notes,
        "ranked_siblings": [
            {"rank": i, "backend": n, "score": s, "reason": r} for i, (n, s, r) in enumerate(plan.ranked_siblings, 1)
        ],
    }
    return json.dumps(payload, indent=2)


def _stub_filename(component_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", component_name).strip("_").lower()
    return f"{safe or 'component'}.py"


def render_stub(plan: BringUpPlan, component: Component) -> str:
    href = component.hf_reference or "(no canonical HF reference found)"
    shape_snippet = ", ".join(f"{k}={v}" for k, v in component.new_shape.items()) or "(no shape fields)"
    return (
        f'"""BRING-UP GAP: `{component.name}` for {plan.new_model_id}.\n'
        f"\n"
        f"This component has no architectural analog in the sibling template\n"
        f"`{plan.backend_name}` ({plan.backend_demo_path}).\n"
        f"\n"
        f"Reference HF source: {href}\n"
        f"New model shape:    {shape_snippet}\n"
        f"\n"
        f"Steps to land this:\n"
        f"  1. Read the HF reference impl above.\n"
        f"  2. Implement the module in TTNN against the shape in `new_shape` (see BRING_UP_PLAN.md).\n"
        f"  3. Add a per-block PCC test under `tests/pcc/`.\n"
        f"  4. Wire it into the scaffolded demo at `demo/`.\n"
        f"  5. Delete this stub.\n"
        f'"""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"\n"
        f"def {component.name}(*args, **kwargs):\n"
        f"    raise NotImplementedError(\n"
        f'        "BRING-UP GAP: implement `{component.name}` for {plan.new_model_id}. "\n'
        f'        "See BRING_UP_PLAN.md and the HF reference at {href}."\n'
        f"    )\n"
    )


def collect_bringup_plan_files(
    *,
    plan: BringUpPlan,
    new_demo_dir_rel: Path,
) -> List[Tuple[Path, bytes, str]]:
    out: List[Tuple[Path, bytes, str]] = []
    out.append(
        (
            new_demo_dir_rel / "BRING_UP_PLAN.md",
            render_markdown(plan).encode("utf-8"),
            "bring-up plan (human)",
        )
    )
    out.append(
        (
            new_demo_dir_rel / "bringup_status.json",
            render_json(plan).encode("utf-8"),
            "bring-up plan (machine)",
        )
    )
    if plan.kernel_findings:
        out.append(
            (
                new_demo_dir_rel / "kernel_findings.json",
                json.dumps({"findings": plan.kernel_findings}, indent=2).encode("utf-8"),
                "kernel-constraint findings (WARN+BLOCKER)",
            )
        )
    for c in plan.components:
        if c.status != NEW:
            continue
        rel = new_demo_dir_rel / "_stubs" / _stub_filename(c.name)
        out.append((rel, render_stub(plan, c).encode("utf-8"), f"NEW-stub for `{c.name}`"))
    return out
