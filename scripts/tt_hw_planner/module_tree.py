"""
Module-tree component discovery (2026-05-23 audit defect 2).

Replaces the filename-grep component decomposition in `bringup_plan.py`
for backends that opt in (`FamilyBackend.use_module_tree=True`) or for
LLM-drafted backends produced by `auto-onboard`.

The old decomposition path looks at:
  1. hand-curated nest keys (`vision_config`, `prompt_encoder_config`,
     `mask_decoder_config`, ...) in the HF config dict
  2. hand-curated primitive names (`patch_embed`, `self_attention`,
     `mlp`, ...), gated on whether a sibling TT template demo has a
     matching filename
This module-tree path looks at the ACTUAL HF model that will run on the
device:
  1. Load the HF `nn.Module` once via AutoModel.from_pretrained.
  2. Walk `named_modules()` and cluster by `type(m).__name__`.
  3. For each cluster of significant size (configurable threshold),
     emit ONE `DiscoveredComponent` with:
       - canonical name derived from the class name
       - the actual `named_modules()` path (the first occurrence)
       - leaf-op count (for complexity-aware LLM model selection)
       - parent path (so multi-instance blocks like `layer.0`, `layer.1`
         collapse to a single "layer" component, just like the
         hand-curated path did)

The output is a list of `DiscoveredComponent` records that
`bringup_plan.py` can transform into its existing `bringup_status.json`
entries with no schema change downstream. Autofill, op-synth,
auto-iterate, and PCC test generation all read the existing
`bringup_status.json` schema, so they're unaffected by where the
component list came from.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class DiscoveredComponent:
    """One component identified by walking the HF module tree.

    Fields mirror what `bringup_plan.Component` ultimately needs; this
    struct exists as a separate type so callers know whether a component
    came from the legacy filename-grep path (`bringup_plan.Component`)
    or the module-tree path (here)."""

    name: str
    submodule_path: str
    class_name: str
    occurrences: int
    leaf_op_count: int
    sample_paths: List[str] = field(default_factory=list)
    status_hint: str = "NEW"


_CONTAINER_CLASS_NAMES: Set[str] = {
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
}


_LEAF_CLASS_NAMES: Set[str] = {
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Embedding",
    "Dropout",
    "Identity",
    "GELU",
    "SiLU",
    "ReLU",
    "Softmax",
    "Tanh",
    "Sigmoid",
    "GELUActivation",
    "QuickGELUActivation",
    "PytorchGELUTanh",
    "SiLUActivation",
    "MishActivation",
    "ClippedGELUActivation",
    "GELUTanh",
    "NewGELUActivation",
    "ACT2FN",
    "ClassInstantier",
}


_HIGH_LEVEL_SUFFIXES: Tuple[str, ...] = (
    "Attention",
    "Block",
    "Layer",
    "Encoder",
    "Decoder",
    "Embeddings",
    "Embedding",
    "Head",
    "PatchEmbed",
    "Stage",
    "MLP",
    "FeedForward",
    "Pooler",
    "Predictor",
    "Mixer",
    "Tokenizer",
    "Backbone",
    "Norm",
    "Transformer",
    "Neck",
    "FPN",
    "Aggregator",
    "Refiner",
    "Projector",
    "Resampler",
    "Adapter",
)


# ADAPT was removed 2026-05-31. The trichotomy REUSE / ADAPT / NEW
# collapses to REUSE / NEW: REUSE is the registry's static claim
# ("works as-is"), NEW means "needs per-component iteration". ADAPT's
# "light parameter tweak" semantics couldn't be enforced at runtime
# (it had the same fragility as REUSE — a static claim contradicted
# by runtime PCC) and had no distinct iteration machinery anyway.
# When REUSE fails PCC, force_adapt_all demotes it straight to NEW.


_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_CAMEL_SEGMENT_RE = re.compile(r"[A-Z][a-z0-9]*")


def _camel_segments(name: str) -> List[str]:
    """Split a CamelCase name into its leading-uppercase segments.
    E.g.  "Sam2HieraBlock"     -> ["Sam2", "Hiera", "Block"]
          "SegformerLayer"     -> ["Segformer", "Layer"]
          "SamMaskDecoderHead" -> ["Sam", "Mask", "Decoder", "Head"]"""
    return _CAMEL_SEGMENT_RE.findall(name)


def _common_prefix_segments(names: List[str]) -> List[str]:
    """Longest common leading sequence of camel segments across all
    class names. Used to strip the per-model prefix (e.g. "Sam2", or
    "Segformer", or "Llama") that would otherwise appear in every
    component name. Only strips when at least 2 distinct classes share
    the prefix -- one-class models keep their full name."""
    if len(names) < 2:
        return []
    segs = [_camel_segments(n) for n in names]
    if not segs or any(not s for s in segs):
        return []
    common: List[str] = []
    for i in range(min(len(s) for s in segs)):
        token = segs[0][i]
        if all(s[i] == token for s in segs):
            common.append(token)
        else:
            break

    while common and any(len(s) <= len(common) for s in segs):
        common.pop()
    return common


def _canonical_component_name(class_name: str, prefix_segments: List[str]) -> str:
    """Convert a HF class name into a snake_case bring-up component
    name, after stripping any per-model prefix.

    Examples (with prefix_segments=["Sam2"]):
      Sam2HieraBlock        -> hiera_block
      Sam2HieraSelfAttention-> hiera_self_attention
      Sam2VisionEncoder     -> vision_encoder
      Sam2MaskDecoderHead   -> mask_decoder_head
    """
    segs = _camel_segments(class_name)
    if prefix_segments and segs[: len(prefix_segments)] == prefix_segments:
        segs = segs[len(prefix_segments) :]
    if not segs:
        segs = _camel_segments(class_name)
    snake = "_".join(s.lower() for s in segs)
    return snake or class_name.lower()


def safe_identifier(name: str) -> str:
    """Sanitize a free-form name into a python-identifier-safe slug.

    Replaces every non-alphanumeric run with a single underscore, strips
    leading/trailing underscores, lowercases the result, and falls back
    to ``"component"`` when the input collapses to empty.

    Canonical implementation used by activation_diff, bringup_loop,
    capture_inputs, and op_emitter — each used to keep its own copy
    of the same two-line regex.
    """
    import re

    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    return safe or "component"


def resolve_dotted(obj: Any, dotted: str) -> Any:
    """Resolve a dotted/bracketed attr path against ``obj``, supporting
    numeric tokens as ``__getitem__`` indices.

    Matches HuggingFace's ``named_modules()`` convention (e.g.
    ``model.layers.0.self_attn``) and the op-planner's
    ``pre_bound[].name`` form (``pre_bound.0.weight``). Numeric tokens
    are indexed; non-numeric tokens go through ``getattr``.

    Raises whatever ``getattr`` / ``__getitem__`` raise on missing
    path elements; callers should catch ``AttributeError``,
    ``IndexError``, ``KeyError``, ``TypeError`` as appropriate.

    Canonical implementation used by op-synth (llm_synth, bringup_loop),
    by activation diffing (activation_diff), and by the input-capture
    harness (capture_inputs). All four previously kept their own
    copies of this function.
    """
    cur = obj
    for tok in dotted.replace("[", ".").replace("]", "").split("."):
        if tok == "":
            continue
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            cur = getattr(cur, tok)
    return cur


def _is_container(mod: Any) -> bool:
    return type(mod).__name__ in _CONTAINER_CLASS_NAMES


def _is_leaf(mod: Any) -> bool:
    """A leaf is a module with no child modules of its own."""
    try:
        return not any(True for _ in mod.children())
    except Exception:
        return True


def _count_leaves(mod: Any) -> int:
    """Recursive leaf count, ignoring container nodes."""
    if _is_leaf(mod):
        return 1
    n = 0
    try:
        for child in mod.children():
            n += _count_leaves(child)
    except Exception:
        return 1
    return n


def _try_rescue_via_op_coverage(class_name: str, info: Dict[str, Any]) -> Optional[str]:
    mod = info.get("first_module")
    if mod is None:
        return None
    try:
        from .op_classifier import classify_ops_in_component

        ops = classify_ops_in_component(mod)
    except Exception:
        return None
    if not ops:
        return None
    # ADAPT removed 2026-05-31. A class with majority-reusable ops used
    # to be tagged ADAPT (light tweak) — now it's just NEW (full
    # per-component iterate). The reusable-ops signal still matters for
    # the LLM prompt (op-REUSE/op-ADAPT/op-NEW is a separate concept
    # from component status) but doesn't influence component status.
    return "NEW"


def _log_dropped_class(
    class_name: str,
    info: Dict[str, Any],
    *,
    reason_leaf: bool,
    reason_occurrence: bool,
) -> None:
    reasons = []
    if reason_leaf:
        reasons.append(f"leaf_count={info['leaf_count_total']} < threshold")
    if reason_occurrence:
        reasons.append(f"occurrences={info['occurrences']} < threshold")
    try:
        from loguru import logger

        logger.warning(
            f"[module_tree] dropping non-whitelist class {class_name!r} "
            f"({'; '.join(reasons)}; op-coverage rescue also declined). "
            f"This module will run on CPU unless added to "
            f"_HIGH_LEVEL_SUFFIXES or onboarded explicitly. "
            f"Sample path: {info['sample_paths'][:1]}"
        )
    except Exception:
        pass


def _classify_status_hint(class_name: str) -> str:
    """Best-guess status (NEW / REUSE) for a module class.

    Always returns NEW. REUSE is reserved for the bring-up plan layer
    when it looks up the reuse_registry; class-name alone is not
    enough evidence to claim REUSE. ADAPT was removed 2026-05-31 —
    the trichotomy collapsed to a dichotomy (REUSE / NEW)."""
    return "NEW"


def _looks_high_level(class_name: str) -> bool:
    """True if a class name carries an architectural-component suffix.

    Singletons of these classes get their own component even when the
    cluster-size heuristic wouldn't promote them."""
    return any(class_name.endswith(s) for s in _HIGH_LEVEL_SUFFIXES)


def discover_components(
    model: Any,
    *,
    min_cluster_size: int = 1,
    min_leaf_count_per_component: int = 2,
    max_components: int = 32,
) -> List[DiscoveredComponent]:
    """Walk an HF `nn.Module` and emit a deduplicated component list.

    Args:
      model:                          a torch.nn.Module (or anything with
                                      `named_modules()`).
      min_cluster_size:               minimum number of instances of a
                                      class for it to qualify as its own
                                      component on cluster-size alone.
                                      Defaults to 1 (any class).
      min_leaf_count_per_component:   filter out trivial leaf-only
                                      submodules (e.g. a bare Dropout)
                                      from being promoted to a
                                      component. Defaults to 2.
      max_components:                 safety cap. Beyond this many
                                      components, the auto-iterate
                                      loop's attempt budget gets
                                      unwieldy. We sort by descending
                                      leaf_op_count and keep the top N.

    Returns components sorted by descending `leaf_op_count` -- the
    biggest / most complex blocks first, which matches the op-count
    tiebreaker used by `demo_wiring.select_maximal_antichain`."""
    if not hasattr(model, "named_modules"):
        return []

    cluster_by_class: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "first_path": None,
            "parent_path": None,
            "occurrences": 0,
            "leaf_count_total": 0,
            "sample_paths": [],
            "first_module": None,
        }
    )

    for path, mod in model.named_modules():
        if path == "":
            continue
        class_name = type(mod).__name__
        if class_name in _CONTAINER_CLASS_NAMES:
            continue
        if class_name in _LEAF_CLASS_NAMES:
            continue
        info = cluster_by_class[class_name]
        if info["first_path"] is None:
            info["first_path"] = path
            info["first_module"] = mod

            parent_parts = path.split(".")
            while parent_parts and parent_parts[-1].isdigit():
                parent_parts.pop()
            info["parent_path"] = ".".join(parent_parts) or path
        info["occurrences"] += 1
        if len(info["sample_paths"]) < 3:
            info["sample_paths"].append(path)
        info["leaf_count_total"] += _count_leaves(mod)

    rescued_hint_override: Dict[str, str] = {}

    qualifying_class_names: List[str] = []
    for class_name, info in cluster_by_class.items():
        is_high_level = _looks_high_level(class_name)
        fails_leaf_count = not is_high_level and info["leaf_count_total"] < min_leaf_count_per_component
        fails_occurrence = info["occurrences"] < min_cluster_size and not is_high_level
        if fails_leaf_count or fails_occurrence:
            rescued = _try_rescue_via_op_coverage(class_name, info)
            if rescued is None:
                _log_dropped_class(
                    class_name,
                    info,
                    reason_leaf=fails_leaf_count,
                    reason_occurrence=fails_occurrence,
                )
                continue
            rescued_hint_override[class_name] = rescued
        qualifying_class_names.append(class_name)
    prefix_segments = _common_prefix_segments(qualifying_class_names)

    discovered: List[DiscoveredComponent] = []
    for class_name in qualifying_class_names:
        info = cluster_by_class[class_name]
        hint = rescued_hint_override.get(class_name, _classify_status_hint(class_name))
        discovered.append(
            DiscoveredComponent(
                name=_canonical_component_name(class_name, prefix_segments),
                submodule_path=str(info["parent_path"]),
                class_name=class_name,
                occurrences=int(info["occurrences"]),
                leaf_op_count=int(info["leaf_count_total"]),
                sample_paths=list(info["sample_paths"]),
                status_hint=hint,
            )
        )

    discovered.sort(key=lambda c: (-c.leaf_op_count, c.name))

    if len(discovered) > max_components:
        discovered = discovered[:max_components]
    return discovered


def discover_components_from_hf_id(
    model_id: str,
    *,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
    load_weights: bool = False,
) -> List[DiscoveredComponent]:
    """Instantiate the model architecture from its HF config and walk
    the tree.

    2026-05-23 audit bug #3: previously used
    ``AutoModel.from_pretrained(model_id)`` which downloads + materializes
    full weights. For a 27B model that's 54 GB of disk + 30+ minutes
    of network time BEFORE the meta-plan LLM call even starts -- so
    the meta-plan's 120s timeout fired during the WEIGHT LOAD, not
    the LLM call.

    Component discovery only needs the ``nn.Module`` skeleton (class
    names + tree topology). We now default to ``AutoModel.from_config(...)``
    which builds the architecture WITHOUT downloading or loading
    weights. Set ``load_weights=True`` only if a downstream caller
    actually needs the parameters (none currently do).

    Caller is responsible for ``del model`` afterwards if memory is
    tight."""
    from transformers import AutoConfig, AutoModel

    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision is not None:
        kwargs["revision"] = revision

    if load_weights:
        model = AutoModel.from_pretrained(model_id, **kwargs)
    else:
        config = AutoConfig.from_pretrained(model_id, **kwargs)
        model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
    try:
        return discover_components(model)
    finally:
        del model
