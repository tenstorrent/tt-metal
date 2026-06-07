# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Composition-tree extractor for emit-e2e demo emission.

Given an HF model + the graduated-stub manifest (``bringup_status.json``),
walks the HF ``nn.Module`` graph to find which attribute path feeds each
graduated component.

Output is a ``CompositionTree`` dataclass consumed by every TaskTemplate
to wire ``hf_model.<path>`` into emitted source code.

Mechanical extraction — no LLM. If an attribute path can't be resolved
for some graduated stub, that stub is recorded in ``cpu_bridges`` (i.e.
the demo will need a CPU bridge instead of a TT call there).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..task_templates._base import CompositionTree


# ─── HF class name → TaskTemplate identifier mapping ─────────────────
# Detected from `config.architectures[0]` + AutoModel-class probing.
# The orchestrator passes the resolved task_class string straight through
# to ``lookup_template``; this map is here for diagnostic purposes only.

_HF_AUTO_CLASSES = (
    "AutoModelForCausalLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForSpeechSeq2Seq",
    "AutoModelForTextToWaveform",
    "AutoModelForVision2Seq",
    "AutoModelForImageClassification",
    "AutoModelForImageSegmentation",
    "AutoModelForCTC",
    "AutoModelForMaskGeneration",
    "AutoPipelineForText2Image",
)


def _normalize_component_name(name: str) -> str:
    """Bring-up tool component names are snake-cased (e.g.
    ``seamless_m4_t_speech_encoder``). HF submodule attribute names are
    camelCase-derived. Map between them by stripping common prefixes
    and lower-casing for fuzzy match."""
    n = name.lower().replace("_", "").replace("-", "")
    return n


def _hf_class_to_normalized(class_name: str) -> str:
    """e.g. SeamlessM4TSpeechEncoder -> seamlessm4tspeechencoder"""
    return class_name.lower().replace("_", "").replace("-", "")


def _walk_module_classes(hf_model: Any) -> List[Tuple[str, str]]:
    """Walk hf_model.named_modules() and collect (dotted_path, class_name).

    The returned list lets us match graduated-component class names to
    actual HF submodule paths.
    """
    found: List[Tuple[str, str]] = []
    if hasattr(hf_model, "named_modules"):
        for path, module in hf_model.named_modules():
            cls_name = type(module).__name__
            found.append((path, cls_name))
    return found


def _load_bringup_status(demo_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """Load ``bringup_status.json`` from a demo dir if present."""
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return None
    try:
        data = json.loads(status_path.read_text())
    except Exception:
        return None
    if isinstance(data, dict):
        return list(data.get("components") or [])
    return None


def _component_class_name(comp: Dict[str, Any]) -> Optional[str]:
    """Pull the original HF class name out of a bringup_status entry.

    The bring-up tool records this as either ``hf_class`` or in
    ``hf_module_path``; try multiple keys for robustness.
    """
    for key in ("hf_class", "class_name", "torch_class", "original_class"):
        v = comp.get(key)
        if isinstance(v, str) and v:
            return v
    # Fall back: snake-case the component name back to a probable class.
    name = comp.get("name", "")
    if isinstance(name, str) and name:
        return "".join(p.capitalize() for p in name.split("_"))
    return None


ROLE_AUDIO_ENCODER = "audio_encoder"
ROLE_TEXT_ENCODER = "text_encoder"
ROLE_TEXT_DECODER = "text_decoder"
ROLE_T2U_MODEL = "t2u_model"
ROLE_VOCODER = "vocoder"
ROLE_LM_HEAD = "lm_head"


def classify_roles(graduated_clean_names: List[str]) -> Dict[str, str]:
    """Map abstract roles -> the clean stub name that plays each role.

    Purely heuristic: looks at clean stub names from dynamic discovery
    and pattern-matches against substrings. NO model-family hardcoding —
    only generic substring rules ("speech" suggests audio, "decoder"
    suggests autoregressive head, etc.) that apply to any HF model.

    A role is omitted from the result if no stub matches it. Templates
    handle missing roles defensively (use ``ctx.composition_tree.roles.get(...)``).

    Each role gets at most ONE assignment (the first match wins, in the
    order the discovery produced them).
    """
    roles: Dict[str, str] = {}

    for clean in graduated_clean_names:
        low = clean.lower()
        # Match each role exactly once.
        if ROLE_AUDIO_ENCODER not in roles:
            if "speech_encoder" in low or "audio_encoder" in low or "acoustic_encoder" in low:
                roles[ROLE_AUDIO_ENCODER] = clean
                continue

        if ROLE_VOCODER not in roles:
            if any(k in low for k in ("code_hifi_gan", "hifi_gan", "vocoder", "wave_glow", "waveglow")):
                # Prefer code_hifi_gan over hifi_gan if both present (more specific)
                if "code_hifi_gan" in low or roles.get(ROLE_VOCODER, "").find("code_hifi") < 0:
                    roles[ROLE_VOCODER] = clean
                continue

        if ROLE_T2U_MODEL not in roles:
            if "text_to_unit" in low or low.startswith("t2u"):
                # Prefer "text_to_unit_for_conditional_generation" over plain
                # "text_to_unit_model" — it's the task-facing class.
                if "for_conditional_generation" in low or ROLE_T2U_MODEL not in roles:
                    roles[ROLE_T2U_MODEL] = clean
                continue

        if ROLE_TEXT_DECODER not in roles:
            # "decoder" by itself (not "decoder_layer", "encoder_decoder")
            # AND not the t2u decoder.
            if low == "decoder" or low.endswith("_decoder"):
                if "encoder" not in low and "_layer" not in low and "_attention" not in low:
                    if "text_to_unit" not in low:
                        roles[ROLE_TEXT_DECODER] = clean
                        continue

        if ROLE_TEXT_ENCODER not in roles:
            # "encoder" by itself, but NOT a speech/audio encoder
            if low == "encoder" or low.endswith("_encoder"):
                if not any(
                    k in low
                    for k in (
                        "speech",
                        "audio",
                        "acoustic",
                        "conformer",
                        "_layer",
                        "_attention",
                    )
                ):
                    if "text_to_unit" not in low:
                        roles[ROLE_TEXT_ENCODER] = clean
                        continue

    return roles


def extract(
    *,
    model_id: str,
    task_class: str,
    hf_model: Any,
    demo_dir: Path,
) -> CompositionTree:
    """Build a CompositionTree by matching graduated components to HF submodules.

    For each entry in ``bringup_status.json``:
      * If the entry has an explicit ``hf_module_path`` field (newer
        graduations), use it directly.
      * Otherwise, walk hf_model.named_modules() and find the first
        path whose class name matches the component's recorded HF class.
      * If no match, record the component name in ``cpu_bridges``.
    """
    components = _load_bringup_status(demo_dir) or []

    module_classes = _walk_module_classes(hf_model)
    class_to_path: Dict[str, str] = {}
    for path, cls in module_classes:
        # Keep the first (shortest) path that matches each class
        if cls not in class_to_path or len(path) < len(class_to_path[cls]):
            class_to_path[cls] = path

    stub_attributes: Dict[str, str] = {}
    cpu_bridges: List[str] = []

    for comp in components:
        comp_name = comp.get("name", "")
        if not isinstance(comp_name, str) or not comp_name:
            continue

        # Preferred: explicit attribute path recorded by the bring-up tool.
        explicit = comp.get("hf_module_path") or comp.get("hf_attribute_path")
        if isinstance(explicit, str) and explicit:
            stub_attributes[comp_name] = explicit
            continue

        # Fallback: match by HF class name.
        cls_name = _component_class_name(comp)
        if cls_name and cls_name in class_to_path:
            stub_attributes[comp_name] = class_to_path[cls_name]
            continue

        # No match → component lives in a CPU bridge.
        cpu_bridges.append(comp_name)

    return CompositionTree(
        model_id=model_id,
        task_class=task_class,
        stub_attributes=stub_attributes,
        cpu_bridges=cpu_bridges,
        multi_task_heads=[],  # populated by orchestrator from multi_task_tasks_for
    )


def detect_task_class(hf_config: Dict[str, Any], probe_pipeline_tag: str = "") -> Optional[str]:
    """Resolve the HF AutoModel class for a model.

    Reads ``config.architectures[0]`` (a concrete subclass like
    ``SeamlessM4TForSpeechToText``) and maps to its AutoModelFor*
    parent via the transformers task-class registry.

    Returns the AutoModel class name string, or None if undetectable.
    """
    arches = hf_config.get("architectures") or []
    if not arches:
        return None

    concrete = str(arches[0])

    # Mapping rules — concrete suffix → AutoModel parent.
    # These are derived from the transformers MODEL_FOR_*_MAPPING tables
    # and kept short on purpose. Extend as new task types arrive.
    if concrete.endswith("ForSpeechToText") or concrete.endswith("ForSpeechSeq2Seq"):
        return "AutoModelForSpeechSeq2Seq"
    if concrete.endswith("ForTextToText") or concrete.endswith("ForConditionalGeneration"):
        # ConditionalGeneration covers seq2seq T2T (BART/NLLB/M2M100/etc.)
        # but ALSO covers Whisper. Prefer SpeechSeq2Seq when pipeline says ASR.
        if probe_pipeline_tag in ("automatic-speech-recognition", "speech-translation"):
            return "AutoModelForSpeechSeq2Seq"
        return "AutoModelForSeq2SeqLM"
    if concrete.endswith("ForTextToSpeech") or concrete.endswith("ForTextToWaveform"):
        return "AutoModelForTextToWaveform"
    if concrete.endswith("ForCausalLM"):
        return "AutoModelForCausalLM"
    if concrete.endswith("ForCTC"):
        return "AutoModelForCTC"
    if concrete.endswith("ForVision2Seq") or concrete.endswith("ForVisionText2Text"):
        return "AutoModelForVision2Seq"
    if concrete.endswith("ForImageClassification"):
        return "AutoModelForImageClassification"
    if concrete.endswith("ForSemanticSegmentation") or concrete.endswith("ForImageSegmentation"):
        return "AutoModelForImageSegmentation"
    if concrete.endswith("ForMaskGeneration") or "SAM" in concrete.upper():
        return "AutoModelForMaskGeneration"
    if concrete.endswith("Pipeline") and "Diffusion" in concrete:
        return "AutoPipelineForText2Image"

    return None


__all__ = [
    "extract",
    "detect_task_class",
]
