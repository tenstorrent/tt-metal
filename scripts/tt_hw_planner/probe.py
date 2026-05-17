# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Static probe of a HuggingFace model.

Pulls the minimal metadata needed for memory planning:
    - pipeline category (LLM / VLM / STT / TTS / Embed / Image / Video / CNN)
    - weight file footprint (safetensors + legacy)
    - dominant on-disk dtype + total parameter count
    - transformer config (for kv-cache math)
    - architecture family detected from config (dense / mla / sw / ssm / moe)

NO model weights are downloaded — only metadata + config.json (a few KB).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .architecture import (
    ArchitectureSpec,
    MemoryModel,
    build_arch_spec,
    detect_architecture,
    select_model,
)


# HuggingFace pipeline_tag -> our category bucket.
PIPELINE_CATEGORY = {
    "text-generation": "LLM",
    "text2text-generation": "LLM",
    "fill-mask": "LLM",
    "conversational": "LLM",
    "image-text-to-text": "VLM",
    "visual-question-answering": "VLM",
    "image-to-text": "VLM",
    "text-to-image": "Image",
    "image-to-image": "Image",
    "text-to-video": "Video",
    "image-to-video": "Video",
    "video-to-video": "Video",
    "automatic-speech-recognition": "STT",
    "audio-to-audio": "STT",
    "text-to-speech": "TTS",
    "text-to-audio": "TTS",
    "feature-extraction": "Embed",
    "sentence-similarity": "Embed",
    "image-classification": "CNN",
    "object-detection": "CNN",
    "image-segmentation": "CNN",
    "depth-estimation": "CNN",
    "image-feature-extraction": "CNN",
    "zero-shot-image-classification": "CNN",
}


# Categories for which we run the transformer config + memory model path.
TRANSFORMER_CATEGORIES = {"LLM", "VLM", "STT", "Embed"}


@dataclass
class ModelProbe:
    """The complete static probe of a HuggingFace model."""

    model_id: str
    category: str
    pipeline_tag: Optional[str]
    library: Optional[str]

    # Weight footprint
    weight_bytes_total: int
    weight_bytes_safetensors: int
    weight_bytes_legacy: int
    saved_dtype: str
    saved_dtype_pretty: str
    total_params: Optional[int]
    bytes_per_param_on_disk: Optional[float]

    # Transformer architecture (None for non-transformer categories)
    arch_spec: Optional[ArchitectureSpec] = None
    arch_family: Optional[str] = None
    memory_model: Optional[MemoryModel] = None

    # Loaded vs missing vs broken — drives confidence
    #   None    - we never tried (non-transformer)
    #   True    - config loaded AND transformer fields found
    #   False   - config loaded BUT fields not at standard paths
    #   "failed"- failed to load at all
    config_status: object = None

    flags: List[str] = field(default_factory=list)
    raw_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _classify_category(pipeline_tag: Optional[str], tags: List[str], library: Optional[str]) -> str:
    if pipeline_tag and pipeline_tag in PIPELINE_CATEGORY:
        return PIPELINE_CATEGORY[pipeline_tag]

    tag_str = " ".join(tags or []).lower()
    lib = (library or "").lower()

    if "diffusers" in lib or "diffusion" in tag_str or "flux" in tag_str:
        return "Image"
    if "sentence-transformers" in lib or "embedding" in tag_str:
        return "Embed"
    if "whisper" in tag_str or "speech-recognition" in tag_str:
        return "STT"
    if "text-to-speech" in tag_str or "tts" in tag_str:
        return "TTS"
    if any(t in tag_str for t in ["resnet", "vit", "convnext", "mobilenet", "efficientnet"]):
        return "CNN"
    if "transformers" in lib:
        return "LLM"
    return "Unknown"


def _sum_weight_files(siblings) -> Tuple[int, int]:
    """Return (safetensors_bytes, legacy_bytes)."""
    sf, legacy = 0, 0
    legacy_exts = (".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".nemo")
    for s in siblings or []:
        size = getattr(s, "size", None) or 0
        name = s.rfilename
        if name.endswith(".safetensors"):
            sf += size
        elif name.endswith(legacy_exts):
            legacy += size
    return sf, legacy


# Safetensors per-element-byte mapping (HF reports element counts here, not bytes).
_DTYPE_ELEMENT_BYTES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8": 1,
    "F8_E8M0": 1,
    "F4": 0.5,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}
_DTYPE_PRETTY = {
    "F32": "fp32",
    "F16": "fp16",
    "BF16": "bf16",
    "F8_E4M3": "fp8",
    "F8_E5M2": "fp8",
    "F8": "fp8",
    "F8_E8M0": "f8_e8m0",
    "F4": "fp4",
}


def _dominant_dtype(parameters, weight_bytes) -> Tuple[str, str, Optional[int], Optional[float]]:
    """
    Return (canonical_dtype, pretty_label, total_params, bytes_per_param_on_disk).

    canonical_dtype is one of: bf16 / fp16 / fp32 / fp8 / fp4 / unknown.
    pretty_label adds (quantized, X.XX B/param on disk) etc.
    """
    if not parameters:
        return "bf16", "bf16 (assumed)", None, None

    total_params = sum(parameters.values())

    # Dominant dtype by *parameter count*, ignoring integer index/routing tables.
    weight_only = {dt: n for dt, n in parameters.items() if not dt.startswith("I") and dt != "BOOL"}
    if weight_only:
        dom = max(weight_only.items(), key=lambda kv: kv[1])[0]
    else:
        dom = max(parameters.items(), key=lambda kv: kv[1])[0]

    canonical = _DTYPE_PRETTY.get(dom, dom.lower())
    pretty = canonical

    bytes_per_param = None
    if total_params > 0 and weight_bytes > 0:
        bytes_per_param = weight_bytes / total_params
        if bytes_per_param < 1.5:
            pretty = f"{canonical} (quantized, {bytes_per_param:.2f} B/param on disk)"
        elif len(parameters) > 1:
            pretty = f"{canonical} (mixed)"
    elif len(parameters) > 1:
        pretty = f"{canonical} (mixed)"

    return canonical, pretty, total_params, bytes_per_param


def _maybe_fetch_config(model_id: str) -> Optional[dict]:
    """Try AutoConfig, fall back to raw config.json download."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return cfg.to_dict()
    except Exception:
        pass

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, "config.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def probe_model(model_id: str) -> ModelProbe:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("ERROR: huggingface_hub not installed. `pip install huggingface_hub`.")

    api = HfApi()
    try:
        info = api.model_info(model_id, files_metadata=True)
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "gated repo" in lower or "restricted" in lower:
            sys.exit(
                f"ERROR: '{model_id}' is a gated HuggingFace repo.\n"
                "  Run `huggingface-cli login` and accept the model's license on\n"
                f"  https://huggingface.co/{model_id} , then re-run this script."
            )
        if "not found" in lower or "repositorynotfounderror" in lower or "404" in lower:
            sys.exit(f"ERROR: '{model_id}' not found on HuggingFace. Check the model ID.")
        if "connection" in lower or "timed out" in lower:
            sys.exit(f"ERROR: Network problem reaching HuggingFace: {msg.splitlines()[0]}")
        raise

    sf_bytes, legacy_bytes = _sum_weight_files(info.siblings)
    weight_bytes = sf_bytes if sf_bytes > 0 else legacy_bytes

    if weight_bytes == 0:
        sys.exit(
            f"ERROR: '{model_id}' has no .safetensors or .bin weight files in its repo.\n"
            "  This script can't estimate memory without weight files. Possible causes:\n"
            "  - GGUF-only model — convert to HF format or use a llama.cpp-style tool.\n"
            "  - Adapter / LoRA repo — point at the base model instead.\n"
            "  - Repo doesn't host weights (template / docs only)."
        )

    parameters = info.safetensors.parameters if info.safetensors else None
    canonical_dtype, pretty_dtype, total_params, bytes_per_param = _dominant_dtype(parameters, weight_bytes)
    if total_params is None and weight_bytes > 0:
        # No safetensors metadata — assume bf16 storage to recover param count.
        total_params = weight_bytes // 2

    category = _classify_category(info.pipeline_tag, info.tags or [], info.library_name)

    probe = ModelProbe(
        model_id=model_id,
        category=category,
        pipeline_tag=info.pipeline_tag,
        library=info.library_name,
        weight_bytes_total=weight_bytes,
        weight_bytes_safetensors=sf_bytes,
        weight_bytes_legacy=legacy_bytes,
        saved_dtype=canonical_dtype,
        saved_dtype_pretty=pretty_dtype,
        total_params=total_params,
        bytes_per_param_on_disk=bytes_per_param,
    )

    if category in TRANSFORMER_CATEGORIES:
        cfg = _maybe_fetch_config(model_id)
        if cfg is None:
            probe.config_status = "failed"
        else:
            probe.raw_config = cfg

            # Walk nested configs to find the transformer fields.
            NESTED_KEYS = (
                "text_config",
                "llm_config",
                "language_config",
                "decoder_config",
                "text_model_config",
                "language_model_config",
            )
            candidates = [cfg] + [cfg.get(k) for k in NESTED_KEYS if isinstance(cfg.get(k), dict)]
            for c in candidates:
                if c.get("hidden_size") and c.get("num_hidden_layers"):
                    text_cfg = c
                    break
            else:
                text_cfg = cfg

            family = detect_architecture(text_cfg)
            arch_spec = build_arch_spec(text_cfg, family)
            probe.arch_spec = arch_spec
            probe.arch_family = family

            if arch_spec.hidden_size and arch_spec.num_layers:
                probe.config_status = True
                probe.memory_model = select_model(arch_spec, total_params, weight_bytes)

                # Surface architectural flags
                if family == "mla":
                    probe.flags.append("MLA (compressed KV cache) detected — DeepSeek family")
                if family == "moe":
                    probe.flags.append(
                        f"MoE detected ({arch_spec.num_experts} experts, " f"top-{arch_spec.experts_per_token})"
                    )
                if family == "ssm":
                    probe.flags.append("State-space model — no per-token KV cache")
                if arch_spec.sliding_window:
                    probe.flags.append(f"Sliding-window attention (window={arch_spec.sliding_window})")
            else:
                probe.config_status = False

    return probe
