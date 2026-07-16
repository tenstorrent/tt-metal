from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


_HF_ID_PART = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
_HF_ID_PATTERN = re.compile(rf"^{_HF_ID_PART}(/{_HF_ID_PART})?$")


def _is_local_model_dir(model_id: str) -> bool:
    return (
        isinstance(model_id, str) and os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "config.json"))
    )


def _validate_hf_id(model_id: str) -> str:
    if _is_local_model_dir(model_id):
        return model_id
    if not isinstance(model_id, str) or not _HF_ID_PATTERN.match(model_id):
        raise ValueError(f"invalid HuggingFace model id: {model_id!r}")
    return model_id


from .architecture import (
    ArchitectureSpec,
    MemoryModel,
    build_arch_spec,
    detect_architecture,
    select_model,
)


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
    "mask-generation": "CNN",
    "zero-shot-object-detection": "CNN",
    "keypoint-detection": "CNN",
    "image-to-3d": "CNN",
    "video-classification": "Video",
}


TRANSFORMER_CATEGORIES = {"LLM", "VLM", "STT", "Embed"}


VISION_ONLY_MODEL_TYPES = {
    "sam",
    "sam2",
    "sam2_video",
    "sam_hiera",
    "clip",
    "siglip",
    "vit",
    "dinov2",
    "swin",
    "yolos",
    "detr",
    "deformable_detr",
    "segformer",
    "maskformer",
    "mask2former",
    "upernet",
    "beit",
    "convnext",
    "mobilenet_v2",
    "mobilenet_v1",
    "resnet",
    "owlvit",
}


AUDIO_ONLY_MODEL_TYPES = {
    "whisper",
    "wav2vec2",
    "hubert",
    "wavlm",
    "sew",
    "unispeech",
    "audio_spectrogram_transformer",
    "clap",
}


def _category_from_model_type(model_type: str) -> Optional[str]:
    mt = (model_type or "").lower()
    if not mt:
        return None
    if mt in VISION_ONLY_MODEL_TYPES:
        return "CNN"
    if mt in AUDIO_ONLY_MODEL_TYPES:
        return "STT"
    if mt in {"llava", "blip-2", "blip2", "idefics", "paligemma", "qwen2_5_vl", "qwen2_vl"}:
        return "VLM"
    return None


@dataclass
class ModelProbe:
    model_id: str
    category: str
    pipeline_tag: Optional[str]
    library: Optional[str]

    weight_bytes_total: int
    weight_bytes_safetensors: int
    weight_bytes_legacy: int
    saved_dtype: str
    saved_dtype_pretty: str
    total_params: Optional[int]
    bytes_per_param_on_disk: Optional[float]

    arch_spec: Optional[ArchitectureSpec] = None
    arch_family: Optional[str] = None
    memory_model: Optional[MemoryModel] = None

    config_status: object = None

    flags: List[str] = field(default_factory=list)
    raw_config: dict = field(default_factory=dict)

    is_composite: bool = False
    submodels: List[str] = field(default_factory=list)


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


def _detect_composite(siblings, raw_config) -> Tuple[bool, List[str]]:
    """Detect a composite / multi-submodel repo from the file list + root config,
    with no weight download (fixes-plan Point 3).

    A composite is a container of ordinary models: >=2 subfolders that each carry
    their own ``config.json``, OR a repo that cannot load as one model
    (``model_index.json`` present with no root ``model_type``). Returns
    ``(is_composite, submodels)``. Standard single-root models (Nemotron/Qwen/XTTS)
    -> ``(False, [])``.
    """
    files = [getattr(s, "rfilename", "") for s in (siblings or [])]
    fileset = set(files)
    subdirs = {f.split("/")[0] for f in files if "/" in f}
    submodels = sorted(d for d in subdirs if f"{d}/config.json" in fileset)
    root_type = bool((raw_config or {}).get("model_type"))
    is_composite = len(submodels) >= 2 or ("model_index.json" in fileset and not root_type)
    return is_composite, submodels


def _sum_weight_files(siblings) -> Tuple[int, int]:
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
    if not parameters:
        return "bf16", "bf16 (assumed)", None, None

    total_params = sum(parameters.values())

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


_TORCH_DTYPE_BYTES = {
    "float32": 4,
    "float": 4,
    "float64": 8,
    "double": 8,
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float8": 1,
    "int8": 1,
    "uint8": 1,
}


def _bytes_per_param_from_config(model_id: str) -> Tuple[int, bool]:
    """Fallback bytes-per-param derived from ``config.json torch_dtype`` when the
    exact safetensors parameter count is unavailable.

    Returns ``(bytes, confident)``. ``confident`` is False when the dtype could
    not be determined and 2 (bf16) was assumed — callers should flag the derived
    parameter count as a low-confidence estimate. Keys on the universal weight
    dtype, never the architecture, so it holds for any repo (LLM / DiT / VAE / CNN).
    """
    cfg = _maybe_fetch_config(model_id) or {}
    td = str(cfg.get("torch_dtype") or "").lower().replace("torch.", "").strip()
    if td in _TORCH_DTYPE_BYTES:
        return _TORCH_DTYPE_BYTES[td], True
    return 2, False


def _maybe_fetch_config(model_id: str) -> Optional[dict]:
    safe_id = _validate_hf_id(model_id)
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(safe_id, trust_remote_code=True)
        return cfg.to_dict()
    except Exception:
        pass

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(safe_id, "config.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _probe_local_model(model_id: str) -> ModelProbe:
    """Build a ModelProbe from a local directory (bypasses the HF Hub API)."""
    weight_exts_legacy = (".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".nemo")
    sf_bytes = 0
    legacy_bytes = 0
    for entry in os.listdir(model_id):
        p = os.path.join(model_id, entry)
        if not os.path.isfile(p):
            continue
        size = os.path.getsize(p)
        if entry.endswith(".safetensors"):
            sf_bytes += size
        elif entry.endswith(weight_exts_legacy):
            legacy_bytes += size
    weight_bytes = sf_bytes if sf_bytes > 0 else legacy_bytes

    cfg = _maybe_fetch_config(model_id) or {}
    pipeline_tag = cfg.get("pipeline_tag")
    library = "transformers"
    category = _classify_category(pipeline_tag, [], library)
    model_type_category = _category_from_model_type(str(cfg.get("model_type", "")))
    if model_type_category:
        category = model_type_category
    elif category == "Unknown" and cfg.get("model_type"):
        category = "LLM"

    total_params = weight_bytes // 4 if weight_bytes > 0 else None
    bytes_per_param = (weight_bytes / total_params) if total_params else None

    probe = ModelProbe(
        model_id=model_id,
        category=category,
        pipeline_tag=pipeline_tag,
        library=library,
        weight_bytes_total=weight_bytes,
        weight_bytes_safetensors=sf_bytes,
        weight_bytes_legacy=legacy_bytes,
        saved_dtype="F32",
        saved_dtype_pretty="fp32",
        total_params=total_params,
        bytes_per_param_on_disk=bytes_per_param,
        raw_config=cfg,
    )
    return probe


def probe_model(model_id: str) -> ModelProbe:
    _validate_hf_id(model_id)
    if _is_local_model_dir(model_id):
        return _probe_local_model(model_id)
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
        _bpp, _confident = _bytes_per_param_from_config(model_id)
        total_params = weight_bytes // _bpp
        bytes_per_param = float(_bpp)
        pretty_dtype = (
            f"{pretty_dtype} (param count est. from config torch_dtype, {_bpp} B/param)"
            if _confident
            else f"{pretty_dtype} (param count est., dtype unknown — assumed bf16, low confidence)"
        )

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

    cfg = _maybe_fetch_config(model_id)
    if cfg is None:
        probe.config_status = "failed"
        return probe

    probe.raw_config = cfg
    probe.is_composite, probe.submodels = _detect_composite(info.siblings, cfg)
    if probe.is_composite:
        _sm = ", ".join(probe.submodels) or "model_index.json (no root model_type)"
        probe.flags.append(
            f"composite repo — {len(probe.submodels)} submodel(s) [{_sm}]; bring up per subfolder, not one root model"
        )

    model_type_category = _category_from_model_type(str(cfg.get("model_type", "")))
    if model_type_category and probe.category in {"LLM", "VLM"} and model_type_category != probe.category:
        probe.flags.append(
            f"Reclassified from {probe.category} to {model_type_category} via "
            f"config.model_type={cfg.get('model_type')!r}"
        )
        probe.category = model_type_category
    elif model_type_category and probe.category == "Unknown":
        probe.category = model_type_category

    if probe.category not in TRANSFORMER_CATEGORIES:
        probe.config_status = None
        return probe

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

        if family == "mla":
            probe.flags.append("MLA (compressed KV cache) detected — DeepSeek family")
        if family == "moe":
            probe.flags.append(f"MoE detected ({arch_spec.num_experts} experts, top-{arch_spec.experts_per_token})")
        if family == "ssm":
            probe.flags.append("State-space model — no per-token KV cache")
        if arch_spec.sliding_window:
            probe.flags.append(f"Sliding-window attention (window={arch_spec.sliding_window})")
    else:
        if probe.category in {"LLM", "VLM"}:
            probe.flags.append(
                "Category downgraded to CNN after config inspection: no causal-LM fields found in config.json."
            )
            probe.category = "CNN"
            probe.arch_spec = None
            probe.arch_family = None
            probe.config_status = None
        else:
            probe.config_status = False

    return probe
