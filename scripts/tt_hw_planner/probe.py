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
    "any-to-any": "VLM",
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

_AMBIGUOUS_PIPELINE_TAGS = {"text-to-audio", "audio-to-audio"}


def _is_low_confidence_category(
    pipeline_tag: Optional[str], model_type_category: Optional[str], arch_changed: bool = False
) -> bool:
    """A category is low-confidence when it was derived from an AMBIGUOUS pipeline_tag
    (e.g. ``text-to-audio`` spans TTS AND music/audio-generation) with no authoritative
    model_type or architecture signal confirming it. Clean tags (text-generation,
    text-to-speech, ...) are reliable and never flagged."""
    return bool(pipeline_tag in _AMBIGUOUS_PIPELINE_TAGS and not model_type_category and not arch_changed)


def _category_from_model_type(model_type: str) -> Optional[str]:
    """Classify a model_type via the installed transformers library's task registries
    ONLY -- a self-maintaining signal (a model_type new to this venv's transformers
    version is picked up without a tool edit; the venv tracks upstream via
    registry_sync). No hand-maintained per-model lists: those never converge (every
    unlisted model is a fresh miss) and were ~75% redundant with this registry anyway.
    Whatever the registry cannot place falls through to the arch-suffix / fingerprint
    layers and finally the LLM residual -- so a brand-new architecture just works
    without a code change. Returns a category or None; never raises."""
    mt = (model_type or "").lower()
    if not mt:
        return None
    return _category_from_transformers_registry(mt)


def _category_from_transformers_registry(model_type: str) -> Optional[str]:
    """Classify an unknown model_type via the installed transformers library's task
    registries. Self-updating: a model_type unknown to the hardcoded tables above is
    still classified as long as the venv's transformers version knows it (and that
    version now tracks upstream tt-metal via registry_sync). Fallback only, so known
    types keep their curated category. Returns a category or None; never raises."""
    try:
        from transformers.models.auto import modeling_auto as _ma
    except Exception:
        return None

    def _has(mapping_name: str) -> bool:
        m = getattr(_ma, mapping_name, None)
        return isinstance(m, dict) and model_type in m

    if _has("MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES") or _has("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES"):
        return "VLM"
    if _has("MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES") or _has("MODEL_FOR_CTC_MAPPING_NAMES"):
        return "STT"
    if (
        _has("MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES")
        or _has("MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES")
        or _has("MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES")
        or _has("MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES")
    ):
        return "CNN"
    if (
        _has("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES")
        or _has("MODEL_FOR_MASKED_LM_MAPPING_NAMES")
        or _has("MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES")
    ):
        return "LLM"
    return None


def _arch_override_category(category: str, cfg: dict) -> str:
    """Trust ``config.architectures`` over a diffusion/unknown pipeline_tag.

    A causal/MoE transformer can carry a diffusion pipeline_tag (e.g.
    HunyuanImage-3.0 is tagged text-to-image but its config declares
    ``architectures=["HunyuanImage3ForCausalMM"]`` with ``num_experts``). The
    pipeline_tag alone lands it in ``Image``, which early-returns before
    ``detect_architecture`` ever runs, so the MoE/attention structure is never
    seen and sibling matching force-fits a diffusion family. When the config
    itself declares a ``*ForCausalLM``/``*ForCausalMM`` architecture, reclassify
    an ``Image``/``Video``/``Unknown`` category to ``LLM`` so architecture
    detection runs. Model-agnostic (matches the architecture suffix, not a
    model name).

    The same authority order (architectures > model_type > pipeline_tag) also
    disambiguates a shared model_type: ``speecht5`` is one model_type serving
    ASR, TTS and voice-conversion, so the model_type table alone can only guess
    (it defaults to STT) and would mislabel ``SpeechT5ForTextToSpeech`` as STT.
    The class-name suffix states the task outright -- ``*ForTextToSpeech`` is
    TTS, ``*ForSpeechToText``/``*ForCTC`` is STT -- so when the current category
    is in the speech family (or Unknown) let the architecture suffix correct it.
    Suffix-matched, not model-name-matched."""
    archs = " ".join(cfg.get("architectures") or [])
    # a *ForCausalLM / *ForCausalMM trunk is a generative LM; a single-modality task tag
    # (e.g. Phi-4-multimodal tagged automatic-speech-recognition, arch Phi4MMForCausalLM)
    # must not override it. Genuine STT uses ConditionalGeneration/CTC, never ForCausalLM,
    # so promoting STT here is safe. TTS is excluded (some AR-TTS legitimately use ForCausalLM).
    if category in {"Image", "Video", "Unknown", "STT"}:
        if re.search(r"ForCausal(LM|MM)\b", archs):
            return "LLM"
    if category in {"TTS", "STT", "Unknown"}:
        if re.search(r"ForTextToSpeech\b", archs):
            return "TTS"
        if re.search(r"For(SpeechToText|CTC)\b", archs):
            return "STT"
    return category


_VALID_CATEGORIES = ("LLM", "VLM", "Image", "Video", "STT", "TTS", "Embed", "CNN", "NLP", "Unknown")
_LLM_CATEGORY_CACHE: dict = {}


def _fetch_model_card_text(model_id: str) -> str:
    """Fetch the model's README (the same prose a human reads to classify it) so the LLM
    resolver reasons over real evidence, not just a few config keys. Local dir or HF repo;
    best-effort, never raises, returns '' on any failure."""
    try:
        if _is_local_model_dir(model_id):
            p = os.path.join(model_id, "README.md")
            return open(p, encoding="utf-8").read() if os.path.isfile(p) else ""
        from huggingface_hub import hf_hub_download

        return open(hf_hub_download(model_id, "README.md"), encoding="utf-8").read()
    except Exception:
        return ""


def _category_from_fingerprint(fingerprint: str) -> Optional[str]:
    """Bridge the structural fingerprint to a category when the deterministic tag/
    model_type path came up ``Unknown`` but the fingerprint DID identify a backbone
    (e.g. Janus ``MultiModalityCausalLM`` -> 'decoder-only causal LM' but model_type
    'multi_modality' has no table/registry home and the 'any-to-any' tag isn't mapped).
    Uses the fact already computed, so no new signal is invented and the LLM residual
    is reserved for a genuinely 'unknown' fingerprint. Returns None if the fingerprint
    itself is unknown."""
    fp = fingerprint.lower()
    if fp.startswith("vlm"):
        return "VLM"
    if fp.startswith("decoder-only") or fp.startswith("ssm") or fp.startswith("autoregressive"):
        return "LLM"
    if fp.startswith("encoder-decoder") or fp.startswith("encoder-only"):
        return "LLM"
    if fp.startswith("vit") or fp.startswith("cnn"):
        return "CNN"
    if fp.startswith("dit") or "diffusion" in fp:
        return "Image"
    return None


def _is_dual_encoder_contrastive(cfg: dict) -> bool:
    """Structural FACT for a contrastive dual-encoder (CLIP / ALIGN / CLAP style): the
    config ships a ``text_config`` alongside a ``vision_config`` or ``audio_config`` and
    the architecture is a bare encoder (no generative *ForCausalLM / *ForConditional-
    Generation / *ForTextToSpeech head). Such models produce embeddings to MATCH inputs,
    not to synthesize -- category Embed. Reading this fact is stable where the LLM is
    flaky (it tends to call an audio/vision contrastive model a generation category).
    Generalizes to any dual-encoder; no per-model list."""
    keys = set(cfg or {})
    has_text = "text_config" in keys
    has_other = "vision_config" in keys or "audio_config" in keys
    archs = " ".join((cfg or {}).get("architectures") or [])
    # a pure contrastive dual-encoder is a BARE encoder (CLIPModel / ClapModel / AlignModel);
    # ANY task head (CLIPSegForImageSegmentation, ...ForConditionalGeneration) means it is a
    # task model built ON a dual encoder, not a contrastive retriever -> let it flow onward.
    task_head = re.search(r"For[A-Z]", archs)
    return has_text and has_other and not task_head


def _is_category_residual(model_type_category: Optional[str], fingerprint: str) -> bool:
    """The genuine residual for the LLM: NO deterministic fact placed this model.
    True only when the model_type carries no category (not in the curated table nor
    the installed transformers task registry) AND the structural fingerprint is
    ``unknown`` (no is_encoder_decoder / arch-suffix / module-tree signal either).
    Every model that any deterministic layer can classify is excluded, so the LLM
    fires on the tail (exotic/config-less arches), never on the common path."""
    return not model_type_category and fingerprint.startswith("unknown")


def _llm_resolve_category(model_id: str, cfg: dict, pipeline_tag: Optional[str], card_text: str = "") -> Optional[str]:
    """Ask the LLM to name the category for a residual model from the facts that DO
    exist -- model_type, architectures, salient config keys (which encode structure,
    e.g. ``sampling_rate``/``codebook_size`` => audio, ``vision_config`` => VLM) and
    the model-card summary. Generalized alternative to a per-model table: it reads the
    same evidence a human would. Gated by ``TT_HW_PLANNER_LLM_CATEGORY`` (default on);
    returns a validated category or None (degrade to the deterministic result, so the
    fail-loud guarantee holds). Cached per model_id; never raises."""
    if os.environ.get("TT_HW_PLANNER_LLM_CATEGORY", "1") == "0":
        return None
    if model_id in _LLM_CATEGORY_CACHE:
        return _LLM_CATEGORY_CACHE[model_id]
    if not card_text:
        card_text = _fetch_model_card_text(model_id)
    result: Optional[str] = None
    try:
        from .llm_synth import extract_json_from_llm_output, invoke_llm_cli_one_shot

        key_cfg = {
            k: cfg.get(k)
            for k in (
                "model_type",
                "architectures",
                "is_encoder_decoder",
                "sampling_rate",
                "codebook_size",
                "num_quantizers",
                "vision_config",
                "audio_config",
                "text_config",
                "num_mel_bins",
                "vocab_size",
                "image_size",
            )
            if k in cfg
        }
        prompt = (
            "Classify this Hugging Face model into exactly ONE hardware bring-up category.\n"
            f"Allowed categories: {', '.join(_VALID_CATEGORIES)}.\n"
            "Definitions: LLM=text-only generative language model; VLM=vision+language; "
            "Image/Video=visual generation; STT=speech->text; TTS=audio synthesis or neural "
            "audio codec; Embed=text embedding/retrieval; CNN=vision classification/detection/"
            "segmentation; NLP=encoder-only text understanding; Unknown=cannot tell.\n"
            "IMPORTANT -- synthesis vs analysis: Image/Video/TTS are ONLY for models that "
            "SYNTHESIZE brand-new media as output. A model that ANALYZES its input is NOT one of "
            "those, even when its task name contains 'generation': vision analysis -- "
            "classification, detection, SEGMENTATION / mask-generation (e.g. Segment-Anything), "
            "depth, keypoints -- is CNN; producing masks or boxes is analysis, not image synthesis. "
            "A contrastive / dual-encoder / retrieval / matching / zero-shot model (CLIP / ALIGN / "
            "CLAP style, which produces embeddings to MATCH inputs) is Embed -- or CNN for a vision "
            "task. Never label an analysis, segmentation, or matching model Image / Video / TTS.\n"
            "For a UNIFIED / OMNI / any-to-any / multimodal model (handles or emits several "
            "modalities, e.g. Qwen-Omni's text+audio+image+video), classify by its CORE trunk: VLM "
            "if it processes vision+language, else LLM. Do NOT reduce it to one secondary output "
            "(never call an omni multimodal model TTS just because it can also speak, or Image just "
            "because it can also draw).\n"
            f"model_id: {model_id}\n"
            f"pipeline_tag: {pipeline_tag}\n"
            f"config (salient keys): {json.dumps(key_cfg)[:1500]}\n"
            f"model card (README, read it to reason about what the model DOES): {card_text[:4000]}\n"
            'Reply with ONLY compact JSON: {"category": "<one of the allowed>"}'
        )

        def _one_vote(_i: int) -> Optional[str]:
            try:
                raw = invoke_llm_cli_one_shot(prompt, model="sonnet", timeout_s=90)
                parsed = extract_json_from_llm_output(raw) or {}
                cand = str(parsed.get("category") or "").strip()
                for c in _VALID_CATEGORIES:
                    if cand.lower() == c.lower():
                        return c
            except Exception:
                return None
            return None

        try:
            votes = max(1, int(os.environ.get("TT_HW_PLANNER_CATEGORY_VOTES", "3")))
        except (TypeError, ValueError):
            votes = 3
        tally: dict = {}
        if votes <= 1:
            picks = [_one_vote(0)]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(votes, 4)) as ex:
                picks = list(ex.map(_one_vote, range(votes)))
        for p in picks:
            if p:
                tally[p] = tally.get(p, 0) + 1
        if tally:
            result = max(tally, key=lambda k: (tally[k], k != "Unknown"))
    except Exception:
        result = None
    _LLM_CATEGORY_CACHE[model_id] = result
    return result


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


_SF_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1}
_SF_DTYPE_PRETTY = {
    "F64": "fp64",
    "F32": "fp32",
    "F16": "fp16",
    "BF16": "bf16",
    "F8_E4M3": "fp8_e4m3",
    "F8_E5M2": "fp8_e5m2",
}


def _bytes_per_param_from_safetensors(model_id: str, sf_files: List[str]) -> Tuple[Optional[int], bool, Optional[str]]:
    """Bytes-per-param from the DOMINANT float weight dtype in an actual safetensors
    file HEADER — the on-disk ground truth. Reads ONE file's header only (no weight
    download). Handles composite / no-index repos where config torch_dtype is absent
    (e.g. LongCat-Video, fp32 weights under dit/). Returns ``(bytes, True)`` or
    ``(None, False)`` when unreadable. Never raises."""
    if not sf_files:
        return None, False, None
    try:
        from huggingface_hub import HfApi

        md = HfApi().parse_safetensors_file_metadata(model_id, sf_files[0])
    except Exception:
        return None, False, None
    counts: dict = {}
    for t in md.tensors.values():
        dt = str(getattr(t, "dtype", ""))
        if dt in _SF_DTYPE_BYTES:
            counts[dt] = counts.get(dt, 0) + 1
    if not counts:
        return None, False, None
    dom = max(counts, key=counts.get)
    return _SF_DTYPE_BYTES[dom], True, _SF_DTYPE_PRETTY.get(dom, dom.lower())


def _bytes_per_param_from_local_safetensors(model_dir: str) -> Tuple[Optional[int], Optional[str]]:
    """Dominant float dtype (bytes, pretty) read directly from a LOCAL safetensors file
    HEADER on disk (8-byte length prefix + JSON header) — the ground truth for a local
    repo whose config has no/ambiguous torch_dtype. Reads one header only. Returns
    ``(bytes, pretty)`` or ``(None, None)``; never raises."""
    import glob
    import struct

    files = sorted(glob.glob(os.path.join(model_dir, "**", "*.safetensors"), recursive=True))
    for f in files[:1]:
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
        except Exception:  # noqa: BLE001
            continue
        counts: dict = {}
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            dt = str(v.get("dtype", ""))
            if dt in _SF_DTYPE_BYTES:
                counts[dt] = counts.get(dt, 0) + 1
        if counts:
            dom = max(counts, key=counts.get)
            return _SF_DTYPE_BYTES[dom], _SF_DTYPE_PRETTY.get(dom, dom.lower())
    return None, None


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


def _read_model_card_frontmatter(model_dir: str) -> dict:
    """Parse ``pipeline_tag`` and ``tags`` from a local repo's ``README.md`` YAML
    frontmatter — the model-card metadata that HF ships in-repo. ``config.json``
    never carries ``pipeline_tag`` (it is hub/model-card metadata), so a local
    probe must read the card. Returns ``{}`` when absent or unparseable."""
    path = os.path.join(model_dir, "README.md")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return {}
    if not text.lstrip().startswith("---"):
        return {}
    start = text.index("---") + 3
    end = text.find("\n---", start)
    if end == -1:
        return {}
    block = text[start:end]
    try:
        import yaml

        meta = yaml.safe_load(block)
        if isinstance(meta, dict):
            tags = meta.get("tags")
            if isinstance(tags, str):
                tags = [tags]
            return {
                "pipeline_tag": meta.get("pipeline_tag"),
                "tags": list(tags) if isinstance(tags, list) else [],
            }
    except Exception:
        pass
    return _parse_frontmatter_lines(block)


def _parse_frontmatter_lines(block: str) -> dict:
    """Dependency-free fallback parser for a README frontmatter block: pulls the
    ``pipeline_tag`` scalar and a block/inline ``tags`` list."""
    pipeline_tag = None
    tags: List[str] = []
    in_tags = False
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("pipeline_tag:"):
            pipeline_tag = line.split(":", 1)[1].strip().strip("'\"") or None
            in_tags = False
        elif line.startswith("tags:"):
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("[") and rest.endswith("]"):
                tags = [t.strip().strip("'\"") for t in rest[1:-1].split(",") if t.strip()]
                in_tags = False
            else:
                in_tags = True
        elif in_tags and line.lstrip().startswith("- "):
            tags.append(line.lstrip()[2:].strip().strip("'\""))
        elif not line.startswith((" ", "\t", "-")):
            in_tags = False
    return {"pipeline_tag": pipeline_tag, "tags": tags}


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
    card = _read_model_card_frontmatter(model_id)
    pipeline_tag = cfg.get("pipeline_tag") or card.get("pipeline_tag")
    card_tags = card.get("tags") or []
    library = cfg.get("library_name") or card.get("library_name") or "transformers"
    category = _classify_category(pipeline_tag, card_tags, library)
    model_type_category = _category_from_model_type(str(cfg.get("model_type", "")))
    if model_type_category:
        category = model_type_category
    elif category == "Unknown" and cfg.get("model_type"):
        category = "LLM"
    category = _arch_override_category(category, cfg)

    from .fingerprint import arch_descriptor as _arch_descriptor

    _fpr = _arch_descriptor(
        model_type=cfg.get("model_type"),
        architectures=cfg.get("architectures"),
        is_encoder_decoder=cfg.get("is_encoder_decoder"),
        pipeline_tag=pipeline_tag,
    )
    if category == "Unknown":
        category = _category_from_fingerprint(_fpr) or category
    if _is_category_residual(model_type_category, _fpr) and _is_dual_encoder_contrastive(cfg):
        category = "Embed"
    elif _is_category_residual(model_type_category, _fpr):
        _llm_cat = _llm_resolve_category(model_id, cfg, pipeline_tag)
        if _llm_cat:
            category = _llm_cat

    _td = str(cfg.get("torch_dtype") or "").lower().replace("torch.", "").strip()
    _bpp = _TORCH_DTYPE_BYTES.get(_td)
    _dtype_pretty = _td or None
    if _bpp is None:
        _hb, _hp = _bytes_per_param_from_local_safetensors(model_id)
        if _hb is not None:
            _bpp, _dtype_pretty = _hb, _hp
    _dtype_confident = _bpp is not None
    if _bpp is None:
        _bpp = 2
    total_params = weight_bytes // _bpp if weight_bytes > 0 else None
    bytes_per_param = float(_bpp) if weight_bytes > 0 else None
    pretty = (
        (_dtype_pretty or "bf16")
        if _dtype_confident
        else f"{_dtype_pretty or 'bf16'} (dtype unknown — assumed bf16, low confidence)"
    )

    probe = ModelProbe(
        model_id=model_id,
        category=category,
        pipeline_tag=pipeline_tag,
        library=library,
        weight_bytes_total=weight_bytes,
        weight_bytes_safetensors=sf_bytes,
        weight_bytes_legacy=legacy_bytes,
        saved_dtype=(_dtype_pretty or "bf16").upper(),
        saved_dtype_pretty=pretty,
        total_params=total_params,
        bytes_per_param_on_disk=bytes_per_param,
        raw_config=cfg,
    )
    if _is_low_confidence_category(pipeline_tag, model_type_category):
        probe.flags.append(
            f"LOW-CONFIDENCE category {category!r}: inferred from the AMBIGUOUS pipeline_tag "
            f"{pipeline_tag!r} with no recognized model_type/architectures — verify. "
            f"('text-to-audio' spans text-to-speech AND music/audio-generation.)"
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
        _src = "config torch_dtype"
        _hdr_pretty = None
        if not _confident:
            _sf = [s.rfilename for s in info.siblings if str(s.rfilename).endswith(".safetensors")]
            _sf_bpp, _sf_conf, _hdr_pretty = _bytes_per_param_from_safetensors(model_id, _sf)
            if _sf_conf:
                _bpp, _confident, _src = _sf_bpp, True, "safetensors header"
        total_params = weight_bytes // _bpp
        bytes_per_param = float(_bpp)
        _base = _hdr_pretty if _hdr_pretty else pretty_dtype
        pretty_dtype = (
            f"{_base} (param count est. from {_src}, {_bpp} B/param)"
            if _confident
            else f"{pretty_dtype} (param count est., dtype unknown — assumed bf16, low confidence)"
        )
        if _hdr_pretty and _confident:
            canonical_dtype = _hdr_pretty

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

    _arch_cat = _arch_override_category(probe.category, cfg)
    _arch_changed = _arch_cat != probe.category
    if _arch_changed:
        probe.flags.append(
            f"Reclassified {probe.category} to {_arch_cat} via " f"config.architectures={cfg.get('architectures')!r}"
        )
        probe.category = _arch_cat

    from .fingerprint import arch_descriptor as _arch_descriptor

    _fpr = _arch_descriptor(
        model_type=cfg.get("model_type"),
        architectures=cfg.get("architectures"),
        is_encoder_decoder=cfg.get("is_encoder_decoder"),
        pipeline_tag=probe.pipeline_tag,
    )
    if probe.category == "Unknown":
        _fp_cat = _category_from_fingerprint(_fpr)
        if _fp_cat:
            probe.flags.append(f"Category Unknown -> {_fp_cat!r} via structural fingerprint {_fpr!r}")
            probe.category = _fp_cat
    if _is_category_residual(model_type_category, _fpr) and _is_dual_encoder_contrastive(cfg):
        if probe.category != "Embed":
            probe.flags.append(
                "Category -> 'Embed' via dual-encoder contrastive fact (text_config + vision/audio_config)"
            )
            probe.category = "Embed"
    elif _is_category_residual(model_type_category, _fpr):
        _llm_cat = _llm_resolve_category(model_id, cfg, probe.pipeline_tag)
        if _llm_cat and _llm_cat != probe.category:
            probe.flags.append(
                f"Category {probe.category!r} -> {_llm_cat!r} by LLM fallback: no deterministic "
                f"fact placed it (model_type/registry unknown, structural fingerprint 'unknown')."
            )
            probe.category = _llm_cat

    if _is_low_confidence_category(probe.pipeline_tag, model_type_category, _arch_changed):
        probe.flags.append(
            f"LOW-CONFIDENCE category {probe.category!r}: inferred from the AMBIGUOUS pipeline_tag "
            f"{probe.pipeline_tag!r} with no recognized model_type/architectures — verify. "
            f"('text-to-audio' spans BOTH text-to-speech and music/audio-generation; sibling "
            f"routing uses the module-tree fingerprint, so a diffusion/DiT trunk still routes correctly.)"
        )

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
