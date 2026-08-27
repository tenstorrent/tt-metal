from __future__ import annotations

import json
import os
import tempfile
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence

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
    "audio-classification": "STT",
    "text-to-speech": "TTS",
    "text-to-audio": "AudioGen",
    "text-to-music": "AudioGen",
    "music-generation": "AudioGen",
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
    # HF publishes 47 pipeline tags; these 19 were absent, so a model carrying one fell through to the
    # keyword guess in _classify_category and, failing that, to "Unknown" -- which drives category
    # routing, the reference loader and the placement plan. Sourced from huggingface.co/api/tasks
    # rather than invented, so the list can be re-diffed when HF adds more.
    "audio-text-to-text": "VLM",
    "video-text-to-text": "VLM",
    "document-question-answering": "VLM",
    "visual-document-retrieval": "VLM",
    "image-text-to-image": "Image",
    "unconditional-image-generation": "Image",
    "text-to-3d": "Image",
    "image-text-to-video": "Video",
    "question-answering": "LLM",
    "summarization": "LLM",
    "translation": "LLM",
    "table-question-answering": "LLM",
    "text-classification": "Embed",
    "token-classification": "Embed",
    "text-ranking": "Embed",
    "zero-shot-classification": "Embed",
    "tabular-classification": "Embed",
    "tabular-regression": "Embed",
    "reinforcement-learning": "Unknown",
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


_VALID_CATEGORIES = ("LLM", "VLM", "Image", "Video", "STT", "TTS", "AudioGen", "Embed", "CNN", "NLP", "Unknown")
_LLM_CATEGORY_CACHE: dict = {}
_AGENT_CATEGORY_CACHE: dict = {}


def _agent_classify_category(model_id: str, cfg: dict, card_text: str = "") -> Optional[str]:
    """Classify a model's bring-up category with a REAL AGENT (claude -p + tools), not a
    one-shot guess: it is handed the config + card as a starting point and may INVESTIGATE
    further (WebFetch the HF page, read files) and VERIFY before answering. This is the
    generalizing path -- it reasons about any model the way a human would, so no per-model
    table or rule is needed. Gated by ``TT_HW_PLANNER_AGENT_CLASSIFY`` (default on); cached
    per model; returns a validated category or None (caller falls back). Never raises."""
    if os.environ.get("TT_HW_PLANNER_AGENT_CLASSIFY", "1") == "0":
        return None
    if model_id in _AGENT_CATEGORY_CACHE:
        return _AGENT_CATEGORY_CACHE[model_id]
    result: Optional[str] = None
    try:
        from ._cli_helpers.agent import resolve_claude_bin
        from .llm_synth import extract_json_from_llm_output, invoke_llm_agent

        salient = {
            k: cfg.get(k)
            for k in ("model_type", "architectures", "is_encoder_decoder", "pipeline_tag", "sampling_rate")
            if k in cfg
        }
        # crisp presence FLAGS for the big nested sub-configs -- dumping the full dicts buried the
        # signal and got truncated, so the agent missed the vision tower (a reasoning-heavy VLM whose card downplays vision).
        salient["has_vision_tower"] = (
            ("vision_config" in cfg) or ("image_token_id" in cfg) or ("vision_start_token_id" in cfg)
        )
        salient["has_audio_config"] = ("audio_config" in cfg) or ("codebook_size" in cfg)
        salient["has_text_config"] = "text_config" in cfg
        prompt = (
            "You are categorizing a Hugging Face model for hardware bring-up on Tenstorrent "
            "accelerators. Investigate the model (you MAY use WebFetch on its Hugging Face page "
            f"https://huggingface.co/{model_id} or read its files) and decide its PRIMARY role, "
            "then answer with exactly one category.\n"
            f"Allowed categories: {', '.join(_VALID_CATEGORIES)}.\n"
            "Guidance: classify by the model's PRIMARY input->output. VLM needs BOTH vision and "
            "language; a vision-only model (classification/detection/segmentation/depth) is CNN; a "
            "text-only embedder/retriever/reranker is Embed; Image/Video/TTS/AudioGen are for models "
            "that SYNTHESIZE that media (not analyze it). Within audio SYNTHESIS, split by what is "
            "produced: a model that synthesizes SPEECH (a spoken voice reading text, voice cloning, "
            "text->spoken-word) is TTS; a model that synthesizes MUSIC or general non-speech audio "
            "(songs, instrumental/ambient tracks, sound effects, foley) is AudioGen. A model that is "
            "not a text/vision/audio deep "
            "network (tabular, classical non-neural ML like trees/boosting, time-series forecasting, "
            "graph, or a pure control/RL policy) is Unknown -- BUT a vision-language-action (robot "
            "policy) model built on a vision+language transformer backbone is VLM (classify by that "
            "backbone). A biological-SEQUENCE model (a transformer over protein/DNA/amino-acid "
            "sequences) is treated like a text model: Embed if it outputs embeddings/representations, "
            "else NLP/LLM. STT is the AUDIO-INPUT bucket: speech->text AND any model that "
            "analyzes/classifies/tags audio (events, emotion, speaker, music -- including spectrogram "
            "transformers) map to STT, not CNN, even though a spectrogram looks image-like. STT "
            "requires AUDIO input. Reading text FROM IMAGES or documents (OCR, e.g. a vision-encoder-"
            "decoder) is VLM, never STT. If the config declares a vision ENCODER (a vision_config "
            "sub-config) AND the model GENERATES text, it is a VLM even when the description stresses "
            "reasoning or text -- the encoder means it accepts images (this does NOT apply to a bare "
            "contrastive dual-encoder, which stays Embed). CRITICAL (autoregressive token generators): "
            "a model whose architecture ends in *ForCausalLM / *ForCausalMM (a causal / autoregressive "
            "LM trunk) that produces images or audio by emitting TOKENS is classified by that trunk -- "
            "LLM (or VLM if it also has a vision ENCODER) -- never Image/TTS/AudioGen. Image/Video/TTS/"
            "AudioGen are ONLY for diffusion / GAN media synthesizers that have NO autoregressive LM "
            "trunk. So a "
            "*ForCausalLM model tagged text-to-image is still LLM: classify by the transformer trunk, "
            "never send it to a diffusion path.\n"
            f"model_id: {model_id}\n"
            f"config (salient keys): {json.dumps(salient)[:1500]}\n"
            f"model card (excerpt): {card_text[:3000]}\n"
            'When done, output ONLY compact JSON on the final line: {"category": "<one allowed>"}'
        )
        _bin = resolve_claude_bin() or "claude"

        def _one_vote(_i: int) -> Optional[str]:
            try:
                raw = invoke_llm_agent(
                    prompt,
                    agent_bin=_bin,
                    model=os.environ.get("TT_HW_PLANNER_AGENT_MODEL", "opus"),
                    timeout_s=240,
                )
                parsed = extract_json_from_llm_output(raw) or {}
                cand = str(parsed.get("category") or "").strip()
                for c in _VALID_CATEGORIES:
                    if cand.lower() == c.lower():
                        return c
                # lenient fallback: the agent reasoned but didn't emit clean JSON. Take the LAST
                # category token that appears in its answer (the conclusion comes last). Prevents a
                # correct-but-unparsed answer from becoming a None (general, not model-specific).
                low = (raw or "").lower()
                best_c, best_i = None, -1
                for c in _VALID_CATEGORIES:
                    j = low.rfind(c.lower())
                    if j > best_i:
                        best_c, best_i = c, j
                return best_c
            except Exception:
                return None

        try:
            votes = max(1, int(os.environ.get("TT_HW_PLANNER_AGENT_VOTES", "3")))
        except (TypeError, ValueError):
            votes = 3
        # self-consistency: run the agent N times CONCURRENTLY (~same wall-clock) and take the
        # majority, so a single flaky run can't decide the category. None votes are ignored.
        if votes <= 1:
            picks = [_one_vote(0)]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=votes) as _ex:
                picks = list(_ex.map(_one_vote, range(votes)))
        tally: dict = {}
        for p in picks:
            if p:
                tally[p] = tally.get(p, 0) + 1
        if tally:
            result = max(tally, key=lambda k: (tally[k], k != "Unknown"))
    except Exception:
        result = None
    _AGENT_CATEGORY_CACHE[model_id] = result
    return result


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


def _has_generative_vlm_fact(cfg: dict) -> bool:
    """DEFINITIVE fact: the config declares a real vision ENCODER (``vision_config``) AND a
    generative arch (*ForCausalLM / *ForCausalMM / *ForConditionalGeneration) -> the model accepts
    images and emits text -> VLM. AUTHORITATIVE (the agent over-weights a text-heavy card and can
    miss the vision tower). CRITICAL: require ``vision_config`` (the encoder), NOT a
    bare ``image_token_id``/``vision_start_token_id`` -- image GENERATORS (HunyuanImage, Emu3) carry
    image-token ids to EMIT image tokens but have no vision encoder; they are LLM-trunk generators
    (issue #3), not VLMs. A bare contrastive dual-encoder (CLIP: vision_config but a *Model class,
    no generative head) is excluded -- it stays Embed."""
    keys = set(cfg or {})
    has_vision_encoder = "vision_config" in keys
    archs = " ".join((cfg or {}).get("architectures") or [])
    generative = re.search(r"For(Causal(LM|MM)|ConditionalGeneration)\b", archs)
    return has_vision_encoder and bool(generative)


def _has_audio_markers(cfg: dict) -> bool:
    """Structural fact that a config describes an audio/waveform model (codec, vocoder,
    speech): sampling_rate / codebook / quantizer / mel bins / audio_config. Used to catch
    the ambiguous 'feature-extraction' tag mislabeling an audio codec (mimi, encodec) as a
    text embedder -- the audio structure contradicts 'text embedding'."""
    keys = set(cfg or {})
    if keys & {"sampling_rate", "codebook_size", "num_quantizers", "num_mel_bins", "audio_config"}:
        return True
    return any("codebook" in k or "quantizer" in k for k in keys)


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
        from .llm_synth import extract_json_from_llm_output, invoke_llm_agent

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
            "Decide by the model's PRIMARY input/output, in this order:\n"
            "1. Does it SYNTHESIZE new media? new images/video -> Image/Video; SPEECH synthesis (a "
            "spoken voice reading text, voice cloning) or a neural speech codec -> TTS; MUSIC or "
            "general non-speech audio synthesis (songs, instrumental/ambient, sound effects) -> "
            "AudioGen. (Producing masks, boxes, depth or embeddings is NOT synthesis -- keep going.)\n"
            "2. Speech/audio -> text (transcription/recognition) -> STT.\n"
            "3. Vision task with NO language output (image classification, detection, SEGMENTATION/"
            "masks incl. Segment-Anything, depth, keypoints, matting, super-resolution) -> CNN.\n"
            "4. Produces embeddings/vectors for retrieval, similarity, matching, reranking, or "
            "contrastive scoring (CLIP/ALIGN/CLAP/BERT-embedder style) -> Embed. A text-only "
            "embedding model is Embed even if multilingual or multi-task -- NOT VLM.\n"
            "5. Takes IMAGES/VIDEO **and** TEXT together and outputs text (captioning, VQA, "
            "doc/chart understanding, or an omni model whose core reasons over vision+language) "
            "-> VLM. VLM REQUIRES BOTH vision and language; a vision-only model is CNN, a "
            "text-only model is never VLM.\n"
            "6. Text in, text out, generative -> LLM (this includes text seq2seq translation/"
            "summarization and any *ForCausalLM trunk). Encoder-only text understanding -> NLP.\n"
            "7. None fit / cannot tell from the evidence -> Unknown.\n"
            "For a UNIFIED/OMNI model, classify by its CORE trunk (VLM if it reasons over "
            "vision+language, else LLM), never by a secondary output it can also emit.\n"
            f"model_id: {model_id}\n"
            f"pipeline_tag: {pipeline_tag}\n"
            f"config (salient keys): {json.dumps(key_cfg)[:1500]}\n"
            f"model card (README, read it to reason about what the model DOES): {card_text[:4000]}\n"
            'Reply with ONLY compact JSON: {"category": "<one of the allowed>"}'
        )

        def _one_vote(_i: int) -> Optional[str]:
            try:
                raw = invoke_llm_agent(prompt, model="opus", timeout_s=200)
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
    # Composite repos have no root model_type. The diffusers recipe
    # (model_index.json "_class_name", e.g. "Flux2KleinPipeline") is the only
    # architecture identity they expose; backend routing uses it as a surrogate.
    pipeline_class: Optional[str] = None


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
    if any(
        t in tag_str for t in ["text-to-music", "music-generation", "musicgen", "text-to-audio", "audio-generation"]
    ):
        return "AudioGen"
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

        path = hf_hub_download(safe_id, ROOT_CONFIG_FILE)
        with open(path) as f:
            return json.load(f)
    except Exception:
        pass

    # Last resort: read the config document directly. AutoConfig only understands
    # transformers-style configs (keyed by ``model_type``); a component of a
    # composite describes itself with ``_class_name`` instead, and a local model
    # directory is not downloadable at all. Both are still perfectly good configs.
    return fetch_repo_json(safe_id, ROOT_CONFIG_FILE)


COMPOSITE_INDEX_FILE = "model_index.json"
ROOT_CONFIG_FILE = "config.json"


def fetch_repo_json(model_id: str, filename: str) -> Optional[dict]:
    """Download and parse one JSON file from a model repo (or read it from a local
    model dir). Returns ``None`` on any failure -- missing file, no access, bad
    JSON. Shared by every caller that needs a raw repo-side JSON document."""
    if isinstance(model_id, str) and os.path.isdir(model_id):
        safe_id = model_id
    else:
        try:
            safe_id = _validate_hf_id(model_id)
        except Exception:
            return None
    if os.path.isdir(safe_id):
        path = os.path.join(safe_id, filename)
        try:
            with open(path) as f:
                doc = json.load(f)
            return doc if isinstance(doc, dict) else None
        except Exception:
            return None
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(safe_id, filename)) as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else None
    except Exception:
        return None


def _repo_access_status(model_id: str, filename: str) -> str:
    """Why fetching ``filename`` from ``model_id`` fails: ``"ok"``, ``"denied"``
    (gated / private / missing repo -- an access problem), ``"absent"`` (repo is
    readable, that file simply is not in it), or ``"unknown"``.

    Classified from the hub client's own exception types, not from message text,
    so the two cases that need OPPOSITE advice are never confused: a gated repo
    needs credentials, a readable repo missing a config needs a different model.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return "unknown"
    try:
        hf_hub_download(_validate_hf_id(model_id), filename)
        return "ok"
    except Exception as exc:
        try:
            from huggingface_hub import errors as _hf_errors
        except Exception:
            return "unknown"
        denied = tuple(
            c
            for c in (
                getattr(_hf_errors, "GatedRepoError", None),
                getattr(_hf_errors, "RepositoryNotFoundError", None),
                getattr(_hf_errors, "LocalTokenNotFoundError", None),
            )
            if isinstance(c, type)
        )
        absent = tuple(
            c
            for c in (
                getattr(_hf_errors, "EntryNotFoundError", None),
                getattr(_hf_errors, "RemoteEntryNotFoundError", None),
            )
            if isinstance(c, type)
        )
        if denied and isinstance(exc, denied):
            return "denied"
        if absent and isinstance(exc, absent):
            return "absent"
        return "unknown"


class _FileEntry:
    """Minimal stand-in for a hub sibling so a local directory can be fed to the
    same composite rule the hub path uses."""

    __slots__ = ("rfilename",)

    def __init__(self, rfilename: str) -> None:
        self.rfilename = rfilename


def _local_siblings(model_dir: str, max_depth: int = 2) -> List[_FileEntry]:
    """Repo-relative file list for a local directory, shaped like hub siblings."""
    out: List[_FileEntry] = []
    base = os.path.abspath(model_dir)
    for root, dirs, files in os.walk(base):
        rel_root = os.path.relpath(root, base)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        if depth >= max_depth:
            dirs[:] = []
        for fname in files:
            rel = fname if rel_root == "." else f"{rel_root}/{fname}"
            out.append(_FileEntry(rel.replace(os.sep, "/")))
    return out


def detect_composite_repo(model_id: str) -> Tuple[bool, List[str]]:
    """``(is_composite, component_names)`` from the file listing alone.

    :func:`probe_model` is expensive -- it may consult an agent to classify the
    model -- so callers that only need to know whether a target is a container of
    models use this instead. It applies the SAME rule as the full probe
    (:func:`_detect_composite`), so the two can never disagree, and reads only the
    file list plus the root config document. Never raises."""
    if not isinstance(model_id, str) or not model_id:
        return (False, [])
    # A composite directory has no root config.json, which is exactly what
    # _is_local_model_dir() requires -- so "is it on disk" is the local test here.
    local = os.path.isdir(model_id)
    if not local:
        try:
            model_id = _validate_hf_id(model_id)
        except Exception:
            return (False, [])
    cfg = fetch_repo_json(model_id, ROOT_CONFIG_FILE)
    try:
        if local:
            siblings: List[_FileEntry] = _local_siblings(model_id)
        else:
            from huggingface_hub import HfApi

            siblings = HfApi().model_info(model_id).siblings or []
    except Exception:
        return (False, [])
    try:
        return _detect_composite(siblings, cfg)
    except Exception:
        return (False, [])


def _surrogate_pipeline_class(cfg: Optional[dict]) -> Optional[str]:
    """Routing surrogate for a config that has no ``model_type``.

    A composite's component describes itself with ``_class_name`` instead. That is
    the only architecture identity it publishes, so routing uses it the same way
    the composite root uses ``model_index.json``'s. Whatever the class is called is
    read from the document -- never assumed."""
    if not isinstance(cfg, dict) or cfg.get("model_type"):
        return None
    cls = cfg.get("_class_name")
    return cls if isinstance(cls, str) and cls else None


def _component_alias(parent_id: str, name: str, target: str) -> str:
    """A stable, uniquely-named path pointing at a component directory.

    Downstream naming (demo folder, overlays, worktrees) derives from the target's
    basename. A component directory is named after its role inside the repo, so
    two different models both contribute a part with the same role and would land
    in the same demo folder. Aliasing under ``<parent>__<part>`` keeps the parent's
    identity attached without copying any weights.

    Falls back to the real directory if the alias cannot be created, so a
    read-only or unusual filesystem degrades to today's behaviour."""
    from .scaffold_demo_folder import _slug

    base = os.environ.get("TT_HW_PLANNER_COMPONENT_BASE") or os.path.join(
        tempfile.gettempdir(), "tt_hw_planner_components"
    )
    # Single separator, deliberately: every downstream name (demo folder, overlay
    # scope, worktree) is derived from this basename through _slug, which collapses
    # any run of non-alphanumerics to one underscore. A doubled separator could
    # therefore never survive, and the component ended up with two spellings --
    # `<parent>__<part>` here and `<parent>_<part>` in the demo folder -- for the
    # same thing. One separator means one name everywhere.
    alias_name = f"{_slug(os.path.basename(parent_id.rstrip('/')))}_{_slug(name)}"
    alias = os.path.join(base, alias_name)
    try:
        os.makedirs(base, exist_ok=True)
        if os.path.islink(alias) or os.path.exists(alias):
            if os.path.realpath(alias) == os.path.realpath(target):
                return alias
            os.unlink(alias)
        os.symlink(os.path.abspath(target), alias)
        return alias
    except OSError:
        return target


def component_targets(model_id: str, submodels: Sequence[str], *, with_weights: bool = False) -> List[Tuple[str, str]]:
    """``[(component_name, local_path)]`` for a composite's parts.

    Each part of a composite repo is an ordinary single-root model: its own
    directory with its own config and weights. Materialising nothing, this returns
    paths the rest of the tool can treat like any local model directory, so a
    composite is brought up by running the existing per-model pipeline once per
    part instead of needing a parallel implementation.

    Names come from the caller (discovered from the repo listing); none are
    assumed here. Parts without a readable config are skipped rather than guessed
    at. Returns ``[]`` when nothing can be resolved.

    ``with_weights=False`` (the default) fetches only each part's config, which is
    all that is needed to enumerate and route -- enumerating must not drag down
    tens of GB of weights. Pass ``with_weights=True`` for the part about to be
    brought up, so its tensors are materialised just before they are needed."""
    names = [n for n in (submodels or []) if n and not n.startswith((".", "/"))]
    if not names:
        return []

    root: Optional[str] = model_id if os.path.isdir(model_id) else None
    if root is None:
        try:
            from huggingface_hub import snapshot_download

            patterns = [f"{n}/*" for n in names] if with_weights else [f"{n}/{ROOT_CONFIG_FILE}" for n in names]
            root = snapshot_download(_validate_hf_id(model_id), allow_patterns=patterns)
        except Exception:
            return []

    out: List[Tuple[str, str]] = []
    for name in names:
        path = os.path.join(root, name)
        if _is_local_model_dir(path):
            out.append((name, _component_alias(model_id, name, path)))
    return out


def missing_config_reason(probe: "ModelProbe", model_id: str) -> str:
    """Why this repo has no usable root config, phrased from evidence.

    Three different situations produce an empty ``raw_config`` and they need
    different answers:

    * the repo could not be READ at all (gated / private / typo) -> an access
      problem, and the only case where credentials are the answer;
    * the repo was read and is a COMPOSITE -> it has no root config by design,
      its architecture lives in the per-component subfolders;
    * the repo was read but exposes no architecture identity at all (no root
      config, no composite index, no component configs) -> it is not a standalone
      model: an adapter/LoRA, a single-file checkpoint, or a weights-only repo.

    Blaming credentials for the last two is what sent a user chasing HF_TOKEN for
    a public repo. Readability is inferred from whether the file listing produced
    anything, never assumed."""
    if getattr(probe, "is_composite", False):
        subs = ", ".join(getattr(probe, "submodels", []) or []) or COMPOSITE_INDEX_FILE
        return (
            f"{model_id} is a composite / multi-component repo [{subs}] -- it has no root "
            f"{ROOT_CONFIG_FILE} by design; each component carries its own. Bring up per component."
        )
    # Metadata is published for gated repos, so a populated probe proves nothing
    # about whether the FILES can be read. Ask the hub directly and let its own
    # error type decide, because "denied" and "absent" need opposite advice.
    status = _repo_access_status(model_id, ROOT_CONFIG_FILE)
    if status == "denied":
        return (
            f"{model_id} cannot be read: access to its files is denied (gated, private, or the repo "
            f"does not exist). Accept the model's terms on its HuggingFace page, then set HF_TOKEN or "
            f"run `huggingface-cli login`."
        )
    if status == "absent":
        return (
            f"{model_id} is readable but exposes no architecture identity: no root {ROOT_CONFIG_FILE}, "
            f"no {COMPOSITE_INDEX_FILE}, and no per-component configs. It is most likely not a standalone "
            f"model (an adapter/LoRA, a single-file checkpoint, or a weights-only repo) -- point the tool "
            f"at the base model it adapts."
        )
    return (
        f"no usable {ROOT_CONFIG_FILE} could be loaded for {model_id}, and the reason could not be "
        f"determined (network or hub error). Re-run; if it persists, check access and that the repo "
        f"publishes a {ROOT_CONFIG_FILE} or {COMPOSITE_INDEX_FILE}."
    )


def _maybe_fetch_pipeline_class(model_id: str) -> Optional[str]:
    """Read ``model_index.json["_class_name"]`` -- the pipeline class (e.g.
    ``Flux2KleinPipeline``). For a composite repo this is the only architecture
    identity on offer, since there is no root ``model_type``; the backend router
    uses it as a surrogate so routing stays deterministic instead of falling
    through to the LLM sibling ranker."""
    cls = (fetch_repo_json(model_id, COMPOSITE_INDEX_FILE) or {}).get("_class_name")
    return cls if isinstance(cls, str) and cls else None


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
    # PRIMARY + VERIFIED: CC agent (3-vote majority) decides every model, incl. vision models --
    # majority voting removes the single-run flakiness, so no unverified authoritative fact is
    # needed. Deterministic facts are the OFFLINE FALLBACK (agent unavailable).
    _agent_cat = _agent_classify_category(model_id, cfg, _fetch_model_card_text(model_id))
    if _agent_cat:
        category = _agent_cat
    else:
        if _has_generative_vlm_fact(cfg):
            category = "VLM"
        else:
            _resid = _is_category_residual(model_type_category, _fpr)
            if _resid and _is_dual_encoder_contrastive(cfg) and category in {"Unknown", "Embed"}:
                category = "Embed"

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
    probe.pipeline_class = _surrogate_pipeline_class(cfg)
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
    # Composite detection reads the HF *file listing* and tolerates a missing root
    # config -- "no root model_type" is itself half the composite signal. It must
    # therefore run BEFORE the config-failure early return, or diffusers-style
    # repos (model_index.json + per-subfolder configs) are never detected.
    probe.is_composite, probe.submodels = _detect_composite(info.siblings, cfg)
    if probe.is_composite:
        probe.pipeline_class = _maybe_fetch_pipeline_class(model_id)
    else:
        probe.pipeline_class = _surrogate_pipeline_class(cfg)
    if cfg is None:
        probe.config_status = "failed"
        return probe

    probe.raw_config = cfg
    if probe.is_composite:
        _sm = ", ".join(probe.submodels) or "model_index.json (no root model_type)"
        probe.flags.append(
            f"composite repo — {len(probe.submodels)} submodel(s) [{_sm}]; bring up per subfolder, not one root model"
        )

    model_type_category = _category_from_model_type(str(cfg.get("model_type", "")))
    if model_type_category and probe.category in {"LLM", "VLM"} and model_type_category != probe.category:
        # Between LLM and VLM, VLM (the vision-inclusive superset) WINS in either direction: a
        # visual-question-answering / image-text-to-text tag reveals vision even when model_type is
        # the text backbone (llama-based VLM), and a VLM model_type reveals vision under a plain
        # text-generation tag. Never erase the vision signal by demoting VLM->LLM.
        _reconciled = "VLM" if "VLM" in {probe.category, model_type_category} else model_type_category
        if _reconciled != probe.category:
            probe.flags.append(
                f"Reclassified from {probe.category} to {_reconciled} (LLM/VLM reconcile; VLM wins) "
                f"via config.model_type={cfg.get('model_type')!r}"
            )
            probe.category = _reconciled
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
    # PRIMARY + VERIFIED: the Claude Code agent (3-vote majority, TT_HW_PLANNER_AGENT_VOTES) decides
    # EVERY model by reasoning over its card+config -- including vision models. The majority vote
    # removes the single-run flakiness that once mis-called a VLM, so no unverified
    # authoritative fact override is needed; the agent's own answer is the (self-consistent) verdict.
    # The deterministic facts below are the OFFLINE FALLBACK, used only if the agent is unavailable.
    _agent_cat = _agent_classify_category(model_id, cfg, _fetch_model_card_text(model_id))
    if _agent_cat:
        if _agent_cat != probe.category:
            probe.flags.append(f"Category {probe.category!r} -> {_agent_cat!r} by CC agent (3-vote, card+config)")
            probe.category = _agent_cat
    else:
        # agent unavailable -> deterministic facts (offline degrade-loud).
        if _has_generative_vlm_fact(cfg):
            probe.category = "VLM"
        else:
            _resid = _is_category_residual(model_type_category, _fpr)
            if _resid and _is_dual_encoder_contrastive(cfg) and probe.category in {"Unknown", "Embed"}:
                if probe.category != "Embed":
                    probe.flags.append(
                        "Category -> 'Embed' via dual-encoder contrastive fact (text_config + vision/audio_config)"
                    )
                    probe.category = "Embed"

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
