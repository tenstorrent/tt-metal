from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


DEFAULT_TEMPLATE_PYTEST_EXCLUDE_K = (
    "not test_perf_device " "and not test_e2e_performant " "and not benchmark " "and not test_perf " "and not stress"
)


@dataclass
class FamilyBackend:
    category: str
    name: str
    demo_path: str
    routing_mode: str
    canonical_hf_id: Optional[str]
    notes: str = ""
    model_type_keys: List[str] = field(default_factory=list)
    pipeline_tags: List[str] = field(default_factory=list)
    smoke_test_entry: Optional[str] = None
    pytest_exclude_k_override: Optional[str] = None

    use_module_tree: bool = False

    def effective_pytest_exclude_k(self) -> str:
        return self.pytest_exclude_k_override or DEFAULT_TEMPLATE_PYTEST_EXCLUDE_K


_BACKENDS: List[FamilyBackend] = [
    FamilyBackend(
        category="LLM",
        name="tt_transformers / simple_text_demo",
        demo_path="models/tt_transformers/demo/simple_text_demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Generic text LLM backend; accepts HF_MODEL for any supported text-causal-LM.",
        model_type_keys=["gemma2", "phi3"],
    ),
    FamilyBackend(
        category="VLM",
        name="tt_transformers / simple_text_demo (multimodal)",
        demo_path="models/tt_transformers/demo/simple_text_demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Generic VLM backend for the Qwen-VL / Gemma3 / Mistral-3.1 lineage.",
        model_type_keys=["gemma3"],
    ),
    FamilyBackend(
        category="STT",
        name="Whisper (distil-large-v3)",
        demo_path="models/demos/audio/whisper",
        routing_mode="template",
        canonical_hf_id="distil-whisper/distil-large-v3",
        notes="Wired for distil-large-v3. Use it directly for that model; "
        "adapt for other Whisper variants or other STT families.",
        model_type_keys=["whisper"],
        pipeline_tags=["automatic-speech-recognition"],
        smoke_test_entry="models/demos/audio/whisper/demo/demo.py",
    ),
    FamilyBackend(
        category="Image",
        name="Stable Diffusion 1.4",
        demo_path="models/demos/vision/generative/stable_diffusion",
        routing_mode="template",
        canonical_hf_id="CompVis/stable-diffusion-v1-4",
        notes="Wired for SD 1.4. Closest template for text-to-image / diffusion models.",
        model_type_keys=["stable_diffusion", "unet", "diffusion"],
        pipeline_tags=["text-to-image", "image-to-image"],
        smoke_test_entry="models/demos/vision/generative/stable_diffusion/wormhole/tests/test_demo.py",
    ),
    FamilyBackend(
        category="CNN",
        name="ViT-base classification",
        demo_path="models/demos/vision/classification/vit",
        routing_mode="template",
        canonical_hf_id="google/vit-base-patch16-224",
        notes="Closest template for ViT / transformer-based image classifiers.",
        model_type_keys=["vit", "beit", "deit", "swin", "convnext"],
        pipeline_tags=["image-classification", "zero-shot-image-classification"],
        smoke_test_entry="models/demos/vision/classification/vit/blackhole/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py",
    ),
    FamilyBackend(
        category="CNN",
        name="ResNet-50 classification",
        demo_path="models/demos/vision/classification/resnet50",
        routing_mode="template",
        canonical_hf_id="microsoft/resnet-50",
        notes="Closest template for convolutional image classifiers (ResNet / " "MobileNet / EfficientNet families).",
        model_type_keys=["resnet", "mobilenet_v1", "mobilenet_v2", "efficientnet"],
        pipeline_tags=["image-classification"],
        smoke_test_entry="models/demos/vision/classification/resnet50/ttnn_resnet/tests/test_demo.py",
    ),
    FamilyBackend(
        category="CNN",
        name="SegFormer (semantic segmentation)",
        demo_path="models/demos/vision/segmentation/segformer",
        routing_mode="template",
        canonical_hf_id="nvidia/segformer-b0-finetuned-ade-512-512",
        notes="Closest template for segmentation / mask-generation models "
        "(SegFormer / MaskFormer / SAM / DETR families). Architecture-specific "
        "adaptation required for non-SegFormer encoders/decoders.",
        model_type_keys=[
            "segformer",
            "maskformer",
            "mask2former",
            "sam",
            "sam2",
            "sam_hiera",
            "detr",
            "deformable_detr",
            "yolos",
            "upernet",
        ],
        pipeline_tags=[
            "image-segmentation",
            "mask-generation",
            "object-detection",
            "zero-shot-object-detection",
        ],
        smoke_test_entry="models/demos/vision/segmentation/segformer/tests/pcc/test_segformer_for_semantic_segmentation.py",
    ),
    FamilyBackend(
        category="CNN",
        name="OWL-ViT (zero-shot object detection)",
        demo_path="models/demos/wormhole/owl_vit",
        routing_mode="template",
        canonical_hf_id="google/owlvit-base-patch32",
        notes="Vision-text grounded detection backend.",
        model_type_keys=["owlvit", "clip", "siglip"],
        pipeline_tags=["zero-shot-object-detection"],
    ),
    FamilyBackend(
        category="Embed",
        name="Sentence-BERT (bert-base)",
        demo_path="models/demos/wormhole/sentence_bert",
        routing_mode="template",
        canonical_hf_id="sentence-transformers/all-MiniLM-L6-v2",
        notes="Closest template for sentence-embedding models. "
        "Backbone is bert-base; adapt for other encoder families. "
        "Decoder-only embedding models (Qwen3-Embedding) have their own "
        "backend entry below; do not add their model_type here.",
        model_type_keys=["bert", "distilbert", "roberta"],
        pipeline_tags=["feature-extraction", "sentence-similarity"],
    ),
    FamilyBackend(
        category="NLP",
        name="BERT-Large (general NLP)",
        demo_path="models/demos/metal_BERT_large_11",
        routing_mode="template",
        canonical_hf_id="bert-large-uncased",
        notes="Closest template for masked-LM / classification BERT variants.",
        model_type_keys=["bert", "distilbert", "roberta", "electra"],
        pipeline_tags=["fill-mask", "text-classification", "token-classification"],
        smoke_test_entry="models/demos/metal_BERT_large_11/tests/test_demo.py",
    ),
    FamilyBackend(
        category="STT",
        name="hf_eager universal (STT)",
        demo_path="models/demos/hf_eager/demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Generic STT/audio backend; CPU eager runner. Use as fallback when no exact template + auto-onboard match exists.",
    ),
    FamilyBackend(
        category="TTS",
        name="hf_eager universal (TTS)",
        demo_path="models/demos/hf_eager/demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Generic TTS backend; CPU eager runner. There is no template-routing TTS backend today, so this is the only path.",
    ),
    FamilyBackend(
        category="Image",
        name="hf_eager universal (Image / diffusion)",
        demo_path="models/demos/hf_eager/demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Generic image-generation backend; CPU eager runner. Falls back here when neither SD template nor a drafted backend matches.",
    ),
    FamilyBackend(
        category="CNN",
        name="hf_eager universal (vision)",
        demo_path="models/demos/hf_eager/demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Generic vision backend; CPU eager runner. Falls back here when none of ViT/ResNet/SegFormer/OWL-ViT templates match exactly and auto-onboard didn't draft one.",
    ),
    FamilyBackend(
        category="Embed",
        name="hf_eager universal (embeddings)",
        demo_path="models/demos/hf_eager/demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Generic embeddings backend; CPU eager runner. Falls back here when Sentence-BERT template doesn't apply.",
    ),
    FamilyBackend(
        category="NLP",
        name="hf_eager universal (NLP)",
        demo_path="models/demos/hf_eager/demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Generic NLP backend; CPU eager runner. Falls back here when BERT-Large template doesn't apply.",
    ),
    FamilyBackend(
        category="Video",
        name="hf_eager universal (Video)",
        demo_path="models/demos/hf_eager/demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Generic video backend; CPU eager runner. There is no template-routing Video backend today, so this is the only path.",
    ),
    FamilyBackend(
        category="Unknown",
        name="hf_eager universal (catch-all)",
        demo_path="models/demos/hf_eager/demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
        notes="Last-resort catch-all when the probe can't classify the category. Runs `AutoModel.from_pretrained` + a synthetic forward on CPU.",
    ),
    FamilyBackend(
        category="Embed",
        name="Qwen3-Embedding (qwen3 decoder)",
        demo_path="models/demos/qwen3_embedding/",
        routing_mode="template",
        canonical_hf_id="Qwen/Qwen3-Embedding-8B",
        notes="Decoder-only transformer (Qwen3 arch) repurposed for dense text embeddings via last-token pooling. Uses RMSNorm, GQA with q_norm/k_norm, SwiGLU MLP, RoPE. Routed to embedding_demo.py which encodes texts via the HF reference and last-token-pools; the tt_transformers components (Attention, MLP, RMSNorm, RoPE) are resolved via reuse_registry derived from BUILDING_BLOCKS.",
        model_type_keys=["qwen3"],
        pipeline_tags=["feature-extraction", "sentence-similarity"],
        smoke_test_entry=None,
        use_module_tree=True,
    ),
    FamilyBackend(
        category="Embed",
        name="Qwen3-Embedding-8B (qwen3 decoder, 8B)",
        demo_path="models/demos/qwen3_embedding_8b",
        routing_mode="template",
        canonical_hf_id="Qwen/Qwen3-Embedding-8B",
        notes="Qwen3-Embedding-8B is a large dense retrieval/embedding model from the Qwen3 family. It uses the standard Qwen3 decoder-only transformer architecture (RMSNorm, GQA, RoPE, SwiGLU MLP) with a mean-pooling head for sentence/document embeddings. Pipeline tag is feature-extraction; suited for semantic search and RAG use cases.",
        model_type_keys=["qwen3"],
        pipeline_tags=["feature-extraction", "sentence-similarity"],
        smoke_test_entry=None,
        use_module_tree=True,
    ),
    FamilyBackend(
        category="VLM",
        name="Mistral-Small-3.1 (mistral3 VLM)",
        demo_path="models/tt_transformers/demo/simple_text_demo.py",
        routing_mode="template",
        canonical_hf_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        notes="Mistral Small 3.1 24B multimodal instruction model. Text path runs via tt_transformers/simple_text_demo.py with HF_MODEL set; vision path is not yet ported.",
        model_type_keys=["mistral3"],
        pipeline_tags=["image-text-to-text"],
        smoke_test_entry="models/tt_transformers/demo/simple_text_demo.py",
        use_module_tree=False,
    ),
    FamilyBackend(
        category="STT",
        name="XTTS-v2 (multilingual TTS)",
        demo_path="models/demos/xtts_v2",
        routing_mode="template",
        canonical_hf_id="/local/ttuser/apande/models/XTTS-v2-hf",
        notes="Coqui XTTS-v2: multilingual zero-shot text-to-speech model. GPT-2 autoregressive backbone generates prosodic audio tokens from text; a HiFi-GAN vocoder decodes tokens to waveforms; a ResNet-based speaker encoder enables voice cloning. Supports 17 languages. Architecture is all-NEW (no existing tt-metal demo to clone).",
        model_type_keys=["xtts"],
        pipeline_tags=["text-to-speech"],
        smoke_test_entry=None,
        use_module_tree=True,
    ),
]


_BY_CATEGORY: Dict[str, List[FamilyBackend]] = {}
for _b in _BACKENDS:
    _BY_CATEGORY.setdefault(_b.category, []).append(_b)


_OVERLAY_BACKENDS_CACHE: Optional[List[FamilyBackend]] = None


def _overlay_backends() -> List[FamilyBackend]:
    """Family backends declared by upstream modules via the generated overlay
    (fixes-plan Point 2a). Pure supplement: a family whose ``name`` OR any
    ``model_type_keys`` already exists in the static ``_BACKENDS`` is dropped so
    the hand-written list always wins. Empty until a module self-declares a
    ``TT_HW_PLANNER_FAMILY`` marker upstream. Cached; never raises."""
    global _OVERLAY_BACKENDS_CACHE
    if _OVERLAY_BACKENDS_CACHE is not None:
        return _OVERLAY_BACKENDS_CACHE
    out: List[FamilyBackend] = []
    try:
        from .registry_sync import load_generated_overlay

        static_names = {b.name for b in _BACKENDS}
        static_mt = {k.lower() for b in _BACKENDS for k in b.model_type_keys}
        for m in load_generated_overlay().get("families", []):
            name = m.get("name") or m.get("concept")
            cat = m.get("category")
            demo = m.get("demo_path") or m.get("tt_path")
            if not (name and cat and demo) or name in static_names:
                continue
            mkeys = [str(k).lower() for k in (m.get("model_type_keys") or [])]
            if any(k in static_mt for k in mkeys):
                continue
            out.append(
                FamilyBackend(
                    category=cat,
                    name=name,
                    demo_path=demo,
                    routing_mode=m.get("routing_mode", "template"),
                    canonical_hf_id=m.get("canonical_hf_id"),
                    notes=m.get("notes", "auto-registered from upstream TT_HW_PLANNER_FAMILY marker"),
                    model_type_keys=mkeys,
                    pipeline_tags=[str(t).lower() for t in (m.get("pipeline_tags") or [])],
                )
            )
    except Exception:
        out = []
    _OVERLAY_BACKENDS_CACHE = out
    return out


def all_backends() -> List[FamilyBackend]:
    return list(_BACKENDS) + _overlay_backends()


def backends_for_category(category: str) -> List[FamilyBackend]:
    static = list(_BY_CATEGORY.get(category, []))
    return static + [b for b in _overlay_backends() if b.category == category]


def pick_backend(
    *,
    category: str,
    model_type: Optional[str] = None,
    pipeline_tag: Optional[str] = None,
) -> Optional[FamilyBackend]:
    """Legacy single-return entry point. Prefer `pick_backend_with_quality`
    for code that needs to distinguish exact / pipeline-tag / category-
    default fallback (the latter is silent and historically caused the
    SAM2-onto-SegFormer-template confusion). Kept for back-compat with
    older callers that don't inspect match quality."""
    backend, _quality = pick_backend_with_quality(category=category, model_type=model_type, pipeline_tag=pipeline_tag)
    return backend


def pick_backend_with_quality(
    *,
    category: str,
    model_type: Optional[str] = None,
    pipeline_tag: Optional[str] = None,
) -> Tuple[Optional[FamilyBackend], str]:
    """Pick a backend AND report how confident the match is.

    Returns ``(backend, quality)`` where ``quality`` is one of:
      - ``"exact"``            -- model_type matched a backend's
                                  `model_type_keys`. Most reliable.
      - ``"pipeline"``         -- model_type missed but pipeline_tag
                                  matched a backend's `pipeline_tags`.
                                  Reasonably reliable; HF pipeline tags
                                  are arch-agnostic on purpose.
      - ``"category-default"`` -- both missed; we fell back to the FIRST
                                  registered backend for this category.
                                  HISTORICALLY THIS IS HOW NEW
                                  ARCHITECTURES SILENTLY GOT MAPPED TO
                                  THE WRONG TEMPLATE (e.g. unknown CNN
                                  -> ViT-base, unknown segmentation ->
                                  SegFormer). Callers driving an
                                  LLM-iterate loop should TREAT THIS AS
                                  AN ERROR unless the user explicitly
                                  opts in.
      - ``"none"``             -- no backend at all (no category match).
    """
    mt = (model_type or "").lower()
    pt = (pipeline_tag or "").lower()
    candidates = backends_for_category(category)

    for b in candidates:
        if mt and mt in {k.lower() for k in b.model_type_keys}:
            return (b, "exact")
    for b in candidates:
        if pt and pt in {t.lower() for t in b.pipeline_tags}:
            return (b, "pipeline")

    if mt:
        for b in all_backends():
            if mt in {k.lower() for k in b.model_type_keys}:
                return (b, "exact")
    if pt:
        for b in all_backends():
            if pt in {t.lower() for t in b.pipeline_tags}:
                return (b, "pipeline")

    if candidates:
        for b in candidates:
            if getattr(b, "routing_mode", "") == "generic":
                return (b, "category-default")
        return (candidates[0], "category-default")
    return (None, "none")


def rank_backends(
    *,
    category: str,
    model_type: Optional[str] = None,
    pipeline_tag: Optional[str] = None,
    top_n: Optional[int] = 3,
) -> List[Tuple[FamilyBackend, int, str]]:
    """Rank candidate family backends best-first with a match score + reason.

    Where :func:`pick_backend_with_quality` returns ONE locked pick, this exposes
    the runners-up so scaffold can surface the top-N sibling candidates (and
    compose per-component reuse across them) instead of one silent choice that
    historically mapped a model onto the wrong family's tree (fixes-plan Point 2b).

    Score: exact model_type = 100 (90 cross-category), pipeline_tag = 70 (60
    cross-category), same-category generic default = 40, same-category other = 30.
    Returns the top ``top_n`` (all if ``top_n`` is falsy), highest score first.
    """
    mt = (model_type or "").lower()
    pt = (pipeline_tag or "").lower()
    ranked: List[Tuple[FamilyBackend, int, str]] = []
    for b in all_backends():
        same_cat = b.category == category
        mkeys = {k.lower() for k in b.model_type_keys}
        pkeys = {t.lower() for t in b.pipeline_tags}
        if mt and mt in mkeys:
            score = 100 if same_cat else 90
            reason = f"exact model_type '{mt}'" + ("" if same_cat else f" (cross-category {b.category})")
        elif pt and pt in pkeys:
            score = 70 if same_cat else 60
            reason = f"pipeline_tag '{pt}'" + ("" if same_cat else f" (cross-category {b.category})")
        elif same_cat:
            generic = getattr(b, "routing_mode", "") == "generic"
            score = 40 if generic else 30
            reason = f"category '{category}' default" + (" (generic runner)" if generic else "")
        else:
            continue
        ranked.append((b, score, reason))
    ranked.sort(key=lambda x: (-x[1], x[0].name))
    return ranked[:top_n] if top_n else ranked


def canonical_hf_ids() -> List[str]:
    return [b.canonical_hf_id for b in _BACKENDS if b.canonical_hf_id]
