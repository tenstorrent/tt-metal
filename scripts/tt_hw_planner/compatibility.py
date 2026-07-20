from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class Status(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    UNNEEDED = "UNNEEDED"


class Effort(str, Enum):
    DROP_IN = "drop-in"
    LIGHT = "light port (days)"
    HEAVY = "heavy port (weeks)"
    NEW = "new building block (months)"
    NONE = "n/a"


@dataclass
class BuildingBlock:
    name: str
    description: str
    needed_when: Callable[[dict], bool]
    tt_path: Optional[str]
    status_when_needed: Status
    effort_when_needed: Effort
    notes: str = ""
    class_name_pattern: Optional[str] = None
    tt_class: Optional[str] = None
    model_type_keys: Optional[Tuple[str, ...]] = None
    registry_tt_path: Optional[str] = None


@dataclass
class CheckResult:
    block: BuildingBlock
    needed: bool
    status: Status
    effort: Effort
    notes: str


@dataclass
class CompatReport:
    model_id: str
    architecture_family: str
    similar_supported_model: Optional[str]
    results: List[CheckResult] = field(default_factory=list)
    overall: str = "UNKNOWN"
    effort_summary: str = ""
    discovery: object = None

    def by_status(self, status: Status) -> List[CheckResult]:
        return [r for r in self.results if r.status == status and r.needed]

    @property
    def in_tt_transformers(self) -> bool:
        return bool(getattr(self.discovery, "in_tt_transformers", False))

    @property
    def in_external_demo(self) -> bool:
        return bool(getattr(self.discovery, "in_external_demo", False))

    @property
    def primary_demo(self):
        return getattr(self.discovery, "primary_demo", None)


LLAMA_FAMILY_MODEL_TYPES = {
    "llama",
    "qwen2",
    "qwen3",
    "qwen2_moe",
    "qwen3_moe",
    "mistral",
    "phi3",
    "phi4",
    "gemma",
    "gemma2",
    "gemma3",
    "mixtral",
    "falcon",
}

MLA_MODEL_TYPES = {"deepseek_v2", "deepseek_v3", "deepseek_v4"}
SSM_MODEL_TYPES = {"mamba", "mamba2", "rwkv", "rwkv5", "rwkv6"}
VLM_MODEL_TYPES = {
    "qwen2_vl",
    "qwen3_vl",
    "qwen2_5_vl",
    "mllama",
    "llama4",
    "mistral3",
    "pixtral",
    "gemma3",
    "gemma4",
    "phi3_v",
    "phi4_multimodal",
}


def detect_family(cfg: dict) -> str:
    mt = (cfg.get("model_type") or "").lower()
    if mt in MLA_MODEL_TYPES:
        return "MLA (DeepSeek-style)"
    if mt in SSM_MODEL_TYPES:
        return "SSM (state-space)"
    if mt in VLM_MODEL_TYPES or cfg.get("vision_config"):
        return "VLM (vision-language)"
    if _is_moe(cfg):
        return "MoE (mixture-of-experts)"
    if mt in LLAMA_FAMILY_MODEL_TYPES:
        return "Llama-family causal LM"
    return f"unknown ({mt or 'no model_type'})"


def _minted_category(cfg: dict) -> str:
    """Honest NEW-category name for an unrecognized model, from the config in
    priority order: model_type -> architectures[0]. Used to LABEL the report
    (traceable, dedup-able) instead of silently characterizing an unknown model
    as a confident LLM. (pipeline_tag, the first choice in the plan, is hub
    metadata not present in the config passed here.)
    """
    mt = str(cfg.get("model_type") or "").strip()
    if mt:
        return mt
    archs = cfg.get("architectures") or []
    if isinstance(archs, (list, tuple)) and archs:
        return str(archs[0])
    return "unrecognized"


SUPPORTED_HF_MODELS = {
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-11B-Vision",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.2-90B-Vision",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-Embedding-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "Qwen/QwQ-32B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-4",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-27b-it",
    "google/medgemma-4b-it",
    "google/medgemma-27b-text-it",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",
    "state-spaces/mamba-2.8b-slimpj",
    "distil-whisper/distil-large-v3",
    "openai/whisper-large-v3",
    "google/vit-base-patch16-224",
    "microsoft/resnet-50",
    "google/mobilenet_v2_1.0_224",
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "CompVis/stable-diffusion-v1-4",
    "google/owlvit-base-patch32",
    "bert-large-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
}


def closest_supported_model(model_id: str, cfg: dict) -> Optional[str]:
    if model_id in SUPPORTED_HF_MODELS:
        return model_id

    mt = (cfg.get("model_type") or "").lower()
    arches = [a.lower() for a in cfg.get("architectures") or []]

    candidates = {
        "qwen2": "Qwen/Qwen2.5-32B",
        "qwen3": "Qwen/Qwen3-32B",
        "qwen2_moe": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "qwen3_moe": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama": "meta-llama/Llama-3.1-8B",
        "llama4": "meta-llama/Llama-3.1-8B",
        "olmo": "meta-llama/Llama-3.1-8B",
        "olmo2": "meta-llama/Llama-3.1-8B",
        "cohere": "meta-llama/Llama-3.1-8B",
        "cohere2": "meta-llama/Llama-3.1-8B",
        "granite": "meta-llama/Llama-3.1-8B",
        "granitemoe": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "internlm": "meta-llama/Llama-3.1-8B",
        "internlm2": "meta-llama/Llama-3.1-8B",
        "internlm3": "meta-llama/Llama-3.1-8B",
        "starcoder2": "meta-llama/Llama-3.1-8B",
        "ministral": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral3": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "phi3": "microsoft/Phi-3.5-mini-instruct",
        "phi4": "microsoft/Phi-4",
        "gemma": "google/gemma-3-27b-it",
        "gemma2": "google/gemma-3-27b-it",
        "gemma3": "google/gemma-3-27b-it",
        "gemma3_text": "google/gemma-3-27b-it",
        "falcon": "tiiuae/falcon-7b-instruct",
        "mllama": "meta-llama/Llama-3.2-11B-Vision",
        "qwen2_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen3_vl": "Qwen/Qwen3-VL-32B-Instruct",
        "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
        "mamba": "state-spaces/mamba-2.8b-slimpj",
        "mamba2": "state-spaces/mamba-2.8b-slimpj",
        "vit": "google/vit-base-patch16-224",
        "beit": "google/vit-base-patch16-224",
        "deit": "google/vit-base-patch16-224",
        "swin": "google/vit-base-patch16-224",
        "convnext": "google/vit-base-patch16-224",
        "resnet": "microsoft/resnet-50",
        "mobilenet_v1": "microsoft/resnet-50",
        "mobilenet_v2": "google/mobilenet_v2_1.0_224",
        "efficientnet": "microsoft/resnet-50",
        "segformer": "nvidia/segformer-b0-finetuned-ade-512-512",
        "maskformer": "nvidia/segformer-b0-finetuned-ade-512-512",
        "mask2former": "nvidia/segformer-b0-finetuned-ade-512-512",
        "sam": "nvidia/segformer-b0-finetuned-ade-512-512",
        "sam2": "nvidia/segformer-b0-finetuned-ade-512-512",
        "sam_hiera": "nvidia/segformer-b0-finetuned-ade-512-512",
        "detr": "nvidia/segformer-b0-finetuned-ade-512-512",
        "deformable_detr": "nvidia/segformer-b0-finetuned-ade-512-512",
        "yolos": "nvidia/segformer-b0-finetuned-ade-512-512",
        "upernet": "nvidia/segformer-b0-finetuned-ade-512-512",
        "owlvit": "google/owlvit-base-patch32",
        "clip": "google/owlvit-base-patch32",
        "siglip": "google/owlvit-base-patch32",
        "stable_diffusion": "CompVis/stable-diffusion-v1-4",
        "unet": "CompVis/stable-diffusion-v1-4",
        "whisper": "distil-whisper/distil-large-v3",
        "bert": "bert-large-uncased",
        "distilbert": "bert-large-uncased",
        "roberta": "bert-large-uncased",
        "electra": "bert-large-uncased",
        "brand_new_xyz": "test/never-going-to-exist",
    }
    if mt in candidates:
        return candidates[mt]
    for a in arches:
        for key, val in candidates.items():
            if key in a:
                return val
    return None


def _text_config(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def _is_moe(cfg: dict) -> bool:
    t = _text_config(cfg)
    return bool(
        t.get("num_local_experts")
        or t.get("n_routed_experts")
        or t.get("num_experts")
        or t.get("moe_intermediate_size")
    )


def _is_mla(cfg: dict) -> bool:
    """True iff the model uses Multi-head Latent Attention (DeepSeek-
    family). Detected by any of the four MLA-specific config keys:
    ``kv_lora_rank``, ``q_lora_rank``, ``qk_rope_head_dim``,
    ``qk_nope_head_dim``. Reads from the text_config when present so
    multimodal variants still classify correctly.
    """
    t = _text_config(cfg)
    return bool(t.get("kv_lora_rank") or t.get("q_lora_rank") or t.get("qk_rope_head_dim") or t.get("qk_nope_head_dim"))


def _is_sliding(cfg: dict) -> bool:
    t = _text_config(cfg)
    return bool(t.get("sliding_window")) or bool(t.get("layer_types"))


def _is_ssm(cfg: dict) -> bool:
    mt = (cfg.get("model_type") or "").lower()
    return mt in SSM_MODEL_TYPES


def _has_vision(cfg: dict) -> bool:
    return bool(cfg.get("vision_config")) or (cfg.get("model_type") or "").lower() in VLM_MODEL_TYPES


def _has_cross_attn(cfg: dict) -> bool:
    t = _text_config(cfg)
    return bool(t.get("cross_attention_layers"))


def _attn_grouping(cfg: dict) -> str:
    t = _text_config(cfg)
    nh = t.get("num_attention_heads") or 0
    nkv = t.get("num_key_value_heads") or nh
    if nh == 0:
        return "unknown"
    if nkv == 1:
        return "MQA"
    if nkv < nh:
        return "GQA"
    return "MHA"


def _rope_scaling_type(cfg: dict) -> Optional[str]:
    """2026-05-23 audit bug #9: also read the newer `rope_parameters`
    field used by transformers 5.x (Phi-3.5, etc.). Previously only
    checked `rope_scaling`, so a model migrated to `rope_parameters`
    would silently report "Standard RoPE [SUPPORTED]" + READY here
    while the runtime might silently drop scaling. The
    kernel_constraints check already warns about this case; this
    aligns the compat building-block view."""
    t = _text_config(cfg)
    rs = t.get("rope_scaling")
    if isinstance(rs, dict):
        v = (rs.get("type") or rs.get("rope_type") or "").lower()
        if v:
            return v
    rp = t.get("rope_parameters")
    if isinstance(rp, dict):
        v = (rp.get("type") or rp.get("rope_type") or "").lower()
        if v:
            return v
    return None


def _hidden_act(cfg: dict) -> str:
    t = _text_config(cfg)
    return (t.get("hidden_act") or t.get("hidden_activation") or "silu").lower()


def _tied_embeddings(cfg: dict) -> bool:
    t = _text_config(cfg)
    return bool(t.get("tie_word_embeddings"))


BUILDING_BLOCKS: List[BuildingBlock] = [
    BuildingBlock(
        name="Token embedding",
        description="Embedding lookup (input ids -> hidden states)",
        needed_when=lambda c: True,
        tt_path="models/tt_transformers/tt/embedding.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="BF16 weights only; ScaledEmbedding variant for Gemma is included.",
    ),
    BuildingBlock(
        name="MHA attention",
        description="Standard multi-head attention",
        needed_when=lambda c: _attn_grouping(c) == "MHA" and not _is_mla(c),
        tt_path="models/tt_transformers/tt/attention.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
    ),
    BuildingBlock(
        name="GQA attention",
        description="Grouped-query attention (num_kv_heads < num_heads)",
        needed_when=lambda c: _attn_grouping(c) == "GQA" and not _is_mla(c),
        tt_path="models/tt_transformers/tt/attention.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="Requires num_attention_heads % num_key_value_heads == 0.",
        class_name_pattern=r".*Attention$",
        tt_class="Attention",
    ),
    BuildingBlock(
        name="MQA attention",
        description="Multi-query attention (single KV head)",
        needed_when=lambda c: _attn_grouping(c) == "MQA" and not _is_mla(c),
        tt_path="models/tt_transformers/tt/attention.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
    ),
    BuildingBlock(
        name="MLA attention",
        description="Multi-head latent attention with compressed KV (DeepSeek-V2/V3)",
        needed_when=_is_mla,
        tt_path="models/demos/deepseek_v3/tt/mla/",
        status_when_needed=Status.PARTIAL,
        effort_when_needed=Effort.HEAVY,
        notes=(
            "Reference impl in demo only; no path in shared tt_transformers "
            "library. Adapting to other MLA models requires lifting the demo "
            "modules into a reusable form."
        ),
    ),
    BuildingBlock(
        name="Q/K RMSNorm",
        description="Per-head Q/K normalization (Qwen3, Phi-4, Olmo2, ...)",
        needed_when=lambda c: (
            (c.get("model_type") or "").lower().startswith(("qwen3", "phi4", "olmo2", "olmoe"))
            or (
                isinstance(c.get("text_config"), dict)
                and (c["text_config"].get("model_type") or "").lower().startswith(("qwen3", "phi4", "olmo2", "olmoe"))
            )
        ),
        tt_path="models/tt_transformers/tt/attention.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="Auto-detected from q_norm / k_norm tensor names in the checkpoint.",
    ),
    BuildingBlock(
        name="Sliding-window attention",
        description="Local-window attention for some/all layers",
        needed_when=_is_sliding,
        tt_path="models/tt_transformers/tt/attention.py",
        status_when_needed=Status.PARTIAL,
        effort_when_needed=Effort.LIGHT,
        notes=(
            "Supported for prefill + decode. NOT supported in combination "
            "with chunked prefill -- raises NotImplementedError. Hybrid "
            "full/sliding patterns work via layer_types config field."
        ),
    ),
    BuildingBlock(
        name="Standard RoPE",
        description="Rotary positional embedding",
        needed_when=lambda c: not _is_ssm(c) and _rope_scaling_type(c) in (None, "default"),
        tt_path="models/tt_transformers/tt/rope.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        class_name_pattern=r".*RotaryEmbedding$",
        tt_class="RotaryEmbedding",
    ),
    BuildingBlock(
        name="Llama-3 RoPE scaling",
        description="Frequency rescaling used by Llama-3.1+",
        needed_when=lambda c: _rope_scaling_type(c) == "llama3",
        tt_path="models/tt_transformers/tt/rope.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
    ),
    BuildingBlock(
        name="YaRN RoPE scaling",
        description="YaRN long-context extrapolation",
        needed_when=lambda c: _rope_scaling_type(c) == "yarn",
        tt_path="models/tt_transformers/tt/rope.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
    ),
    BuildingBlock(
        name="LongRoPE (Phi-3)",
        description="Phi-3 long-context RoPE scaling",
        needed_when=lambda c: _rope_scaling_type(c) == "longrope",
        tt_path="models/tt_transformers/tt/rope.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
    ),
    BuildingBlock(
        name="mRoPE",
        description="Multimodal RoPE used by some VL models",
        needed_when=lambda c: _rope_scaling_type(c) == "mrope",
        tt_path=None,
        status_when_needed=Status.MISSING,
        effort_when_needed=Effort.HEAVY,
        notes=(
            "Code path currently warns and drops the scaling - inference "
            "may diverge from HF reference. Needs a real kernel."
        ),
    ),
    BuildingBlock(
        name="ALiBi positional bias",
        description="Linear bias positional encoding (Falcon-1, MPT)",
        needed_when=lambda c: bool(_text_config(c).get("alibi_bias_max")),
        tt_path=None,
        status_when_needed=Status.MISSING,
        effort_when_needed=Effort.HEAVY,
    ),
    BuildingBlock(
        name="RMSNorm (text)",
        description="Root-mean-square normalization between blocks",
        needed_when=lambda c: not _is_ssm(c),
        tt_path="models/tt_transformers/tt/distributed_norm.py (wraps models/common/rmsnorm.py)",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip.",
        class_name_pattern=r".*RMSNorm$",
        tt_class="RMSNorm",
        registry_tt_path="models/common/rmsnorm.py",
    ),
    BuildingBlock(
        name="Extra Gemma-style norms",
        description="pre_/post_feedforward_layernorm (Gemma 2/3/4)",
        needed_when=lambda c: (c.get("model_type") or "").lower().startswith("gemma"),
        tt_path="models/tt_transformers/tt/decoder.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="Auto-detected from checkpoint keys; activates only if present.",
    ),
    BuildingBlock(
        name="SwiGLU MLP",
        description="Gate/up/down projections with SiLU gating",
        needed_when=lambda c: _hidden_act(c) in ("silu", "swiglu", "gelu_pytorch_tanh")
        and not _is_moe(c)
        and not _is_ssm(c),
        tt_path="models/tt_transformers/tt/mlp.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh.",
        class_name_pattern=r".*MLP$",
        tt_class="MLP",
    ),
    BuildingBlock(
        name="MoE routing (Mixtral-style)",
        description="Top-k expert routing + weighted combine",
        needed_when=lambda c: _is_moe(c) and not _is_mla(c),
        tt_path="models/tt_transformers/tt/mixtral_moe.py",
        status_when_needed=Status.PARTIAL,
        effort_when_needed=Effort.LIGHT,
        notes=(
            "Generic MoE block currently hard-codes num_devices=8 and top-2. "
            "Adapting to other top-k or device counts needs a small refactor. "
            "Larger-scale MoE (DeepSeek/GPT-OSS) has standalone demos."
        ),
        class_name_pattern=r".*M[Oo][Ee]$",
        tt_class="TtMoeLayer",
    ),
    BuildingBlock(
        name="DeepSeek-style MoE (routed + shared)",
        description="DeepSeek MoE with both routed experts and a shared expert",
        needed_when=lambda c: _is_mla(c) and _is_moe(c),
        tt_path="models/demos/deepseek_v3/tt/moe.py",
        status_when_needed=Status.PARTIAL,
        effort_when_needed=Effort.HEAVY,
        notes="Lives in deepseek_v3 demo only; not generalized for other MLA+MoE models.",
    ),
    BuildingBlock(
        name="SSM / Mamba blocks",
        description="State-space sequence model (no attention)",
        needed_when=_is_ssm,
        tt_path="models/demos/wormhole/mamba/tt/",
        status_when_needed=Status.PARTIAL,
        effort_when_needed=Effort.HEAVY,
        notes="Separate stack from tt_transformers; covers state-spaces/mamba-2.8b-slimpj. Other Mamba variants would need adaptation.",
        class_name_pattern=r".*Mamba\d*Mixer$",
        tt_class="TtMambaSSM",
        registry_tt_path="models/demos/wormhole/mamba/tt/mamba_ssm.py",
    ),
    BuildingBlock(
        name="LM head",
        description="Output projection over vocabulary",
        needed_when=lambda c: True,
        tt_path="models/tt_transformers/tt/lm_head.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="Sharded vocab + all_reduce; pads to padded_vocab_size.",
    ),
    BuildingBlock(
        name="Tied embeddings",
        description="LM head shares weights with token embedding",
        needed_when=_tied_embeddings,
        tt_path="models/tt_transformers/tt/load_checkpoints.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="standardize_hf_keys auto-duplicates embed_tokens -> lm_head.",
    ),
    BuildingBlock(
        name="Vision tower",
        description="Image encoder (ViT-style)",
        needed_when=_has_vision,
        tt_path="models/tt_transformers/tt/multimodal/ + models/demos/{qwen25_vl,qwen3_vl,multimodal/gemma3,mistral_24b}/",
        status_when_needed=Status.PARTIAL,
        effort_when_needed=Effort.LIGHT,
        notes=(
            "Three vision stacks exist: Llama-vision, Pixtral/Mistral, and "
            "Qwen-VL. A new VLM family typically needs adapting one of these "
            "to the model-specific patch / projector layout."
        ),
    ),
    BuildingBlock(
        name="Cross-attention (vision -> text)",
        description="Image-conditioned cross-attention layers in the decoder",
        needed_when=_has_cross_attn,
        tt_path="models/tt_transformers/tt/multimodal/llama_cross_attention.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="Llama-vision style cross-attention; other VLMs use feature concat instead.",
    ),
    BuildingBlock(
        name="Checkpoint key remap",
        description="Mapping HF state-dict keys to TT module fields",
        needed_when=lambda c: True,
        tt_path="models/tt_transformers/tt/load_checkpoints.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes=(
            "convert_hf_to_meta() handles QKV permutation; "
            "_no_qkv_permute() handles HF-RoPE-native checkpoints. "
            "New families may need entries in map_hf_to_meta_keys()."
        ),
    ),
    BuildingBlock(
        name="Tokenizer",
        description="HF tokenizer / processor for tokens and (optional) image input",
        needed_when=lambda c: True,
        tt_path="HF AutoTokenizer.from_pretrained(HF_MODEL)",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="If the HF repo doesn't ship tokenizer files, add an entry to base_model_tokenizer_mapping in model_config.py.",
    ),
    BuildingBlock(
        name="Generator / inference loop",
        description="Prefill + decode orchestration with KV cache",
        needed_when=lambda c: True,
        tt_path="models/tt_transformers/tt/generator.py",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="Chunked prefill, paged KV, tracing, sampling.",
    ),
    BuildingBlock(
        name="Top-k / sampling",
        description="Logits-to-token sampling",
        needed_when=lambda c: True,
        tt_path="ttnn.argmax / ttnn.topk + on-device sampling helpers",
        status_when_needed=Status.SUPPORTED,
        effort_when_needed=Effort.DROP_IN,
        notes="Top-p is composed in host code; multi-core top-k requires k<=64 and power-of-2 dim.",
    ),
]


_OVERALL_FROM_STATUSES = [
    (
        lambda r: r.by_status(Status.MISSING),
        "BLOCKED",
        "Some required building blocks have no TT implementation. Porting "
        "requires writing new kernels / modules from scratch.",
    ),
    (
        lambda r: r.by_status(Status.PARTIAL),
        "FEASIBLE WITH WORK",
        "All needed blocks exist somewhere in tt-metal but not all in the "
        "shared library. Plan on lifting/adapting demo code.",
    ),
    (
        lambda _: [],
        "READY",
        "All required blocks already exist in models/tt_transformers/. Likely "
        "drop-in via the existing porting checklist.",
    ),
]


def _aggregate_overall(report: CompatReport) -> None:
    disc = report.discovery
    is_supported = bool(getattr(disc, "is_supported", False)) or (report.model_id in SUPPORTED_HF_MODELS)
    if is_supported:
        in_external = bool(getattr(disc, "in_external_demo", False))
        in_ttt = bool(getattr(disc, "in_tt_transformers", False))
        demo_path = getattr(disc, "primary_demo", None)
        arch_set = getattr(disc, "arch_compatibility", frozenset())
        arch_tag = (next(iter(arch_set)) + "-only") if (arch_set and len(arch_set) == 1) else None
        if in_external and demo_path is not None:
            tag = f"external demo, {arch_tag}" if arch_tag else "external demo"
            report.overall = f"ALREADY SUPPORTED ({tag})"
            note = (
                f"Supported, but the demo lives outside tt_transformers at "
                f"`{demo_path.as_posix()}`. `prepare` routes there automatically; "
                f"`scaffold` does not apply (its tables only affect tt_transformers)."
            )
            if arch_tag:
                note += (
                    f"  Demo path is restricted to {arch_tag.replace('-only', '')}; "
                    f"`prepare` will refuse to emit a run command for any other arch."
                )
            report.effort_summary = note
        elif in_ttt:
            report.overall = "ALREADY SUPPORTED"
            report.effort_summary = (
                "Driven by tt_transformers/simple_text_demo.py. `prepare` will "
                "build a simple_text_demo invocation with the right MESH_DEVICE."
            )
        else:
            report.overall = "ALREADY SUPPORTED"
            report.effort_summary = (
                "Listed as supported but no demo file in models/ references "
                "this HF id literally. `prepare` will assume the tt_transformers "
                "path; if that fails, the model lives at a non-obvious entry "
                "point and the demo file should be located manually."
            )
        return

    targeted = getattr(disc, "target_entry", None) is not None
    if targeted:
        report.overall = "TARGETED (no demo wired)"
        report.effort_summary = (
            "Tracked in models/model_targets.yaml as a future target, but no "
            "demo or test file references this HF id literally. Architecture "
            "compatibility (below) tells you what would be needed to actually "
            "run it."
        )

    for predicate, label, summary in _OVERALL_FROM_STATUSES:
        if predicate(report):
            if targeted:
                report.effort_summary += f"  Architectural verdict: {label}. {summary}"
            else:
                report.overall = label
                report.effort_summary = summary
            return


def check_compatibility(model_id: str, cfg: dict) -> CompatReport:
    family = detect_family(cfg)
    is_unknown = family.startswith("unknown")
    minted = _minted_category(cfg) if is_unknown else None
    if is_unknown:
        family = f"unknown / NEW category '{minted}' — attempting generic LLM path as fallback"
    closest = closest_supported_model(model_id, cfg)

    results: List[CheckResult] = []
    for blk in BUILDING_BLOCKS:
        needed = bool(blk.needed_when(cfg))
        if not needed:
            results.append(
                CheckResult(
                    block=blk,
                    needed=False,
                    status=Status.UNNEEDED,
                    effort=Effort.NONE,
                    notes="",
                )
            )
            continue
        status = blk.status_when_needed
        notes = blk.notes
        if is_unknown and status == Status.SUPPORTED:
            status = Status.PARTIAL
            notes = (
                f"LLM fallback for unrecognized model (NEW category '{minted}'): assumed by the "
                f"generic decoder path, NOT a confident match — verify against the real architecture. " + (notes or "")
            ).strip()
        results.append(
            CheckResult(
                block=blk,
                needed=True,
                status=status,
                effort=blk.effort_when_needed,
                notes=notes,
            )
        )

    try:
        from .discovery import discover_model

        discovery = discover_model(model_id)
    except Exception:
        discovery = None

    report = CompatReport(
        model_id=model_id,
        architecture_family=family,
        similar_supported_model=closest,
        results=results,
        discovery=discovery,
    )
    _aggregate_overall(report)
    return report
