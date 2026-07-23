# SPDX-License-Identifier: Apache-2.0
"""Structural architecture fingerprint — one representation derived the same way for
the target model AND every registered backend, so sibling routing matches
structure-to-structure (backbone + feature flags) instead of hand-written task
labels / pipeline tags. This is the generalized alternative to per-model patches:
whatever HF signal is present (model_type, architectures class names, or the
instantiated module tree) is folded into a single normalized descriptor, and the
same descriptor is computed for backends at sync time. Pure/heuristic; never raises.
"""
from __future__ import annotations

import re
from typing import List, Optional


_MODEL_TYPE_FAMILY = {
    "llama": ("decoder-only causal LM", []),
    "mistral": ("decoder-only causal LM", []),
    "mixtral": ("decoder-only causal LM", ["MoE"]),
    "qwen2": ("decoder-only causal LM", []),
    "qwen3": ("decoder-only causal LM", []),
    "gemma": ("decoder-only causal LM", []),
    "gemma2": ("decoder-only causal LM", []),
    "phi": ("decoder-only causal LM", []),
    "phi3": ("decoder-only causal LM", []),
    "falcon": ("decoder-only causal LM", []),
    "glm": ("decoder-only causal LM", []),
    "glm4_moe": ("decoder-only causal LM", ["MoE"]),
    "hunyuan": ("decoder-only causal LM", ["MoE"]),
    "hunyuan_image_3": ("decoder-only causal LM", ["MoE", "image-tokens"]),
    "hunyuan_image_3_moe": ("decoder-only causal LM", ["MoE", "image-tokens"]),
    "nemotron_h": ("decoder-only causal LM", ["Mamba/SSM", "MoE"]),
    "voxtral": ("decoder-only causal LM", ["audio"]),
    "emu3": ("decoder-only causal LM", ["image-tokens"]),
    "qwen2_vl": ("VLM (ViT + decoder LM)", []),
    "qwen2_5_vl": ("VLM (ViT + decoder LM)", []),
    "qwen3_vl": ("VLM (ViT + decoder LM)", []),
    "llava": ("VLM (ViT + decoder LM)", []),
    "paligemma": ("VLM (ViT + decoder LM)", []),
    "idefics": ("VLM (ViT + decoder LM)", []),
    "mistral3": ("VLM (ViT + decoder LM)", []),
    "gemma3": ("VLM (ViT + decoder LM)", []),
    "whisper": ("encoder-decoder transformer", ["speech"]),
    "seamless_m4t": ("encoder-decoder transformer", ["conformer", "vocoder"]),
    "seamless_m4t_v2": ("encoder-decoder transformer", ["conformer", "vocoder"]),
    "t5": ("encoder-decoder transformer", []),
    "bart": ("encoder-decoder transformer", []),
    "speecht5": ("encoder-decoder transformer", ["TTS", "vocoder"]),
    "bert": ("encoder-only transformer", []),
    "roberta": ("encoder-only transformer", []),
    "distilbert": ("encoder-only transformer", []),
    "electra": ("encoder-only transformer", []),
    "vit": ("ViT (vision transformer)", []),
    "deit": ("ViT (vision transformer)", []),
    "beit": ("ViT (vision transformer)", []),
    "swin": ("ViT (vision transformer)", []),
    "resnet": ("CNN/conv", []),
    "mobilenet_v1": ("CNN/conv", []),
    "mobilenet_v2": ("CNN/conv", []),
    "efficientnet": ("CNN/conv", []),
    "convnext": ("CNN/conv", []),
    "segformer": ("CNN/conv", ["segmentation"]),
    "stable_diffusion": ("diffusion UNet+VAE", []),
    "unet": ("diffusion UNet+VAE", []),
    "flux": ("DiT (diffusion transformer)", []),
    "flux1": ("DiT (diffusion transformer)", []),
    "sd3": ("DiT (diffusion transformer)", []),
    "stable_diffusion_35_large": ("DiT (diffusion transformer)", []),
    "mochi": ("DiT (diffusion transformer)", ["video"]),
    "wan": ("DiT (diffusion transformer)", ["video"]),
    "ltx": ("DiT (diffusion transformer)", ["video"]),
    "qwenimage": ("DiT (diffusion transformer)", []),
    "acestep": ("DiT (diffusion transformer)", ["audio", "VAE"]),
    "xtts": ("autoregressive-GPT TTS", ["vocoder"]),
    "vibevoice": ("decoder-only causal LM", ["diffusion-head", "vocoder"]),
    "kokoro": ("non-transformer TTS (StyleTTS/iSTFTNet)", ["vocoder"]),
}

_FLAG_KEYWORDS = [
    ("moe", "MoE"),
    ("mamba", "Mamba/SSM"),
    ("state-space", "Mamba/SSM"),
    ("state space", "Mamba/SSM"),
    ("vocoder", "vocoder"),
    ("hifigan", "vocoder"),
    ("hifi-gan", "vocoder"),
    ("hifi_gan", "vocoder"),
    ("timestep", "diffusion-timestep"),
    ("conformer", "conformer"),
]


def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _backbone_from_text(text: str) -> Optional[str]:
    if any(k in text for k in ("transformer2d", "dit ", "ditlayer", "ditmodel", "diffusion transformer")):
        return "DiT (diffusion transformer)"
    if ("causallm" in text or "causalmm" in text or "decoder-only" in text) and "encoder" not in text:
        return "decoder-only causal LM"
    if "mamba" in text or "state-space" in text or "state space" in text:
        return "SSM/Mamba (hybrid)"
    if ("unet" in text or "stable diffusion" in text or "latent diffusion" in text) and "dit" not in text:
        return "diffusion UNet+VAE"
    if ("encoder" in text and "decoder" in text) or "encoderdecoder" in text or "seq2seq" in text:
        return "encoder-decoder transformer"
    if any(k in text for k in ("resnet", "convnext", "mobilenet", "conv1d", "conv2d")):
        return "CNN/conv"
    if any(k in text for k in ("vit", "beit", "deit", "swin")):
        return "ViT (vision transformer)"
    return None


def arch_descriptor(
    *,
    model_type: Optional[str] = None,
    architectures: Optional[List[str]] = None,
    notes: str = "",
    components: Optional[List[dict]] = None,
    pipeline_tag: Optional[str] = None,
) -> str:
    """Return a compact structural descriptor, e.g. 'decoder-only causal LM [MoE]' or
    'encoder-decoder transformer [TTS, vocoder]'. Resolution order: exact HF
    model_type table -> architectures class-name / component / notes inference. The
    SAME function is called for the target and for every backend, so routing can
    compare backbone-to-backbone rather than task-label-to-task-label."""
    nmt = _norm(model_type)
    backbone: Optional[str] = None
    flags: List[str] = []

    fam = _MODEL_TYPE_FAMILY.get(nmt) or _MODEL_TYPE_FAMILY.get((model_type or "").lower())
    if fam:
        backbone, flags = fam[0], list(fam[1])

    text = " ".join(
        filter(
            None,
            [
                (model_type or "").lower(),
                " ".join(architectures or []).lower(),
                (notes or "").lower(),
                " ".join((c.get("class_name") or c.get("name") or "") for c in (components or [])).lower(),
            ],
        )
    )
    if backbone is None:
        backbone = _backbone_from_text(text) or "unknown"

    for kw, tag in _FLAG_KEYWORDS:
        if kw in text and tag not in flags:
            flags.append(tag)

    return backbone + (" [" + ", ".join(flags) + "]" if flags else "")
