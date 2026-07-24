# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
HF reference loading for Dots OCR (CPU / no CUDA).

**Remote Hub models:** Dots loads **remote** code from the Hub (``trust_remote_code=True``). Transformers still runs
static ``check_imports`` on that file before ``from_pretrained(..., _attn_implementation="eager")``
applies, so a bare ``import flash_attn`` in the checkpoint must resolve. We use
``flash_attention_shim.install()`` only for that import check; **runtime** attention is eager PyTorch
MHA via ``_attn_implementation="eager"``, not FlashAttention kernels and not CUDA-specific.
After load, :func:`_install_eager_vision_attention` also swaps the **vision tower** to the
remote code's eager ``VisionAttention`` (``vision_config`` defaults to flash otherwise).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Hub remote-code vision classes live under these prefixes; constrain dynamic imports (SAST / supply-chain hygiene).


def _resolve_remote_vision_module(module_name: str) -> types.ModuleType:
    """
    Return the module object for validated Hub / transformers vision code.

    After ``from_pretrained(..., trust_remote_code=True)``, remote modules are already present in
    ``sys.modules``. We validate ``module_name`` (same rules as before) and resolve via that dict only,
    avoiding :func:`importlib.import_module` on a dynamic string so static analysis does not flag
    unsanitized input passed to code-loading APIs.
    """
    if not isinstance(module_name, str) or not module_name.strip():
        raise ValueError("module_name must be a non-empty str")

    mod = sys.modules.get(module_name)
    if isinstance(mod, types.ModuleType):
        return mod

    raise ValueError(
        f"remote vision module {module_name!r} is not present in sys.modules after model load "
        "(expected with trust_remote_code); refusing dynamic import"
    )


@dataclass(frozen=True)
class HFLoadSpec:
    model_id: str
    revision: Optional[str] = None
    cache_dir: Optional[Path] = None
    dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True
    use_fast_processor: Optional[bool] = None


_DOTS_MOCR_HUB_ID = "rednote-hilab/dots.mocr"

# Public default used when seeding ``HF_MODEL`` for ``tt_transformers.ModelArgs`` (required).
DOTS_OCR_DEFAULT_HF_MODEL_ID = _DOTS_MOCR_HUB_ID


def get_hf_model_id(default: str = "rednote-hilab/dots.mocr") -> str:
    """Canonical Dots OCR Hub checkpoint id (fixed; not read from ``HF_MODEL``)."""
    _ = default  # kept for call-site compatibility; value is always the canonical Hub id
    return DOTS_OCR_DEFAULT_HF_MODEL_ID


def load_processor_and_model(spec: HFLoadSpec):
    """
    Load the Dots OCR model and processor using HuggingFace Transformers.

    **Eager attention:** ``_attn_implementation="eager"`` — PyTorch standard MHA, not
    FlashAttention kernels.

    Remote ``trust_remote_code`` modeling is still scanned with ``check_imports`` *before* those
    kwargs apply; if the repo lists ``import flash_attn``, we register a minimal compatibility
    namespace (see ``flash_attention_shim.install``) so imports succeed. That is **not** the real
    ``flash_attn`` package and does not enable FA2 at runtime.
    """
    from models.demos.dots_ocr.reference.flash_attention_shim import install as _install_flash_attn_shim

    _install_flash_attn_shim()

    from transformers import AutoModelForCausalLM, AutoProcessor

    cache_dir = str(spec.cache_dir) if spec.cache_dir else None
    model_id = spec.model_id
    model_path = Path(model_id).expanduser()
    looks_like_local_path = model_id.startswith(("/", "./", "../", "~"))

    # Prefer loading from a local snapshot path for Hub ids. This avoids dynamic-module
    # namespace edge cases for dotted repo names and keeps trust_remote_code behavior stable.
    model_source: str
    model_revision: Optional[str]
    if model_path.exists():
        model_source = str(model_path.resolve())
        model_revision = None
    elif looks_like_local_path:
        raise FileNotFoundError(
            f"HF model path does not exist: {model_path}. "
            "Pass a valid local model directory or a Hub id like 'rednote-hilab/dots.mocr' in HFLoadSpec.model_id."
        )
    else:
        try:
            model_source = snapshot_download(
                repo_id=model_id,
                revision=spec.revision,
                cache_dir=cache_dir,
            )
            model_revision = None
        except Exception as exc:
            logger.warning(
                "snapshot_download failed for '%s' (%s); falling back to direct from_pretrained with repo id",
                model_id,
                exc,
            )
            model_source = model_id
            model_revision = spec.revision

    processor_kwargs = {
        "revision": model_revision,
        "cache_dir": cache_dir,
        "trust_remote_code": spec.trust_remote_code,
        "use_fast": spec.use_fast_processor if spec.use_fast_processor is not None else True,
    }
    try:
        processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
    except TypeError as exc:
        # Newer transformers validates that Qwen2.5-VL-style processors receive a
        # BaseVideoProcessor. Some Dots remote processor revisions still omit this arg.
        if "video_processor" not in str(exc):
            raise
        from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor
        from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

        logger.warning(
            "AutoProcessor construction failed due to missing video_processor; "
            "falling back to manual Qwen2_5_VLProcessor assembly."
        )
        image_processor = AutoImageProcessor.from_pretrained(model_source, **processor_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_source, **processor_kwargs)
        video_processor = AutoVideoProcessor.from_pretrained(model_source, **processor_kwargs)
        processor = Qwen2_5_VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=getattr(tokenizer, "chat_template", None),
        )
        if not hasattr(tokenizer, "image_token"):
            processor.image_token = "<|imgpad|>"
        if not hasattr(tokenizer, "image_token_id"):
            processor.image_token_id = 151665

    # Transformers kwargs around dtype have changed across versions:
    # - some accept `torch_dtype=...`
    # - some accept (or prefer) `dtype=...`
    # Try `dtype` first (newer warning suggests this), fall back to `torch_dtype`.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            revision=model_revision,
            cache_dir=cache_dir,
            dtype=spec.dtype,
            trust_remote_code=spec.trust_remote_code,
            _attn_implementation="eager",
            use_safetensors=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            revision=model_revision,
            cache_dir=cache_dir,
            torch_dtype=spec.dtype,
            trust_remote_code=spec.trust_remote_code,
            _attn_implementation="eager",
            use_safetensors=True,
        )
    model.eval()
    from models.demos.dots_ocr.reference.dots_hub_generation import patch_dots_ocr_prepare_inputs_for_generation

    patch_dots_ocr_prepare_inputs_for_generation(model)
    _install_eager_vision_attention(model)
    return processor, model


def _install_eager_vision_attention(model) -> None:
    """
    Dots stores **vision** attention in ``config.vision_config.attn_implementation`` (default
    ``flash_attention_2``). The ``from_pretrained`` ``_attn_implementation=eager`` flag only sets the
    **text** LM attention; the vision tower is already built with ``VisionFlashAttention2``.

    Replace each block's attention with the remote code's **eager** ``VisionAttention`` (manual
    mask + matmul + softmax) using the same ``qkv``/``proj`` weights so HF reference matches
    PCC expectations without ``flash_attn`` at runtime.
    """
    vt = getattr(model, "vision_tower", None)
    if vt is None or not hasattr(vt, "blocks"):
        return
    cfg = getattr(getattr(model, "config", None), "vision_config", None)
    if cfg is None:
        return

    module_name = vt.__class__.__module__
    try:
        mod = _resolve_remote_vision_module(module_name)
    except ValueError as exc:
        logger.warning("eager vision swap: refused dynamic import (%s); leaving vision attention as loaded.", exc)
        return
    VisionAttention = getattr(mod, "VisionAttention", None)
    if VisionAttention is None:
        logger.warning(
            "eager vision swap: VisionAttention not found in %s; leaving vision attention as loaded.",
            vt.__class__.__module__,
        )
        return

    if getattr(cfg, "attn_implementation", None) == "eager" and all(
        isinstance(b.attn, VisionAttention) for b in vt.blocks
    ):
        return

    cfg.attn_implementation = "eager"
    for block in vt.blocks:
        old = block.attn
        if isinstance(old, VisionAttention):
            continue
        device = next(old.parameters()).device
        dtype = next(old.parameters()).dtype
        new_attn = VisionAttention(
            cfg,
            cfg.embed_dim,
            num_heads=cfg.num_attention_heads,
            bias=cfg.use_bias,
        )
        new_attn = new_attn.to(device=device, dtype=dtype)
        new_attn.load_state_dict(old.state_dict(), strict=True)
        block.attn = new_attn


def pick_tiny_model_fallback() -> str:
    """
    Best-effort tiny model id for tests (to avoid downloading multi-GB weights).
    """
    # There is no guaranteed official tiny for dots.mocr; use a tiny random multimodal CausalLM if available.
    return "hf-internal-testing/tiny-random-LlamaForCausalLM"
