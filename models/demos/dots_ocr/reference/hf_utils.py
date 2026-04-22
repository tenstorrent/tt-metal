# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
HF reference loading for Dots OCR (CPU / no CUDA).

**Remote Hub models:** Dots loads **remote** code from the Hub (``trust_remote_code=True``). Transformers still runs
static ``check_imports`` on that file before ``from_pretrained(..., _attn_implementation="eager")``
applies, so a bare ``import flash_attn`` in the checkpoint must resolve. We use
``_flash_attn_shim.install()`` only for that import check; **runtime** attention is eager PyTorch
MHA via ``_attn_implementation="eager"``, not FlashAttention kernels and not CUDA-specific.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class HFLoadSpec:
    model_id: str
    revision: Optional[str] = None
    cache_dir: Optional[Path] = None
    dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True
    use_fast_processor: Optional[bool] = None


def get_hf_model_id(default: str = "rednote-hilab/dots.mocr") -> str:
    return os.environ.get("HF_MODEL", default)


def load_processor_and_model(spec: HFLoadSpec):
    """
    Load the Dots OCR model and processor using HuggingFace Transformers.

    **Eager attention:** ``_attn_implementation="eager"`` — PyTorch standard MHA, not
    FlashAttention kernels.

    Remote ``trust_remote_code`` modeling is still scanned with ``check_imports`` *before* those
    kwargs apply; if the repo lists ``import flash_attn``, we register a minimal compatibility
    namespace (see ``_flash_attn_shim.install``) so imports succeed. That is **not** the real
    ``flash_attn`` package and does not enable FA2 at runtime.
    """
    from models.demos.dots_ocr.reference._flash_attn_shim import install as _install_flash_attn_shim

    _install_flash_attn_shim()

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        cache_dir=str(spec.cache_dir) if spec.cache_dir else None,
        trust_remote_code=spec.trust_remote_code,
        use_fast=spec.use_fast_processor if spec.use_fast_processor is not None else True,
    )

    # Transformers kwargs around dtype have changed across versions:
    # - some accept `torch_dtype=...`
    # - some accept (or prefer) `dtype=...`
    # Try `dtype` first (newer warning suggests this), fall back to `torch_dtype`.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=str(spec.cache_dir) if spec.cache_dir else None,
            dtype=spec.dtype,
            trust_remote_code=spec.trust_remote_code,
            _attn_implementation="eager",
            use_safetensors=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=str(spec.cache_dir) if spec.cache_dir else None,
            torch_dtype=spec.dtype,
            trust_remote_code=spec.trust_remote_code,
            _attn_implementation="eager",
            use_safetensors=True,
        )
    model.eval()
    from models.demos.dots_ocr.reference._dots_hub_generation_patch import patch_dots_ocr_prepare_inputs_for_generation

    patch_dots_ocr_prepare_inputs_for_generation(model)
    return processor, model


def pick_tiny_model_fallback() -> str:
    """
    Best-effort tiny model id for tests (to avoid downloading multi-GB weights).
    If you have a local/internal tiny checkpoint, set HF_MODEL.
    """
    # There is no guaranteed official tiny for dots.mocr; use a tiny random multimodal CausalLM if available.
    # Users/CI should set HF_MODEL explicitly when needed.
    return os.environ.get("HF_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
