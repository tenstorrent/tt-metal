# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class HFLoadSpec:
    model_id: str
    revision: Optional[str] = None
    cache_dir: Optional[Path] = None
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True


def get_hf_model_id(default: str = "rednote-hilab/dots.mocr") -> str:
    return os.environ.get("HF_MODEL", default)


def load_processor_and_model(spec: HFLoadSpec):
    """
    Load the Dots OCR model and processor using HuggingFace Transformers.

    Note:
    - Dots uses custom code (trust_remote_code=True).
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        cache_dir=str(spec.cache_dir) if spec.cache_dir else None,
        trust_remote_code=spec.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        cache_dir=str(spec.cache_dir) if spec.cache_dir else None,
        torch_dtype=spec.torch_dtype,
        trust_remote_code=spec.trust_remote_code,
    )
    model.eval()
    return processor, model


def pick_tiny_model_fallback() -> str:
    """
    Best-effort tiny model id for tests (to avoid downloading multi-GB weights).
    If you have a local/internal tiny checkpoint, set HF_MODEL.
    """
    # There is no guaranteed official tiny for dots.mocr; use a tiny random multimodal CausalLM if available.
    # Users/CI should set HF_MODEL explicitly when needed.
    return os.environ.get("HF_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
