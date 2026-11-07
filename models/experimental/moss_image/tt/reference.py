# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native, locally reviewed upstream MOSS-Image CUDA pipeline."""
from __future__ import annotations

import torch
from diffusers import AutoencoderKLFlux2
from transformers import AutoModel, AutoTokenizer

from .checkpoint import VENDORED_PIPELINE_SHA256, resolve_checkpoint, sha256
from .vendor import pipeline_moss_image
from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer


def load_reference(variant="Base", checkpoint=None, *, device="cuda:0", local_files_only=False):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA reference requires an available CUDA device; CPU fallback is forbidden")
    source = __import__("pathlib").Path(pipeline_moss_image.__file__)
    if sha256(source) != VENDORED_PIPELINE_SHA256:
        raise RuntimeError("Vendored official pipeline differs from the reviewed pinned source")
    path = resolve_checkpoint(variant, checkpoint, local_files_only=local_files_only)
    transformer = MOSSImageTransformer.from_pretrained(
        path / "transformer", torch_dtype=torch.bfloat16, local_files_only=True
    ).eval()
    text_encoder = AutoModel.from_pretrained(
        path / "text_encoder", dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True, trust_remote_code=False)
    vae = AutoencoderKLFlux2.from_pretrained(path / "vae", torch_dtype=torch.float32, local_files_only=True).eval()
    scheduler = MOSSImageScheduler.from_pretrained(path / "scheduler", local_files_only=True)
    pipe = MOSSImagePipeline(transformer, scheduler, text_encoder, tokenizer, vae).to(device)
    return pipe, path
