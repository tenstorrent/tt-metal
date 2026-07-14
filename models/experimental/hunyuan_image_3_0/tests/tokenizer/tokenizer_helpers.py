# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared constants and upstream-repo resolution for tokenizer tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, MODEL_DIR

CONFIG_PATH = Path(__file__).resolve().parents[2] / "ref/tokenizer/assets/config.json"
PROMPT = "a cat on a mat"
RECAPTION_PROMPT = "make the sky more dramatic at sunset"
IMAGE_SIZE = 1024
HF_IMAGE_SIZE = "1024x1024"
MAX_LENGTH = 10000
VIT_LAYERS = 1

_ENCODE_PREFIXES = (
    "vae.",
    "vision_model.",
    "vision_aligner.",
    "patch_embed.",
    "time_embed.",
    "timestep_emb.",
    "model.wte.",
)


def resolve_upstream_repo() -> Path | None:
    if env := os.environ.get("HUNYUAN_UPSTREAM"):
        p = Path(env)
        if p.is_dir():
            return p
    return None


def ensure_upstream_in_path() -> Path | None:
    upstream = resolve_upstream_repo()
    if upstream is not None and str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    return upstream


UPSTREAM = resolve_upstream_repo()
HAS_UPSTREAM = UPSTREAM is not None
HAS_WEIGHTS = (MODEL_DIR / "model.safetensors.index.json").is_file()
HAS_INSTRUCT = (INSTRUCT_MODEL_DIR / "model.safetensors.index.json").is_file()


def rope_pairs(info_row):
    return [(sl.start, sl.stop, hw) for sl, hw in info_row]
