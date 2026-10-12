# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host side of DeepSeek-V4.1-Flash image input: the checkpoint's own ``inference/image_processor.py`` (resize plan, patchify, span layout) driven through
the ``VisionArgs`` of the config, so prompts expand exactly like the reference (``prepare_vl_inputs``).

An image span is ``[IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END]``; every position carries ``image_token_id`` in the token ids and the
token type tells the positions apart (TEXT = -1 elsewhere).
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

from models.demos.blackhole.deepseek_v41_flash.tt.model_args import CKPT

_mod = {}


def _checkpoint_module(name, sub):
    """Import ``<ckpt>/<sub>/<name>.py`` (the checkpoint ships its reference implementation next to the weights)."""
    if name not in _mod:
        path = os.path.join(CKPT, sub, name + ".py")
        spec = importlib.util.spec_from_file_location(f"dsv41_ckpt_{name}", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _mod[name] = m
    return _mod[name]


def image_processor():
    return _checkpoint_module("image_processor", "inference")


_ip = image_processor()
TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = (
    _ip.TEXT,
    _ip.IMAGE_START,
    _ip.IMAGE,
    _ip.IMAGE_NEW_LINE,
    _ip.IMAGE_END,
)


def checkpoint_args(vargs):
    """The attribute names ``image_processor`` reads, from ``tt.vision.VisionArgs``."""
    return SimpleNamespace(
        vision_patch_size=vargs.patch,
        vision_downsample_ratio=vargs.ratio,
        vision_max_n_token=vargs.max_tokens,
        vision_min_pixels=vargs.min_pixels,
        vision_max_wh_ratio=vargs.max_wh_ratio,
        image_token_id=vargs.image_token_id,
        vision_enabled=vargs.n_layers > 0,
    )


def prepare_vl_inputs(prompt, images, tokenizer, vargs):
    """-> (tokens list, token_types list, [ImageInput] or None); ``images``: records ``{"data"|"url"|"source": ...}`` in prompt order."""
    enc = os.path.join(CKPT, "encoding")
    if enc not in sys.path:
        sys.path.insert(0, enc)
    return _ip.prepare_vl_inputs(prompt, images, tokenizer, checkpoint_args(vargs))


def image_inputs_from_pil(images, vargs):
    """Per-image ``ImageInput`` (patches, grids, span types) for ``start`` = 0, from PIL images or records; the span position is set by the caller."""
    cargs = checkpoint_args(vargs)
    out = []
    for rec in images:
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = _ip.load_image(rec, cargs)
        out.append(_ip.ImageInput(0, patches, n_vit_h, n_vit_w, _ip.image_token_types(n_llm_h, n_llm_w)))
    return out


def image_span_len(width, height, vargs):
    """Tokens (incl. delimiters) an image of this pixel size takes: the processor's resize plan, no pixels needed."""
    n_llm_h, n_llm_w, _, _ = _ip.plan_image_grid(width, height, checkpoint_args(vargs))
    return _ip.num_image_tokens(n_llm_h, n_llm_w)


def token_types_tensor(token_types):
    return torch.as_tensor(token_types, dtype=torch.int64)
