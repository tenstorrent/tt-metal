# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Self-contained input construction for NVIDIA LocateAnything-3B.

Replicates the HF `LocateAnythingImageProcessor` + `LocateAnythingProcessor`
chat-template exactly, but WITHOUT importing the repo's processor module
(which hard-imports cv2/lmdb/decord that are not installed here).

Used by both the torch CPU reference and the tt-nn device port so inputs are
byte-identical.
"""
import glob
import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

# --- special tokens / ids (from config.json) ---
IMAGE_TOKEN = "<IMG_CONTEXT>"
IMAGE_START_TOKEN = "<img>"
IMAGE_END_TOKEN = "</img>"
IMAGE_TOKEN_INDEX = 151665

_PROMPT = "Locate all the instances that matches the following description: "

# image normalization (preprocessor_config.json)
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
PATCH_SIZE = 14
MERGE = (2, 2)
IN_TOKEN_LIMIT = 25600  # preprocessor_config.json


def find_model_path():
    """Locate the downloaded LocateAnything-3B snapshot dir."""
    env = os.environ.get("LA_MODEL_PATH")
    if env and os.path.isdir(env):
        return env
    pat = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--LocateAnything-3B/snapshots/*/")
    cands = sorted(glob.glob(pat))
    if not cands:
        raise FileNotFoundError(f"No LocateAnything-3B snapshot found under {pat}")
    return cands[-1].rstrip("/")


def _rescale(image: Image.Image, in_token_limit=IN_TOKEN_LIMIT) -> Image.Image:
    """Exact port of LocateAnythingImageProcessor.rescale."""
    w, h = image.size
    p = PATCH_SIZE
    if (w // p) * (h // p) > in_token_limit:
        scale = math.sqrt(in_token_limit / ((w // p) * (h // p)))
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    new_w, new_h = image.size
    pad_h = MERGE[0] * p
    pad_w = MERGE[1] * p
    target_w = math.ceil(new_w / pad_w) * pad_w
    target_h = math.ceil(new_h / pad_h) * pad_h
    if target_w != new_w or target_h != new_h:
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    w, h = image.size
    if w // p >= 512 or h // p >= 512:
        raise ValueError("Exceed pos emb")
    return image


def preprocess_image(image: Image.Image, in_token_limit=IN_TOKEN_LIMIT):
    """Returns (pixel_values [L,3,14,14] float, grid_hw (H_patches, W_patches))."""
    image = _rescale(image.convert("RGB"), in_token_limit)
    t = TF.to_tensor(image)  # [3,H,W] in [0,1]
    t = TF.normalize(t, MEAN, STD)
    C, H, W = t.shape
    p = PATCH_SIZE
    patches = t.reshape(C, H // p, p, W // p, p)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous().view(-1, C, p, p)
    grid_hw = (H // p, W // p)
    return patches, grid_hw


def num_image_tokens(grid_hw):
    """Merged vision-token count for one image."""
    return (grid_hw[0] * grid_hw[1]) // (MERGE[0] * MERGE[1])


def build_chat_text(query: str, n_img_tokens: int) -> str:
    """Replicates LocateAnythingProcessor.py_apply_chat_template + media replacement."""
    text_body = _PROMPT + query + "."
    img_block = f"<image 1>{IMAGE_START_TOKEN}{IMAGE_TOKEN * n_img_tokens}{IMAGE_END_TOKEN}"
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{img_block}{text_body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_inputs(tokenizer, image: Image.Image, query: str, in_token_limit=IN_TOKEN_LIMIT):
    """Full input bundle for both reference and device runs.

    Returns dict: input_ids [1,S], attention_mask [1,S], pixel_values [L,3,14,14],
    image_grid_hws np.int32 [1,2], grid_hw tuple, n_img_tokens int.
    """
    pixel_values, grid_hw = preprocess_image(image, in_token_limit)
    n_tok = num_image_tokens(grid_hw)
    text = build_chat_text(query, n_tok)
    enc = tokenizer([text], return_tensors="pt")
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))
    n_in_ids = int((input_ids[0] == IMAGE_TOKEN_INDEX).sum().item())
    assert n_in_ids == n_tok, f"image-token mismatch: ids={n_in_ids} expected={n_tok}"
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "pixel_values": pixel_values,
        "image_grid_hws": np.array([grid_hw], dtype=np.int32),
        "grid_hw": grid_hw,
        "n_img_tokens": n_tok,
    }


def load_test_image(path=None):
    """Load a deterministic test image. Falls back to a synthetic image."""
    if path is None:
        mp = find_model_path()
        for name in ("teaser.jpg", "coco_lvis.png", "dense_object_detection.png", "referring.png"):
            cand = os.path.join(mp, "assets", name)
            if os.path.exists(cand) and os.path.getsize(cand) > 0:
                path = cand
                break
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB"), path
    # synthetic deterministic fallback
    g = np.zeros((448, 448, 3), dtype=np.uint8)
    g[112:336, 112:336] = (200, 80, 40)
    g[50:120, 300:400] = (40, 160, 220)
    return Image.fromarray(g), "synthetic-448x448"
