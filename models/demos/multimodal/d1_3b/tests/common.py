# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for d1-3B tests. No ttnn imports: the reference generator runs in a plain CPU env."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from models.demos.multimodal.d1_3b.tt.checkpoint import (  # noqa: F401  (re-exported for the tests)
    D1_HF_MODEL,
    D1_REVISION,
    remote_modules,
    resolve_checkpoint,
)

HERE = Path(__file__).parent
EXAMPLES_PATH = HERE / "examples.json"
REFERENCE_DIR = HERE / "reference"
# Hidden states, full logits and pixel inputs are too large for the repo; they go here.
SIDECAR_DIR = Path(os.environ.get("D1_SIDECAR_DIR", Path.home() / ".cache" / "d1_3b" / "reference"))
IMAGE_CACHE = Path(os.environ.get("D1_IMAGE_CACHE", Path.home() / ".cache" / "d1_3b" / "images"))

REMOTE_IMAGES = {
    "coco_000000039769.jpg": (
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e",
    ),
}


def load_examples():
    return json.loads(EXAMPLES_PATH.read_text())


def load_state(example):
    if "state_file" not in example:
        return example["state"]
    path = HERE / example["state_file"]
    return json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()


def _synthetic_red_circle():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(img).ellipse((96, 96, 288, 288), fill=(220, 20, 20))
    return img


def load_image(name):
    from PIL import Image

    if name == "synthetic_red_circle.png":
        return _synthetic_red_circle()
    url, sha = REMOTE_IMAGES[name]
    path = IMAGE_CACHE / name
    if not path.is_file():
        IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        raise ValueError(f"{path} sha256 {digest} != pinned {sha}")
    return Image.open(path).convert("RGB")


def load_images(example):
    return [load_image(n) for n in example.get("images", [])]


def ids_sha256(input_ids):
    return hashlib.sha256(json.dumps([int(i) for i in input_ids]).encode()).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
