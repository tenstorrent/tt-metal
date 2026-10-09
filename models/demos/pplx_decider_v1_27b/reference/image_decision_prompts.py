# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic 8-row image decision prompt set for the stage-12 vision goldens.

Each row is an autojev ``DecisionInput`` with ``"images": [<png path>]`` (one image). The images
are drawn with PIL from a fixed seed (colored regions, shapes, a bar chart, a rendered receipt,
a progress bar), so the ground truth of every question is known by construction.

Encoding reproduces the reference app (snapshot ``source/src/autojev/model.py`` ``prepare``):
``open_image`` -> ``decision_messages(row, codes)`` (one ``{"type": "image"}`` before the text)
-> ``apply_chat_template(..., enable_thinking=False)`` -> ``processor(text=[text], images=[img],
padding=True)`` with ``image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}``;
batch 1, no padding.

CLI (writes the PNGs, prints the per-row table, needs no weights)::

    python -m models.demos.pplx_decider_v1_27b.reference.image_decision_prompts [--image-dir DIR]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageDraw, ImageFont

from models.demos.pplx_decider_v1_27b.reference.decision_prompts import MAX_LENGTH, AppTokenizer, bucket_of

SEED = 20261012
APP_IMAGE_SIZE = {"shortest_edge": 65536, "longest_edge": 262144}
IMAGE_TOKEN_ID = 248056
DEFAULT_IMAGE_DIR = Path("/local/ttuser/gtobar/artifacts/pplx_decider/goldens/vision/images")
STATE = "The user attached one image. Answer from what the image shows."

RGB = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 80, 210),
    "yellow": (240, 210, 30),
    "purple": (140, 50, 170),
    "orange": (245, 140, 20),
    "grey": (128, 128, 128),
}
FIVE_LEVELS = ["1: very low", "2: low", "3: medium", "4: high", "5: very high"]


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)  # Pillow's bundled font: host independent


def shape(draw: ImageDraw.ImageDraw, kind: str, cx: int, cy: int, r: int, color) -> None:
    box = (cx - r, cy - r, cx + r, cy + r)
    if kind == "circle":
        draw.ellipse(box, fill=color)
    elif kind == "square":
        draw.rectangle(box, fill=color)
    else:  # triangle
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


def scatter(rng: random.Random, w: int, h: int, r: int, n: int) -> list[tuple[int, int]]:
    """``n`` non-overlapping centers (min distance 2.4 r) inside a margin."""
    pts: list[tuple[int, int]] = []
    while len(pts) < n:
        p = (rng.randint(r + 8, w - r - 8), rng.randint(r + 8, h - r - 8))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > (2.4 * r) ** 2 for q in pts):
            pts.append(p)
    return pts


# ----------------------------------------------------------------------------------------------
# Image builders: each returns (PIL image, row question, expected answer key)
# ----------------------------------------------------------------------------------------------


def img_dominant_color(rng: random.Random):
    im = Image.new("RGB", (256, 256), RGB["blue"])
    d = ImageDraw.Draw(im)
    for (cx, cy), c in zip(scatter(rng, 256, 256, 14, 3), ("red", "green", "yellow")):
        shape(d, "square", cx, cy, 14, RGB[c])
    q = {
        "type": "choice",
        "instructions": "Which color covers most of the image?",
        "criteria": {c: None for c in ("red", "green", "blue", "yellow", "purple")},
    }
    return im, q, "blue"


def img_count_circles(rng: random.Random):
    w, h = 1024, 768
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    pts = scatter(rng, w, h, 70, 6)
    kinds = ["circle"] * 4 + ["square", "triangle"]
    for (cx, cy), k, c in zip(pts, kinds, ("red", "blue", "green", "orange", "purple", "grey")):
        shape(d, k, cx, cy, 70, RGB[c])
    q = {
        "type": "choice",
        "instructions": "How many circles are in the image? Squares and triangles do not count.",
        "criteria": {n: None for n in ("2", "3", "4", "5", "6")},
    }
    return im, q, "4"


def img_receipt(rng: random.Random):
    w, h = 360, 540
    im = Image.new("RGB", (w, h), (250, 248, 240))
    d = ImageDraw.Draw(im)
    f, fb = font(22), font(30)
    d.text((w // 2, 30), "CORNER CAFE", font=fb, fill="black", anchor="mm")
    items = [("Latte", 4.75), ("Bagel", 3.20), ("Orange juice", 3.90), ("Muffin", 2.85), ("Water", 1.50)]
    y = 80
    for name, price in items:
        d.text((24, y), name, font=f, fill="black")
        d.text((w - 24, y), f"${price:.2f}", font=f, fill="black", anchor="ra")
        y += 40
    subtotal = sum(p for _, p in items)
    tax = round(subtotal * 0.08, 2)
    total = round(subtotal + tax, 2)
    d.line((20, y + 5, w - 20, y + 5), fill="black", width=2)
    y += 20
    for label, val, fnt in (("Subtotal", subtotal, f), ("Tax 8%", tax, f), ("TOTAL", total, fb)):
        d.text((24, y), label, font=fnt, fill="black")
        d.text((w - 24, y), f"${val:.2f}", font=fnt, fill="black", anchor="ra")
        y += 44
    d.text((w // 2, h - 30), "Thank you!", font=f, fill="black", anchor="mm")
    expected = f"${total:.2f}"
    distract = [f"${subtotal:.2f}", f"${total + 4.10:.2f}", f"${total - 3.35:.2f}"]
    opts = sorted([expected] + distract)
    q = {
        "type": "choice",
        "instructions": "What is the TOTAL amount printed on the receipt?",
        "criteria": {o: None for o in opts},
    }
    return im, q, expected


def img_tallest_bar(rng: random.Random):
    w, h = 512, 512
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    vals = [42, 61, 35, 88, 57]
    f = font(24)
    base, top = h - 60, 40
    d.line((40, base, w - 20, base), fill="black", width=3)
    bw = (w - 80) // len(days)
    for i, (day, v) in enumerate(zip(days, vals)):
        x0 = 50 + i * bw + 10
        d.rectangle((x0, base - (base - top) * v // 100, x0 + bw - 20, base), fill=RGB["blue"])
        d.text((x0 + (bw - 20) // 2, base + 25), day, font=f, fill="black", anchor="mm")
    q = {
        "type": "choice",
        "instructions": "In the bar chart, which day has the tallest bar?",
        "criteria": {day: None for day in days},
    }
    return im, q, "Thu"


def _three_shapes(rng: random.Random, w: int, h: int, spec: list[tuple[str, str]]):
    im = Image.new("RGB", (w, h), (235, 235, 235))
    d = ImageDraw.Draw(im)
    r = min(w, h) // 8
    for (cx, cy), (k, c) in zip(scatter(rng, w, h, r, len(spec)), spec):
        shape(d, k, cx, cy, r, RGB[c])
    return im


def img_red_circle_yes(rng: random.Random):
    im = _three_shapes(rng, 320, 240, [("circle", "red"), ("square", "blue"), ("triangle", "green")])
    return im, {"type": "noul", "instructions": "Is there a red circle in the image?"}, "true"


def img_red_circle_no(rng: random.Random):
    im = _three_shapes(rng, 448, 320, [("square", "red"), ("circle", "blue"), ("triangle", "green")])
    return im, {"type": "noul", "instructions": "Is there a red circle in the image?"}, "false"


def img_brightness_dark(rng: random.Random):
    w, h = 300, 500
    im = Image.new("RGB", (w, h), (14, 14, 18))
    d = ImageDraw.Draw(im)
    for cx, cy in scatter(rng, w, h, 20, 4):
        shape(d, "circle", cx, cy, 20, (40, 40, 48))
    q = {"type": "score", "instructions": "Rate the overall brightness of the image.", "criteria": FIVE_LEVELS}
    return im, q, "0"


def img_progress_bar(rng: random.Random):
    w, h = 640, 360
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    x0, x1, y0, y1 = 40, w - 40, h // 2 - 30, h // 2 + 30
    d.rectangle((x0, y0, x1, y1), outline="black", width=4, fill=(225, 225, 225))
    d.rectangle((x0 + 4, y0 + 4, x0 + 4 + int((x1 - x0 - 8) * 0.9), y1 - 4), fill=RGB["green"])
    q = {
        "type": "score",
        "instructions": "How full is the progress bar?",
        "criteria": ["0-20% full", "21-40% full", "41-60% full", "61-80% full", "81-100% full"],
    }
    return im, q, "4"


@dataclass
class ImageSpec:
    id: str
    make: Callable[[random.Random], tuple[Image.Image, dict, str]]


SPECS = [
    ImageSpec("v01_dominant_color", img_dominant_color),
    ImageSpec("v02_count_circles", img_count_circles),
    ImageSpec("v03_receipt_total", img_receipt),
    ImageSpec("v04_tallest_bar", img_tallest_bar),
    ImageSpec("v05_red_circle_yes", img_red_circle_yes),
    ImageSpec("v06_red_circle_no", img_red_circle_no),
    ImageSpec("v07_brightness_dark", img_brightness_dark),
    ImageSpec("v08_progress_fill", img_progress_bar),
]


# ----------------------------------------------------------------------------------------------
# App-exact encoding
# ----------------------------------------------------------------------------------------------


def image_tokenizer(snapshot=None) -> AppTokenizer:
    """``AppTokenizer`` with the app's image processor pixel budget (``model.py:139``)."""
    tok = AppTokenizer(snapshot) if snapshot is not None else AppTokenizer()
    tok.processor.image_processor.size = dict(APP_IMAGE_SIZE)
    return tok


def encode(tok: AppTokenizer, row: dict) -> dict[str, torch.Tensor]:
    """``DecisionModel.prepare`` for one row: processor outputs (input_ids, attention_mask,
    pixel_values, image_grid_thw, mm_token_type_ids; image keys absent for text-only rows)."""
    assert tok.processor.image_processor.size == APP_IMAGE_SIZE
    images = [tok.autojev.open_image(v) for v in row.get("images", [])]
    enc = dict(tok.processor(text=[tok.text(row)], images=images or None, padding=True, return_tensors="pt"))
    if int(enc["attention_mask"].sum()) != enc["input_ids"].shape[1]:
        raise AssertionError("batch-1 encoding must have no padding")
    if enc["input_ids"].shape[1] > MAX_LENGTH:
        raise ValueError("prompt exceeds the 8192-token limit")
    return enc


def build_image_prompt_set(tok: AppTokenizer, image_dir: Path = DEFAULT_IMAGE_DIR) -> list[dict]:
    """Draw (or redraw identically) the 8 PNGs and encode each row exactly as the app does."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, spec in enumerate(SPECS):
        im, question, expected = spec.make(random.Random(SEED + k))
        path = image_dir / f"{spec.id}.png"
        im.save(path, format="PNG")
        row = {"state": STATE, "question": question, "images": [str(path)]}
        enc = encode(tok, row)
        ids = enc["input_ids"][0]
        t, gh, gw = enc["image_grid_thw"][0].tolist()
        n_img = int((ids == IMAGE_TOKEN_ID).sum())
        assert n_img == t * gh * gw // 4
        rows.append(
            {
                "id": spec.id,
                "type": question["type"],
                "row": row,
                "meta": {"expected": expected},
                "count": tok.count(row),
                "image_size": list(im.size),
                "grid_thw": [t, gh, gw],
                "patches": t * gh * gw,
                "image_tokens": n_img,
                "seq_len": int(ids.shape[0]),
                "bucket": bucket_of(int(ids.shape[0])),
                "input_ids": ids.tolist(),
                "enc": enc,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    args = parser.parse_args()
    for r in build_image_prompt_set(image_tokenizer(), args.image_dir):
        print(
            f"{r['id']:<22} {r['type']:<6} size={r['image_size']} grid={r['grid_thw']} patches={r['patches']:<5} "
            f"(%128={r['patches'] % 128:<3}) img_tok={r['image_tokens']:<4} seq={r['seq_len']:<4} "
            f"bucket={r['bucket']} count={r['count']} expected={r['meta']['expected']}"
        )


if __name__ == "__main__":
    main()
