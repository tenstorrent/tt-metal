# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Build COCO val2017 images/ + binary masks/ for IoU/Dice eval. Requires: pip install pycocotools"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as mask_util
except ImportError as e:
    raise SystemExit("Install pycocotools: pip install pycocotools") from e


def parse_args():
    p = argparse.ArgumentParser(description="COCO val subset → images/ + masks/ for one category")
    p.add_argument(
        "--coco-root", type=Path, required=True, help="Contains val2017/ and annotations/instances_val2017.json"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/demos/attention_denseunet/eval/datasets/coco_val2017_person_subset"),
    )
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--category", type=str, default="person")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    return p.parse_args()


def build_union_mask(anns: list, height: int, width: int) -> np.ndarray:
    union = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        seg = ann["segmentation"]
        if isinstance(seg, dict):
            # COCO may store RLE as:
            # - compressed dict: {"counts": "<bytes-like str>", "size": [h, w]}
            # - uncompressed dict: {"counts": [..], "size": [h, w]}
            # pycocotools.decode accepts compressed RLE directly, but uncompressed
            # dicts must be converted via frPyObjects first.
            counts = seg.get("counts")
            if isinstance(counts, list):
                rle = mask_util.frPyObjects(seg, height, width)
                m = mask_util.decode(rle).astype(np.uint8)
            else:
                m = mask_util.decode(seg).astype(np.uint8)
        elif isinstance(seg, list) and len(seg) > 0:
            rles = mask_util.frPyObjects(seg, height, width)
            m = mask_util.decode(rles)
            if m.ndim == 3:
                m = (m.sum(axis=2) > 0).astype(np.uint8)
            else:
                m = m.astype(np.uint8)
        else:
            continue
        union = np.maximum(union, m)
    return union


def main():
    args = parse_args()
    coco_root = args.coco_root.resolve()
    ann_path = coco_root / "annotations" / "instances_val2017.json"
    img_dir = coco_root / "val2017"
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)
    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)

    with ann_path.open() as f:
        coco = json.load(f)
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    if args.category not in name_to_id:
        raise ValueError(f"Unknown category {args.category!r}")
    cat_id = name_to_id[args.category]
    images = {im["id"]: im for im in coco["images"]}
    by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] == cat_id:
            by_image[ann["image_id"]].append(ann)

    eligible = sorted((i for i in by_image if by_image[i]), key=lambda i: images[i]["file_name"])
    selected = eligible[: args.max_images]
    if not selected:
        raise RuntimeError(f"No images with category {args.category!r}")

    out_images = args.out_dir / "images"
    out_masks = args.out_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    for image_id in selected:
        meta = images[image_id]
        file_name = meta["file_name"]
        h0, w0 = meta["height"], meta["width"]
        src = img_dir / file_name
        if not src.is_file():
            print(f"warning: missing {src}")
            continue
        union = build_union_mask(by_image[image_id], h0, w0)
        mask_img = Image.fromarray((union > 0).astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize((args.width, args.height), Image.Resampling.NEAREST)
        shutil.copy2(src, out_images / file_name)
        stem = Path(file_name).stem
        mask_img.save(out_masks / f"{stem}.png")

    n = sum(1 for _ in out_images.iterdir())
    print(f"Wrote {n} pairs under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
