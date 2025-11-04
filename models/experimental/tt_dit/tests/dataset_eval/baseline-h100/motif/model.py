# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import json
import time
import argparse
import csv
import urllib
import subprocess
from pathlib import Path
from loguru import logger
from PIL import Image

from clip_encoder import CLIPEncoder
from fid_score import calculate_fid_score


COCO_CAPTIONS_DOWNLOAD_PATH = (
    "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/"
    "text_to_image/coco2014/captions/captions_source.tsv"
)


def load_coco_prompts(captions_path, start_from=0, num_prompts=16):
    """loads prompts from COCO captions tsv, downloading if missing"""
    prompts = []
    if not os.path.isfile(captions_path):
        logger.info(f"file {captions_path} not found. downloading...")
        urllib.request.urlretrieve(COCO_CAPTIONS_DOWNLOAD_PATH, captions_path)
        logger.info("download complete.")

    with open(captions_path, "r") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)  # skip header
        for index, row in enumerate(reader):
            if index < start_from:
                continue
            if index >= start_from + num_prompts:
                break
            if len(row) >= 3:
                prompts.append(row[2].strip())
    logger.info(f"loaded {len(prompts)} coco prompts from {captions_path}")
    return prompts


def generate_images_with_inference(prompts_file, output_dir, num_prompts):
    """forces motif inference to regenerate only the requested number of prompts"""
    logger.info("clearing any previous images...")
    os.makedirs(output_dir, exist_ok=True)
    for f in Path(output_dir).rglob("*.png"):
        f.unlink()

    # create temporary subset prompt file
    tmp_prompts_file = os.path.join(output_dir, "subset_prompts.txt")
    with open(prompts_file, "r") as f_in, open(tmp_prompts_file, "w") as f_out:
        lines = [line.strip() for line in f_in if line.strip()]
        subset = lines[:num_prompts]
        for l in subset:
            f_out.write(l + "\n")

    logger.info(f"generating {len(subset)} images using motif inference.py ...")

    cmd = [
        "python",
        "inference.py",
        "--model-config",
        "configs/mmdit_xlarge_hq.json",
        "--model-ckpt",
        "checkpoints/pytorch_model_fsdp.bin",
        "--seed",
        "7777",
        "--steps",
        "20",
        "--resolution",
        "1024",
        "--prompt-file",
        tmp_prompts_file,  # use subset
        "--guidance-scales",
        "4.0",
        "--output-dir",
        output_dir,
        "--batch-size",
        "1",
    ]
    subprocess.run(cmd, check=True)
    logger.info("image generation complete.")


def compute_clip_scores(images_dir, prompts):
    """computes average clip similarity across generated images and corresponding prompts"""
    clip_encoder = CLIPEncoder(clip_version="ViT-B/32", pretrained="openai")
    scores = []

    # find all pngs recursively
    all_images = sorted(Path(images_dir).rglob("*.png"))
    if len(all_images) < len(prompts):
        logger.warning(f"found only {len(all_images)} images, expected {len(prompts)}")

    for idx, (prompt, img_path) in enumerate(zip(prompts, all_images)):
        try:
            image = Image.open(img_path).convert("RGB")
            score = clip_encoder.get_clip_score(prompt, image)
            scores.append(score.item())
        except Exception as e:
            logger.error(f"error processing {img_path}: {e}")

    if not scores:
        return 0.0

    avg_score = sum(scores) / len(scores)
    logger.info(f"average clip score = {avg_score:.4f} ({avg_score * 100:.2f} CLIPScore)")
    return avg_score


def compute_fid(images_dir, coco_stats="val2014.npz"):
    """computes fid score against coco statistics"""
    if not os.path.exists(coco_stats):
        logger.info(f"COCO stats not found at {coco_stats}, downloading...")
        os.makedirs(os.path.dirname(coco_stats) or ".", exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/"
            "text_to_image/tools/val2014.npz",
            coco_stats,
        )
        logger.info("download complete.")

    all_images = [Image.open(f).convert("RGB") for f in sorted(Path(images_dir).rglob("*.png"))]
    if not all_images:
        logger.error("no images found for fid computation")
        return None

    logger.info(f"computing fid over {len(all_images)} images ...")
    fid = calculate_fid_score(all_images, coco_stats)
    logger.info(f"fid score = {fid:.4f}")
    return fid


def main():
    parser = argparse.ArgumentParser(description="Motif Dataset Evaluation")
    parser.add_argument("--images-dir", type=str, required=True, help="directory containing generated images")
    parser.add_argument("--prompts-file", type=str, required=True, help="path to prompts file (.tsv or .txt)")
    parser.add_argument("--output-json", type=str, default="dataset_eval_results.json", help="output results json")
    parser.add_argument("--start-from", type=int, default=0, help="starting index for captions.tsv")
    parser.add_argument("--num-prompts", type=int, default=16, help="number of prompts to evaluate")
    args = parser.parse_args()

    start_time = time.time()

    # load prompts dynamically depending on file type
    if args.prompts_file.endswith(".tsv"):
        prompts = load_coco_prompts(args.prompts_file, args.start_from, args.num_prompts)
    else:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

    # always regenerate fresh images
    generate_images_with_inference(args.prompts_file, args.images_dir, args.num_prompts)

    # automatically detect where the actual images were saved
    def find_deepest_image_dir(root):
        """walks the output dir and returns the first subdir containing pngs"""
        for dirpath, _, filenames in os.walk(root):
            if any(fn.lower().endswith(".png") for fn in filenames):
                return dirpath
        return root

    image_root = find_deepest_image_dir(args.images_dir)
    logger.info(f"found generated images under: {image_root}")

    # compute metrics at the detected path
    clip_score = compute_clip_scores(image_root, prompts)
    fid_score = compute_fid(image_root, coco_stats="val2014.npz")
    total_time = time.time() - start_time

    results = {
        "num_images": len(prompts),
        "clip_score": clip_score,
        "fid_score": fid_score,
        "total_time_sec": total_time,
        "images_dir": image_root,
    }

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"saved results to {args.output_json}")
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
