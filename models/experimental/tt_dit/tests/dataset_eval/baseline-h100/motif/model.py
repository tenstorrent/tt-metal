# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import time
import json
import urllib
import torch
import pytest
import subprocess
import statistics
from loguru import logger
from tqdm import tqdm
from PIL import Image

from clip_encoder import CLIPEncoder
from fid_score import calculate_fid_score

# === constants ===
COCO_CAPTIONS_DOWNLOAD_PATH = (
    "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/"
    "text_to_image/coco2014/captions/captions_source.tsv"
)
OUT_DIR = "test_reports"
RESULTS_FILE = "motif_test_results.json"
MOTIF_DIR = "./Motif-Image-6B-Preview"


@pytest.mark.parametrize(
    "model_name,image_w,image_h,guidance_scale,num_inference_steps",
    [("motif", 1024, 1024, 7.5, 50)],
)
@pytest.mark.parametrize("captions_path", ["captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["val2014.npz"])
def test_accuracy_motif(
    model_name,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    captions_path,
    coco_statistics_path,
    evaluation_range,
):
    start_from, num_prompts = evaluation_range
    prompts = load_coco_prompts(captions_path, start_from, num_prompts)
    os.makedirs(OUT_DIR, exist_ok=True)

    # === device setup ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    # === verify motif dir ===
    config_path = os.path.join(MOTIF_DIR, "configs", "mmdit_xlarge_hq.json")
    ckpt_path = os.path.join(MOTIF_DIR, "checkpoints", "pytorch_model_fsdp.bin")
    if not os.path.isfile(config_path) or not os.path.isfile(ckpt_path):
        raise RuntimeError(f"missing model files. expected:\n" f"  - {config_path}\n" f"  - {ckpt_path}")

    # === begin eval ===
    logger.info(f"generating {num_prompts} images via motif native pipeline...")
    os.makedirs("generated_images", exist_ok=True)

    clip = CLIPEncoder()
    images, clip_scores, times = [], [], []

    for i, prompt in enumerate(tqdm(prompts, desc="Generating images", ncols=80)):
        logger.info(f"[{i+1}/{num_prompts}] prompt: {prompt[:80]}...")
        output_dir = "generated_images"
        t0 = time.time()

        # === run motif inference via subprocess ===
        cmd = [
            "python",
            os.path.join(MOTIF_DIR, "inference.py"),
            "--model-config",
            config_path,
            "--model-ckpt",
            ckpt_path,
            "--steps",
            str(num_inference_steps),
            "--resolution",
            str(image_w),
            "--batch-size",
            "1",
            "--output-dir",
            output_dir,
        ]

        # write the prompt to a temp text file (Motif expects --prompt-file)
        prompt_file = os.path.join(output_dir, f"prompt_{i+1}.txt")
        with open(prompt_file, "w") as f:
            f.write(prompt)

        cmd = [
            "python",
            os.path.join(MOTIF_DIR, "inference.py"),
            "--model-config",
            config_path,
            "--model-ckpt",
            ckpt_path,
            "--steps",
            str(num_inference_steps),
            "--resolution",
            str(image_w),
            "--batch-size",
            "1",
            "--output-dir",
            output_dir,
            "--prompt-file",
            prompt_file,
        ]

        subprocess.run(cmd, check=True)

        t1 = time.time()
        elapsed = t1 - t0

        # === pick the latest generated image ===
        generated_files = sorted(
            [f for f in os.listdir(output_dir) if f.endswith(".png")],
            key=lambda x: os.path.getmtime(os.path.join(output_dir, x)),
            reverse=True,
        )
        if not generated_files:
            raise RuntimeError("no output images found after inference run.")
        output_path = os.path.join(output_dir, generated_files[0])
        img_pil = Image.open(output_path).convert("RGB")

        # === clip score ===
        clip_score = 100 * clip.get_clip_score(prompt, img_pil).item()

        images.append(img_pil)
        clip_scores.append(clip_score)
        times.append(elapsed)

        tqdm.write(f"✓ image {i+1} ({elapsed:.2f}s, CLIP={clip_score:.2f})")

    # === compute metrics ===
    avg_clip = sum(clip_scores) / len(clip_scores)
    std_clip = statistics.stdev(clip_scores) if len(clip_scores) > 1 else 0.0

    fid_score = "N/A"
    if len(images) > 1 and os.path.isfile(coco_statistics_path):
        fid_score = calculate_fid_score(images, coco_statistics_path)
    elif len(images) > 1:
        logger.warning("fid skipped: coco stats file missing.")

    avg_time = sum(times) / len(times)
    min_time, max_time = min(times), max(times)

    # === log results ===
    logger.info("=== RESULTS ===")
    logger.info(f"average clip: {avg_clip:.2f}")
    logger.info(f"std dev clip: {std_clip:.2f}")
    logger.info(f"fid score: {fid_score}")
    logger.info(f"avg time: {avg_time:.2f}s  (min={min_time:.2f}s, max={max_time:.2f}s)")

    # === save to json ===
    results = {
        "model": model_name,
        "metadata": {
            "device": device.upper(),
            "start_from": start_from,
            "num_prompts": num_prompts,
            "image_w": image_w,
            "image_h": image_h,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "backend": "motif_native_pipeline",
        },
        "metrics": {
            "average_clip": avg_clip,
            "deviation_clip": std_clip,
            "fid_score": fid_score,
            "average_inference_time": avg_time,
            "min_inference_time": min_time,
            "max_inference_time": max_time,
        },
        "individual_clip_scores": [float(x) for x in clip_scores],
    }

    with open(f"{OUT_DIR}/{RESULTS_FILE}", "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"results saved to {OUT_DIR}/{RESULTS_FILE}")

    print(f"\n🎉 Test completed successfully!")
    print(f"📊 Avg CLIP: {avg_clip:.2f} | FID: {fid_score}")
    print(f"⚡ Avg Time: {avg_time:.2f}s | Saved: generated_images/")


# === helper ===
def load_coco_prompts(captions_path, start_from, num_prompts):
    if not os.path.isfile(captions_path):
        logger.info(f"{captions_path} not found, downloading...")
        urllib.request.urlretrieve(COCO_CAPTIONS_DOWNLOAD_PATH, captions_path)
        logger.info("download complete.")
    prompts = []
    with open(captions_path, "r") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i < start_from:
                continue
            if i >= start_from + num_prompts:
                break
            prompts.append(row[2])
    return prompts
