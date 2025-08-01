# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
import shutil
from loguru import logger

# Install kagglehub if not installed
try:
    import kagglehub
except ImportError:
    logger.info("Installing kagglehub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub


def download_lgg_dataset():
    # Define dataset and target folder
    DATASET = "mateuszbuda/lgg-mri-segmentation"
    TARGET_FOLDER = "models/experimental/segmentation_evaluation/imageset"

    # Optionally remove previous cache if needed (only if you always want fresh download)
    cache_dir = "/home/ubuntu/.cache/kagglehub/datasets/mateuszbuda/lgg-mri-segmentation/versions/2"
    if os.path.exists(cache_dir):
        logger.info(f"Removing cached dataset: {cache_dir}")
        shutil.rmtree(cache_dir)

    # Download the dataset
    logger.info(f"Downloading dataset: {DATASET}...")
    dataset_path = kagglehub.dataset_download(DATASET)
    logger.info(f"Dataset downloaded to: {dataset_path}")

    if not os.path.exists(dataset_path):
        logger.error(f"Error: Dataset was not downloaded to '{dataset_path}'.")
        sys.exit(1)

    if not os.listdir(dataset_path):
        logger.error(f"Error: Dataset directory '{dataset_path}' is empty.")
        sys.exit(1)

    # Create target directory
    os.makedirs(TARGET_FOLDER, exist_ok=True)

    source_root = os.path.join(dataset_path, "lgg-mri-segmentation/kaggle_3m")
    for folder in os.listdir(source_root):
        src = os.path.join(source_root, folder)
        dst = os.path.join(TARGET_FOLDER, folder)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    logger.info(f"Successfully moved '{source_root}' to '{TARGET_FOLDER}'")

    # Ensure prediction output folder also exists
    os.makedirs("models/experimental/segmentation_evaluation/pred_image_set", exist_ok=True)


def download_ade20k_dataset(dest_root="models/demos/segformer/demo/validation_data_ade20k", max_samples=2000):
    if os.path.exists(dest_root) and len(os.listdir(os.path.join(dest_root, "images"))) >= max_samples:
        print(f"ADE20K dataset already exists with {len(os.listdir(os.path.join(dest_root, 'images')))} images.")
        return

    # Optionally remove previous cache if needed (only if you always want fresh download)
    cache_dir = "/home/ubuntu/.cache/kagglehub/datasets/awsaf49/ade20k-dataset/versions/2"
    if os.path.exists(cache_dir):
        logger.info(f"Removing cached dataset: {cache_dir}")
        shutil.rmtree(cache_dir)

    os.makedirs(dest_root, exist_ok=True)
    logger.info("Downloading ADE20K dataset from Kaggle...")

    dataset_path = kagglehub.dataset_download("awsaf49/ade20k-dataset")
    logger.info(f"ADE20K downloaded to: {dataset_path}")

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        logger.error("ADE20K dataset not properly downloaded or is empty.")
        return

    # Move only first 500 validation images and masks
    image_src = os.path.join(dataset_path, "ADEChallengeData2016", "images", "validation")
    mask_src = os.path.join(dataset_path, "ADEChallengeData2016", "annotations", "validation")

    image_dst = os.path.join(dest_root, "images")
    mask_dst = os.path.join(dest_root, "annotations")

    os.makedirs(image_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)

    img_files = sorted(os.listdir(image_src))[:max_samples]
    mask_files = sorted(os.listdir(mask_src))[:max_samples]

    for f in img_files:
        shutil.move(os.path.join(image_src, f), os.path.join(image_dst, f))

    for f in mask_files:
        shutil.move(os.path.join(mask_src, f), os.path.join(mask_dst, f))
    logger.info(f"Moved {len(img_files)} images and {len(mask_files)} masks to {dest_root}")


def main(model_name):
    if model_name == "vanilla_unet":
        download_lgg_dataset()
    elif model_name == "segformer":
        download_ade20k_dataset()


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "vanilla_unet"
    main(model_name)
