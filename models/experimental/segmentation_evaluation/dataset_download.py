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

shutil.rmtree("/home/ubuntu/.cache/kagglehub/datasets/mateuszbuda/lgg-mri-segmentation/versions/2")

# Define dataset and target folder
DATASET = "mateuszbuda/lgg-mri-segmentation"
TARGET_FOLDER = "models/experimental/segmentation_evaluation/imageset"
# REQUIRED_SUBFOLDER = "lgg-mri-segmentation/kaggle_3m/TCGA_CS_4944_20010208"

# Download the dataset
logger.info(f"Downloading dataset: {DATASET}...")
dataset_path = kagglehub.dataset_download(DATASET)
logger.info(f"dataset_path: {dataset_path}")

if not os.path.exists(dataset_path):
    logger.error(f"Error: Dataset was not downloaded to '{dataset_path}'.")
    sys.exit(1)

# Check if dataset_path contains files/folders
if not os.listdir(dataset_path):
    logger.error(f"Error: Dataset directory '{dataset_path}' is empty.")
    sys.exit(1)

# Ensure target folder exists
os.makedirs(TARGET_FOLDER, exist_ok=True)

# source_subfolder = os.path.join(dataset_path, REQUIRED_SUBFOLDER)

# Check if required subfolder exists
# if not os.path.exists(source_subfolder):
#     logger.error(
#         f"Error: Required subfolder '{REQUIRED_SUBFOLDER}' not found in '{dataset_path}'\n"
#         f"This usually happens when the dataset download failed or the folder is corrupted.\n"
#         f"Try deleting the folder:\n  'models/experimental/segmentation_evaluation/imageset'\n"
#         f"Then rerun this script to download it again."
#     )
#     sys.exit(1)

source_root = os.path.join(dataset_path, "lgg-mri-segmentation/kaggle_3m")
for folder in os.listdir(source_root):
    src = os.path.join(source_root, folder)
    dst = os.path.join(TARGET_FOLDER, folder)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)


# shutil.move(source_subfolder, os.path.join(TARGET_FOLDER, os.path.basename(REQUIRED_SUBFOLDER)))

logger.info(f"Successfully moved '{source_root}' to '{TARGET_FOLDER}'")

os.makedirs("models/experimental/segmentation_evaluation/pred_image_set", exist_ok=True)
