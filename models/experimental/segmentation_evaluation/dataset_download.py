# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
import shutil

# Install kagglehub if not installed
try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

# When the similar error is faced "Error: Required subfolder 'lgg-mri-segmentation/kaggle_3m/TCGA_CS_4944_20010208' not found in '/home/ubuntu/.cache/kagglehub/datasets/mateuszbuda/lgg-mri-segmentation/versions/2'",
# Try deleting the "imageset" folder in "models/experimental/segmentation_evaluation" directory, and uncomment the below line and try running again.
# shutil.rmtree("/home/ubuntu/.cache/kagglehub/datasets/mateuszbuda/lgg-mri-segmentation/versions/2")

# Define dataset and target folder
DATASET = "mateuszbuda/lgg-mri-segmentation"
TARGET_FOLDER = "models/experimental/segmentation_evaluation/imageset"
REQUIRED_SUBFOLDER = "lgg-mri-segmentation/kaggle_3m/TCGA_CS_4944_20010208"

# Download the dataset
print(f"Downloading dataset: {DATASET}...")
dataset_path = kagglehub.dataset_download(DATASET)
print("dataset_path: ", dataset_path)

if not os.path.exists(dataset_path):
    print(f"Error: Dataset was not downloaded to '{dataset_path}'.")
    sys.exit(1)

# ✅ **Check if dataset_path contains files/folders**
if not os.listdir(dataset_path):
    print(f"Error: Dataset directory '{dataset_path}' is empty.")
    sys.exit(1)

# Ensure target folder exists
os.makedirs(TARGET_FOLDER, exist_ok=True)

source_subfolder = os.path.join(dataset_path, REQUIRED_SUBFOLDER)

# **Check if required subfolder exists**
if not os.path.exists(source_subfolder):
    print(f"Error: Required subfolder '{REQUIRED_SUBFOLDER}' not found in '{dataset_path}'")
    sys.exit(1)

shutil.move(source_subfolder, os.path.join(TARGET_FOLDER, os.path.basename(REQUIRED_SUBFOLDER)))

print(f"Successfully moved '{REQUIRED_SUBFOLDER}' to '{TARGET_FOLDER}'")

os.makedirs("models/experimental/segmentation_evaluation/pred_image_set", exist_ok=True)
