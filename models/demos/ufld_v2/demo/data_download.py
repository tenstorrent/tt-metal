# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import kagglehub
from loguru import logger

DATASET_PATH = "manideep1108/tusimple"
TARGET_FOLDER = "models/demos/ufld_v2/demo/image_data"
REQUIRED_IMAGES_SUBFOLDER = "TUSimple/test_set/clips"
REQUIRED_LABELS = "TUSimple/test_label.json"
path = kagglehub.dataset_download(DATASET_PATH)

os.makedirs(TARGET_FOLDER, exist_ok=True)
source_images_subfolder = os.path.join(path, REQUIRED_IMAGES_SUBFOLDER)
source_labels_subfolder = os.path.join(path, REQUIRED_LABELS)
shutil.move(source_images_subfolder, os.path.join(TARGET_FOLDER, os.path.basename(REQUIRED_IMAGES_SUBFOLDER)))
logger.info(f"Successfully moved '{REQUIRED_IMAGES_SUBFOLDER}' to '{TARGET_FOLDER}'")

shutil.move(source_labels_subfolder, os.path.join(TARGET_FOLDER, os.path.basename(REQUIRED_LABELS)))
logger.info(f"Successfully moved '{REQUIRED_LABELS}' to '{TARGET_FOLDER}'")
