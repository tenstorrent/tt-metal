# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ATSS-specific config and constants.
The Swin-L backbone itself lives in models/experimental/swin_l/ —
this file only contains ATSS detection model settings.

Environment variable overrides
-------------------------------
ATSS_CHECKPOINT   Path to the .pth checkpoint file. If set and the file
                  exists, it is used directly (skipping auto-download).
ATSS_CONFIG       Path to the mmdet config .py file. If set and the file
                  exists, it is used directly.
"""

import os
import subprocess
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _MODEL_DIR / "weights"

_CHECKPOINT_FILENAME = "atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth"
_CONFIG_FILENAME = "atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco.py"
_MMDET_CONFIG_NAME = "atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco"

# ATSS input resolution (mmdet config: scale=(2000, 1200) with keep_ratio)
ATSS_INPUT_H = 1200
ATSS_INPUT_W = 2000

# Which Swin-L stages to output (skip stage 0 for ATSS)
ATSS_OUT_INDICES = (1, 2, 3)

# FPN config
ATSS_FPN_IN_CHANNELS = [384, 768, 1536]  # from Swin-L stages 1, 2, 3
ATSS_FPN_OUT_CHANNELS = 256
ATSS_FPN_NUM_OUTS = 5

# DyHead config
ATSS_DYHEAD_NUM_BLOCKS = 6
ATSS_DYHEAD_IN_CHANNELS = 256
ATSS_DYHEAD_OUT_CHANNELS = 256

# ATSS Head config
ATSS_NUM_CLASSES = 80
ATSS_NUM_ANCHORS = 1
ATSS_STRIDES = (8, 16, 32, 64, 128)

# Preprocessing
ATSS_PIXEL_MEAN = (123.675, 116.28, 103.53)
ATSS_PIXEL_STD = (58.395, 57.12, 57.375)
ATSS_PAD_SIZE_DIVISOR = 128

# Post-processing
ATSS_SCORE_THR = 0.05
ATSS_NMS_IOU_THR = 0.6
ATSS_NMS_PRE = 1000
ATSS_MAX_PER_IMG = 100


def _download_checkpoint():
    """Download the ATSS checkpoint via ``mim download mmdet``."""
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ATSS] Checkpoint not found. Downloading to {_WEIGHTS_DIR} ...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "mim",
                "download",
                "mmdet",
                "--config",
                _MMDET_CONFIG_NAME,
                "--dest",
                str(_WEIGHTS_DIR),
            ],
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise FileNotFoundError(
            f"Auto-download failed ({exc}). Install openmim + mmdet and retry, or "
            f"manually place the checkpoint at: {_WEIGHTS_DIR / _CHECKPOINT_FILENAME}\n"
            f"  pip install openmim && mim install mmdet mmengine\n"
            f"  mim download mmdet --config {_MMDET_CONFIG_NAME} --dest {_WEIGHTS_DIR}"
        ) from exc


def get_checkpoint_path() -> str:
    """Return the path to the ATSS .pth checkpoint.

    Resolution order:
      1. ATSS_CHECKPOINT env var (if set and file exists)
      2. Local weights/ directory
      3. Auto-download via ``mim download mmdet``
    """
    env = os.environ.get("ATSS_CHECKPOINT")
    if env and Path(env).is_file():
        return env

    path = _WEIGHTS_DIR / _CHECKPOINT_FILENAME
    if path.is_file():
        return str(path)
    _download_checkpoint()
    if path.is_file():
        return str(path)
    raise FileNotFoundError(
        f"Checkpoint not found at {path} even after download attempt.\n"
        f"Set ATSS_CHECKPOINT env var, manually place the file, or run:\n"
        f"  mim download mmdet --config {_MMDET_CONFIG_NAME} --dest {_WEIGHTS_DIR}"
    )


def get_config_path() -> str:
    """Return the path to the ATSS mmdet config .py file.

    Resolution order:
      1. ATSS_CONFIG env var (if set and file exists)
      2. Local weights/ directory
    """
    env = os.environ.get("ATSS_CONFIG")
    if env and Path(env).is_file():
        return env

    path = _WEIGHTS_DIR / _CONFIG_FILENAME
    if path.is_file():
        return str(path)
    raise FileNotFoundError(f"Config not found at {path}.\n" f"Set ATSS_CONFIG env var or place the file manually.")


ATSS_CHECKPOINT = get_checkpoint_path()
ATSS_CONFIG = get_config_path()
