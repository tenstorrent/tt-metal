# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger

GOLDEN_ROOT = Path("/mnt/MLPerf/huggingface/sdxl_unet_loop_golden")


def golden_ci_mode(is_ci_env, is_ci_v2_env):
    # Only trust the cache in CIv1 with the shared mount actually present.
    if not is_ci_env or is_ci_v2_env:
        return False
    if not GOLDEN_ROOT.parent.is_dir():
        logger.warning(f"SDXL golden: {GOLDEN_ROOT.parent} not mounted; golden cache disabled")
        return False
    return True


def mlperf_writable():
    return os.getenv("MLPERF_READ_ONLY", "true").strip().lower() == "false"


def load_golden(name, expected_metadata):
    path = GOLDEN_ROOT / name
    if not path.is_file():
        logger.info(f"SDXL golden: {path} not found; computing torch reference")
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.warning(f"SDXL golden: failed to read {path} ({e}); computing torch reference")
        return None
    if payload.get("metadata") != expected_metadata:
        logger.warning(
            f"SDXL golden: metadata mismatch at {path}; computing torch reference.\n"
            f"  cached:   {payload.get('metadata')}\n  expected: {expected_metadata}"
        )
        return None
    logger.info(f"SDXL golden: using cached reference from {path}")
    return payload


def save_golden(name, metadata, golden):
    if not mlperf_writable():
        logger.info(f"SDXL golden: MLPerf mounted read-only, not saving {name}")
        return
    path = GOLDEN_ROOT / name
    try:
        GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata, "golden": golden}, path)
        logger.info(f"SDXL golden: saved {len(golden)} iterations to {path}")
    except Exception as e:
        logger.warning(f"SDXL golden: failed to save {path} ({e})")
