# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for ATSS Swin-L DyHead PCC tests.

Checkpoint path is resolved from common.py (auto-downloads if missing).
Override with env var ATSS_CKPT if needed.
"""

import os
from pathlib import Path

import pytest
import torch


def _mmdet_importable():
    try:
        from mmdet.apis import init_detector  # noqa: F401

        return True
    except ImportError:
        return False


def _get_config_and_checkpoint():
    from models.experimental.atss_swin_l_dyhead.common import ATSS_CHECKPOINT, ATSS_CONFIG

    config = os.environ.get("ATSS_CONFIG", ATSS_CONFIG)
    ckpt = os.environ.get("ATSS_CKPT", ATSS_CHECKPOINT)
    return config, ckpt


MMDET_SKIP = "mmdet not importable"
CKPT_SKIP = "ATSS checkpoint not found. Set ATSS_CKPT env var."


@pytest.fixture(scope="module")
def atss_ckpt_path():
    _, ckpt_path = _get_config_and_checkpoint()
    if not Path(ckpt_path).is_file():
        pytest.skip(CKPT_SKIP)
    return ckpt_path


@pytest.fixture(scope="module")
def atss_config_path():
    config_path, _ = _get_config_and_checkpoint()
    if not Path(config_path).is_file():
        pytest.skip("ATSS config not found")
    return config_path


@pytest.fixture(scope="module")
def atss_mmdet_model(atss_config_path, atss_ckpt_path):
    """Load full mmdet model for reference comparisons."""
    if not _mmdet_importable():
        pytest.skip(MMDET_SKIP)
    from mmdet.apis import init_detector

    model = init_detector(atss_config_path, atss_ckpt_path, device="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def atss_ref_model(atss_ckpt_path):
    """Load standalone reference model."""
    from models.experimental.atss_swin_l_dyhead.reference.model import (
        build_atss_model,
        load_mmdet_checkpoint,
    )

    model = build_atss_model()
    load_mmdet_checkpoint(model, atss_ckpt_path)
    model.eval()
    return model


@pytest.fixture(scope="module")
def sample_input_640():
    torch.manual_seed(42)
    return torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.float32)


@pytest.fixture(scope="module")
def reset_seeds():
    torch.manual_seed(42)
