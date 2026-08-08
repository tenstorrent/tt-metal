# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for ATSS Swin-L DyHead PCC tests.

Checkpoint and config paths are resolved by common.py which supports
env var overrides (ATSS_CHECKPOINT, ATSS_CONFIG) and auto-download.
"""

import pytest
import torch


def _mmdet_importable():
    try:
        from mmdet.apis import init_detector  # noqa: F401

        return True
    except ImportError:
        return False


MMDET_SKIP = "mmdet not importable"
CKPT_SKIP = "ATSS checkpoint not found. Set ATSS_CHECKPOINT env var."


@pytest.fixture(scope="module")
def atss_ckpt_path():
    from models.experimental.atss_swin_l_dyhead.common import get_checkpoint_path

    try:
        ckpt_path = get_checkpoint_path()
    except FileNotFoundError:
        pytest.skip(CKPT_SKIP)
    return ckpt_path


@pytest.fixture(scope="module")
def atss_config_path():
    from models.experimental.atss_swin_l_dyhead.common import get_config_path

    try:
        config_path = get_config_path()
    except FileNotFoundError:
        pytest.skip("ATSS config not found. Set ATSS_CONFIG env var.")
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
