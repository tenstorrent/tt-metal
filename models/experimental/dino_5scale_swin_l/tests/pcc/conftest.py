# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Shared fixtures for Swin-L PCC tests.
"""

import os
from pathlib import Path

import pytest


def _mmdet_importable():
    try:
        from mmdet.apis import init_detector  # noqa: F401

        return True
    except ImportError:
        return False


def _get_config_and_checkpoint():
    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    config = base / "models/experimental/dino_5scale_swin_l/reference/dino_5scale_swin_l.py"
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
    ckpt = ckpt_dir / "dino_5scale_swin_l.pth"
    if not ckpt.is_file():
        ckpt = ckpt_dir / "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
    return str(config), str(ckpt)


MMDET_SKIP = "mmdet not importable; export PYTHONPATH=/home/ubuntu/tt-metal:$HOME/.local/lib/python3.10/site-packages"
CKPT_SKIP = (
    "Checkpoint not found; run: mim download mmdet --config dino-5scale_swin-l_8xb2-36e_coco "
    "--dest models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
)


@pytest.fixture(scope="module")
def swin_l_ref():
    """Module-scoped PyTorch Swin-L reference (loaded once, shared across tests)."""
    if not _mmdet_importable():
        pytest.skip(MMDET_SKIP)
    config_path, ckpt_path = _get_config_and_checkpoint()
    if not Path(ckpt_path).is_file():
        pytest.skip(CKPT_SKIP)
    from models.experimental.dino_5scale_swin_l.reference.swin_l_reference import SwinLReference

    return SwinLReference(config_path, ckpt_path)


@pytest.fixture(scope="module")
def swin_l_ckpt_path():
    """Return checkpoint path string."""
    _, ckpt_path = _get_config_and_checkpoint()
    if not Path(ckpt_path).is_file():
        pytest.skip(CKPT_SKIP)
    return ckpt_path
