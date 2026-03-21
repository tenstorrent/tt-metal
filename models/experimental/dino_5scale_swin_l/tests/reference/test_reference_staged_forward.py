# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test: PyTorch reference (DINOStagedForward) loads and runs backbone + neck.
Requires mmdet and checkpoint. If mmdet was installed with pip into user site-packages
(because venv site-packages is not writeable), run pytest with PYTHONPATH including
both the repo root (first!) and user site-packages, e.g.:
  export PYTHONPATH=/home/ubuntu/tt-metal:$HOME/.local/lib/python3.10/site-packages
"""
import os
import pytest
import torch

from models.experimental.dino_5scale_swin_l.common import DINO_INPUT_H, DINO_INPUT_W


def _mmdet_importable():
    try:
        from mmdet.apis import init_detector  # noqa: F401

        return True
    except ImportError:
        return False


def _get_config_and_checkpoint():
    from pathlib import Path

    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    config = base / "models/experimental/dino_5scale_swin_l/reference/dino_5scale_swin_l.py"
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
    ckpt = ckpt_dir / "dino_5scale_swin_l.pth"
    if not ckpt.is_file():
        ckpt = ckpt_dir / "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
    return str(config), str(ckpt)


# Skip if mmdet not importable (e.g. venv doesn't see user site-packages)
MMDET_SKIP_REASON = (
    "mmdet not importable. With python_env, pip often installs to user site-packages; "
    "run: export PYTHONPATH=/home/ubuntu/tt-metal:$HOME/.local/lib/python3.10/site-packages"
)


@pytest.mark.skipif(not _mmdet_importable(), reason=MMDET_SKIP_REASON)
@pytest.mark.skipif(
    not __import__("pathlib").Path(_get_config_and_checkpoint()[1]).is_file(),
    reason="Checkpoint not found; run: mim download mmdet --config dino-5scale_swin-l_8xb2-36e_coco --dest models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l",
)
def test_reference_staged_forward_backbone_and_neck():
    """Load reference, run backbone then neck, check shapes."""
    from models.experimental.dino_5scale_swin_l.reference.dino_staged_forward import DINOStagedForward

    config_path, ckpt_path = _get_config_and_checkpoint()
    ref = DINOStagedForward(config_path, ckpt_path, device="cpu")
    B = 1
    x = torch.rand(B, 3, DINO_INPUT_H, DINO_INPUT_W)
    feats = ref.forward_backbone(x)
    assert len(feats) == 4
    # C2, C3, C4, C5: channels 192, 384, 768, 1536
    assert feats[0].dim() == 4 and feats[0].shape[0] == B
    memory, spatial_shapes, level_start_index = ref.forward_neck(feats)
    assert memory.dim() == 3 and memory.shape[0] == B and memory.shape[2] == 256
    assert spatial_shapes.shape[0] == 5 and level_start_index.shape[0] == 6
