# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: the full `TT_MoonViT_Vision_Model` (patch embed, all 27 encoder blocks, final norm, patch merger,
projector) against the reference `KimiK3VisionModel` in fp32 (`goldens.to_fp32`), on small images.
Not the bf16 reference: its own rounding is larger than the device's error. `test_full_size.py`
runs it on a full-size image against the saved fp32 outputs.

`test_vision_model_small_image` is the CI end-to-end check: a 48x48-patch image (672x672 px), so the
CPU reference takes seconds. CI runs it with the checkpoint's weights. The config and kernels stay at
production size."""

from __future__ import annotations

import copy

import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    SYNTHETIC,
    REAL,
    assert_quality,
    ATTENTION_GRIDS,
    RING,
    random_patches,
    ring_parallel,
)
from models.experimental.kimi_k3_moonvit_v2.tests.goldens import to_fp32
from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Vision_Model


# (t, h, w) in patches for `test_vision_model_small_image`: 2,304 patches, 576 after the 2x2 merge.
SMALL_IMAGE_GRID = [(1, 48, 48)]
RELATIVE_RMSE = 0.07


def check_vision_model(mesh_device, sp_axis, num_links, config, reference, grid) -> None:
    """The model on the ring on random patches of `grid`, against the fp32 reference computed here."""
    sequence_parallel, ccl = ring_parallel(mesh_device, sp_axis, num_links)
    tt = TT_MoonViT_Vision_Model(config, mesh_device=mesh_device, sequence_parallel=sequence_parallel, ccl=ccl)
    tt.load_torch_state_dict(reference.state_dict())
    pixels, grid_thws = random_patches(config, grid)
    with torch.no_grad():
        (golden,) = to_fp32(copy.deepcopy(reference))(pixels.float(), grid_thws)
    assert_quality(golden, tt.run(pixels, grid_thws), relative_rmse=RELATIVE_RMSE)


@RING
@SYNTHETIC
@REAL
@ATTENTION_GRIDS
def test_vision_model(mesh_device, sp_axis, num_links, config, reference, grid):
    check_vision_model(mesh_device, sp_axis, num_links, config, reference, grid)


@RING
@SYNTHETIC
@REAL
def test_vision_model_small_image(mesh_device, sp_axis, num_links, config, reference):
    check_vision_model(mesh_device, sp_axis, num_links, config, reference, SMALL_IMAGE_GRID)
