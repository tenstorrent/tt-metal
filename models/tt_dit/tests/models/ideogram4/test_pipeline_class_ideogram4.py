# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Drive the Ideogram4Pipeline class end-to-end on device (TP=4, all models
# resident: encoder + cond/uncond transformers + VAE), real weights.
# =============================================================================

import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.ideogram4.pipeline import Ideogram4Pipeline

PROMPT = "a watercolor painting of a red panda reading a book under a cherry tree, soft morning light"


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [pytest.param((2, 4), (1, 4), 1, id="tp4")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
@pytest.mark.parametrize(("height", "width"), [(512, 512)], ids=["512px"])
def test_pipeline_class(*, mesh_device, submesh_shape, tp_axis, height, width) -> None:
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)
    img = pipe(PROMPT, height=height, width=width, preset="V4_TURBO_12", seed=1234)

    out = "/localdev/cglagovich/ideogram4_pipeline_class.png"
    Image.fromarray(img).save(out)
    logger.info(f"pipeline class saved {out} shape={img.shape} std={img.std():.1f}")
    assert img.shape == (height, width, 3)
    assert img.std() > 1.0
