# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.flux3_klein.upsampler_flux3_klein import DEVICE_PARAMS, Flux3KleinUpsampler


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_upsampler(*, mesh_device: ttnn.MeshDevice) -> None:
    upsampler = Flux3KleinUpsampler(mesh_device)
    requests = [
        ("A man walking his dog in the snow", None),
        ("Make it a night scene with street lights", Image.new("RGB", (1024, 1024), (120, 140, 90))),
    ]
    for prompt, image in requests:
        t0 = time.time()
        caption = upsampler(prompt, image)
        logger.info(f"{prompt!r} -> {len(caption)} characters in {time.time() - t0:.1f} s: {caption}")
        assert caption and caption != prompt
    upsampler.close()
