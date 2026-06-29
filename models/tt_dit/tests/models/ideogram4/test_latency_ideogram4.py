# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# End-to-end latency of the device-resident Ideogram4Pipeline (TP=4, all models
# on device). Warms the program cache, then times encode / denoise / decode.
# Wall-clock (program-cache warm); ttnn trace capture would remove per-op host
# dispatch on top of this.
# =============================================================================

import pytest
from loguru import logger

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
@pytest.mark.parametrize(("height", "width", "preset"), [(512, 512, "V4_TURBO_12")], ids=["512px_turbo12"])
def test_latency(*, mesh_device, submesh_shape, tp_axis, height, width, preset) -> None:
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)

    pipe(PROMPT, height=height, width=width, preset=preset, seed=1)  # warmup (compile + cache)
    pipe(PROMPT, height=height, width=width, preset=preset, seed=2)  # timed (warm)

    t = pipe.timings
    logger.info(
        f"LATENCY TABLE {height}px {preset}: total {t['total']:.2f}s | encode {t['encode']:.2f}s | "
        f"denoise {t['denoise']:.2f}s = {t['denoise_per_step']*1000:.0f}ms/step | decode {t['decode']:.2f}s"
    )
