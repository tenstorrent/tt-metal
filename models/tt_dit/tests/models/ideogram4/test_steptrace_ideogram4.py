# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Per-step latent-std trajectory at 512px vs 1024px to localize WHERE the hi-res
# latent collapses (gradual compounding vs a specific t-value / step).

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
def test_step_trace(*, mesh_device, submesh_shape, tp_axis) -> None:
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)
    pipe._verbose_steps = True
    for h in (512, 1024):
        pipe(PROMPT, height=h, width=h, preset="V4_TURBO_12", seed=1234)
        zs = [s[4] for s in pipe.step_trace]
        ts = [s[0] for s in pipe.step_trace]
        logger.info(f"TRACE {h}px z_std/step: {[f'{z:.3f}' for z in zs]}")
        logger.info(f"TRACE {h}px t/step:     {[f'{t:.3f}' for t in ts]}")
