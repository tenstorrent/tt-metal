# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# The hi-res wash-out is a SAMPLING effect (text-guidance dilution at high token
# counts), not a compute bug (denoiser PCC 0.998 at all t/seq). Test the fix:
# counter the ~2.5x dilution with higher guidance_scale at 1024px. A recovered
# latent std (~0.9+) and a contrasty image confirm the diagnosis + the fix.

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
def test_hires_guidance_fix(*, mesh_device, submesh_shape, tp_axis) -> None:
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)
    # baseline (preset gw=7) then higher gw to counter ~2.5x dilution at 1024px
    for gw in (None, 12.0, 18.0, 25.0):
        img = pipe(PROMPT, height=1024, width=1024, preset="V4_TURBO_12", seed=1234, guidance_scale=gw)
        zstd = pipe.step_trace[-1][4]
        tag = "preset7" if gw is None else f"gw{int(gw)}"
        out = f"/localdev/cglagovich/ideogram4_1024_{tag}.png"
        Image.fromarray(img).save(out)
        logger.info(f"HIRES-FIX 1024px {tag}: final z_std={zstd:.3f} image_std={img.std():.1f} -> {out}")
