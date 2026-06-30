# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Drive the Ideogram4Pipeline class end-to-end on device (TP=4, all models
# resident: encoder + cond/uncond transformers + VAE), real weights.
#
# Resolution status (task #24):
#   * 512px  -> FAITHFUL (image std ~46).
#   * 1024px / 2048px -> the denoiser under-converges the latent at 4096/16384 tokens
#     (latent std ~0.59 vs ~0.98 at 512px); the image decodes flat (std ~10) on BOTH
#     the device AND the reference host VAE, so the VAE is ruled out. Preset-independent
#     (TURBO_12 and QUALITY_48 both flat). The per-call denoiser is verified at 4096
#     tokens (PCC 0.998) and the schedule matches the reference exactly, so the leading
#     suspect is real-weight bf16 fidelity compounding over the 4x longer attention
#     (consistent with the earlier outlier finding). OPEN: needs a reference-pipeline
#     trajectory comparison and/or higher attention fidelity at long sequence.
# =============================================================================

import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.ideogram4.pipeline import Ideogram4Pipeline
from .test_generate_ideogram4 import PROMPT_JSON

# Ideogram 4 was trained exclusively on structured-JSON captions; the in-distribution
# JSON prompt is the known-good format (plain prose triggers in-weights moderation
# washout at >=1024px per the status notes). Reuse the verified caption.
PROMPT = PROMPT_JSON


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [
        pytest.param((2, 4), (1, 4), 1, id="tp4"),
        pytest.param((4, 2), (4, 2), 1, id="sp4tp2"),  # full-mesh SP4xTP2 denoiser default
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}], indirect=True
)
@pytest.mark.parametrize(
    ("height", "width"), [(512, 512), (1024, 1024), (2048, 2048)], ids=["512px", "1024px", "2048px"]
)
@pytest.mark.parametrize("preset", ["V4_TURBO_12", "V4_QUALITY_48"])
def test_pipeline_class(*, mesh_device, submesh_shape, tp_axis, height, width, preset) -> None:
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)
    img = pipe(PROMPT, height=height, width=width, preset=preset, seed=1234)

    out = f"/data/cglagovich/ideogram4_pipeline_class_{height}_{preset}.png"
    Image.fromarray(img).save(out)
    logger.info(f"pipeline class saved {out} shape={img.shape} std={img.std():.1f} preset={preset}")
    assert img.shape == (height, width, 3)
    assert img.std() > 1.0
