# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Drive the Ideogram4Pipeline class end-to-end on device (TP=4, all models
# resident: encoder + cond/uncond transformers + VAE) with real weights, across
# presets and resolutions. Saves each image and asserts it is non-degenerate
# (correct shape, std > 1). A generation smoke test, not a PCC gate.
# =============================================================================

import os

import numpy as np
import pytest
from loguru import logger

from ....pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from ....utils.test import line_params, ring_params

# The pipeline needs the gated fp8 checkpoint (create_pipeline raises ValueError without it);
# skip (don't error) when it isn't configured, matching the component tests.
_NEEDS_WEIGHTS = pytest.mark.skipif(
    not os.environ.get("IDEOGRAM4_WEIGHTS"), reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)"
)


def _typography_prompt_json() -> str:
    """Typography-heavy in-distribution JSON caption that showcases Ideogram 4's
    best-in-class text rendering (crisp legible lettering + bbox layout + palette).

    Ideogram 4 was trained exclusively on structured-JSON captions; quoting the exact
    text strings and placing them with bbox is the reliable way to get them rendered.
    """
    import json

    caption = {
        "aspect_ratio": "1:1",
        "high_level_description": (
            "A bold retro travel poster with large, crisp typography. A condensed sans-serif "
            'headline reading "EXPLORE THE WILD" arcs across the top, a clean letter-spaced '
            'subtitle "NATIONAL PARKS · EST. 1872" sits beneath it, and a small banner at the '
            'bottom reads "ADVENTURE AWAITS". Set over a stylized sunrise mountain landscape '
            "with pine silhouettes; flat vintage screen-print look, sharp legible lettering."
        ),
        "colour_palette": ["#F4A259", "#2E4600", "#1B3A4B", "#F2E8CF", "#BC4749"],
        "compositional_deconstruction": {
            "background": (
                "Stylized sunrise sky in warm cream and orange bands, layered mountain ridges in "
                "deep green and teal, a row of pine-tree silhouettes along the lower third, flat "
                "screen-print texture with subtle grain."
            ),
            "elements": [
                {
                    "type": "text",
                    "bbox": [70, 70, 290, 930],
                    "text": "EXPLORE THE WILD",
                    "desc": "Large bold condensed sans-serif headline in cream, all caps, gentle "
                    "upward arc, high contrast against the sky.",
                },
                {
                    "type": "text",
                    "bbox": [300, 180, 400, 820],
                    "text": "NATIONAL PARKS · EST. 1872",
                    "desc": "Smaller clean uppercase subtitle in warm orange, generously "
                    "letter-spaced, centered beneath the headline.",
                },
                {
                    "type": "text",
                    "bbox": [840, 250, 960, 750],
                    "text": "ADVENTURE AWAITS",
                    "desc": "Small bold uppercase banner text in cream on a deep-red rounded ribbon.",
                },
                {
                    "type": "obj",
                    "bbox": [410, 60, 880, 940],
                    "desc": "Layered mountain range at sunrise with a bold circular sun disc, deep "
                    "green and teal ridges, vintage poster shading.",
                },
            ],
        },
    }
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


# Typography showcase prompt (JSON is the known-good format; plain prose washes out
# at >=1024px per the in-weights-moderation finding).
PROMPT = _typography_prompt_json()


# Trace region for the per-branch transformer-forward traces (cond + uncond).
# 60MB headroom; bump if "trace region" OOM at higher resolution.
# l1_small_size=32KB: only the VAE conv draws from L1_SMALL (proven sufficient in the VAE tests);
# the denoiser matmuls use none, so 64KB over-reserved the shared L1 region and made the SP4xTP2
# path's resident ring/all-gather buffers clash with the matmul circular buffers.
_LINE_PIPE = {**line_params, "l1_small_size": 32768, "trace_region_size": 60000000}
_RING_PIPE = {**ring_params, "l1_small_size": 32768, "trace_region_size": 60000000}


# Pipeline mesh configs: the parallel config (tp/sp/num_links/topology) is DISCOVERED from the
# mesh shape via Ideogram4Pipeline._PRESETS, so tests only pick a mesh shape + its device_params.
_PIPE_CONFIGS = [
    pytest.param((2, 4), _LINE_PIPE, id="sp4tp2"),  # BH loudbox 2x4: SP4 x TP2
    pytest.param((4, 8), _RING_PIPE, id="bh_galaxy_sp8tp4"),  # BH Galaxy: TP4 x SP8 (Ring)
]


@_NEEDS_WEIGHTS
@pytest.mark.parametrize(("mesh_device", "device_params"), _PIPE_CONFIGS, indirect=True)
@pytest.mark.parametrize(
    ("height", "width"), [(512, 512), (1024, 1024), (2048, 2048)], ids=["512px", "1024px", "2048px"]
)
@pytest.mark.parametrize("preset", ["V4_TURBO_12", "V4_QUALITY_48"])
def test_pipeline_class(*, mesh_device, height, width, preset) -> None:
    # Parallel config is discovered from the mesh shape (_PRESETS); no per-test tp/sp/links wiring.
    pipe = Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device, height=height, width=width)
    img = pipe(prompts=[PROMPT], preset=preset, seed=1234, traced=False)[0]

    out_dir = os.environ.get("IDEOGRAM4_OUT_DIR", "generated")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"ideogram4_pipeline_class_{height}_{preset}.png")
    img.save(out)
    arr = np.asarray(img)
    logger.info(f"pipeline class saved {out} shape={arr.shape} std={arr.std():.1f} preset={preset}")
    assert arr.shape == (height, width, 3)
    assert arr.std() > 1.0


@_NEEDS_WEIGHTS
@pytest.mark.parametrize(("mesh_device", "device_params"), _PIPE_CONFIGS, indirect=True)
def test_2048_presets(*, mesh_device) -> None:
    """UNTRACED 2048px generation across all 3 sampler presets, saved for manual review.

    Builds the pipeline ONCE and loops the presets (avoids a per-preset model reload).
    Untraced (traced=False) per the serving plan; 2048px doesn't benefit from tracing.
    """
    pipe = Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device, height=2048, width=2048)
    for preset in ("V4_TURBO_12", "V4_DEFAULT_20", "V4_QUALITY_48"):
        img = pipe(prompts=[PROMPT], preset=preset, seed=1234, traced=False)[0]
        out_dir = os.environ.get("IDEOGRAM4_OUT_DIR", "generated")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"ideogram4_2048_{preset}.png")
        img.save(out)
        arr = np.asarray(img)
        logger.info(f"2048px UNTRACED {preset}: saved {out} std={arr.std():.1f}")
        assert arr.shape == (2048, 2048, 3)
        assert arr.std() > 1.0
