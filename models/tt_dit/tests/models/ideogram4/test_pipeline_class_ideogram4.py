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

    # Warmup run: triggers JIT compile + program-cache fill (NOT timed).
    pipe(PROMPT, height=height, width=width, preset=preset, seed=1234)
    # Timed run: program cache is warm, so the timings exclude JIT compile.
    img = pipe(PROMPT, height=height, width=width, preset=preset, seed=1234)
    t = pipe.timings
    logger.info(
        f"WARM E2E LATENCY {height}px {preset}: total={t['total']:.2f}s | encode={t['encode']:.2f}s "
        f"| denoise={t['denoise']:.2f}s ({t['denoise_per_step']*1000:.0f}ms/step) | decode={t['decode']:.2f}s"
    )

    out = f"/data/cglagovich/ideogram4_pipeline_class_{height}_{preset}.png"
    Image.fromarray(img).save(out)
    logger.info(f"pipeline class saved {out} shape={img.shape} std={img.std():.1f} preset={preset}")
    assert img.shape == (height, width, 3)
    assert img.std() > 1.0
