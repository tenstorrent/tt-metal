# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein serving: after ``warmup`` a request compiles nothing, at every warmed size, for prompts whose attended
text lengths differ from warmup's and from each other (160 rows up to the full 4096-row sequence), without and with a
reference of every served reference size.

The attended text length reaches ring SDPA as a device tensor of the request's plan, and the text encoder runs one of
the ENCODER_ROWS row counts that warmup compiled for every text bucket.

Env: FLUX3_KLEIN_SERVE_SIZES (default 1024, comma-separated sides or WxH), FLUX3_KLEIN_SERVE_OUT (image dir).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.flux3_klein.packing import MAX_TEXT_LENGTH
from ....pipelines.flux3_klein.pipeline_flux3_klein import Flux3KleinPipeline, Flux3KleinRequest
from ....utils.test import ring_params_8k

FILLER = (
    "a sunlit harbor town with pastel houses, fishing boats, gulls over the water, a lighthouse on the cliff, market "
    "stalls with fruit and flowers, children chasing a dog along the pier, clouds over distant hills,"
).split()
TEXT_ROW_TARGETS = (160, 560, 1360, 2480, MAX_TEXT_LENGTH)
# references of other shapes than warmup()'s bucket-filling squares: aspect kept, never cropped, every canvas
REFERENCE_SHAPES = (
    (0, 0),
    (1024, 1024),
    (2048, 2048),
    (6000, 4000),
    (1080, 1920),
    (4096, 1024),
    (1400, 1000),
    (256, 256),
)


def _prompt_for(rows: int, attended) -> str:
    """The shortest filler prompt whose attended text length is at least ``rows``."""
    prompt = lambda n: " ".join(FILLER[i % len(FILLER)] for i in range(n))  # noqa: E731
    words = 1
    while attended(prompt(words)) < rows:
        words *= 2
    lo, hi = words // 2, words
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if attended(prompt(mid)) < rows else (lo, mid)
    return prompt(hi)


def _reference(width: int, height: int) -> Image.Image:
    ramp = np.add.outer(np.linspace(0, 255, height), np.linspace(0, 127, width)).astype(np.uint8)
    return Image.fromarray(np.stack([ramp] * 3, axis=-1))


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_serving(*, mesh_device: ttnn.MeshDevice) -> None:
    sizes = tuple(
        tuple(map(int, s.split("x"))) if "x" in s else (int(s), int(s))
        for s in os.environ.get("FLUX3_KLEIN_SERVE_SIZES", "1024").split(",")
    )
    out = Path(os.environ.get("FLUX3_KLEIN_SERVE_OUT", "flux3_klein_serving"))
    out.mkdir(parents=True, exist_ok=True)

    pipeline = Flux3KleinPipeline.create(mesh_device)

    def attended_rows(prompt: str) -> int:
        return pipeline._prompt_encoder.tokenize(prompt, MAX_TEXT_LENGTH)[2]

    prompts = {f"text{rows}": _prompt_for(rows, attended_rows) for rows in TEXT_ROW_TARGETS}
    attended = {name: attended_rows(prompt) for name, prompt in prompts.items()}
    logger.info(f"SERVE attended text rows {attended}, warmup {attended_rows('warmup')}")
    assert attended_rows("warmup") not in attended.values(), attended
    assert len(set(attended.values())) == len(attended), attended
    assert attended[f"text{MAX_TEXT_LENGTH}"] == MAX_TEXT_LENGTH

    pipeline.warmup(sizes=sizes)
    failures = []
    for width, height in sizes:
        for ref_w, ref_h in REFERENCE_SHAPES:
            images = (_reference(ref_w, ref_h),) if ref_w else ()
            for name, prompt in prompts.items():
                ttnn.synchronize_device(mesh_device)
                programs = mesh_device.num_program_cache_entries()
                result = pipeline.generate(Flux3KleinRequest(prompt, width=width, height=height, seed=0, images=images))
                ttnn.synchronize_device(mesh_device)
                new_programs = mesh_device.num_program_cache_entries() - programs
                ref = f"{ref_w}x{ref_h}" if ref_w else "none"
                tag = f"{width}x{height} ref {ref} {name} prompt"
                logger.info(f"SERVE {tag}: new programs {new_programs}, {result.timings}")
                result.image.save(out / f"{width}x{height}_ref{ref}_{name}.png")
                if new_programs:
                    failures.append(f"{tag}: {new_programs} new programs")
                if float(np.asarray(result.image, dtype=np.float32).std()) < 10:
                    failures.append(f"{tag}: flat image")
    pipeline.close()
    assert not failures, "\n".join(failures)
