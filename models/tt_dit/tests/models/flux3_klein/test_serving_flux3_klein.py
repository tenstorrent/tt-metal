# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein serving: after ``warmup`` the DiT compiles nothing, at every warmed size, for prompts whose attended text
lengths differ from warmup's and from each other (at least 160 rows, at least 2000 rows, the full 4096-row sequence;
warmup's prompt attends 80), without a reference and with references that fill or only partly fill a reference token
bucket.

The prompt length and the reference's size within its bucket reach ring SDPA only as the device tensors the transformer
writes per request, so every request runs the programs warmup compiled for its bucket. A reference of a new shape still
compiles the VAE encoder, so the test encodes each reference once before its requests and asserts no new program-cache
entries around every request.

Env: FLUX3_KLEIN_SERVE_SIZES (default 1024, comma-separated), FLUX3_KLEIN_SERVE_OUT (image dir).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.flux3_klein.packing import DOWNSAMPLING_RATIO, MAX_TEXT_LENGTH
from ....pipelines.flux3_klein.pipeline_flux3_klein import (
    REFERENCE_TOKEN_BUCKETS,
    WARMUP_PROMPT,
    Flux3KleinPipeline,
    Flux3KleinRequest,
    prepare_reference,
    reference_bucket,
)
from ....utils.test import ring_params_8k

_FILLER = [
    "a",
    "sunlit",
    "harbor",
    "town",
    "with",
    "pastel",
    "houses,",
    "fishing",
    "boats,",
    "gulls",
    "over",
    "the",
    "water,",
    "a",
    "lighthouse",
    "on",
    "the",
    "cliff,",
    "market",
    "stalls",
    "with",
    "fruit",
    "and",
    "flowers,",
    "children",
    "chasing",
    "a",
    "dog",
    "along",
    "the",
    "pier,",
    "clouds",
    "over",
    "distant",
    "hills,",
]
TEXT_ROW_TARGETS = (160, 2000, MAX_TEXT_LENGTH)
REFERENCE_SIDES = (1024, 2048, 768, 1536)


def _prompt_for(rows: int, attended) -> str:
    """The shortest filler prompt whose attended text length is at least ``rows``."""
    words = 1
    while attended(" ".join(_FILLER[i % len(_FILLER)] for i in range(words))) < rows:
        words *= 2
    lo, hi = words // 2, words
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if attended(" ".join(_FILLER[i % len(_FILLER)] for i in range(mid))) < rows else (lo, mid)
    return " ".join(_FILLER[i % len(_FILLER)] for i in range(hi))


def _reference(side: int) -> Image.Image:
    ramp = np.linspace(0, 255, side, dtype=np.uint8)
    return Image.fromarray(np.stack([np.add.outer(ramp, ramp // 2).astype(np.uint8)] * 3, axis=-1))


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("model", ["distilled"])
def test_serving(*, mesh_device: ttnn.MeshDevice, model: str) -> None:
    sizes = tuple(int(s) for s in os.environ.get("FLUX3_KLEIN_SERVE_SIZES", "1024").split(","))
    out = Path(os.environ.get("FLUX3_KLEIN_SERVE_OUT", "flux3_klein_serving"))
    out.mkdir(parents=True, exist_ok=True)

    pipeline = Flux3KleinPipeline.create(mesh_device, models=(model,))

    def attended_rows(prompt: str) -> int:
        return pipeline._prompt_encoder.tokenize(prompt, MAX_TEXT_LENGTH)[2]

    prompts = {f"text{rows}": _prompt_for(rows, attended_rows) for rows in TEXT_ROW_TARGETS}
    attended = {name: attended_rows(prompt) for name, prompt in prompts.items()}
    warmup_rows = attended_rows(WARMUP_PROMPT)
    logger.info(f"SERVE attended text rows {attended}, warmup {warmup_rows}")
    assert warmup_rows not in attended.values(), (warmup_rows, attended)
    assert len(set(attended.values())) == len(attended), attended
    assert attended[f"text{MAX_TEXT_LENGTH}"] == MAX_TEXT_LENGTH

    references = {None: None, **{side: _reference(side) for side in REFERENCE_SIDES}}
    tokens = {side: (side // DOWNSAMPLING_RATIO) ** 2 for side in REFERENCE_SIDES}
    assert set(REFERENCE_TOKEN_BUCKETS) <= set(tokens.values()), "a reference must fill each bucket"
    assert any(
        t not in REFERENCE_TOKEN_BUCKETS for t in tokens.values()
    ), "a reference must leave its bucket part empty"

    pipeline.warmup(sizes=sizes)
    failures = []
    for side, reference in references.items():
        if reference is not None:
            pipeline._vae.encode(prepare_reference(reference))
            logger.info(f"SERVE reference {side}: {tokens[side]} tokens in bucket {reference_bucket(tokens[side])}")
        for size in sizes:
            for name, prompt in prompts.items():
                ttnn.synchronize_device(mesh_device)
                programs = mesh_device.num_program_cache_entries()
                images = () if reference is None else (reference,)
                result = pipeline.generate(
                    Flux3KleinRequest(prompt, model=model, width=size, height=size, seed=0, images=images)
                )
                ttnn.synchronize_device(mesh_device)
                new_programs = mesh_device.num_program_cache_entries() - programs
                tag = f"{size} ref {side} {name} prompt"
                logger.info(f"SERVE {tag}: new programs {new_programs}, {result.timings}")
                result.image.save(out / f"{size}_ref{side}_{name}.png")
                if new_programs:
                    failures.append(f"{tag}: {new_programs} new programs")
                if float(np.asarray(result.image, dtype=np.float32).std()) < 10:
                    failures.append(f"{tag}: flat image")
    pipeline.close()
    assert not failures, "\n".join(failures)
