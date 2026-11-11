# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein serving: after ``warmup`` the DiT compiles nothing, at every warmed size, for a short prompt and for one
that fills the whole 4096-row text sequence, without a reference and with references that fill or only partly fill a
reference token bucket.

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
    Flux3KleinPipeline,
    Flux3KleinRequest,
    prepare_reference,
    reference_bucket,
)
from ....utils.test import ring_params_8k

SHORT_PROMPT = "A red apple on a wooden table."
_FILLER = (
    "a sunlit harbor town with pastel houses, fishing boats, gulls over the water, a lighthouse on the cliff, "
    "market stalls with fruit and flowers, children chasing a dog along the pier, clouds over distant hills, "
)
LONG_PROMPT = _FILLER * 64  # truncated to the full text sequence
# square references: each bucket filled (as warmup does) and each only partly filled
REFERENCE_SIDES = (1024, 2048, 768, 1536)


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
    attended = {
        name: pipeline._prompt_encoder.tokenize(prompt, MAX_TEXT_LENGTH)[2]
        for name, prompt in (("short", SHORT_PROMPT), ("long", LONG_PROMPT))
    }
    logger.info(f"SERVE attended text rows {attended}")
    assert attended["long"] == MAX_TEXT_LENGTH
    assert attended["short"] < MAX_TEXT_LENGTH

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
            # a new reference shape compiles the VAE encoder only: do it outside the measured requests
            pipeline._vae.encode(prepare_reference(reference))
            logger.info(f"SERVE reference {side}: {tokens[side]} tokens in bucket {reference_bucket(tokens[side])}")
        for size in sizes:
            for name, prompt in (("short", SHORT_PROMPT), ("long", LONG_PROMPT)):
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
