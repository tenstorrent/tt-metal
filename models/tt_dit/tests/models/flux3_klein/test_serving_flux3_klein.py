# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein serving: after ``warmup`` a request compiles nothing, at every warmed size and reference count, for a
short prompt and for one that fills the whole 4096-row text sequence.

The text length reaches ring SDPA only as the device tensors the transformer writes per request, so every prompt
length runs the programs warmup compiled. Around every request the test asserts no new program-cache entries.

Env: FLUX3_KLEIN_SERVE_SIZES (default 1024, comma-separated), FLUX3_KLEIN_SERVE_OUT (image dir).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.flux3_klein.packing import MAX_TEXT_LENGTH
from ....pipelines.flux3_klein.pipeline_flux3_klein import MAX_REFERENCES, Flux3KleinPipeline, Flux3KleinRequest
from ....utils.test import ring_params_8k

SHORT_PROMPT = "A red apple on a wooden table."
_FILLER = (
    "a sunlit harbor town with pastel houses, fishing boats, gulls over the water, a lighthouse on the cliff, "
    "market stalls with fruit and flowers, children chasing a dog along the pier, clouds over distant hills, "
)
LONG_PROMPT = _FILLER * 64  # truncated to the full text sequence


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

    pipeline.warmup(sizes=sizes)
    reference = Image.new("RGB", (1024, 1024), (90, 120, 160))
    failures = []
    for size in sizes:
        for references in range(MAX_REFERENCES + 1):
            for name, prompt in (("short", SHORT_PROMPT), ("long", LONG_PROMPT)):
                ttnn.synchronize_device(mesh_device)
                programs = mesh_device.num_program_cache_entries()
                request = Flux3KleinRequest(
                    prompt, model=model, width=size, height=size, seed=0, images=(reference,) * references
                )
                result = pipeline.generate(request)
                ttnn.synchronize_device(mesh_device)
                new_programs = mesh_device.num_program_cache_entries() - programs
                logger.info(f"SERVE {size} refs {references} {name}: new programs {new_programs}, {result.timings}")
                result.image.save(out / f"{size}_refs{references}_{name}.png")
                if new_programs:
                    failures.append(f"{size} refs {references} {name} prompt: {new_programs} new programs")
                if float(np.asarray(result.image, dtype=np.float32).std()) < 10:
                    failures.append(f"{size} refs {references} {name} prompt: flat image")
    pipeline.close()
    assert not failures, "\n".join(failures)
