# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional raw-text-to-image test; not an assertion of CUDA image parity."""
import os

import pytest
import torch
import ttnn
from PIL import Image

from models.experimental.moss_image.tests.conftest import verify_weights


@pytest.mark.parametrize("moss_assets", ["Base", "SFT"], indirect=True)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param((1, 1), {"l1_small_size": 131072}, id="1-card"),
        pytest.param((1, 2), {"l1_small_size": 131072, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="2-cards"),
    ],
    indirect=True,
)
def test_text_to_image(moss_assets, mesh_device, tmp_path, record_property):
    from models.experimental.moss_image.demo import generate

    checkpoint, capture = moss_assets
    verify_weights(checkpoint, capture, None)
    # The quick functional case still traverses the entire model. Full 1024px,
    # 50-step runs are explicitly requested rather than silently substituted.
    full = os.environ.get("MOSS_IMAGE_FULL_E2E") == "1"
    resolution, steps = (1024, 50) if full else (256, 2)
    output = tmp_path / "generation"
    result = generate(
        checkpoint,
        mesh_device,
        output,
        variant=capture.manifest["arguments"]["variant"],
        prompt="the quick brown fox jumps over the lazy dog",
        height=resolution,
        width=resolution,
        steps=steps,
        seed=42,
        guidance_scale=3.0,
    )
    record_property("scope", "functional native TT generation; final CUDA image parity remains unqualified")
    record_property("steps", steps)
    record_property("resolution", resolution)
    assert result["status"] in ("complete", "completed", "passed"), result
    image = Image.open(output / "image.png")
    assert image.size == (resolution, resolution)
    pixels = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8).float()
    assert pixels.std() > 1.0, "Constant or nearly constant output is not a useful smoke test"
