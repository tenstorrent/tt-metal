# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Full Z-Image-Turbo pipeline test — warmup + generate on (1,4) mesh.

Verifies the complete end-to-end flow: text encoder → DIT denoising loop → VAE
decode, including Metal Trace capture and replay for all three models.

The pipeline opens its own mesh device internally, so no device fixture is needed.
"""

from PIL import Image

from models.demos.z_image_turbo.tt.z_image_turbo import ZImageTurbo


def test_z_image_turbo_pipeline(tmp_path):
    pipeline = ZImageTurbo()

    warmup_image = pipeline.warmup(steps=9, seed=42)
    assert isinstance(warmup_image, Image.Image), "Warmup did not return a PIL Image"
    assert warmup_image.size == (512, 512), f"Expected 512x512, got {warmup_image.size}"

    image = pipeline("a cat sitting on a mat", steps=9, seed=42)
    assert isinstance(image, Image.Image), "Forward did not return a PIL Image"
    assert image.size == (512, 512), f"Expected 512x512, got {image.size}"

    out_path = tmp_path / "test_output.png"
    image.save(str(out_path))
    assert out_path.exists(), "Output image was not saved"
