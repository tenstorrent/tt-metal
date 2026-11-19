# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttnn

from models.experimental.moss_image.tests.assets import check
from models.experimental.moss_image.tt.scheduler import guided_euler_step


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_cfg_euler(moss_assets, mesh_device, record_property):
    _, capture = moss_assets

    def packed(value):
        return value.flatten(2).transpose(1, 2).contiguous()

    def upload(key):
        return ttnn.from_torch(
            packed(capture.load(key)),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    step = capture.manifest["steps"][0]
    actual, velocity = guided_euler_step(
        upload("steps/000/input_latents"),
        upload("steps/000/positive_prediction"),
        upload("steps/000/negative_prediction"),
        capture.manifest["guidance_scale"],
        step["next_sigma"] - step["sigma"],
    )
    for name, value, key in (
        ("velocity", velocity, "steps/000/guided_prediction"),
        ("latents", actual, "steps/000/updated_latents"),
    ):
        host = ttnn.to_torch(ttnn.get_device_tensors(value)[0])
        check(
            host,
            packed(capture.load(key)),
            lambda key, value: record_property(f"{name}.{key}", value),
            min_pcc=0.999999,
            max_relative_rms=1e-5,
        )
