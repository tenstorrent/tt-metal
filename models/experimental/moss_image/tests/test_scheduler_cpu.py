# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Capture-free TT CFG/Euler checks against the upstream scheduler on CPU.

This is a component reference, not CPU execution of the full image pipeline.
Both replicas are checked on a two-card mesh; no collective is required here.
"""

import pytest
import torch
import ttnn

from models.experimental.moss_image.tests.assets import check
from models.experimental.moss_image.tt.scheduler import guided_euler_step
from models.experimental.moss_image.tt.vendor.pipeline_moss_image import MOSSImageScheduler


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param((1, 1), {"l1_small_size": 32768}, id="1-card"),
        pytest.param((1, 2), {"l1_small_size": 32768, "fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="2-cards"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "step_index,guidance_scale,prediction_dtype,tokens",
    [
        pytest.param(0, 3.0, torch.bfloat16, 64, id="first-cfg3-bf16"),
        pytest.param(24, 1.0, torch.float32, 96, id="middle-cfg1-fp32"),
        pytest.param(49, 3.0, torch.bfloat16, 65, id="final-cfg3-bf16-padded"),
    ],
)
def test_cfg_euler_cpu(mesh_device, step_index, guidance_scale, prediction_dtype, tokens, record_property):
    generator = torch.Generator(device="cpu").manual_seed(20261023)
    shape = (1, tokens, 128)
    sample = torch.randn(shape, generator=generator, dtype=torch.float32)
    positive = torch.randn(shape, generator=generator).to(prediction_dtype)
    negative = torch.randn(shape, generator=generator).to(prediction_dtype)

    # Use the pinned upstream schedule and diffusers step implementation instead
    # of rebuilding Euler integration with the same expression as the TT code.
    # The official pipeline promotes each prediction before applying CFG.
    scheduler = MOSSImageScheduler(shift=5.0)
    scheduler.set_timesteps(50, device="cpu")
    scheduler.set_begin_index(step_index)
    expected_velocity = negative.float() + guidance_scale * (positive.float() - negative.float())
    timestep = scheduler.timesteps[step_index]
    dt = float(scheduler.sigmas[step_index + 1] - scheduler.sigmas[step_index])
    expected_sample = scheduler.step(expected_velocity, timestep, sample, return_dict=False)[0]
    assert dt < 0

    def upload(value):
        return ttnn.from_torch(
            value,
            dtype=ttnn.bfloat16 if value.dtype == torch.bfloat16 else ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None,
        )

    actual_sample, actual_velocity = guided_euler_step(
        upload(sample), upload(positive), upload(negative), guidance_scale, dt
    )
    record_property("reference", "CPU PyTorch CFG and pinned MOSS/diffusers Euler scheduler; no capture assets")
    record_property("step_index", step_index)
    record_property("guidance_scale", guidance_scale)
    for name, actual, expected in (
        ("velocity", actual_velocity, expected_velocity),
        ("latents", actual_sample, expected_sample),
    ):
        assert actual.dtype == ttnn.float32
        replicas = ttnn.get_device_tensors(actual)
        assert len(replicas) == mesh_device.get_num_devices()
        for replica, tensor in enumerate(replicas):
            check(
                ttnn.to_torch(tensor),
                expected,
                lambda key, value: record_property(f"{name}.replica_{replica}.{key}", value),
                min_pcc=0.999999,
                max_relative_rms=1e-5,
            )
