# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5 probe (DO NOT DELETE): L1 INTERLEAVED output on the largest golden shapes, fp32 LayerNorm with gamma +
beta (the largest CB set). Each call must either pass or be refused with UnsupportedAxisValue carrying the numbers
(never a CB clash / hang). Prints which."""

import pytest
import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue

from .test_all_gather_rms_norm import DEVICE_PARAMS, TL, _run_case, _skip_if_unrunnable

SHAPES = [
    (1, 1, 2048, 1280),
    (1, 1, 2048, 384),
    (1, 640, 1792),
    (1, 1, 640, 1792),
    (128, 8192),
    (1, 1, 128, 4096),
    (4, 1, 512, 512),
    (2, 4, 128, 512),
]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("shape", SHAPES, ids=["x".join(map(str, s)) for s in SHAPES])
def test_l1_out_fit(mesh_device, shape, dtype):
    _skip_if_unrunnable(mesh_device, 1, ttnn.Topology.Linear)
    try:
        _run_case(
            mesh_device,
            shape,
            cluster_axis=1,
            dtype=dtype,
            norm_type="layer_norm",
            gamma_layout=TL,
            with_beta=True,
            topology=ttnn.Topology.Linear,
            num_links=2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(f"L1FIT {shape} {dtype}: PASS")
    except UnsupportedAxisValue as e:
        print(f"L1FIT {shape} {dtype}: REFUSED {e}")
