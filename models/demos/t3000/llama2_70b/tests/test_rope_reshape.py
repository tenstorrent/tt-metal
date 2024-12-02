# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 64, 128],
    ],
)
def test_reshape_rope(
    device,
    shape,
    function_level_defaults,
):
    pt_input = torch.randn((*shape[:2], shape[2] * shape[3]), dtype=torch.float32)

    tt_input = ttnn.from_torch(
        pt_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    logger.info(f"pt_input shape: {pt_input.shape}")
    logger.info(f"tt_input shape: {tt_input.shape}")

    pt_out = pt_input.reshape(shape[0], shape[1], shape[2], shape[3])

    tt_out = ttnn.reshape(tt_input, shape)
    tt_out = ttnn.to_torch(tt_out)

    logger.info(f"pt_out shape: {pt_out.shape}")
    logger.info(f"tt_out shape: {tt_out.shape}")

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 64, 128],
    ],
)
def test_reshape_rope_t3k(
    t3k_mesh_device,
    shape,
    function_level_defaults,
):
    pt_input = torch.randn((*shape[:2], shape[2] * shape[3]), dtype=torch.float32)

    tt_input = ttnn.from_torch(
        pt_input,
        device=t3k_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )
    logger.info(f"pt_input shape: {pt_input.shape}")
    logger.info(f"tt_input shape: {tt_input.shape}")

    pt_out = pt_input.reshape(shape[0], shape[1], shape[2], shape[3])

    tt_out = ttnn.reshape(tt_input, shape)
    tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(t3k_mesh_device, dim=-1))[..., : shape[3]]

    logger.info(f"pt_out shape: {pt_out.shape}")
    logger.info(f"tt_out shape: {tt_out.shape}")

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing
