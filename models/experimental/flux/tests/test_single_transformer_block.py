# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import torch
import ttnn
from loguru import logger

from ..tt import utils
from ..tt.single_transformer_block import FluxSingleTransformerBlock, FluxSingleTransformerBlockParameters
from ..tt.utils import allocate_tensor_on_device_like, assert_quality

if TYPE_CHECKING:
    from ..reference import FluxTransformer as FluxTransformerReference


@pytest.mark.parametrize(
    ("block_index", "sequence_length"),
    [
        (0, 4096 + 512),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 716800}], indirect=True)
@pytest.mark.parametrize("use_tracing", [False])  # Tracing currently causes a mesh device to hang.
def test_single_transformer_block(
    *,
    mesh_device: ttnn.MeshDevice,
    use_tracing: bool,
    block_index: int,
    sequence_length: int,
    parent_torch_model: FluxTransformerReference,
) -> None:
    batch_size, _ = mesh_device.shape

    torch.manual_seed(0)

    torch_model: SingleTransformerBlockReference = parent_torch_model.single_transformer_blocks[block_index].to(
        torch.float32
    )

    logger.debug("creating TT-NN model...")
    parameters = FluxSingleTransformerBlockParameters.from_torch(
        torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b
    )
    tt_model = FluxSingleTransformerBlock(parameters, num_heads=torch_model.num_heads)

    embedding_dim = 3072

    combined = torch.randn((batch_size, sequence_length, embedding_dim))
    time = torch.randn((batch_size, embedding_dim))
    imagerot1 = torch.randn([sequence_length, 128], dtype=torch.float32)
    imagerot2 = torch.randn([sequence_length, 128], dtype=torch.float32)

    sharded = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, -1))
    batch_sharded = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, None))
    unsharded = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_combined_host = ttnn.from_torch(combined, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=sharded)
    tt_time_host = ttnn.from_torch(
        time.unsqueeze(1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=batch_sharded
    )
    tt_imagerot1_host = ttnn.from_torch(imagerot1, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded)
    tt_imagerot2_host = ttnn.from_torch(imagerot2, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded)

    with torch.no_grad():
        combined_output = torch_model(combined=combined, time_embed=time, image_rotary_emb=(imagerot1, imagerot2))

    tt_combined = allocate_tensor_on_device_like(tt_combined_host, device=mesh_device)
    tt_time = allocate_tensor_on_device_like(tt_time_host, device=mesh_device)
    tt_imagerot1 = allocate_tensor_on_device_like(tt_imagerot1_host, device=mesh_device)
    tt_imagerot2 = allocate_tensor_on_device_like(tt_imagerot2_host, device=mesh_device)

    model_args = dict(  # noqa: C408
        combined=tt_combined,
        time_embed=tt_time,
        image_rotary_emb=(tt_imagerot1, tt_imagerot2),
    )

    if use_tracing:
        # cache
        logger.debug("caching...")
        tt_model.forward(**model_args)

        # trace
        logger.debug("tracing...")
        tid = ttnn.begin_trace_capture(mesh_device)
        tt_combined_output = tt_model.forward(**model_args)
        ttnn.end_trace_capture(mesh_device, tid)

        # execute
        logger.debug("executing...")
        ttnn.copy_host_to_device_tensor(tt_combined_host, tt_combined)
        ttnn.copy_host_to_device_tensor(tt_time_host, tt_time)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        ttnn.execute_trace(mesh_device, tid)
    else:
        # compile
        logger.debug("compiling...")
        tt_model.forward(**model_args)

        # execute
        logger.debug("executing...")
        ttnn.copy_host_to_device_tensor(tt_combined_host, tt_combined)
        ttnn.copy_host_to_device_tensor(tt_time_host, tt_time)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        utils.signpost("start")
        tt_combined_output = tt_model.forward(**model_args)
        utils.signpost("end")

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1))
    assert_quality(combined_output, tt_combined_output, pcc=0.99933, mse=2300, mesh_composer=composer)
