# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os


import pytest
import torch
import ttnn

from models.experimental.flux.tt.attention import Attention, AttentionParameters
from models.experimental.flux.tt.utils import allocate_tensor_on_device_like, assert_quality

from models.experimental.flux.reference import FluxTransformer as FluxTransformerReference
from models.experimental.flux.reference.attention import Attention as AttentionReference


@pytest.mark.parametrize(
    ("block_index", "spatial_sequence_length", "prompt_sequence_length"),
    [
        (0, 4096, 512),
        (0, 4096 + 512, 0),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE") or "N300", len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 517120}], indirect=True)
@pytest.mark.parametrize("use_tracing", [False])  # Tracing currently causes a mesh device to hang.
def test_attention(
    *,
    mesh_device: ttnn.MeshDevice,
    use_tracing: bool,
    block_index: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
    model_location_generator,
) -> None:
    separate_prompt = prompt_sequence_length != 0
    batch_size, _ = mesh_device.shape

    torch.manual_seed(0)

    # Load the model from checkpoint
    model_path = model_location_generator("black-forest-labs/FLUX.1-schnell", model_subdir="Flux1_Schnell")
    flux_model = FluxTransformerReference.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.float32
    )

    torch_model: AttentionReference = flux_model.transformer_blocks[block_index].attn.to(torch.float32)

    parameters = AttentionParameters.from_torch(
        torch_model.state_dict(),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    tt_model = Attention(parameters, num_heads=torch_model.num_heads)

    spatial = torch.randn((batch_size, spatial_sequence_length, 3072))
    prompt = torch.randn((batch_size, prompt_sequence_length, 3072)) if separate_prompt else None
    imagerot1 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128])
    imagerot2 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128])

    sharded = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, -1))
    unsharded = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_spatial_host = ttnn.from_torch(spatial, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=sharded)
    tt_prompt_host = (
        ttnn.from_torch(prompt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=sharded)
        if separate_prompt
        else None
    )
    tt_imagerot1_host = ttnn.from_torch(imagerot1, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded)
    tt_imagerot2_host = ttnn.from_torch(imagerot2, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded)

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(
            spatial=spatial, prompt=prompt, image_rotary_emb=(imagerot1, imagerot2)
        )

    tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=mesh_device)
    tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=mesh_device) if separate_prompt else None
    tt_imagerot1 = allocate_tensor_on_device_like(tt_imagerot1_host, device=mesh_device)
    tt_imagerot2 = allocate_tensor_on_device_like(tt_imagerot2_host, device=mesh_device)

    model_args = dict(  # noqa: C408
        spatial=tt_spatial,
        prompt=tt_prompt,
        image_rotary_emb=(tt_imagerot1, tt_imagerot2),
    )

    if use_tracing:
        # cache
        tt_model.forward(**model_args)

        # trace
        tid = ttnn.begin_trace_capture(mesh_device)
        tt_spatial_output, tt_prompt_output = tt_model.forward(**model_args)
        ttnn.end_trace_capture(mesh_device, tid)

        # execute
        ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
        if separate_prompt:
            ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        ttnn.execute_trace(mesh_device, tid)
    else:
        # compile
        tt_model.forward(**model_args)

        # execute
        ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
        if separate_prompt:
            ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
        ttnn.copy_host_to_device_tensor(tt_imagerot1_host, tt_imagerot1)
        ttnn.copy_host_to_device_tensor(tt_imagerot2_host, tt_imagerot2)
        tt_spatial_output, tt_prompt_output = tt_model.forward(**model_args)

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1))

    assert_quality(spatial_output, tt_spatial_output, pcc=0.9939, mse=0.0003, mesh_composer=composer)
    if separate_prompt:
        assert_quality(prompt_output, tt_prompt_output, pcc=0.9939, mse=0.02, mesh_composer=composer)
