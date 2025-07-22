# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import ttnn

from ..reference.normalization import RmsNorm
from ..tt.fun_normalization import sd_layer_norm, sd_rms_norm
from ..tt.fun_normalization import TtLayerNormParameters, TtRmsNormParameters
from ..tt.utils import assert_quality
from ..tt.parallel_config import StableDiffusionParallelManager


@pytest.mark.parametrize(
    "input_shape, distributed",
    [
        [(4096, 3072), True],  # Prompt, spatial in transformer block
        [(4096, 3072), False],  # Output layernorm in transformer
    ],
)
@pytest.mark.parametrize(
    (
        "mesh_device",
        "cfg",
        "sp",
        "tp",
        "topology",
        "num_links",
    ),
    [
        [(2, 4), (1, 0), (2, 0), (4, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 3],
    ],
    ids=[
        "t3k_cfg1_sp2_tp4",
        "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("affine", [True, False], ids=["affine", "noaffine"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_layer_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    input_shape: list[int],
    affine: bool,
    distributed: bool,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
) -> None:
    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp
    parallel_manager = StableDiffusionParallelManager(
        mesh_device,
        cfg_factor,
        sp_factor,
        tp_factor,
        sp_factor,
        tp_factor,
        topology,
        cfg_axis=cfg_axis,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
    )
    submesh = parallel_manager.submesh_devices[0]
    input_shape = [1, *input_shape]

    torch.manual_seed(0)

    torch_model = torch.nn.LayerNorm(input_shape[-1:], eps=1.0, elementwise_affine=affine)

    parameters = TtLayerNormParameters.from_torch(
        torch_model.state_dict(),
        device=submesh,
        dtype=ttnn.bfloat8_b,
        weight_shape=input_shape[-1:],
        eps=torch_model.eps,
        distributed=distributed,
        parallel_config=parallel_manager.dit_parallel_config,
    )

    torch_input_tensor = torch.randn(input_shape)

    dims = [None, None]
    dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = 1
    if distributed:
        dims[parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis] = 2
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, tuple(submesh.shape), dims),
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    # Unsqueeze shape to 4D
    buffer_shape = list(tt_input_tensor.padded_shape)
    buffer_shape = [1] * (4 - len(buffer_shape)) + buffer_shape

    parallel_manager.maybe_init_persistent_buffers(
        KV_shape=[1, 1, 32, 32],  # dummy
        spatial_shape=buffer_shape,
        prompt_shape=buffer_shape,
    )

    tt_output = sd_layer_norm(tt_input_tensor, parameters, parallel_manager, cfg_index=0)
    if not distributed:
        dims[parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis] = 0
    composer = ttnn.ConcatMesh2dToTensor(submesh, tuple(submesh.shape), dims)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=composer)
    if not distributed:
        tt_output_torch = tt_output_torch[:1]  # Concatenated replicas on batch, slice one.
    assert_quality(torch_output, tt_output_torch, pcc=0.99990)


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 24, 4096, 64],
    ],
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_rms_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    input_shape: list[int],
) -> None:
    torch.manual_seed(0)

    torch_model = RmsNorm(dim=input_shape[-1], eps=1.0)
    torch.nn.init.normal_(torch_model.weight)

    parameters = TtRmsNormParameters.from_torch(
        torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b, eps=torch_model.eps
    )

    torch_input_tensor = torch.randn(input_shape)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
    )

    torch_output = torch_model(torch_input_tensor)

    tt_output = sd_rms_norm(tt_input_tensor, parameters)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[: input_shape[0]]

    assert_quality(torch_output, tt_output_torch, pcc=0.99985)
