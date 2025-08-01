# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
import math

from ..reference.feed_forward import FeedForward
from ..tt.fun_feed_forward import TtFeedForwardParameters, sd_feed_forward
from ..tt.utils import assert_quality, from_torch_fast_2d
from ..tt.parallel_config import StableDiffusionParallelManager

TILE_SIZE = 32


@pytest.mark.parametrize(
    (
        "batch_size",
        "embedding_dim",
        "num_heads",
        "sequence_length",
        "shard_sequence",
    ),
    [
        (1, 2432, 38, 4096, True),
        (1, 2432, 38, 333, False),
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
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_feed_forward(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    embedding_dim: int,
    num_heads: int,
    sequence_length: int,
    cfg: int,
    sp: int,
    tp: int,
    topology: ttnn.Topology,
    shard_sequence: bool,
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
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    torch_model = FeedForward(dim=embedding_dim, dim_out=embedding_dim).to(dtype=torch_dtype)
    torch_model.eval()

    ## heads padding
    pad_embedding_dim = (bool)(num_heads) % tp_factor
    if pad_embedding_dim:
        head_size = embedding_dim // num_heads
        num_heads = math.ceil(num_heads / tp_factor) * tp_factor
        hidden_dim_padding = (num_heads * head_size) - embedding_dim
    else:
        hidden_dim_padding = 0

    parameters = TtFeedForwardParameters.from_torch(
        torch_model.state_dict(),
        dtype=ttnn_dtype,
        device=submesh,
        hidden_dim_padding=hidden_dim_padding,
        parallel_config=parallel_manager.dit_parallel_config,
    )

    torch.manual_seed(0)
    torch_input_tensor = torch.randn((batch_size, sequence_length, embedding_dim), dtype=torch_dtype)

    input_padded_4d = torch_input_tensor.unsqueeze(1)
    if pad_embedding_dim:
        input_padded_4d = torch.nn.functional.pad(
            input_padded_4d, pad=(0, hidden_dim_padding), mode="constant", value=0
        )
    if input_padded_4d.shape[2] % TILE_SIZE:
        input_padded_4d = torch.nn.functional.pad(
            input_padded_4d,
            pad=(0, 0, 0, TILE_SIZE - (input_padded_4d.shape[2] % TILE_SIZE)),
            mode="constant",
            value=0,
        )

    dims = [None, None]
    if shard_sequence:
        dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = 2
    tt_input_tensor = from_torch_fast_2d(
        input_padded_4d,
        mesh_device=submesh,
        mesh_shape=parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
        dims=dims,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    buffer_shape = list(tt_input_tensor.padded_shape)
    buffer_shape[3] //= parallel_manager.dit_parallel_config.tensor_parallel.factor

    parallel_manager.maybe_init_persistent_buffers(
        KV_shape=[1, 1, 32, 32],  # dummy
        spatial_shape=buffer_shape,
        prompt_shape=buffer_shape,
    )

    is_spatial = not shard_sequence

    tt_output = sd_feed_forward(
        tt_input_tensor,
        parameters,
        parallel_manager=parallel_manager,
        cfg_index=0,
        is_spatial=is_spatial,
    )
    dims = [None, None]
    dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = 2
    dims[parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            submesh,
            mesh_shape=tuple(submesh.shape),
            dims=dims,
        ),
    )
    tt_output_torch = tt_output_torch[:, :, 0:sequence_length, :embedding_dim]

    assert_quality(torch_output, tt_output_torch, pcc=0.998_800, shard_dim=-1)
