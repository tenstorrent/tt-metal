# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The stage-5 diffusion stack split across the mesh matches the same stack replicated.

Stage 5 is the reason for the whole exercise: at 145 frames its context is 9.7 GB and its
``context_and_x`` buffer 19.4 GB on every chip. It carries more than the deterministic stages —
context injection, AdaLN modulation from a shared timestep embedding, a fused SwiGLU and the
output projection — but every one of those is per-site, so the sharded version differs from the
replicated one in exactly one place: the halo that feeds the attention.

The modulation path is worth watching. It comes from a timestep, not from the volume, so it is
identical on every device and must stay that way — a shard that modulated with its own slice of
anything would drift block by block, and eight blocks is enough for that to show.
"""

import pytest
import torch

import ttnn

from ....models.vae.diffvae_ltx import MeshShardConfig
from ....models.vae.diffvae_ltx_stage5 import DiffVAEStage5, DiffVAEStage5Config, Grid
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality

_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}


def _config(kernel) -> DiffVAEStage5Config:
    return DiffVAEStage5Config(
        dim=64,
        head_dim=32,
        kernel_size=kernel,
        context_channels=64,
        mlp_hidden=128,
        num_blocks=2,
        patch_size=4,
        out_channels=3,
    )


def _random_state(cfg: DiffVAEStage5Config, padded_patch: int, generator) -> dict[str, torch.Tensor]:
    def randn(*shape):
        return torch.randn(*shape, generator=generator) * 0.05

    state = {
        "conv_in_x_t.weight": randn(cfg.dim, cfg.patch_channels),
        "conv_in_x_t.bias": randn(cfg.dim),
        "norm_out.weight": randn(cfg.dim),
        "conv_out.weight": randn(cfg.patch_channels, cfg.dim),
        "conv_out.bias": randn(cfg.patch_channels),
        "t_embedder.mlp.linear_1.weight": randn(cfg.t_emb_dim, 256),
        "t_embedder.mlp.linear_1.bias": randn(cfg.t_emb_dim),
        "t_embedder.mlp.linear_2.weight": randn(cfg.t_emb_dim, cfg.t_emb_dim),
        "t_embedder.mlp.linear_2.bias": randn(cfg.t_emb_dim),
        "shared_adaln.proj.weight": randn(7 * cfg.dim, cfg.t_emb_dim),
        "shared_adaln.proj.bias": randn(7 * cfg.dim),
    }
    for block in range(cfg.num_blocks):
        prefix = f"diff_blocks.{block}."
        state |= {
            prefix + "scale_shift_table": randn(7, cfg.dim),
            prefix + "context_proj.weight": randn(cfg.dim, cfg.context_channels),
            prefix + "context_proj.bias": randn(cfg.dim),
            prefix + "norm1.weight": randn(cfg.dim),
            prefix + "norm2.weight": randn(cfg.dim),
            prefix + "attn.qkv.weight": randn(3 * cfg.dim, cfg.dim),
            prefix + "attn.qkv.bias": randn(3 * cfg.dim),
            prefix + "attn.proj.weight": randn(cfg.dim, cfg.dim),
            prefix + "attn.proj.bias": randn(cfg.dim),
            prefix + "attn.q_norm.weight": randn(cfg.head_dim),
            prefix + "attn.k_norm.weight": randn(cfg.head_dim),
            prefix + "mlp.w_gate.weight": randn(cfg.mlp_hidden, cfg.dim),
            prefix + "mlp.w_up.weight": randn(cfg.mlp_hidden, cfg.dim),
            prefix + "mlp.w_down.weight": randn(cfg.dim, cfg.mlp_hidden),
        }
    return state


@pytest.mark.parametrize(
    "mesh_device, device_params, grid_hw, kernel",
    [
        ((4, 8), _FABRIC, (32, 64), (3, 7, 7)),
        ((4, 8), _FABRIC, (32, 64), (11, 11, 11)),  # the shipped stage-5 kernel
    ],
    indirect=["mesh_device", "device_params"],
)
def test_sharded_stage5_matches_replicated(*, mesh_device, grid_hw, kernel):
    mesh = tuple(mesh_device.shape)
    cfg = _config(kernel)
    grid = Grid(batch=1, t=4, h=grid_hw[0], w=grid_hw[1])

    generator = torch.Generator().manual_seed(0)
    stage5 = DiffVAEStage5(cfg, mesh_device=mesh_device)
    stage5.load_torch_state_dict(_random_state(cfg, stage5.padded_patch_channels, generator))

    channels = cfg.context_channels + cfg.dim
    buffer = torch.randn(1, grid.t, grid.h, grid.w, channels, generator=generator)
    timestep = ttnn.from_torch(
        torch.tensor([[[[1.0]]]]), device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )

    replicated = stage5.forward_diff_step(
        ttnn.from_torch(
            buffer.reshape(1, 1, grid.sites, channels),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
        timestep,
        grid,
    )
    reference = ttnn.to_torch(ttnn.get_device_tensors(replicated)[0])
    reference = reference.reshape(1, 1, grid.sites, -1).reshape(1, grid.t, grid.h, grid.w, -1)

    ccl = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    shard = MeshShardConfig(ccl=ccl, mesh=mesh)
    local = DiffVAEStage5.shard_grid(grid, shard)

    sharded_in = ttnn.from_torch(
        buffer,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh, dims=[2, 3]),
    )
    sharded_in = ttnn.to_layout(ttnn.reshape(sharded_in, (1, 1, local.sites, channels)), ttnn.TILE_LAYOUT)

    sharded = stage5.forward_diff_step(sharded_in, timestep, grid, shard=shard)
    gathered = ttnn.to_torch(
        ttnn.reshape(sharded, (1, local.t, local.h, local.w, -1)),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[2, 3], mesh_shape=mesh),
    )

    assert tuple(gathered.shape) == tuple(reference.shape), f"{tuple(gathered.shape)} != {tuple(reference.shape)}"
    assert_quality(reference, gathered, pcc=0.999)
