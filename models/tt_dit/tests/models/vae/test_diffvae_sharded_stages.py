# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The deterministic stages produce the same volume split across the mesh as replicated on it.

Covers what the single-block test does not: the entry from a replicated volume into a split
one, several blocks in sequence each doing their own halo exchange, and the pixel-shuffle
upsample carrying a shard across a stage boundary.

The upsample is the interesting part. It only ever redistributes a token's own channels, so a
shard's extent scales by the stride and its boundaries stay where its neighbours' do — no
reshard, no collective. If that were wrong the seams would drift one stage at a time, which is
why the check runs through two stages rather than one.

The split starts after stage 0 because the latent grid is the one grid a 4x8 mesh does not
divide, and also the one small enough that replicating it costs nothing.
"""

import pytest
import torch

import ttnn

from ....models.vae.diffvae_ltx import DeterministicStages, MeshShardConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality

_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}

IN_CHANNELS = 32
STAGE_CHANNELS = (64, 64, 32)
STAGE_DEPTHS = (2, 2)
STAGE_KERNELS = ((3, 7, 7), (3, 5, 5))
# Second element is the channel reduction: an upsample outputs in_channels // reduction, so
# these have to track STAGE_CHANNELS the way the shipped config's do.
UPSAMPLES = (((1, 2, 2), 1), ((2, 2, 2), 2))


def _random_state(generator) -> dict[str, torch.Tensor]:
    def randn(*shape):
        return torch.randn(*shape, generator=generator) * 0.05

    state = {"conv_in.weight": randn(STAGE_CHANNELS[0], IN_CHANNELS), "conv_in.bias": randn(STAGE_CHANNELS[0])}
    for stage, depth in enumerate(STAGE_DEPTHS):
        dim = STAGE_CHANNELS[stage]
        hidden = (int(dim * 4.0) + 15) // 16 * 16
        for block in range(depth):
            prefix = f"det_stages.{stage}.{block}."
            state |= {
                prefix + "norm1.weight": randn(dim),
                prefix + "norm2.weight": randn(dim),
                prefix + "attn.qkv.weight": randn(3 * dim, dim),
                prefix + "attn.qkv.bias": randn(3 * dim),
                prefix + "attn.proj.weight": randn(dim, dim),
                prefix + "attn.proj.bias": randn(dim),
                prefix + "attn.q_norm.weight": randn(64),
                prefix + "attn.k_norm.weight": randn(64),
                prefix + "mlp.w_gate.weight": randn(hidden, dim),
                prefix + "mlp.w_up.weight": randn(hidden, dim),
                prefix + "mlp.w_down.weight": randn(dim, hidden),
            }
        stride, reduction = UPSAMPLES[stage]
        out = stride[0] * stride[1] * stride[2] * dim // reduction
        state |= {f"upsamples.{stage}.proj.weight": randn(out, dim), f"upsamples.{stage}.proj.bias": randn(out)}
    return state


@pytest.mark.parametrize(
    "mesh_device, device_params, latent_dims",
    [((4, 8), _FABRIC, (4, 16, 32))],
    indirect=["mesh_device", "device_params"],
)
def test_sharded_stages_match_replicated(*, mesh_device, latent_dims):
    mesh = tuple(mesh_device.shape)
    generator = torch.Generator().manual_seed(0)

    stages = DeterministicStages(
        in_channels=IN_CHANNELS,
        stage_channels=STAGE_CHANNELS,
        stage_depths=STAGE_DEPTHS,
        stage_kernels=STAGE_KERNELS,
        upsamples=UPSAMPLES,
        head_dim=64,
        mesh_device=mesh_device,
    )
    stages.load_torch_state_dict(_random_state(generator))

    latent = torch.randn(1, *latent_dims, IN_CHANNELS, generator=generator).reshape(-1, IN_CHANNELS)
    upload = lambda: ttnn.from_torch(latent, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    replicated, whole_dims = stages(upload(), dims=latent_dims)
    reference = ttnn.to_torch(ttnn.get_device_tensors(replicated)[0]).reshape(1, *whole_dims, STAGE_CHANNELS[-1])

    ccl = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    sharded, local_dims = stages(upload(), dims=latent_dims, shard=MeshShardConfig(ccl=ccl, mesh=mesh, enter_stage=1))

    assert local_dims == (
        whole_dims[0],
        whole_dims[1] // mesh[0],
        whole_dims[2] // mesh[1],
    ), f"shard {local_dims} does not tile {whole_dims} over {mesh}"
    gathered = ttnn.to_torch(
        ttnn.reshape(sharded, (1, *local_dims, STAGE_CHANNELS[-1])),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[2, 3], mesh_shape=mesh),
    )
    assert tuple(gathered.shape) == tuple(reference.shape), f"{tuple(gathered.shape)} != {tuple(reference.shape)}"
    assert_quality(reference, gathered, pcc=0.999)
