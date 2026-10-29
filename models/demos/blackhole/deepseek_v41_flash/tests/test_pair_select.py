# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""pair_select (tt/pair_select.py): the drafter sampler tail as one program == the argmax / gather / arithmetic chain of ``DSV41DeviceHead.sample_global_rm`` + the embedding lookup.
Random fp32 logits sharded over the 8 mesh columns, with planted ties inside a column and across columns (the lowest index must win); U = 1..6 user rows."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import VOCAB, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.pair_select import pair_select
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.parametrize("U", [1, 2, 4, 6])
@pytest.mark.parametrize("tile_pairs", [False, True])
@torch.no_grad()
def test_pair_select(mesh_device, U, tile_pairs):
    md = mesh_device
    rows, cols = tuple(md.shape)
    shard = VOCAB // cols
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    fake = SimpleNamespace(cols=cols, md=md)
    torch.manual_seed(7 + U)
    emb = torch.randn(VOCAB, 256).to(torch.bfloat16)
    emb_d = ttnn.from_torch(
        emb,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    for trial in range(6):
        lg = torch.randn(1, 1, U, VOCAB).float()
        if trial % 3 == 1:  # tie inside one column: two equal maxima, the lower index must win
            for u in range(U):
                c = (u + trial) % cols
                a, b = c * shard + 11 + u, c * shard + 5000 + u
                lg[0, 0, u, a] = lg[0, 0, u, b] = 50.0
        if trial % 3 == 2:  # tie across columns: the lower column must win
            for u in range(U):
                a, b = (u % 3) * shard + 77, (u % 3 + 2) * shard + 3
                lg[0, 0, u, a] = lg[0, 0, u, b] = 60.0
        lg_d = ttnn.from_torch(
            lg,
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols)),
        )
        want = lg[0, 0].argmax(-1)  # first maximum
        ref = DSV41DeviceHead.sample_global_rm(fake, lg_d, mc, ccl)
        pair = DSV41DeviceHead.sample_pair_rm(fake, lg_d, mc, ccl, tile_pairs=tile_pairs)
        tok, me = pair_select(pair, shard, emb_d, tile_pairs=tile_pairs)
        ttnn.synchronize_device(md)
        for r in range(rows):
            for c in (0, cols - 1):
                dev = r * cols + c
                t_ref = ttnn.to_torch(ttnn.get_device_tensors(ref)[dev]).reshape(-1).long()
                t_new = ttnn.to_torch(ttnn.get_device_tensors(tok)[dev]).reshape(-1).long()
                e_new = ttnn.to_torch(ttnn.get_device_tensors(me)[dev]).reshape(U, 256)
                assert torch.equal(t_ref, want), (trial, t_ref, want)
                assert torch.equal(t_new, want), (trial, r, c, t_new, want)
                assert torch.equal(e_new, emb[want]), (trial, r, c)
