# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of block_keys, the golden's topology: tensors replicated over the whole system mesh (the golden
suite opens it with ttnn.open_mesh_device()). The golden's largest and smallest block_keys inputs; every chip's output
is checked byte-exact against the golden reference. (The golden also sets FABRIC_2D; block_keys moves nothing between
chips, so this opens the mesh without it.)"""

import pytest
import torch
import ttnn
from loguru import logger

ROWS = 96


@pytest.mark.parametrize(("W", "K", "B"), [(65536, 512, 8), (512, 16, 2)], ids=["w65536_k512_b8", "w512_k16_b2"])
def test_golden_cut_mesh(system_mesh, block_keys_op, bk, W, K, B):
    mesh = system_mesh
    system = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
    logger.info(f"block_keys golden mesh cut: {mesh.get_num_devices()} devices, mesh shape {mesh.shape}")
    assert mesh.get_num_devices() == system, (mesh.get_num_devices(), system)
    g = torch.Generator().manual_seed(W + K)
    nb = W // B
    ids = torch.randint(0, nb, (1, 1, ROWS, K), generator=g)
    ids[..., 0, -5:] = bk.SENT
    ids[..., 1, 3] = nb
    ids[..., 2, :] = 0
    pos = torch.randint(0, K * B, (1, 1, ROWS, min(512, K * B)), generator=g)
    pos[..., 4, :7] = bk.SENT
    pt, it = bk.up(pos, mesh, mesh=True), bk.up(ids, mesh, mesh=True)
    want = bk.ref_keys(pos, ids, B, W)
    keys = block_keys_op(pt, it, block=B, width=W)
    bk.assert_dram_rm_u32(keys, list(pos.shape))
    shards = bk.down_all(keys)
    assert len(shards) == mesh.get_num_devices()
    for d, got in enumerate(shards):
        bk.assert_exact(got, want, f"mesh device {d}: W={W} K={K} B={B}")
