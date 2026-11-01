# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/fuse_attn.py ids_build vs the unfused chain of prefill_sparse.attend_dyn (slice / concat / compact_rows), U=2 users x C=512 rows, one chip, exact equality."""
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import fuse_attn as FA
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_sparse as PS

U, C = 2, 512


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="p")], indirect=True)
@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("perm_per_user", [True, False])
@torch.no_grad()
def test_ids_build(mesh_device, first, perm_per_user):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t.to(torch.int32).contiguous(),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    torch.manual_seed(1)
    win = torch.randint(0, 170000, (U, 1, C, 128))
    ids = torch.randint(0, 170000, (U, 1, C, 512))
    nvalid = torch.randint(1, 513, (U, 1, C, 1))
    ids = torch.where(
        torch.arange(512).view(1, 1, 1, -1) < nvalid, ids, torch.full_like(ids, -1)
    )  # sentinel tail 0xFFFFFFFF
    perm = PS.perm_table(first).repeat(U if perm_per_user else 1, 1, 1, 1)
    # make the permutation differ per user
    if perm_per_user:
        perm[1] = PS.perm_table(not first)[0]
    t_win, t_ids, t_perm = up(win), up(ids), up(perm)
    got = FA.ids_build(t_win, t_ids, t_perm, U, C)
    for u in range(U):
        pu = perm[u : u + 1] if perm_per_user else perm
        ref = PS.compact_rows(ttnn.concat([up(win[u : u + 1]), up(ids[u : u + 1])], dim=3), up(pu))
        r = ttnn.to_torch(ttnn.get_device_tensors(ref)[0]).to(torch.int64) & 0xFFFFFFFF
        g = ttnn.to_torch(ttnn.get_device_tensors(got[u])[0]).to(torch.int64) & 0xFFFFFFFF
        eq = torch.equal(r.reshape(g.shape), g)
        print(f"ids_build first={first} perm_per_user={perm_per_user} user {u}: equal {eq}", flush=True)
        assert eq
