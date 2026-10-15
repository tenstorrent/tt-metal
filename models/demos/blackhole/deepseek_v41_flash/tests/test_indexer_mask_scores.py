# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tt/indexer.py ``mask_scores`` ([1,1,U,n] layout) against the old [U,1,1,n] formulation: bit-identical masked scores, and the DRAM it needs at the 1M spec-verify shape (24 rows x 262144)."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import mask_scores


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@torch.no_grad()
def test_mask_scores(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt, lay: ttnn.from_torch(t, device=md, dtype=dt, layout=lay, mesh_mapper=rep)
    for U, n in ((6, 4096), (24, 262144)):
        s = torch.randn(U, 1, 1, n).to(torch.bfloat16)
        nv = torch.randint(1, n, (U, 1, 1, 1)).float()
        scores = up(s, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        nvalid = up(nv, ttnn.float32, ttnn.TILE_LAYOUT)
        ar = up(torch.arange(n, dtype=torch.float32).reshape(1, 1, 1, n), ttnn.float32, ttnn.TILE_LAYOUT)
        got = ttnn.to_torch(ttnn.get_device_tensors(mask_scores(scores, nvalid, ar))[0]).reshape(U, n)
        want = torch.where(torch.arange(n).view(1, -1) < nv.view(U, 1), s.view(U, n).float(), torch.tensor(-1e30)).to(
            torch.bfloat16
        )
        print(f"MS U={U} n={n}: bit-exact {bool(torch.equal(got, want))}", flush=True)
        assert torch.equal(got, want)
