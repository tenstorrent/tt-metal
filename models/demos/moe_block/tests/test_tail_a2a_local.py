# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tail_a2a's local mode (MoEOptions residual "owners" with tail "fused"; DeepSeek-V4.1 P.23 OWN_ROWS): out = tilize(y)
+ se on each chip's own rows [S/N, H], one program for the mesh, no all-to-all - against torch to a bf16 ulp (the add's
rounding), eager and prepared, with and without se, workers inside max_grid. Any mesh: MOE_MESH (default 2x4) and
MOE_FABRIC ("" FABRIC_2D, torus_x, torus_xy: a Galaxy 8x4 opens the torus)."""

import os

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.tt.tail_a2a import TailA2A

R, C = (int(v) for v in os.environ.get("MOE_MESH", "2x4").split("x"))
_FABRIC = {"torus_x": "FABRIC_2D_TORUS_X", "torus_xy": "FABRIC_2D_TORUS_XY"}.get(os.environ.get("MOE_FABRIC", ""))


def _params():
    p = fabric2d_device_params(l1_small_size=0)
    if _FABRIC:
        p["fabric_config"] = getattr(ttnn.FabricConfig, _FABRIC)
    return p


@pytest.mark.parametrize("mesh_device, device_params", [((R, C), _params())], indirect=True)
@pytest.mark.parametrize("H, ROWS", [(5120, 640 * C // 4), (4096, 1024)])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("with_se", [True, False])
def test_tail_a2a_local(mesh_device, device_params, H, ROWS, prepared, with_se):
    S = ROWS // C
    tail = TailA2A(
        mesh_device,
        rows=ROWS,
        H=H,
        topology=ttnn.Topology.Linear,
        num_links=2,
        prepared=prepared,
        local=True,
        max_grid=(11, 10),
    )
    assert tail.out_shape() == [1, 1, S, H]
    assert all(c.x < 11 and c.y < 10 for c in tail.workers) and len(tail.workers) <= 110
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    g_ = torch.Generator().manual_seed(1)
    for call in range(2):
        y = torch.randn(R, C, S, H, generator=g_).to(torch.bfloat16)
        se = torch.randn(R, C, S, H, generator=g_).to(torch.bfloat16)
        yt = ttnn.from_torch(
            y, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        st = (
            ttnn.from_torch(se, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)
            if with_se
            else None
        )
        out = tail.run(yt, st)
        assert tuple(out.shape) == (1, 1, S, H) and out.layout == ttnn.TILE_LAYOUT
        parts = [ttnn.to_torch(t).reshape(S, H).float() for t in ttnn.get_device_tensors(out)]
        want = (y.float() + se.float()).to(torch.bfloat16).float() if with_se else y.float()
        for d, got in enumerate(parts):
            ref = want[d // C, d % C]
            # (one bf16 rounding of the add: the device's packer and torch may differ by an ulp on ties)
            diff = (got - ref).abs() - ref.abs() / 128
            assert diff.max() <= 1e-6, f"call {call}: chip {divmod(d, C)} off by {diff.max():.3g} beyond a bf16 ulp"
        for t in (yt, st, out):
            if t is not None:
                ttnn.deallocate(t)
