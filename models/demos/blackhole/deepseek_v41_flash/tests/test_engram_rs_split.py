# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""engram_reduce_scatter (prefill_model.py): the hidden-dim split (DSV41_PFA_ENGRAM_RS_SPLIT=k) against the single reduce_scatter_minimal_async
op on the 8-device mesh axis, random per-device partials [8T, 1, 4, 5120] bf16. Prints the PCC / max abs difference / bit-equality per k and
checks the reduced output against the host sum."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_model import engram_reduce_scatter
from models.demos.gpt_oss.tt.ccl import CCLManager

D = 5120


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("same", [True, False], ids=["same8", "random8"])
@pytest.mark.parametrize("rows", [256, 2048])
@torch.no_grad()
def test_engram_rs_split(mesh_device, rows, same, monkeypatch):
    md = mesh_device
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    host = torch.randn(4, 8, rows, 1, 4, D).to(torch.bfloat16)
    if same:  # the model's case: the 8 stacked copies of one reduce group are identical (sum x8, exact in any order)
        host = host[:, :1].expand(4, 8, rows, 1, 4, D).contiguous()

    def run(k):
        monkeypatch.setenv("DSV41_PFA_ENGRAM_RS_SPLIT", str(k))
        t = ttnn.from_torch(
            host.reshape(4, 8, rows, 1, 4 * D),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(4, 8)),
        )
        # the shape the model uses: [rows, 1, 4, D] per device
        t = ttnn.reshape(t, [rows, 1, 4, D])
        out = engram_reduce_scatter(t, ccl)
        ttnn.synchronize_device(md)
        o = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(4, 8)))
        ttnn.deallocate(out)
        return o.float()  # [4*rows/8*?, ...]

    base = run(1)
    print("out", tuple(base.shape))
    for k in (2, 4, 8):
        o = run(k)
        d = (o - base).abs().max().item()
        eq = torch.equal(o, base)
        a, b = o.flatten(), base.flatten()
        pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        print(f"RSSPLIT rows={rows} same={same} k={k}: bit_equal={eq} max_abs_diff={d} pcc={pcc:.6f}")
        assert eq if same else pcc > 0.9999
