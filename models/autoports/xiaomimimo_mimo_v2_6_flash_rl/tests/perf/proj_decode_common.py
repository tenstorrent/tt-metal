# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the MiMo decode projection perf / correctness scripts (tt/proj_stream.py, Galaxy row 0).

Per chip of a TP=8 SWA layer: qkv [4096, 1920] BFP8 and o [1024, 4096] BFP8 (DRAM interleaved, HiFi2), router
[4096, 256] bf16 (DRAM-sharded, HiFi4, fp32 out). Random weights at the real shapes, dtypes and layouts; 4 users in
rows 0-3 of the 32-row activation tile.
"""

import torch

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import moe_decode_common as common
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.common import MeshPlan
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.proj_stream import DecodeProjections

HIDDEN, QKV_WIDTH, O_IN, EXPERTS, USERS = 4096, 1920, 1024, 256, 4


def build(row, seed=0):
    plan = MeshPlan(row)
    proj = DecodeProjections(plan, HIDDEN, QKV_WIDTH, O_IN, EXPERTS)
    g = torch.Generator().manual_seed(seed)
    rep = plan.replicate()

    def dev(t, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            t, device=row, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, mesh_mapper=rep
        )

    w_qkv = dev(torch.randn(HIDDEN, QKV_WIDTH, generator=g) * 0.02, ttnn.bfloat8_b)
    w_o = dev(torch.randn(O_IN, HIDDEN, generator=g) * 0.02, ttnn.bfloat8_b)
    w_router = dev(torch.randn(HIDDEN, EXPERTS, generator=g) * 0.02, ttnn.bfloat16, proj.router_mm.weight_memory_config)
    weights = proj.prepare_weights(w_qkv, w_o, w_router)
    return plan, proj, weights


def inputs(row, step, seed=0):
    """x for qkv / router [1, 1, 32, 4096] (DRAM) and for o [1, 1, 32, 1024] (L1), users in rows 0-3, rest zero."""
    g = torch.Generator().manual_seed(1000 + step + 100 * seed)
    rep = ttnn.ReplicateTensorToMesh(row)
    xs = []
    for k, mem in ((HIDDEN, ttnn.DRAM_MEMORY_CONFIG), (O_IN, ttnn.L1_MEMORY_CONFIG)):
        x = torch.zeros(1, 1, 32, k)
        x[0, 0, :USERS] = torch.randn(USERS, k, generator=g)
        xs.append(
            ttnn.from_torch(
                x.bfloat16(),
                device=row,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=rep,
            )
        )
    return xs


def run(proj, weights, x_h, x_o):
    w_qkv, w_o, w_router = weights
    return proj.qkv(x_h, w_qkv), proj.o(x_o, w_o), proj.router(x_h, w_router)


open_row, close = common.open_row, common.close
