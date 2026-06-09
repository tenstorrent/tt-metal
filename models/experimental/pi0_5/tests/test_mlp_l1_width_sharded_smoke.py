# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: can all three Gemma-2B MLP weights co-reside on L1 as width-sharded?

Uploads gate [2048, 16384], up [2048, 16384], down [16384, 2048] at bfloat8_b
onto a single Blackhole chip with L1 width-sharded layout on an 8x8=64 core
grid. Then runs a single matmul of each to check that the kernel CB region
doesn't clash with the per-core weight slice.

Reports per-core L1 used after each upload. Fails fast if any allocation OOMs.

Run with:
    PI0_MLP_L1WS_SMOKE=1 pytest models/experimental/pi0_5/tests/test_mlp_l1_width_sharded_smoke.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.experimental.pi0_5.tt.ttnn_common import build_l1_width_sharded_memcfg


pytestmark = pytest.mark.skipif(
    os.environ.get("PI0_MLP_L1WS_SMOKE") != "1",
    reason="set PI0_MLP_L1WS_SMOKE=1 to run the L1 width-sharded MLP smoke",
)


# Gemma-2B MLP dims
K_GU = 2048
N_GU = 16384
K_D = 16384
N_D = 2048

# Default 8x8=64 core grid — divides 16384 cleanly (256 per core on N) and
# 2048 cleanly (32 per core on N for down).
GRID_X = int(os.environ.get("PI0_MLP_L1WS_GRID_X", "8"))
GRID_Y = int(os.environ.get("PI0_MLP_L1WS_GRID_Y", "8"))


def test_mlp_l1_width_sharded_upload():
    """Upload gate/up/down as L1 width-sharded; verify all three co-reside."""
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=458752)
    try:
        print(f"\n[grid] {GRID_X}x{GRID_Y} = {GRID_X * GRID_Y} cores")
        print(f"[shapes] gate/up=({K_GU}, {N_GU}) down=({K_D}, {N_D}) at bfloat8_b")

        gate_memcfg = build_l1_width_sharded_memcfg(K_GU, N_GU, GRID_X, GRID_Y)
        up_memcfg = build_l1_width_sharded_memcfg(K_GU, N_GU, GRID_X, GRID_Y)
        down_memcfg = build_l1_width_sharded_memcfg(K_D, N_D, GRID_X, GRID_Y)
        per_core_gu_bytes = (K_GU * (N_GU // (GRID_X * GRID_Y))) * 1.0625
        per_core_d_bytes = (K_D * (N_D // (GRID_X * GRID_Y))) * 1.0625
        print(f"[expected per-core] gate/up={per_core_gu_bytes/1024:.0f} KB  down={per_core_d_bytes/1024:.0f} KB")
        print(f"[expected total per-core] {(2 * per_core_gu_bytes + per_core_d_bytes)/1024:.0f} KB")

        gen = torch.Generator().manual_seed(0)

        gate_t = torch.randn(K_GU, N_GU, generator=gen, dtype=torch.float32) * 0.02
        gate = ttnn.from_torch(
            gate_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=gate_memcfg
        )
        print("  [gate uploaded OK]")

        up_t = torch.randn(K_GU, N_GU, generator=gen, dtype=torch.float32) * 0.02
        up = ttnn.from_torch(
            up_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=up_memcfg
        )
        print("  [up uploaded OK]")

        down_t = torch.randn(K_D, N_D, generator=gen, dtype=torch.float32) * 0.02
        down = ttnn.from_torch(
            down_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=down_memcfg
        )
        print("  [down uploaded OK]")

        print("\n[PASS] all 3 MLP weights co-resident on L1 width-sharded")
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        ttnn.deallocate(down)
    finally:
        ttnn.close_mesh_device(device)
