# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Minimal standalone repro — B1: all_gather one-tile scatter-write assert (Blackhole 1x4).

An `all_gather` whose per-packet payload is a SINGLE tile makes
`minimal_default_writer.cpp` pre-initialize a fabric *scatter*-write header with
chunk_count == 1. `populate_unicast_scatter_write_fields` (tt_metal/fabric/hw/inc/api_common.h)
asserts `chunk_count >= NOC_SCATTER_WRITE_MIN_CHUNKS (==2)`, so a watcher-enabled BRISC
trips `assert_and_hang` and the watcher stops the device. (Without watcher the ASSERT
compiles out, so the bug is silent — always run this under TT_METAL_WATCHER.)

Trigger = FP32: an FP32 tile is 4 KiB, which fills one fabric packet, so
`num_tiles_to_write_per_packet = min(4, packet_bytes/tile_bytes) = 1`.
The SAME op in bf16 (2 KiB tile -> 2 tiles/packet -> chunk_count==2) passes: that A/B
proves the one-chunk scatter packet is the sole trigger.

No model / weights needed — only a 1x4 mesh.

Run (watcher ON so the assert is live):
  TT_METAL_WATCHER=120 TT_METAL_HOME=$METAL PYTHONPATH=$METAL \
    python3 repros/ccl_one_tile_scatter_assert.py
Expected on buggy build: watcher assert at api_common.h (~L260/277), device stopped.
Expected on fixed build (if constexpr num_tiles_to_write_per_packet>1 guard): both pass.

Or as pytest:
  TT_METAL_WATCHER=120 pytest -svq repros/ccl_one_tile_scatter_assert.py
"""
from __future__ import annotations

import torch
import ttnn

MESH_SHAPE = (1, 4)          # Blackhole P300 qb2, 1x4 line mesh
GATHER_DIM = 1
NUM_LINKS = 2
# Per-device input [1,1,64,32]; gathered along dim=1 across 4 devices -> [1,4,64,32].
# 64x32 FP32 = two 32x32 tiles of 4 KiB each -> one tile per fabric packet.
PER_DEV = (1, 1, 64, 32)
FULL = (1, MESH_SHAPE[1], 64, 32)


def _run_all_gather(mesh_device, dtype):
    """Gather a full tensor sharded on dim=1; return (ok, observed_shape)."""
    host = torch.arange(FULL[0] * FULL[1] * FULL[2] * FULL[3], dtype=torch.float32).reshape(FULL)
    local = ttnn.from_torch(
        host.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, GATHER_DIM), mesh_shape=ttnn.MeshShape(*MESH_SHAPE)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gathered = ttnn.all_gather(                 # plain all_gather manages its own semaphores
        local,
        dim=GATHER_DIM,
        num_links=NUM_LINKS,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=None,
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(mesh_device)        # on the buggy build the watcher assert fires by here
    observed = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).float()
    shape = list(observed.shape)
    ttnn.deallocate(gathered, True)
    ttnn.deallocate(local, True)
    return shape == list(FULL), shape


def _repro():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
    try:
        # PASS control (bf16 -> 2 tiles/packet -> chunk_count==2)
        ok_bf16, sh_bf16 = _run_all_gather(mesh_device, ttnn.bfloat16)
        print(f"[bf16  / 2-tiles-per-packet] shape={sh_bf16} ok={ok_bf16}  (expected: PASS)")

        # FAIL trigger (fp32 -> 1 tile/packet -> scatter chunk_count==1 -> assert_and_hang)
        ok_fp32, sh_fp32 = _run_all_gather(mesh_device, ttnn.float32)
        print(f"[fp32  / 1-tile-per-packet ] shape={sh_fp32} ok={ok_fp32}  "
              f"(buggy build never reaches here: watcher stops device at api_common.h scatter assert)")
        return ok_bf16 and ok_fp32
    finally:
        ttnn.close_mesh_device(mesh_device)


def test_all_gather_one_tile_scatter():
    assert _repro(), "one-tile FP32 all_gather did not complete (scatter chunk_count<2 assert / hang)"


if __name__ == "__main__":
    import sys
    sys.exit(0 if _repro() else 1)
