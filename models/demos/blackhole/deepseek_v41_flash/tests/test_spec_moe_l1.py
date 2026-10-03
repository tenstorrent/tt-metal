# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug: L1 allocator state of the drafter's 128-expert MoE block (DraftMoEBlock) around warmup / first real forward (the M1 'static CB clash')."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.mtp import DraftMoEBlock, load_mtp_moe


def view(md, tag):
    if tag != "start":
        ttnn.synchronize_device(md)
    mv = ttnn.get_memory_view(md, ttnn.BufferType.L1)
    print(
        f"L1VIEW {tag:40s} allocated/bank {mv.total_bytes_allocated_per_bank:8d} largest_free {mv.largest_contiguous_bytes_free_per_bank:8d} base {ttnn.get_allocator_base_address(md, ttnn.BufferType.L1)}",
        flush=True,
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_spec_moe_l1(mesh_device):
    import faulthandler

    faulthandler.dump_traceback_later(600, exit=True)
    md = mesh_device
    T = int(os.environ.get("DSV41_T", "20"))
    sh = _Shards()
    w = load_mtp_moe(0, sh)
    _ = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    print("weights loaded", flush=True)
    from models.demos.gpt_oss.tt.ccl import CCLManager

    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    print("ccl built", flush=True)
    blk = DraftMoEBlock(md, w, T)
    print("block built", flush=True)
    view(md, "after block init")
    blk.warmup()
    view(md, "after warmup")
    rep = ttnn.ReplicateTensorToMesh(md)
    x = torch.randn(1, 1, T, 5120)
    for mem in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        h = ttnn.from_torch(
            x, device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mem, mesh_mapper=rep
        )
        ht = ttnn.from_torch(
            x.reshape(T, 1, 1, 5120),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        view(md, f"before forward (h in {mem.buffer_type})")
        try:
            out = blk.forward(h, ht)
            ttnn.synchronize_device(md)
            view(md, f"after forward OK (h in {mem.buffer_type})")
            ttnn.deallocate(out)
        except Exception as e:
            print(f"FORWARD FAILED (h in {mem.buffer_type}):", str(e)[:300], flush=True)
            try:
                ttnn.dump_device_memory_state(md.get_devices()[0], "l1dump_fail_")
            except Exception as e2:
                print("dump failed", repr(e2)[:200], flush=True)
            break
