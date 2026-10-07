# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident live mask for the decode MoE modules: the mask [1,1,64,1] is derived ON DEVICE from per-row dead/live flags or live counts (MiMoMoEDecode.mask_from_flags / mask_from_count), no host copy
inside the step. Checks (4x8, real layer MIMO_LAYER=5): (1) derived masks equal the host-built ones for ragged per-row patterns, several flag dtypes/layouts, (2) ring and unified decode MoE give BIT-IDENTICAL outputs with
the derived mask vs the host mask, eager and in a trace (the flags/counts are updated in place between replays by the test harness only, standing in for the decode step's device-side update), (3) no allocation
after capture is needed (replays with new flags match the eager result for the new flags)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.moe_decode import SLOTS, MiMoMoEDecode
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    HIDDEN,
    ROWS,
    MiMoPrefillMoE,
    build_expert_weights,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.moe_ring import MiMoRingMoEDecode, RingExpertWeights
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

D1 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
LAYER = int(os.environ.get("MIMO_LAYER", "5"))
T = int(os.environ.get("MIMO_T", "8"))


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
def test_moe_mask_device(mesh_device):
    md = mesh_device
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    W = load_layer_experts(LAYER)
    gw, gb = load_gate(LAYER)
    gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
    for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
        ttnn.deallocate(getattr(gate, n))
    moe = MiMoPrefillMoE(
        md,
        build_expert_weights(md, lambda e: W[e], nd_sharded=True),
        ckc=make_ckc("hifi2", fp32acc=True, packer_l1_acc=True),
        expert_op="unified",
        cf=4,
    )
    uni = MiMoMoEDecode(md, gate, moe, router="fast", dead="spread32")
    ring = MiMoRingMoEDecode(md, gate, RingExpertWeights(md, LAYER, w_of_expert=lambda e: W[e]), router="fast")
    a = torch.load(f"{D1}/layer_{LAYER}.pt", mmap=True)["prefill"]
    sel = torch.arange(0, 512, 16)[:32]
    x = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel][:, :T].reshape(ROWS, 1, T, HIDDEN).clone()
    xd = ttnn.from_torch(
        x,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=m2,
    )

    def host_mask_of(counts):
        return uni.make_mask_host(counts)

    def dev_mask_to_host(m):
        d = ttnn.get_device_tensors(m)
        return torch.stack([ttnn.to_torch(d[r * COLS]).reshape(-1)[:SLOTS].float() for r in range(ROWS)])

    patterns = [[T, T, T, T], [T, max(1, T // 2), 3, 1], [1, 1, 1, 1], [2, 0, T, 5]]
    # (1) derived == host
    for cnt in patterns:
        cnt = [min(c, T) for c in cnt]
        ref = host_mask_of(cnt).reshape(ROWS, SLOTS).float()
        flags = torch.zeros(ROWS, 1, 1, 32)
        for r, c in enumerate(cnt):
            flags[r, 0, 0, :c] = 1
        for name, mk in (
            (
                "u32 RM [rows,1,1,32]",
                lambda: ttnn.from_torch(
                    flags.to(torch.int32), device=md, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=m2
                ),
            ),
            (
                "bf16 tile",
                lambda: ttnn.from_torch(
                    flags.bfloat16(), device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=m2
                ),
            ),
            (
                "i32 RM",
                lambda: ttnn.from_torch(
                    flags.to(torch.int32), device=md, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=m2
                ),
            ),
        ):
            got = dev_mask_to_host(uni.mask_from_flags(mk()))
            assert torch.equal(got, ref), (name, cnt, got[:, :10], ref[:, :10])
        cn = torch.tensor(cnt, dtype=torch.int32).reshape(ROWS, 1, 1, 1)
        for name, mk in (
            (
                "count u32",
                lambda: ttnn.from_torch(cn, device=md, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=m2),
            ),
            (
                "count bf16",
                lambda: ttnn.from_torch(
                    cn.bfloat16(), device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=m2
                ),
            ),
        ):
            got = dev_mask_to_host(uni.mask_from_count(mk()))
            assert torch.equal(got, ref), (name, cnt)
    print("RESULT derived masks (flags u32/i32/bf16, counts u32/bf16) == host masks for 4 ragged patterns", flush=True)

    # (2)+(3) forward with host mask vs device-derived mask, eager and traced with in-place flag updates
    def out_host(t):
        d = ttnn.get_device_tensors(t)
        return torch.stack([ttnn.to_torch(d[r * COLS]).reshape(-1, HIDDEN).float() for r in range(ROWS)])

    flags_dev = ttnn.from_torch(
        torch.ones(ROWS, 1, 1, 32).to(torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=m2,
    )

    def set_flags(cnt):
        fl = torch.zeros(ROWS, 1, 1, 32)
        for r, c in enumerate(cnt):
            fl[r, 0, 0, :c] = 1
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(fl.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=m2),
            flags_dev,
        )

    for name, dec in (("unified", uni), ("ring", ring)):
        for cnt in patterns[:3]:
            cnt = [min(c, T) for c in cnt]
            hm = uni.alloc_mask(host_mask_of(cnt))
            o_host = out_host(dec.forward(xd, hm))
            set_flags(cnt)
            o_dev = out_host(dec.forward(xd, uni.mask_from_flags(flags_dev)))
            assert torch.equal(o_host, o_dev), (name, cnt, float((o_host - o_dev).abs().max()))
        # traced: derive the mask inside the trace from flags_dev, replay for other flags
        set_flags([T] * ROWS)
        dec.forward(xd, uni.mask_from_flags(flags_dev))
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        ot = dec.forward(xd, uni.mask_from_flags(flags_dev))
        ttnn.end_trace_capture(md, tid, cq_id=0)
        for cnt in patterns:
            cnt = [min(c, T) for c in cnt]
            set_flags(cnt)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            hm = uni.alloc_mask(host_mask_of(cnt))
            ref = out_host(dec.forward(xd, hm))
            got = out_host(ot)
            live = torch.zeros(ROWS, T, dtype=torch.bool)
            for r, c in enumerate(cnt):
                live[r, :c] = True
            assert torch.equal(got[:, :T][live], ref[:, :T][live]), (name, cnt)
        ttnn.release_trace(md, tid)
        print(
            f"RESULT {name}: device-derived mask forward bit-identical to host-mask forward (eager 3 patterns, traced with in-place flag updates, 4 patterns, live rows)",
            flush=True,
        )
