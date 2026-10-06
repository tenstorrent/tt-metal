"""(sd overlap test) Single-layer stage timing + PCC of the MoE collectives and their fused / overlapped alternatives (pf_moeoverlap).
Env: DSV41_UM_LAYER (3), DSV41_UM_N tokens per mesh row (4096), DSV41_UM_REAL=1 (N=512: real post-norm FFN inputs of the S=128 dump), DSV41_UM_RING (1),
DSV41_UM_STAGES (1: per-stage times), DSV41_MO_VARIANTS (comma list of reduce modes to compare, default hidden,a2a)."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt import moe_weights as mw
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import (
    COLS,
    ROWS,
    DSV41UnifiedMoE,
    get_shared,
    route_cols,
)
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

D = 5120


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 400_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_moe_overlap(mesh_device):
    md = mesh_device
    layer = int(os.environ.get("DSV41_UM_LAYER", "3"))
    N = int(os.environ.get("DSV41_UM_N", "4096"))
    w = mw.load_moe_layer(layer)
    blk = DSV41MoEBlock(md, w, batch_per_device=32, gate_bias_shift=0.0)
    blk.warmup()
    get_shared(md, N)
    ring = os.environ.get("DSV41_UM_RING", "1") == "1"
    um = (
        DSV41UnifiedMoE(md, layer, ring=(blk.decode.expert_state.tt_w0_w1, blk.decode.expert_state.tt_w2))
        if ring
        else DSV41UnifiedMoE(md, layer)
    )
    torch.manual_seed(0)
    x = torch.randn(ROWS, N, D).to(torch.bfloat16)
    if os.environ.get(
        "DSV41_UM_REAL"
    ):  # real post-norm FFN inputs of the CPU dumps: N=512 (S=128 x 16 users) or N=4096 (one 4096-token user, rolled per row)
        if N == 512:
            d_ = torch.load(f"/mnt/tt-data/ssinghal/dsv4-prefill-s128/layer_{layer}.pt", mmap=True)["prefill"]["ffn_in"]
            x = d_.float().reshape(ROWS, N, D).to(torch.bfloat16)
        else:
            assert N == 4096
            d_ = torch.load(f"/mnt/tt-data/ssinghal/dsv4-prefill-s4096b1f/layer_{layer}.pt", mmap=True)["prefill"][
                "ffn_in"
            ]
            x = torch.stack([d_.float().reshape(N, D).roll(r * 1024, 0) for r in range(ROWS)]).to(torch.bfloat16)
    n_own = N // COLS
    xo = x.reshape(ROWS, N // 256, COLS, 32, D).permute(0, 2, 1, 3, 4).reshape(ROWS, COLS, n_own, D)
    xod = ttnn.from_torch(
        xo,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(ROWS, COLS)),
    )
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    dev = lambda t, r=0, c=0: ttnn.to_torch(ttnn.get_device_tensors(t)[r * COLS + c])
    free = lambda *ts: [ttnn.deallocate(t) for t in ts]

    # ---- intermediates shared by the stage timings
    h = mc.allgather(xod, ccl, axis=1, dim=2)
    sc_all, ix_all, junk = route_cols(blk.gate, xod, h, mc, ccl)
    x_rm = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [1, N, D])
    part = um.forward(x_rm, sc_all, ix_all)
    ttnn.synchronize_device(md)

    def T(name, fn):
        print(f"MO time N={N} {name:34s} {chain_ms(md, fn):.3f} ms", flush=True)

    from models.demos.blackhole.deepseek_v41_flash.tt.moe_overlap import SDOverlap, SegTrace
    from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import shared_big
    from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_v2 import DSV41SharedExpertV2

    g_ = torch.Generator().manual_seed(5)
    w0, w1 = [(torch.randn(1, 1, D, 2304, generator=g_) * 0.02) for _ in range(2)]
    w2 = torch.randn(1, 1, 2304, D, generator=g_) * 0.02
    sh = DSV41SharedExpertV2(md, w0, w1, w2)
    ov = SDOverlap.get(md)
    print(f"MO sub-device grid {ov.gx}x{ov.gy}, shared grid {ov.s_grid}", flush=True)
    ov.split_weights(sh)
    res = {}

    def seq():
        so = shared_big(sh, xod)
        p = um.forward(x_rm, sc_all, ix_all)
        res["seq"] = (so, p)

    def ovl():
        box = []
        p = um.forward(x_rm, sc_all, ix_all, overlap=(ov, lambda: box.append(ov.shared(sh, xod))))
        res["ovl"] = (box[0], p)

    def timed(name, fn, seg=None, iters=10):
        fn()  # compile
        ttnn.synchronize_device(md)
        if seg is None:
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            fn()
            ttnn.end_trace_capture(md, tid, cq_id=0)
            run = lambda: ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        else:
            ov.seg = seg
            seg.begin()
            fn()
            seg.end()
            ov.seg = None
            run = seg.replay
        import time

        for _ in range(3):
            run()
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(iters):
            run()
        ttnn.synchronize_device(md)
        print(f"MO time N={N} {name}: {(time.perf_counter() - t) / iters * 1e3:.3f} ms", flush=True)

    seq()
    ovl()
    ttnn.synchronize_device(md)
    for r in (0, 3):
        for c in (0, 5):
            a, b = dev(res["seq"][0], r, c).float(), dev(res["ovl"][0], r, c).float()
            pa, pb = dev(res["seq"][1], r, c).float(), dev(res["ovl"][1], r, c).float()
            print(
                f"MO eager shared pcc seq vs overlap ({r},{c}): {pcc(a, b):.6f}  moe partial: {pcc(pa, pb):.6f} equal {bool(torch.equal(pa, pb))}",
                flush=True,
            )
    timed("sequential (shared_big + MoE section)", seq)
    seg = SegTrace(md)
    timed("overlapped (segmented trace)", ovl, seg=seg)
    ttnn.synchronize_device(md)
    for _ in range(5):
        seg.replay()
    ttnn.synchronize_device(md)
    for r in (0, 3):
        for c in (0, 5):
            a, b = dev(res["seq"][0], r, c).float(), dev(res["ovl"][0], r, c).float()
            pa, pb = dev(res["seq"][1], r, c).float(), dev(res["ovl"][1], r, c).float()
            print(
                f"MO traced shared pcc seq vs overlap ({r},{c}): {pcc(a, b):.6f}  moe partial: {pcc(pa, pb):.6f} equal {bool(torch.equal(pa, pb))}",
                flush=True,
            )
