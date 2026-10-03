# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-call cost (inside a long trace) of the device embedding, Engram layer and LM head, and variants."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

T, D = 4, 5120


def chain_ms(md, fn):
    def run(k):
        def f():
            for _ in range(k):
                fn()

        fn()
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        f()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(20):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        r = (time.perf_counter() - t) / 20 * 1e3
        ttnn.release_trace(md, tid)
        return r

    return (run(3) - run(1)) / 2


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_head_engram_cost(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    sh = _Shards()
    emb = DSV41DeviceEmbedding(md, sh.get("embed.weight"))
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    eng = DSV41DeviceEngram(md, 1, sh)
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard
    )
    x = up(torch.randn(rows * T, 1, 4, D))
    pre = up(torch.rand(rows * T, 1, 1, 4))
    rws = up(torch.randn(rows * T, 1, 1, eng.kin), ttnn.bfloat16)
    toks = emb.upload_tokens(torch.randint(0, 1000, (rows * T,)))
    p = lambda n, ms: print(f"HEC {n:44s} {ms * 1e3:8.1f} us", flush=True)
    p("embedding.forward", chain_ms(md, lambda: emb.forward(toks)))
    p("head.forward (collapse+norm+vocab matmul)", chain_ms(md, lambda: head.forward(x, pre)))
    p("engram.forward (layer 1)", chain_ms(md, lambda: eng.forward(x, rws)))
    # --- breakdown of the head
    y = ttnn.reshape(ttnn.matmul(pre, x, compute_kernel_config=head.ckc32), [1, 1, T, D])
    xb = ttnn.typecast(y, ttnn.bfloat16)
    p("  head: collapse matmul", chain_ms(md, lambda: ttnn.matmul(pre, x, compute_kernel_config=head.ckc32)))
    p(
        "  head: vocab matmul default cfg",
        chain_ms(md, lambda: ttnn.matmul(xb, head.head_w, compute_kernel_config=head.ckc, dtype=ttnn.float32)),
    )
    for gy, gx in ((8, 8), (4, 8)):
        try:
            p(
                f"  head: vocab matmul core_grid {gy}x{gx}",
                chain_ms(
                    md,
                    lambda: ttnn.matmul(
                        xb,
                        head.head_w,
                        compute_kernel_config=head.ckc,
                        dtype=ttnn.float32,
                        core_grid=ttnn.CoreGrid(y=gy, x=gx),
                    ),
                ),
            )
        except Exception as e:
            print("HEC core_grid variant failed", str(e)[:100], flush=True)
    p(
        "  head: vocab matmul bf16 out",
        chain_ms(md, lambda: ttnn.matmul(xb, head.head_w, compute_kernel_config=head.ckc, dtype=ttnn.bfloat16)),
    )
    # --- breakdown of engram
    p(
        "  engram: wkv matmul default cfg",
        chain_ms(md, lambda: ttnn.matmul(rws, eng.wkv_T, compute_kernel_config=eng.ckc, dtype=ttnn.bfloat16)),
    )
    for gy, gx in ((8, 8), (4, 8)):
        try:
            p(
                f"  engram: wkv matmul core_grid {gy}x{gx}",
                chain_ms(
                    md,
                    lambda: ttnn.matmul(
                        rws,
                        eng.wkv_T,
                        compute_kernel_config=eng.ckc,
                        dtype=ttnn.bfloat16,
                        core_grid=ttnn.CoreGrid(y=gy, x=gx),
                    ),
                ),
            )
        except Exception as e:
            print("HEC core_grid variant failed", str(e)[:100], flush=True)

    # ---- column-sharded wkv (tensor parallel) vs replicated: same outputs, less time
    cfg, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    eng_tp = DSV41DeviceEngram(md, 1, sh, mesh_config=cfg, ccl=ccl)
    a = ttnn.to_torch(ttnn.get_device_tensors(eng.forward(x, rws))[0]).float()
    b = ttnn.to_torch(ttnn.get_device_tensors(eng_tp.forward(x, rws))[0]).float()
    print(f"HEC engram TP vs replicated: PCC {R.pcc(a, b):.6f}  max|diff| {(a - b).abs().max().item():.3e}", flush=True)
    p("engram.forward replicated (layer 1)", chain_ms(md, lambda: eng.forward(x, rws)))
    p("engram.forward TP over columns (layer 1)", chain_ms(md, lambda: eng_tp.forward(x, rws)))
