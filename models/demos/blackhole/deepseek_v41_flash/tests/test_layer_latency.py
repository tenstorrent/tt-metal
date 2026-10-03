# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Eager decode latency of one DeepSeek-V4.1-Flash layer (4x8 Blackhole, batch 16), with a per-section breakdown.

State is synthetic (zero caches, position S), weights are real. Numbers are wall-clock of the host-driven eager
path: they include host dispatch overhead, which tracing would remove.
"""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.reference.calibrate import calibrate_gate_cutoff
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention, DSV41CompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.layer import DSV41Layer
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("layer_id,S", [(0, 40), (2, 40), (20, 40)])
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_layer_latency(mesh_device, layer_id, S):
    rows, cols = tuple(mesh_device.shape)
    per_row, B = 4, rows * 4
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=256)
    ratio = blk.attn.compress_ratio
    mesh_config = mesh_4x8()
    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    weights = R.dequantized_attention_weights(blk)
    if ratio == 0:
        attn = DSV41Attention(
            mesh_device, mesh_config, ccl, weights, blk.attn.freqs_cis, users_per_row=per_row, max_seq=256
        )
    else:
        comp = blk.attn.compressor
        comp_w = {"wkv": comp.wkv.weight.data.float(), "norm": comp.norm.weight.data.float()}
        if ratio > 1:
            comp_w["wgate"] = comp.wgate.weight.data.float()
        attn = DSV41CompressedAttention(
            mesh_device,
            mesh_config,
            ccl,
            weights,
            blk.attn.freqs_cis,
            ratio,
            comp_w,
            users_per_row=per_row,
            max_comp=128,
        )
        z = torch.zeros(B, 128, 512)
        attn.load_state(
            z,
            torch.zeros(B, S // ratio, 512),
            torch.zeros(B, max(ratio, 1), 512),
            torch.full((B, max(ratio, 1), 512), -1e30),
        )
    mhc = lambda n: (
        getattr(blk, f"hc_{n}_fn").data,
        getattr(blk, f"hc_{n}_base").data,
        getattr(blk, f"hc_{n}_scale").data,
    )
    layer = DSV41Layer(
        mesh_device,
        mesh_config,
        ccl,
        attn,
        norms={"attn_norm": blk.attn_norm.weight.data.float(), "ffn_norm": blk.ffn_norm.weight.data.float()},
        mhc_params={"attn": mhc("attn"), "ffn": mhc("ffn")},
        moe_weights=load_moe_layer(layer_id),
        gate_bias_shift=calibrate_gate_cutoff(layer_id, seed=99),
        users_per_row=per_row,
    )
    shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(rows, cols))
    up = lambda t: ttnn.from_torch(
        t.float(),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    h, pm = R.embed_tokens(torch.randint(1000, 100000, (B, 1)))
    tt_x, tt_pre = up(h.reshape(B, 1, 4, 5120)), up(pm.reshape(B, 1, 1, 4))
    st = attn.step_inputs(torch.full((B,), S))

    import functools
    import gc
    import os

    if os.environ.get("DSV_OP_TRACE"):  # diagnostics: allocator reading after each MoE decode op
        log = open(os.environ["DSV_OP_TRACE"], "w")

        def wrap(owner, name):
            fn = getattr(owner, name)

            @functools.wraps(fn)
            def traced(*a, **k):
                before = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
                r = fn(*a, **k)
                ttnn.synchronize_device(mesh_device)
                mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
                log.write(
                    f"{name:42s} alloc {before.total_bytes_allocated_per_bank:7d}->{mv.total_bytes_allocated_per_bank:7d}  largest_free {before.largest_contiguous_bytes_free_per_bank:8d}->{mv.largest_contiguous_bytes_free_per_bank:8d}\n"
                )
                log.flush()
                return r

            setattr(owner, name, traced)

        for nm in (
            "all_to_all_dispatch_metadata",
            "moe_compute",
            "deepseek_moe_post_combine_tilize",
            "deepseek_moe_fast_reduce_nc_fused",
        ):
            wrap(ttnn.experimental, nm)
        wrap(ttnn, "reduce_scatter")
        wrap(ttnn.experimental.deepseek.moe, "generalized_moe_gate")

    def l1_used(tag):
        try:
            mv = ttnn.get_memory_view(
                mesh_device.get_device(0) if hasattr(mesh_device, "get_device") else mesh_device, ttnn.BufferType.L1
            )
            print(
                f"L1[{tag}]: allocated/bank {mv.total_bytes_allocated_per_bank}  free/bank {mv.total_bytes_free_per_bank}  largest_free {mv.largest_contiguous_bytes_free_per_bank}"
            )
        except Exception as e:  # diagnostics only
            print(f"L1[{tag}]: n/a ({type(e).__name__}: {e})")

    l1_used("before first forward")
    for i in range(3):  # warm-up: program compile + caches
        print(f"--- warm-up iteration {i}")
        out, nxt = layer.forward(tt_x, tt_pre, st)
        ttnn.synchronize_device(mesh_device)
        l1_used(f"after forward {i} (outputs alive)")
        ttnn.deallocate(out)
        ttnn.deallocate(nxt)
        del out, nxt
        gc.collect()
        l1_used(f"after forward {i} (outputs freed, gc)")
    n = 20
    t = time.perf_counter()
    for _ in range(n):
        out, nxt = layer.forward(tt_x, tt_pre, st)
    ttnn.synchronize_device(mesh_device)
    eager = (time.perf_counter() - t) / n
    prof = {}
    for _ in range(n):
        layer.forward(tt_x, tt_pre, st, profile=prof)
    rows_txt = "  ".join(f"{k}={v / n * 1e3:.2f}" for k, v in prof.items())
    print(
        f"LATENCY layer {layer_id} (ratio {ratio}, batch {B}): eager {eager * 1e3:.2f} ms/layer | serialised sections (ms): {rows_txt}"
    )

    # traced: capture one layer step and replay it (no per-op host dispatch)
    traced = None
    try:
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        t_out, t_nxt = layer.forward(tt_x, tt_pre, st)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        for _ in range(3):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        n2 = 50
        t = time.perf_counter()
        for _ in range(n2):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        traced = (time.perf_counter() - t) / n2
        ttnn.release_trace(mesh_device, tid)
    except Exception as e:
        print(f"TRACE FAILED layer {layer_id}: {type(e).__name__}: {str(e)[:300]}")
    if traced is not None:
        print(
            f"LATENCY_TRACED layer {layer_id} (ratio {ratio}, batch {B}): {traced * 1e3:.3f} ms/layer  (x40 layers = {traced * 40 * 1e3:.1f} ms/token -> {1.0 / (traced * 40):.1f} tok/s/user, layer-only extrapolation)"
        )
