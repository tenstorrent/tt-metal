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
@pytest.mark.parametrize("layer_id,S", [(2, 40)])
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_layer_breakdown(mesh_device, layer_id, S):
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

    def traced_ms(fn, n=50):
        fn()  # compile
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        for _ in range(3):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        t = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t) / n
        ttnn.release_trace(mesh_device, tid)
        return dt * 1e3

    T = per_row
    h_bf = layer._norm(layer._to_row(layer.mhc_attn.collapse(tt_x, tt_pre)), layer.attn_norm_w)  # bf16 [1,1,T,D]
    h_tok = ttnn.reshape(ttnn.to_layout(h_bf, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, 5120])
    a32 = layer._to_tok(ttnn.typecast(h_bf, ttnn.float32))

    def mhc_side(mhc):
        pre, post, comb = mhc.mixes(tt_x)
        hh = layer._norm(mhc.collapse(tt_x, tt_pre), layer.attn_norm_w)
        return mhc.expand(a32, tt_x, post, comb)

    def chain_ms(fn):
        """Per-call time inside a long trace: (t(3 calls) - t(1 call)) / 2, so the trace launch latency drops out."""

        def rep(k):
            def f():
                for _ in range(k):
                    fn()

            return f

        return (traced_ms(rep(3), n=20) - traced_ms(rep(1), n=20)) / 2

    routing = layer.moe.gate.forward(h_bf)  # (weights, indices), persistent inputs for the forced-routing timing

    res = {
        "CHAIN mhc_mixes": chain_ms(lambda: layer.mhc_attn.mixes(tt_x)),
        "CHAIN mhc_collapse+norm": chain_ms(
            lambda: layer._norm(layer._to_row(layer.mhc_attn.collapse(tt_x, tt_pre)), layer.attn_norm_w)
        ),
        "CHAIN mhc_expand": chain_ms(lambda: layer.mhc_attn.expand(a32, tt_x, *layer.mhc_attn.mixes(tt_x)[1:])),
        "CHAIN attention": chain_ms(lambda: attn.forward(h_bf, st)),
        "CHAIN router_exact": chain_ms(lambda: layer.moe.gate._forward_exact(h_bf)),
        "CHAIN router_kernel": chain_ms(lambda: layer.moe.gate._forward_kernel(h_bf)),
        "CHAIN moe_forced_routing(no router)": chain_ms(lambda: layer.moe.forward(h_bf, h_tok, routing)),
        "CHAIN moe_full": chain_ms(lambda: layer.moe.forward(h_bf, h_tok)),
        "CHAIN allgather": chain_ms(
            lambda: layer.mesh_config.allgather(layer.moe.forward(h_bf, h_tok, routing), layer.ccl, axis=1, dim=3)
        ),
        "CHAIN shared_expert": chain_ms(lambda: layer.shared.forward(h_bf)),
        "mhc_mixes": traced_ms(lambda: layer.mhc_attn.mixes(tt_x)),
        "mhc_collapse+norm": traced_ms(
            lambda: layer._norm(layer._to_row(layer.mhc_attn.collapse(tt_x, tt_pre)), layer.attn_norm_w)
        ),
        "mhc_expand": traced_ms(lambda: layer.mhc_attn.expand(a32, tt_x, *layer.mhc_attn.mixes(tt_x)[1:])),
        "attention": traced_ms(lambda: attn.forward(h_bf, st)),
        "moe(gate+experts+RS)": traced_ms(lambda: layer.moe.forward(h_bf, h_tok)),
        "moe_allgather": traced_ms(
            lambda: layer.mesh_config.allgather(layer.moe.forward(h_bf, h_tok), layer.ccl, axis=1, dim=3)
        ),
    }
    print("BREAKDOWN layer", layer_id, " ".join(f"{k}={v:.3f}ms" for k, v in res.items()))
