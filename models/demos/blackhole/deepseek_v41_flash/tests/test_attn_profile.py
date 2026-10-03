# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage in-trace cost of a compressed attention layer ((t(3) - t(1)) / 2 chained). Random weights, real shapes.
ATTN_IMPL=ref (tests/attention_ref_impl.py) or new (tt/attention.py). Prints 'PR name us'."""

import importlib
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_probe_matmul import chain_ms
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def make_weights(bf=torch.bfloat16):
    g = lambda *s: (torch.randn(*s) * 0.02).to(bf)
    return {
        "wq_a": g(1280, 5120),
        "q_norm": torch.ones(1280),
        "wq_b": g(32768, 1280),
        "wkv": g(512, 5120),
        "kv_norm": torch.ones(512),
        "wo_a": g(8192, 4096),
        "wo_b": g(5120, 8192),
        "attn_sink": torch.randn(64),
    }


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
@pytest.mark.parametrize("ratio,S", [(2, 24), (2, 23), (1, 24), (0, 24)])
@torch.no_grad()
def test_attn_profile(mesh_device, ratio, S):
    md = mesh_device
    mod = importlib.import_module(
        "models.demos.blackhole.deepseek_v41_flash.tests.attention_ref_impl"
        if os.environ.get("ATTN_IMPL", "new") == "ref"
        else "models.demos.blackhole.deepseek_v41_flash.tt.attention"
    )
    torch.manual_seed(0)
    rows, cols = tuple(md.shape)
    T, B = 4, rows * 4
    freqs = torch.polar(torch.ones(256, 32), torch.rand(256, 32))
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w = make_weights()
    if ratio == 0:
        attn = mod.DSV41Attention(md, mesh_4x8(), ccl, w, freqs, users_per_row=T, max_seq=256)
        attn.load_window(torch.randn(B, S, 512))
    else:
        cw = {"wkv": (torch.randn(512, 5120) * 0.02), "norm": torch.ones(512)}
        if ratio > 1:
            cw["wgate"] = torch.randn(512, 5120) * 0.02
        attn = mod.DSV41CompressedAttention(md, mesh_4x8(), ccl, w, freqs, ratio, cw, users_per_row=T, max_comp=128)
        attn.load_state(
            torch.randn(B, 128, 512),
            torch.randn(B, S // ratio, 512),
            torch.randn(B, ratio, 512),
            torch.randn(B, ratio, 512),
        )
    st = attn.step_inputs(torch.full((B,), S))
    shard = ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols))
    x = ttnn.from_torch(
        torch.randn(1, 1, B, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    res = {"forward": chain_ms(md, lambda: attn.forward(x, st))}
    if os.environ.get("ATTN_STAGES", "1") == "1" and hasattr(attn, "_rope_inv"):
        lat = None
        if ratio:
            res["compress_step"] = chain_ms(md, lambda: attn._compress_step(x, st))
            lat = attn._compress_step(x, st)
        res["_qkv"] = chain_ms(md, lambda: attn._qkv(x, st, lat))
        q, k, v = attn._qkv(x, st, lat)
        res["write_cache"] = chain_ms(
            md, lambda: attn._write_cache(attn.cache, k, st["pos_ring"] if ratio else st["pos"])
        )
        if ratio:
            o = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                attn.cache,
                attn.cache,
                is_causal=False,
                attn_mask=st["mask"],
                attention_sink=attn.sinks,
                scale=attn.scale,
                program_config=attn._sdpa_cfg(attn._k_chunk),
                compute_kernel_config=attn.ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            res["sdpa"] = chain_ms(
                md,
                lambda: ttnn.transformer.scaled_dot_product_attention_decode(
                    q,
                    attn.cache,
                    attn.cache,
                    is_causal=False,
                    attn_mask=st["mask"],
                    attention_sink=attn.sinks,
                    scale=attn.scale,
                    program_config=attn._sdpa_cfg(attn._k_chunk),
                    compute_kernel_config=attn.ckc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
        else:
            f = lambda: ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                attn.cache,
                attn.cache,
                cur_pos_tensor=st["pos"],
                sliding_window_size=128,
                attention_sink=attn.sinks,
                scale=attn.scale,
                program_config=attn._sdpa_cfg(128),
                compute_kernel_config=attn.ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            o = f()
            res["sdpa"] = chain_ms(md, f)
        res["_finish"] = chain_ms(md, lambda: attn._finish(o, st))
        res["  rope_inv"] = chain_ms(md, lambda: attn._rope_inv(o, st))
        oi = attn._rope_inv(o, st)
        res["  nlp_concat"] = chain_ms(md, lambda: ttnn.experimental.nlp_concat_heads_decode(oi, num_heads=9))
        c = ttnn.experimental.nlp_concat_heads_decode(oi, num_heads=9)
        res["  wo_a"] = chain_ms(md, lambda: attn._lin(c, attn.wo_a, "OA"))
        a1 = attn._lin(c, attn.wo_a, "OA")
        res["  wo_b"] = chain_ms(md, lambda: attn._lin(a1, attn.wo_b, "OB"))
        part = attn._lin(a1, attn.wo_b, "OB")

        def ar(t):  # MeshConfig.allreduce without the input deallocation (so it can be replayed on the same tensor)
            sc = ttnn.experimental.reduce_scatter_minimal_async(
                t,
                dim=3,
                multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
                num_links=ccl.num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ccl.topology,
                cluster_axis=1,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )
            return ttnn.experimental.all_gather_async(
                sc,
                dim=3,
                cluster_axis=1,
                mesh_device=md,
                topology=ccl.topology,
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                num_links=ccl.num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=ccl.get_barrier_semaphore(),
            )

        res["  allreduce"] = chain_ms(md, lambda: ar(part))
        res["  wqkv"] = chain_ms(md, lambda: attn._lin(x, attn.wqkv, "QKV"))
        y = attn._lin(x, attn.wqkv, "QKV")
        qa = ttnn.slice(y, [0, 0, 0, 0], [1, 1, 4, 1280])
        res["  slice q"] = chain_ms(md, lambda: ttnn.slice(y, [0, 0, 0, 0], [1, 1, 4, 1280]))
        qr = ttnn.rms_norm(qa, weight=attn.q_norm, epsilon=1e-20)
        res["  rms_q"] = chain_ms(md, lambda: ttnn.rms_norm(qa, weight=attn.q_norm, epsilon=1e-20))
        res["  wq_b"] = chain_ms(md, lambda: attn._lin(qr, attn.wq_b, "QB"))
        kvn = ttnn.rms_norm(ttnn.slice(y, [0, 0, 0, 1280], [1, 1, 4, 1792]), weight=attn.kv_norm, epsilon=1e-20)
        qq = attn._lin(qr, attn.wq_b, "QB")
        oq = o
        res["  rope_heads(q)"] = chain_ms(md, lambda: attn._rope_heads(oq, st["Ch"], st["Sh"]))
        cat = ttnn.concat([kvn, qq, kvn, kvn], dim=3)
        res["  concat"] = chain_ms(md, lambda: ttnn.concat([kvn, qq, kvn, kvn], dim=3))
        res["  nlp_create"] = chain_ms(
            md,
            lambda: ttnn.experimental.nlp_create_qkv_heads_decode(
                cat, num_heads=9, num_kv_heads=1, memory_config=attn._ucfg
            ),
        )
        res["  to_dram(q)"] = chain_ms(md, lambda: ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG))
        res["  to_sharded(q)"] = chain_ms(md, lambda: ttnn.to_memory_config(q, attn._ucfg))
    for k, v in res.items():
        print(f"PR ratio={ratio} S={S} {k:20s} {v * 1e3:8.1f} us", flush=True)
