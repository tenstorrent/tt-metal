# DP4 x TP8 prefill, step 3: sequence-parallel (SP) TP8 layer on a mesh column (cluster_axis 0, 8 chips), every
# column a separate prompt. Residual [M/8, 8192] bf16 per chip (this chip's rows), all traffic in bf8:
#   local RMSNorm -> all-gather rows -> QKV -> heads + RoPE -> causal SDPA (8 q / 1 kv head per chip) -> concat
#   -> WO + reduce-scatter rows (ttnn.experimental.matmul_reduce_scatter_prefill, fp32 adds per hop) -> + residual
#   -> local RMSNorm -> all-gather rows -> FF1|FF3 (one packed matmul) -> silu*mul -> FF2 + reduce-scatter -> + residual
# Prints per-part times (synced), the full layer (host loop, traced), and an all-gather link/worker sweep.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_dp4_tp8_sp_layer.py -s
# env: TP8_M (4096,16384), TP8_ITERS (10), TP8_FID (mixed: QKV/WO/FF2 HiFi2 + FF13 LoFi | lofi: all LoFi)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, D, FF, NQ, NKV, TP = 8192, 128, 28672, 64, 8, 8
MS = [int(m) for m in (os.environ.get("TP8_M") or "4096,16384").split(",")]
ITERS = int(os.environ.get("TP8_ITERS") or "10")
FID = os.environ.get("TP8_FID") or "mixed"
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _ckc(mesh, f):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 300_000_000, "l1_small_size": 32768}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dp4_tp8_sp_layer(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    hifi2 = _ckc(mesh, ttnn.MathFidelity.HiFi2)
    lofi = _ckc(mesh, ttnn.MathFidelity.LoFi)
    ck_attn = lofi if FID == "lofi" else hifi2
    ck_ff2 = lofi if FID == "lofi" else hifi2
    mmrs_attn = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi if FID == "lofi" else ttnn.MathFidelity.HiFi2
    )
    mmrs_ff2 = mmrs_attn

    def t(shape, dt, scale=1.0):
        return ttnn.from_torch(
            torch.randn(shape) * scale,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dt,
            memory_config=DRAM,
            mesh_mapper=rep,
        )

    wqkv = t((1, 1, H, (NQ + 2 * NKV) * D // TP), ttnn.bfloat8_b, 0.02)
    wo = t((NQ * D // TP, H), ttnn.bfloat8_b, 0.02)  # 2D for matmul_reduce_scatter_prefill
    w13 = t((1, 1, H, 2 * FF // TP), ttnn.bfloat4_b, 0.02)
    w2 = t((FF // TP, H), ttnn.bfloat8_b, 0.02)
    nw = ttnn.from_torch(
        torch.ones(1, 1, H // 32, 32), device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep
    )
    tm = torch.zeros(1, 1, 32, 32)
    tm[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
    tm[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
    trans = ttnn.from_torch(tm, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep)

    def mmcfg(M, K, N):
        return ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )

    def mm(a, w, ck, dtype=ttnn.bfloat8_b):
        return ttnn.experimental.minimal_matmul(
            input_tensor=a,
            weight_tensor=w,
            config=mmcfg(0, 0, 0),
            compute_kernel_config=ck,
            memory_config=DRAM,
            dtype=dtype,
        )

    def ag(x, links=2, **kw):
        return ttnn.all_gather(
            x, 2, cluster_axis=0, topology=ttnn.Topology.Ring, num_links=links, memory_config=DRAM, **kw
        )

    for M in MS:
        tag = f"TP8SP M={M} fid={FID}"
        x = t((1, 1, M // TP, H), ttnn.bfloat16)
        ang = torch.randn(1, 1, M, D)
        cos = ttnn.from_torch(
            torch.cos(ang), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep
        )
        sin = ttnn.from_torch(
            torch.sin(ang), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep
        )
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=256, exp_approx_mode=False
        )

        def layer(x, parts=None):
            def mark(name):
                if parts is not None:
                    ttnn.synchronize_device(mesh)
                    parts.append((name, time.perf_counter()))

            mark("start")
            h = ttnn.rms_norm(x, epsilon=1e-5, weight=nw, memory_config=DRAM)
            h8 = ttnn.typecast(h, ttnn.bfloat8_b)
            ttnn.deallocate(h)
            mark("norm1")
            hg = ag(h8)
            ttnn.deallocate(h8)
            mark("ag1")
            qkv = mm(hg, wqkv, ck_attn, ttnn.bfloat16)
            ttnn.deallocate(hg)
            mark("qkv")
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=NQ // TP, num_kv_heads=NKV // TP, transpose_k_heads=False, memory_config=DRAM
            )
            ttnn.deallocate(qkv)
            q2 = ttnn.experimental.rotary_embedding_llama(q, cos, sin, trans, is_decode_mode=False)
            k2 = ttnn.experimental.rotary_embedding_llama(k, cos, sin, trans, is_decode_mode=False)
            ttnn.deallocate(q), ttnn.deallocate(k)
            k8, v8 = ttnn.typecast(k2, ttnn.bfloat8_b), ttnn.typecast(v, ttnn.bfloat8_b)
            ttnn.deallocate(k2), ttnn.deallocate(v)
            mark("heads_rope")
            o = ttnn.transformer.scaled_dot_product_attention(
                q2, k8, v8, is_causal=True, program_config=sdpa_cfg, compute_kernel_config=hifi2, memory_config=DRAM
            )
            ttnn.deallocate(q2), ttnn.deallocate(k8), ttnn.deallocate(v8)
            mark("sdpa")
            a = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)
            ttnn.deallocate(o)
            a8 = ttnn.typecast(a, ttnn.bfloat8_b)
            ttnn.deallocate(a)
            mark("concat")
            r1 = ttnn.experimental.matmul_reduce_scatter_prefill(
                a8,
                wo,
                cluster_axis=0,
                scatter_dim=-2,
                topology=ttnn.Topology.Ring,
                num_links=2,
                compute_kernel_config=mmrs_attn,
                memory_config=DRAM,
            )
            ttnn.deallocate(a8)
            mark("wo_mmrs")
            x1 = ttnn.add(x, r1, dtype=ttnn.bfloat16, memory_config=DRAM)
            ttnn.deallocate(r1)
            h = ttnn.rms_norm(x1, epsilon=1e-5, weight=nw, memory_config=DRAM)
            h8 = ttnn.typecast(h, ttnn.bfloat8_b)
            ttnn.deallocate(h)
            mark("add_norm2")
            hg = ag(h8)
            ttnn.deallocate(h8)
            mark("ag2")
            f13 = mm(hg, w13, lofi)
            ttnn.deallocate(hg)
            mark("ff13")
            f1, f3 = ttnn.split(f13, FF // TP, dim=-1)
            ttnn.deallocate(f13)
            f = ttnn.mul(
                f1, f3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat8_b, memory_config=DRAM
            )
            ttnn.deallocate(f1), ttnn.deallocate(f3)
            mark("swiglu")
            r2 = ttnn.experimental.matmul_reduce_scatter_prefill(
                f,
                w2,
                cluster_axis=0,
                scatter_dim=-2,
                topology=ttnn.Topology.Ring,
                num_links=2,
                compute_kernel_config=mmrs_ff2,
                memory_config=DRAM,
            )
            ttnn.deallocate(f)
            mark("ff2_mmrs")
            x2 = ttnn.add(x1, r2, dtype=ttnn.bfloat16, memory_config=DRAM)
            ttnn.deallocate(r2), ttnn.deallocate(x1)
            mark("add2")
            return x2

        try:
            ttnn.deallocate(layer(x))
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                ttnn.deallocate(layer(x))
            ttnn.synchronize_device(mesh)
            host = (time.perf_counter() - t0) / ITERS * 1e3
            parts = []
            ttnn.deallocate(layer(x, parts))
            logger.info(
                f"{tag} parts (synced, ms): "
                + " ".join(f"{parts[i][0]}={1e3 * (parts[i][1] - parts[i - 1][1]):.2f}" for i in range(1, len(parts)))
            )
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = layer(x)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            tr = (time.perf_counter() - t0) / ITERS * 1e3
            ttnn.release_trace(mesh, tid)
            ttnn.deallocate(out)
            P = tr * 80 / 1e3
            logger.info(
                f"{tag} SUMMARY layer {host:.2f} ms host / {tr:.2f} ms traced | per prompt per column {P:.2f} s | "
                f"32 at once on one Galaxy (8 per column FIFO): mean {4.5 * P:.2f} / max {8 * P:.2f} s"
            )
        except Exception as e:
            logger.warning(f"{tag} layer ERROR {str(e)[:600]}")

        # all-gather bandwidth sweep on the column axis (bf8 rows, the AG1/AG2 payload)
        h8 = t((1, 1, M // TP, H), ttnn.bfloat8_b)
        nbytes = 7 / 8 * M * H * 1088 / 1024
        for links in (1, 2):
            for kw in ({}, {"num_workers_per_link": 2}, {"num_workers_per_link": 4}, {"num_workers_per_link": 8}):
                try:
                    ttnn.deallocate(ag(h8, links, **kw))
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(ITERS):
                        ttnn.deallocate(ag(h8, links, **kw))
                    ttnn.synchronize_device(mesh)
                    ms = (time.perf_counter() - t0) / ITERS * 1e3
                    logger.info(
                        f"{tag} AG links={links} {kw}: {ms:.3f} ms = {nbytes / ms / 1e6:.0f} GB/s received per chip"
                    )
                except Exception as e:
                    logger.warning(f"{tag} AG links={links} {kw} ERROR {str(e)[:200]}")
        ttnn.deallocate(h8)
        for z in (x, cos, sin):
            ttnn.deallocate(z)
