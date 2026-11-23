# DP4 x TP8 prefill, step 2: one Llama-3.3-70B layer built from the tt_dit tensor-parallel stack (Blackhole-tuned fused
# collectives), cluster_axis 0 = the 8 chips of a mesh column, every column a separate prompt (weights replicated
# across columns). Hidden-sharded residual [M, 8192/8] per chip:
#   DistributedRMSNorm -> QKV (all-gather + matmul fused) -> heads + RoPE -> causal SDPA (8 q / 1 kv head per chip)
#   -> WO (matmul + reduce-scatter fused, residual add fused in its epilogue)
#   -> DistributedRMSNorm -> FF1|FF3 (all-gather + matmul + SwiGLU fused) -> FF2 (matmul + reduce-scatter + residual)
# Times the layer with a host loop and with a captured trace. Compare: tests/test_dp4_tp8_prefill_bench.py (same layer
# from unfused ops: 4K 5.17 ms, 16K 23.2 ms); today's TP32 prefill ~2.96 ms per 4K prompt-layer (vLLM 32 x 4K).
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_dp4_tp8_fused_layer.py -s
# env: TP8_M (default 4096,16384), TP8_ITERS (10), TP8_FF13 (lofi | hifi2), TP8_ATTN (hifi2 | lofi)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers.linear import ColParallelLinear, RowParallelLinear
from models.tt_dit.layers.normalization import DistributedRMSNorm
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

H, D, FF, NQ, NKV, TP = 8192, 128, 28672, 64, 8, 8
MS = [int(m) for m in (os.environ.get("TP8_M") or "4096,16384").split(",")]
ITERS = int(os.environ.get("TP8_ITERS") or "10")


def _ckc(mesh, fid):
    f = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}[fid]
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )


class TP8Layer:
    def __init__(self, mesh, ccl):
        self.mesh, self.ccl = mesh, ccl
        self.pc = DiTParallelConfig(
            cfg_parallel=ParallelFactor(1, 1),
            tensor_parallel=ParallelFactor(TP, 0),
            sequence_parallel=ParallelFactor(1, 1),
        )
        kw = dict(mesh_device=mesh, mesh_axis=0, ccl_manager=ccl, bias=False)
        self.n1 = DistributedRMSNorm(H, norm_eps=1e-5, mesh_axis=0, mesh_device=mesh, ccl_manager=ccl)
        self.n2 = DistributedRMSNorm(H, norm_eps=1e-5, mesh_axis=0, mesh_device=mesh, ccl_manager=ccl)
        self.qkv = ColParallelLinear(H, (NQ + 2 * NKV) * D, dtype=ttnn.bfloat8_b, **kw)
        self.wo = RowParallelLinear(NQ * D, H, dtype=ttnn.bfloat8_b, **kw)
        self.ff13 = ColParallelLinear(H, FF, activation_fn="swiglu", dtype=ttnn.bfloat4_b, **kw)
        self.ff2 = RowParallelLinear(FF, H, dtype=ttnn.bfloat8_b, **kw)
        g = torch.Generator().manual_seed(0)
        for m, shp in (
            (self.qkv, ((NQ + 2 * NKV) * D, H)),
            (self.wo, (H, NQ * D)),
            (self.ff13, (2 * FF, H)),
            (self.ff2, (H, FF)),
        ):
            m.load_torch_state_dict({"weight": torch.randn(shp, generator=g) * 0.02})
        for n in (self.n1, self.n2):
            n.load_torch_state_dict({"weight": torch.ones(H)})
        self.ck_attn = _ckc(mesh, os.environ.get("TP8_ATTN") or "hifi2")
        self.ck_ff13 = _ckc(mesh, os.environ.get("TP8_FF13") or "lofi")
        self.ck_ff2 = _ckc(mesh, "hifi2")
        self.ck_sdpa = _ckc(mesh, "hifi2")
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.ones = ttnn.from_torch(
            torch.ones(1, 1, 1, H // TP), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep
        )
        tm = torch.zeros(1, 1, 32, 32)
        tm[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
        tm[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
        self.trans = ttnn.from_torch(tm, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep)
        self.grid = mesh.compute_with_storage_grid_size()

    def rope(self, M):
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        ang = torch.randn(1, 1, M, D)
        return [
            ttnn.from_torch(f(ang), device=self.mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep)
            for f in (torch.cos, torch.sin)
        ]

    def __call__(self, x, cos, sin, sdpa_cfg, parts=None):
        def mark(name):
            if parts is not None:
                ttnn.synchronize_device(self.mesh)
                parts.append((name, time.perf_counter()))

        mark("start")
        h = self.n1(x, dtype=ttnn.bfloat8_b)
        mark("norm1")
        qkv = self.qkv(h, parallel_config=self.pc, compute_kernel_config=self.ck_attn, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        mark("qkv_agmm")
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=NQ // TP,
            num_kv_heads=NKV // TP,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q2 = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self.trans, is_decode_mode=False)
        k2 = ttnn.experimental.rotary_embedding_llama(k, cos, sin, self.trans, is_decode_mode=False)
        ttnn.deallocate(q), ttnn.deallocate(k)
        k8 = ttnn.typecast(k2, ttnn.bfloat8_b)
        v8 = ttnn.typecast(v, ttnn.bfloat8_b)
        ttnn.deallocate(k2), ttnn.deallocate(v)
        mark("heads_rope")
        o = ttnn.transformer.scaled_dot_product_attention(
            q2,
            k8,
            v8,
            is_causal=True,
            program_config=sdpa_cfg,
            compute_kernel_config=self.ck_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q2), ttnn.deallocate(k8), ttnn.deallocate(v8)
        mark("sdpa")
        a = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o)
        a8 = ttnn.typecast(a, ttnn.bfloat8_b)
        ttnn.deallocate(a)
        x1 = self.wo.forward_fused_addcmul(
            a8, x, self.ones, 1.0, compute_kernel_config=self.ck_attn, dtype=ttnn.bfloat16
        )
        ttnn.deallocate(a8)
        mark("wo_mmrs")
        h2 = self.n2(x1, dtype=ttnn.bfloat8_b)
        mark("norm2")
        f = self.ff13(h2, parallel_config=self.pc, compute_kernel_config=self.ck_ff13, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(h2)
        mark("ff13_agmm_swiglu")
        x2 = self.ff2.forward_fused_addcmul(
            f, x1, self.ones, 1.0, compute_kernel_config=self.ck_ff2, dtype=ttnn.bfloat16
        )
        ttnn.deallocate(f)  # x1 / x2 live in the CCL manager's persistent reduce-scatter buffers: never free them
        mark("ff2_mmrs")
        return x2


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 300_000_000, "l1_small_size": 32768}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dp4_tp8_fused_layer(mesh_device, device_params):
    mesh = mesh_device
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Ring)
    layer = TP8Layer(mesh, ccl)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    for M in MS:
        tag = f"TP8F M={M}"
        x = ttnn.from_torch(
            torch.randn(1, 1, M, H // TP), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep
        )
        cos, sin = layer.rope(M)
        best = None
        for qc, kc in ((256, 512), (256, 256)):
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=layer.grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
            )
            try:
                layer(x, cos, sin, cfg)
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    layer(x, cos, sin, cfg)
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - t0) / ITERS * 1e3
            except Exception as e:
                logger.warning(f"{tag} sdpa q{qc}k{kc} ERROR {str(e)[:400]}")
                continue
            logger.info(f"{tag} host loop (sdpa q{qc} k{kc}): {ms:.2f} ms/layer")
            if best is None or ms < best[0]:
                best = (ms, cfg)
        if best is None:
            continue
        cfg = best[1]
        parts = []
        layer(x, cos, sin, cfg, parts)
        logger.info(
            f"{tag} parts (synced, ms): "
            + " ".join(f"{parts[i][0]}={1e3 * (parts[i][1] - parts[i - 1][1]):.2f}" for i in range(1, len(parts)))
        )
        try:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = layer(x, cos, sin, cfg)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            tr = (time.perf_counter() - t0) / ITERS * 1e3
            ttnn.release_trace(mesh, tid)
        except Exception as e:
            tr = float("nan")
            logger.warning(f"{tag} trace ERROR {str(e)[:400]}")
        lay = min(best[0], tr) if tr == tr else best[0]
        P = lay * 80 / 1e3
        logger.info(
            f"{tag} SUMMARY layer {best[0]:.2f} ms host / {tr:.2f} ms traced | per prompt per column {P:.2f} s "
            f"| 32 at once on one Galaxy (8 per column FIFO): mean {4.5 * P:.2f} / max {8 * P:.2f} s"
        )
        ttnn.deallocate(x), ttnn.deallocate(cos), ttnn.deallocate(sin)
