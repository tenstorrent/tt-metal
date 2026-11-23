# DP4 x TP8 prefill feasibility on the 8x4 BH Galaxy: each mesh column (8 chips, cluster_axis 0) prefills ITS OWN
# lane's prompt with Megatron-style TP8, so 4 prompts run at once (SPMD: same program on every column, different data).
# Per chip: 8 q heads + 1 kv head (= the decode KV layout: row r holds kv head r for lane c, so the KV needs no move),
# full-K matmuls, collectives only along the column. Today's TP32 prefill splits one prompt over all 32 chips and
# spends ~1.7 of ~3.5 ms per 4K layer in all-gather / reduce-scatter (prof_cpf), with matmuls ~0.95 ms.
# Times one Llama-3.3-70B layer per chip, ops one after another (no fusion, no trace) and each piece alone:
#   AG(x_sp)->QKV, SDPA(8q/1kv, causal), WO->RS, AG->FF1,FF3,mul, FF2->RS   (norms, RoPE, residual adds left out)
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_dp4_tp8_prefill_bench.py -s
# env: TP8_M (comma list, default 4096,8192,16384), TP8_LINKS (2), TP8_WORKERS (4), TP8_ITERS (5)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, D, FF = 8192, 128, 28672
TP = 8
NQ, NKV = 64 // TP, 8 // TP  # per chip: 8 q heads, 1 kv head
MS = [int(m) for m in (os.environ.get("TP8_M") or "4096,8192,16384").split(",")]
LINKS = int(os.environ.get("TP8_LINKS", "2"))
WORKERS = int(os.environ.get("TP8_WORKERS", "4"))
ITERS = int(os.environ.get("TP8_ITERS", "5"))


def _run(fn):
    out = fn()
    for o in out if isinstance(out, (list, tuple)) else [out]:
        ttnn.deallocate(o)


def _timed(mesh, fn):
    _run(fn)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _run(fn)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / ITERS * 1e3


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dp4_tp8_prefill_bench(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    dram = ttnn.DRAM_MEMORY_CONFIG
    rep = ttnn.ReplicateTensorToMesh(mesh)
    ckc = {
        f: ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        for f in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi)
    }
    hifi2, lofi = ckc[ttnn.MathFidelity.HiFi2], ckc[ttnn.MathFidelity.LoFi]

    def t(shape, dt, scale=1.0):
        return ttnn.from_torch(
            torch.randn(shape) * scale,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dt,
            memory_config=dram,
            mesh_mapper=rep,
        )

    wqkv = t((1, 1, H, (NQ + 2 * NKV) * D), ttnn.bfloat8_b, 0.02)
    wo = t((1, 1, NQ * D, H), ttnn.bfloat8_b, 0.02)
    w1, w3 = t((1, 1, H, FF // TP), ttnn.bfloat4_b, 0.02), t((1, 1, H, FF // TP), ttnn.bfloat4_b, 0.02)
    w2 = t((1, 1, FF // TP, H), ttnn.bfloat8_b, 0.02)

    def mm(a, b, k):
        # best of ttnn.linear and minimal_matmul is picked per op below; this is the minimal_matmul form
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )
        return ttnn.experimental.minimal_matmul(
            input_tensor=a, weight_tensor=b, config=cfg, compute_kernel_config=k, memory_config=dram
        )

    def lin(a, b, k):
        return ttnn.linear(
            a,
            b,
            compute_kernel_config=k,
            dtype=ttnn.bfloat8_b,
            memory_config=dram,
            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        )

    def ag(x):
        return ttnn.all_gather(x, 2, cluster_axis=0, topology=ttnn.Topology.Ring, num_links=LINKS, memory_config=dram)

    def rs(x):
        return ttnn.reduce_scatter(
            x,
            2,
            cluster_axis=0,
            topology=ttnn.Topology.Ring,
            num_links=LINKS,
            num_workers_per_link=WORKERS,
            memory_config=dram,
        )

    for M in MS:
        x_sp = t((1, 1, M // TP, H), ttnn.bfloat8_b)
        x = t((1, 1, M, H), ttnn.bfloat8_b)
        attn_flat = t((1, 1, M, NQ * D), ttnn.bfloat8_b)
        h = t((1, 1, M, FF // TP), ttnn.bfloat8_b)
        part = t((1, 1, M, H), ttnn.bfloat8_b)
        q = t((1, NQ, M, D), ttnn.bfloat16)
        k = t((1, NKV, M, D), ttnn.bfloat8_b)
        v = t((1, NKV, M, D), ttnn.bfloat8_b)
        tag = f"TP8 M={M}"
        res = {}

        # matmuls: pick the faster of linear / minimal_matmul per op
        pick = {}
        for name, (a, b, kk) in {
            "qkv": (x, wqkv, hifi2),
            "wo": (attn_flat, wo, hifi2),
            "ff1": (x, w1, lofi),
            "ff3": (x, w3, lofi),
            "ff2": (h, w2, hifi2),
        }.items():
            best = None
            for label, f in (("minimal", mm), ("linear", lin)):
                try:
                    ms = _timed(mesh, lambda: f(a, b, kk))
                except Exception as e:
                    logger.warning(f"{tag} {name} {label} ERROR {str(e)[:200]}")
                    continue
                logger.info(f"{tag} {name} {label}: {ms:.3f} ms")
                if best is None or ms < best[0]:
                    best = (ms, f)
            res[name], pick[name] = best[0], best[1]

        sdpa_best = None
        for qc, kc in ((128, 256), (256, 256), (128, 512), (256, 512)):
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
            )
            try:
                ms = _timed(
                    mesh,
                    lambda: ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, is_causal=True, program_config=cfg, compute_kernel_config=hifi2, memory_config=dram
                    ),
                )
            except Exception as e:
                logger.warning(f"{tag} sdpa q{qc} k{kc} ERROR {str(e)[:200]}")
                continue
            logger.info(f"{tag} sdpa q{qc} k{kc}: {ms:.3f} ms")
            if sdpa_best is None or ms < sdpa_best[0]:
                sdpa_best = (ms, cfg)
        res["sdpa"] = sdpa_best[0]

        for name, fn in (("ag", lambda: ag(x_sp)), ("rs", lambda: rs(part))):
            try:
                res[name] = _timed(mesh, fn)
                logger.info(f"{tag} {name} (links={LINKS}): {res[name]:.3f} ms")
            except Exception as e:
                logger.warning(f"{tag} {name} ERROR {str(e)[:300]}")
                res[name] = float("nan")

        def layer():
            outs = []
            xg = ag(x_sp)
            qkv = pick["qkv"](xg, wqkv, hifi2)
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, program_config=sdpa_best[1], compute_kernel_config=hifi2, memory_config=dram
            )
            p1 = pick["wo"](attn_flat, wo, hifi2)
            r1 = rs(p1)
            xg2 = ag(r1)
            f1, f3 = pick["ff1"](xg2, w1, lofi), pick["ff3"](xg2, w3, lofi)
            hh = ttnn.mul(f1, f3, memory_config=dram)
            p2 = pick["ff2"](hh, w2, hifi2)
            r2 = rs(p2)
            for z in (xg, qkv, o, p1, r1, xg2, f1, f3, hh, p2):
                ttnn.deallocate(z)
            outs.append(r2)
            return outs

        try:
            res["layer"] = _timed(mesh, layer)
        except Exception as e:
            logger.warning(f"{tag} layer ERROR {str(e)[:300]}")
            res["layer"] = float("nan")
        mm_sum = sum(res[n] for n in ("qkv", "wo", "ff1", "ff3", "ff2"))
        ccl_sum = 2 * res["ag"] + 2 * res["rs"]
        prompt_s = res["layer"] * 80 / 1e3  # one prompt on one column
        ideal_s = max(mm_sum + res["sdpa"], ccl_sum) * 80 / 1e3  # if the collectives hid under compute
        logger.info(
            f"{tag} SUMMARY matmuls {mm_sum:.2f} sdpa {res['sdpa']:.2f} ccl(2AG+2RS) {ccl_sum:.2f} "
            f"serial layer {res['layer']:.2f} ms | per prompt per column {prompt_s:.2f} s serial, {ideal_s:.2f} s overlapped "
            f"| 32 at once (8 per column, FIFO): mean {4.5 * prompt_s:.2f} / max {8 * prompt_s:.2f} s serial, "
            f"mean {4.5 * ideal_s:.2f} / max {8 * ideal_s:.2f} s overlapped"
        )
        for z in (x_sp, x, attn_flat, h, part, q, k, v):
            ttnn.deallocate(z)
