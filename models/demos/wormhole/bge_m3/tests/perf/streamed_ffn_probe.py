# SPDX-License-Identifier: Apache-2.0
"""Streamed / L1-intermediate FFN probe (precision-preserving DRAM-traffic lever).

Per-ASIC MLP (DP=2): Wi [M=49152,K=1024,N=4096]+GELU -> Wo [M=49152,K=4096,N=1024].
The [49152,4096] bf8 intermediate (~214MB) round-trips DRAM (Wi writes, Wo reads).
The bf4 probe proved halving that traffic = -11ms wall. Keeping the intermediate
FULLY in L1 (bf8, PCC preserved) should eliminate the whole round-trip.

Full-size L1 OOMs, so CHUNK M: for each M-chunk run Wi->L1 then Wo(L1)->DRAM, so
the intermediate never hits DRAM. Sweep chunk count; compare wall + PCC vs the
baseline two-full-matmul (DRAM intermediate).
"""
import time

import pytest
import torch
from loguru import logger

import ttnn

M, K, N = 49152, 1024, 4096  # per-ASIC Wi shape; Wo is [M,4096]->[M,1024]
N_ITERS = 15


def _wi_cfg():
    return ttnn.MinimalMatmulConfig(
        M_block_size=16,
        K_block_size=16,
        N_block_size=8,
        subblock_h=4,
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    )


def _wo_cfg():
    return ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=32,
        N_block_size=4,
        subblock_h=4,
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 90_000_000, "num_command_queues": 1}], indirect=True)
def test_streamed_ffn(mesh_device):
    torch.manual_seed(0)
    x = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.1
    wi = torch.randn(1, 1, K, 4096, dtype=torch.bfloat16) * 0.05
    wo = torch.randn(1, 1, 4096, N // (N // 1024), dtype=torch.bfloat16) * 0.05  # N_wo=1024
    wo = torch.randn(1, 1, 4096, 1024, dtype=torch.bfloat16) * 0.05

    dev = mesh_device
    mk = lambda t, dt=ttnn.bfloat8_b: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tx, twi, two = mk(x), mk(wi), mk(wo)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    ref = torch.nn.functional.gelu(x.float() @ wi.float()) @ wo.float()

    def pcc(t):
        got = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].float()
        return torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()

    def bench(name, body):
        out = body()
        ttnn.synchronize_device(dev)
        p = pcc(out)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = body()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.release_trace(dev, tid)
        ts.sort()
        logger.info(f"{name:<30} wall min={ts[0]:.3f} med={ts[len(ts)//2]:.3f} ms  pcc={p:.5f}")

    # BASELINE: two full matmuls, DRAM intermediate
    def baseline():
        act = ttnn.experimental.minimal_matmul(
            input_tensor=tx,
            weight_tensor=twi,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
            config=_wi_cfg(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ckc,
        )
        o = ttnn.experimental.minimal_matmul(
            input_tensor=act,
            weight_tensor=two,
            config=_wo_cfg(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ckc,
        )
        ttnn.deallocate(act)
        return o

    bench("baseline_bf8_intermediate", baseline)

    # bf4 intermediate: reproduce the model's -11ms lever in isolation.
    def bf4_inter():
        act = ttnn.experimental.minimal_matmul(
            input_tensor=tx,
            weight_tensor=twi,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
            config=_wi_cfg(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat4_b,
            compute_kernel_config=ckc,
        )
        two_bf4 = ttnn.typecast(two, ttnn.bfloat4_b)
        o = ttnn.experimental.minimal_matmul(
            input_tensor=act,
            weight_tensor=two_bf4,
            config=_wo_cfg(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ckc,
        )
        ttnn.deallocate(act)
        ttnn.deallocate(two_bf4)
        return o

    try:
        bench("bf4_intermediate", bf4_inter)
    except Exception as e:
        logger.error(f"bf4_intermediate FAILED: {str(e)[:160]}")

    # CHUNKED L1: split M; per chunk Wi->L1, Wo(L1)->DRAM. Intermediate never hits DRAM.
    # Sweep smaller Wo CB configs to leave L1 headroom for the intermediate tensor.
    wo_variants = {
        "wo_k32n4": ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=32,
            N_block_size=4,
            subblock_h=4,
            subblock_w=2,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        ),
        "wo_k16n4": ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=16,
            N_block_size=4,
            subblock_h=4,
            subblock_w=2,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        ),
        "wo_k8n4": ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=4,
            subblock_h=4,
            subblock_w=2,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        ),
        "wo_k8n2": ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=2,
            subblock_h=4,
            subblock_w=2,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        ),
    }
    for nchunks in (16, 32):
        if M % nchunks != 0 or (M // nchunks) % 32 != 0:
            continue
        cm = M // nchunks
        for wname, wcfg in wo_variants.items():

            def chunked(cm=cm, nchunks=nchunks, wcfg=wcfg):
                outs = []
                for i in range(nchunks):
                    xc = ttnn.slice(tx, [0, 0, i * cm, 0], [1, 1, (i + 1) * cm, K])
                    act = ttnn.experimental.minimal_matmul(
                        input_tensor=xc,
                        weight_tensor=twi,
                        fused_activation=(ttnn.UnaryOpType.GELU, True),
                        config=_wi_cfg(),
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat8_b,
                        compute_kernel_config=ckc,
                    )
                    oc = ttnn.experimental.minimal_matmul(
                        input_tensor=act,
                        weight_tensor=two,
                        config=wcfg,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat8_b,
                        compute_kernel_config=ckc,
                    )
                    ttnn.deallocate(act)
                    ttnn.deallocate(xc)
                    outs.append(oc)
                o = ttnn.concat(outs, dim=2)
                for oc in outs:
                    ttnn.deallocate(oc)
                return o

            try:
                bench(f"chunked_L1_n{nchunks}_{wname}", chunked)
            except Exception as e:
                logger.error(f"chunked_L1_n{nchunks}_{wname} FAILED: {str(e)[:120]}")
