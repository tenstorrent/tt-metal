# SPDX-License-Identifier: Apache-2.0
"""Decompose SDPA cost: matmul-only (QK^T + PV) vs full SDPA at the exact shape.
The delta bounds the softmax/reduction/correction/format overhead — answers
whether 30.59ms/op is matmul-bound or softmax-bound WITHOUT editing C++ kernels.

Per SDPA op the real kernel does, per (batch,head): [Sq,DH]x[DH,Sk] -> [Sq,Sk]
scores, softmax over Sk, then [Sq,Sk]x[Sk,DH] -> [Sq,DH]. We emulate just the
two matmuls at the same total FLOP to get an empirical matmul lower bound.
Device-kernel time via signposts (10 iters, 2 warmup dropped).
"""
import os

import pytest
import torch
from loguru import logger

import ttnn

try:
    from tracy import signpost
except ImportError:

    def signpost(*a, **k):
        return None


# Full SDPA logical shape (matches #215 in-model): B6 HQ32 HKV16 Sq4096 Sk8192 D64
B, HQ, Sq, Sk, DH = 6, 32, 4096, 8192, 64
N = 10


@pytest.mark.parametrize("device_params", [{"trace_region_size": 10_000_000, "num_command_queues": 1}], indirect=True)
def test_sdpa_decomp(mesh_device):
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("TT_METAL_DEVICE_PROFILER=1 required")
    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True
    )
    dev = mesh_device
    BF8 = ttnn.bfloat8_b

    def t(shape):
        return ttnn.from_torch(
            torch.randn(*shape, dtype=torch.bfloat16),
            dtype=BF8,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Fold (B,H) into batch dim for plain matmul: M=Sq, K=DH, N=Sk (QK^T);
    # then M=Sq, K=Sk, N=DH (PV). Use B*HQ = 192 as leading batch.
    BH = B * HQ
    q = t((BH, Sq, DH))
    kt = t((BH, DH, Sk))  # K already transposed to [DH,Sk]
    scores = t((BH, Sq, Sk))
    v = t((BH, Sk, DH))

    def qk():
        return ttnn.matmul(q, kt, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def pv():
        return ttnn.matmul(scores, v, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    for name, fn in [("qk_matmul", qk), ("pv_matmul", pv)]:
        try:
            for _ in range(2):
                o = fn()
                ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            signpost(name)
            for _ in range(N):
                o = fn()
                ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            logger.info(f"OK {name}")
        except Exception as e:
            logger.error(f"FAIL {name}: {str(e)[:100]}")
    signpost("end")
