# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hammer the Q-norm op of Molmo2 attention prefill in isolation to separate a model/hardware problem from a kernel one.

    python models/demos/molmo2/tests/rmsnorm_loop.py [iterations] [seq_len] [heads] [fp32_dest_acc 1/0]
"""
import sys
import time

import torch
from loguru import logger

import ttnn

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
seq = int(sys.argv[2]) if len(sys.argv) > 2 else 25344
heads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
FP32_ACC = (sys.argv[4] != "0") if len(sys.argv) > 4 else True

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
rep = ttnn.ReplicateTensorToMesh(mesh)
torch.manual_seed(0)
x = ttnn.from_torch(
    torch.randn(1, heads, seq, 128),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=rep,
)
w = ttnn.from_torch(
    torch.randn(1, 1, 128 // 32, 32),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=rep,
)
# Same compute config as models/common/rmsnorm.py RMSNorm (used by Molmo2 attention q_norm/k_norm).
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=FP32_ACC,
    packer_l1_acc=True,
)
logger.info(f"rms_norm x={tuple(x.shape)} loops={iters}")
t = time.time()
for i in range(iters):
    y = ttnn.rms_norm(x, weight=w, epsilon=1e-6, compute_kernel_config=cfg)
    ttnn.deallocate(y)
    if i % 100 == 99:
        ttnn.synchronize_device(mesh)
        logger.info(f"ITER {i + 1} ok {time.time() - t:.1f}s")
ttnn.synchronize_device(mesh)
ttnn.close_mesh_device(mesh)
logger.info("LOOP DONE")
