# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi K2.6 5K device-profiler sweep for the fused routed-expert block.

Run with ``scripts/run_safe_pytest.sh --profile``.  The profiler CSV preserves
one row per dispatch, ordered by the ``COUNTS``/``REPS`` loops below.
"""

import os

import pytest
import torch
import ttnn

from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs


TILE = 32
K = KimiK26Config.EMB_SIZE
N = KimiK26Config.MOE_INTERMEDIATE_SIZE
KIMI_EXPERTS_PER_TOKEN = 8
KIMI_5K_SEQ_LEN_PER_CHIP = 640
CAPACITY = KIMI_EXPERTS_PER_TOKEN * KIMI_5K_SEQ_LEN_PER_CHIP
GRID = ttnn.CoreCoord(11, 8)
COUNTS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
REPS = int(os.environ.get("KIMI_K26_5K_PERF_REPS", "7"))
# Kimi K2.6 can route all 8 tokens from each of its 640 tokens on this chip to
# one expert, so a 5K dispatch group requires a 5120-row expert region.
NUM_ROUTED_EXPERTS = KimiK26Config.NUM_ROUTED_EXPERTS
GLOBAL_EXPERT_ID = 137
LOCAL_EXPERT_ID = 3


def to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_kimi_k26_5k_cpp_binding_perf_matrix(device):
    torch.manual_seed(42)
    x = to_device(
        torch.randn((1, 1, CAPACITY, K), dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )
    gate_up_memory_config, down_memory_config = weight_memory_configs(device, K, N, core_grid=GRID)
    weights = [
        to_device(host, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device, memory_config)
        for host, memory_config in zip(
            (
                torch.randn((K, N), dtype=torch.bfloat16),
                torch.randn((K, N), dtype=torch.bfloat16),
                torch.randn((N, K), dtype=torch.bfloat16),
            ),
            (gate_up_memory_config, gate_up_memory_config, down_memory_config),
        )
    ]
    ids = torch.tensor([(11 + 37 * local) % NUM_ROUTED_EXPERTS for local in range(8)], dtype=torch.int32)
    ids[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    ids = to_device(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    count_tensors = {}
    for count in COUNTS:
        host = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.int32)
        host[GLOBAL_EXPERT_ID] = count
        count_tensors[count] = to_device(host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    op = ttnn.experimental.deepseek_prefill.moe_fused_swiglu

    # Compile/cache warmup is deliberately excluded from the reported matrix.
    op(
        x,
        *weights,
        count_tensors[512],
        ids,
        LOCAL_EXPERT_ID,
        input_m_tiles=CAPACITY // TILE,
        compute_kernel_config=config,
        core_grid=GRID,
    )
    ttnn.ReadDeviceProfiler(device)

    for count in COUNTS:
        for _ in range(REPS):
            op(
                x,
                *weights,
                count_tensors[count],
                ids,
                LOCAL_EXPERT_ID,
                input_m_tiles=CAPACITY // TILE,
                compute_kernel_config=config,
                core_grid=GRID,
            )
            ttnn.ReadDeviceProfiler(device)
