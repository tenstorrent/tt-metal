# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi K2.6 / GLM 5.1 5K device-profiler sweep for the fused routed-expert block.

Run with ``scripts/run_safe_pytest.sh --profile``.  The profiler CSV preserves
one row per dispatch, ordered by the ``COUNTS``/``REPS`` loops below.
"""

import os

import pytest
import torch
import ttnn

from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

TILE = 32
EXPERTS_PER_TOKEN = 8
SEQ_LEN_PER_CHIP = 640
CAPACITY = EXPERTS_PER_TOKEN * SEQ_LEN_PER_CHIP
GRID = ttnn.CoreCoord(11, 8)
COUNTS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
REPS = int(os.environ.get("MOE_FUSED_SWIGLU_PERF_REPS", "7"))
# Kimi K2.6 can route all 8 tokens from each of its 640 tokens on this chip to
# one expert, so a 5K dispatch group requires a 5120-row expert region.
GLOBAL_EXPERT_ID = 137
LOCAL_EXPERT_ID = 3
MODELS = (KimiK26Config, GLM51Config)


def to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("model_config", MODELS, ids=["kimi-k26", "glm-51"])
@pytest.mark.parametrize(
    "input_dtype,input_layout",
    [(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)],
    ids=["row-major-bf16", "tile-bf8"],
)
@pytest.mark.parametrize(
    "weight_memory_layout",
    [ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.ND_SHARDED],
    ids=["interleaved", "nd-sharded"],
)
def test_5k_cpp_binding_perf_matrix(device, model_config, input_dtype, input_layout, weight_memory_layout):
    torch.manual_seed(42)
    k = model_config.EMB_SIZE
    n = model_config.MOE_INTERMEDIATE_SIZE
    num_routed_experts = model_config.NUM_ROUTED_EXPERTS
    x = to_device(
        torch.randn((1, 1, CAPACITY, k), dtype=torch.bfloat16),
        input_dtype,
        input_layout,
        device,
    )
    if weight_memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED:
        gate_up_memory_config, down_memory_config = weight_memory_configs(device, k, n, core_grid=GRID)
    else:
        gate_up_memory_config = down_memory_config = ttnn.DRAM_MEMORY_CONFIG
    weights = [
        to_device(host, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device, memory_config)
        for host, memory_config in zip(
            (
                torch.randn((k, n), dtype=torch.bfloat16),
                torch.randn((k, n), dtype=torch.bfloat16),
                torch.randn((n, k), dtype=torch.bfloat16),
            ),
            (gate_up_memory_config, gate_up_memory_config, down_memory_config),
        )
    ]
    ids = torch.tensor([(11 + 37 * local) % num_routed_experts for local in range(8)], dtype=torch.int32)
    ids[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    ids = to_device(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    count_tensors = {}
    for count in COUNTS:
        host = torch.zeros(num_routed_experts, dtype=torch.int32)
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
