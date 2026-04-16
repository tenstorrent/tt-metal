# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: MatmulMultiCoreReuse with FUSE_BIAS.

Exercises the fused-bias code path in
``bmm_large_block_zm_fused_bias_activation.cpp`` under the
``MatmulMultiCoreReuseOptimizedProgramFactory``. Bias is a 1xN row vector.

FUSE_BIAS is the heaviest of the phase-1 kernel changes: the fused kernel was
rewritten from 422 to 298 lines and is the primary caller of both the unified
``matmul_block`` helper and the ``bias_add_helpers`` helper (see
``project_matmul_helpers_phase1_context.md``).

Run (from the repo root, after ``./build_metal.sh``):

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/reuse_fuse_bias.py
"""

from __future__ import annotations

import os
import sys

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _perf_harness import (  # noqa: E402
    DEFAULT_MEASURE_ITERS,
    DEFAULT_WARMUP_ITERS,
    HarnessConfig,
    pick_compute_kernel_config,
    run_warmup_and_measure,
)


SCRIPT_LABEL = "reuse_fuse_bias"


def build_inputs(device):
    torch.manual_seed(0)
    m_size, k_size, n_size = 512, 512, 512
    grid_y, grid_x = 1, 1

    in0_shape = [1, 1, m_size, k_size]
    in1_shape = [1, 1, k_size, n_size]
    bias_shape = [1, 1, 1, n_size]

    torch_a = torch.randn(in0_shape).to(torch.bfloat16)
    torch_b = torch.randn(in1_shape).to(torch.bfloat16)
    torch_bias = torch.randn(bias_shape).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m_size // 32,
        per_core_N=n_size // 32,
    )
    return a, b, bias, program_config


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        a, b, bias, program_config = build_inputs(device)
        compute_config = pick_compute_kernel_config()

        def run_once():
            # ttnn.linear dispatches to matmul with bias fused.
            return ttnn.linear(
                a,
                b,
                bias=bias,
                program_config=program_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=compute_config,
            )

        run_warmup_and_measure(
            run_once,
            device=device,
            config=HarnessConfig(
                warmup_iters=DEFAULT_WARMUP_ITERS,
                measure_iters=DEFAULT_MEASURE_ITERS,
            ),
            label=SCRIPT_LABEL,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
