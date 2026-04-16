# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: MatmulMultiCoreReuse (non-bias) via non-multicast sharded bmm.

Shape mirrors the ``test_sharded_matmul[bmm]`` case that regressed and was
fixed by the ``#ifdef ROW_MAJOR_OUTPUT`` gate in
``reader_writer_bmm_tile_layout_in1.cpp`` on this branch (see
``project_matmul_helpers_phase1_context.md``).

- Factory: MatmulMultiCoreReuseOptimizedProgramFactory (entered via
  ``MatmulMultiCoreReuseProgramConfig`` on the public API).
- Batched matmul: a[B0, B1, M, K] @ b[B0, B1, K, N], no bcast_batch.
- Both inputs height-sharded, output height-sharded.

Run (from the repo root, after ``./build_metal.sh``):

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/reuse_bmm_sharded.py
"""

from __future__ import annotations

import os
import sys

import torch
import ttnn

# Allow ``import _perf_harness`` when script is run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _perf_harness import (  # noqa: E402
    DEFAULT_MEASURE_ITERS,
    DEFAULT_WARMUP_ITERS,
    HarnessConfig,
    pick_compute_kernel_config,
    run_warmup_and_measure,
)


SCRIPT_LABEL = "reuse_bmm_sharded"


def build_inputs(device):
    torch.manual_seed(0)

    # 7x7 batch, M=K=N multiples of 32, designed to hit the non-multicast
    # sharded bmm path. Grid is (7, 7) height-sharded, per-core shard is
    # (m_size, k_size) on input a and (k_size, n_size) on input b.
    b0, b1 = 7, 7
    m_size, k_size, n_size = 384, 64, 384

    grid_y, grid_x = 7, 7
    in0_shape = [b0, b1, m_size, k_size]
    in1_shape = [b0, b1, k_size, n_size]

    torch_a = torch.randn(in0_shape).to(torch.bfloat16)
    torch_b = torch.randn(in1_shape).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT)

    a = ttnn.to_device(a, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.to_device(b, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Match test_sharded_matmul[bmm]: height-sharded with explicit shard shape.
    a_sharded_mem_cfg = ttnn.create_sharded_memory_config(
        in0_shape,
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    b_sharded_mem_cfg = ttnn.create_sharded_memory_config(
        in1_shape,
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    a = ttnn.to_memory_config(a, a_sharded_mem_cfg)
    b = ttnn.to_memory_config(b, b_sharded_mem_cfg)

    # Program config for the Reuse factory.
    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=k_size // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m_size // 32,
        per_core_N=n_size // 32,
    )
    return a, b, program_config


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        a, b, program_config = build_inputs(device)
        compute_config = pick_compute_kernel_config()
        out_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )

        def run_once():
            return ttnn.matmul(
                a,
                b,
                program_config=program_config,
                memory_config=out_mem_config,
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
