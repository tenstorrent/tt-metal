# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: SDPA decode (scaled_dot_product_attention_decode).

Exercises the flash-decode kernel chain in
``sdpa_flash_decode.cpp`` — the phase-1 branch migrated two call sites here
to the unified ``matmul_block`` + ``matmul_reduce_inplace`` helpers (see
``project_matmul_helpers_phase1_context.md``).

Shape: b=8, nh=8, nkv=1 (MQA), seq_len=8192, d=128. Representative of
Llama2-70B-style decode serving a small-ish context.

Run (from the repo root, after ``./build_metal.sh``):

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/sdpa_decode.py
"""

from __future__ import annotations

import math
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


SCRIPT_LABEL = "sdpa_decode"


def num_to_corerange(b):
    """Return a single CoreRange covering ``b`` cores row-major from (0,0).

    Intentionally simple: phase 2 may replace with
    ``tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils.num_to_corerange``
    if that import works without pulling in pytest dependencies.
    """
    assert 1 <= b <= 64
    y = (b + 7) // 8  # round up
    x = min(b, 8)
    return ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))


def nearest_pow_2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def get_chunk_size(max_context, s):
    # Matches the helper in sdpa_test_utils: 128 for small contexts, 512 up
    # to 32k, 1024 beyond.
    if max_context <= 2048:
        return 128
    if max_context <= 32768:
        return 512
    return 1024


def build_inputs(device):
    torch.manual_seed(1234)
    b, nh, nkv, s, d = 8, 8, 1, 8192, 128
    grid_size = (8, 6)  # Llama2-70B decode grid

    compute_grid = device.compute_with_storage_grid_size()
    grid_x = min(grid_size[0], compute_grid.x)
    grid_y = min(grid_size[1], compute_grid.y)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))

    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    K = torch.randn(b, nkv, s, d) * 0.5
    V = torch.randn(b, nkv, s, d) * 0.5
    Q = torch.randn(1, b, nh, d) * 0.5

    tt_K = ttnn.as_tensor(K, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_V = ttnn.as_tensor(V, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram_memcfg)
    tt_Q = ttnn.as_tensor(
        Q[:, :, :nh],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
    )

    start_indices = [s // 2 for _ in range(b)]
    max_start_idx = max(start_indices)
    scale = d**-0.5

    q_chunk_size = padded_num_heads
    k_chunk_size = get_chunk_size(max_start_idx + 1, s)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    return tt_Q, tt_K, tt_V, start_indices, scale, program_config


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        tt_Q, tt_K, tt_V, start_indices, scale, program_config = build_inputs(device)
        compute_kernel_config = pick_compute_kernel_config(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def run_once():
            return ttnn.transformer.scaled_dot_product_attention_decode(
                tt_Q,
                tt_K,
                tt_V,
                cur_pos=start_indices,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
