# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: SDPA prefill (scaled_dot_product_attention, is_causal=True).

Targets the SDPA prefill compute kernel chain, which is the primary consumer
of the unified ``matmul_block`` helper's ``transpose=True`` template mode (for
the Q@K^T matmul) plus the ``matmul_reduce_inplace`` helper for the in-place
reduce step. See ``project_matmul_helpers_phase1_context.md`` for the full
list of SDPA call sites migrated to the new helper.

Shape: b=1, nh=8, seq_len=2048, d=128. Representative of a mid-sized LLM
prefill path (e.g. Llama2-style head dims).

Run (from the repo root, after ``./build_metal.sh``):

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/sdpa_prefill.py
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


SCRIPT_LABEL = "sdpa_prefill"


def build_inputs(device):
    torch.manual_seed(1234)
    b, nh, nkv, s, d = 1, 8, 8, 2048, 128

    q_chunk_size = 128
    k_chunk_size = 128

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    Q = torch.randn(b, nh, s, d) * 0.5
    K = torch.randn(b, nkv, s, d) * 0.5
    V = torch.randn(b, nkv, s, d) * 0.5

    tt_Q = ttnn.from_torch(
        Q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    tt_K = ttnn.from_torch(
        K,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    tt_V = ttnn.from_torch(
        V,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    return tt_Q, tt_K, tt_V, program_config


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        tt_Q, tt_K, tt_V, program_config = build_inputs(device)
        compute_kernel_config = pick_compute_kernel_config(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def run_once():
            return ttnn.transformer.scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                is_causal=True,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
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
