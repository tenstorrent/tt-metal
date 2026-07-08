# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone SDPA chunk/grid sweep for BGE-M3 B12/S8192 on N300 (single Wormhole chip).

SDPA is 66.7% of the B12/S8192 prefill device time. This sweep isolates the
encoder (non-causal) SDPA op for the exact model shape and finds the fastest
(q_chunk, k_chunk, grid, max_cores_per_head_batch) via the device profiler,
using the same trace-capture measurement pattern as
models/tt_dit/utils/sweep_mm_block_sizes.py.

Shape (per the model): Q/K/V = [B=12, n_heads=16, S=8192, head_dim=64], bf16,
non-causal, additive attn_mask [B,1,S,S] bf16, scale = head_dim**-0.5.

Run (device profiler, from tt-metal root):
    TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r \\
      --no-runtime-analysis -v -m pytest \\
      models/demos/wormhole/bge_m3/tests/perf/sweep_sdpa_b12_s8192.py -k sweep -sv

Reads the resulting ops CSV yourself with tt-perf-report, or check the printed
per-combo timings (signposted).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


B = 12
N_HEADS = 16
S = 8192
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5

# Candidate chunk sizes (must divide S=8192; S/128=64 q-chunks etc.)
Q_CHUNKS = [128, 256, 512]
K_CHUNKS = [128, 256, 512, 1024]
# Grids to try on the 8x8=64-core N300.
GRIDS = [(8, 8), (8, 4), (7, 8)]
# max_cores_per_head_batch candidates.
MAX_CORES = [16, 8, 32]


def _combos():
    out = []
    for gx, gy in GRIDS:
        for q in Q_CHUNKS:
            for k in K_CHUNKS:
                for mc in MAX_CORES:
                    out.append((gx, gy, q, k, mc))
    return out


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 90_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_sdpa_sweep(mesh_device):
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        pytest.fail("Set TT_METAL_DEVICE_PROFILER=1 and run under python -m tracy.")

    torch.manual_seed(0)
    q = ttnn.from_torch(
        torch.randn(B, N_HEADS, S, HEAD_DIM, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k = ttnn.from_torch(
        torch.randn(B, N_HEADS, S, HEAD_DIM, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v = ttnn.from_torch(
        torch.randn(B, N_HEADS, S, HEAD_DIM, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Additive mask [B,1,S,S] bf16 (all-zero = no-op, same cost as real mask).
    mask = ttnn.from_torch(
        torch.zeros(B, 1, S, S, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    combos = _combos()
    logger.info(f"SDPA sweep: {len(combos)} combos")

    def run(gx, gy, qc, kc, mc):
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=False,
            max_cores_per_head_batch=mc,
        )
        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, attn_mask=mask, scale=SCALE,
            program_config=pc, compute_kernel_config=ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(out)

    # Warmup + filter OOM/invalid combos.
    valid = []
    for c in combos:
        try:
            run(*c)
            ttnn.synchronize_device(mesh_device)
            valid.append(c)
        except Exception as e:
            logger.warning(f"skip {c}: {str(e)[:80]}")
    ttnn.synchronize_device(mesh_device)

    # Measured: single start/stop window; SDPA op rows appear in combo order.
    # Print the ordered combo list so the CSV rows can be matched by index.
    logger.info("SWEEP_ORDER: " + " ".join(f"g{gx}x{gy}_q{qc}_k{kc}_mc{mc}" for gx, gy, qc, kc, mc in valid))
    signpost("start")
    for c in valid:
        run(*c)
        ttnn.synchronize_device(mesh_device)
    signpost("stop")
    logger.info(f"SDPA sweep done: {len(valid)}/{len(combos)} valid combos")
