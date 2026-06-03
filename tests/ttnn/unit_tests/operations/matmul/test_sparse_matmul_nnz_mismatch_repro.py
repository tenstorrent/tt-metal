# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproducer: ttnn.sparse_matmul deadlocks when the declared `nnz` is
GREATER than the actual number of non-zero entries in the `sparsity` tensor.

Single device, no mesh / no fabric / no model — the deadlock is entirely inside
the on-chip in0 multicast of the sparse_matmul 1D-mcast program.

Root cause (in0-mcast handshake):
  * With `nnz` supplied, the factory sets `get_batch_from_reader = false` and wires:
      - in0 SENDER  (reader_bmm_tile_layout_in0_sender_padding.cpp): loops over ALL
        batchB experts and `continue`s (issues NO multicast) for every zero `sparsity`
        entry -> it multicasts exactly `count_nonzero(sparsity)` times.
      - in0 RECEIVER (reader_bmm_tile_layout_in0_receiver.cpp): loops a FIXED
        `batch = num_batch_compute = nnz` times, waiting on the in0 semaphore each time.
      - compute (bmm_large_block...): also loops `nnz` times.
  * So the op silently requires `count_nonzero(sparsity) == nnz`.
  * If `count_nonzero(sparsity) < nnz`, the receivers wait on an in0 multicast that the
    sender never issues -> they block forever in noc_semaphore_wait, compute blocks in
    llk_wait_tiles, and the op never completes (device wedges).

Observed in production as a gpt-oss-120B decode hang on Blackhole, where the MoE router's
top-k softmax weights flush small values to exactly 0 on BH, so the per-token sparsity has
fewer than `nnz=num_experts_per_tok` non-zeros for some tokens.

How to run (single Blackhole/Wormhole device):

  # Control — declared nnz == actual non-zeros -> PASSES quickly:
  pytest tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_nnz_mismatch_repro.py::test_nnz_matches_actual_passes -x -v

  # Bug — declared nnz (4) > actual non-zeros (1) -> HANGS (deadlock).
  # Use the operation timeout so it surfaces as a timeout instead of wedging forever:
  TT_METAL_OPERATION_TIMEOUT_SECONDS=30 \
    pytest tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_nnz_mismatch_repro.py::test_nnz_greater_than_actual_hangs -x -v

The only difference between the two tests is the value passed as `nnz`.
"""

import math

import pytest
import torch
from loguru import logger

import ttnn


def _run_sparse_matmul(device, *, declared_nnz, actual_nnz):
    """Build a tiny single-token MoE-style sparse_matmul and run it.

    in0 : [1, 1, M, K]            (dense activation, broadcast over experts)
    in1 : [1, E, K, N]            (per-expert weights; B is the sparse operand)
    sparsity : [1, 1, 1, E]       (row-major bf16; `actual_nnz` non-zero entries)
    """
    # Shapes chosen so the 1D mcast program spans the full (4,4) core grid:
    #   N=512, tile_w=32 -> Nt=16 == num_cores -> one sender + 15 receivers.
    #   M=32, tile_h=32  -> Mt=1 (single-token decode shape).
    m, k, n = 32, 128, 512
    num_experts = 8
    tile_h, tile_w = 32, 32
    core_x, core_y = 4, 4
    assert actual_nnz <= num_experts and declared_nnz <= num_experts

    torch.manual_seed(0)
    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    # sparsity with EXACTLY `actual_nnz` non-zero entries (the first `actual_nnz` experts).
    sparsity = torch.zeros((1, 1, 1, num_experts), dtype=torch.float32)
    sparsity[..., :actual_nnz] = 1.0
    sparsity = sparsity.to(dtype=torch.bfloat16)
    assert int((sparsity != 0).sum().item()) == actual_nnz

    in0_t = ttnn.from_torch(
        in0, tile=ttnn.Tile((tile_h, 32)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    in1_t = ttnn.from_torch(
        in1, tile=ttnn.Tile((32, tile_w)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sparsity_t = ttnn.from_torch(
        sparsity, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"declared nnz={declared_nnz}, actual count_nonzero(sparsity)={actual_nnz}")
    out_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=declared_nnz,  # <-- the only knob that matters for the deadlock
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=ttnn.Tile([tile_h, tile_w]),
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=m // tile_h,
            per_core_N=int(math.ceil(n / tile_w)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )
    # Force completion (host readback). With the bug, the op above never finishes.
    _ = ttnn.to_torch(out_t)
    logger.info("sparse_matmul completed")


def test_nnz_matches_actual_passes(device):
    """Control: declared nnz == actual non-zeros -> sender mcasts == receiver waits -> OK."""
    _run_sparse_matmul(device, declared_nnz=1, actual_nnz=1)


def test_nnz_greater_than_actual_hangs(device):
    """Bug: declared nnz (4) > actual non-zeros (1).

    Sender multicasts in0 only once (one non-zero expert), but each receiver waits for 4
    multicasts -> receivers deadlock in noc_semaphore_wait. The op never completes; run
    with TT_METAL_OPERATION_TIMEOUT_SECONDS to surface it as a timeout.
    """
    _run_sparse_matmul(device, declared_nnz=4, actual_nnz=1)
