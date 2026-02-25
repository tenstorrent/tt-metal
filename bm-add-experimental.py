# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark ttnn.experimental.add: DRAM interleaved vs DRAM sharded performance.

Focus: compare DRAM interleaved (ttnn.DRAM_MEMORY_CONFIG) vs DRAM HEIGHT_SHARDED
for attention-like shapes:
  (B, H, S, hid/H), (B, 1, S, S), (B, H, S, S), (B, 1, S, hid*8/3)
  B=1/2/4, S=2048/4096, H=32, hid=2048/4096.

Run with:
  pytest bm-add-experimental.py -v -s
  pytest bm-add-experimental.py -v -s --benchmark-only
  pytest bm-add-experimental.py -k interleaved -v -s --benchmark-only
  pytest bm-add-experimental.py -k sharded -v -s --benchmark-only
"""

import time
import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

# Attention-like shape parameters
# B = batch, S = sequence length, H = heads, hid = hidden dim
BATCH_SIZES = (1, 2, 4)
SEQ_LENGTHS = (2048, 4096)
NUM_HEADS = 32
HIDDEN_DIMS = (2048, 4096)

SHAPE_KINDS = {
    # "BHS_head": lambda b, s, h, hid: (b, h, s, hid // h),        # (B, H, S, hid/H)
    "B1SS": lambda b, s, h, hid: (b, 1, s, s),  # (B, 1, S, S)
    # "BHSS": lambda b, s, h, hid: (b, h, s, s),                   # (B, H, S, S)
    # "B1S_hid83": lambda b, s, h, hid: (b, 1, s, (hid * 8) // 3), # (B, 1, S, hid*8/3)
}


def _get_shape(shape_kind, batch_size, seq_len, num_heads, hidden_dim):
    return SHAPE_KINDS[shape_kind](batch_size, seq_len, num_heads, hidden_dim)


BENCHMARK_CONFIGS = [(sk, b, s, h) for sk in SHAPE_KINDS for b in BATCH_SIZES for s in SEQ_LENGTHS for h in HIDDEN_DIMS]


def get_optimal_dram_bank_to_reader_assignment(device, noc=ttnn.NOC.NOC_0):
    """Python equivalent of C++ get_optimal_dram_bank_to_reader_assignment (test_dram_read.cpp).
    Returns (all_worker_cores_ordered, all_worker_cores) where:
      - all_worker_cores_ordered: list of CoreCoord in optimal DRAM bank -> reader order
      - all_worker_cores: CoreRangeSet of single-core ranges for those coords
    """
    all_worker_cores_ordered = device.get_optimal_dram_bank_to_logical_worker_assignment(noc)
    all_worker_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores_ordered]
    )
    return all_worker_cores_ordered, all_worker_cores


# ---- DRAM interleaved benchmarks ----
@pytest.mark.parametrize(
    "config",
    BENCHMARK_CONFIGS,
    ids=lambda c: f"interleaved_{c[0]}_B{c[1]}_S{c[2]}_hid{c[3]}",
)
def test_benchmark_add_ng_dram_interleaved(benchmark, device, config):
    """Benchmark ttnn.experimental.add with DRAM interleaved (ttnn.DRAM_MEMORY_CONFIG)."""
    shape_kind, batch_size, seq_len, hidden_dim = config
    shape = _get_shape(shape_kind, batch_size, seq_len, NUM_HEADS, hidden_dim)

    torch.manual_seed(0)
    torch_a = torch.randn(shape, dtype=torch.bfloat16) * 0.1
    torch_b = torch.randn(shape, dtype=torch.bfloat16) * 0.1

    a_tt = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run_add():
        ttnn.mul(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)

    benchmark.pedantic(run_add, iterations=5, rounds=3, warmup_rounds=1)


@pytest.mark.parametrize(
    "config",
    BENCHMARK_CONFIGS,
    ids=lambda c: f"sharded_{c[0]}_B{c[1]}_S{c[2]}_hid{c[3]}",
)
def test_benchmark_experimental_add_dram_sharded(benchmark, device, config):
    """Benchmark ttnn.experimental.add with DRAM HEIGHT_SHARDED."""
    shape_kind, batch_size, seq_len, hidden_dim = config
    shape = _get_shape(shape_kind, batch_size, seq_len, NUM_HEADS, hidden_dim)

    _, shard_grid = get_optimal_dram_bank_to_reader_assignment(device, ttnn.NOC.NOC_0)

    torch.manual_seed(0)

    torch_a = torch.randn(shape, dtype=torch.bfloat16) * 0.1
    torch_b = torch.randn(shape, dtype=torch.bfloat16) * 0.1

    a_tt = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run_add():
        ttnn.experimental.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=shard_grid)  #
        ttnn.synchronize_device(device)

    benchmark.pedantic(run_add, iterations=5, rounds=3, warmup_rounds=1)
