# SPDX-License-Identifier: Apache-2.0
"""Counter-argument test for Q4 ("SDPA is not memory-bound").

Runs a deliberately memory-bound op (large BF16 eltwise add on DRAM-interleaved
tensors: ~1 add per 3 elements moved, so bandwidth-bound) through the SAME
perf-counter pipeline. If the L1/unpacker utilization and the math-thread
input-wait counters rise here while staying near zero for SDPA, that proves the
counters CAN detect memory pressure — i.e. SDPA's ~0%/~5% is a real
not-memory-bound result, not a dead counter.

Run:
    python -m tracy --perf-counter-multipass --profiler-capture-perf-counters=all \
        -m pytest analysis/membound_test.py::test_membound -s
"""
import os
import pytest
import torch


_env = os.environ.get("MEMBOUND_N", "8192")
NS = [int(x) for x in _env.split(",") if x.strip()]


@pytest.mark.parametrize("n", NS)
def test_membound(device, n):
    import ttnn

    # Large square BF16 tensors in DRAM. Eltwise add = 1 flop per element,
    # 3 element-moves (2 read + 1 write) => strongly bandwidth-bound.
    a = torch.randn(1, 1, n, n)
    b = torch.randn(1, 1, n, n)
    tt_a = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    tt_b = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    out = ttnn.add(tt_a, tt_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    print(f"[membound] add done N={n} out.shape={out.shape}", flush=True)
    tt_a.deallocate()
    tt_b.deallocate()
    out.deallocate()


_mm = os.environ.get("MEMBOUND_MM_N", "8192")
MM_NS = [int(x) for x in _mm.split(",") if x.strip()]


@pytest.mark.parametrize("n", MM_NS)
def test_membound_matmul(device, n):
    """DRAM-read-bound matmul: [n, K] x [K, n] with K = 1 tile (32).

    ~1 tile-MAC produced per 3 tile-moves (2 read + 1 write), no data reuse
    across output tiles, all in DRAM => the reader cannot keep the compute
    input CB full, so the math thread should stall on input (math_sem_wait rises)
    and the L1/unpacker ports run hot — the opposite of compute-bound SDPA.
    """
    import ttnn

    K = 32  # one tile of contraction
    a = torch.randn(1, 1, n, K)
    b = torch.randn(1, 1, K, n)
    tt_a = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    tt_b = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    out = ttnn.matmul(tt_a, tt_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    print(f"[membound] matmul done n={n} K={K} out.shape={out.shape}", flush=True)
    tt_a.deallocate()
    tt_b.deallocate()
    out.deallocate()
