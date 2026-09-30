# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf research (not for merge): SDPA column math on the TRISC2 RISC-V vector unit (Zve32f)."""

import math
import os

import numpy as np
import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

KERNEL = "tests/ttnn/unit_tests/operations/sdpa/kernels/rvv_column_bench.cpp"


def bf16_bits(x):
    return (np.asarray(x, np.float32).view(np.uint32) + 0x7FFF + ((np.asarray(x, np.float32).view(np.uint32) >> 16) & 1)) >> 16


def from_bf16(bits):
    return (np.asarray(bits, np.uint32) << 16).view(np.float32)


def round_bf16(x):
    return from_bf16(bf16_bits(x))


def column_offsets():
    # Column 0 of a face-ordered 32x32 tile: element index of each row.
    return np.array([16 * r for r in range(16)] + [512 + 16 * r for r in range(16)])


@pytest.mark.skipif(os.getenv("TEST_SDPA_PERF_RVV") != "1", reason="Opt-in perf research")
def test_rvv_column_math(device):
    if not is_blackhole():
        pytest.skip("RVV is Blackhole TRISC2 only")
    rng = np.random.default_rng(0)
    scale = 1 / math.sqrt(128)
    buf = np.zeros(8192, np.uint8)
    m_old = round_bf16(rng.uniform(-5, 5, 32).astype(np.float32))
    m_new = round_bf16(m_old + rng.uniform(0, 3, 32).astype(np.float32))
    old_tile = np.zeros(1024, np.uint16)
    new_tile = np.zeros(1024, np.uint16)
    old_tile[column_offsets()] = bf16_bits(m_old)
    new_tile[column_offsets()] = bf16_bits(m_new)
    buf[0:2048] = old_tile.view(np.uint8)
    buf[2048:4096] = new_tile.view(np.uint8)
    hi = round_bf16(rng.uniform(1, 100, 32).astype(np.float32))
    lo = round_bf16(rng.uniform(-0.01, 0.01, 32).astype(np.float32))
    chunk = rng.uniform(1, 50, 32).astype(np.float32)
    buf[6304:6688] = np.concatenate([hi, lo, chunk]).astype(np.float32).view(np.uint8)

    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core, [64, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    host = torch.from_numpy(buf.view(np.float32).reshape(64, 32).copy())
    scratch = ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [scratch.buffer_address(), int(np.float32(scale).view(np.uint32))]
    config = ttnn.ComputeConfigDescriptor()
    config.enable_trisc2_rvv = True
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        core_ranges=core,
        compile_time_args=[],
        runtime_args=rt,
        config=config,
    )
    # generic_op wants at least one input and one output; the kernel only touches `scratch`.
    dummy = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.generic_op([dummy, scratch], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[]))
    out = ttnn.to_torch(scratch).numpy().reshape(-1).view(np.uint8)

    stats = out[6272:6304].view(np.uint32)
    c = out[4096:4224].view(np.float32)
    c_tile = out[4224:6272].view(np.uint16)[column_offsets()]
    folded = out[6304:6688].view(np.float32)
    ref_c = np.exp(np.float64(scale) * (m_old.astype(np.float64) - m_new))
    c_err = np.max(np.abs(c - ref_c) / ref_c)
    c_tile_err = np.max(np.abs(from_bf16(c_tile) - ref_c) / ref_c)
    total = (hi.astype(np.float64) + lo) * c + chunk
    fold_err = np.max(np.abs(folded[:32].astype(np.float64) + folded[32:64] - total) / total)
    reps = int(stats[5])
    print(
        f"RVV vlmax e32m1={stats[0]} e32m4={stats[1]} | ticks per 32-row op: "
        f"c_fp32={stats[2] / reps:.1f} c_bf16_tile={stats[3] / reps:.1f} hi_lo_fold={stats[4] / reps:.1f} | "
        f"rel err c={c_err:.2e} c_bf16={c_tile_err:.2e} fold={fold_err:.2e}"
    )
    assert c_err < 1e-5 and c_tile_err < 8e-3 and fold_err < 1e-5
