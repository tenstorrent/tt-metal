# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tt/pf_fp4.fp4_fused (one generic_op) vs tt/pf_tune.fp4_fast (bf16 slice chain): must be bit-identical; prints the device time of both."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import pf_fp4, pf_tune


@pytest.mark.parametrize(
    "shape,scale",
    [((2, 8, 256, 128), 0.05), ((2, 8, 256, 128), 1.0), ((2, 8, 256, 128), 30.0), ((4, 64, 128, 128), 1.0)],
)
def test_fp4_fused_matches(device, shape, scale):
    torch.manual_seed(0)
    x = (torch.randn(*shape) * scale).to(torch.bfloat16)
    x[0, 0, 0] = 0  # all-zero blocks
    x[0, 1, 1, :32] = 6.0 * 2.0**3  # amax exactly 6 * 2^k
    x[0, 1, 2, :32] = 3.0 * 2.0**-2
    x[0, 1, 3, 32:64] = -(6.0 * 2.0**-9)
    x[0, 1, 4, 64:96] = 1e-38  # tiny
    t = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ref = pf_tune.fp4_fast(t)
    got = pf_fp4.fp4_fused(t)
    r, g = ttnn.to_torch(ref), ttnn.to_torch(got)
    ndiff = int((r != g).sum())
    print(f"FP4 fused {shape} scale {scale}: {ndiff} of {r.numel()} elements differ", flush=True)
    for name, fn in (("fast", pf_tune.fp4_fast), ("fused", pf_fp4.fp4_fused)):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(10):
            o = fn(t)
        ttnn.synchronize_device(device)
        print(f"  {name}: {(time.perf_counter() - t0) / 10 * 1e3:.3f} ms / call (host wall)", flush=True)
    assert ndiff == 0


@pytest.mark.parametrize(
    "shape", [(8, 1, 32, 128), (12, 1, 32, 128), (1, 1, 32, 128), (16, 1, 32, 128), (2, 8, 256, 128)]
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000}], indirect=True)
def test_fp4_nb1_matches(device, shape):
    """DSV41_FP4_NB1: one core per 32-column tile (nb=1) vs the whole-row block (nb=4): bit-identical; traced device-side time of both (k calls per trace, difference of two trace lengths)."""
    torch.manual_seed(2)
    x = (torch.randn(*shape) * torch.logspace(-2, 2, shape[-2]).reshape(1, 1, -1, 1)).to(torch.bfloat16)
    x[..., 3, :32] = 0
    x[..., 5, 32:64] = 6.0 * 2.0**4
    t = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    a, b = pf_fp4.fp4_fused(t, nb=4), pf_fp4.fp4_fused(t, nb=1)
    ndiff = int((ttnn.to_torch(a) != ttnn.to_torch(b)).sum())
    print(f"FP4 nb1 {shape}: {ndiff} elements differ", flush=True)

    def traced_us(nb, k=8, reps=20):
        def cap(kk):
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(kk):
                pf_fp4.fp4_fused(t, nb=nb)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            return tid

        def run(tid):
            for _ in range(3):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(reps):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            return (time.perf_counter() - t0) / reps * 1e6

        t1, tk = cap(1), cap(k)
        return (run(tk) - run(t1)) / (k - 1)

    print(f"FP4 nb1 {shape}: nb4 {traced_us(4):.1f} us, nb1 {traced_us(1):.1f} us per call (traced)", flush=True)
    assert ndiff == 0


# the chunk shapes of the indexer in the grid cells: q [U, 64 heads, C, 128] and the pooled keys [U, 1, C/4, 128]
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 2048, 128),  # 60k B=4 (U=1, C=2048); also the 2048-row chunk of the ratio-4 indexer at long context
        (4, 64, 1024, 128),  # 4k B=16
        (8, 64, 512, 128),  # 4k B=32
        (1, 64, 1024, 128),
        (1, 1, 512, 128),
        (4, 1, 256, 128),
        (8, 1, 128, 128),
        (1, 1, 32, 128),
    ],
)
def test_fp4_fused_chunk_shapes(device, shape):
    torch.manual_seed(1)
    x = (torch.randn(*shape) * torch.logspace(-2, 2, shape[-2]).reshape(1, 1, -1, 1)).to(torch.bfloat16)
    x[..., 3, :32] = 0
    x[..., 5, 32:64] = 6.0 * 2.0**4
    t = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    r, g = ttnn.to_torch(pf_tune.fp4_fast(t)), ttnn.to_torch(pf_fp4.fp4_fused(t))
    ndiff = int((r != g).sum())
    print(f"FP4 fused chunk shape {shape}: {ndiff} of {r.numel()} elements differ", flush=True)
    assert ndiff == 0
