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
