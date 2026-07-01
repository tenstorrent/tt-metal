# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for ttnn.experimental.transpose_rm — precision-preserving
inner-axis ROW_MAJOR transpose used by the two-pass FFT composite
(commit 3c).
"""

import os
import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "1") == "0",
    reason="TT_FFT_NATIVE=1 not set; new ProgramDescriptor path is gated.",
)


_DTYPES = [
    (ttnn.float32,  torch.float32,  "fp32"),
    (ttnn.bfloat16, torch.bfloat16, "bf16"),
]

# (B, A, C) — last two dims swapped on output.
# Cover small / mid / large factorisations the FFT composite will need.
_SHAPES = [
    (1,   32,   32),     # smallest
    (1,   32,   64),
    (1,   64,   32),
    (1,   64,   64),
    (1,   64,  128),
    (1,  128,  128),     # N=16K composite case
    (1,  256,  128),     # N=32K composite case
    (1,  256,  256),     # N=65K composite case
    (1,  512,  512),     # N=262K
    (1, 1024, 1024),     # N=1M composite case (largest)
    (2,  256,  128),
    (4,  128,  128),
]


@pytest.mark.parametrize("tt_dtype,torch_dtype,label", _DTYPES, ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("B,A,C", _SHAPES, ids=[f"B{b}x{a}x{c}" for (b, a, c) in _SHAPES])
def test_transpose_rm_correctness(device, B, A, C, tt_dtype, torch_dtype, label):
    torch.manual_seed(0)
    x_fp32 = torch.randn(B, A, C, dtype=torch.float32)
    x = x_fp32.to(torch_dtype)

    tt_x = ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_y = ttnn.experimental.transpose_rm(tt_x)
    got = ttnn.to_torch(tt_y).to(torch.float32)
    ref = x.to(torch.float32).transpose(-2, -1).contiguous()

    # Bit-exact: transpose_rm is pure data movement, no math.
    assert got.shape == ref.shape, f"shape mismatch: {got.shape} vs {ref.shape}"
    diff = (got - ref).abs().max().item()
    assert diff == 0.0, (
        f"[{label}] {B}x{A}x{C} transpose_rm max abs diff {diff:.2e} "
        f"(expected bit-exact; transpose is pure data movement)"
    )


def test_transpose_rm_program_cache_hit(device):
    """Standalone program-cache verification — single shape, two calls."""
    B, A, C = 1, 128, 128
    torch.manual_seed(0)
    x = torch.randn(B, A, C, dtype=torch.float32)

    tt_x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.experimental.transpose_rm(tt_x)
    n_after_warmup = device.num_program_cache_entries()

    tt_x2 = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.experimental.transpose_rm(tt_x2)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"transpose_rm program cache regression: "
        f"{n_after_warmup} → {n_after_repeat}"
    )
