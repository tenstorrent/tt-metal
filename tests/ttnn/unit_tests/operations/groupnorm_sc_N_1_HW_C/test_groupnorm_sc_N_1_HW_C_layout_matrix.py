# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layout matrix for groupnorm_sc_N_1_HW_C (Refinement 2 — padding-independent non-aligned paths).

/memory-layouts §7: every layout pair the op claims x aligned / HW-non-aligned / C-non-aligned / both shapes x
{bf16, fp32}. Output is always TILE. Plus the two cases this refinement pins for the next phase:

* ``test_rm_hw_non_aligned_two_images``: RM input, HW % 32 != 0, N = 2 with a large per-image offset — before
  Refinement 2 the stick reader read the next image's rows into the first image's statistics, which passes
  silently for N = 1 / num_groups = 1 unless the second image differs.
* ``test_tile_padding_garbage_independent``: TILE input whose pad rows / lanes carry a large finite garbage
  value (``ttnn.from_torch(pad_value=...)``) — the statistics must not depend on the padding contents.
"""

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

# The acceptance test is not a package (no __init__.py): load its reference / PCC helpers by path.
import importlib.util as _ilu
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location("_gn_acceptance", _Path(__file__).with_name("test_groupnorm_sc_N_1_HW_C.py"))
_acc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_acc)
PCC_BY_DTYPE, TORCH_DTYPE, pcc, pytorch_reference = (
    _acc.PCC_BY_DTYPE,
    _acc.TORCH_DTYPE,
    _acc.pcc,
    _acc.pytorch_reference,
)


def _check(output, expected, shape, dtype):
    assert output.dtype == dtype
    assert output.layout == ttnn.TILE_LAYOUT
    assert list(output.shape) == list(shape)
    got = ttnn.to_torch(output).to(torch.float32)
    exp = expected.to(torch.float32)
    assert torch.isfinite(got).all(), "output contains NaN/Inf"
    p = pcc(got, exp)
    assert p >= PCC_BY_DTYPE[dtype], f"PCC {p:.6f} < {PCC_BY_DTYPE[dtype]}"
    return got, exp


SHAPES = [
    pytest.param((1, 1, 32, 32), 1, id="32x32_aligned_small"),
    pytest.param((1, 1, 32, 64), 2, id="32x64_aligned"),
    pytest.param((1, 1, 64, 128), 4, id="64x128_aligned"),
    pytest.param((1, 1, 256, 640), 32, id="256x640_aligned_straddle"),
    pytest.param((1, 1, 64, 48), 2, id="64x48_C_non_aligned"),
    pytest.param((1, 1, 17, 64), 1, id="17x64_HW_non_aligned"),
    pytest.param((1, 1, 50, 128), 1, id="50x128_HW_non_aligned"),
    pytest.param((1, 1, 47, 50), 1, id="47x50_both_non_aligned"),
    pytest.param((2, 1, 100, 80), 4, id="2x100x80_both_non_aligned_batch2"),
]


@pytest.mark.parametrize("dtype", [pytest.param(ttnn.bfloat16, id="bf16"), pytest.param(ttnn.float32, id="fp32")])
@pytest.mark.parametrize("shape,num_groups", SHAPES)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_groupnorm_sc_N_1_HW_C_layout_matrix(device, shape, num_groups, input_layout, dtype):
    if input_layout == ttnn.ROW_MAJOR_LAYOUT and dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        pytest.skip("Block formats do not support ROW_MAJOR layout")
    torch.manual_seed(7)
    x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    expected = pytorch_reference(x, num_groups)
    _check(groupnorm_sc_N_1_HW_C(ttnn_x, num_groups), expected, shape, dtype)


@pytest.mark.parametrize("shape,num_groups", [((2, 1, 100, 128), 1), ((2, 1, 17, 64), 1), ((3, 1, 50, 96), 3)])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["rm", "tile"])
def test_rm_hw_non_aligned_two_images(device, shape, num_groups, input_layout):
    """Images with different offsets: reading image n+1's rows into image n's statistics shows up as a
    per-image mean shift (PCC drop and a large per-image mean error).

    The offsets stay O(std): the op's one-pass variance (E[x^2] - mean^2, tf32-class FPU operands) is the
    documented precision limitation of verification_report.md, and offsets of 20-60 x std collapse it on
    tile-ALIGNED shapes too (probe_009: (3,1,64,96) PCC 0.34 at offsets 20/40/60, 0.99998 at 2/4/6)."""
    torch.manual_seed(11)
    N = shape[0]
    x = torch.randn(shape, dtype=torch.float32)
    for n in range(N):
        x[n] += 2.0 * (n + 1) * (-1) ** n  # image n offset: -2, +4, -6, ...
    x = x.to(torch.bfloat16)
    ttnn_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    expected = pytorch_reference(x, num_groups)
    got, exp = _check(groupnorm_sc_N_1_HW_C(ttnn_x, num_groups), expected, shape, ttnn.bfloat16)
    for n in range(N):
        assert abs(float(got[n].mean())) < 0.05, f"image {n}: output mean {float(got[n].mean()):.3f} (expected ~0)"


@pytest.mark.parametrize("shape,num_groups", [((1, 1, 17, 64), 1), ((2, 1, 50, 80), 4), ((1, 1, 100, 200), 8)])
@pytest.mark.parametrize("pad_value", [1000.0, -333.0])
def test_tile_padding_garbage_independent(device, shape, num_groups, pad_value):
    """TILE input whose padding rows (HW % 32) / lanes (C % 32) hold a large finite garbage value."""
    torch.manual_seed(5)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ttnn_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=pad_value,
    )
    expected = pytorch_reference(x, num_groups)
    _check(groupnorm_sc_N_1_HW_C(ttnn_x, num_groups), expected, shape, ttnn.bfloat16)
