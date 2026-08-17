# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Reproduction of the FLOAT32-input bilinear grid_sample failure seen through the
# forge/ONNX -> tt-mlir -> ttnn path.
#
# The three cases below mirror, op-for-op, the TTNN IR emitted by forge for
#   onnx.GridSample(mode="bilinear", padding_mode="zeros", align_corners=...)
# with a NON-precomputed (raw [-1,1]) grid and an fp32 data tensor:
#
#   %0 = to_layout(arg0 : NCHW row-major f32)      -> TILE   f32
#   %1 = permute(%0, [0,2,3,1])  (NCHW -> NHWC)    -> TILE   f32
#   %2 = to_layout(%1)                             -> ROW_MAJOR f32   <-- grid_sample input
#   %3 = grid_sample(%2, grid)  bilinear, zeros, use_precomputed_grid=false
#   %4 = to_layout(%3)                             -> TILE   f32
#   %5 = permute(%4, [0,3,1,2])  (NHWC -> NCHW)    -> TILE   f32
#
# All tensors are FLOAT32, DRAM INTERLEAVED, matching the #ttnn_layout / #dram
# attributes in the IR. Golden is torch.nn.functional.grid_sample (identical
# semantics to onnx.GridSample opset-18), computed and compared in NCHW.
#
# REGRESSION TEST for the fp32-input bilinear fix. Before the fix these failed with a large
# PCC drop (~0.10-0.45) and outputs ~2-5x too large: the reader always emits the 4 bilinear
# weights as bf16, but the scalar CB was declared with the input dtype, so for an fp32 input the
# compute unpacked the bf16 weight bytes as fp32 and corrupted every weight. The fix declares the
# scalar CB as bf16 unconditionally (grid_sample_bilinear_program_factory.cpp). These now pass at
# PCC ~1.0, matching the bf16-input path.

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG
FP32 = ttnn.float32


def _grid_with_padding_coverage(shape, seed=1):
    """Coordinates spanning [-1.1, 1.1] so ~9% land outside the image and must
    come back as zeros under padding_mode='zeros' (matches the forge test)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.1, 1.1, shape).astype(np.float32)


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


# (data_shape NCHW, grid_shape, align_corners) — exactly the three failing forge cases.
_CASES = [
    ((1, 32, 8, 8), (1, 4, 4, 2), True),
    ((1, 32, 8, 8), (1, 4, 4, 2), False),
    ((1, 64, 96, 96), (1, 128, 64, 2), True),
]


@pytest.mark.parametrize("data_shape, grid_shape, align_corners", _CASES)
def test_grid_sample_fp32_bilinear_repro(device, data_shape, grid_shape, align_corners):
    n, c, h_in, w_in = data_shape
    _, gh, gw, _ = grid_shape

    rng = np.random.default_rng(0)
    data_nchw = torch.from_numpy(rng.standard_normal(data_shape).astype(np.float32))  # NCHW f32
    grid = torch.from_numpy(_grid_with_padding_coverage(grid_shape))  # (N,gh,gw,2) f32

    # ---- Golden: torch (== onnx GridSample opset-18), NCHW in/out ----
    golden_nchw = F.grid_sample(
        data_nchw, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    )  # (N, C, gh, gw)

    # ---- TTNN: replicate the forge IR op chain, all FLOAT32 / DRAM interleaved ----
    # arg0: NCHW row-major f32
    t_arg0 = ttnn.from_torch(data_nchw, layout=RM, dtype=FP32, device=device, memory_config=DRAM)
    # arg1: grid row-major f32 (fed to grid_sample unchanged)
    t_grid = ttnn.from_torch(grid, layout=RM, dtype=FP32, device=device, memory_config=DRAM)

    t0 = ttnn.to_layout(t_arg0, TILE)  # %0  row-major -> tile
    t1 = ttnn.permute(t0, (0, 2, 3, 1))  # %1  NCHW -> NHWC
    t2 = ttnn.to_layout(t1, RM)  # %2  tile -> row-major  (grid_sample input)

    t3 = ttnn.grid_sample(  # %3
        t2,
        t_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
        batch_output_channels=False,
        use_precomputed_grid=False,
    )

    t4 = ttnn.to_layout(t3, TILE)  # %4  row-major -> tile
    t5 = ttnn.permute(t4, (0, 3, 1, 2))  # %5  NHWC -> NCHW
    got_nchw = ttnn.to_torch(t5)

    pcc = _pcc(golden_nchw, got_nchw)
    print(
        f"\n[fp32 bilinear repro] shape={data_shape} grid={grid_shape} ac={align_corners} "
        f"PCC={pcc:.5f}  golden_absmax={golden_nchw.abs().max():.3f}  ttnn_absmax={got_nchw.float().abs().max():.3f}"
    )

    assert list(got_nchw.shape) == [n, c, gh, gw], f"shape {list(got_nchw.shape)} != {[n, c, gh, gw]}"
    # Passes after the fp32 scalar-CB fix (was PCC ~0.10-0.45 before).
    assert_with_pcc(golden_nchw, got_nchw, 0.99)
