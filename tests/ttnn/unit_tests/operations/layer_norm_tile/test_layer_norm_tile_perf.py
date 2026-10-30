# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf guard set for --profile runs (device kernel time); correctness is a smoke check only.

Each case dispatches the op RUNS_PER_CASE times in a row; the profiler CSV holds one row per dispatch, in
collection order. Take the minimum per case."""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm

WIDTH = 768
RUNS_PER_CASE = 2
PERF_SHAPES = [
    (32, 1, 32, WIDTH),  # 32 tile-rows
    (16, 1, 128, WIDTH),  # 64 tile-rows
    (8, 1, 512, WIDTH),  # 128 tile-rows
    (124, 1, 33, WIDTH),  # 248 tile-rows, non-aligned S
    (256, 1, 16, WIDTH),  # 256 tile-rows, multi-block with residual at fp32 DEST
]

CONFIGS = {
    "default": None,
    "fp32_acc_off": ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False),
    "lofi": ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
}


def make_affine(vec, affine_form, device):
    if affine_form == "tile_dram":
        return ttnn.from_torch(
            vec, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return ttnn.from_torch(
        vec.reshape(1, 1, WIDTH // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def to_device(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )


@pytest.mark.parametrize("config_name", list(CONFIGS))
@pytest.mark.parametrize("with_residual", [False, True], ids=["no_residual", "residual"])
@pytest.mark.parametrize("affine_form", ["tile_dram", "rm_l1"])
@pytest.mark.parametrize("shape", PERF_SHAPES, ids=lambda s: "x".join(str(d) for d in s))
def test_layer_norm_tile_perf(device, shape, affine_form, with_residual, config_name):
    torch.manual_seed(0)
    x = to_device(torch.randn(shape).to(torch.bfloat16), device)
    residual = to_device(torch.randn(shape).to(torch.bfloat16), device) if with_residual else None
    affine = make_affine(torch.randn(WIDTH).to(torch.bfloat16), affine_form, device)
    kwargs = {} if CONFIGS[config_name] is None else {"compute_kernel_config": CONFIGS[config_name]}
    for _ in range(RUNS_PER_CASE):
        out = layer_norm(x, affine, affine, residual, **kwargs)
        assert tuple(out.shape) == shape
        ttnn.deallocate(out)
