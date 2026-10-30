# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verifier gap checks: DRAM-interleaved input, affine forms outside the acceptance matrix (mixed layouts,
ROW_MAJOR in DRAM), and the refusals validate() must raise as NotImplementedError."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.layer_norm import layer_norm

WIDTH = 768
PCC = 0.9995


def to_affine(vec, form, device):
    if form == "tile_dram":
        return ttnn.from_torch(
            vec, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    shapes = {
        "rm_l1_sticks": ((1, 1, WIDTH // 32, 32), ttnn.L1_MEMORY_CONFIG),
        "rm_l1_one_stick": ((1, 1, 1, WIDTH), ttnn.L1_MEMORY_CONFIG),
        "rm_dram_sticks": ((1, 1, WIDTH // 32, 32), ttnn.DRAM_MEMORY_CONFIG),
        "rm_dram_one_stick": ((1, 1, 1, WIDTH), ttnn.DRAM_MEMORY_CONFIG),
        "rm_l1_sticks_48": ((1, 1, WIDTH // 48, 48), ttnn.L1_MEMORY_CONFIG),
    }
    shape, memory_config = shapes[form]
    return ttnn.from_torch(
        vec.reshape(shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def run(device, shape, gamma_form, beta_form, input_memory_config, with_residual=False):
    torch.manual_seed(7)
    x = torch.randn(shape).to(torch.bfloat16)
    residual = torch.randn(shape).to(torch.bfloat16) if with_residual else None
    gamma = torch.randn(shape[-1]).to(torch.bfloat16)
    beta = torch.randn(shape[-1]).to(torch.bfloat16)
    ttnn_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    ttnn_residual = None
    if with_residual:
        ttnn_residual = ttnn.from_torch(
            residual, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
        )
    out = layer_norm(ttnn_x, to_affine(gamma, gamma_form, device), to_affine(beta, beta_form, device), ttnn_residual)
    x_ref = x.float() + residual.float() if with_residual else x.float()
    expected = torch.nn.functional.layer_norm(x_ref, (shape[-1],), weight=gamma.float(), bias=beta.float())
    return out, expected


@pytest.mark.parametrize("with_residual", [False, True], ids=["no_residual", "residual"])
@pytest.mark.parametrize("shape", [(32, 1, 32, WIDTH), (7, 1, 300, WIDTH)], ids=["32x32", "7x300"])
def test_dram_interleaved_input(device, shape, with_residual):
    out, expected = run(device, shape, "rm_l1_sticks", "rm_l1_sticks", ttnn.DRAM_MEMORY_CONFIG, with_residual)
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_with_pcc(expected, ttnn.to_torch(out).float(), PCC)


@pytest.mark.parametrize(
    "gamma_form, beta_form",
    [
        ("rm_l1_one_stick", "rm_l1_one_stick"),
        ("tile_dram", "rm_l1_sticks"),
        ("rm_l1_sticks", "tile_dram"),
        ("rm_dram_sticks", "rm_dram_sticks"),
        ("rm_dram_one_stick", "tile_dram"),
    ],
)
def test_affine_forms(device, gamma_form, beta_form):
    out, expected = run(device, (16, 1, 128, WIDTH), gamma_form, beta_form, ttnn.L1_MEMORY_CONFIG, True)
    assert_with_pcc(expected, ttnn.to_torch(out).float(), PCC)


def test_rejects_w_non_aligned(device, expect_error):
    with expect_error(NotImplementedError, "alignment"):
        run(device, (4, 1, 256, 760), "tile_dram", "tile_dram", ttnn.L1_MEMORY_CONFIG)


def test_rejects_rm_affine_stick_not_tile_multiple(device, expect_error):
    with expect_error(NotImplementedError, "stick width"):
        run(device, (16, 1, 128, WIDTH), "rm_l1_sticks_48", "rm_l1_sticks_48", ttnn.L1_MEMORY_CONFIG)


def test_rejects_height_sharded_input(device, expect_error):
    torch.manual_seed(7)
    shape = (1, 1, 32 * 8, WIDTH)
    sharded = ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    x = ttnn.from_torch(
        torch.randn(shape).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded,
    )
    gamma = to_affine(torch.randn(WIDTH).to(torch.bfloat16), "tile_dram", device)
    with expect_error(NotImplementedError, "memory_layout"):
        layer_norm(x, gamma, gamma)
