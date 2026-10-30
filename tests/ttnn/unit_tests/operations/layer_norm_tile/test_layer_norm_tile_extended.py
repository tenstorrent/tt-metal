# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verifier gap checks: DRAM-interleaved input, one-stick and mixed affine forms, and the refusals validate()
must raise as NotImplementedError (registry support refusals)."""

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
    if form == "rm_l1_sticks":
        shape, memory_config = (1, 1, WIDTH // 32, 32), ttnn.L1_MEMORY_CONFIG
    elif form == "rm_l1_one_stick":
        shape, memory_config = (1, 1, 1, WIDTH), ttnn.L1_MEMORY_CONFIG
    else:
        assert form == "rm_dram_sticks"
        shape, memory_config = (1, 1, WIDTH // 32, 32), ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(
        vec.reshape(shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def run(device, shape, gamma_form, beta_form, input_memory_config):
    torch.manual_seed(7)
    x = torch.randn(shape).to(torch.bfloat16)
    gamma = torch.randn(shape[-1]).to(torch.bfloat16)
    beta = torch.randn(shape[-1]).to(torch.bfloat16)
    ttnn_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    out = layer_norm(ttnn_x, to_affine(gamma, gamma_form, device), to_affine(beta, beta_form, device))
    expected = torch.nn.functional.layer_norm(x.float(), (shape[-1],), weight=gamma.float(), bias=beta.float())
    return out, expected


@pytest.mark.parametrize("shape", [(32, 1, 32, WIDTH), (7, 1, 300, WIDTH)], ids=["32x32", "7x300"])
def test_dram_interleaved_input(device, shape):
    out, expected = run(device, shape, "rm_l1_sticks", "rm_l1_sticks", ttnn.DRAM_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_with_pcc(expected, ttnn.to_torch(out).float(), PCC)


@pytest.mark.parametrize(
    "gamma_form, beta_form",
    [("rm_l1_one_stick", "rm_l1_one_stick"), ("tile_dram", "rm_l1_sticks"), ("rm_l1_sticks", "tile_dram")],
)
def test_affine_forms(device, gamma_form, beta_form):
    out, expected = run(device, (16, 1, 128, WIDTH), gamma_form, beta_form, ttnn.L1_MEMORY_CONFIG)
    assert_with_pcc(expected, ttnn.to_torch(out).float(), PCC)


def test_rejects_w_non_aligned(device, expect_error):
    with expect_error(NotImplementedError, "alignment"):
        run(device, (4, 1, 256, 760), "tile_dram", "tile_dram", ttnn.L1_MEMORY_CONFIG)


def test_rm_affine_in_dram(device, expect_error):
    # Row-0 halves are read at stick offsets o and o + 32 B; a DRAM NoC alignment above 32 B cannot serve them.
    if 32 % ttnn.get_dram_alignment() == 0:
        out, expected = run(device, (16, 1, 128, WIDTH), "rm_dram_sticks", "rm_dram_sticks", ttnn.L1_MEMORY_CONFIG)
        assert_with_pcc(expected, ttnn.to_torch(out).float(), PCC)
    else:
        with expect_error(NotImplementedError, "ROW_MAJOR gamma"):
            run(device, (16, 1, 128, WIDTH), "rm_dram_sticks", "rm_dram_sticks", ttnn.L1_MEMORY_CONFIG)
