# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5: the grouped gamma/beta multicast block (AffineBlock in layer_norm_tile_stream.hpp).

Covers ROW_MAJOR sticks wider than one tile column (the coalesced per-page reads, split at group boundaries:
7 columns per ROW_MAJOR group, 8 per TILE group), partial last groups, both placements, and the beta read on
the injector's writer RISC on and off.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_tile import layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_layer_norm_tile import (
    MEMORY_CONFIG,
    reference_layer_norm,
    to_device,
)

descriptor = importlib.import_module("ttnn.operations.layer_norm_tile.layer_norm_tile_program_descriptor")

DRAM = ttnn.BufferType.DRAM
L1 = ttnn.BufferType.L1


def rel_rms(expected, actual):
    expected = expected.to(torch.float64)
    return ((actual.to(torch.float64) - expected).pow(2).mean().sqrt() / expected.std()).item()


def run_case(device, width, stick, affine_layout, affine_buf, with_residual=True):
    torch.manual_seed(7)
    shape = (2, 1, 64, width)
    x = torch.randn(shape).to(torch.bfloat16)
    res = torch.randn(shape).to(torch.bfloat16) if with_residual else None
    # Distinct per-column scales so a misplaced tile column shows up in the error.
    gamma = (torch.randn(width) * (1 + torch.arange(width) / 32)).to(torch.bfloat16)
    beta = (torch.randn(width) + torch.arange(width) / 32).to(torch.bfloat16)

    def affine(vector):
        if affine_layout == ttnn.TILE_LAYOUT:
            return to_device(vector.reshape(1, 1, 1, width), device, affine_buf, layout=affine_layout)
        return to_device(vector.reshape(1, 1, width // stick, stick), device, affine_buf, layout=affine_layout)

    out = layer_norm(
        to_device(x, device, DRAM),
        affine(gamma),
        affine(beta),
        to_device(res, device, DRAM) if with_residual else None,
        memory_config=MEMORY_CONFIG[DRAM],
    )
    expected = reference_layer_norm(x, gamma, beta, res)
    error = rel_rms(expected, ttnn.to_torch(out).to(torch.float32))
    assert error < 0.02, f"rel-RMS {error}"


@pytest.mark.parametrize(
    "width, stick",
    [
        (768, 32),  # one column per page (production sticks)
        (768, 192),  # 6 columns per page: spans split at the 7-column group boundary
        (768, 768),  # one page: one read per group
        (224, 224),  # exactly one full ROW_MAJOR group
        (256, 128),  # 8 columns: a one-column last group
        (32, 32),  # Wt = 1
    ],
)
@pytest.mark.parametrize("affine_buf", [L1, DRAM], ids=["affL1", "affDRAM"])
def test_row_major_sticks(device, width, stick, affine_buf):
    run_case(device, width, stick, ttnn.ROW_MAJOR_LAYOUT, affine_buf)


@pytest.mark.parametrize("width", [256, 288, 768])
def test_tile_groups(device, width):
    run_case(device, width, None, ttnn.TILE_LAYOUT, DRAM)


@pytest.mark.parametrize("beta_on_writer", [False, True])
@pytest.mark.parametrize("affine_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
@pytest.mark.parametrize("with_residual", [False, True], ids=["nores", "res"])
def test_beta_on_writer(device, monkeypatch, beta_on_writer, affine_layout, with_residual):
    monkeypatch.setattr(descriptor, "AFFINE_BETA_ON_WRITER", beta_on_writer)
    run_case(device, 768, 768, affine_layout, L1, with_residual=with_residual)
