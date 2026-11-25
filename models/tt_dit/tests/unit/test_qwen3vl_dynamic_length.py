# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Single-device pieces of the Qwen3-VL decoder's dynamic-length path: one program for every row count."""

from types import SimpleNamespace

import pytest
import torch

import ttnn

from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlContext, _MatmulRotateHalf, _rotate_half
from ...utils.tensor import bf16_tensor, to_torch

HEAD_DIM = 128
# Long, short, long: a program compiled at one row count must serve a larger and a smaller one.
ROWS = (320, 32, 64, 96, 640)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(("heads", "programs"), [(16, 1), (2, 2)], ids=["q", "k"])
def test_rotate_half_matmul(mesh_device: ttnn.MeshDevice, heads: int, programs: int) -> None:
    """`x @ R` equals slice + neg + concat value for value. K changes program once, where its rows pass head_dim."""
    torch.manual_seed(0)
    rotate = _MatmulRotateHalf(mesh_device, HEAD_DIM)
    added = 0
    for rows in ROWS:
        x = bf16_tensor(torch.randn(1, heads, rows, HEAD_DIM) * 3, device=mesh_device)
        expected = to_torch(_rotate_half(x))
        entries = mesh_device.num_program_cache_entries()
        actual = to_torch(rotate(x))
        added += mesh_device.num_program_cache_entries() - entries
        assert torch.equal(actual, expected), f"rows={rows}"
    assert added == programs, f"{added} rotation programs over rows {ROWS}"


def test_context_dynamic_length_needs_sp_and_capacity(expect_error) -> None:
    with expect_error(ValueError, "dynamic_length"):
        Qwen3VlContext(device=None, tp_axis=0, ccl_manager=object(), dynamic_length=True)
    with expect_error(ValueError, "dynamic_length"):
        Qwen3VlContext(device=None, tp_axis=0, ccl_manager=object(), sp_axis=1, dynamic_length=True)


def test_context_dynamic_length_needs_line_reduce_scatter(expect_error) -> None:
    blackhole = SimpleNamespace(arch=lambda: ttnn.device.Arch.BLACKHOLE)
    ring = SimpleNamespace(reduce_scatter_topology=ttnn.Topology.Ring)
    with expect_error(ValueError, "reduce_scatter_topology"):
        Qwen3VlContext(
            device=blackhole, tp_axis=0, ccl_manager=ring, sp_axis=1, kv_gather_capacity=5120, dynamic_length=True
        )


def test_context_dynamic_length_is_blackhole_only(expect_error) -> None:
    wormhole = SimpleNamespace(arch=lambda: ttnn.device.Arch.WORMHOLE_B0)
    with expect_error(ValueError, "Blackhole only"):
        Qwen3VlContext(
            device=wormhole, tp_axis=0, ccl_manager=object(), sp_axis=1, kv_gather_capacity=5120, dynamic_length=True
        )
