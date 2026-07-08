# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `tile_reorder` data-movement example (shuffle vs. do-the-work).

See ttnn/ttnn/operations/examples/tile_reorder/README.md for what this
illustrates and why. Tests:

    # both methods produce identical, correct output
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py::test_tile_reorder_correctness

    # measured device kernel duration, scatter vs relocate
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py::test_tile_reorder_device_perf
"""

import pytest
import torch

import ttnn
from ttnn.operations.examples.tile_reorder import tile_reorder

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

try:
    from tracy import signpost
except ImportError:

    def signpost(_message):
        pass


TILE = 32
SHAPE = (1024, 1024)  # 32x32 = 1024 tiles
N_WARMUP = 5
N_PROFILE_ITERS = 20


def _reference(torch_input):
    """Reverse the 32-wide column blocks (whole-tile reversal along width)."""
    h, w = torch_input.shape
    return torch_input.reshape(h, w // TILE, TILE).flip(1).reshape(h, w).contiguous()


def _make_input(device):
    torch.manual_seed(0)
    torch_input = torch.rand(SHAPE, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return torch_input, tt_input


@pytest.mark.parametrize("method", ["scatter", "relocate"], ids=["scatter", "relocate"])
def test_tile_reorder_correctness(device, method):
    """Both methods must produce identical, correct output."""
    torch_input, tt_input = _make_input(device)
    expected = _reference(torch_input)

    out = ttnn.to_torch(tile_reorder(tt_input, method=method)).to(torch.float32)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, 0.9999)


@pytest.mark.parametrize("method", ["scatter", "relocate"], ids=["scatter", "relocate"])
def test_tile_reorder_workload(device, method):
    """Profileable body: correctness check + a signpost-bracketed reorder loop.

    Every op row between signposts is GenericOpDeviceOperation (the reorder).
    """
    torch_input, tt_input = _make_input(device)
    expected = _reference(torch_input)

    out = ttnn.to_torch(tile_reorder(tt_input, method=method)).to(torch.float32)
    assert_with_pcc(expected, out, 0.9999)

    for _ in range(N_WARMUP):
        tile_reorder(tt_input, method=method)
    ttnn.synchronize_device(device)

    signpost("start")
    for _ in range(N_PROFILE_ITERS):
        tile_reorder(tt_input, method=method)
    ttnn.synchronize_device(device)
    signpost("stop")


@pytest.mark.models_device_performance_bare_metal
def test_tile_reorder_device_perf():
    """Measure device kernel duration for scatter vs relocate and assert the win.

    The result is a pure tile relocation; `relocate` moves each whole 2 KB tile,
    while `scatter` writes the same bytes as 4 smaller 512 B faces. The whole-tile
    shuffle MUST be at least as fast — this is the number the workflow should
    internalize.
    """
    from models.perf.device_perf_utils import run_device_perf

    test_file = "tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py"
    subdir = "example_tile_reorder"

    per_op = {}
    for method in ("scatter", "relocate"):
        results = run_device_perf(
            f"pytest {test_file}::test_tile_reorder_workload[{method}]",
            subdir,
            1,  # one profiler pass; the workload's 20-iter signpost loop already averages
            ["DEVICE KERNEL"],
            batch_size=1,
            op_name="GenericOpDeviceOperation",
            has_signposts=True,
        )
        per_op[method] = results["AVG DEVICE KERNEL DURATION [ns]"] / N_PROFILE_ITERS

    logger.info(
        "\n=== tile_reorder device perf (whole-tile shuffle vs per-face scatter) ==="
        "\n  scatter  (4 x 512 B faces / tile): %10.1f ns/op"
        "\n  relocate (1 x 2 KB page / tile)  : %10.1f ns/op" % (per_op["scatter"], per_op["relocate"])
    )
    assert (
        per_op["relocate"] <= per_op["scatter"]
    ), f"relocate ({per_op['relocate']:.1f} ns) should be <= scatter ({per_op['scatter']:.1f} ns)"
