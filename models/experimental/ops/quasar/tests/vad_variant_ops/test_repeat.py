# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.repeat`` (ROW_MAJOR).

Shape source (kernel cache): ``repeat_last_dim_rm`` ct = [src_page_bytes, num_repeats, ..]
and ``repeat_higher_dim_rm`` ct = [page_bytes, .., lower_pages, rep_dim_size, ..] pin the row
width and last-dim repeat count [H]; the remaining extents are representative [L]. One
user-level repeat can emit several of these kernels (``repeat/repeat.cpp:160-186``).
Repeat is value-preserving: checked bit-exact.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q = U.NUM_QUERY
RM = ttnn.ROW_MAJOR_LAYOUT

# fmt: off
# (shape, repeats, mem)
CASES = [
    pytest.param((1, Q, 1, 2), (1, 1, 1, 4), "dram", id="RP1-last-2to8"),
    pytest.param((1, 100, 2), (1, 1, 16), "dram", id="RP2-last-2to32"),
    pytest.param((1, 100, 16), (1, 1, 8), "dram", id="RP3-last-16to128"),
    pytest.param((1, 100, 8), (1, 1, 16), "l1", id="RP4-last-8to128-l1"),
    pytest.param((1, 1, 32), (1, 10, 1), "dram", id="RP5-higher-w32"),
    pytest.param((1, 1, 24, 4), (1, U.NUM_CAMS, 1, 1), "dram", id="RP6-higher-w4-24rows"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("shape, repeats, mem", CASES)
def test_repeat(ttnn_mesh_device, reset_seeds, shape, repeats, mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, mem, layout=RM)
    out = ttnn.repeat(x, repeats)
    U.assert_lossless(x_torch.repeat(repeats), out, mesh_device=mesh)
