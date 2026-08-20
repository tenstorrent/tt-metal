# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for pad's codegen implementation (ttnn.pad(..., implementation="codegen")).

Correctness and performance across the port's declared coverage are settled by the generator's
sweep. What lives here are the cases that sweep cannot reach -- shapes far outside its grid whose
host-side sizing decisions therefore have no other coverage.
"""

import pytest
import torch

import ttnn


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


# build_pad_codegen_params only clamps read_batch/write_batch once the output pages approach the
# per-core L1 budget, around 80KB sticks. Every shape in the codegen coverage sweep is orders of
# magnitude below that cliff, so neither the clamp nor the alignment its page size is charged at has
# coverage there.
#
# Output widths straddle dram_alignment deliberately: 20032*4 is a 32B multiple, 20004*4 is 16B- but
# not 32B-aligned. The second separates a budget page charged at 16B from the dram_alignment pitch
# the circular buffer is actually built at.
#
# Input widths stay DRAM-aligned so these remain on the codegen path. A ragged input stick is
# perf-demoted onto the staging path and routed native, and native pad cannot allocate its own
# circular buffers at these widths -- a pre-existing native limit this port neither causes nor fixes.
@pytest.mark.parametrize(
    "h, w, padding, torch_padding",
    [
        (4, 20000, ((0, 0), (0, 0), (0, 1), (0, 32)), (0, 32, 0, 1)),
        (4, 20000, ((0, 0), (0, 0), (0, 1), (0, 4)), (0, 4, 0, 1)),
        (2, 40000, ((0, 0), (0, 0), (0, 1), (0, 8)), (0, 8, 0, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_pad_codegen_rm_wide_stick_l1_clamp(device, h, w, padding, torch_padding, dtype):
    torch.manual_seed(0)

    torch_input_tensor = random_torch_tensor(dtype, (1, 1, h, w))
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)
    output_tensor = ttnn.to_torch(ttnn.pad(input_tensor, padding=padding, value=0, implementation="codegen"))

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


# supported_by_codegen must reject a partial input tile: a tile-page copy moves that tile verbatim
# and nothing fills its remainder lanes with the pad value. The generator only reaches this case by
# first running build_fill_partial_tile, which is not part of this port's kernel set, so `auto` has
# to route these to native and still produce the right answer.
@pytest.mark.parametrize(
    "shape, padding, torch_padding",
    [
        ((1, 1, 40, 64), ((0, 0), (0, 0), (0, 32), (0, 32)), (0, 32, 0, 32)),
        ((1, 1, 64, 40), ((0, 0), (0, 0), (0, 32), (0, 32)), (0, 32, 0, 32)),
        ((1, 1, 40, 40), ((0, 0), (0, 0), (0, 64), (0, 64)), (0, 64, 0, 64)),
    ],
)
def test_pad_codegen_tile_partial_input_routes_native(device, shape, padding, torch_padding):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape).bfloat16().float()
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    output_tensor = ttnn.to_torch(ttnn.pad(input_tensor, padding=padding, value=0))

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)
