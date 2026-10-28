# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.topk on Quasar: the BFLOAT16 / UINT16 path that Quasar supports, and the inputs it rejects."""

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import is_quasar


def check_topk(torch_input, ttnn_values, ttnn_indices, k, largest):
    ref_values, _ = torch.topk(torch_input, k, dim=-1, largest=largest, sorted=True)
    values = ttnn.to_torch(ttnn_values)
    indices = ttnn.to_torch(ttnn_indices, dtype=torch.int64)
    assert torch.equal(values, ref_values)
    # Equal values can come in any order, so check that each index points at its value.
    assert torch.equal(torch.gather(torch_input, -1, indices), values)


@pytest.mark.parametrize(
    "shape, k",
    [
        ((1, 1, 32, 64), 1),
        ((1, 1, 32, 64), 32),
        ((1, 1, 64, 256), 10),
        # Eligible for the multi-core factory on WH/BH (see topk_multicore_structurally_eligible), which
        # Quasar must not select.
        ((1, 2, 32, 1024), 64),
        # k above 64 needs more than 2 output tiles, so the sort merges into a growing set of tiles.
        ((1, 1, 32, 256), 96),
    ],
)
@pytest.mark.parametrize("largest", [True, False])
def test_topk_bfloat16(shape, k, largest, device):
    # randn gives negative and positive values, so the sort compares values of both signs.
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True)

    check_topk(torch_input, ttnn_values, ttnn_indices, k, largest)


def test_topk_indices_above_int16_max(device):
    # On Quasar the 16-bit indices travel in an Int16 buffer. Indices from 32768 to 65535 have bit 15
    # set, so they must reach the output as raw bits, without any signed conversion.
    torch.manual_seed(0)
    W = 40960
    k = 32
    torch_input = torch.rand((1, 1, 32, W), dtype=torch.bfloat16) - 0.5
    positions = torch.arange(32768, W, (W - 32768) // k)[:k]
    torch_input[..., positions] = torch.arange(1, k + 1, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=True, sorted=True)

    check_topk(torch_input, ttnn_values, ttnn_indices, k, largest=True)
    indices = ttnn.to_torch(ttnn_indices, dtype=torch.int64)
    assert torch.equal(indices, positions.flip(0).expand(1, 1, 32, k))


def test_topk_several_rows_per_core(device):
    # One core gets all 4 tile rows, so the kernels run several rows through the same buffers.
    torch.manual_seed(0)
    k = 8
    torch_input = torch.randn((1, 1, 128, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    one_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

    ttnn_values, ttnn_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=True, sorted=True, sub_core_grids=one_core)

    check_topk(torch_input, ttnn_values, ttnn_indices, k, largest=True)


def test_topk_uint16_indices_tensor(device):
    # The caller's index labels are read instead of generated. Labels at or above 32768 check that
    # they pass through the 16-bit index buffer without change.
    torch.manual_seed(0)
    k = 32
    shape = (1, 1, 32, 64)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    labels = (65535 - torch.arange(shape[-1])).expand(shape)
    indices_tensor = ttnn.from_torch(labels.to(torch.int32), ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_values, ttnn_labels = ttnn.topk(
        ttnn_input, k, dim=-1, largest=True, sorted=True, indices_tensor=indices_tensor
    )

    ref_values, _ = torch.topk(torch_input, k, dim=-1, largest=True, sorted=True)
    values = ttnn.to_torch(ttnn_values)
    positions = 65535 - ttnn.to_torch(ttnn_labels, dtype=torch.int64)
    assert torch.equal(values, ref_values)
    assert torch.equal(torch.gather(torch_input, -1, positions), values)


@pytest.mark.skipif(not is_quasar(), reason="Quasar-only validation")
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat8_b])
def test_topk_rejects_input_dtype(dtype, device, expect_error):
    torch_input = torch.randn((1, 1, 32, 64), dtype=torch.float32)
    ttnn_input = ttnn.from_torch(torch_input, dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "TopK on Quasar supports only BFLOAT16 input"):
        ttnn.topk(ttnn_input, 32, dim=-1)


@pytest.mark.skipif(not is_quasar(), reason="Quasar-only validation")
def test_topk_rejects_32bit_indices(device, expect_error):
    shape = (1, 1, 32, 64)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    indices_tensor = ttnn.from_torch(
        torch.arange(shape[-1]).expand(shape).to(torch.int32), ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
    )

    with expect_error(RuntimeError, "TopK on Quasar supports only UINT16 indices"):
        ttnn.topk(ttnn_input, 32, dim=-1, indices_tensor=indices_tensor)
