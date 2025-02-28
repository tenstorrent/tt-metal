# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


def run_topk_test(N, C, H, W, k, largest, dtype, device, sub_core_grids=None):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    pyt_topk_values, pyt_topk_indices = torch.topk(input, k, dim=-1, largest=largest, sorted=True)

    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    if sub_core_grids is not None:
        print(f"unit test: sub_core_grids: {sub_core_grids}")
        try:
            ttnn_topk_values, ttnn_topk_indices = ttnn.topk(
                ttnn_input, k, dim=-1, largest=largest, sorted=True, sub_core_grids=sub_core_grids
            )
        except Exception as e:
            print(f"unit test: sub_core_grids: {sub_core_grids}")
            raise e
    else:
        print(f"unit test: sub_core_grids: {sub_core_grids}")
        ttnn_topk_values, ttnn_topk_indices = ttnn.topk(ttnn_input, k, dim=-1, largest=largest, sorted=True)

    print(f"topk done")
    print(f"unit test: ttnn_topk_values: {ttnn_topk_values}")
    print(f"unit test: ttnn_topk_indices: {ttnn_topk_indices}")

    assert list(ttnn_topk_values.padded_shape) == [N, C, H, k]
    assert list(ttnn_topk_indices.padded_shape) == [N, C, H, k]

    ttnn_torch_values = ttnn.to_torch(ttnn_topk_values)
    ttnn_torch_indices = ttnn.to_torch(ttnn_topk_indices).to(torch.int64)

    if dtype == ttnn.bfloat8_b:
        pcc_values = 0.99
    else:
        pcc_values = 1.0

    # pcc is not a good measure for the raw indices
    # if index 49 and index 8 are tied, the order of the indices can be different
    # but the values associated with the indices should be the same
    # if index 7 and 8 are tied, but swapped, the pcc will be better than if index 49 and 8 are tied but swapped
    # rounding may also cause more ties than expected
    # the bigger we get, the tighter the distribution of the top 32 elements, so the pcc will be worse as stability/rounding will cause more ties
    # use cosine similarity on the gathered indices as this will show the top elements are all about the same
    ttnn_torch_gather_from_indices = torch.gather(input, -1, ttnn_torch_indices.to(torch.int64))
    cosine = torch.nn.CosineSimilarity(dim=-1)
    ttnn_torch_cosine = torch.mean(cosine(pyt_topk_values, ttnn_torch_gather_from_indices))

    print(f"unit test: ttnn_torch_cosine: {ttnn_torch_cosine}")

    assert ttnn_torch_cosine > 0.99, "Cosine similarity between topk values and gather from indices is less than 0.99"

    print(f"unit test: pyt_topk_values: {pyt_topk_values}")
    print(f"unit test: ttnn_torch_values: {ttnn_torch_values}")

    breakpoint()
    assert_with_pcc(pyt_topk_values, ttnn_torch_values, pcc_values)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        # ttnn.float32, top bits in float32 get cut off somewhere, LLK does not work for this
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
        # "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, k,",
    (
        (1, 1, 32, 64, 32),
        (1, 1, 32, 8192, 32),
        (1, 1, 2048, 64, 32),
        (1, 1, 32, 32768, 32),
        (1, 1, 8192, 64, 32),
    ),
)
@pytest.mark.parametrize("largest", (True, False))
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_topk(N, C, H, W, k, largest, dtype, device):
    run_topk_test(N, C, H, W, k, largest, dtype, device)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b,),
    ids=[
        "BFLOAT8_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, k,",
    (
        (1, 1, 32, 16 * 1024, 32),
        # (1, 1, 32, 64, 32),
    ),
)
@pytest.mark.parametrize(
    "sub_core_grids",
    [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 8)),
            ]
        ),
    ],
)
@pytest.mark.parametrize("largest", (True,))
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_topk_sub_core_grids(N, C, H, W, k, largest, dtype, mesh_device, sub_core_grids):
    device = mesh_device.get_device(mesh_device.get_device_ids()[0])
    print(f"unit test: sub_core_grids: {sub_core_grids}")
    run_topk_test(N, C, H, W, k, largest, dtype, device, sub_core_grids=sub_core_grids)
