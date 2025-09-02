import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param([(1, 1, 1), (8, 16, 32)], id="broadcast_lhs_1"),
        pytest.param([(1, 1, 32), (8, 16, 32)], id="broadcast_lhs_2"),
        pytest.param([(1, 16, 32), (8, 16, 32)], id="broadcast_lhs_3"),
        pytest.param([(8, 16, 32), (1, 1, 1)], id="broadcast_rhs_1"),
        pytest.param([(8, 16, 32), (1, 1, 32)], id="broadcast_rhs_2"),
        pytest.param([(8, 16, 32), (1, 16, 32)], id="broadcast_rhs_3"),
        pytest.param([(8, 16, 1), (1, 1, 32)], id="broadcast_both_1"),
        pytest.param([(1, 1, 32), (8, 16, 1)], id="broadcast_both_2"),
        pytest.param([(8, 1, 32), (8, 16, 1)], id="broadcast_both_3"),
        pytest.param([(8, 16, 1), (8, 1, 32)], id="broadcast_both_4"),
    ],
)
def test_lt_implicit_broadcast(device, shapes):
    torch.manual_seed(0)

    min_int = torch.iinfo(torch.int32).min
    max_int = torch.iinfo(torch.int32).max
    torch_input_tensor_a = torch.randint(low=min_int, high=max_int, size=shapes[0], dtype=torch.int32)
    torch_input_tensor_b = torch.randint(low=min_int, high=max_int, size=shapes[1], dtype=torch.int32)
    # torch_input_tensor_a = torch.tensor([209652396, -698378168])
    # torch_input_tensor_b = torch.tensor([-2060646285, -1347192322])
    # torch_input_tensor_b = -2060646285
    # print("input_a: ", torch_input_tensor_a)
    # print("input_b: ", torch_input_tensor_b)

    torch_output_tensor = torch.lt(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.lt(input_tensor_a, input_tensor_b, use_legacy=False)
    # print(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # print(torch_output_tensor)
    # print(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    torch.equal()
