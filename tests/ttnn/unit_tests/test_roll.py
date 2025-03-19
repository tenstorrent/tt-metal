import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_tensor, shifts, dim",
    [
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 1, 0),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], -1, 0),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 0, 0),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 4, 0),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 5, 0),
        ([[1, 1], [1, 1], [1, 1], [1, 1]], 1, 1),
        (
            [
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
                [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29], [30, 31, 32, 33, 34], [35, 36, 37, 38, 39]],
                [[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]],
                [[60, 61, 62, 63, 64], [65, 66, 67, 68, 69], [70, 71, 72, 73, 74], [75, 76, 77, 78, 79]],
                [[80, 81, 82, 83, 84], [85, 86, 87, 88, 89], [90, 91, 92, 93, 94], [95, 96, 97, 98, 99]],
                [
                    [100, 101, 102, 103, 104],
                    [105, 106, 107, 108, 109],
                    [110, 111, 112, 113, 114],
                    [115, 116, 117, 118, 119],
                ],
            ],
            [-2, -1],
            [0, 1],
        ),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], [1, 0], [0, 1]),
        ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], [1, 0, -2], [0, 1, 2]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]], [0, -1, -2], [0, 1, 2]),
        (
            [
                [
                    [
                        [[[[[5, 6], [7, 8]], [[9, 10], [11, 12]]], [[[13, 14], [15, 16]], [[17, 18], [19, 20]]]]],
                        [[[[[21, 22], [23, 24]], [[25, 26], [27, 28]]], [[[29, 30], [31, 32]], [[33, 34], [35, 36]]]]],
                    ]
                ]
            ],
            [0, 1, 2],
            [1, 4, 7],
        ),
    ],
)
def test_roll(device, input_tensor, shifts, dim):
    tensor = torch.tensor(input_tensor, dtype=torch.float32)
    ttnn_tensor = ttnn.from_torch(tensor)
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pytorch_out = torch.roll(tensor, shifts, dim)
    ttnn_out = ttnn.roll(ttnn_tensor, shifts, dim)
    ttnn_result_torch = ttnn.to_torch(ttnn_out)
    print(pytorch_out)
    print(ttnn_result_torch)
    assert_with_pcc(pytorch_out, ttnn_result_torch)
    # print(pytorch_out)
    # print(ttnn_result_torch)
    assert (
        pytorch_out.shape == ttnn_result_torch.shape
    ), f"Shapes don't match: {pytorch_out.shape} vs {ttnn_result_torch.shape}"
