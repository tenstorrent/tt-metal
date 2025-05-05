import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_div(device):
    x = torch.load("tests/ttnn/unit_tests/dumps/torch_1.pt")
    tt_x = torch.load("tests/ttnn/unit_tests/dumps/ttnn_1.pt")

    assert_with_pcc(x, tt_x, 0.9999)

    y = torch.load("tests/ttnn/unit_tests/dumps/torch_2.pt")
    tt_y = torch.load("tests/ttnn/unit_tests/dumps/ttnn_2.pt")

    assert_with_pcc(y, tt_y, 0.9999)

    x = torch.load("tests/ttnn/unit_tests/dumps/torch_1.pt")
    y = torch.load("tests/ttnn/unit_tests/dumps/torch_2.pt")

    tt_x = torch.load("tests/ttnn/unit_tests/dumps/ttnn_1.pt")
    tt_y = torch.load("tests/ttnn/unit_tests/dumps/ttnn_2.pt")
    tt_x = ttnn.from_torch(tt_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_y = ttnn.from_torch(tt_y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_output = x / y
    ttnn_output = ttnn.div(tt_x, tt_y)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(ttnn_output, torch_output, 0.9999)


def test_div_torch():
    x = torch.load("tests/ttnn/unit_tests/dumps/torch_1.pt")
    y = torch.load("tests/ttnn/unit_tests/dumps/torch_2.pt")

    tt_x = torch.load("tests/ttnn/unit_tests/dumps/ttnn_1.pt")
    tt_y = torch.load("tests/ttnn/unit_tests/dumps/ttnn_2.pt")

    torch_output = x / y
    ttnn_output = tt_x / tt_y

    assert_with_pcc(ttnn_output, torch_output, 0.9999)
