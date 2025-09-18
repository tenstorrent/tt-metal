import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("height", [1])
@pytest.mark.parametrize("width", [1])
def test_add_not_okay(device, height, width):
    torch_input = torch.tensor(16777216, dtype=torch.bfloat16)
    torch_other = torch.tensor(-8192, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_other = ttnn.from_torch(torch_other, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.add(tt_input, tt_other, dtype=ttnn.float32)
    actual_output = ttnn.to_torch(tt_output)

    print("actual_output ", actual_output)
    assert actual_output == 16769024  # 16777216-8192=16769024(=19 bits)


@pytest.mark.parametrize("height", [1])
@pytest.mark.parametrize("width", [1])
def test_copy_19_bis_okay(device, height, width):
    torch_input = torch.tensor(16769024, dtype=torch.float32)
    torch_other = torch.tensor(0, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_other = ttnn.from_torch(torch_other, layout=ttnn.TILE_LAYOUT, device=device)
    print("tt_input ", tt_input)
    print("tt_other ", tt_other)
    tt_output = ttnn.add(tt_input, tt_other, dtype=ttnn.float32)
    print("tt_output ", tt_output)  # 16769024(=19 bits)
