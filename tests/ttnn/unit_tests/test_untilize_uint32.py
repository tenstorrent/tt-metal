import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8, 32])
@pytest.mark.parametrize("sentence_size", [128, 256, 258, 512, 1024])
# @pytest.mark.parametrize(
#     "vocabulary_size", [512, 30522, 2048]
# )
def test_untilize_uint32(device, batch_size, sentence_size):
    torch_input_tensor = torch.randint(0, 10, (batch_size, sentence_size), dtype=torch.int32)
    ttnn_input = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    output_tt = ttnn.to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT)
    output_torch = ttnn.to_torch(output_tt)

    assert_with_pcc(torch_input_tensor, output_torch)
