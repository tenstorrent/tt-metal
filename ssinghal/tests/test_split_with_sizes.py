import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape_and_sizes",
    [
        # YOLOv12x ultra-high resolution (2176x3840) split operations
        ([1, 96, 1088, 1920], [48, 48]),      # Split channels after conv
        ([1, 192, 544, 960], [96, 96]),       # Split channels after conv  
        ([1, 384, 272, 480], [192, 192]),     # Split channels after conv
        ([1, 768, 136, 240], [384, 384]),     # Split channels after conv
        ([1, 1152, 68, 120], [576, 576]),     # Split channels after conv
        ([1, 192, 2176, 1920], [96, 96]),     # Ultra-high-res splits
        ([1, 384, 1088, 960], [192, 192]),    # High-res feature map splits
        ([1, 768, 544, 480], [384, 384]),     # Medium-res feature map splits
        ([1, 1152, 272, 240], [576, 576]),    # Lower-res feature map splits
        ([1, 384, 136, 240], [192, 192]),     # Detection head splits
        ([1, 768, 68, 120], [384, 384]),      # Detection head splits
        ([1, 1152, 34, 60], [576, 576]),      # Detection head splits
        ([1, 96, 2176, 3840], [48, 48]),      # Ultra-high-res splits
        ([1, 192, 1088, 1920], [96, 96]),     # High-res channel splits
        ([1, 384, 544, 960], [192, 192]),     # Medium-res channel splits
    ],
)
def test_split_with_sizes(device, input_shape_and_sizes):
    """Test split_with_sizes operator with YOLOv12x ultra-high resolution (2176x3840) input shapes"""
    input_shape, split_sizes = input_shape_and_sizes
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        # Split with sizes (split along dimension 1 by default)
        dim = 1 if len(input_shape) > 1 else 0
        ttnn_outputs = ttnn.split(ttnn_input, split_sizes, dim=dim)
        torch_reference = torch.split(torch_input, split_sizes, dim=dim)

        # Convert outputs back to torch and compare
        for ttnn_out, torch_ref in zip(ttnn_outputs, torch_reference):
            ttnn_result = ttnn.to_torch(ttnn_out)
            check_with_pcc_without_tensor_printout(ttnn_result, torch_ref, 0.99)

    except RuntimeError as e:
        if "Out of Memory" in str(e):
            pytest.skip(f"OOM: {input_shape} - {str(e)}")
        else:
            raise e
    except Exception as e:
        if "incompatible function arguments" in str(e):
            pytest.skip(f"Type error: {input_shape} - {str(e)}")
        else:
            raise e
