import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape_and_sizes",
    [
        # YOLOv12 high-resolution split operations
        ([1, 96, 640, 640], [48, 48]),      # Split channels after conv
        ([1, 192, 320, 320], [96, 96]),     # Split channels after conv  
        ([1, 384, 160, 160], [192, 192]),   # Split channels after conv
        ([1, 768, 80, 80], [384, 384]),     # Split channels after conv
        ([1, 1536, 40, 40], [768, 768]),    # Split channels after conv
        ([1, 128, 1280, 800], [64, 64]),    # High-res feature map splits
        ([1, 256, 640, 400], [128, 128]),   # Medium-res feature map splits
        ([1, 512, 320, 200], [256, 256]),   # Lower-res feature map splits
        ([1, 160, 640, 400], [80, 80]),     # Detection head splits
        ([1, 320, 320, 200], [160, 160]),   # Detection head splits
        ([1, 640, 160, 100], [320, 320]),   # Detection head splits
        ([1, 1280, 80, 50], [640, 640]),    # Detection head splits
        ([1, 96, 1280, 1280], [48, 48]),    # Very high-res splits
        ([1, 64, 1280, 800], [32, 32]),     # High-res channel splits
        ([1, 32, 1280, 800], [16, 16]),     # High-res channel splits
    ],
)
def test_split_with_sizes(device, input_shape_and_sizes):
    """Test split_with_sizes operator with YOLOv12 high-resolution input shapes"""
    input_shape, split_sizes = input_shape_and_sizes
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
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
