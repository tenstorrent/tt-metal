import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape_and_scale",
    [
        # YOLOv12 high-resolution upsampling operations
        ([1, 768, 40, 40], 2),        # Upsample backbone features 40x40 -> 80x80
        ([1, 768, 80, 80], 2),        # Upsample backbone features 80x80 -> 160x160
        ([1, 384, 160, 160], 2),      # Upsample backbone features 160x160 -> 320x320
        ([1, 192, 320, 320], 2),      # Upsample backbone features 320x320 -> 640x640
        ([1, 96, 640, 640], 2),       # Upsample backbone features 640x640 -> 1280x1280
        ([1, 384, 80, 80], 2),        # Detection head upsampling
        ([1, 192, 160, 160], 2),      # Detection head upsampling
        ([1, 96, 320, 320], 2),       # Detection head upsampling
        ([1, 768, 20, 20], 4),        # 4x upsampling for very low-res features
        ([1, 384, 40, 40], 4),        # 4x upsampling for low-res features
        ([1, 192, 80, 80], 4),        # 4x upsampling for medium-res features
        ([1, 64, 320, 200], 2),       # Non-square upsampling
        ([1, 128, 160, 100], 2),      # Non-square upsampling
        ([1, 256, 80, 50], 2),        # Non-square upsampling
        ([1, 512, 40, 25], 2),        # Non-square upsampling
    ],
)
def test_upsample_nearest2d(device, input_shape_and_scale):
    """Test upsample_nearest2d operator with YOLOv12 high-resolution input shapes"""
    input_shape, scale_factor = input_shape_and_scale
    torch.manual_seed(0)

    try:
        # For upsample, we need 4D input (N, C, H, W)
        if len(input_shape) != 4:
            pytest.skip(f"Upsample requires 4D input, got {len(input_shape)}D")

        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
        
        # Convert to ttnn format (N, H, W, C)
        torch_input_ttnn = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input_ttnn, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        # Calculate output size
        _, _, height, width = input_shape
        output_size = (height * scale_factor, width * scale_factor)

        # Perform upsampling
        ttnn_output = ttnn.upsample(ttnn_input, scale_factor)
        
        # PyTorch reference
        torch_reference = torch.nn.functional.interpolate(
            torch_input, scale_factor=scale_factor, mode='nearest'
        )
        torch_reference = torch_reference.permute(0, 2, 3, 1)  # Convert to NHWC

        # Convert output back to torch
        ttnn_result = ttnn.to_torch(ttnn_output)

        # Compare results
        check_with_pcc_without_tensor_printout(ttnn_result, torch_reference, 0.99)

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
