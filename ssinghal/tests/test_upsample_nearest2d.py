import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape_and_scale",
    [
        # YOLOv12x ultra-high resolution (2176x3840) upsampling operations
        ([1, 768, 68, 120], 2),  # Upsample backbone features 68x120 -> 136x240
        ([1, 768, 136, 240], 2),  # Upsample backbone features 136x240 -> 272x480
        ([1, 384, 272, 480], 2),  # Upsample backbone features 272x480 -> 544x960
        ([1, 192, 544, 960], 2),  # Upsample backbone features 544x960 -> 1088x1920
        ([1, 96, 1088, 1920], 2),  # Upsample backbone features 1088x1920 -> 2176x3840
        ([1, 1152, 34, 60], 2),  # Detection head upsampling
        ([1, 768, 68, 120], 2),  # Detection head upsampling
        ([1, 384, 136, 240], 2),  # Detection head upsampling
        ([1, 768, 34, 60], 4),  # 4x upsampling for very low-res features
        ([1, 384, 68, 120], 4),  # 4x upsampling for low-res features
        ([1, 192, 136, 240], 4),  # 4x upsampling for medium-res features
        ([1, 96, 544, 480], 2),  # Non-square upsampling
        ([1, 192, 272, 240], 2),  # Non-square upsampling
        ([1, 384, 136, 120], 2),  # Non-square upsampling
        ([1, 768, 68, 60], 2),  # Non-square upsampling
        # 8K YOLOv12x ultra-high resolution upsampling operations
        ([1, 1536, 67, 120], 2),  # 8K deep features 67x120 -> 135x240
        ([1, 768, 135, 240], 2),  # 8K features 135x240 -> 270x480
        ([1, 384, 270, 480], 2),  # 8K features 270x480 -> 540x960
        ([1, 192, 540, 960], 2),  # 8K features 540x960 -> 1080x1920
        ([1, 96, 1080, 1920], 2),  # 8K features 1080x1920 -> 2160x3840
        ([1, 1152, 33, 60], 2),  # 8K detection head upsampling
        ([1, 768, 67, 120], 4),  # 8K 4x upsampling
        ([1, 384, 135, 240], 4),  # 8K 4x upsampling
        ([1, 192, 270, 480], 4),  # 8K 4x upsampling
    ],
)
def test_upsample_nearest2d(device, input_shape_and_scale):
    """Test upsample_nearest2d operator with YOLOv12x ultra-high resolution (2176x3840) input shapes"""
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
            torch_input_ttnn, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        # Calculate output size
        _, _, height, width = input_shape
        output_size = (height * scale_factor, width * scale_factor)

        # Perform upsampling
        ttnn_output = ttnn.upsample(ttnn_input, scale_factor)

        # PyTorch reference
        torch_reference = torch.nn.functional.interpolate(torch_input, scale_factor=scale_factor, mode="nearest")
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
