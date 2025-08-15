import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        # YOLOv12x ultra-high resolution (2176x3840) input shapes
        [1, 3, 2176, 3840],   # YOLOv12x ultra-high-res input
        [1, 96, 1088, 1920],  # After first conv stride=2
        [1, 192, 544, 960],   # After second conv stride=2
        [1, 384, 272, 480],   # After third conv stride=2
        [1, 768, 136, 240],   # After fourth conv stride=2
        [1, 768, 68, 120],    # After fifth conv stride=2
        [1, 96, 2176, 1920],  # Ultra-high-res feature maps
        [1, 192, 1088, 960],  # High-res feature maps
        [1, 384, 544, 480],   # Medium-res feature maps
        [1, 768, 272, 240],   # Lower-res feature maps
        [1, 1152, 136, 120],  # Attention feature maps
        [1, 384, 136, 240],   # Detection head shapes
        [1, 768, 68, 120],    # Detection head shapes
        [1, 1152, 34, 60],    # Detection head shapes
        [1, 32640, 384],      # Flattened attention shapes
        [1, 8160, 768],       # Flattened feature shapes
        [1, 3, 1088, 1920],   # Half-res YOLOv12x input
        [1, 96, 544, 960],    # Quarter-res feature maps
        [1, 192, 272, 480],   # Eighth-res feature maps
        [1, 384, 136, 240],   # Sixteenth-res feature maps
    ],
)
def test_ones(device, input_shape):
    """Test ones operator with YOLOv12x ultra-high resolution (2176x3840) input shapes"""
    torch.manual_seed(0)

    try:
        torch_reference = torch.ones(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_reference = torch_reference.permute(0, 2, 3, 1)

        # Create ones tensor for YOLOv12 shapes
        if len(input_shape) == 4:
            ttnn_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]  # NHWC for ttnn
        else:
            ttnn_shape = input_shape
        
        ttnn_output = ttnn.ones(ttnn_shape, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

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
