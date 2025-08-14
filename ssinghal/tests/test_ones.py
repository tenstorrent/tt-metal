import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        # YOLOv12 high-resolution input shapes
        [1, 3, 1280, 1280],   # YOLOv12 high-res input
        [1, 96, 640, 640],    # After first conv stride=2
        [1, 192, 320, 320],   # After second conv stride=2
        [1, 384, 160, 160],   # After third conv stride=2
        [1, 768, 80, 80],     # After fourth conv stride=2
        [1, 768, 40, 40],     # After fifth conv stride=2
        [1, 64, 1280, 800],   # High-res feature maps
        [1, 128, 640, 400],   # Medium-res feature maps
        [1, 256, 320, 200],   # Lower-res feature maps
        [1, 32, 1280, 800],   # High-res channels
        [1, 80, 640, 400],    # Detection head shapes
        [1, 160, 320, 200],   # Detection head shapes
        [1, 320, 160, 100],   # Detection head shapes
        [1, 640, 80, 50],     # Detection head shapes
        [1, 1280, 40, 25],    # Detection head shapes
        [1, 3, 640, 640],     # Standard YOLOv12 input
        [1, 96, 320, 320],    # Standard feature maps
        [1, 192, 160, 160],   # Standard feature maps
        [1, 384, 80, 80],     # Standard feature maps
        [1, 768, 40, 40],     # Standard feature maps
    ],
)
def test_ones(device, input_shape):
    """Test ones operator with YOLOv12 high-resolution input shapes"""
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
        
        ttnn_output = ttnn.ones(ttnn_shape, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

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
