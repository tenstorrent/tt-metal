import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape,split_size,dim",
    [
        # YOLOv12x ultra-high resolution (2176x3840) split operations from graph.py
        ([1, 192, 1088, 1920], 96, 1),  # C3k2 layer: split 192 channels into 2x96
        ([1, 384, 544, 960], 192, 1),  # C3k2 layer: split 384 channels into 2x192
        ([1, 768, 272, 480], 384, 1),  # C3k2 layer: split 768 channels into 2x384
        ([1, 4, 171360], 2, 1),  # Detection head: split 4 channels into 2x2
        # Additional YOLOv12x feature map split patterns
        ([1, 96, 2176, 3840], 48, 1),  # Ultra-high-res feature maps
        ([1, 192, 1088, 1920], 96, 1),  # High-res feature maps
        ([1, 384, 544, 960], 192, 1),  # Medium-res feature maps
        ([1, 768, 136, 240], 384, 1),  # Lower-res feature maps
        ([1, 1152, 136, 120], 576, 1),  # Attention feature maps
        # Flattened attention and detection head splits
        ([1, 32640, 384], 16320, 1),  # Large flattened attention split
        ([1, 8160, 768], 4080, 1),  # Medium flattened feature split
        ([1, 2040, 512], 1020, 1),  # Smaller feature split
        # Various split dimensions for robustness
        ([1, 3, 2176, 3840], 1, 1),  # Split input channels
        ([1, 96, 1088, 1920], 24, 1),  # Quarter channel splits
        ([1, 768, 68, 120], 256, 1),  # Third channel splits
        # 8K YOLOv12x ultra-high resolution split operations
        ([1, 3, 4320, 7680], 1, 1),  # Split 8K input channels
        ([1, 96, 2160, 3840], 48, 1),  # Split 8K first conv channels
        ([1, 192, 1080, 1920], 96, 1),  # Split 8K second conv channels
        ([1, 384, 540, 960], 192, 1),  # Split 8K third conv channels
        ([1, 768, 270, 480], 384, 1),  # Split 8K fourth conv channels
        ([1, 1536, 135, 240], 768, 1),  # Split 8K fifth conv channels
    ],
)
def test_split(device, input_shape, split_size, dim):
    """Test split operator with YOLOv12x ultra-high resolution (2176x3840) input shapes"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.split(ttnn_input, split_size, dim)

        torch_reference = torch.split(torch_input, split_size, dim)

        # Convert ttnn outputs back to torch for comparison
        ttnn_torch_outputs = [ttnn.to_torch(output) for output in ttnn_output]

        # Check each split output
        for ttnn_torch_output, torch_ref_output in zip(ttnn_torch_outputs, torch_reference):
            if len(input_shape) == 4:
                ttnn_torch_output = ttnn_torch_output.permute(0, 3, 1, 2)

            passed, pcc_message = check_with_pcc_without_tensor_printout(ttnn_torch_output, torch_ref_output)
            assert passed, f"PCC check failed for split output: {pcc_message}"

        print(f"✅ Split test passed for shape {input_shape}, split_size={split_size}, dim={dim}")

    except Exception as e:
        if "OutOfMemoryError" in str(e) or "out of memory" in str(e).lower():
            print(f"⚠️  OOM for shape {input_shape}, split_size={split_size}, dim={dim}: {e}")
            pytest.skip(f"OOM: {e}")
        else:
            print(f"❌ Error for shape {input_shape}, split_size={split_size}, dim={dim}: {e}")
            raise
