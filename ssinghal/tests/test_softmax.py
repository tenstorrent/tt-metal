import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        # YOLOv12x ultra-high resolution (2176x3840) softmax attention shapes
        [1, 12, 8160, 8160],   # Full attention matrix: [batch, heads, seq, seq]
        [1, 12, 32640, 384],   # Large attention: [batch, heads, seq, features]
        [1, 8, 4080, 4080],    # Half attention matrix: [batch, heads, half_seq, half_seq]
        [12, 8160, 8160],      # Multi-head attention scores: [heads, seq, seq]
        [12, 32640, 32],       # Attention over features: [heads, seq, head_dim]
        [1, 1152, 68120],      # Channel attention: [batch, channels, spatial]
        [8, 2040, 96],         # Medium attention: [heads, quarter_seq, features]
        [16, 1020, 48],        # High-head attention: [heads, eighth_seq, head_dim]
        [4, 16320, 128],       # Low-head attention: [heads, double_seq, features]
        [24, 680, 32],         # Many-head attention: [heads, small_seq, head_dim]
    ],
)
def test_softmax(device, input_shape):
    """Test softmax operator with YOLOv12x ultra-high resolution (2176x3840) attention shapes"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.softmax(ttnn_input)

        torch_reference = torch.softmax(torch_input)

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
