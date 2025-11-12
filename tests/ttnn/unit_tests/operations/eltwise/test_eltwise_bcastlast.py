import pytest
import torch
import ttnn


@pytest.mark.parametrize("N", [1, 3, 9])
def test_eltwise_add_with_row_values(device, N):
    torch.manual_seed(0)

    # Create N tiles filled with ones for SrcA
    # Each tile is 32x32 (standard tile size)
    tile_h, tile_w = 32, 32
    torch_input_tensor_a = torch.ones((1, 1, N * tile_h, tile_w), dtype=torch.bfloat16)

    # Create N tiles where each tile has constant value in row i = value(i) for SrcB
    torch_input_tensor_b = torch.zeros((1, 1, N * tile_h, tile_w), dtype=torch.bfloat16)
    for i in range(N * tile_h):
        torch_input_tensor_b[0, 0, i, :] = float(i)

    # Convert to ttnn tensors
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Run eltwise add
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Compute golden output
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    # Convert back to torch and run PCC check
    output_tensor = ttnn.to_torch(output_tensor)
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999
