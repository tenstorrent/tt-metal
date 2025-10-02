import torch
import ttnn

w = 2


def mlir_const_eval_simulation():
    """
    The result should be the same [w] tensor 7 times.
    """
    print("Running MLIR const_eval simulation...")

    # Get device
    device = ttnn.open_device(device_id=0)

    ttnn_to_torch_dtype = {
        ttnn.float32: torch.float32,
        ttnn.bfloat16: torch.bfloat16,
    }
    dtype = ttnn.float32
    # dtype = ttnn.bfloat16

    try:
        # Create a scalar input tensor (similar to tensor<f32>)
        scalar_value = 0.00001
        torch_input = torch.tensor(scalar_value, dtype=ttnn_to_torch_dtype[dtype])

        # Convert to ttnn tensor
        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=dtype,
            device=device,
            # layout=ttnn.TILE_LAYOUT,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        check_input = ttnn.to_torch(input_tensor)
        print(check_input)

        # Step 1: Reshape scalar to [1, 1, 1, 1]
        reshaped = ttnn.reshape(input_tensor, [1, 1, 1, 1])

        repeated = ttnn.repeat(reshaped, (1, 1, 1, w))

        result = ttnn.to_torch(repeated)

        # Expected result: tensor filled with the scalar value repeated w times
        expected = torch.full([1, 1, 1, w], scalar_value, dtype=ttnn_to_torch_dtype[dtype])

        print("Result:", result)
        print("Expected:", expected)
        assert torch.allclose(result, expected), "The result does not match the expected output."
        print("Test passed: The result matches the expected output.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    mlir_const_eval_simulation()
