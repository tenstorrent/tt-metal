import torch
import ttnn


def reproduce_program_cache_concat_issue():
    # Initialize device
    device = ttnn.open_device(device_id=0)

    # Enable program cache (this is key!)
    # ttnn.enable_program_cache(device)

    # Test parameters from the failing test
    shapes = ((1, 1, 64, 64), (1, 1, 128, 64))
    dim = -2
    layout = ttnn.TILE_LAYOUT
    dtype = ttnn.bfloat16

    input_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    output_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    def run_concat_once():
        print(f"Running concat with shapes: {shapes}, dim: {dim}")

        # Create input tensors
        inputs = []
        tt_inputs = []
        for i, shape in enumerate(shapes):
            tensor_shape = torch.Size(shape)
            inputs.append(torch.rand(tensor_shape).to(torch.bfloat16))
            tt_inputs.append(ttnn.Tensor(inputs[i], dtype).to(layout).to(device, input_mem_config))

        # Expected result
        expected = torch.concat(inputs, dim)

        # Run ttnn concat
        result = ttnn.concat(tt_inputs, dim, memory_config=output_mem_config)

        # Convert back and compare
        actual = result.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)

        return torch.allclose(expected, actual, rtol=1e-2, atol=1e-2)

    print("=== First concat run (should populate program cache) ===")
    success1 = run_concat_once()
    print(f"First run success: {success1}")

    print("\n=== Creating dummy tensor (like in test) ===")
    tmp = ttnn.zeros([1, 256, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    print("Dummy tensor created")

    print("\n=== Second concat run (should use cached program) ===")
    success2 = run_concat_once()  # This should crash and show the traceback
    print(f"Second run success: {success2}")

    print(f"\nOverall test result: {success1 and success2}")


def test_large_tensor_count():
    """Test with large tensor count to trigger the original error"""
    device = ttnn.open_device(device_id=0)
    # ttnn.enable_program_cache(device)

    # Test with 55 tensors to trigger the original 278 argument error
    num_tensors = 55
    base_shape = (1, 1, 32, 32)
    dim = 3  # concat on width

    print(f"=== Testing with {num_tensors} tensors (should trigger argument limit) ===")

    shapes = [base_shape] * num_tensors

    # Create tensors
    inputs = []
    tt_inputs = []
    for i in range(num_tensors):
        inputs.append(torch.rand(base_shape).to(torch.bfloat16))
        tt_inputs.append(ttnn.Tensor(inputs[i], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device))

    print("First run...")
    result1 = ttnn.concat(tt_inputs, dim)  # Let this crash if it will

    print("Second run...")
    result2 = ttnn.concat(tt_inputs, dim)  # Let this crash if it will


if __name__ == "__main__":
    print("=== Testing program cache concat issue ===")
    reproduce_program_cache_concat_issue()

    print("\n" + "=" * 50)
    print("=== Testing large tensor count ===")
    test_large_tensor_count()
