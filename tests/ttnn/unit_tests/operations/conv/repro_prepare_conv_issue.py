#!/usr/bin/env python3
import torch
import ttnn

# Open device
print("Opening device...")
device = ttnn.open_device(device_id=0)

try:
    # Create a weight tensor for conv2d in bf16 with row major layout on host
    # Typical conv2d weight shape: [out_channels, in_channels, kernel_h, kernel_w]
    # Using a small example: [64, 32, 3, 3]
    weight_shape = (64, 32, 3, 3)

    print(f"\nCreating torch tensor with shape {weight_shape}...")
    torch_weight = torch.randn(weight_shape, dtype=torch.bfloat16)

    print("Converting to ttnn tensor in ROW_MAJOR layout on host...")
    ttnn_weight = ttnn.from_torch(
        torch_weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT
    )

    print(f"\nInput tensor info:")
    print(f"  Shape: {ttnn_weight.shape}")
    print(f"  Dtype: {ttnn_weight.dtype}")
    print(f"  Layout: {ttnn_weight.layout}")
    print(f"  Is on device: {ttnn_weight.is_allocated()}")
    if ttnn_weight.is_allocated():
        print(f"  Storage type: {ttnn_weight.storage_type()}")

    # Call prepare_conv_weights with output_dtype=ttnn.bfloat8_b
    # Need to provide all required parameters for prepare_conv_weights
    print("\nCalling ttnn.prepare_conv_weights with output_dtype=ttnn.bfloat8_b...")

    # Conv2d parameters (example values)
    batch_size = 1
    in_channels = 32
    out_channels = 64
    input_height = 32
    input_width = 32
    kernel_size = [3, 3]
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1
    has_bias = False

    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_weight,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.TILE_LAYOUT,
        weights_format="OIHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        has_bias=has_bias,
        groups=groups,
        device=device,
        input_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat8_b
    )

    print(f"\nOutput tensor info:")
    print(f"  Shape: {prepared_weight.shape}")
    print(f"  Dtype: {prepared_weight.dtype}")
    print(f"  Layout: {prepared_weight.layout}")
    print(f"  Is on device: {prepared_weight.is_allocated()}")
    if prepared_weight.is_allocated():
        print(f"  Storage type: {prepared_weight.storage_type()}")

    # Check if tensor is actually on device
    print(f"\n{'='*60}")
    print("RESULT:")
    print(f"{'='*60}")
    if prepared_weight.is_allocated():
        storage_type = prepared_weight.storage_type()
        print(f"Tensor is ALLOCATED with storage type: {storage_type}")
        if "DEVICE" in str(storage_type).upper():
            print("✓ Tensor appears to be on DEVICE")
        else:
            print("✗ Tensor appears to be on HOST")
    else:
        print("✗ Tensor is NOT allocated (likely on host)")

    print(f"Dtype: {prepared_weight.dtype}")
    print(f"Expected dtype: ttnn.bfloat8_b")
    if "bfp8" in str(prepared_weight.dtype).lower() or "bfloat8_b" in str(prepared_weight.dtype).lower():
        print("✓ Dtype matches expected bfp8")
    else:
        print(f"✗ Dtype does NOT match (got {prepared_weight.dtype})")

finally:
    # Close device
    print("\nClosing device...")
    ttnn.close_device(device)
    print("Done!")
