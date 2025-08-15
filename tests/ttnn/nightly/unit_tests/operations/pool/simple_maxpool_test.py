import torch
import ttnn
import math

device = ttnn.CreateDevice(0, l1_small_size=8192)

in_n = 1
in_h = 159
in_w = 159
in_c = 32
kernel_size = [3, 3]
stride = [1, 1]
padding = [1, 1]
dilation = [1, 1]
shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
tensor_shape = (in_n, in_c, in_h, in_w)  # NCHW format

# Create tensor filled with height and width coordinates
torch.manual_seed(0)
# torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

# Create tensor where each element equals its HW coordinate (h * in_w + w)
torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
# torch_input = torch.zeros(tensor_shape, dtype=torch.bfloat16)
# for n in range(in_n):
#     for c in range(in_c):
#         for h in range(in_h):
#             for w in range(in_w):
#                 coordinate_value = c  # h * in_w + w
#                 torch_input[n, c, h, w] = coordinate_value

ttnn_input_shape = (1, 1, in_n * in_h * in_w, in_c)
torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)  # NHW, C
ttnn_input = ttnn.from_torch(torch_input_reshaped, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

# ttnn_output = ttnn.max_pool2d(
#     input_tensor=ttnn_input,
#     batch_size=in_n,
#     input_h=in_h,
#     input_w=in_w,
#     channels=in_c,
#     kernel_size=kernel_size,
#     stride=stride,
#     padding=padding,  # ttnn is padding in the order (top, bottom, left, right)
#     dilation=dilation,
#     applied_shard_scheme=shard_scheme,
# )

# print("Output without indices:")
# print(ttnn.to_torch(ttnn_output))

ttnn_output, indices = ttnn.max_pool2d(
    input_tensor=ttnn_input,
    batch_size=in_n,
    input_h=in_h,
    input_w=in_w,
    channels=in_c,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,  # ttnn is padding in the order (top, bottom, left, right)
    dilation=dilation,
    applied_shard_scheme=shard_scheme,
    return_indices=True,
)

print("\nTTNN max pool results:")
print("Output shape:", ttnn.to_torch(ttnn_output).shape)
# print("Output:\n", ttnn.to_torch(ttnn_output))
print("Indices shape:", ttnn.to_torch(indices).shape)
# print("Indices:\n", ttnn.to_torch(indices))

# Run PyTorch max pool for reference
torch_output, torch_indices = torch.nn.functional.max_pool2d(
    torch_input, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True
)

# Reshape torch output to match TTNN format (NCHW -> NHWC)
torch_output_reshaped = torch_output.permute(0, 2, 3, 1)  # N, H, W, C
torch_indices_reshaped = torch_indices.permute(0, 2, 3, 1)  # N, H, W, C

print("PyTorch max pool results:")
print("Output shape:", torch_output_reshaped.shape)
# print("Output:\n", torch_output_reshaped)
print("Indices shape:", torch_indices_reshaped.shape)
# print("Indices:\n", torch_indices_reshaped)

# Compare outputs using allclose
ttnn_output_torch = ttnn.to_torch(ttnn_output)
ttnn_indices_torch = ttnn.to_torch(indices)

# Reshape TTNN outputs to match PyTorch shape for comparison
# TTNN output is in shape (1, 1, output_h*output_w, channels)
# Calculate output dimensions using the pooling formula
pad_h = padding[0] * 2  # padding is [top/bottom, left/right] but we need total padding
pad_w = padding[1] * 2
dilation_h = dilation[0]
dilation_w = dilation[1]
kernel_h = kernel_size[0]
kernel_w = kernel_size[1]
stride_h = stride[0]
stride_w = stride[1]

output_h = math.floor((in_h + pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
output_w = math.floor((in_w + pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1

ttnn_output_reshaped = ttnn_output_torch.reshape(in_n, output_h, output_w, in_c)
ttnn_indices_reshaped = ttnn_indices_torch.reshape(in_n, output_h, output_w, in_c)

output_match = torch.allclose(torch_output_reshaped, ttnn_output_reshaped)
indices_match = torch.allclose(torch_indices_reshaped.float(), ttnn_indices_reshaped.float())

print(f"Output values match (allclose): {output_match}")
print(f"Indices values match (allclose): {indices_match}")

# Print detailed mismatch information if outputs don't match
if not output_match:
    print("\n=== OUTPUT MISMATCHES ===")
    diff = torch.abs(torch_output_reshaped - ttnn_output_reshaped)
    max_diff = torch.max(diff)
    print(f"Maximum absolute difference: {max_diff}")

    # Find positions where values don't match (with a small tolerance)
    mismatch_mask = diff > 1e-5
    mismatch_positions = torch.nonzero(mismatch_mask, as_tuple=False)

    if len(mismatch_positions) > 0:
        print(f"Number of mismatched elements: {len(mismatch_positions)}")
        print("All mismatches (n, h, w, c): torch_val vs ttnn_val (diff):")
        for i, pos in enumerate(mismatch_positions):
            n, h, w, c = pos
            torch_val = torch_output_reshaped[n, h, w, c]
            ttnn_val = ttnn_output_reshaped[n, h, w, c]
            diff_val = diff[n, h, w, c]
            print(f"  [{n}, {h}, {w}, {c}]: {torch_val:.6f} vs {ttnn_val:.6f} (diff: {diff_val:.6f})")

# Count total output elements
total_output_elements = torch_output_reshaped.numel()
print(f"\nTotal output elements: {total_output_elements}")

# Analyze indices mismatches
tie_breaking_differences = 0
value_differences = 0
has_actual_errors = False

if not indices_match:
    print("\n=== INDICES MISMATCHES ===")
    torch_indices_float = torch_indices_reshaped.float()
    ttnn_indices_float = ttnn_indices_reshaped.float()
    diff = torch.abs(torch_indices_float - ttnn_indices_float)
    max_diff = torch.max(diff)
    print(f"Maximum absolute difference: {max_diff}")

    # Find positions where indices don't match
    mismatch_mask = diff > 1e-5
    mismatch_positions = torch.nonzero(mismatch_mask, as_tuple=False)

    if len(mismatch_positions) > 0:
        print(f"Number of mismatched elements: {len(mismatch_positions)}")
        print("All mismatches (n, h, w, c): torch_idx vs ttnn_idx (diff):")
        for i, pos in enumerate(mismatch_positions):
            n, h, w, c = pos
            torch_idx = torch_indices_float[n, h, w, c]
            ttnn_idx = ttnn_indices_float[n, h, w, c]
            diff_val = diff[n, h, w, c]

            # Convert linear indices back to spatial coordinates in the input tensor
            # PyTorch uses NCHW format for indexing in max_pool2d
            torch_idx_int = int(torch_idx.item())
            ttnn_idx_int = int(ttnn_idx.item())

            # Convert linear index to (h, w) coordinates within the pooling window
            # For PyTorch indices, they are relative to the flattened spatial dimensions (H*W)
            torch_h = torch_idx_int // in_w
            torch_w = torch_idx_int % in_w
            ttnn_h = ttnn_idx_int // in_w
            ttnn_w = ttnn_idx_int % in_w

            # Get the actual input values at these positions
            torch_input_val = torch_input[n, c, torch_h, torch_w] if torch_h < in_h and torch_w < in_w else float("nan")
            ttnn_input_val = torch_input[n, c, ttnn_h, ttnn_w] if ttnn_h < in_h and ttnn_w < in_w else float("nan")

            print(
                f"  output [{n}, {h}, {w}, {c}]: torch_idx={torch_idx:.0f} vs ttnn_idx={ttnn_idx:.0f} (diff: {diff_val:.0f})"
            )
            print(f"    Torch chose input[{n},{c},{torch_h},{torch_w}] = {torch_input_val:.6f}")
            print(f"    TTNN chose input[{n},{c},{ttnn_h},{ttnn_w}] = {ttnn_input_val:.6f}")

            # Check if this is a valid tie-breaking difference
            # Two conditions must be satisfied:
            # 1. The values must be the same
            # 2. Both indices must be within the same kernel window

            values_same = abs(torch_input_val - ttnn_input_val) < 1e-6

            # Calculate the top-left corner of the kernel window for this output position
            # Given output position (h, w), the top-left of the kernel window is:
            kernel_top_left_h = h * stride_h - padding[0]  # padding[0] is top padding
            kernel_top_left_w = w * stride_w - padding[1]  # padding[1] is left padding

            # Check if both indices are within the same kernel window
            kernel_bottom_right_h = kernel_top_left_h + kernel_h - 1
            kernel_bottom_right_w = kernel_top_left_w + kernel_w - 1

            torch_in_window = (
                kernel_top_left_h <= torch_h <= kernel_bottom_right_h
                and kernel_top_left_w <= torch_w <= kernel_bottom_right_w
            )
            ttnn_in_window = (
                kernel_top_left_h <= ttnn_h <= kernel_bottom_right_h
                and kernel_top_left_w <= ttnn_w <= kernel_bottom_right_w
            )

            same_kernel_window = torch_in_window and ttnn_in_window

            print(
                f"    Kernel window: top_left=({kernel_top_left_h},{kernel_top_left_w}), bottom_right=({kernel_bottom_right_h},{kernel_bottom_right_w})"
            )
            print(f"    Torch index in window: {torch_in_window}, TTNN index in window: {ttnn_in_window}")

            if values_same and same_kernel_window:
                print(f"    -> Same input values AND same kernel window! This is a valid tie-breaking difference.")
                tie_breaking_differences += 1
            elif values_same and not same_kernel_window:
                print(f"    -> Same input values but DIFFERENT kernel windows! This is an error.")
                value_differences += 1
                has_actual_errors = True
            else:
                print(f"    -> Different input values! Value difference: {abs(torch_input_val - ttnn_input_val):.6f}")
                value_differences += 1
                has_actual_errors = True
            print()

print(f"\n=== SUMMARY ===")
print(f"Total output elements: {total_output_elements}")
print(f"Tie-breaking differences: {tie_breaking_differences}")
print(f"Value or window differences(actual errors): {value_differences}")

# Updated test result logic - only fail if there are actual value differences or output mismatches
test_passed = output_match and (not has_actual_errors)

if test_passed:
    print("\n✓ Test PASSED: TTNN and PyTorch outputs match!")
    if tie_breaking_differences > 0:
        print(f"  Note: {tie_breaking_differences} tie-breaking differences in indices are acceptable.")
else:
    print("\n✗ Test FAILED:")
    if not output_match:
        print("  - Output values do not match")
    if has_actual_errors:
        print(f"  - {value_differences} actual value differences found in indices")
