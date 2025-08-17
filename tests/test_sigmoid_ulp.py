import torch
import ttnn
import numpy as np


def spacing(x: torch.Tensor):
    """Calculate spacing (ULP size) for bfloat16 and float32 inputs."""
    # If input is bfloat16, use abs(x) * 2^-7 for normal numbers
    if x.dtype == torch.bfloat16:
        ulp = torch.abs(x).to(torch.float32) * 2**-7
        # For zero, use smallest positive normal bfloat16 (2^-126)
        ulp = torch.where(x == 0, torch.tensor(2**-126, dtype=torch.float32), ulp)
        return ulp
    else:
        # For float32, use nextafter
        direction = torch.where(x >= 0, torch.inf, -torch.inf)
        next_val = torch.nextafter(x, direction)
        return torch.abs(next_val - x)


def ulp_difference(a, b):
    """Calculate ULP (Units in the Last Place) difference between two tensors for bfloat16 and float32."""
    # Ensure inputs are torch tensors
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    # Move to CPU and ensure same device/dtype
    a = a.detach().cpu()
    b = b.detach().cpu()
    # Handle special cases (inf, nan)
    mask_valid = torch.isfinite(a) & torch.isfinite(b)
    if not torch.any(mask_valid):
        return torch.tensor([float("inf")])
    # Use the larger absolute value for spacing calculation
    max_abs = torch.maximum(torch.abs(a), torch.abs(b))
    # Calculate spacing (distance to next representable float)
    ulp_spacing = spacing(max_abs)
    # Calculate ULP difference
    abs_diff = torch.abs(a - b)
    ulp_diff = abs_diff / ulp_spacing
    return ulp_diff[mask_valid]


# Use a reasonable range for sigmoid (-10 to 10)
# Values outside this range will be very close to 0 or 1
N_RANGE = -10
P_RANGE = 10


def test_ttnn_sigmoid():
    x_values = torch.linspace(N_RANGE, P_RANGE, 1024 * 1024, dtype=torch.bfloat16).reshape(-1, 32, 32)

    with ttnn.manage_device(0) as dev:
        # Convert to ttnn tensor with BFLOAT16 and tile layout
        ttnn_input = ttnn.from_torch(x_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        # Test both normal and approximate modes
        for approx_mode in [False, True]:
            mode_str = "approximate" if approx_mode else "normal"
            # ttnn_result = ttnn.sigmoid(ttnn_input, fast_and_approximate_mode=approx_mode)
            ttnn_result = ttnn.sigmoid_accurate(ttnn_input)
            ttnn_output = ttnn.to_torch(ttnn_result)
            torch_reference = torch.sigmoid(x_values)

            print("\n" + "=" * 60)
            print(f"COMPARISON RESULTS ({mode_str} mode)")
            print("=" * 60)

            # Calculate various error metrics
            abs_diff = torch.abs(ttnn_output - torch_reference)
            rel_diff = abs_diff.float() / (torch.abs(torch_reference.float()) + 1e-9)

            print(f"Max absolute difference: {abs_diff.max():.6e}")
            print(f"Mean absolute difference: {abs_diff.mean():.6e}")
            print(f"Max relative difference: {rel_diff.max():.6e}")
            print(f"Mean relative difference: {rel_diff.mean():.6e}")

            # Calculate ULP difference
            ulp_diffs = ulp_difference(ttnn_output, torch_reference)

            print(f"\nULP Statistics:")
            print(f"Max ULP difference: {torch.max(ulp_diffs):.2f}")
            print(f"Mean ULP difference: {torch.mean(ulp_diffs):.2f}")
            print(f"Median ULP difference: {torch.median(ulp_diffs):.2f}")

            N = 30
            # Print some sample comparisons (equally spaced subset)
            total_values = len(x_values.flatten())
            indices = np.linspace(0, total_values - 1, N, dtype=int)

            print(f"\nSample comparisons ({N} equally spaced values):")
            print("Input\t\tTorch sigmoid\tTTNN sigmoid\tAbs Diff\tULP Diff\tOne ULP size")
            print("-" * 70)
            for i in indices:
                inp = x_values.flatten()[i].item()
                torch_val = torch_reference.flatten()[i].item()
                ttnn_val = ttnn_output.flatten()[i].item()
                abs_d = abs_diff.flatten()[i].item()
                ulp_d = ulp_diffs[i] if i < len(ulp_diffs) else 0
                torch_ulp = spacing(torch.tensor(torch_val, dtype=torch.bfloat16)).item()
                print(f"{inp:8.3f}\t{torch_val:10.6f}\t{ttnn_val:10.6f}\t{abs_d:.2e}\t{ulp_d:8.2f}\t{torch_ulp:.2e}")

            reasonable_ulp = torch.sum(ulp_diffs < 2) / len(ulp_diffs) * 100
            print(f"\nPercentage of values with ULP < 2: {reasonable_ulp:.1f}%")


if __name__ == "__main__":
    test_ttnn_sigmoid()
