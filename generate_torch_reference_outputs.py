#!/usr/bin/env python3
"""
Standalone script to generate PyTorch reference outputs for conv2d and matmul.
Saves outputs to files for cross-machine comparison.

This script uses the same configuration as conv2d_vs_matmul_ulp_comparison.ipynb:
- M = batch * output_height * output_width
- N = out_channels = 32
- K varies by changing input_channels
- Fixed: batch=1, input_height=32, input_width=32, kernel=3x3, padding=(1,1), stride=(1,1)
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
import platform
import sys
import hashlib
import argparse

# Configuration
SEED = 42
INPUT_CHANNELS_SWEEP = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
OUTPUT_DIR = "torch_reference_outputs"

# Fixed parameters
BATCH = 1
INPUT_HEIGHT = 8
INPUT_WIDTH = 4
OUT_CHANNELS = 32
KERNEL_H = 3
KERNEL_W = 3
PADDING = (1, 1)
STRIDE = (1, 1)
USE_BIAS = True

# Data types to test
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Generation methods
GEN_METHODS = ["rand", "randn"]


def generate_conv2d_inputs(
    batch,
    in_channels,
    input_height,
    input_width,
    out_channels,
    kernel_h,
    kernel_w,
    dtype,
    seed=0,
    device="cpu",
    method="rand",
):
    """Generate random input tensors for conv2d using either rand or randn."""
    torch.manual_seed(seed)

    gen_fn = torch.rand if method == "rand" else torch.randn

    input_tensor = gen_fn((batch, in_channels, input_height, input_width), dtype=dtype, device=device)
    weight_tensor = gen_fn((out_channels, in_channels, kernel_h, kernel_w), dtype=dtype, device=device)
    bias_tensor = gen_fn((out_channels,), dtype=dtype, device=device)

    return input_tensor, weight_tensor, bias_tensor


def conv2d_to_matmul_shape(
    batch, in_channels, input_height, input_width, out_channels, kernel_h, kernel_w, padding, stride=(1, 1)
):
    """Calculate the equivalent matmul shape for a conv2d operation."""
    output_height = (input_height + 2 * padding[0] - kernel_h) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - kernel_w) // stride[1] + 1

    M = batch * output_height * output_width
    K = in_channels * kernel_h * kernel_w
    N = out_channels

    return M, K, N, output_height, output_width


def generate_matmul_inputs(M, K, N, dtype, seed=0, device="cpu", method="rand"):
    """Generate random input tensors for matmul using either rand or randn."""
    torch.manual_seed(seed)
    gen_fn = torch.rand if method == "rand" else torch.randn
    A = gen_fn((M, K), dtype=dtype, device=device)
    B = gen_fn((K, N), dtype=dtype, device=device)
    return A, B


def run_torch_conv2d(input_tensor, weight_tensor, bias_tensor, padding, stride, use_bias=True):
    """Run conv2d in PyTorch."""
    return torch.nn.functional.conv2d(
        input_tensor, weight_tensor, bias=bias_tensor if use_bias else None, stride=stride, padding=padding
    )


def run_torch_matmul(A, B):
    """Run matmul in PyTorch."""
    return torch.matmul(A, B)


def save_tensor_to_file(tensor, filepath):
    """Save a PyTorch tensor to a numpy file.

    Note: BFloat16 tensors are converted to float32 for storage since NumPy
    doesn't natively support bfloat16. The original dtype is preserved in metadata.
    """
    # Convert bfloat16 to float32 for storage (NumPy doesn't support bfloat16)
    if tensor.dtype == torch.bfloat16:
        tensor_np = tensor.to(torch.float32).detach().cpu().numpy()
    else:
        tensor_np = tensor.detach().cpu().numpy()
    np.save(filepath, tensor_np)


def compute_tensor_hash(tensor):
    """Compute SHA256 hash of a tensor for verification.

    Note: For bfloat16 tensors, the hash is computed on the float32 representation
    to match how the tensors are saved (since NumPy doesn't support bfloat16).
    """
    # Convert to numpy (handling bfloat16)
    if tensor.dtype == torch.bfloat16:
        tensor_np = tensor.to(torch.float32).detach().cpu().numpy()
    else:
        tensor_np = tensor.detach().cpu().numpy()
    tensor_bytes = tensor_np.tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def collect_system_info(device):
    """Collect system information for reproducibility."""
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "device": str(device),
    }

    # Add CUDA info if using GPU
    if device.type == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_count"] = torch.cuda.device_count()

    return info


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate PyTorch reference outputs for conv2d and matmul")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to use: "cpu", "cuda", or "cuda:0", "cuda:1", etc. (default: cpu)',
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="highest",
        choices=["highest", "high", "medium"],
        help='Float32 matmul precision: "highest", "high", or "medium" (default: highest)',
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow TF32 for float32 operations on Ampere+ GPUs (faster, slight accuracy loss)",
    )
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)

    # Check if CUDA is requested but not available
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"ERROR: CUDA device requested but CUDA is not available.")
        print(f"Falling back to CPU.")
        device = torch.device("cpu")

    # Set float32 matmul precision
    torch.set_float32_matmul_precision(args.precision)

    # Set CUDA precision flags (only relevant for CUDA)
    if device.type == "cuda":
        # TF32 for float32 operations (matmul and conv)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

        # Always use bfloat16 accumulation for bfloat16 operations (no FP32 accumulation)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    print("=" * 80)
    print("PyTorch Reference Output Generator")
    print("=" * 80)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Collect system info
    system_info = collect_system_info(device)
    system_info["float32_matmul_precision"] = args.precision

    # Record CUDA precision settings if applicable
    if device.type == "cuda":
        system_info["cuda_allow_tf32_matmul"] = args.allow_tf32
        system_info["cuda_allow_tf32_cudnn"] = args.allow_tf32
        system_info["cuda_bf16_reduced_precision"] = True  # Always enabled for CUDA

    print("\nSystem Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Save system info
    with open(output_dir / "system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)

    # Configuration info
    config_info = {
        "seed": SEED,
        "input_channels_sweep": INPUT_CHANNELS_SWEEP,
        "batch": BATCH,
        "input_height": INPUT_HEIGHT,
        "input_width": INPUT_WIDTH,
        "out_channels": OUT_CHANNELS,
        "kernel_h": KERNEL_H,
        "kernel_w": KERNEL_W,
        "padding": PADDING,
        "stride": STRIDE,
        "use_bias": USE_BIAS,
        "dtypes": list(DTYPES.keys()),
        "gen_methods": GEN_METHODS,
    }

    # Calculate K values
    k_values = [ic * KERNEL_H * KERNEL_W for ic in INPUT_CHANNELS_SWEEP]
    config_info["k_values"] = k_values

    print("\nConfiguration:")
    print(f"  Input channels sweep: {INPUT_CHANNELS_SWEEP}")
    print(f"  K values: {k_values}")
    print(
        f"  Fixed: batch={BATCH}, input={INPUT_HEIGHT}x{INPUT_WIDTH}, "
        f"out_channels={OUT_CHANNELS}, kernel={KERNEL_H}x{KERNEL_W}"
    )
    print(f"  Padding: {PADDING}, Stride: {STRIDE}")
    print(f"  Use bias: {USE_BIAS}")
    print(f"  Seed: {SEED}")
    print(f"  Generation methods: {GEN_METHODS}")

    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_info, f, indent=2)

    print("\n" + "=" * 80)
    print("Generating Reference Outputs")
    print("=" * 80)

    # Run for each dtype and generation method combination
    for dtype_name, dtype in DTYPES.items():
        for method in GEN_METHODS:
            group_name = f"{dtype_name}_{method}"
            print(f"\nProcessing: {group_name}")
            print("-" * 80)

            # Create subdirectory for this group
            group_dir = output_dir / group_name
            group_dir.mkdir(exist_ok=True)

            # Iterate over K values
            for ic, k in zip(INPUT_CHANNELS_SWEEP, k_values):
                print(f"  K={k} (input_channels={ic})...", end=" ", flush=True)

                # Calculate matmul shape
                M, K_matmul, N, output_height, output_width = conv2d_to_matmul_shape(
                    BATCH, ic, INPUT_HEIGHT, INPUT_WIDTH, OUT_CHANNELS, KERNEL_H, KERNEL_W, PADDING, STRIDE
                )

                # Generate conv2d inputs with specified method
                input_tensor, weight_tensor, bias_tensor = generate_conv2d_inputs(
                    BATCH, ic, INPUT_HEIGHT, INPUT_WIDTH, OUT_CHANNELS, KERNEL_H, KERNEL_W, dtype, SEED, device, method
                )

                # Generate matmul inputs with specified method
                A, B = generate_matmul_inputs(M, K_matmul, N, dtype, SEED, device, method)

                # Run PyTorch conv2d
                conv2d_output = run_torch_conv2d(input_tensor, weight_tensor, bias_tensor, PADDING, STRIDE, USE_BIAS)

                # Run PyTorch matmul
                matmul_output = run_torch_matmul(A, B)

                # Save only outputs (inputs can be regenerated from seed)
                prefix = f"k{k:04d}_ic{ic:03d}"

                # Save conv2d output
                save_tensor_to_file(conv2d_output, group_dir / f"{prefix}_conv2d_output.npy")

                # Save matmul output
                save_tensor_to_file(matmul_output, group_dir / f"{prefix}_matmul_output.npy")

                # Compute hashes for all input tensors
                input_hashes = {
                    "conv2d": {
                        "input": compute_tensor_hash(input_tensor),
                        "weight": compute_tensor_hash(weight_tensor),
                        "bias": compute_tensor_hash(bias_tensor),
                    },
                    "matmul": {
                        "A": compute_tensor_hash(A),
                        "B": compute_tensor_hash(B),
                    },
                }

                # Save shapes, dtypes, hashes, and generation method
                metadata = {
                    "shapes": {
                        "conv2d": {
                            "input": list(input_tensor.shape),
                            "weight": list(weight_tensor.shape),
                            "bias": list(bias_tensor.shape),
                            "output": list(conv2d_output.shape),
                        },
                        "matmul": {
                            "A": list(A.shape),
                            "B": list(B.shape),
                            "output": list(matmul_output.shape),
                        },
                    },
                    "dtypes": {
                        "conv2d": str(input_tensor.dtype),
                        "matmul": str(A.dtype),
                    },
                    "gen_method": method,
                    "input_hashes": input_hashes,
                }

                with open(group_dir / f"{prefix}_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                print("Done")

    print("\n" + "=" * 80)
    print(f"Reference outputs saved to: {output_dir.absolute()}")
    print("=" * 80)

    # Create a README
    readme_content = f"""# PyTorch Reference Outputs

Generated on: {platform.platform()}
PyTorch version: {torch.__version__}
Python version: {sys.version.split()[0]}

## Configuration

- Seed: {SEED}
- Input channels: {INPUT_CHANNELS_SWEEP}
- K values: {k_values}
- Batch: {BATCH}
- Input size: {INPUT_HEIGHT}x{INPUT_WIDTH}
- Output channels: {OUT_CHANNELS}
- Kernel: {KERNEL_H}x{KERNEL_W}
- Padding: {PADDING}
- Stride: {STRIDE}
- Use bias: {USE_BIAS}

## Directory Structure

```
{OUTPUT_DIR}/
├── system_info.json          # System and library versions
├── config.json                # Test configuration
├── README.md                  # This file
├── bfloat16_rand/             # BFloat16 with torch.rand
│   ├── k0072_ic008_conv2d_output.npy
│   ├── k0072_ic008_matmul_output.npy
│   ├── k0072_ic008_metadata.json
│   └── ...
├── bfloat16_randn/            # BFloat16 with torch.randn
│   └── ...
├── float32_rand/              # Float32 with torch.rand
│   └── ...
└── float32_randn/             # Float32 with torch.randn
    └── ...
```

## File Naming Convention

Format: `k<K>_ic<input_channels>_<operation>_output.npy`

Examples:
- `k0072_ic008_conv2d_output.npy` - Conv2d output for K=72, input_channels=8
- `k0072_ic008_matmul_output.npy` - Matmul output for K=72, input_channels=8
- `k0072_ic008_metadata.json` - Metadata including shapes, dtypes, and input hashes

**Note**: Input tensors are NOT saved. They can be regenerated using the same seed ({SEED}).

## Metadata Files

Each test case has a `metadata.json` file containing:
- **shapes**: Tensor dimensions for all inputs and outputs
- **dtypes**: Data types used (bfloat16 or float32)
- **input_hashes**: SHA256 hashes of all input tensors for verification

The hashes allow you to verify that the same inputs were generated on different machines
by regenerating them with the same seed and comparing hashes.

## Loading Data and Verifying Inputs

```python
import numpy as np
import torch
import json

# Load metadata
with open('bfloat16/k0072_ic008_metadata.json') as f:
    metadata = json.load(f)
    print(f"Input hash: {{metadata['input_hashes']['conv2d']['input']}}")
    print(f"Dtype: {{metadata['dtypes']['conv2d']}}")

# Load output
conv2d_output = np.load('bfloat16/k0072_ic008_conv2d_output.npy')
matmul_output = np.load('bfloat16/k0072_ic008_matmul_output.npy')

# Regenerate inputs using the same seed and verify hashes
# (See load_torch_reference_outputs.py for helper functions)
```

**Note on BFloat16 Storage**: Since NumPy doesn't natively support bfloat16, bfloat16 output tensors
are converted to float32 for storage. The original dtype is preserved in the metadata file.
Float32 tensors are saved directly in float32 format.
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    num_groups = len(DTYPES) * len(GEN_METHODS)
    print(
        f"\nGenerated {len(INPUT_CHANNELS_SWEEP)} test cases for {num_groups} groups ({len(DTYPES)} dtypes × {len(GEN_METHODS)} methods)"
    )
    print(f"Total files: {len(INPUT_CHANNELS_SWEEP) * num_groups * 3}")  # 2 outputs + 1 metadata file per case


if __name__ == "__main__":
    main()
