#!/usr/bin/env python3
"""
Script to generate PyTorch reference outputs using CUDA in miniconda environment.
Checks environment setup and runs generation with appropriate settings.
"""

import sys
import subprocess
import json
from pathlib import Path


def check_environment():
    """Check if the environment is properly set up for CUDA generation."""
    print("=" * 80)
    print("Environment Check")
    print("=" * 80)

    errors = []
    warnings = []

    # Check PyTorch
    try:
        import torch

        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        errors.append("PyTorch not installed")
        print("✗ PyTorch: NOT INSTALLED")
        return False

    # Check CUDA availability
    if not torch.cuda.is_available():
        errors.append("CUDA not available")
        print("✗ CUDA: NOT AVAILABLE")
        print("  Install with: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia")
        return False

    print(f"✓ CUDA available: {torch.version.cuda}")
    print(f"✓ CUDA devices: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"  - cuda:{i}: {torch.cuda.get_device_name(i)}")

    # Check numpy
    try:
        import numpy as np

        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        warnings.append("NumPy not installed (will be needed)")

    # Check if generator script exists
    if not Path("generate_torch_reference_outputs.py").exists():
        errors.append("generate_torch_reference_outputs.py not found")
        print("✗ Generator script: NOT FOUND")
        return False

    print(f"✓ Generator script: Found")

    if errors:
        print("\n❌ Errors found:")
        for error in errors:
            print(f"  - {error}")
        return False

    if warnings:
        print("\n⚠ Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print("\n✓ Environment check passed!")
    return True


def print_config(device, precision, allow_tf32):
    """Print generation configuration."""
    print("\n" + "=" * 80)
    print("Generation Configuration")
    print("=" * 80)
    print(f"Device:              {device}")
    print(f"Float32 precision:   {precision}")
    print(f"TF32 for float32:    {'Enabled' if allow_tf32 else 'Disabled'}")
    print(f"BF16 accumulation:   BF16 (always enabled on CUDA)")
    print("=" * 80)


def run_generation(device="cuda:0", precision="highest", allow_tf32=False):
    """Run the generation script with specified settings."""
    cmd = ["python", "generate_torch_reference_outputs.py", "--device", device, "--precision", precision]

    if allow_tf32:
        cmd.append("--allow-tf32")

    print("\n" + "=" * 80)
    print("Starting Generation...")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Generation failed with error code {e.returncode}")
        return False


def print_summary():
    """Print summary of generated data."""
    system_info_path = Path("torch_reference_outputs/system_info.json")

    if not system_info_path.exists():
        print("\n⚠ Warning: system_info.json not found")
        return

    with open(system_info_path) as f:
        info = json.load(f)

    print("\n" + "=" * 80)
    print("✓ Generation Complete!")
    print("=" * 80)
    print(f"\nOutput directory: torch_reference_outputs/")
    print(f"\nSystem info:")
    print(f"  Device:               {info['device']}")
    print(f"  PyTorch:              {info['pytorch_version']}")
    print(f"  CUDA:                 {info.get('cuda_version', 'N/A')}")
    print(f"  GPU:                  {info.get('gpu_name', 'N/A')}")
    print(f"  Float32 precision:    {info['float32_matmul_precision']}")
    print(f"  TF32 (matmul):        {info.get('cuda_allow_tf32_matmul', False)}")
    print(f"  TF32 (cudnn):         {info.get('cuda_allow_tf32_cudnn', False)}")
    print(f"  BF16 reduced prec:    {info.get('cuda_bf16_reduced_precision', False)}")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Verify: python load_torch_reference_outputs.py")
    print("  2. Use in tests: from load_torch_reference_outputs import get_conv2d_inputs")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PyTorch reference outputs using CUDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default GPU (cuda:0) with highest precision
  python generate_cuda_reference.py

  # Use specific GPU
  python generate_cuda_reference.py --device cuda:1

  # Enable TF32 for faster generation (Ampere+ GPUs)
  python generate_cuda_reference.py --allow-tf32

  # Use high precision mode
  python generate_cuda_reference.py --precision high --allow-tf32

  # Just check environment without generating
  python generate_cuda_reference.py --check-only
        """,
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to use (default: cuda:0)")

    parser.add_argument(
        "--precision",
        type=str,
        default="highest",
        choices=["highest", "high", "medium"],
        help="Float32 matmul precision (default: highest)",
    )

    parser.add_argument("--allow-tf32", action="store_true", help="Allow TF32 for float32 operations (Ampere+ GPUs)")

    parser.add_argument("--check-only", action="store_true", help="Only check environment, don't generate")

    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the errors above.")
        sys.exit(1)

    if args.check_only:
        print("\n✓ Environment check complete. Ready to generate.")
        sys.exit(0)

    # Print configuration
    print_config(args.device, args.precision, args.allow_tf32)

    # Confirm
    if not args.yes:
        response = input("\nProceed with generation? (y/n): ")
        if response.lower() not in ["y", "yes"]:
            print("Generation cancelled.")
            sys.exit(0)

    # Run generation
    success = run_generation(args.device, args.precision, args.allow_tf32)

    if success:
        print_summary()
        sys.exit(0)
    else:
        print("\n❌ Generation failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
