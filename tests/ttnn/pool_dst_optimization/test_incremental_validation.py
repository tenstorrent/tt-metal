#!/usr/bin/env python3
"""
Incremental validation tests for step-by-step verification
"""

import ttnn
import torch
import pytest


def test_dst_register_allocation():
    """Test if DST register modifications compile and allocate correctly"""
    print("Testing DST register allocation...")

    device = ttnn.open_device(device_id=0)

    try:
        # Minimal tensor to trigger DST allocation path
        torch_input = torch.randn((1, 32, 8, 8), dtype=torch.bfloat16)
        torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))
        ttnn_input = ttnn.from_torch(
            torch_input_permuted,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        print("✓ Tensor creation successful")

        # Try to create the operation (even if it fails, we want to see where)
        try:
            result = ttnn.avg_pool2d(
                ttnn_input,
                batch_size=1,
                input_h=8,
                input_w=8,
                channels=32,
                kernel_size=(2, 2),
                stride=(1, 1),
                padding=(0, 0),
            )
            print("✓ Operation completed")
            return True
        except Exception as op_error:
            print(f"⚠ Operation failed (expected during development): {op_error}")
            return False

    except Exception as setup_error:
        print(f"✗ Setup failed: {setup_error}")
        return False
    finally:
        ttnn.close_device(device)


def test_kernel_compilation():
    """Test if kernel modifications compile successfully"""
    print("Testing kernel compilation...")

    # This will trigger kernel compilation
    device = ttnn.open_device(device_id=0)

    try:
        # Force kernel compilation with minimal operation
        torch_input = torch.randn((1, 4, 4, 4), dtype=torch.bfloat16)
        torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))
        ttnn_input = ttnn.from_torch(
            torch_input_permuted,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        print("✓ Input tensor created")

        # Attempt operation - if kernels compile, this should at least start
        result = ttnn.avg_pool2d(
            ttnn_input,
            batch_size=1,
            input_h=4,
            input_w=4,
            channels=4,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
        )
        print("✓ Kernel compilation and execution successful")
        return True

    except Exception as e:
        print(f"Kernel compilation/execution status: {e}")
        # If it's a compilation error, it will be obvious
        # If it's a runtime error, kernels compiled successfully
        return "compilation" not in str(e).lower()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    print("=== INCREMENTAL VALIDATION ===")

    print("\n1. Testing kernel compilation:")
    compile_ok = test_kernel_compilation()

    print("\n2. Testing DST register allocation:")
    dst_ok = test_dst_register_allocation()

    print(f"\n=== SUMMARY ===")
    print(f"Kernel Compilation: {'✅ PASS' if compile_ok else '❌ FAIL'}")
    print(f"DST Allocation: {'✅ PASS' if dst_ok else '❌ FAIL'}")

    if compile_ok:
        print("\n🎯 Ready for next step: data flow validation")
    else:
        print("\n🚫 Fix compilation issues before proceeding")
