#!/usr/bin/env python3
"""
Baseline validator using existing test infrastructure
"""

import pytest
import ttnn
import torch
from tests.ttnn.nightly.unit_tests.operations.pool.test_avgpool2d import run_avg_pool2d


def test_baseline_functionality():
    """Test baseline avgpool functionality using existing test infrastructure"""
    print("\n" + "=" * 60)
    print("BASELINE VALIDATION TEST")
    print("=" * 60)

    # Use a simple configuration that should work
    input_shape = [1, 32, 16, 16]
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)

    print(f"Testing configuration:")
    print(f"  Input shape: {input_shape}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Stride: {stride}")
    print(f"  Padding: {padding}")

    # Open device
    device = ttnn.open_device(device_id=0)
    tensor_map = {}

    try:
        # Run the existing avgpool test
        result = run_avg_pool2d(
            device=device,
            tensor_map=tensor_map,
            input_shape=input_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            divisor_override=None,
            count_include_pad=True,
            shard_scheme=None,
            run_twice=False,
            in_dtype=ttnn.bfloat16,
            nightly_skips=False,
            skips_enabled=False,
            out_dtype=ttnn.bfloat16,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        print("✅ BASELINE TEST PASSED!")
        print("Current avgpool implementation is working correctly.")
        return True

    except Exception as e:
        print(f"❌ BASELINE TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    success = test_baseline_functionality()
    if success:
        print("\n🎯 READY TO PROCEED WITH OPTIMIZATION IMPLEMENTATION")
    else:
        print("\n🚫 NEED TO FIX BASELINE ISSUES BEFORE PROCEEDING")
