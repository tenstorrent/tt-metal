#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0

Quick test script to verify YOLOS-small works on Tenstorrent hardware.
This script can be run standalone with minimal dependencies.
"""

import os
import sys


def test_basic_imports():
    """Test that basic imports work"""
    print("=" * 80)
    print("YOLOS-small Tenstorrent Test - Step 1: Imports")
    print("=" * 80)

    try:
        import torch

        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False

    try:
        import ttnn

        print(f"✓ TTNN imported successfully")

        # Check devices
        num_devices = ttnn.get_num_devices()
        print(f"✓ Tenstorrent devices detected: {num_devices}")

        if num_devices == 0:
            print("✗ No Tenstorrent devices found!")
            return False

    except ImportError as e:
        print(f"✗ TTNN not found: {e}")
        print("  Install tt-metal first")
        return False
    except Exception as e:
        print(f"✗ TTNN error: {e}")
        return False

    return True


def test_device_access():
    """Test opening and closing Tenstorrent device"""
    print("\n" + "=" * 80)
    print("YOLOS-small Tenstorrent Test - Step 2: Device Access")
    print("=" * 80)

    try:
        import ttnn

        print("Opening device 0...")
        device = ttnn.open_device(device_id=0)
        print(f"✓ Device opened successfully: {device}")

        print("Closing device...")
        ttnn.close_device(device)
        print("✓ Device closed successfully")

        return True

    except Exception as e:
        print(f"✗ Device access failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_simple_ttnn_op():
    """Test a simple TTNN operation"""
    print("\n" + "=" * 80)
    print("YOLOS-small Tenstorrent Test - Step 3: Simple TTNN Operation")
    print("=" * 80)

    try:
        import torch

        import ttnn

        device = ttnn.open_device(device_id=0)

        # Create a simple tensor
        print("Creating test tensor...")
        torch_tensor = torch.randn(1, 10, 384)
        print(f"  Shape: {torch_tensor.shape}")

        # Convert to TTNN
        print("Converting to TTNN...")
        ttnn_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        print(f"✓ TTNN tensor created: {ttnn_tensor.shape}")

        # Simple operation
        print("Testing GELU activation...")
        result = ttnn.gelu(ttnn_tensor)
        print(f"✓ GELU successful: {result.shape}")

        # Convert back
        result_torch = ttnn.to_torch(result)
        print(f"✓ Converted back to PyTorch: {result_torch.shape}")

        ttnn.close_device(device)
        print("✓ All basic operations working!")

        return True

    except Exception as e:
        print(f"✗ TTNN operation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_yolos_reference():
    """Test YOLOS reference implementation"""
    print("\n" + "=" * 80)
    print("YOLOS-small Tenstorrent Test - Step 4: Reference Model")
    print("=" * 80)

    # Check if we're in the right directory
    if not os.path.exists("reference"):
        print("✗ Not in yolos_small directory. Please cd to /workdir/ttnn-bounty-30874/yolos_small")
        return False

    try:
        from reference.config import get_yolos_small_config
        from reference.modeling_yolos import YolosForObjectDetection

        print("Creating YOLOS config...")
        config = get_yolos_small_config()
        print(f"✓ Config: {config.hidden_size}D, {config.num_hidden_layers} layers")

        print("Creating PyTorch model...")
        model = YolosForObjectDetection(config)
        print(f"✓ Model created with ~30.7M parameters")

        # Test forward pass
        import torch

        print("Testing forward pass...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 864)
            logits, boxes = model(dummy_input)
            print(f"✓ Forward pass successful!")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Boxes shape: {boxes.shape}")

        return True

    except Exception as e:
        print(f"✗ Reference model failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_yolos_ttnn_stage1():
    """Test YOLOS TTNN implementation Stage 1"""
    print("\n" + "=" * 80)
    print("YOLOS-small Tenstorrent Test - Step 5: TTNN Stage 1")
    print("=" * 80)

    try:
        import torch
        from reference.config import get_yolos_small_config
        from reference.modeling_yolos import YolosForObjectDetection as PyTorchYolos
        from yolos_ttnn.common import OptimizationConfig
        from yolos_ttnn.modeling_yolos import YolosForObjectDetection as TtnnYolos

        import ttnn

        print("Creating reference model...")
        config = get_yolos_small_config()
        pytorch_model = PyTorchYolos(config)
        pytorch_model.eval()

        print("Opening Tenstorrent device...")
        device = ttnn.open_device(device_id=0)

        print("Creating TTNN model (Stage 1)...")
        opt_config = OptimizationConfig.stage1()
        ttnn_model = TtnnYolos(
            config=config,
            device=device,
            reference_model=pytorch_model,
            opt_config=opt_config,
        )
        print("✓ TTNN model created!")

        # Test forward pass
        print("Testing TTNN forward pass...")
        dummy_input = torch.randn(1, 3, 512, 864)

        # Convert to TTNN, using dtype appropriate for Stage 1
        from yolos_ttnn.common import OptimizationConfig, convert_to_ttnn_tensor, get_dtype_for_stage

        input_dtype = get_dtype_for_stage(OptimizationConfig.stage1())
        ttnn_input = convert_to_ttnn_tensor(dummy_input, device, dtype=input_dtype)

        print("Running inference...")
        logits, boxes = ttnn_model(ttnn_input)

        print(f"✓ TTNN inference successful!")
        print(f"  Logits shape: {ttnn.to_torch(logits).shape}")
        print(f"  Boxes shape: {ttnn.to_torch(boxes).shape}")

        ttnn.close_device(device)

        print("\n" + "🎉" * 40)
        print("SUCCESS! YOLOS-small Stage 1 works on Tenstorrent hardware!")
        print("🎉" * 40)

        return True

    except Exception as e:
        print(f"✗ TTNN Stage 1 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "YOLOS-small Tenstorrent Hardware Test" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Device Access", test_device_access),
        ("Simple TTNN Operation", test_simple_ttnn_op),
        ("PyTorch Reference Model", test_yolos_reference),
        ("TTNN Stage 1", test_yolos_ttnn_stage1),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            if not result:
                print(f"\n⚠️  Test '{name}' failed, stopping here.")
                break
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
            break

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n" + "🎉" * 40)
        print("ALL TESTS PASSED!")
        print("YOLOS-small is ready on Tenstorrent hardware!")
        print("🎉" * 40)
        print("\nNext steps:")
        print("  - Run: python3 demo.py --stage 1")
        print("  - Run: python3 demo.py --stage 2")
        print("  - Run: python3 demo.py --stage 3 --benchmark")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""
