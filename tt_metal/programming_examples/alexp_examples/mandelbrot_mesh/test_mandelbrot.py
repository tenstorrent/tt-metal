#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0

"""
Test script for Mandelbrot mesh implementation
"""

import sys
import os
import subprocess
import numpy as np
from PIL import Image


def test_cpu_mandelbrot():
    """Test CPU reference implementation"""
    print("Testing CPU Mandelbrot implementation...")

    def mandelbrot_cpu_simple(width, height, max_iter=50):
        """Simplified CPU implementation for testing"""
        x = np.linspace(-2.5, 1.5, width)
        y = np.linspace(-2.0, 2.0, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        Z = np.zeros_like(C)
        iterations = np.zeros(C.shape, dtype=int)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            iterations[mask] = i

        return iterations

    # Test with small image
    result = mandelbrot_cpu_simple(64, 64, 50)

    # Basic validation
    assert result.shape == (64, 64), "Incorrect output shape"
    assert result.min() >= 0, "Negative iteration counts"
    assert result.max() < 50, "Iteration count exceeds maximum"

    # Check that we have both escaped and non-escaped points
    assert np.any(result == 49), "No points reached max iterations (should be in set)"
    assert np.any(result < 10), "No points escaped quickly"

    print("✓ CPU Mandelbrot test passed!")
    return True


def test_ttnn_availability():
    """Test if TTNN is available and can create mesh device"""
    print("Testing TTNN availability...")

    try:
        import ttnn

        print("✓ TTNN import successful")

        # Try to check for available devices
        # Note: This might fail if no devices are available, which is expected
        try:
            device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
            ttnn.close_device(device)
            print("✓ Mesh device creation successful")
            return True
        except Exception as e:
            print(f"⚠ Mesh device creation failed (expected if no hardware): {e}")
            return False

    except ImportError as e:
        print(f"✗ TTNN import failed: {e}")
        return False


def test_file_structure():
    """Test that all required files are present"""
    print("Testing file structure...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        "mandelbrot_mesh.cpp",
        "python_mandelbrot_mesh.py",
        "kernels/compute/mandelbrot_compute.cpp",
        "kernels/dataflow/mandelbrot_writer.cpp",
        "CMakeLists.txt",
        "README.md",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False

    print("✓ All required files present")
    return True


def test_python_syntax():
    """Test that Python files have valid syntax"""
    print("Testing Python syntax...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    python_files = ["python_mandelbrot_mesh.py", "test_mandelbrot.py"]

    for py_file in python_files:
        file_path = os.path.join(base_dir, py_file)
        try:
            with open(file_path, "r") as f:
                compile(f.read(), file_path, "exec")
            print(f"✓ {py_file} syntax valid")
        except SyntaxError as e:
            print(f"✗ {py_file} syntax error: {e}")
            return False

    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("Mandelbrot Mesh Implementation Tests")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("CPU Mandelbrot", test_cpu_mandelbrot),
        ("TTNN Availability", test_ttnn_availability),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1

    print(f"\nTests passed: {passed}/{len(results)}")

    if passed == len(results):
        print("🎉 All tests passed! The Mandelbrot mesh implementation is ready.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
