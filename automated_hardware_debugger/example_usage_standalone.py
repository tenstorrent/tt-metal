#!/usr/bin/env python3
"""
Example usage of the Automated Hardware Debugging Tool

This script shows various ways to use the standalone debugging tool
for different hardware operations and configurations.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display the results"""
    print(f"\n{'='*80}")
    print(f"EXAMPLE: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            if result.stdout:
                print("Output:")
                print(result.stdout)

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT - Command took too long")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def main():
    """Demonstrate various usage examples"""
    print("🚀 AUTOMATED HARDWARE DEBUGGING TOOL - USAGE EXAMPLES")
    print("=" * 80)
    print("This script demonstrates different ways to use the debugging tool")
    print("Note: These examples assume you have the test environment set up")

    # Check if the tool exists
    tool_path = "./automated_hardware_debugger.py"
    if not os.path.exists(tool_path):
        print(f"❌ Error: Tool not found at {tool_path}")
        print("Please ensure automated_hardware_debugger.py is in the current directory")
        sys.exit(1)

    # Example 1: Basic usage with permute test
    run_command(
        [
            "python3",
            tool_path,
            "--test-file",
            "tests/ttnn/unit_tests/operations/test_permute.py",
            "--function",
            "test_permute_5d_blocked",
            "--max-nops",
            "20",  # Reduced for faster example
            "--iterations",
            "3",  # Reduced for faster example
        ],
        "Basic debugging of permute 5D blocked test",
    )

    # Example 2: Quick debugging with minimal parameters
    run_command(
        [
            "python3",
            tool_path,
            "--test-file",
            "tests/ttnn/unit_tests/operations/test_permute.py",
            "--function",
            "test_permute_5d_blocked",
            "--max-nops",
            "10",
            "--iterations",
            "2",
        ],
        "Quick debugging with minimal parameters",
    )

    # Example 3: Custom backup directory
    run_command(
        [
            "python3",
            tool_path,
            "--test-file",
            "tests/ttnn/unit_tests/operations/test_permute.py",
            "--function",
            "test_permute_5d_blocked",
            "--backup-dir",
            "/tmp/debug_backups_example",
            "--max-nops",
            "15",
            "--iterations",
            "3",
        ],
        "Using custom backup directory",
    )

    # Example 4: Show help
    run_command(["python3", tool_path, "--help"], "Display help information")

    print("\n" + "=" * 80)
    print("EXAMPLE USAGE PATTERNS")
    print("=" * 80)

    print(
        """
## For Different Operations:

# Debug permute operation
./automated_hardware_debugger.py \\
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \\
    --function test_permute_5d_blocked

# Debug convolution operation (if available)
./automated_hardware_debugger.py \\
    --test-file tests/ttnn/unit_tests/operations/test_conv2d.py \\
    --function test_conv2d_basic

# Debug matrix multiplication (if available)
./automated_hardware_debugger.py \\
    --test-file tests/ttnn/unit_tests/operations/test_matmul.py \\
    --function test_matmul_2d

## For Different Performance Requirements:

# Fast debugging (good for development)
./automated_hardware_debugger.py \\
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \\
    --function test_permute_5d_blocked \\
    --max-nops 25 \\
    --iterations 3

# Thorough debugging (good for analysis)
./automated_hardware_debugger.py \\
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \\
    --function test_permute_5d_blocked \\
    --max-nops 100 \\
    --iterations 10

# Ultra-fast debugging (good for CI/CD)
./automated_hardware_debugger.py \\
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \\
    --function test_permute_5d_blocked \\
    --max-nops 10 \\
    --iterations 2

## Integration with Build Systems:

# Makefile target
debug-permute:
\tpython3 automated_hardware_debugger.py \\
\t\t--test-file tests/ttnn/unit_tests/operations/test_permute.py \\
\t\t--function test_permute_5d_blocked \\
\t\t--max-nops 50 \\
\t\t--iterations 5

# GitHub Actions workflow
- name: Run Hardware Debugging
  run: |
    python3 automated_hardware_debugger.py \\
      --test-file tests/ttnn/unit_tests/operations/test_permute.py \\
      --function test_permute_5d_blocked \\
      --max-nops 30 \\
      --iterations 3

# Jenkins pipeline
stage('Hardware Debug') {
    steps {
        script {
            sh '''
                python3 automated_hardware_debugger.py \\
                  --test-file tests/ttnn/unit_tests/operations/test_permute.py \\
                  --function test_permute_5d_blocked \\
                  --max-nops 40 \\
                  --iterations 4
            '''
        }
    }
}
    """
    )

    print("\n📋 Examples completed!")
    print("Check the output above to see how the tool works with different configurations.")
    print("Remember: All file modifications are temporary and automatically restored!")


if __name__ == "__main__":
    main()
