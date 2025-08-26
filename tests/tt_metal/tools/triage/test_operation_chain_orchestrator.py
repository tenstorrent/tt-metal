# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest, os, yaml, glob, subprocess, time
import ttnn

# How long to wait for operations before considering them hung
OPERATION_TIMEOUT_SECONDS = 15


def get_llk_header_path():
    """Get the path to the LLK header file"""
    return "tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h"


def add_infinite_loop_to_llk_header():
    """Add the infinite loop to the LLK header file"""
    header_path = get_llk_header_path()

    with open(header_path, "r") as f:
        content = f.read()

    # Check if the infinite loop is already there
    if "for (;;) {}  // Infinite loop to test timeout - only for ADD operations" in content:
        print("Infinite loop already present in LLK header")
        return True

    # Add the infinite loop - look for the more specific pattern
    target = "    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))\n    {\n        if constexpr (src_b_bcast_type == BroadcastType::COL)"
    replacement = """    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))
    {
        if constexpr (eltwise_binary_type == ELWADD) {
            for (;;) {}  // Infinite loop to test timeout - only for ADD operations
        }
        if constexpr (src_b_bcast_type == BroadcastType::COL)"""

    modified_content = content.replace(target, replacement)

    if content != modified_content:
        with open(header_path, "w") as f:
            f.write(modified_content)
        print("Added infinite loop to LLK header for ADD operations")
        return True

    print("Failed to add infinite loop to LLK header - target pattern not found")
    return False


def remove_infinite_loop_from_llk_header():
    """Remove the infinite loop from the LLK header file"""
    header_path = get_llk_header_path()

    with open(header_path, "r") as f:
        content = f.read()

    # Remove the infinite loop block (including any extra newlines)
    target = """        if constexpr (eltwise_binary_type == ELWADD) {
            for (;;) {}  // Infinite loop to test timeout - only for ADD operations
        }
"""

    modified_content = content.replace(target, "")

    if content != modified_content:
        with open(header_path, "w") as f:
            f.write(modified_content)
        print("Removed infinite loop from LLK header")
        return True

    print("Infinite loop not found in LLK header")
    return False


def run_dump_ops():
    """Run dump_ops.py script to show hanging operation if any"""
    print("\n=== Running dump_ops.py to analyze potential hang ===")
    try:
        result = subprocess.run(
            ["python", "./scripts/debugging_scripts/dump_ops.py", "--mapping-file=generated/inspector/ops/ops.yaml"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.stdout:
            print("dump_ops.py output:")
            print(result.stdout)
        if result.stderr:
            print("dump_ops.py errors:")
            print(result.stderr)

        return result.stdout, result.returncode == 0
    except Exception as e:
        print(f"Error running dump_ops.py: {e}")
        return "", False


def reset_device():
    """Reset the device using tt-smi"""
    print("Resetting device with tt-smi -r...")
    try:
        result = subprocess.run(["tt-smi", "-r"], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("Device reset successful")
            return True
        else:
            print(f"Device reset failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error resetting device: {e}")
        return False


def run_operation_chain_script(timeout_seconds=None):
    """Run the operation chain script with optional timeout"""
    script_path = "tests/tt_metal/tools/triage/run_operation_chain.py"

    try:
        if timeout_seconds:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ, "TT_METAL_HOME": os.getcwd(), "PYTHONPATH": os.getcwd()},
            )
        else:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                env={**os.environ, "TT_METAL_HOME": os.getcwd(), "PYTHONPATH": os.getcwd()},
            )

        return result.stdout, result.stderr, result.returncode, False  # False = not timed out

    except subprocess.TimeoutExpired:
        return "", f"Operation timed out after {timeout_seconds} seconds", -1, True  # True = timed out


def test_operation_chain_timeout_workflow():
    """Test complete operation chain timeout detection and recovery workflow"""

    print("=== PHASE 1: Testing timeout behavior with infinite loop ===")

    # Step 1: Add infinite loop to LLK header
    print("Step 1: Adding infinite loop to LLK header for ADD operations")
    if not add_infinite_loop_to_llk_header():
        pytest.fail("Failed to add infinite loop to LLK header")

    try:
        # Step 2: Run operation script that should timeout
        print(f"Step 2: Running operation script with {OPERATION_TIMEOUT_SECONDS}s timeout")
        stdout, stderr, returncode, timed_out = run_operation_chain_script(OPERATION_TIMEOUT_SECONDS)

        if not timed_out:
            pytest.fail("Operation script should have timed out but completed unexpectedly")

        print(f"✓ Operation script timed out as expected after {OPERATION_TIMEOUT_SECONDS} seconds")

        # Step 3: Dump ops and verify correct output
        print("Step 3: Running dump_ops to analyze the hang")
        dump_output, dump_success = run_dump_ops()

        if not dump_success:
            pytest.fail("dump_ops.py failed to run")

        # Verify that dump_ops shows the hang on ADD operation
        if "ttnn.add" not in dump_output:
            pytest.fail("dump_ops output should show ttnn.add operation as the hang point")

        print("✓ dump_ops correctly identified the hang on ADD operation")

        # Step 4: Reset device and revert LLK header change
        print("Step 4: Resetting device and reverting LLK header")

        if not reset_device():
            print("Warning: Device reset failed, continuing anyway")

        if not remove_infinite_loop_from_llk_header():
            pytest.fail("Failed to remove infinite loop from LLK header")

        print("✓ Device reset and LLK header reverted")

        # Give device time to reset
        time.sleep(3)

        print("\n=== PHASE 2: Testing normal behavior without infinite loop ===")

        # Step 5: Run again and make sure it passes
        print("Step 5: Running operation script again - should complete normally")
        stdout, stderr, returncode, timed_out = run_operation_chain_script(OPERATION_TIMEOUT_SECONDS)

        if timed_out:
            pytest.fail("Operation script should not have timed out after removing infinite loop")

        if returncode != 0:
            pytest.fail(f"Operation script failed after removing infinite loop: {stderr}")

        if "Test completed successfully" not in stdout:
            pytest.fail("Operation script did not complete successfully after removing infinite loop")

        print("✓ Operation script completed successfully without infinite loop")
        print("\n=== ALL PHASES COMPLETED SUCCESSFULLY ===")
        print("✓ Demonstrated timeout detection using subprocess timeout")
        print("✓ Verified dump_ops analysis of hung operations")
        print("✓ Confirmed device reset and LLK header revert functionality")
        print("✓ Validated normal operation after cleanup")

    except Exception as e:
        # Cleanup on failure
        print(f"\n=== CLEANUP ON FAILURE: {e} ===")
        reset_device()
        remove_infinite_loop_from_llk_header()
        raise


def test_operation_chain_normal():
    """Test that operation chain works normally without infinite loop"""

    # Ensure no infinite loop is present
    remove_infinite_loop_from_llk_header()

    print("=== Testing normal operation chain execution ===")
    stdout, stderr, returncode, timed_out = run_operation_chain_script(30)  # 30s timeout for safety

    if timed_out:
        pytest.fail("Normal operation chain should not timeout")

    if returncode != 0:
        pytest.fail(f"Normal operation chain failed: {stderr}")

    if "Test completed successfully" not in stdout:
        pytest.fail("Normal operation chain did not complete successfully")

    print("✓ Normal operation chain completed successfully")
