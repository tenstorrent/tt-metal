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
        # Enable Inspector RPC for dump_ops
        env = {
            **os.environ,
            "TT_METAL_HOME": os.getcwd(),
            "PYTHONPATH": os.getcwd(),
            "TT_METAL_INSPECTOR_LOG_PATH": "generated/inspector",
        }

        # Try with Inspector RPC first, fall back to mapping file
        result = subprocess.run(
            ["python", "./scripts/debugging_scripts/dump_ops.py"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

        if result.stdout:
            print("\ndump_ops.py output:")
            print("-" * 40)
            print(result.stdout)
            print("-" * 40)
        if result.stderr and "Error" in result.stderr:
            print("\ndump_ops.py errors:")
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


def start_operation_chain_in_background():
    """Start the operation chain script in background and return the process"""
    script_path = "tests/tt_metal/tools/triage/run_operation_chain.py"

    # Enable Inspector RPC for dump_ops to work
    env = {
        **os.environ,
        "TT_METAL_HOME": os.getcwd(),
        "PYTHONPATH": os.getcwd(),
        "TT_METAL_INSPECTOR_LOG_PATH": "generated/inspector",
        "PYTHONUNBUFFERED": "1",  # Force unbuffered output
    }

    process = subprocess.Popen(
        ["python", "-u", script_path],  # -u for unbuffered
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=0,  # No buffering
        env=env,
    )

    return process


def test_normal_operation_chain():
    """Test that operation chain works normally without infinite loop"""
    try:
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

    finally:
        # Always clean up
        remove_infinite_loop_from_llk_header()
        # Clean up any serialized files
        import glob

        for f in glob.glob("generated/inspector/*.capnp.bin"):
            try:
                os.remove(f)
            except:
                pass


def test_timeout_workflow():
    """Test complete operation chain timeout detection and recovery workflow"""

    print("=== PHASE 1: Testing timeout behavior with infinite loop ===")

    # Step 1: Add infinite loop to LLK header
    print("Step 1: Adding infinite loop to LLK header for ADD operations")
    if not add_infinite_loop_to_llk_header():
        pytest.fail("Failed to add infinite loop to LLK header")

    process = None
    try:
        # Step 2: Start operation script in background (it will hang)
        print("Step 2: Starting operation script in background (will hang on ADD)")
        process = start_operation_chain_in_background()

        # Monitor subprocess output to see when it reaches the hang point
        print("\n=== Monitoring subprocess output ===")
        import select
        import fcntl

        fl = fcntl.fcntl(process.stdout.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)

        start_time = time.time()
        timeout = 15
        hang_detected = False

        while time.time() - start_time < timeout:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                try:
                    line = process.stdout.readline()
                    if line:
                        print(f"SUBPROCESS: {line.rstrip()}")
                        # Check if we've reached the hang point
                        if "Step 4: result3 + 1.0" in line:
                            print("\n⚠️ Subprocess reached ADD operation (potential hang point)")
                            hang_detected = True
                            time.sleep(3)  # Give it time to actually hang
                            break
                except:
                    pass

            # Check if process terminated
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"Process unexpectedly terminated. stdout: {stdout}, stderr: {stderr}")

        if not hang_detected:
            print("Warning: Did not detect hang message, but continuing...")

        print("=== End subprocess monitoring ===\n")

        # Check if process is still running (it should be hanging)
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Process unexpectedly terminated. stdout: {stdout}, stderr: {stderr}")

        print("✓ Operation script is hanging as expected")

        # Step 3: Run dump_ops while the process is still hanging
        print("Step 3: Running dump_ops to analyze the hang (while process is still running)")
        dump_output, dump_success = run_dump_ops()

        if not dump_success:
            pytest.fail("dump_ops.py failed to run")

        # Verify that dump_ops shows the hang on ADD operation
        if "ttnn.add" not in dump_output and "ttnn::add" not in dump_output:
            print(f"dump_ops output: {dump_output}")
            pytest.fail("dump_ops output should show ttnn.add or ttnn::add operation as the hang point")

        print("✓ dump_ops correctly identified the hang on ADD operation")

        # Parse and show details about the hanging ADD operations
        import re

        add_count = dump_output.count("ttnn::add") + dump_output.count("ttnn.add")
        print(f"   Found ADD operations on {add_count} core(s)")

        # Show which cores have the ADD operation
        matches = re.findall(r"(\d+\s*/\s*\d+,\d+)\s*│\s*ttnn[:.]add", dump_output)
        if matches:
            print(f"   ADD operations hanging on cores: {', '.join(matches)}")

        # Step 4: Kill the hanging process
        print("Step 4: Terminating the hanging process")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            print("✓ Hanging process terminated")

        # Step 5: Reset device and revert LLK header change
        print("Step 5: Resetting device and reverting LLK header")

        if not reset_device():
            print("Warning: Device reset failed, continuing anyway")

        if not remove_infinite_loop_from_llk_header():
            pytest.fail("Failed to remove infinite loop from LLK header")

        print("✓ Device reset and LLK header reverted")

        # Give device time to reset
        time.sleep(3)

        print("\n=== PHASE 2: Testing normal behavior without infinite loop ===")

        # Step 6: Run again and make sure it passes
        print("Step 6: Running operation script again - should complete normally")
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
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
                process.wait()
        reset_device()
        remove_infinite_loop_from_llk_header()
        raise
    finally:
        # Make sure process is terminated
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                process.kill()
                process.wait()


def test_hang_detection_with_dump_ops():
    """Test operation chain with hang detection using dump_ops"""
    import select
    import fcntl
    import re

    print("=" * 60)
    print("TEST: Operation Chain with Hang Detection")
    print("=" * 60)

    process = None
    try:
        # Add infinite loop to cause hang
        print("\n1. Adding infinite loop to LLK header to simulate hang...")
        if not add_infinite_loop_to_llk_header():
            pytest.fail("Failed to add infinite loop to LLK header")
        print("   ✓ Infinite loop added to ADD operations")

        print("\n2. Starting operation chain subprocess...")
        process = start_operation_chain_in_background()
        print(f"   Process PID: {process.pid}")

        # Monitor subprocess output
        print("\n3. Monitoring subprocess output:")
        print("-" * 40)

        # Make stdout non-blocking
        fl = fcntl.fcntl(process.stdout.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)

        start_time = time.time()
        timeout = 10
        operation_started = False

        while time.time() - start_time < timeout:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                try:
                    line = process.stdout.readline()
                    if line:
                        print(f"   {line.rstrip()}")
                        if "Step 4:" in line or "may hang" in line:
                            operation_started = True
                            print("\n   ⚠️  Reached ADD operation (will hang)")
                            time.sleep(3)  # Give it time to hang
                            break
                except:
                    pass

            # Check if process terminated unexpectedly
            if process.poll() is not None:
                pytest.fail(f"Process terminated unexpectedly with code: {process.poll()}")

        print("-" * 40)

        # Verify process is still running (hung)
        if process.poll() is not None:
            pytest.fail("Process should be hanging but has terminated")

        if not operation_started:
            pytest.fail("Did not reach ADD operation")

        print("\n4. Process is hanging - running dump_ops.py:")
        dump_output, success = run_dump_ops()

        if not success:
            pytest.fail("dump_ops failed to run")

        # Verify dump_ops shows ADD operations
        if "ttnn::add" not in dump_output and "ttnn.add" not in dump_output:
            pytest.fail("dump_ops should show ttnn.add or ttnn::add operations")

        print("   ✅ dump_ops correctly identified ADD operations hanging on device!")

        # Count cores with ADD operations
        add_count = dump_output.count("ttnn::add") + dump_output.count("ttnn.add")
        print(f"   Found ADD operations on {add_count} core(s)")

        # Show which cores have the ADD operation
        matches = re.findall(r"(\d+\s*/\s*\d+,\d+)\s*│\s*ttnn[:.]add", dump_output)
        if matches:
            print(f"   ADD operations found on cores: {', '.join(matches)}")

        print("\n✓ Test passed: Hang detection with dump_ops works correctly")

    finally:
        # Clean up
        print("\n5. Cleaning up...")

        # Kill hanging process
        if process and process.poll() is None:
            print("   Terminating hanging subprocess...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        # Remove infinite loop
        if not remove_infinite_loop_from_llk_header():
            print("   ⚠️  Failed to remove infinite loop from LLK header")
        else:
            print("   ✓ Infinite loop removed from LLK header")

        # Reset device
        if not reset_device():
            print("   ⚠️  Device reset failed")
        else:
            print("   ✓ Device reset successful")

        # Clean up any serialized files
        import glob

        for f in glob.glob("generated/inspector/*.capnp.bin"):
            try:
                os.remove(f)
            except:
                pass

    print("=" * 60)


def test_timeout_with_rpc_serialization():
    """Test operation timeout with RPC serialization on exit"""
    import select
    import fcntl
    import re

    print("=" * 60)
    print("TEST: Operation Timeout with RPC Serialization")
    print("=" * 60)

    # Clean up any old serialized files first
    print("\n1. Cleaning up old serialized Inspector files...")
    import glob

    old_files = glob.glob("generated/inspector/*.capnp.bin")
    for f in old_files:
        os.remove(f)
    print(f"   Removed {len(old_files)} old serialized files")

    # Add infinite loop to cause hang
    print("\n2. Adding infinite loop to LLK header to cause hang...")
    if not add_infinite_loop_to_llk_header():
        print("   ⚠️  Failed to add infinite loop")
        return False

    print("   ✓ Infinite loop added to ADD operations")

    process = None
    try:
        # Start subprocess with operation timeout
        print("\n3. Starting operation chain with 10 second timeout...")
        script_path = "tests/tt_metal/tools/triage/run_operation_chain.py"

        env = {
            **os.environ,
            "TT_METAL_HOME": os.getcwd(),
            "PYTHONPATH": os.getcwd(),
            "TT_METAL_INSPECTOR_LOG_PATH": "generated/inspector",
            "TT_METAL_OPERATION_TIMEOUT_SECONDS": "10",  # Set 10 second timeout for operations
        }

        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        print(f"   Process PID: {process.pid}")

        # Monitor subprocess output
        print("\n4. Waiting for subprocess to timeout (expecting ~10 seconds):")
        print("-" * 40)

        start_time = time.time()
        max_wait = 20  # Maximum time to wait for process
        timeout_detected = False
        rpc_serialized = False
        process_output = []

        try:
            # Wait for process with timeout
            stdout, stderr = process.communicate(timeout=max_wait)

            # Process terminated, check exit code
            exit_code = process.returncode
            print(f"\n   Process terminated with exit code: {exit_code}")

            # Process stdout
            for line in stdout.splitlines():
                if line.strip():
                    process_output.append(f"   STDOUT: {line}")
                    if "timeout detected" in line.lower():
                        timeout_detected = True
                    if "serializ" in line.lower():
                        rpc_serialized = True

            # Process stderr
            for line in stderr.splitlines():
                if line.strip():
                    process_output.append(f"   STDERR: {line}")
                    if "timeout" in line.lower():
                        timeout_detected = True
                    if "serializ" in line.lower():
                        rpc_serialized = True

            # Print last 30 lines of output
            print("\n   Last 30 lines of process output:")
            for line in process_output[-30:]:
                print(line)
                if "timeout detected" in line.lower():
                    print("   ✅ TIMEOUT DETECTED!")
                if "serializ" in line.lower() and "Inspector" in line:
                    print("   ✅ RPC SERIALIZATION DETECTED!")

        except subprocess.TimeoutExpired:
            print(f"\n   ⚠️  Process did not terminate within {max_wait} seconds")
            # Kill the process
            process.kill()
            stdout, stderr = process.communicate()
            print("   Process killed")

        elapsed = time.time() - start_time
        print(f"\n   Process ran for {elapsed:.1f} seconds")
        print("-" * 40)

        # Check for serialized files
        print("\n5. Checking for serialized Inspector files:")
        serialized_files = glob.glob("generated/inspector/*.capnp.bin")
        if serialized_files:
            print(f"   ✓ Found {len(serialized_files)} serialized files:")
            for f in serialized_files:
                print(f"     - {os.path.basename(f)}")
        else:
            print("   ⚠️  No serialized files found!")

        # Now try to run dump_ops - it should use serialized data since RPC is down
        print("\n6. Running dump_ops after timeout (RPC should be down):")
        dump_output, dump_success = run_dump_ops()

        if dump_success:
            print("   ✓ dump_ops ran successfully")

            # Check if it's using serialized data or fallback
            if "using serialized" in dump_output.lower() or "fallback" in dump_output.lower():
                print("   ✓ dump_ops using serialized/fallback data (RPC is down)")
            elif "ttnn::add" in dump_output or "ttnn.add" in dump_output:
                print("   ✓ dump_ops found ADD operations")
                # Count how many cores show ADD operations
                add_count = dump_output.count("ttnn::add") + dump_output.count("ttnn.add")
                if add_count > 0:
                    print(f"   Found ADD operations on {add_count} core(s)")
        else:
            print("   ⚠️  dump_ops failed to run")

        # Summary
        print("\n7. Test Summary:")
        if timeout_detected:
            print("   ✓ Operation timeout was detected")
        else:
            print("   ⚠️  Operation timeout was NOT detected")

        if rpc_serialized:
            print("   ✓ RPC serialization occurred on timeout")
        else:
            print("   ⚠️  RPC serialization was NOT detected")

        if serialized_files:
            print(f"   ✓ Serialized files were created ({len(serialized_files)} files)")
        else:
            print("   ⚠️  No serialized files were created")

        success = timeout_detected or dump_success

    finally:
        # Clean up
        print("\n8. Cleaning up...")

        # Kill process if still running
        if process and process.poll() is None:
            print("   Terminating subprocess...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        # Remove infinite loop
        if remove_infinite_loop_from_llk_header():
            print("   ✓ Infinite loop removed from LLK header")
        else:
            print("   ⚠️  Failed to remove infinite loop from LLK header")

        # Reset device
        if reset_device():
            print("   ✓ Device reset successful")
        else:
            print("   ⚠️  Device reset failed")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    return success


if __name__ == "__main__":
    # Run the demo when script is executed directly
    import sys

    # Check for different test modes
    if "--test-timeout" in sys.argv:
        print("Running timeout test with RPC serialization")
        test_operation_timeout_with_rpc_serialization()
    elif "--with-hang" in sys.argv:
        print("Running demo WITH hang simulation (will modify LLK header)")
        demo_subprocess_with_dump_ops(with_hang=True)
    else:
        print("Running demo WITHOUT hang simulation")
        demo_subprocess_with_dump_ops(with_hang=False)
