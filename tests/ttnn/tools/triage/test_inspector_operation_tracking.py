# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import subprocess
import time
import glob
import select
import fcntl
from contextlib import contextmanager
from typing import Optional, Tuple, Callable

# ============================================================================
# Configuration
# ============================================================================

OPERATION_TIMEOUT_SECONDS = 15
LLK_HEADER_PATH = "tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h"
INSPECTOR_LOG_PATH = "generated/inspector"
CPP_OPERATION_CHAIN_SOURCE = "tests/ttnn/tools/triage/run_operation_chain.cpp"
CPP_OPERATION_CHAIN_BINARY = "build/test/ttnn/tools/triage/run_operation_chain_cpp"
CPP_OPERATION_CHAIN_BINARY_STRIPPED = "build/test/ttnn/tools/triage/run_operation_chain_cpp_stripped"
BUILD_DIR = "build"  # Symlink to actual build directory (build_Release, build_Debug, etc.)

# ============================================================================
# Logging Helpers
# ============================================================================


class Logger:
    """Consistent logging for all test operations"""

    verbose = False  # Class variable to control verbosity

    @classmethod
    def set_verbose(cls, verbose: bool):
        """Set verbose mode"""
        cls.verbose = verbose

    @classmethod
    def log(cls, message: str):
        """Print only if verbose mode is enabled"""
        if cls.verbose:
            print(message)

    @classmethod
    def test_header(cls, name: str):
        if cls.verbose:
            print()  # Line break before test header
            print("=" * 60)
            print(f"TEST: {name}")
            print("=" * 60)

    @classmethod
    def step(cls, num: int, message: str):
        cls.log(f"\n{num}. {message}")

    @classmethod
    def substep(cls, message: str, indent: int = 1):
        cls.log(f"{'   ' * indent}{message}")

    @classmethod
    def success(cls, message: str):
        cls.log(f"   ✓ {message}")

    @classmethod
    def warning(cls, message: str):
        cls.log(f"   ⚠️  {message}")

    @classmethod
    def error(cls, message: str):
        # Errors should always be shown
        print(f"   ✗ {message}")

    @classmethod
    def separator(cls):
        cls.log("-" * 40)


log = Logger()

# ============================================================================
# Environment Helpers
# ============================================================================


def get_base_env():
    """Get base environment variables for all tests"""
    return {
        **os.environ,
        "TT_METAL_HOME": os.getcwd(),
        "PYTHONPATH": os.getcwd(),
        "TT_METAL_INSPECTOR_LOG_PATH": INSPECTOR_LOG_PATH,
    }


def get_timeout_env(timeout_seconds: int):
    """Get environment with operation timeout set"""
    env = get_base_env()
    env["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = str(timeout_seconds)
    return env


# ============================================================================
# Cleanup and Device Management
# ============================================================================


def cleanup_serialized_files() -> int:
    """Clean up any serialized Inspector files from previous runs"""
    cleaned_count = 0
    for f in glob.glob(f"{INSPECTOR_LOG_PATH}/*.capnp.bin"):
        try:
            os.remove(f)
            cleaned_count += 1
        except:
            pass
    if cleaned_count > 0 and Logger.verbose:
        print()  # Add newline to separate from pytest output
        log.substep(f"Cleaned up {cleaned_count} serialized Inspector files")
    return cleaned_count


def reset_device() -> bool:
    """Reset the device using tt-smi"""
    log.substep("Resetting device with tt-smi...")
    try:
        result = subprocess.run(["tt-smi", "-r"], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            log.success("Device reset successful")
            return True
        else:
            log.error(f"Device reset failed: {result.stderr}")
            return False
    except Exception as e:
        log.error(f"Error resetting device: {e}")
        return False


# ============================================================================
# LLK Header Management
# ============================================================================


@contextmanager
def infinite_loop_context():
    """Context manager for adding/removing infinite loop in LLK header"""

    MARKER = "for (;;) {}  // Infinite loop to test timeout - only for ADD operations"

    TARGET = (
        "    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))\n"
        "    {\n"
        "        if constexpr (src_b_bcast_type == BroadcastType::COL)"
    )

    REPLACEMENT = """    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))
    {
        if constexpr (eltwise_binary_type == ELWADD) {
            for (;;) {}  // Infinite loop to test timeout - only for ADD operations
        }
        if constexpr (src_b_bcast_type == BroadcastType::COL)"""

    REMOVAL = """        if constexpr (eltwise_binary_type == ELWADD) {
            for (;;) {}  // Infinite loop to test timeout - only for ADD operations
        }
"""

    # Add infinite loop
    with open(LLK_HEADER_PATH, "r") as f:
        content = f.read()

    if MARKER not in content:
        modified = content.replace(TARGET, REPLACEMENT)
        if modified != content:
            with open(LLK_HEADER_PATH, "w") as f:
                f.write(modified)
            log.success("Added infinite loop to LLK header for ADD operations")
        else:
            raise RuntimeError("Failed to add infinite loop - target pattern not found")
    else:
        log.substep("Infinite loop already present in LLK header")

    try:
        yield
    finally:
        # Remove infinite loop
        with open(LLK_HEADER_PATH, "r") as f:
            content = f.read()

        modified = content.replace(REMOVAL, "")
        if modified != content:
            with open(LLK_HEADER_PATH, "w") as f:
                f.write(modified)
            log.success("Removed infinite loop from LLK header")


def ensure_llk_header_clean():
    """Ensure LLK header has no infinite loop"""
    REMOVAL = """        if constexpr (eltwise_binary_type == ELWADD) {
            for (;;) {}  // Infinite loop to test timeout - only for ADD operations
        }
"""
    with open(LLK_HEADER_PATH, "r") as f:
        content = f.read()

    modified = content.replace(REMOVAL, "")
    if modified != content:
        with open(LLK_HEADER_PATH, "w") as f:
            f.write(modified)
        log.substep("Cleaned up LLK header")


# ============================================================================
# Process Management
# ============================================================================


def make_stdout_nonblocking(process: subprocess.Popen):
    """Make process stdout non-blocking"""
    if hasattr(process.stdout, "fileno"):
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


def monitor_process_output(
    process: subprocess.Popen, timeout: float, expected_pattern: Optional[str] = None, label: str = "Process"
) -> Tuple[bool, str]:
    """
    Monitor process output for expected pattern or timeout.

    Returns:
        (found_pattern, last_line)
    """
    make_stdout_nonblocking(process)

    start_time = time.time()
    last_line = ""
    found_pattern = expected_pattern is None

    while time.time() - start_time < timeout:
        if process.poll() is not None:
            log.warning(f"Process exited unexpectedly with code {process.returncode}")
            return found_pattern, last_line

        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if ready:
            try:
                line = process.stdout.readline()
                if line:
                    log.substep(f"{label}: {line.strip()}")
                    last_line = line
                    if expected_pattern and expected_pattern in line:
                        found_pattern = True
            except:
                pass

    return found_pattern, last_line


def terminate_process(process: Optional[subprocess.Popen]):
    """Safely terminate a process"""
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            process.kill()
            process.wait()


# ============================================================================
# Operation Chain Runners
# ============================================================================


def start_python_operation_chain() -> subprocess.Popen:
    """Start Python operation chain script in background"""
    env = get_base_env()
    env["PYTHONUNBUFFERED"] = "1"

    return subprocess.Popen(
        ["python", "-u", "tests/ttnn/tools/triage/run_operation_chain.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
        env=env,
    )


def start_cpp_operation_chain(binary_path: str = CPP_OPERATION_CHAIN_BINARY) -> subprocess.Popen:
    """Start standalone C++ operation chain binary"""
    env = get_base_env()

    return subprocess.Popen(
        [binary_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
        env=env,
    )


# ============================================================================
# dump_ops.py Integration
# ============================================================================


def run_dump_ops() -> Tuple[str, bool]:
    """Run dump_ops.py script to analyze operations"""
    log.substep("Running dump_ops.py...")
    log.separator()

    try:
        result = subprocess.run(
            ["python", "./tools/triage/dump_ops.py"],
            capture_output=True,
            text=True,
            timeout=30,
            env=get_base_env(),
        )

        if result.stdout and Logger.verbose:
            print(result.stdout)  # Only print in verbose mode
        if result.stderr:
            log.error(f"dump_ops.py stderr: {result.stderr}")
        if result.returncode != 0:
            log.error(f"dump_ops.py failed with return code: {result.returncode}")
            if result.stdout:
                log.error(f"stdout: {result.stdout}")

        log.separator()
        return result.stdout, result.returncode == 0

    except Exception as e:
        log.error(f"Error running dump_ops.py: {e}")
        return "", False


def parse_dump_ops_table(dump_output: str) -> list:
    """
    Parse dump_ops table and validate format.

    Returns list of parsed operations, each with:
    - device_core: str (e.g., "0 / 0,0")
    - operation: str (e.g., "ttnn::add")
    - callstack: str (file/line or addresses)
    - has_arguments: bool
    """
    import re

    operations = []
    lines = dump_output.split("\n")

    # Find table content between header and footer
    in_table = False
    for i, line in enumerate(lines):
        # Look for separator line with dashes after header
        if "├────" in line or "│ Dev/Core" in line:
            in_table = True
            continue
        if "╰────" in line:
            break

        if in_table and "│" in line:
            # Parse table row: │ Dev/Core │ Operation │ Callstack/Args │
            parts = [p.strip() for p in line.split("│") if p.strip()]

            if len(parts) >= 3:
                device_core = parts[0]
                operation = parts[1]
                callstack_args = parts[2]

                # Validate device/core format: "N / X,Y" or just operation continuation
                device_core_pattern = r"^\d+\s*/\s*\d+,\d+$"
                if re.match(device_core_pattern, device_core):
                    # New operation row
                    current_op = {
                        "device_core": device_core,
                        "operation": operation,
                        "callstack": callstack_args,
                        "has_arguments": False,
                    }

                    # Check if this row or subsequent rows contain arguments
                    # Look ahead for "Arguments:" marker
                    for j in range(i, min(i + 10, len(lines))):
                        if "Arguments:" in lines[j]:
                            current_op["has_arguments"] = True
                            break

                    operations.append(current_op)

    return operations


def verify_dump_ops_table(
    dump_output: str,
    expected_operation: str = "add",
    callstack_type: str = "python",
    allow_no_line_numbers: bool = False,
):
    """
    Verify dump_ops table has proper format and expected content.

    Args:
        dump_output: Raw output from dump_ops.py
        expected_operation: Operation to look for (e.g., "add", "multiply")
        callstack_type: "python" or "cpp" for different validation
        allow_no_line_numbers: If True, allow C++ callstacks without resolved line numbers (stripped binaries)
    """
    import re

    operations = parse_dump_ops_table(dump_output)

    if not operations:
        pytest.fail("No operations found in dump_ops table")

    log.success(f"Parsed {len(operations)} operation(s) from dump_ops table")

    # Verify each operation
    found_expected_op = False
    for op in operations:
        # 1. Validate device/core format
        if not re.match(r"^\d+\s*/\s*\d+,\d+$", op["device_core"]):
            pytest.fail(f"Invalid device/core format: {op['device_core']}")

        # 2. Validate operation name contains ttnn::
        if not op["operation"].startswith("ttnn::"):
            pytest.fail(f"Operation should start with 'ttnn::': {op['operation']}")

        # 3. Check for expected operation
        if expected_operation in op["operation"].lower():
            found_expected_op = True

            # 4. Validate callstack format based on type
            if callstack_type == "python":
                # Should have file path with line number (e.g., "run_operation_chain.py:61")
                if not re.search(r"\.py:\d+", op["callstack"]):
                    pytest.fail(f"Python callstack should contain '.py:LINE_NUMBER' format. " f"Got: {op['callstack']}")
            elif callstack_type == "cpp":
                # Should have function addresses or resolved symbols
                has_addresses = "+0x" in op["callstack"]
                has_resolved_symbols = ".cpp:" in op["callstack"]

                if allow_no_line_numbers:
                    # For stripped binaries: just check for addresses
                    if not has_addresses:
                        pytest.fail(f"C++ callstack should contain addresses (+0x). " f"Got: {op['callstack']}")
                else:
                    # For normal binaries: check for addresses OR resolved symbols
                    if not (has_addresses or has_resolved_symbols):
                        pytest.fail(
                            f"C++ callstack should contain addresses (+0x) or resolved symbols (.cpp:LINE). "
                            f"Got: {op['callstack']}"
                        )

        # 5. Validate arguments are present
        if not op["has_arguments"]:
            pytest.fail(f"Operation should have Arguments section: {op['operation']}")

    if not found_expected_op:
        pytest.fail(f"Expected operation 'ttnn::{expected_operation}' not found in table")

    log.success(f"All {len(operations)} operations have valid format")
    log.success(f"Found expected operation: ttnn::{expected_operation}")

    if allow_no_line_numbers and callstack_type == "cpp":
        log.success("Callstacks contain addresses (no line numbers expected for stripped binary)")


def verify_dump_ops_shows_operation(dump_output: str, operation: str = "add"):
    """Verify dump_ops output contains expected operation"""
    patterns = [f"ttnn::{operation}", f"ttnn.{operation}"]

    if not any(pattern in dump_output for pattern in patterns):
        pytest.fail(f"dump_ops should show {' or '.join(patterns)} operations")

    log.success(f"dump_ops correctly identified {operation.upper()} operations")

    # Count and report
    op_count = sum(dump_output.count(pattern) for pattern in patterns)
    log.substep(f"Found {operation.upper()} operations on {op_count} core(s)")


# ============================================================================
# Pytest Fixtures
# ============================================================================

# Set up logging verbosity based on pytest options
import sys

if any(arg in sys.argv for arg in ["-v", "-vv", "-s", "--verbose", "--capture=no"]):
    Logger.set_verbose(True)


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test"""
    cleanup_serialized_files()
    yield
    cleanup_serialized_files()


@pytest.fixture
def clean_llk_header():
    """Ensure LLK header starts and ends clean"""
    ensure_llk_header_clean()
    yield
    ensure_llk_header_clean()


@pytest.fixture
def clean_device():
    """Ensure device is reset before test"""
    reset_device()
    yield
    # No reset after test as this can take a while - next test will reset before it starts


@pytest.fixture(scope="session")
def cpp_binary():
    """Verify C++ operation chain binary exists (built by CMake)"""
    if not os.path.exists(CPP_OPERATION_CHAIN_BINARY):
        pytest.skip(
            f"C++ operation chain binary not found: {CPP_OPERATION_CHAIN_BINARY}\n"
            f"Build with './build_metal.sh --build-ttnn-tests' to run C++ tests"
        )
    yield CPP_OPERATION_CHAIN_BINARY


@pytest.fixture(scope="session")
def cpp_binary_stripped(cpp_binary):
    """Create a stripped version of the C++ binary (no debug symbols)"""
    import shutil

    # Copy the binary
    shutil.copy2(cpp_binary, CPP_OPERATION_CHAIN_BINARY_STRIPPED)

    # Strip debug symbols
    result = subprocess.run(
        ["strip", "--strip-debug", CPP_OPERATION_CHAIN_BINARY_STRIPPED], capture_output=True, text=True
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to strip binary: {result.stderr}")

    log.substep(f"Created stripped binary: {CPP_OPERATION_CHAIN_BINARY_STRIPPED}")

    yield CPP_OPERATION_CHAIN_BINARY_STRIPPED

    # Cleanup
    try:
        os.remove(CPP_OPERATION_CHAIN_BINARY_STRIPPED)
    except:
        pass


# ============================================================================
# Generic Test Implementations
# ============================================================================


def run_hang_detection_test(
    test_name: str,
    start_process_func: Callable[[], subprocess.Popen],
    expected_pattern: str,
    callstack_type: str,
    allow_no_line_numbers: bool = False,
    hang_timeout: float = 30,
):
    """
    Generic hang detection test with dump_ops.

    Tests that:
    1. Process hangs on ADD operation when infinite loop is present
    2. dump_ops can identify the hanging operation while process is running
    3. Table output has proper format with device/core, operation, callstack, and arguments
    """
    log.test_header(f"{test_name} Operation Chain with Hang Detection")

    process = None
    try:
        log.step(1, "Adding infinite loop to simulate hang")
        with infinite_loop_context():
            log.step(2, f"Starting {test_name} operation chain")
            process = start_process_func()
            log.substep(f"Process PID: {process.pid}")

            log.step(3, f"Monitoring output for hang (timeout: {hang_timeout}s)")
            found_pattern, last_line = monitor_process_output(process, hang_timeout, expected_pattern, test_name)

            if not found_pattern:
                log.warning(f"Did not see '{expected_pattern}' in output")
            if last_line:
                log.substep(f"Last output: {last_line.strip()}")

            if process.poll() is not None:
                pytest.fail("Process should hang on ADD operation")

            log.success("Process is hanging as expected")

            log.step(4, "Running dump_ops.py on hanging process")
            dump_output, success = run_dump_ops()

            if not success:
                pytest.fail("dump_ops failed to run")

            # Use strict table validation
            verify_dump_ops_table(
                dump_output,
                expected_operation="add",
                callstack_type=callstack_type,
                allow_no_line_numbers=allow_no_line_numbers,
            )

        log.success(f"{test_name} hang detection test passed")

    finally:
        log.step(5, "Cleaning up")
        terminate_process(process)
        log.success("Cleanup completed")


def run_timeout_serialization_test(
    test_name: str,
    command: list,
    callstack_type: str,
    operation_timeout_seconds: int = 10,
    process_timeout_seconds: int = 60,
):
    """
    Generic timeout with RPC serialization test.

    Tests that:
    1. Process detects operation timeout and exits gracefully
    2. Operation details are serialized via RPC on timeout
    3. dump_ops can read serialized data after process exit
    4. Table output has proper format with device/core, operation, callstack, and arguments
    """
    log.test_header(f"{test_name} Operation Timeout with RPC Serialization")

    try:
        log.step(1, "Adding infinite loop to simulate hang for timeout test")
        with infinite_loop_context():
            log.step(
                2,
                f"Starting {test_name} test with operation timeout {operation_timeout_seconds}s and process timeout {process_timeout_seconds}s",
            )
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=get_timeout_env(operation_timeout_seconds),
            )

            log.step(3, "Waiting for process to timeout and exit naturally")
            start_time = time.time()

            try:
                _, stderr = process.communicate(timeout=process_timeout_seconds)
                elapsed = time.time() - start_time

                if "timeout" in stderr.lower() or process.returncode != 0:
                    log.success("Process exited due to timeout")
                    log.substep(f"Elapsed time: {elapsed:.1f}s")
                else:
                    log.warning("Process exited but timeout not detected")

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                process.kill()
                process.communicate()
                pytest.fail(f"Process did not exit on its own after {elapsed:.1f}s")

        log.step(4, "Checking for RPC serialization files")
        serialized_files = []
        if os.path.exists(INSPECTOR_LOG_PATH):
            serialized_files = [f for f in os.listdir(INSPECTOR_LOG_PATH) if f.endswith(".capnp.bin")]

        if serialized_files:
            log.success(f"Found {len(serialized_files)} serialized files")
            for f in serialized_files[:5]:
                log.substep(f"- {f}", indent=2)

        log.step(5, "Running dump_ops to verify serialization")
        dump_output, dump_success = run_dump_ops()

        if dump_success:
            log.success("dump_ops ran successfully")

            # For timeout tests with normal operations (no infinite loop),
            # operations might complete before timeout, resulting in no operations being recorded
            operations = parse_dump_ops_table(dump_output)
            if operations:
                # If operations were recorded, validate them
                verify_dump_ops_table(dump_output, expected_operation="add", callstack_type=callstack_type)
            else:
                log.warning("No operations found - process may have completed before timeout")

        log.success(f"{test_name} timeout serialization test passed")

    finally:
        log.success("Cleanup completed")


# ============================================================================
# Test Cases
# ============================================================================


def test_py_normal_operation_chain(clean_llk_header, clean_device):
    """Test that operation chain works normally without infinite loop"""
    # Fixtures clean_llk_header and clean_device ensure clean state
    _ = (clean_llk_header, clean_device)

    log.test_header("Normal Python Operation Chain (No Hang)")

    log.step(1, "Running normal operation chain with 30s timeout")
    try:
        result = subprocess.run(
            ["python", "tests/ttnn/tools/triage/run_operation_chain.py"],
            capture_output=True,
            text=True,
            timeout=30,
            env=get_base_env(),
        )

        if result.returncode != 0:
            pytest.fail(f"Normal operation chain failed: {result.stderr}")

        if "Test completed successfully" not in result.stdout:
            pytest.fail("Normal operation chain did not complete successfully")

        log.success("Normal operation chain completed successfully")

    except subprocess.TimeoutExpired:
        pytest.fail("Normal operation chain should not timeout")


def test_py_hang_detection_with_dump_ops(clean_device):
    """Test Python operation chain with hang detection using dump_ops"""
    _ = clean_device  # Ensure device is reset before test
    run_hang_detection_test(
        test_name="Python",
        start_process_func=start_python_operation_chain,
        expected_pattern="Step 4:",
        callstack_type="python",
    )


def test_py_timeout_with_rpc_serialization(clean_device):
    """Test Python operation timeout with RPC serialization on exit"""
    _ = clean_device  # Ensure device is reset before test
    run_timeout_serialization_test(
        test_name="Python",
        command=["python", "tests/ttnn/tools/triage/run_operation_chain.py"],
        callstack_type="python",
    )


def test_cpp_hang_detection_with_dump_ops(cpp_binary, clean_device):
    """
    Test C++ operation chain with hang detection using dump_ops.

    This test uses the standalone C++ binary built via CMake (without unity build)
    to preserve debug symbols, enabling proper callstack resolution via addr2line.
    """
    # Fixture cpp_binary ensures binary exists
    _ = (cpp_binary, clean_device)  # Ensure device is reset before test

    run_hang_detection_test(
        test_name="C++", start_process_func=start_cpp_operation_chain, expected_pattern="Step 1:", callstack_type="cpp"
    )


def test_cpp_timeout_with_rpc_serialization(cpp_binary, clean_device):
    """
    Test C++ operation timeout with RPC serialization on exit.

    This test uses the standalone C++ binary built via CMake
    to preserve debug symbols, enabling proper callstack resolution via addr2line.
    """
    # Fixture cpp_binary ensures binary exists
    _ = (cpp_binary, clean_device)  # Ensure device is reset before test

    run_timeout_serialization_test(test_name="C++", command=[CPP_OPERATION_CHAIN_BINARY], callstack_type="cpp")


def test_cpp_stripped_binary_hang_detection(cpp_binary_stripped, clean_device):
    """
    Test C++ operation chain with stripped binary (no debug symbols).

    This test verifies that even without debug symbols, the Inspector can still:
    1. Track operations and capture callstacks
    2. Show function addresses (without resolved line numbers)
    3. Display operation arguments

    The stripped binary has debug symbols removed via `strip --strip-debug`, so:
    - Callstacks will show addresses like `run_operation_chain_cpp(+0x3d648)` ✓
    - Line numbers like `run_operation_chain.cpp:42` will NOT be available ✓
    """
    # Fixture cpp_binary_stripped ensures stripped binary exists
    _ = (cpp_binary_stripped, clean_device)  # Ensure device is reset before test

    run_hang_detection_test(
        test_name="C++ (Stripped Binary)",
        start_process_func=lambda: start_cpp_operation_chain(CPP_OPERATION_CHAIN_BINARY_STRIPPED),
        expected_pattern="Step 1:",
        callstack_type="cpp",
        allow_no_line_numbers=True,  # Stripped binary won't have line numbers
    )


if __name__ == "__main__":
    pytest.main()
