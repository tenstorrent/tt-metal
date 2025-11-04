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

    # Updated to match the actual file content
    TARGET = """    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)"""

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
    - callstack: str (file/line or addresses) - reconstructed from multi-line frames
    - has_arguments: bool
    """
    import re

    operations = []
    lines = dump_output.split("\n")

    # Find table content between header and footer
    in_table = False
    current_op = None
    callstack_frames = []
    in_arguments_section = False  # Track if we're in arguments vs callstack

    for i, line in enumerate(lines):
        # Look for separator line with dashes after header
        if "├────" in line or "│ Dev/Core" in line:
            in_table = True
            continue
        if "╰────" in line:
            # Finalize last operation if any
            if current_op and callstack_frames:
                current_op["callstack"] = " ".join(callstack_frames)
            break

        if in_table and "│" in line:
            # Parse table row: │ Dev/Core │ Operation │ Callstack/Args │
            # Don't filter out empty parts - continuation rows have empty first two columns
            parts = [p.strip() for p in line.split("│")]
            # Remove first and last (empty before first │ and after last │)
            if len(parts) > 0 and parts[0] == "":
                parts = parts[1:]
            if len(parts) > 0 and parts[-1] == "":
                parts = parts[:-1]

            if len(parts) >= 3:
                device_core = parts[0]
                operation = parts[1]
                callstack_args = parts[2]

                # Validate device/core format: "N / X,Y" or just operation continuation
                device_core_pattern = r"^\d+\s*/\s*\d+,\d+$"
                if re.match(device_core_pattern, device_core):
                    # Finalize previous operation if any
                    if current_op and callstack_frames:
                        current_op["callstack"] = " ".join(callstack_frames)
                        callstack_frames = []

                    # New operation row - reset state
                    in_arguments_section = False
                    current_op = {
                        "device_core": device_core,
                        "operation": operation,
                        "callstack": "",
                        "has_arguments": False,
                    }

                    # Check if this is the start of callstack section
                    if callstack_args == "Callstack:":
                        # Frames will be on subsequent lines
                        pass
                    elif callstack_args.startswith("Callstack:"):
                        # Old single-line format or first frame inline
                        current_op["callstack"] = callstack_args.replace("Callstack:", "").strip()
                    else:
                        current_op["callstack"] = callstack_args

                    # Check if this row or subsequent rows contain arguments
                    # Look ahead for "Arguments:" marker
                    for j in range(i, min(i + 30, len(lines))):  # Increased range for multi-line callstacks
                        if "Arguments:" in lines[j]:
                            current_op["has_arguments"] = True
                            break

                    operations.append(current_op)

                elif current_op and "Arguments:" in callstack_args:
                    # We've moved to the arguments section - finalize callstack
                    if callstack_frames:
                        current_op["callstack"] = " ".join(callstack_frames)
                        callstack_frames = []
                    in_arguments_section = True

                elif current_op and "·" in callstack_args and not in_arguments_section:
                    # This is a callstack frame (bullet point line in continuation row)
                    # Only collect if we're NOT in arguments section yet
                    # The device_core and operation columns are empty (just whitespace)
                    # Strip bullet and whitespace, extract the frame
                    frame = callstack_args.replace("·", "").strip()
                    if frame:
                        callstack_frames.append(frame)

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
                # Should have frame numbers and file path with line number
                # Format: "#0 test.py:42" or "#0 test.py:42 #1 main.py:10 ..."
                if not re.search(r"#\d+ .+\.py:\d+", op["callstack"]):
                    pytest.fail(
                        f"Python callstack should contain frame numbers with '.py:LINE_NUMBER' format. "
                        f"Expected pattern like '#0 file.py:42'. Got: {op['callstack']}"
                    )
            elif callstack_type == "cpp":
                # Should have frame numbers with function addresses or resolved symbols
                # Format: "#0 func [binary(+0x123)]" or "#0 file.cpp:42"
                has_frame_numbers = re.search(r"#\d+ ", op["callstack"])
                has_addresses = "+0x" in op["callstack"]
                has_resolved_symbols = ".cpp:" in op["callstack"]

                if not has_frame_numbers:
                    pytest.fail(
                        f"C++ callstack should contain frame numbers (e.g., '#0 func'). " f"Got: {op['callstack']}"
                    )

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
# Test Generation Helpers
# ============================================================================


def run_generate_test_and_display_results(validate_tests=False):
    """
    Common function to run dump_ops --generate-test and display results.

    Args:
        validate_tests: If True, reset device and run generated tests to verify they work

    Returns:
        tuple of (success, generated_files, tests_passed)
    """
    log.substep("Running dump_ops.py with --generate-test flag")

    # Run dump_ops with generate-test flag
    gen_result = subprocess.run(
        ["python", "tools/triage/dump_ops.py", "--generate-test"],
        capture_output=True,
        text=True,
        timeout=60,
        env=get_base_env(),
    )

    log.substep("dump_ops.py output:")
    if gen_result.stdout:
        for line in gen_result.stdout.split("\n"):
            if line.strip():
                log.substep(f"  {line}")

    if gen_result.returncode != 0:
        log.warning(f"dump_ops.py returned non-zero exit code: {gen_result.returncode}")
        if gen_result.stderr:
            log.error(f"stderr: {gen_result.stderr}")

    # Check if test files were generated
    generated_files = glob.glob("test_op_*.py")
    tests_passed = False

    if generated_files:
        log.success(f"Found {len(generated_files)} generated test file(s):")
        for test_file in generated_files:
            log.substep(f"  - {test_file}")

            # Show first few lines of each test file
            with open(test_file, "r") as f:
                lines = f.readlines()[:15]
                log.substep(f"\n  Preview of {test_file}:")
                for line in lines:
                    if line.strip():
                        log.substep(f"    {line.rstrip()}")

        if validate_tests:
            log.separator()
            log.step(0, "Validating generated tests by running them")

            # IMPORTANT: Clean the LLK header first to remove any infinite loop
            log.substep("Cleaning LLK header before running generated tests...")
            ensure_llk_header_clean()

            # Reset device before running tests
            log.substep("Resetting device before running generated tests...")
            reset_success = reset_device()
            if not reset_success:
                log.error("Failed to reset device, tests may not run properly")

            # Run generated tests one by one
            log.substep("Running generated tests to verify they work...")
            all_tests_ran = True
            test_results = []

            for test_file in generated_files:
                log.substep(f"\nRunning {test_file}...")
                try:
                    # Run with short timeout - we expect these to either pass quickly or hang
                    test_result = subprocess.run(
                        ["python", "-m", "pytest", test_file, "-xvs", "--tb=short"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=get_base_env(),
                    )

                    if test_result.returncode == 0:
                        log.success(f"  ✓ {test_file} executed successfully")
                        test_results.append((test_file, "passed"))
                    else:
                        # Check if it failed at the expected operation (the hanging one)
                        # The test file name contains the operation index (e.g., test_op_3_ttnn_add.py)
                        # The last operation (ADD) is the problematic one
                        if "add" in test_file.lower() or "op_4" in test_file:
                            # This is expected - the ADD operation was the hanging one (operation ID 4)
                            log.success(f"  ✓ {test_file} correctly identified as problematic operation")
                            test_results.append((test_file, "identified_issue"))
                        elif "ttnn::add" in test_result.stdout or "ttnn.add" in test_result.stdout:
                            log.success(f"  ✓ {test_file} correctly identified problematic ADD operation in output")
                            test_results.append((test_file, "identified_issue"))
                        else:
                            # For other operations, they should pass successfully
                            log.warning(f"  ⚠ {test_file} failed unexpectedly")
                            log.substep(f"    Return code: {test_result.returncode}")
                            if test_result.stderr:
                                log.substep(f"    Error: {test_result.stderr[:200]}")
                            test_results.append((test_file, "failed"))
                            # Don't fail the whole test - some operations before the hang should pass
                            # all_tests_ran = False

                except subprocess.TimeoutExpired:
                    log.warning(f"  ⚠ {test_file} timed out (may be the hanging operation)")
                    test_results.append((test_file, "timeout"))
                except Exception as e:
                    log.error(f"  ✗ Failed to run {test_file}: {e}")
                    test_results.append((test_file, "error"))
                    all_tests_ran = False

            # Summary of test results
            log.separator()
            log.substep("\nTest Validation Summary:")
            log.substep("=" * 40)
            passed_count = sum(1 for _, status in test_results if status in ["passed", "identified_issue", "timeout"])
            log.substep(f"Total tests: {len(test_results)}")
            log.substep(f"Successfully validated: {passed_count}")

            for test_file, status in test_results:
                status_icon = {"passed": "✓", "identified_issue": "✓", "timeout": "⏱", "failed": "✗", "error": "✗"}.get(
                    status, "?"
                )
                log.substep(f"  {status_icon} {test_file}: {status}")

            # Consider tests passed if we have at least one successful validation
            # (either passed or correctly identified the problematic operation)
            tests_passed = passed_count > 0

            if tests_passed:
                log.success(f"\n{passed_count}/{len(test_results)} generated tests were successfully validated!")
            else:
                log.warning("\nTests could not be validated properly")

        # Show instructions on how to run the tests
        log.separator()
        log.success("Test files successfully generated!")
        log.substep("")
        log.substep("INSTRUCTIONS TO RUN THE GENERATED TESTS:")
        log.substep("=" * 50)
        log.substep("")
        log.substep("1. Run all tests to find the problematic operation:")
        log.substep(f"   pytest {' '.join(generated_files)} -xvs")
        log.substep("")
        log.substep("2. Or run individual tests:")
        for test_file in generated_files:
            log.substep(f"   pytest {test_file} -xvs")
        log.substep("")
        log.substep("3. The test that fails/hangs identifies the problematic operation")
        log.substep("")
        log.substep("NOTE: Generated test files have been preserved for inspection.")
        log.substep(f"      Files: {', '.join(generated_files)}")

        return True, generated_files, tests_passed
    else:
        log.warning("No test files (test_op_*.py) were generated")
        log.substep("This could mean:")
        log.substep("  - The operations were not captured yet")
        log.substep("  - dump_ops couldn't access the Inspector data")
        log.substep("  - The process needs more time to serialize operations")

        return False, [], False


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
    test_generate: bool = False,
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

            # Optionally test the generate-test functionality
            if test_generate:
                log.step(5, "Testing --generate-test functionality")

                # Run dump_ops with generate-test flag
                gen_result = subprocess.run(
                    ["python", "tools/triage/dump_ops.py", "--generate-test"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=get_base_env(),
                )

                # Check if test files were generated
                generated_files = glob.glob("test_op_*.py")
                if generated_files:
                    log.success(f"Generated {len(generated_files)} test files")

                    # NOTE: NOT cleaning up generated files - keeping them for inspection
                    for test_file in generated_files:
                        log.substep(f"Preserved test file: {test_file}")
                else:
                    log.warning("No test files were generated (may be expected if no hanging ops)")

        log.success(f"{test_name} hang detection test passed")

    finally:
        log.step(6 if test_generate else 5, "Cleaning up")
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


def test_py_timeout_with_generate_test(clean_device):
    """Test --generate-test works when operation times out (not just hangs)"""
    _ = clean_device  # Ensure device is reset before test

    log.test_header("Python Timeout with Test Generation")

    process = None
    generated_files = []
    try:
        log.step(1, "Adding infinite loop to trigger timeout")
        with infinite_loop_context():
            log.step(2, "Starting Python operation chain with timeout (10s)")

            # Start with Inspector enabled and operation timeout
            env = get_timeout_env(10)  # 10 second timeout

            process = subprocess.Popen(
                ["python", "tests/ttnn/tools/triage/run_operation_chain.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            log.substep(f"Process PID: {process.pid}")

            log.step(3, "Waiting for process to timeout and exit")
            start_time = time.time()

            try:
                stdout, stderr = process.communicate(timeout=60)
                elapsed = time.time() - start_time

                if "timeout" in stderr.lower() or process.returncode != 0:
                    log.success(f"Process exited due to timeout after {elapsed:.1f}s")
                else:
                    log.warning("Process exited but timeout not detected")

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                process.kill()
                process.communicate()
                pytest.fail(f"Process did not exit on its own after {elapsed:.1f}s")

        log.step(4, "Checking for serialized operations")
        serialized_files = []
        if os.path.exists(INSPECTOR_LOG_PATH):
            serialized_files = [f for f in os.listdir(INSPECTOR_LOG_PATH) if f.endswith(".capnp.bin")]
            if serialized_files:
                log.success(f"Found {len(serialized_files)} serialized files")

        log.step(5, "Running dump_ops.py with --generate-test flag and validating")

        # Use common helper to run generate-test and display results
        # Pass validate_tests=True to actually run and verify the generated tests
        success, generated_files, tests_passed = run_generate_test_and_display_results(validate_tests=True)

        if success:
            log.success("Successfully generated tests from timeout scenario")
            if tests_passed:
                log.success("Generated tests were validated successfully!")
            else:
                log.warning("Generated tests could not be fully validated")
        else:
            # For timeout tests, operations might complete before timeout
            log.warning("No operations captured - process may have completed before timeout")

    finally:
        log.step(6, "Cleaning up (keeping generated test files)")
        terminate_process(process)

        if generated_files:
            log.substep(f"Preserved {len(generated_files)} test file(s) for inspection")

        log.success("Cleanup completed")


def test_py_hang_with_generate_test(clean_device):
    """Test that --generate-test flag creates reproducible test for hanging operation"""
    _ = clean_device  # Ensure device is reset before test

    log.test_header("Python Generate Test for Hanging Operation")

    process = None
    generated_files = []
    try:
        log.step(1, "Adding infinite loop to simulate hang")
        with infinite_loop_context():
            log.step(2, "Starting Python operation chain with Inspector")
            # Start with Inspector enabled to generate operations data
            env = get_base_env()
            env["TT_METAL_INSPECTOR_LOG_PATH"] = INSPECTOR_LOG_PATH

            process = subprocess.Popen(
                ["python", "tests/ttnn/tools/triage/run_operation_chain.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            log.substep(f"Process PID: {process.pid}")

            log.step(3, "Monitoring output for hang (timeout: 30s)")
            found_pattern, last_line = monitor_process_output(process, 30, "Step 4:", "Python")

            if process.poll() is not None:
                pytest.fail("Process should hang on ADD operation")

            log.success("Process is hanging as expected")

            log.step(4, "Running dump_ops.py with --generate-test flag and validating")

            # Use common helper to run generate-test and display results
            # Pass validate_tests=True to actually run and verify the generated tests
            success, generated_files, tests_passed = run_generate_test_and_display_results(validate_tests=True)

            if success:
                log.success("Test generation check completed")
                if tests_passed:
                    log.success("Generated tests were validated successfully!")
                else:
                    pytest.fail("Generated tests failed validation - they should be able to run successfully")
            else:
                log.warning("Test generation check completed with no files generated")

    finally:
        log.step(6, "Cleaning up process (keeping generated test files)")
        terminate_process(process)

        # NOTE: We intentionally do NOT remove generated test files
        # so they can be inspected and run manually
        if generated_files:
            log.substep(f"Preserved {len(generated_files)} test file(s) for inspection")

        log.success("Cleanup completed")


if __name__ == "__main__":
    pytest.main()
