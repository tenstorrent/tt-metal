#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


"""
Convenience script to run cluster validation for 4x4 BH quietbox tests.
This will eventually be replaced when tt-run supports top-level config.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from loguru import logger

# MPI Configuration
MPI_HOSTS = "10.140.20.237,10.140.20.239,10.140.20.238,10.140.20.240"
RANKFILE = "tests/scale_out/4x_bh_quietbox/rankfile/4x4.txt"
RANK_BINDING = "tests/scale_out/4x_bh_quietbox/rank_bindings/4x4.yaml"

MPI_COMMON_ARGS = (
    f"--allow-run-as-root --tag-output --host {MPI_HOSTS} "
    f"--map-by rankfile:file={RANKFILE} --mca btl self,tcp "
    f"--mca btl_tcp_if_include enp10s0f1np1"
)

# Retry Configuration
MAX_RETRIES = 10  # Maximum number of attempts before giving up

# Validation command
VALIDATION_CMD = (
    "./build/tools/scaleout/run_cluster_validation --hard-fail "
    "--factory-descriptor-path tests/scale_out/4x_bh_quietbox/factory_system_descriptors/"
    "factory_system_descriptor_4x_bh_quietbox.textproto"
)


def header(message):
    """Wrap a message with separator lines."""
    sep = "=" * 42
    return f"{sep}\n{message}\n{sep}"


def run_synchronized_reset():
    """
    Run synchronized reset across all hosts by calling reset.sh.

    This ensures all machines reset simultaneously to avoid link skew issues.

    Returns:
        bool: True if reset was successful, False otherwise.
    """
    logger.info("")
    logger.info(header("Running synchronized reset across cluster..."))

    reset_script = Path(__file__).parent / "distributed_reset.sh"

    try:
        if not reset_script.exists():
            logger.error(f"Reset script not found: {reset_script}")
            return False

        result = subprocess.run([str(reset_script)], capture_output=True, text=True)

        if result.returncode == 0:
            logger.success("✓ Synchronized reset completed successfully")

            # Wait for devices to stabilize after reset
            logger.info("Waiting 3 seconds for devices to stabilize...")
            time.sleep(3)
            logger.info("")
            return True
        else:
            logger.error("✗ Reset failed")
            if result.stdout:
                logger.error(f"stdout: {result.stdout}")
            if result.stderr:
                logger.error(f"stderr: {result.stderr}")
            logger.info("")
            return False

    except Exception as e:
        logger.error(f"Error during reset: {e}")
        return False


def run_cluster_validation(attempt_number=1):
    """
    Run cluster validation with MPI.

    Args:
        attempt_number: Current attempt number (for logging).

    Returns:
        int: Exit code from the validation command.
    """
    if attempt_number > 1:
        logger.info(header(f"Running cluster validation (attempt {attempt_number}/{MAX_RETRIES})..."))
    else:
        logger.info(header("Running cluster validation..."))

    # Construct the full tt-run command
    cmd = ["tt-run", "--mpi-args", MPI_COMMON_ARGS, "--rank-binding", RANK_BINDING, "bash", "-c", VALIDATION_CMD]

    # Run command and capture output
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as temp_log:
        temp_log_path = temp_log.name

        try:
            with open(temp_log_path, "w") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

            logger.info("")

            if result.returncode == 0:
                logger.success(header("✓ Cluster validation PASSED"))
                # Show output on success
                with open(temp_log_path, "r") as f:
                    logger.info(f.read())
                os.unlink(temp_log_path)
            else:
                logger.error(header(f"✗ Cluster validation FAILED (exit code: {result.returncode})"))
                # On failure, provide helpful information
                logger.info("")
                logger.info("Logs available in: cluster_validation_logs/")
                logger.info(f"Full output saved to: {temp_log_path}")
                logger.info("")

            return result.returncode

        except FileNotFoundError:
            logger.error("'tt-run' command not found. Make sure it's in your PATH.")
            os.unlink(temp_log_path)
            return 127
        except Exception as e:
            logger.error(f"Error running cluster validation: {e}")
            if os.path.exists(temp_log_path):
                os.unlink(temp_log_path)
            return 1


def print_instructions():
    """Print usage instructions."""
    logger.info(
        """
INSTRUCTIONS:

Distributed Reset:
  Since distributed/MPI reset introduces too much reset skew, we need to manually
  reset the system to ensure all links come up correctly. We can do this by running
  'tt-smi -r' in synchronized tmux panes on each of the four machines.

Troubleshooting:
  If cluster_validation is failing, you need to repeat reset on all machines to
  ensure all links come up correctly.

  If successful, it will report:
  "[1,0]<stdout>: All connections match between FSD and GSD (64 connections)"
"""
    )


def run_validation_with_retry():
    """
    Run cluster validation in a loop with automatic reset on failure.

    Returns:
        int: Final exit code (0 on success, non-zero on failure).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        # Run validation
        exit_code = run_cluster_validation(attempt_number=attempt)

        # If successful, we're done!
        if exit_code == 0:
            logger.info("")
            logger.success(f"✓ Validation succeeded on attempt {attempt}/{MAX_RETRIES}")
            return 0

        # If this was the last attempt, give up
        if attempt >= MAX_RETRIES:
            logger.info("")
            logger.error(f"✗ Validation failed after {MAX_RETRIES} attempts")
            logger.error("Please check the logs and manually investigate the issue.")
            return exit_code

        # Otherwise, run reset and try again
        logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed. Running synchronized reset...")
        reset_success = run_synchronized_reset()

        if not reset_success:
            logger.error("Reset failed. Aborting retry loop.")
            logger.error("You may need to manually run 'tt-smi -r' on all machines.")
            return exit_code

        logger.info("Reset completed. Retrying validation...")
        logger.info("")

    return 1  # Should never reach here


def main():
    """Main entry point."""
    # Configure loguru with custom colors for different log levels
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", colorize=True, level="INFO")

    # Set environment variables for MPI
    os.environ["MPI_HOSTS"] = MPI_HOSTS
    os.environ["RANKFILE"] = RANKFILE
    os.environ["RANK_BINDING"] = RANK_BINDING
    os.environ["MPI_COMMON_ARGS"] = MPI_COMMON_ARGS

    # Run validation with retry loop
    exit_code = run_validation_with_retry()

    # Exit with the same code as the validation command
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
