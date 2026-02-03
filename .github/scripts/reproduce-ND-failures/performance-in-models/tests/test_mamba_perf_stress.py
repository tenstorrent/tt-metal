# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Stress test to reproduce: Mamba model device performance marginally below threshold

Original failure: device-perf / N300 WH B0 Set 2 device perf - 2026-02-01
Error: AVG DEVICE KERNEL SAMPLES/S 19646.28 is outside of expected range (19649.12, 20451.13)

The original test expected ~20050 samples/s with a 2% margin, but measured 19646 samples/s
which was just 0.015% below the lower threshold of 19649 samples/s.

This test amplifies the non-deterministic performance variance by running multiple iterations.

Run with:
    cd /tt-metal
    source /opt/venv/bin/activate
    export TT_METAL_HOME=/tt-metal
    export PYTHONPATH=/tt-metal
    export LD_LIBRARY_PATH=/tt-metal/build/lib
    export ARCH_NAME=wormhole_b0
    export LOGURU_LEVEL=INFO

    # Run via pytest (20 iterations)
    pytest .github/scripts/reproduce-ND-failures/performance-in-models/tests/test_mamba_perf_stress.py -v -x --timeout=600 2>&1 | tee .github/scripts/reproduce-ND-failures/performance-in-models/logs/run.log

    # Or run directly via Python
    python .github/scripts/reproduce-ND-failures/performance-in-models/tests/test_mamba_perf_stress.py --iterations 20 2>&1 | tee .github/scripts/reproduce-ND-failures/performance-in-models/logs/run.log
"""

import argparse
import sys
import time

import pytest
from loguru import logger
from tracy.process_model_log import get_samples_per_s

# Import the same utilities used by the original test
from models.perf.device_perf_utils import run_device_perf

# Test parameters matching the original failing test
BATCH_SIZE = 32
EXPECTED_LAYER_DURATION_MS = 1.596  # ms per layer
MARGIN = 0.020  # 2% margin (same as original)
NUM_LAYERS = 1  # For device perf test
SUBDIR = "ttnn_mamba"

# Calculate expected performance
EXPECTED_LAYER_DURATION_NS = EXPECTED_LAYER_DURATION_MS * 1e6  # ms to ns
EXPECTED_TOTAL_SAMPLES_PER_S = get_samples_per_s(EXPECTED_LAYER_DURATION_NS, BATCH_SIZE)
LOWER_THRESHOLD = (1 - MARGIN) * EXPECTED_TOTAL_SAMPLES_PER_S
UPPER_THRESHOLD = (1 + MARGIN) * EXPECTED_TOTAL_SAMPLES_PER_S


def run_single_iteration(iteration_num: int) -> tuple[float, bool]:
    """
    Run a single performance measurement iteration.

    NOTE: This spawns an external pytest process that uses the device,
    so the caller must NOT hold the device.

    Returns:
        Tuple of (measured_samples_per_s, passed)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Iteration {iteration_num}: Starting performance measurement")
    logger.info(f"Expected: {EXPECTED_TOTAL_SAMPLES_PER_S:.2f} samples/s")
    logger.info(f"Lower threshold: {LOWER_THRESHOLD:.2f} samples/s")
    logger.info(f"Upper threshold: {UPPER_THRESHOLD:.2f} samples/s")
    logger.info(f"{'='*60}")

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    command = "pytest models/demos/wormhole/mamba/tests/test_mamba_model.py::test_device_perf[1]"

    try:
        post_processed_results = run_device_perf(command, SUBDIR, 1, cols, BATCH_SIZE)

        inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
        measured = post_processed_results.get(inference_time_key, 0)

        passed = LOWER_THRESHOLD <= measured <= UPPER_THRESHOLD

        if passed:
            logger.info(f"Iteration {iteration_num}: PASSED - {measured:.2f} samples/s (within range)")
        else:
            if measured < LOWER_THRESHOLD:
                logger.error(
                    f"Iteration {iteration_num}: FAILED (TOO SLOW) - "
                    f"{measured:.2f} samples/s < {LOWER_THRESHOLD:.2f} lower threshold "
                    f"(difference: {LOWER_THRESHOLD - measured:.4f})"
                )
            else:
                logger.error(
                    f"Iteration {iteration_num}: FAILED (TOO FAST) - "
                    f"{measured:.2f} samples/s > {UPPER_THRESHOLD:.2f} upper threshold"
                )

        return measured, passed

    except Exception as e:
        logger.error(f"Iteration {iteration_num}: ERROR - {e}")
        return 0.0, False


# IMPORTANT: No device fixture - run_device_perf spawns its own pytest subprocess
# that opens the device internally. We must NOT hold the device here.
@pytest.mark.parametrize("iteration", range(20))
def test_mamba_perf_stress(iteration):
    """
    Stress test that runs the Mamba device performance check multiple times.

    This test reproduces the non-deterministic failure where performance
    sometimes dips just below the 2% margin threshold.

    NOTE: Does NOT use device fixture because run_device_perf spawns an external
    pytest subprocess that needs exclusive access to the device.
    """
    logger.info(f"Starting stress test iteration {iteration}")

    measured, passed = run_single_iteration(iteration)

    # Assert to trigger test failure on threshold violation
    if not passed:
        if measured < LOWER_THRESHOLD:
            pytest.fail(
                f"Performance regression detected! "
                f"Measured {measured:.2f} samples/s < {LOWER_THRESHOLD:.2f} threshold "
                f"(expected ~{EXPECTED_TOTAL_SAMPLES_PER_S:.2f} samples/s)"
            )
        else:
            pytest.fail(
                f"Performance faster than expected! "
                f"Measured {measured:.2f} samples/s > {UPPER_THRESHOLD:.2f} threshold"
            )


def stress_test_loop(iterations: int = 20, stop_on_fail: bool = True):
    """
    Run the stress test in a loop without pytest.

    Args:
        iterations: Number of iterations to run
        stop_on_fail: If True, stop on first failure
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"Starting Mamba Performance Stress Test")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Expected performance: {EXPECTED_TOTAL_SAMPLES_PER_S:.2f} samples/s")
    logger.info(f"Margin: {MARGIN*100:.1f}%")
    logger.info(f"Valid range: [{LOWER_THRESHOLD:.2f}, {UPPER_THRESHOLD:.2f}]")
    logger.info(f"{'#'*60}\n")

    results = []
    failures = []

    for i in range(iterations):
        measured, passed = run_single_iteration(i + 1)
        results.append((measured, passed))

        if not passed:
            failures.append((i + 1, measured))
            if stop_on_fail:
                logger.error(f"\nStopping on first failure at iteration {i + 1}")
                break

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"STRESS TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total iterations: {len(results)}")
    logger.info(f"Passed: {sum(1 for _, p in results if p)}")
    logger.info(f"Failed: {len(failures)}")

    if results:
        measurements = [m for m, _ in results if m > 0]
        if measurements:
            logger.info(f"Min performance: {min(measurements):.2f} samples/s")
            logger.info(f"Max performance: {max(measurements):.2f} samples/s")
            logger.info(f"Avg performance: {sum(measurements)/len(measurements):.2f} samples/s")

    if failures:
        logger.error(f"\nFailures occurred at iterations: {[f[0] for f in failures]}")
        for iter_num, measured in failures:
            logger.error(f"  Iteration {iter_num}: {measured:.2f} samples/s")
        return 1
    else:
        logger.info(f"\nAll iterations passed!")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mamba performance stress test")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations to run (default: 20)")
    parser.add_argument(
        "--no-stop-on-fail", action="store_true", help="Continue running after failure (default: stop on first failure)"
    )
    args = parser.parse_args()

    sys.exit(stress_test_loop(iterations=args.iterations, stop_on_fail=not args.no_stop_on_fail))
