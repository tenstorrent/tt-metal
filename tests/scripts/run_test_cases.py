#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
import pty
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import signal

from loguru import logger


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    exit_code: int = 0


@dataclass
class TestCase:
    name: str
    command: str
    env: Dict[str, str] = None
    timeout: int = 3600

    def __post_init__(self):
        if self.env is None:
            self.env = {}


class TestRunner:
    def __init__(self, fail_fast: bool = False, working_dir: str = None):
        self.fail_fast = fail_fast
        self.working_dir = working_dir or os.getcwd()
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def run_command(self, test_case: TestCase) -> TestResult:
        """Run a single test command with color preservation."""
        logger.info("=" * 60)
        logger.info(f"Running: \033[36m{test_case.name}\033[0m")
        logger.info(f"Command: \033[33m{test_case.command}\033[0m")
        logger.info("=" * 60)
        print()  # Add blank line before subprocess output

        # Prepare environment with color forcing
        env = os.environ.copy()
        env.update(test_case.env)
        env.update(
            {
                "FORCE_COLOR": "1",
                "PY_COLORS": "1",
                "ANSI_COLORS_DISABLED": "0",
                "TERM": "xterm-256color",
            }
        )
        env.pop("NO_COLOR", None)

        start_time = time.time()

        try:
            # Use pty to preserve colors and show real-time output
            master_fd, slave_fd = pty.openpty()

            process = subprocess.Popen(
                test_case.command,
                shell=True,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
                cwd=self.working_dir,
                preexec_fn=os.setsid,
            )

            os.close(slave_fd)  # Close slave in parent

            # Read and display output in real-time
            try:
                while True:
                    try:
                        data = os.read(master_fd, 1024)
                        if not data:
                            break
                        print(data.decode("utf-8", errors="replace"), end="", flush=True)
                    except OSError:
                        break

                # Wait for process with timeout
                try:
                    exit_code = process.wait(timeout=test_case.timeout)
                except subprocess.TimeoutExpired:
                    logger.error(f"Test timed out after {test_case.timeout} seconds")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    exit_code = -1

            finally:
                os.close(master_fd)

        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            exit_code = -2

        duration = time.time() - start_time
        passed = exit_code == 0

        # Log result
        print()
        if passed:
            logger.success(f"PASSED: {test_case.name} ({duration:.2f}s)")
        else:
            logger.error(f"FAILED: {test_case.name} ({duration:.2f}s) - Exit code: {exit_code}")

        return TestResult(name=test_case.name, passed=passed, duration=duration, exit_code=exit_code)

    def run_tests(self, test_cases: List[TestCase]) -> bool:
        """Run all test cases and return True if all passed."""
        logger.info(f"Starting test run with {len(test_cases)} test cases")
        logger.info(f"Fail-fast: {'enabled' if self.fail_fast else 'disabled'}")
        logger.info(f"Working directory: {self.working_dir}")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"[{i}/{len(test_cases)}] Starting: {test_case.name}")

            result = self.run_command(test_case)
            self.results.append(result)

            if not result.passed and self.fail_fast:
                logger.error(f"Fail-fast enabled. Stopping after failure: {test_case.name}")
                break

        return self.print_summary()

    def print_summary(self) -> bool:
        """Print test summary and return True if all tests passed."""
        total_duration = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests

        print()
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total: {total_tests} | " f"Passed: {passed_tests} | " f"Failed: {failed_tests}")
        logger.info(f"Duration: {total_duration:.2f}s")

        if failed_tests > 0:
            logger.error("Failed tests:")
            for result in self.results:
                if not result.passed:
                    logger.error(f"  • {result.name} ({result.duration:.2f}s)")

        all_passed = failed_tests == 0
        if all_passed:
            logger.success("ALL TESTS PASSED!")
        else:
            logger.error(f"{failed_tests} TEST(S) FAILED")

        return all_passed


def load_test_cases(config_file: str) -> List[TestCase]:
    """Load test cases from JSON configuration file."""
    with open(config_file, "r") as f:
        data = json.load(f)

    return [TestCase(**test_data) for test_data in data.get("tests", [])]


def setup_logging():
    """Setup loguru logging with distinctive formatting."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="[RUNNER] {time:HH:mm:ss} | {message}", colorize=True)


def main():
    parser = argparse.ArgumentParser(description="Run test cases", formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--test-cases", required=True, help="JSON file with test definitions")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--working-dir", help="Working directory (default: current)")

    args = parser.parse_args()
    setup_logging()

    if not os.path.exists(args.test_cases):
        logger.error(f"Test cases file not found: {args.test_cases}")
        sys.exit(1)

    try:
        test_cases = load_test_cases(args.test_cases)
        logger.info(f"Loaded {len(test_cases)} test cases from {args.test_cases}")

        runner = TestRunner(fail_fast=args.fail_fast, working_dir=args.working_dir)
        all_passed = runner.run_tests(test_cases)

        sys.exit(0 if all_passed else 1)

    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
