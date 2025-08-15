#!/usr/bin/env python3
"""
Fabric Elastic Channels Performance Sweep

This script runs the fabric_elastic_channels_host_test with different parameter combinations
and collects comprehensive performance metrics for analysis.

Features:
- Two execution modes: pipe mode (single process, faster) and traditional mode (multiple processes)
- Comprehensive CSV output with all test configuration parameters and calculated metrics
- Consolidated results across multiple test runs for longitudinal analysis
- Column documentation for easy data analysis

Output Files:
- results_TIMESTAMP.csv: Individual test run results with all columns
- consolidated_results.csv: All test runs combined for cross-run analysis
- column_documentation.txt: Explanation of all CSV columns
- results_TIMESTAMP.json: JSON format results
- summary.json: Statistical summary of the test run

Example Analysis:
The comprehensive CSV format enables easy analysis with tools like pandas:
    import pandas as pd
    df = pd.read_csv('consolidated_results.csv')

    # Find best configurations by throughput
    best_configs = df.nlargest(10, 'throughput_gbps')

    # Analyze performance by worker count
    by_workers = df.groupby('n_workers')['throughput_gbps'].mean()

    # Compare execution modes
    mode_comparison = df.groupby('execution_mode')['throughput_gbps'].describe()
"""

import subprocess
import re
import csv
import json
import itertools
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import threading
import queue


@dataclass
class TestConfig:
    """Test configuration parameters"""

    n_chunks: int
    chunk_n_pkts: int
    rx_chunk_n_pkts: int
    packet_size: int
    bidirectional: bool
    message_size: int
    total_messages: int
    n_workers: int
    fabric_mcast_factor: int


@dataclass
class TestResult:
    """Test result containing performance metrics"""

    config: TestConfig
    throughput_gbps: float
    throughput_msgs_per_sec: float
    total_messages_sent: int
    total_cycles: int
    duration_us: int
    success: bool
    error_msg: Optional[str] = None
    raw_output: Optional[str] = None


class FabricChannelsSweep:
    """Main class for running parameter sweeps and collecting comprehensive results

    This class handles:
    - Running test sweeps in either pipe mode (single process) or traditional mode (multiple processes)
    - Collecting detailed performance metrics and test configurations
    - Generating comprehensive CSV files with all differentiation columns and calculated metrics
    - Creating consolidated results across multiple test runs
    - Providing analysis documentation for all collected data
    """

    def __init__(self, executable_path: str, output_dir: str = "sweep_results", use_pipe_mode: bool = True):
        self.executable_path = executable_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[TestResult] = []
        self.use_pipe_mode = use_pipe_mode
        self.last_consolidated_count = 0  # Track how many results we've already saved to consolidated file

    def parse_test_output(self, output: str, config: TestConfig) -> TestResult:
        """Parse test output to extract performance metrics"""
        result = TestResult(
            config=config,
            throughput_gbps=0.0,
            throughput_msgs_per_sec=0.0,
            total_messages_sent=0,
            total_cycles=0,
            duration_us=0,
            success=False,
            raw_output=output,
        )

        try:
            # Extract throughput in GB/s
            gbps_match = re.search(r"Throughput: ([\d.]+) GB/s", output)
            if gbps_match:
                result.throughput_gbps = float(gbps_match.group(1))

            # Extract throughput in messages/second
            msgs_per_sec_match = re.search(r"Throughput: ([\d.]+) messages/second", output)
            if msgs_per_sec_match:
                result.throughput_msgs_per_sec = float(msgs_per_sec_match.group(1))

            # Extract total messages sent
            total_msgs_match = re.search(r"Total messages sent: (\d+)", output)
            if total_msgs_match:
                result.total_messages_sent = int(total_msgs_match.group(1))

            # Extract total cycles
            total_cycles_match = re.search(r"Total_cycles: (\d+)", output)
            if total_cycles_match:
                result.total_cycles = int(total_cycles_match.group(1))

            # Extract duration
            duration_match = re.search(r"Test completed in (\d+) microseconds", output)
            if duration_match:
                result.duration_us = int(duration_match.group(1))

            # Check if test completed successfully
            if "Test completed successfully" in output:
                result.success = True

        except Exception as e:
            result.error_msg = f"Failed to parse output: {str(e)}"

        return result

    def config_to_json(self, config: TestConfig) -> str:
        """Convert TestConfig to JSON string expected by C++ binary"""
        config_dict = {
            "n_chunks": config.n_chunks,
            "chunk_n_pkts": config.chunk_n_pkts,
            "rx_chunk_n_pkts": config.rx_chunk_n_pkts,
            "packet_size": config.packet_size,
            "bidirectional_mode": config.bidirectional,
            "message_size": config.message_size,
            "total_messages": config.total_messages,
            "n_workers": config.n_workers,
            "fabric_mcast_factor": config.fabric_mcast_factor,
        }
        return json.dumps(config_dict)

    def run_single_test(self, config: TestConfig, timeout: int = 300) -> TestResult:
        """Run a single test with given configuration"""
        cmd = [
            self.executable_path,
            str(config.n_chunks),
            str(config.chunk_n_pkts),
            str(config.rx_chunk_n_pkts),
            str(config.packet_size),
            str(1 if config.bidirectional else 0),
            str(config.message_size),
            str(config.total_messages),
            str(config.n_workers),
            str(config.fabric_mcast_factor),
        ]

        print(f"Running: {' '.join(cmd)}")

        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=os.getcwd())

            output = process.stdout + process.stderr
            result = self.parse_test_output(output, config)

            if process.returncode != 0 and result.success:
                result.success = False
                result.error_msg = f"Process returned {process.returncode}"

            if not result.success and result.error_msg is None:
                result.error_msg = f"Process failed with return code {process.returncode}"

            return result

        except subprocess.TimeoutExpired as e:
            timeout_output = ""
            try:
                if e.stdout:
                    timeout_output += e.stdout.decode()
                if e.stderr:
                    timeout_output += e.stderr.decode()
                if not timeout_output:
                    timeout_output = "No output captured (timeout)"
            except:
                timeout_output = "Could not decode timeout output"

            return TestResult(
                config=config,
                throughput_gbps=0.0,
                throughput_msgs_per_sec=0.0,
                total_messages_sent=0,
                total_cycles=0,
                duration_us=0,
                success=False,
                error_msg=f"Test timed out after {timeout} seconds",
                raw_output=timeout_output,
            )
        except Exception as e:
            return TestResult(
                config=config,
                throughput_gbps=0.0,
                throughput_msgs_per_sec=0.0,
                total_messages_sent=0,
                total_cycles=0,
                duration_us=0,
                success=False,
                error_msg=str(e),
                raw_output="No output captured (exception during execution)",
            )

    def run_pipe_mode_sweep(self, configs: List[TestConfig], max_failures: int = 10) -> None:
        """Run sweep using pipe mode - launch binary once and send configs via stdin"""
        total_configs = len(configs)
        print(f"Starting pipe mode sweep with {total_configs} test configurations")

        # Launch binary in pipe mode
        cmd = [self.executable_path, "--pipe-mode"]
        print(f"Launching: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                cwd=os.getcwd(),
            )

            # Start background thread to read output
            output_queue = queue.Queue()
            output_buffer = []

            def read_output():
                """Read output from the process and buffer it"""
                for line in iter(process.stdout.readline, ""):
                    if line:
                        output_queue.put(line.rstrip())
                        output_buffer.append(line.rstrip())

            output_thread = threading.Thread(target=read_output)
            output_thread.daemon = True
            output_thread.start()

            failures = 0
            for i, config in enumerate(configs):
                print(f"\n[{i+1}/{total_configs}] Sending test configuration:")
                print(f"  {config}")

                try:
                    # Send JSON config to process stdin
                    json_config = self.config_to_json(config)
                    process.stdin.write(json_config + "\n")
                    process.stdin.flush()

                    # Wait for this test to complete by looking for completion messages
                    test_output = []
                    test_completed = False
                    test_success = False

                    # Give it some time to process
                    timeout_counter = 0
                    max_timeout = 300  # 5 minutes

                    while not test_completed and timeout_counter < max_timeout:
                        try:
                            # Check for new output with short timeout
                            line = output_queue.get(timeout=1.0)
                            test_output.append(line)

                            # Check for completion markers
                            if "Test completed successfully" in line:
                                test_success = True
                                test_completed = True
                            elif "Test failed with exception:" in line or "failed" in line.lower():
                                test_success = False
                                test_completed = True

                        except queue.Empty:
                            timeout_counter += 1
                            continue

                    # Parse the collected output for this test
                    combined_output = "\n".join(test_output)
                    result = self.parse_test_output(combined_output, config)

                    # Override success based on our parsing if we detected completion
                    if test_completed:
                        result.success = test_success
                    elif timeout_counter >= max_timeout:
                        result.success = False
                        result.error_msg = "Test timed out"

                    self.results.append(result)

                    if result.success:
                        print(f"  ✓ Success: {result.throughput_gbps:.2f} GB/s")
                    else:
                        failures += 1
                        print(f"  ✗ Failed: {result.error_msg}")

                        if failures >= max_failures:
                            print(f"\nStopping sweep after {failures} consecutive failures")
                            break

                    # Save intermediate results
                    if (i + 1) % 10 == 0:
                        self.save_results()

                except Exception as e:
                    failures += 1
                    error_result = TestResult(
                        config=config,
                        throughput_gbps=0.0,
                        throughput_msgs_per_sec=0.0,
                        total_messages_sent=0,
                        total_cycles=0,
                        duration_us=0,
                        success=False,
                        error_msg=f"Exception during test: {str(e)}",
                        raw_output="",
                    )
                    self.results.append(error_result)
                    print(f"  ✗ Exception: {str(e)}")

                    if failures >= max_failures:
                        print(f"\nStopping sweep after {failures} consecutive failures")
                        break

            # Send empty line to signal completion
            try:
                process.stdin.write("\n")
                process.stdin.flush()
                process.stdin.close()
            except:
                pass

            # Wait for process to complete
            process.wait(timeout=30)

        except Exception as e:
            print(f"Failed to run pipe mode sweep: {e}")

        print(
            f"\nCompleted pipe mode sweep: {len([r for r in self.results if r.success])}/{len(self.results)} tests successful"
        )

    def is_invalid_config(self, config: TestConfig) -> bool:
        if config.message_size > config.packet_size:
            return True
        if config.n_workers > config.n_chunks:
            return True
        if config.fabric_mcast_factor > 10:
            return True

        if config.chunk_n_pkts * config.n_chunks > 16:
            return True
        return False

    def generate_sweep_configs(self, sweep_params: Dict) -> List[TestConfig]:
        """Generate all test configurations from sweep parameters"""
        configs = []

        # Create cartesian product of all parameter combinations
        param_names = list(sweep_params.keys())
        param_values = [sweep_params[name] for name in param_names]

        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))

            # Validate configuration before adding
            config = TestConfig(**param_dict)
            # Add basic validation using the is_invalid_config method
            if not self.is_invalid_config(config) and all(
                getattr(config, field) > 0
                for field in ["n_chunks", "chunk_n_pkts", "packet_size", "message_size", "total_messages", "n_workers"]
            ):
                configs.append(config)
            else:
                print(f"Skipping invalid config: {config}")

        return configs

    def run_sweep(self, sweep_params: Dict, max_failures: int = 10) -> None:
        """Run parameter sweep using either pipe mode or traditional mode"""
        configs = self.generate_sweep_configs(sweep_params)
        total_configs = len(configs)

        print(f"Generated {total_configs} test configurations")

        if self.use_pipe_mode:
            print("Using pipe mode (single process, multiple test cases)")
            self.run_pipe_mode_sweep(configs, max_failures)
        else:
            print("Using traditional mode (one process per test case)")
            self.run_traditional_sweep(configs, max_failures)

        print(f"\nCompleted sweep: {len([r for r in self.results if r.success])}/{len(self.results)} tests successful")

    def run_traditional_sweep(self, configs: List[TestConfig], max_failures: int = 10) -> None:
        """Run sweep using traditional mode - launch new process for each test"""
        total_configs = len(configs)
        failures = 0

        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{total_configs}] Running test configuration:")
            print(f"  {config}")

            result = self.run_single_test(config)
            self.results.append(result)

            if result.success:
                print(f"  ✓ Success: {result.throughput_gbps:.2f} GB/s")
            else:
                failures += 1
                print(f"  ✗ Failed: {result.error_msg}")
                if result.raw_output:
                    print(f"  Binary output:")
                    # Indent each line of output for better readability
                    for line in result.raw_output.strip().split("\n"):
                        print(f"    {line}")

                if failures >= max_failures:
                    print(f"\nStopping sweep after {failures} consecutive failures")
                    break

            # Save intermediate results
            if (i + 1) % 10 == 0:
                self.save_results()

    def save_results(self) -> None:
        """Save results to comprehensive CSV with all relevant differentiation columns"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save as comprehensive CSV
        csv_file = self.output_dir / f"results_{timestamp}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Comprehensive header with all relevant differentiation columns
            header = [
                # Test identification
                "test_id",
                "timestamp",
                "execution_mode",
                # Configuration parameters (sorted for consistency)
                "n_chunks",
                "chunk_n_pkts",
                "rx_chunk_n_pkts",
                "packet_size",
                "bidirectional",
                "message_size",
                "total_messages",
                "n_workers",
                "fabric_mcast_factor",
                # Calculated configuration metrics
                "total_buffer_size",
                "messages_per_worker",
                "bytes_per_chunk",
                "total_packets_per_chunk",
                # Performance results
                "throughput_gbps",
                "throughput_msgs_per_sec",
                "total_messages_sent",
                "total_cycles",
                "duration_us",
                # Calculated performance metrics
                "avg_cycles_per_message",
                "avg_us_per_message",
                "bandwidth_utilization_percent",
                "messages_per_second_per_worker",
                # Test status
                "success",
                "error_msg",
            ]
            writer.writerow(header)

            # Data rows with comprehensive information
            for i, result in enumerate(self.results):
                config = result.config

                # Calculate derived metrics
                total_buffer_size = config.n_chunks * config.chunk_n_pkts * config.packet_size
                messages_per_worker = config.total_messages if config.n_workers > 0 else 0
                bytes_per_chunk = config.chunk_n_pkts * config.packet_size
                total_packets_per_chunk = config.chunk_n_pkts

                # Performance calculations
                avg_cycles_per_message = (
                    (result.total_cycles / result.total_messages_sent) if result.total_messages_sent > 0 else 0
                )
                avg_us_per_message = (
                    (result.duration_us / result.total_messages_sent) if result.total_messages_sent > 0 else 0
                )

                # Bandwidth utilization (assuming theoretical max of 100 GB/s - adjust as needed)
                theoretical_max_gbps = 100.0  # This should be adjusted based on hardware specs
                bandwidth_utilization = (
                    (result.throughput_gbps / theoretical_max_gbps * 100) if theoretical_max_gbps > 0 else 0
                )

                messages_per_second_per_worker = (
                    (result.throughput_msgs_per_sec / config.n_workers) if config.n_workers > 0 else 0
                )

                row = [
                    # Test identification
                    i + 1,  # test_id
                    timestamp,
                    "pipe" if self.use_pipe_mode else "traditional",
                    # Configuration parameters
                    config.n_chunks,
                    config.chunk_n_pkts,
                    config.rx_chunk_n_pkts,
                    config.packet_size,
                    config.bidirectional,
                    config.message_size,
                    config.total_messages,
                    config.n_workers,
                    config.fabric_mcast_factor,
                    # Calculated configuration metrics
                    total_buffer_size,
                    messages_per_worker,
                    bytes_per_chunk,
                    total_packets_per_chunk,
                    # Performance results
                    result.throughput_gbps,
                    result.throughput_msgs_per_sec,
                    result.total_messages_sent,
                    result.total_cycles,
                    result.duration_us,
                    # Calculated performance metrics
                    round(avg_cycles_per_message, 2),
                    round(avg_us_per_message, 2),
                    round(bandwidth_utilization, 2),
                    round(messages_per_second_per_worker, 2),
                    # Test status
                    result.success,
                    result.error_msg or "",
                ]
                writer.writerow(row)

        # Save as JSON (excluding raw_output for size reasons)
        json_file = self.output_dir / f"results_{timestamp}.json"
        with open(json_file, "w") as f:
            results_without_raw_output = []
            for result in self.results:
                result_dict = asdict(result)
                result_dict.pop("raw_output", None)  # Remove raw_output to keep file size manageable
                results_without_raw_output.append(result_dict)
            json.dump(results_without_raw_output, f, indent=2)

        print(f"Results saved to {csv_file} and {json_file}")

        # Also create a consolidated results file for easy analysis
        self.save_consolidated_results()

    def save_consolidated_results(self) -> None:
        """Save only new results to consolidated CSV file to avoid duplication"""
        consolidated_file = self.output_dir / "consolidated_results.csv"

        # Only save new results since last consolidated save
        new_results = self.results[self.last_consolidated_count :]
        if not new_results:
            return  # Nothing new to save

        # Check if file already exists to determine if we need to write header
        file_exists = consolidated_file.exists()

        with open(consolidated_file, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header only if file doesn't exist
            if not file_exists:
                header = [
                    # Test identification
                    "run_timestamp",
                    "test_id",
                    "execution_mode",
                    # Configuration parameters
                    "n_chunks",
                    "chunk_n_pkts",
                    "rx_chunk_n_pkts",
                    "packet_size",
                    "bidirectional",
                    "message_size",
                    "total_messages",
                    "n_workers",
                    "fabric_mcast_factor",
                    # Calculated configuration metrics
                    "total_buffer_size",
                    "messages_per_worker",
                    "bytes_per_chunk",
                    "total_packets_per_chunk",
                    # Performance results
                    "throughput_gbps",
                    "throughput_msgs_per_sec",
                    "total_messages_sent",
                    "total_cycles",
                    "duration_us",
                    # Calculated performance metrics
                    "avg_cycles_per_message",
                    "avg_us_per_message",
                    "bandwidth_utilization_percent",
                    "messages_per_second_per_worker",
                    # Test status
                    "success",
                    "error_msg",
                ]
                writer.writerow(header)

            # Write only new data rows
            current_timestamp = time.strftime("%Y%m%d_%H%M%S")
            for i, result in enumerate(new_results, start=self.last_consolidated_count):
                config = result.config

                # Calculate derived metrics (same as save_results)
                total_buffer_size = config.n_chunks * config.chunk_n_pkts * config.packet_size
                messages_per_worker = config.total_messages if config.n_workers > 0 else 0
                bytes_per_chunk = config.chunk_n_pkts * config.packet_size
                total_packets_per_chunk = config.chunk_n_pkts

                avg_cycles_per_message = (
                    (result.total_cycles / result.total_messages_sent) if result.total_messages_sent > 0 else 0
                )
                avg_us_per_message = (
                    (result.duration_us / result.total_messages_sent) if result.total_messages_sent > 0 else 0
                )

                theoretical_max_gbps = 100.0
                bandwidth_utilization = (
                    (result.throughput_gbps / theoretical_max_gbps * 100) if theoretical_max_gbps > 0 else 0
                )
                messages_per_second_per_worker = (
                    (result.throughput_msgs_per_sec / config.n_workers) if config.n_workers > 0 else 0
                )

                row = [
                    # Test identification
                    current_timestamp,
                    i + 1,  # Global test ID across all results
                    "pipe" if self.use_pipe_mode else "traditional",
                    # Configuration parameters
                    config.n_chunks,
                    config.chunk_n_pkts,
                    config.rx_chunk_n_pkts,
                    config.packet_size,
                    config.bidirectional,
                    config.message_size,
                    config.total_messages,
                    config.n_workers,
                    config.fabric_mcast_factor,
                    # Calculated configuration metrics
                    total_buffer_size,
                    messages_per_worker,
                    bytes_per_chunk,
                    total_packets_per_chunk,
                    # Performance results
                    result.throughput_gbps,
                    result.throughput_msgs_per_sec,
                    result.total_messages_sent,
                    result.total_cycles,
                    result.duration_us,
                    # Calculated performance metrics
                    round(avg_cycles_per_message, 2),
                    round(avg_us_per_message, 2),
                    round(bandwidth_utilization, 2),
                    round(messages_per_second_per_worker, 2),
                    # Test status
                    result.success,
                    result.error_msg or "",
                ]
                writer.writerow(row)

        # Update the count of consolidated results
        self.last_consolidated_count = len(self.results)

        if new_results:
            print(f"Added {len(new_results)} new results to consolidated file: {consolidated_file}")

    def create_column_documentation(self) -> None:
        """Create a documentation file explaining all CSV columns"""
        doc_file = self.output_dir / "column_documentation.txt"

        documentation = """
CSV Column Documentation for Fabric Elastic Channels Test Results
================================================================

Test Identification Columns:
- run_timestamp: When this test run was executed (YYYYMMDD_HHMMSS format)
- test_id: Sequential test case number within this run (1, 2, 3, ...)
- execution_mode: How the test was run ("pipe" for single process mode, "traditional" for separate processes)

Configuration Parameters (Test Variables):
- n_chunks: Number of data chunks used in the test
- chunk_n_pkts: Number of packets per chunk for transmission
- rx_chunk_n_pkts: Number of packets per chunk for reception
- packet_size: Size of each packet in bytes
- bidirectional: Whether the test runs in both directions (True/False)
- message_size: Size of each message in bytes
- total_messages: Total number of messages to send per worker
- n_workers: Number of worker threads/processes
- fabric_mcast_factor: Fabric multicast factor setting

Calculated Configuration Metrics (Derived from Parameters):
- total_buffer_size: Total buffer size used (n_chunks * chunk_n_pkts * packet_size)
- messages_per_worker: Messages assigned per worker (total_messages per worker)
- bytes_per_chunk: Number of bytes per chunk (chunk_n_pkts * packet_size)
- total_packets_per_chunk: Same as chunk_n_pkts (for clarity in analysis)

Performance Results (Measured Output):
- throughput_gbps: Achieved throughput in gigabytes per second
- throughput_msgs_per_sec: Achieved throughput in messages per second
- total_messages_sent: Actual total messages sent (all workers combined)
- total_cycles: Total CPU/hardware cycles consumed
- duration_us: Test duration in microseconds

Calculated Performance Metrics (Derived from Results):
- avg_cycles_per_message: Average cycles needed per message (total_cycles / total_messages_sent)
- avg_us_per_message: Average microseconds per message (duration_us / total_messages_sent)
- bandwidth_utilization_percent: Percentage of theoretical maximum bandwidth used
- messages_per_second_per_worker: Per-worker message throughput (throughput_msgs_per_sec / n_workers)

Test Status:
- success: Whether the test completed successfully (True/False)
- error_msg: Error message if test failed (empty if successful)

Notes:
- All calculated metrics will be 0 if the test failed
- bandwidth_utilization_percent assumes 100 GB/s theoretical maximum (adjust as needed for your hardware)
- Empty error_msg indicates successful test completion
- Multiple test runs will be appended to consolidated_results.csv for easy cross-run analysis
"""

        with open(doc_file, "w") as f:
            f.write(documentation)

        print(f"Column documentation saved to: {doc_file}")

    def cleanup_consolidated_file(self) -> None:
        """Remove duplicates from consolidated results file"""
        consolidated_file = self.output_dir / "consolidated_results.csv"

        if not consolidated_file.exists():
            print("No consolidated file found to clean up")
            return

        print(f"Cleaning up duplicates in {consolidated_file}...")

        # Read the file and identify unique rows
        unique_rows = []
        seen_configs = set()

        with open(consolidated_file, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            unique_rows.append(header)

            for row in reader:
                if len(row) < 15:  # Skip malformed rows
                    continue

                # Create a signature from test configuration to identify duplicates
                # Using: run_timestamp, test_id, and key config parameters
                signature = (
                    row[0],  # run_timestamp
                    row[1],  # test_id
                    row[3],  # n_chunks
                    row[4],  # chunk_n_pkts
                    row[5],  # rx_chunk_n_pkts
                    row[6],  # packet_size
                    row[7],  # bidirectional
                    row[8],  # message_size
                    row[9],  # total_messages
                    row[10],  # n_workers
                    row[11],  # fabric_mcast_factor
                )

                if signature not in seen_configs:
                    seen_configs.add(signature)
                    unique_rows.append(row)

        # Create backup of original file
        backup_file = self.output_dir / "consolidated_results_backup.csv"
        consolidated_file.rename(backup_file)

        # Write cleaned file
        with open(consolidated_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(unique_rows)

        total_rows = len(unique_rows) - 1  # Subtract header
        print(f"Cleanup complete: {total_rows} unique test results retained")
        print(f"Original file backed up as: {backup_file}")
        print(f"Cleaned file saved as: {consolidated_file}")

    def generate_summary(self) -> None:
        """Generate summary statistics and best configurations"""
        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            print("No successful test results to summarize")
            return

        # Find best configurations
        best_throughput = max(successful_results, key=lambda r: r.throughput_gbps)

        # Generate summary
        summary = {
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(self.results) if self.results else 0,
            "best_throughput_gbps": best_throughput.throughput_gbps,
            "best_config": asdict(best_throughput.config),
            "throughput_stats": {
                "min": min(r.throughput_gbps for r in successful_results),
                "max": max(r.throughput_gbps for r in successful_results),
                "avg": sum(r.throughput_gbps for r in successful_results) / len(successful_results),
            },
        }

        # Save summary
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print(f"\n=== SWEEP SUMMARY ===")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']} ({summary['success_rate']:.1%})")
        print(f"Best throughput: {summary['best_throughput_gbps']:.2f} GB/s")
        print(f"Best config: {summary['best_config']}")
        print(
            f"Throughput range: {summary['throughput_stats']['min']:.2f} - {summary['throughput_stats']['max']:.2f} GB/s"
        )
        print(f"Average throughput: {summary['throughput_stats']['avg']:.2f} GB/s")
        print(f"Summary saved to {summary_file}")


def create_default_sweep_params() -> Dict:
    """Create default sweep parameters"""
    return {
        "n_chunks": [1, 2, 4],
        "chunk_n_pkts": [2, 3],
        "rx_chunk_n_pkts": [10, 11, 12, 13, 14, 15, 16],
        "packet_size": [4352],
        "bidirectional": [True],
        "message_size": [16, 2048, 4096],
        "total_messages": [10000],
        "n_workers": [1, 2, 3, 4],
        "fabric_mcast_factor": [1, 2, 3, 4],
    }


def main():
    parser = argparse.ArgumentParser(description="Fabric Elastic Channels Performance Sweep")
    parser.add_argument(
        "--output-dir", default="sweep_results", help="Output directory for results (default: sweep_results)"
    )
    parser.add_argument(
        "--config-file", help="JSON file with custom sweep parameters (if not provided, uses default parameters)"
    )
    parser.add_argument(
        "--max-failures", type=int, default=10, help="Maximum consecutive failures before stopping (default: 10)"
    )
    parser.add_argument(
        "--mode",
        choices=["pipe", "traditional"],
        default="pipe",
        help="Execution mode: 'pipe' for single process with JSON configs via stdin, 'traditional' for one process per test (default: pipe)",
    )
    parser.add_argument(
        "--cleanup-duplicates",
        action="store_true",
        help="Clean up duplicate entries in consolidated_results.csv and exit",
    )

    args = parser.parse_args()

    # Handle cleanup mode
    if args.cleanup_duplicates:
        print("Running in cleanup mode...")
        sweep = FabricChannelsSweep("", args.output_dir, False)  # Dummy setup for cleanup
        sweep.cleanup_consolidated_file()
        sys.exit(0)

    # Use hardcoded executable path
    executable_path_str = "./build/test/tt_metal/tt_fabric/fabric_elastic_channels_host_test"
    executable_path = Path(executable_path_str)
    if not executable_path.exists():
        print(f"Error: Executable not found: {executable_path_str}")
        print("Please build the test first:")
        print("  cd build && make fabric_elastic_channels_host_test -j$(nproc)")
        sys.exit(1)

    # Load sweep parameters
    if args.config_file:
        try:
            with open(args.config_file, "r") as f:
                sweep_params = json.load(f)
            print(f"Loaded sweep parameters from {args.config_file}")
        except FileNotFoundError:
            print(f"Error: Config file not found: {args.config_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}")
            sys.exit(1)
    else:
        sweep_params = create_default_sweep_params()
        print("Using default sweep parameters (no config file provided)")

    # Run sweep
    use_pipe_mode = args.mode == "pipe"
    sweep = FabricChannelsSweep(executable_path_str, args.output_dir, use_pipe_mode)

    # Create column documentation for analysis
    sweep.create_column_documentation()

    try:
        sweep.run_sweep(sweep_params, args.max_failures)
        sweep.save_results()
        sweep.generate_summary()
    except KeyboardInterrupt:
        print("\nSweep interrupted by user")
        sweep.save_results()
        sweep.generate_summary()
    except Exception as e:
        print(f"Sweep failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
