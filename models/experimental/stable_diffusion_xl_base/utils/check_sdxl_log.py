# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import re
import sys
import numpy as np
from typing import List, Tuple
import argparse


def extract_times(log_file: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract timing information for denoising loop, VAE decoding, and output tensor read from log file.

    Args:
        log_file: Path to the log file

    Returns:
        Tuple of (denoising_times, vae_times, output_read_times)
    """
    denoising_pattern = r"Denoising loop for \d+ promts completed in (\d+\.\d+) seconds"
    vae_pattern = r"On device VAE decoding completed in (\d+\.\d+) seconds"
    output_read_pattern = r"Output tensor read completed in (\d+\.\d+) seconds"

    denoising_times = []
    vae_times = []
    output_read_times = []

    try:
        with open(log_file, "r") as f:
            for line in f:
                # Check for denoising loop time
                denoising_match = re.search(denoising_pattern, line)
                if denoising_match:
                    denoising_times.append(float(denoising_match.group(1)))

                # Check for VAE decoding time
                vae_match = re.search(vae_pattern, line)
                if vae_match:
                    vae_times.append(float(vae_match.group(1)))

                # Check for output tensor read time
                output_read_match = re.search(output_read_pattern, line)
                if output_read_match:
                    output_read_times.append(float(output_read_match.group(1)))
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

    return denoising_times, vae_times, output_read_times


def calculate_stats(times: List[float]) -> Tuple[float, float, float, float]:
    """
    Calculate statistics for a list of times.

    Args:
        times: List of time values

    Returns:
        Tuple of (average, minimum, maximum, standard deviation)
    """
    if not times:
        return 0.0, 0.0, 0.0, 0.0

    avg = np.mean(times)
    min_val = np.min(times)
    max_val = np.max(times)
    std = np.std(times)

    return avg, min_val, max_val, std


def print_stats(name: str, stats: Tuple[float, float, float, float], count: int) -> None:
    """
    Print statistics in a formatted way.

    Args:
        name: Name of the metric
        stats: Tuple of (average, minimum, maximum, standard deviation)
        count: Number of samples
    """
    avg, min_val, max_val, std = stats

    print(f"{name} Statistics (samples: {count}):")
    print(f"  Average: {avg:.2f} seconds")
    print(f"  Minimum: {min_val:.2f} seconds")
    print(f"  Maximum: {max_val:.2f} seconds")
    print(f"  Std Dev: {std:.2f} seconds")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze performance timing from log files")
    parser.add_argument("log_file", type=str, help="Path to the log file to analyze")

    # Parse arguments
    args = parser.parse_args()

    log_file = args.log_file
    denoising_times, vae_times, output_read_times = extract_times(log_file)

    # Calculate statistics
    denoising_stats = calculate_stats(denoising_times)
    vae_stats = calculate_stats(vae_times)
    output_read_stats = calculate_stats(output_read_times)

    # Print results
    print()
    print("=" * 80)
    print(f"Log Analysis Results for: {log_file}")
    print("-" * 50)
    print_stats("Denoising Loop", denoising_stats, len(denoising_times))
    print_stats("VAE Decoding", vae_stats, len(vae_times))
    print_stats("Output Tensor Read", output_read_stats, len(output_read_times))
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
