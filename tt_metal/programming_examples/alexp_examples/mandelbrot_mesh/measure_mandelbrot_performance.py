#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple Performance Measurement Script for Mandelbrot Implementations

This script provides an easy way to measure and compare performance
of different Mandelbrot implementations on Tenstorrent hardware.
"""

import subprocess
import time
import json
import os
import sys
from pathlib import Path
import argparse


class MandelbrotPerformanceMeasurement:
    def __init__(self, build_dir="/home/tt-metal-apv/build-cmake"):
        self.build_dir = Path(build_dir)
        self.results_dir = Path("./performance_results")
        self.results_dir.mkdir(exist_ok=True)

    def measure_execution_time(self, executable_path, args=None, num_runs=3):
        """Measure execution time of a program with multiple runs for accuracy."""
        if args is None:
            args = []

        times = []

        print(f"🚀 Measuring performance of: {executable_path}")
        print(f"   Arguments: {' '.join(args)}")
        print(f"   Runs: {num_runs}")

        for run in range(num_runs):
            print(f"   Run {run + 1}/{num_runs}...", end=" ", flush=True)

            start_time = time.perf_counter()

            try:
                result = subprocess.run(
                    [str(executable_path)] + args, capture_output=True, text=True, timeout=300  # 5 minute timeout
                )

                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

                if result.returncode == 0:
                    times.append(execution_time)
                    print(f"{execution_time:.2f}ms ✅")
                else:
                    print(f"❌ Failed (exit code: {result.returncode})")
                    print(f"   Error: {result.stderr}")
                    return None

            except subprocess.TimeoutExpired:
                print("❌ Timeout")
                return None
            except Exception as e:
                print(f"❌ Error: {e}")
                return None

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            return {
                "times": times,
                "avg_ms": avg_time,
                "min_ms": min_time,
                "max_ms": max_time,
                "std_dev": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5,
            }

        return None

    def calculate_throughput_metrics(self, timing_result, width=1024, height=1024, max_iterations=100):
        """Calculate throughput and performance metrics."""
        if not timing_result:
            return None

        total_pixels = width * height
        total_operations = total_pixels * max_iterations
        execution_time_s = timing_result["avg_ms"] / 1000.0

        metrics = {
            "total_pixels": total_pixels,
            "total_operations": total_operations,
            "execution_time_ms": timing_result["avg_ms"],
            "pixels_per_second": total_pixels / execution_time_s,
            "operations_per_second": total_operations / execution_time_s,
            "estimated_gflops": (total_operations * 10) / (execution_time_s * 1e9),  # ~10 FLOPs per iteration
            "timing_stats": timing_result,
        }

        return metrics

    def measure_mandelbrot_implementations(self, image_size=1024, max_iterations=100, num_runs=3):
        """Measure performance of available Mandelbrot implementations."""

        implementations = [
            {"name": "Multi-Core Mesh", "executable": "mandelbrot_multi_core_mesh", "args": []},
            {"name": "Simple Mesh", "executable": "mandelbrot_mesh_simple", "args": []},
            {"name": "Regular Mesh", "executable": "mandelbrot_mesh", "args": []},
        ]

        results = {}

        print(f"📊 Performance Measurement Configuration:")
        print(f"   • Image size: {image_size}x{image_size}")
        print(f"   • Max iterations: {max_iterations}")
        print(f"   • Measurement runs: {num_runs}")
        print(f"   • Build directory: {self.build_dir}")
        print("")

        for impl in implementations:
            executable_path = self.build_dir / "programming_examples" / impl["executable"]

            if executable_path.exists():
                timing_result = self.measure_execution_time(executable_path, impl["args"], num_runs)

                if timing_result:
                    metrics = self.calculate_throughput_metrics(timing_result, image_size, image_size, max_iterations)
                    results[impl["name"]] = metrics
                else:
                    print(f"❌ Failed to measure {impl['name']}")
            else:
                print(f"⚠️  Executable not found: {executable_path}")

        return results

    def print_comparison_table(self, results):
        """Print a formatted comparison table of results."""
        if not results:
            print("❌ No results to display")
            return

        print("\n🏆 Performance Comparison Results")
        print("=" * 80)

        # Header
        print(f"{'Implementation':<20} {'Time (ms)':<12} {'Pixels/sec':<15} {'Ops/sec':<15} {'GFLOPS':<10}")
        print("-" * 80)

        # Sort by execution time (fastest first)
        sorted_results = sorted(results.items(), key=lambda x: x[1]["execution_time_ms"])

        baseline_time = None

        for name, metrics in sorted_results:
            if baseline_time is None:
                baseline_time = metrics["execution_time_ms"]

            speedup = baseline_time / metrics["execution_time_ms"] if metrics["execution_time_ms"] > 0 else 0

            print(
                f"{name:<20} {metrics['execution_time_ms']:<12.2f} "
                f"{metrics['pixels_per_second']:<15.2e} "
                f"{metrics['operations_per_second']:<15.2e} "
                f"{metrics['estimated_gflops']:<10.2f}"
            )

        print("-" * 80)
        print(f"Fastest implementation: {sorted_results[0][0]}")

        # Show speedup comparison
        print(f"\nSpeedup vs slowest:")
        slowest_time = sorted_results[-1][1]["execution_time_ms"]
        for name, metrics in sorted_results:
            speedup = slowest_time / metrics["execution_time_ms"]
            print(f"  {name}: {speedup:.2f}x")

    def save_results_to_json(self, results, filename="mandelbrot_performance_results.json"):
        """Save results to JSON file for later analysis."""
        output_file = self.results_dir / filename

        # Add timestamp and configuration
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {"build_dir": str(self.build_dir), "results_dir": str(self.results_dir)},
            "results": results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n📄 Results saved to: {output_file}")

    def run_cpu_reference_benchmark(self, width=1024, height=1024, max_iterations=100):
        """Run a simple CPU reference implementation for comparison."""
        print("\n🖥️  Running CPU reference benchmark...")

        import math

        def mandelbrot_cpu(width, height, max_iterations):
            x_min, x_max = -2.5, 1.5
            y_min, y_max = -2.0, 2.0

            dx = (x_max - x_min) / width
            dy = (y_max - y_min) / height

            result = []

            for y in range(height):
                for x in range(width):
                    cx = x_min + x * dx
                    cy = y_min + y * dy

                    zx, zy = 0, 0
                    iterations = 0

                    for iterations in range(max_iterations):
                        if zx * zx + zy * zy > 4.0:
                            break
                        zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy

                    result.append(iterations)

            return result

        start_time = time.perf_counter()
        mandelbrot_cpu(width, height, max_iterations)
        end_time = time.perf_counter()

        cpu_time_ms = (end_time - start_time) * 1000

        cpu_metrics = self.calculate_throughput_metrics(
            {"avg_ms": cpu_time_ms, "min_ms": cpu_time_ms, "max_ms": cpu_time_ms, "times": [cpu_time_ms]},
            width,
            height,
            max_iterations,
        )

        print(f"   CPU execution time: {cpu_time_ms:.2f} ms")

        return cpu_metrics


def main():
    parser = argparse.ArgumentParser(description="Measure Mandelbrot implementation performance")
    parser.add_argument("--size", type=int, default=1024, help="Image size (width=height)")
    parser.add_argument("--iterations", type=int, default=100, help="Max Mandelbrot iterations")
    parser.add_argument("--runs", type=int, default=3, help="Number of measurement runs")
    parser.add_argument("--build-dir", default="/home/tt-metal-apv/build-cmake", help="Build directory")
    parser.add_argument("--include-cpu", action="store_true", help="Include CPU reference benchmark")
    parser.add_argument("--output", help="Output JSON filename")

    args = parser.parse_args()

    print("🚀 Tenstorrent Mandelbrot Performance Measurement")
    print("=" * 50)

    # Initialize measurement system
    perf_measure = MandelbrotPerformanceMeasurement(args.build_dir)

    # Check if build directory exists
    if not Path(args.build_dir).exists():
        print(f"❌ Build directory not found: {args.build_dir}")
        print("Please build the project first or specify correct build directory with --build-dir")
        return 1

    # Measure TT-Metal implementations
    results = perf_measure.measure_mandelbrot_implementations(args.size, args.iterations, args.runs)

    # Add CPU reference if requested
    if args.include_cpu:
        try:
            cpu_metrics = perf_measure.run_cpu_reference_benchmark(args.size, args.size, args.iterations)
            results["CPU Reference"] = cpu_metrics
        except Exception as e:
            print(f"❌ CPU benchmark failed: {e}")

    # Display results
    if results:
        perf_measure.print_comparison_table(results)

        # Save results
        output_filename = args.output or "mandelbrot_performance_results.json"
        perf_measure.save_results_to_json(results, output_filename)

        print("\n🎉 Performance measurement completed successfully!")

        # Performance tips
        print("\n💡 Performance Optimization Tips:")
        print("   • Use larger image sizes to better utilize parallel cores")
        print("   • Higher iteration counts show compute-bound performance")
        print("   • Multi-core implementations should show significant speedup")
        print("   • Check profiler results (if enabled) for detailed analysis")

    else:
        print("❌ No implementations could be measured")
        print("Make sure executables are built and available in the build directory")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
