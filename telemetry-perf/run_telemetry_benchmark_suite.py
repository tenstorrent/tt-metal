#!/usr/bin/env python3
"""
Main Orchestrator for Comprehensive Telemetry Performance Benchmark Suite

Coordinates execution of all benchmark tests:
  1. Core hypothesis validation (--mmio-only vs full mode)
  2. Single-device comprehensive benchmark
  3. Multi-device comprehensive benchmark
  4. Sustained workload test
  5. Final analysis and report generation

Supports two phases:
  - Phase 1 (Reduced): ~2-3 hours, validates core hypotheses
  - Phase 2 (Full): ~9-12 hours, comprehensive analysis
"""

import sys
import time
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from telemetry_benchmark_utils import save_results_json, load_results_json


class BenchmarkOrchestrator:
    """Orchestrates execution of all benchmark tests."""

    def __init__(self, phase: str = "reduced", output_dir: str = "/tmp"):
        self.phase = phase
        self.output_dir = Path(output_dir)
        self.start_time = None
        self.results = {}
        self.errors = []

    def run_script(self, script_path: str, args: List[str] = None, timeout: int = None) -> Dict[str, Any]:
        """
        Run a Python script and capture results.

        Args:
            script_path: Path to Python script
            args: Command line arguments
            timeout: Timeout in seconds

        Returns:
            Dictionary with status and result file path
        """
        script_name = Path(script_path).stem
        print(f"\n{'='*80}")
        print(f"Running: {script_name}")
        print(f"{'='*80}")

        cmd = ["python3", script_path]
        if args:
            cmd.extend(args)

        start = time.time()

        try:
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)

            duration = time.time() - start

            print(f"\nCompleted in {duration:.1f}s")
            print(f"Return code: {result.returncode}")

            if result.returncode != 0:
                print(f"STDERR:\n{result.stderr}")
                return {
                    "script": script_name,
                    "status": "failed",
                    "return_code": result.returncode,
                    "duration_sec": duration,
                    "error": result.stderr,
                }

            return {
                "script": script_name,
                "status": "success",
                "return_code": result.returncode,
                "duration_sec": duration,
                "stdout": result.stdout[:1000] if result.stdout else "",  # First 1000 chars
            }

        except subprocess.TimeoutExpired:
            duration = time.time() - start
            print(f"ERROR: Script timed out after {timeout}s")
            return {"script": script_name, "status": "timeout", "duration_sec": duration, "timeout_sec": timeout}

        except Exception as e:
            duration = time.time() - start
            print(f"ERROR: {e}")
            return {"script": script_name, "status": "error", "duration_sec": duration, "error": str(e)}

    def run_core_hypothesis_test(self) -> Dict[str, Any]:
        """Run core hypothesis validation test."""
        print("\n" + "=" * 80)
        print("STEP 1: CORE HYPOTHESIS VALIDATION")
        print("=" * 80)

        result = self.run_script(str(SCRIPT_DIR / "validate_mmio_only.py"), timeout=3600)  # 1 hour timeout

        self.results["core_hypothesis"] = result

        # Load results if successful
        if result["status"] == "success":
            try:
                data = load_results_json("/tmp/mmio_validation_results.json")
                result["analysis"] = data.get("analysis", {})
            except Exception as e:
                print(f"Warning: Could not load results: {e}")

        return result

    def run_single_device_benchmark(self) -> Dict[str, Any]:
        """Run single-device comprehensive benchmark."""
        print("\n" + "=" * 80)
        print("STEP 2: SINGLE-DEVICE COMPREHENSIVE BENCHMARK")
        print("=" * 80)

        # Timeout: Phase 1 ~1.5 hours, Phase 2 ~6 hours
        timeout = 6000 if self.phase == "reduced" else 25000

        result = self.run_script(
            str(SCRIPT_DIR / "comprehensive_single_device_benchmark.py"), args=[self.phase], timeout=timeout
        )

        self.results["single_device"] = result

        # Load results if successful
        if result["status"] == "success":
            try:
                data = load_results_json(f"/tmp/single_device_results_{self.phase}.json")
                result["analysis"] = data.get("analysis", {})
            except Exception as e:
                print(f"Warning: Could not load results: {e}")

        return result

    def run_multi_device_benchmark(self) -> Dict[str, Any]:
        """Run multi-device comprehensive benchmark."""
        print("\n" + "=" * 80)
        print("STEP 3: MULTI-DEVICE COMPREHENSIVE BENCHMARK")
        print("=" * 80)

        # Timeout: Phase 1 ~2 hours, Phase 2 ~5 hours
        timeout = 8000 if self.phase == "reduced" else 20000

        result = self.run_script(
            str(SCRIPT_DIR / "comprehensive_multi_device_benchmark.py"), args=[self.phase], timeout=timeout
        )

        self.results["multi_device"] = result

        # Load results if successful
        if result["status"] == "success":
            try:
                data = load_results_json(f"/tmp/multi_device_results_{self.phase}.json")
                result["analysis"] = data.get("analysis", {})
            except Exception as e:
                print(f"Warning: Could not load results: {e}")

        return result

    def run_sustained_workload_test(self) -> Dict[str, Any]:
        """Run sustained workload test."""
        print("\n" + "=" * 80)
        print("STEP 4: SUSTAINED WORKLOAD TEST")
        print("=" * 80)

        # Reduced: 1000 iterations (~20 min), Full: 300s (~5 min)
        if self.phase == "reduced":
            args = ["1000"]  # 1000 iterations
            timeout = 2400  # 40 min timeout
        else:
            args = ["300s"]  # 5 minutes
            timeout = 1200  # 20 min timeout

        result = self.run_script(str(SCRIPT_DIR / "sustained_workload_test.py"), args=args, timeout=timeout)

        self.results["sustained_workload"] = result

        # Load results if successful
        if result["status"] == "success":
            try:
                data = load_results_json("/tmp/sustained_workload_results.json")
                result["analysis"] = data.get("analysis", {})
            except Exception as e:
                print(f"Warning: Could not load results: {e}")

        return result

    def generate_final_report(self) -> str:
        """
        Generate final markdown report summarizing all results.

        Returns:
            Path to generated report
        """
        print("\n" + "=" * 80)
        print("GENERATING FINAL REPORT")
        print("=" * 80)

        total_duration = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Comprehensive Telemetry Performance Benchmark Report

**Phase:** {self.phase.upper()}
**Date:** {timestamp}
**Total Duration:** {total_duration/3600:.2f} hours

---

## Executive Summary

"""

        # Core hypothesis results
        if "core_hypothesis" in self.results:
            ch_result = self.results["core_hypothesis"]
            report += f"""### Core Hypothesis Validation

**Status:** {ch_result['status']}
**Duration:** {ch_result.get('duration_sec', 0)/60:.1f} minutes

"""
            if ch_result["status"] == "success" and "analysis" in ch_result:
                analysis = ch_result["analysis"]
                validated = analysis.get("hypothesis_validated", False)

                if validated:
                    report += """**Result:** ✓ HYPOTHESIS VALIDATED

The test confirms that `--mmio-only` successfully prevents ERISC contention:
- MMIO-only mode shows <5% impact on all multi-chip workloads
- Full mode shows ≥10% impact or failures on all multi-chip workloads
- Conclusion: **Use --mmio-only flag for multi-chip telemetry**

"""
                else:
                    report += """**Result:** ✗ HYPOTHESIS NOT VALIDATED

Some tests did not meet expected criteria. Review detailed results for more information.

"""
            else:
                report += f"""**Result:** Test {ch_result['status']}

"""

        # Single-device results
        if "single_device" in self.results:
            sd_result = self.results["single_device"]
            report += f"""### Single-Device Benchmark

**Status:** {sd_result['status']}
**Duration:** {sd_result.get('duration_sec', 0)/3600:.2f} hours

"""
            if sd_result["status"] == "success" and "analysis" in sd_result:
                summary = sd_result["analysis"].get("summary", {})
                report += f"""**Results:**
- Total tests: {summary.get('total_tests', 0)}
- Successful: {summary.get('successful_tests', 0)}
- Failed: {summary.get('failed_tests', 0)}
- Significant impacts (uncorrected): {summary.get('significant_impacts_uncorrected', 0)}
- Monotonic relationships detected: {summary.get('monotonic_relationships', 0)}

"""

                if summary.get("significant_impacts_uncorrected", 0) == 0:
                    report += """**Conclusion:** No significant telemetry impact detected on single-device workloads.

"""
                else:
                    report += """**Conclusion:** Some significant impacts detected. Review detailed results.

"""
            else:
                report += f"""**Result:** Test {sd_result['status']}

"""

        # Multi-device results
        if "multi_device" in self.results:
            md_result = self.results["multi_device"]
            report += f"""### Multi-Device Benchmark

**Status:** {md_result['status']}
**Duration:** {md_result.get('duration_sec', 0)/3600:.2f} hours

"""
            if md_result["status"] == "success" and "analysis" in md_result:
                summary = md_result["analysis"].get("summary", {})
                report += f"""**Results:**
- Total tests: {summary.get('total_tests', 0)}
- Successful: {summary.get('successful_tests', 0)}
- Failed: {summary.get('failed_tests', 0)}
- Significant impacts (uncorrected): {summary.get('significant_impacts_uncorrected', 0)}
- Monotonic relationships detected: {summary.get('monotonic_relationships', 0)}
- Notable MMIO vs Full differences: {summary.get('mmio_vs_full_differences', 0)}

"""

                if summary.get("mmio_vs_full_differences", 0) > 0:
                    report += """**Conclusion:** MMIO-only and full modes show different impact profiles.

"""
                else:
                    report += """**Conclusion:** No significant difference between MMIO-only and full modes detected.

"""
            else:
                report += f"""**Result:** Test {md_result['status']}

"""

        # Sustained workload results
        if "sustained_workload" in self.results:
            sw_result = self.results["sustained_workload"]
            report += f"""### Sustained Workload Test

**Status:** {sw_result['status']}
**Duration:** {sw_result.get('duration_sec', 0)/60:.1f} minutes

"""
            if sw_result["status"] == "success" and "analysis" in sw_result:
                summary = sw_result["analysis"].get("summary", {})
                baseline_drift = summary.get("baseline_drift", 0)
                max_drift = summary.get("max_additional_drift", 0)
                exceeded = summary.get("drift_threshold_exceeded", False)

                report += f"""**Results:**
- Baseline drift: {baseline_drift:+.2f}%
- Max additional drift (with telemetry): {max_drift:+.2f}%
- Drift threshold (5%) exceeded: {exceeded}

"""

                if not exceeded:
                    report += """**Conclusion:** No significant performance drift detected over sustained workload.

"""
                else:
                    report += """**Conclusion:** ⚠️ Significant drift detected. Telemetry may accumulate overhead over time.

"""
            else:
                report += f"""**Result:** Test {sw_result['status']}

"""

        # Overall conclusions
        report += """---

## Overall Conclusions

"""

        # Determine overall conclusions
        core_validated = False
        if "core_hypothesis" in self.results:
            ch_result = self.results["core_hypothesis"]
            if ch_result["status"] == "success" and "analysis" in ch_result:
                core_validated = ch_result["analysis"].get("hypothesis_validated", False)

        if core_validated:
            report += """### Recommendation

**Use `--mmio-only` flag for multi-chip telemetry:**
```bash
./build/tools/tt-telemetry/tt-telemetry --mmio-only --logging-interval 100ms
```

This configuration:
- Prevents ERISC contention on multi-chip workloads
- Shows minimal (<5%) performance impact
- Provides sufficient telemetry coverage for monitoring

"""
        else:
            report += """### Recommendation

Review detailed test results to understand telemetry impact on your specific workload.

"""

        report += """---

## Detailed Results

Detailed results are available in the following files:
- Core hypothesis: `/tmp/mmio_validation_results.json`
- Single-device: `/tmp/single_device_results_{phase}.json`
- Multi-device: `/tmp/multi_device_results_{phase}.json`
- Sustained workload: `/tmp/sustained_workload_results.json`

---

## Test Configuration

"""

        report += f"""**Phase:** {self.phase}

"""

        if self.phase == "reduced":
            report += """### Phase 1 (Reduced) Configuration

- **Tensor sizes:** 3 (1024², 8192², 17408²)
- **Polling frequencies:** 6 (60s, 1s, 100ms, 10ms, 1ms, 100us)
- **Single-device operations:** 3 (matmul, add, to_memory_config)
- **Multi-device operations:** 1 (AllGather)
- **Device counts:** 3 (2, 4, 8)
- **Memory configs:** DRAM only
- **Samples per config:** 100
- **Warmup iterations:** 20

**Estimated runtime:** 2-3 hours

"""
        else:
            report += """### Phase 2 (Full) Configuration

- **Tensor sizes:** 5 (1024² to 17408²)
- **Polling frequencies:** 12 (60s to 100us)
- **Single-device operations:** 5 (matmul, add, concat, to_memory_config, reshape)
- **Multi-device operations:** 3 (AllGather, ReduceScatter, AllReduce)
- **Device counts:** 3 (2, 4, 8)
- **Memory configs:** L1 + DRAM
- **Samples per config:** 100
- **Warmup iterations:** adaptive (up to 50)

**Estimated runtime:** 9-12 hours

"""

        report += f"""---

*Report generated on {timestamp}*
*Total benchmark duration: {total_duration/3600:.2f} hours*
"""

        # Save report
        report_path = self.output_dir / f"telemetry_final_report_{self.phase}.md"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"Report saved to: {report_path}")

        return str(report_path)

    def run(self) -> int:
        """
        Run complete benchmark suite.

        Returns:
            Exit code (0 = success)
        """
        self.start_time = time.time()

        print("=" * 80)
        print(f"COMPREHENSIVE TELEMETRY PERFORMANCE BENCHMARK SUITE - {self.phase.upper()} PHASE")
        print("=" * 80)
        print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Estimate duration
        if self.phase == "reduced":
            print("Estimated duration: 2-3 hours")
        else:
            print("Estimated duration: 9-12 hours")

        # Run all tests
        try:
            # Step 1: Core hypothesis validation
            self.run_core_hypothesis_test()

            # Step 2: Single-device benchmark
            self.run_single_device_benchmark()

            # Step 3: Multi-device benchmark
            self.run_multi_device_benchmark()

            # Step 4: Sustained workload test
            self.run_sustained_workload_test()

        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            self.errors.append("Interrupted by user")

        except Exception as e:
            print(f"\n\nBenchmark failed with exception: {e}")
            self.errors.append(str(e))

        # Generate final report
        try:
            report_path = self.generate_final_report()
        except Exception as e:
            print(f"Failed to generate report: {e}")
            report_path = None

        # Save orchestrator results
        orchestrator_results = {
            "phase": self.phase,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_sec": time.time() - self.start_time,
            "results": self.results,
            "errors": self.errors,
            "report_path": report_path,
        }

        save_results_json(orchestrator_results, str(self.output_dir / f"benchmark_orchestrator_{self.phase}.json"))

        # Print summary
        print("\n" + "=" * 80)
        print("BENCHMARK SUITE COMPLETE")
        print("=" * 80)
        print(f"\nTotal duration: {(time.time() - self.start_time)/3600:.2f} hours")

        successful = sum(1 for r in self.results.values() if r.get("status") == "success")
        total = len(self.results)
        print(f"Successful tests: {successful}/{total}")

        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")

        if report_path:
            print(f"\nFinal report: {report_path}")

        print("=" * 80)

        # Return 0 if all tests successful, 1 otherwise
        return 0 if successful == total and not self.errors else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Telemetry Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run reduced phase (~2-3 hours)
  python3 run_telemetry_benchmark_suite.py --phase reduced

  # Run full phase (~9-12 hours)
  python3 run_telemetry_benchmark_suite.py --phase full

  # Specify output directory
  python3 run_telemetry_benchmark_suite.py --phase reduced --output /tmp/results
        """,
    )

    parser.add_argument(
        "--phase",
        choices=["reduced", "full"],
        default="reduced",
        help="Benchmark phase: 'reduced' (~2-3 hours) or 'full' (~9-12 hours)",
    )

    parser.add_argument("--output", default="/tmp", help="Output directory for results (default: /tmp)")

    args = parser.parse_args()

    # Create orchestrator and run
    orchestrator = BenchmarkOrchestrator(phase=args.phase, output_dir=args.output)

    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
