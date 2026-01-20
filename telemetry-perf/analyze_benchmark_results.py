#!/usr/bin/env python3
"""
Post-Analysis Script for Telemetry Benchmark Results

Loads all benchmark results and generates:
  - Detailed statistical analysis
  - Impact distribution analysis
  - Monotonicity validation
  - Recommendations for production deployment

Can be run independently after benchmarks complete.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from statistics import mean, median, stdev
import numpy as np

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from telemetry_benchmark_utils import (
    load_results_json,
    parse_frequency_to_hz,
    calculate_impact_percentage,
    analyze_monotonicity,
    apply_multiple_comparison_correction,
    save_results_json,
)


class BenchmarkAnalyzer:
    """Analyzes completed benchmark results."""

    def __init__(self, phase: str = "reduced", output_dir: str = "/tmp"):
        self.phase = phase
        self.output_dir = Path(output_dir)
        self.results = {}
        self.analysis = {}

    def load_results(self):
        """Load all result files."""
        print("Loading benchmark results...")

        files = {
            "core_hypothesis": f"/tmp/mmio_validation_results.json",
            "single_device": f"/tmp/single_device_results_{self.phase}.json",
            "multi_device": f"/tmp/multi_device_results_{self.phase}.json",
            "sustained_workload": f"/tmp/sustained_workload_results.json",
        }

        for name, filepath in files.items():
            try:
                self.results[name] = load_results_json(filepath)
                print(f"  ✓ Loaded {name}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
                self.results[name] = None

    def analyze_impact_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of performance impacts."""
        print("\nAnalyzing impact distribution...")

        analysis = {"single_device": {}, "multi_device": {}}

        # Single-device impacts
        if self.results.get("single_device"):
            sd_results = self.results["single_device"].get("results", [])
            impacts = []

            for result in sd_results:
                config = result.get("config", {})
                if config.get("telemetry_enabled") and result.get("samples"):
                    # Find corresponding baseline
                    baseline = self._find_baseline(sd_results, config)
                    if baseline and baseline.get("samples"):
                        baseline_mean = mean(baseline["samples"])
                        telemetry_mean = mean(result["samples"])
                        impact = calculate_impact_percentage(baseline_mean, telemetry_mean)
                        impacts.append(impact)

            if impacts:
                analysis["single_device"] = {
                    "n_tests": len(impacts),
                    "mean_impact": float(mean(impacts)),
                    "median_impact": float(median(impacts)),
                    "std_impact": float(stdev(impacts)) if len(impacts) > 1 else 0,
                    "min_impact": float(min(impacts)),
                    "max_impact": float(max(impacts)),
                    "impacts_over_5pct": sum(1 for i in impacts if abs(i) >= 5.0),
                    "impacts_over_10pct": sum(1 for i in impacts if abs(i) >= 10.0),
                }

                print(f"  Single-device: {len(impacts)} tests")
                print(f"    Mean impact: {analysis['single_device']['mean_impact']:.2f}%")
                print(f"    Median impact: {analysis['single_device']['median_impact']:.2f}%")
                print(f"    Impacts >5%: {analysis['single_device']['impacts_over_5pct']}")

        # Multi-device impacts
        if self.results.get("multi_device"):
            md_results = self.results["multi_device"].get("results", [])

            mmio_impacts = []
            full_impacts = []

            for result in md_results:
                config = result.get("config", {})
                mode = config.get("telemetry_mode")

                if mode in ["mmio_only", "full"] and result.get("samples"):
                    # Find corresponding baseline
                    baseline = self._find_baseline(md_results, config)
                    if baseline and baseline.get("samples"):
                        baseline_mean = mean(baseline["samples"])
                        telemetry_mean = mean(result["samples"])
                        impact = calculate_impact_percentage(baseline_mean, telemetry_mean)

                        if mode == "mmio_only":
                            mmio_impacts.append(impact)
                        else:
                            full_impacts.append(impact)

            if mmio_impacts:
                analysis["multi_device"]["mmio_only"] = {
                    "n_tests": len(mmio_impacts),
                    "mean_impact": float(mean(mmio_impacts)),
                    "median_impact": float(median(mmio_impacts)),
                    "std_impact": float(stdev(mmio_impacts)) if len(mmio_impacts) > 1 else 0,
                    "min_impact": float(min(mmio_impacts)),
                    "max_impact": float(max(mmio_impacts)),
                    "impacts_over_5pct": sum(1 for i in mmio_impacts if abs(i) >= 5.0),
                    "impacts_over_10pct": sum(1 for i in mmio_impacts if abs(i) >= 10.0),
                }

                print(f"  Multi-device (MMIO-only): {len(mmio_impacts)} tests")
                print(f"    Mean impact: {analysis['multi_device']['mmio_only']['mean_impact']:.2f}%")
                print(f"    Impacts >5%: {analysis['multi_device']['mmio_only']['impacts_over_5pct']}")

            if full_impacts:
                analysis["multi_device"]["full"] = {
                    "n_tests": len(full_impacts),
                    "mean_impact": float(mean(full_impacts)),
                    "median_impact": float(median(full_impacts)),
                    "std_impact": float(stdev(full_impacts)) if len(full_impacts) > 1 else 0,
                    "min_impact": float(min(full_impacts)),
                    "max_impact": float(max(full_impacts)),
                    "impacts_over_5pct": sum(1 for i in full_impacts if abs(i) >= 5.0),
                    "impacts_over_10pct": sum(1 for i in full_impacts if abs(i) >= 10.0),
                }

                print(f"  Multi-device (Full): {len(full_impacts)} tests")
                print(f"    Mean impact: {analysis['multi_device']['full']['mean_impact']:.2f}%")
                print(f"    Impacts >5%: {analysis['multi_device']['full']['impacts_over_5pct']}")

        return analysis

    def _find_baseline(self, results: List[Dict], config: Dict) -> Dict:
        """Find baseline result matching a configuration."""
        # Match operation, shape/devices, memory config
        for result in results:
            result_config = result.get("config", {})

            # Check if this is a baseline
            if result_config.get("telemetry_enabled") == True:
                continue
            if result_config.get("telemetry_mode") != "none":
                continue

            # Match operation
            if config.get("operation") != result_config.get("operation"):
                continue

            # Match shape or num_devices
            if config.get("tensor_shape") and config.get("tensor_shape") != result_config.get("tensor_shape"):
                continue
            if config.get("num_devices") and config.get("num_devices") != result_config.get("num_devices"):
                continue

            # Match memory config if applicable
            if config.get("memory_config") and config.get("memory_config") != result_config.get("memory_config"):
                continue

            return result

        return None

    def analyze_frequency_sensitivity(self) -> Dict[str, Any]:
        """Analyze sensitivity to polling frequency."""
        print("\nAnalyzing frequency sensitivity...")

        analysis = {"single_device": [], "multi_device": []}

        # Single-device frequency analysis
        if self.results.get("single_device"):
            sd_data = self.results["single_device"]
            if "analysis" in sd_data and "statistical_comparisons" in sd_data["analysis"]:
                comparisons = sd_data["analysis"]["statistical_comparisons"]

                # Group by operation/shape/memory
                grouped = {}
                for comp in comparisons:
                    key = (comp["operation"], tuple(comp["shape"]), comp["memory_config"])
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(comp)

                # Analyze each group
                for (op, shape, mem), comps in grouped.items():
                    if len(comps) >= 3:
                        freqs = [parse_frequency_to_hz(c["polling_interval"]) for c in comps]
                        impacts = [c["impact_percent"] for c in comps]

                        monotonicity = analyze_monotonicity(freqs, impacts)

                        analysis["single_device"].append(
                            {
                                "operation": op,
                                "shape": shape,
                                "memory_config": mem,
                                "n_frequencies": len(comps),
                                "monotonicity": monotonicity,
                                "max_impact": max(abs(i) for i in impacts),
                                "impact_range": max(impacts) - min(impacts),
                            }
                        )

                print(f"  Single-device: analyzed {len(analysis['single_device'])} operation groups")
                monotonic = sum(1 for a in analysis["single_device"] if a["monotonicity"].get("monotonic"))
                print(f"    Monotonic relationships: {monotonic}/{len(analysis['single_device'])}")

        # Multi-device frequency analysis
        if self.results.get("multi_device"):
            md_data = self.results["multi_device"]
            if "analysis" in md_data and "statistical_comparisons" in md_data["analysis"]:
                comparisons = md_data["analysis"]["statistical_comparisons"]

                # Group by operation/devices/mode
                grouped = {}
                for comp in comparisons:
                    key = (comp["operation"], comp["num_devices"], comp["telemetry_mode"])
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(comp)

                # Analyze each group
                for (op, devices, mode), comps in grouped.items():
                    if len(comps) >= 3:
                        freqs = [parse_frequency_to_hz(c["polling_interval"]) for c in comps]
                        impacts = [c["impact_percent"] for c in comps]

                        monotonicity = analyze_monotonicity(freqs, impacts)

                        analysis["multi_device"].append(
                            {
                                "operation": op,
                                "num_devices": devices,
                                "telemetry_mode": mode,
                                "n_frequencies": len(comps),
                                "monotonicity": monotonicity,
                                "max_impact": max(abs(i) for i in impacts),
                                "impact_range": max(impacts) - min(impacts),
                            }
                        )

                print(f"  Multi-device: analyzed {len(analysis['multi_device'])} operation groups")
                monotonic = sum(1 for a in analysis["multi_device"] if a["monotonicity"].get("monotonic"))
                print(f"    Monotonic relationships: {monotonic}/{len(analysis['multi_device'])}")

        return analysis

    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate production deployment recommendations."""
        print("\nGenerating recommendations...")

        recommendations = {"recommended_config": {}, "safe_polling_frequencies": [], "warnings": [], "notes": []}

        # Check core hypothesis
        core_validated = False
        if self.results.get("core_hypothesis"):
            ch_data = self.results["core_hypothesis"]
            if "analysis" in ch_data:
                core_validated = ch_data["analysis"].get("hypothesis_validated", False)

        if core_validated:
            recommendations["recommended_config"]["multi_chip"] = {
                "flag": "--mmio-only",
                "reason": "Prevents ERISC contention on multi-chip workloads",
            }
            print("  ✓ Recommend --mmio-only for multi-chip")
        else:
            recommendations["warnings"].append("Core hypothesis not validated. Review multi-chip telemetry carefully.")

        # Determine safe polling frequencies based on impact distribution
        if "impact_distribution" in self.analysis:
            sd_dist = self.analysis["impact_distribution"].get("single_device", {})

            # Find frequencies with <2% average impact
            if self.results.get("single_device"):
                sd_data = self.results["single_device"]
                if "analysis" in sd_data and "statistical_comparisons" in sd_data["analysis"]:
                    comparisons = sd_data["analysis"]["statistical_comparisons"]

                    # Group by frequency
                    freq_impacts = {}
                    for comp in comparisons:
                        freq = comp["polling_interval"]
                        if freq not in freq_impacts:
                            freq_impacts[freq] = []
                        freq_impacts[freq].append(abs(comp["impact_percent"]))

                    # Find frequencies with mean impact <2%
                    for freq, impacts in sorted(freq_impacts.items(), key=lambda x: parse_frequency_to_hz(x[0])):
                        avg_impact = mean(impacts)
                        if avg_impact < 2.0:
                            recommendations["safe_polling_frequencies"].append(
                                {
                                    "frequency": freq,
                                    "average_impact": round(avg_impact, 2),
                                    "max_impact": round(max(impacts), 2),
                                }
                            )

            if recommendations["safe_polling_frequencies"]:
                print(f"  ✓ Found {len(recommendations['safe_polling_frequencies'])} safe polling frequencies")
                # Recommend the highest safe frequency
                safe_freqs = recommendations["safe_polling_frequencies"]
                best_freq = max(safe_freqs, key=lambda x: parse_frequency_to_hz(x["frequency"]))
                recommendations["recommended_config"]["polling_interval"] = best_freq["frequency"]
                print(f"  ✓ Recommended polling interval: {best_freq['frequency']}")
            else:
                recommendations["notes"].append(
                    "No polling frequency showed <2% average impact. Use longest interval that meets monitoring needs."
                )
                recommendations["recommended_config"]["polling_interval"] = "1s"

        # Check sustained workload results
        if self.results.get("sustained_workload"):
            sw_data = self.results["sustained_workload"]
            if "analysis" in sw_data:
                sw_analysis = sw_data["analysis"]
                summary = sw_analysis.get("summary", {})

                if summary.get("drift_threshold_exceeded"):
                    recommendations["warnings"].append(
                        f"Performance drift >{summary.get('max_additional_drift', 0):.1f}% detected in sustained workload. "
                        "Monitor for accumulation effects in long-running deployments."
                    )
                else:
                    recommendations["notes"].append("No significant drift detected in sustained workload tests.")
                    print("  ✓ No drift in sustained workloads")

        return recommendations

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        print("\nGenerating summary report...")

        report = f"""# Telemetry Benchmark Analysis Summary

**Phase:** {self.phase.upper()}

---

## Impact Distribution

"""

        # Impact distribution
        if "impact_distribution" in self.analysis:
            dist = self.analysis["impact_distribution"]

            if "single_device" in dist and dist["single_device"]:
                sd = dist["single_device"]
                report += f"""### Single-Device Workloads

- **Tests analyzed:** {sd['n_tests']}
- **Mean impact:** {sd['mean_impact']:+.2f}%
- **Median impact:** {sd['median_impact']:+.2f}%
- **Range:** {sd['min_impact']:+.2f}% to {sd['max_impact']:+.2f}%
- **Tests with >5% impact:** {sd['impacts_over_5pct']} ({100*sd['impacts_over_5pct']/sd['n_tests']:.1f}%)
- **Tests with >10% impact:** {sd['impacts_over_10pct']} ({100*sd['impacts_over_10pct']/sd['n_tests']:.1f}%)

"""

            if "multi_device" in dist:
                md = dist["multi_device"]

                if "mmio_only" in md:
                    mmio = md["mmio_only"]
                    report += f"""### Multi-Device Workloads (MMIO-only)

- **Tests analyzed:** {mmio['n_tests']}
- **Mean impact:** {mmio['mean_impact']:+.2f}%
- **Median impact:** {mmio['median_impact']:+.2f}%
- **Range:** {mmio['min_impact']:+.2f}% to {mmio['max_impact']:+.2f}%
- **Tests with >5% impact:** {mmio['impacts_over_5pct']} ({100*mmio['impacts_over_5pct']/mmio['n_tests']:.1f}%)

"""

                if "full" in md:
                    full = md["full"]
                    report += f"""### Multi-Device Workloads (Full mode)

- **Tests analyzed:** {full['n_tests']}
- **Mean impact:** {full['mean_impact']:+.2f}%
- **Median impact:** {full['median_impact']:+.2f}%
- **Range:** {full['min_impact']:+.2f}% to {full['max_impact']:+.2f}%
- **Tests with >5% impact:** {full['impacts_over_5pct']} ({100*full['impacts_over_5pct']/full['n_tests']:.1f}%)

"""

        # Frequency sensitivity
        report += """---

## Frequency Sensitivity

"""

        if "frequency_sensitivity" in self.analysis:
            freq_sens = self.analysis["frequency_sensitivity"]

            if freq_sens.get("single_device"):
                monotonic = sum(1 for a in freq_sens["single_device"] if a["monotonicity"].get("monotonic"))
                total = len(freq_sens["single_device"])
                report += f"""### Single-Device

- **Operation groups analyzed:** {total}
- **Monotonic relationships detected:** {monotonic} ({100*monotonic/total:.1f}%)

"""
                if monotonic > 0:
                    report += "**Monotonic operations:**\n"
                    for analysis in freq_sens["single_device"]:
                        if analysis["monotonicity"].get("monotonic"):
                            report += f"- {analysis['operation']} ({analysis['shape'][2]}x{analysis['shape'][3]}, {analysis['memory_config']}): "
                            report += f"tau={analysis['monotonicity']['tau']:.3f}, p={analysis['monotonicity']['p_value']:.4f}\n"
                    report += "\n"

            if freq_sens.get("multi_device"):
                monotonic = sum(1 for a in freq_sens["multi_device"] if a["monotonicity"].get("monotonic"))
                total = len(freq_sens["multi_device"])
                report += f"""### Multi-Device

- **Operation groups analyzed:** {total}
- **Monotonic relationships detected:** {monotonic} ({100*monotonic/total:.1f}%)

"""
                if monotonic > 0:
                    report += "**Monotonic operations:**\n"
                    for analysis in freq_sens["multi_device"]:
                        if analysis["monotonicity"].get("monotonic"):
                            report += f"- {analysis['operation']} ({analysis['num_devices']} devices, {analysis['telemetry_mode']}): "
                            report += f"tau={analysis['monotonicity']['tau']:.3f}, p={analysis['monotonicity']['p_value']:.4f}\n"
                    report += "\n"

        # Recommendations
        report += """---

## Production Recommendations

"""

        if "recommendations" in self.analysis:
            recs = self.analysis["recommendations"]

            if recs.get("recommended_config"):
                report += "### Recommended Configuration\n\n```bash\n"
                config = recs["recommended_config"]

                cmd = "./build/tools/tt-telemetry/tt-telemetry"
                if config.get("multi_chip", {}).get("flag"):
                    cmd += f" {config['multi_chip']['flag']}"
                if config.get("polling_interval"):
                    cmd += f" --logging-interval {config['polling_interval']}"
                cmd += " --port 7070"

                report += f"{cmd}\n```\n\n"

                if config.get("multi_chip", {}).get("reason"):
                    report += f"**Rationale:** {config['multi_chip']['reason']}\n\n"

            if recs.get("safe_polling_frequencies"):
                report += "### Safe Polling Frequencies\n\n"
                report += "Frequencies with <2% average impact:\n\n"
                for freq_info in recs["safe_polling_frequencies"]:
                    report += f"- **{freq_info['frequency']}**: avg impact {freq_info['average_impact']}%, "
                    report += f"max impact {freq_info['max_impact']}%\n"
                report += "\n"

            if recs.get("warnings"):
                report += "### ⚠️ Warnings\n\n"
                for warning in recs["warnings"]:
                    report += f"- {warning}\n"
                report += "\n"

            if recs.get("notes"):
                report += "### Notes\n\n"
                for note in recs["notes"]:
                    report += f"- {note}\n"
                report += "\n"

        report += """---

*Analysis complete. Review detailed results files for more information.*
"""

        return report

    def run(self) -> int:
        """Run complete analysis."""
        print("=" * 80)
        print(f"BENCHMARK RESULTS ANALYSIS - {self.phase.upper()} PHASE")
        print("=" * 80)

        # Load results
        self.load_results()

        if not any(self.results.values()):
            print("\nERROR: No results loaded. Run benchmarks first.")
            return 1

        # Run analyses
        try:
            self.analysis["impact_distribution"] = self.analyze_impact_distribution()
            self.analysis["frequency_sensitivity"] = self.analyze_frequency_sensitivity()
            self.analysis["recommendations"] = self.generate_recommendations()
        except Exception as e:
            print(f"\nERROR during analysis: {e}")
            import traceback

            traceback.print_exc()
            return 1

        # Generate report
        try:
            report = self.generate_summary_report()
            report_path = self.output_dir / f"telemetry_analysis_summary_{self.phase}.md"
            with open(report_path, "w") as f:
                f.write(report)
            print(f"\nSummary report saved to: {report_path}")
        except Exception as e:
            print(f"\nERROR generating report: {e}")
            import traceback

            traceback.print_exc()
            return 1

        # Save analysis results
        try:
            analysis_path = self.output_dir / f"telemetry_analysis_{self.phase}.json"
            save_results_json(self.analysis, str(analysis_path))
            print(f"Analysis data saved to: {analysis_path}")
        except Exception as e:
            print(f"\nERROR saving analysis: {e}")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze telemetry benchmark results", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--phase", choices=["reduced", "full"], default="reduced", help="Benchmark phase to analyze")

    parser.add_argument("--output", default="/tmp", help="Output directory for analysis results")

    args = parser.parse_args()

    analyzer = BenchmarkAnalyzer(phase=args.phase, output_dir=args.output)
    return analyzer.run()


if __name__ == "__main__":
    sys.exit(main())
