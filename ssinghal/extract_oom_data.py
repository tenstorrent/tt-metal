import re
import csv
import json


def extract_oom_failures_from_results():
    """Extract OOM failures from test result files"""

    # List of operators we tested
    tested_operators = ["add", "silu", "relu", "sigmoid", "view", "permute"]

    all_oom_failures = []

    for operator in tested_operators:
        result_file = f"ssinghal/test_results/{operator}_results.txt"

        try:
            with open(result_file, "r") as f:
                content = f.read()

            print(f"Processing {operator}...")

            # Find all SKIPPED lines with OOM information
            skipped_pattern = r"SKIPPED \[1\] [^:]+:\d+: OOM: (\[[^\]]+\]) - .*?allocate (\d+) B.*?store (\d+) B"

            matches = re.findall(skipped_pattern, content, re.DOTALL)

            for match in matches:
                input_shape = match[0]
                total_memory = int(match[1])
                per_bank_memory = int(match[2])

                oom_failure = {
                    "operator": operator,
                    "input_shape": input_shape,
                    "total_memory_B": total_memory,
                    "total_memory_MB": round(total_memory / (1024 * 1024), 2),
                    "per_bank_memory_B": per_bank_memory,
                    "per_bank_memory_KB": round(per_bank_memory / 1024, 2),
                    "num_banks": 64,  # Always 64 based on the pattern
                }

                all_oom_failures.append(oom_failure)
                print(f"  Found OOM: {input_shape} - {oom_failure['total_memory_MB']} MB")

        except FileNotFoundError:
            print(f"  No results file found for {operator}")
        except Exception as e:
            print(f"  Error processing {operator}: {e}")

    return all_oom_failures


def create_comprehensive_oom_report(oom_failures):
    """Create comprehensive OOM analysis report"""

    # Group by operator
    by_operator = {}
    for failure in oom_failures:
        op = failure["operator"]
        if op not in by_operator:
            by_operator[op] = []
        by_operator[op].append(failure)

    # Group by shape
    by_shape = {}
    for failure in oom_failures:
        shape = failure["input_shape"]
        if shape not in by_shape:
            by_shape[shape] = []
        by_shape[shape].append(failure)

    # Memory usage analysis
    memory_stats = {
        "min_memory_MB": min(f["total_memory_MB"] for f in oom_failures) if oom_failures else 0,
        "max_memory_MB": max(f["total_memory_MB"] for f in oom_failures) if oom_failures else 0,
        "avg_memory_MB": sum(f["total_memory_MB"] for f in oom_failures) / len(oom_failures) if oom_failures else 0,
        "total_failures": len(oom_failures),
    }

    report = {
        "summary": {
            "total_oom_failures": len(oom_failures),
            "operators_with_oom": len(by_operator),
            "unique_problematic_shapes": len(by_shape),
            "memory_statistics": memory_stats,
        },
        "failures_by_operator": by_operator,
        "failures_by_shape": by_shape,
        "all_failures": oom_failures,
    }

    return report


def main():
    print("Extracting OOM failure data from test results...")

    oom_failures = extract_oom_failures_from_results()

    print(f"\nFound {len(oom_failures)} OOM failures total")

    if oom_failures:
        # Create comprehensive report
        report = create_comprehensive_oom_report(oom_failures)

        # Save JSON report
        with open("ssinghal/comprehensive_oom_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Save CSV for easy analysis
        with open("ssinghal/oom_failures_detailed.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Operator",
                    "Input_Shape",
                    "Total_Memory_B",
                    "Total_Memory_MB",
                    "Per_Bank_Memory_B",
                    "Per_Bank_Memory_KB",
                    "Num_Banks",
                ]
            )

            for failure in oom_failures:
                writer.writerow(
                    [
                        failure["operator"],
                        failure["input_shape"],
                        failure["total_memory_B"],
                        failure["total_memory_MB"],
                        failure["per_bank_memory_B"],
                        failure["per_bank_memory_KB"],
                        failure["num_banks"],
                    ]
                )

        # Print summary
        print(f"\n=== OOM FAILURE ANALYSIS ===")
        print(f"Total OOM failures: {report['summary']['total_oom_failures']}")
        print(f"Operators affected: {report['summary']['operators_with_oom']}")
        print(f"Unique problematic shapes: {report['summary']['unique_problematic_shapes']}")

        stats = report["summary"]["memory_statistics"]
        print(f"\nMemory requirements:")
        print(f"  Min: {stats['min_memory_MB']} MB")
        print(f"  Max: {stats['max_memory_MB']} MB")
        print(f"  Avg: {stats['avg_memory_MB']:.1f} MB")

        print(f"\nBy operator:")
        for op, failures in report["failures_by_operator"].items():
            print(f"  {op}: {len(failures)} failures")

        print(f"\nMost problematic shapes (>100MB):")
        large_failures = [f for f in oom_failures if f["total_memory_MB"] > 100]
        large_failures.sort(key=lambda x: x["total_memory_MB"], reverse=True)

        for failure in large_failures[:10]:  # Top 10
            print(f"  {failure['input_shape']}: {failure['total_memory_MB']} MB ({failure['operator']})")

        print(f"\nReports saved:")
        print(f"  - ssinghal/comprehensive_oom_report.json")
        print(f"  - ssinghal/oom_failures_detailed.csv")

    else:
        print("No OOM failures found in the test results.")


if __name__ == "__main__":
    main()
