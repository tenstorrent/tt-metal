#!/usr/bin/env python3
import os
import glob
from datetime import datetime


def analyze_test_results():
    results_dir = "ssinghal/test_results"

    print("=" * 50)
    print("OPERATOR TESTS SUMMARY")
    print("=" * 50)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Find all result files
    operator_results = glob.glob(f"{results_dir}/*_results.txt")

    total_passed = 0
    total_failed = 0
    failed_tests = []
    passed_tests = []

    print("OPERATOR TEST RESULTS:")
    print("-" * 30)

    for result_file in sorted(operator_results):
        test_name = os.path.basename(result_file).replace("_results.txt", "")
        with open(result_file, "r") as f:
            content = f.read()
            if "FAILED" in content or "ERROR" in content:
                print(f"  ✗ {test_name}")
                total_failed += 1
                failed_tests.append(test_name)
            else:
                print(f"  ✓ {test_name}")
                total_passed += 1
                passed_tests.append(test_name)

    print()
    print("=" * 50)
    print("FINAL SUMMARY:")
    print("=" * 50)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    if total_passed + total_failed > 0:
        success_rate = (total_passed / (total_passed + total_failed)) * 100
        print(f"Success Rate: {success_rate:.1f}%")

    if failed_tests:
        print(f"\nFailed Tests ({len(failed_tests)}):")
        for failed_test in failed_tests:
            print(f"  ✗ {failed_test}")

    if passed_tests:
        print(f"\nPassed Tests ({len(passed_tests)}):")
        for passed_test in passed_tests:
            print(f"  ✓ {passed_test}")

    print(f"\nDetailed results available in ssinghal/test_results/")


if __name__ == "__main__":
    analyze_test_results()
