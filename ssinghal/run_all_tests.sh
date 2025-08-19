#!/bin/bash

# Test runner for operator tests only
# Runs only the operator tests from ssinghal/tests/ folder

export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

echo "Running operator tests..."

# Activate virtual environment
source python_env/bin/activate

# Create results directory
mkdir -p ssinghal/test_results

echo "Starting test execution..."

# Count operator tests only
operator_tests=($(find ssinghal/tests -name "test_*.py" -type f 2>/dev/null | sort))
total_count=${#operator_tests[@]}

echo "Found $total_count operator tests"
echo ""

counter=1

echo "=========================================="
echo "Operator Tests (ssinghal/tests/)"
echo "=========================================="

for test_file in "${operator_tests[@]}"; do
    test_name=$(basename "$test_file" .py)
    echo "($counter/$total_count) Testing $test_name..."
    python_env/bin/python -m pytest "$test_file" -v --tb=short > "ssinghal/test_results/${test_name}_results.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✓ PASSED - Results saved to ssinghal/test_results/${test_name}_results.txt"
    else
        echo "  ✗ FAILED - Check ssinghal/test_results/${test_name}_results.txt for details"
    fi
    counter=$((counter + 1))
done

echo ""
echo "All operator tests completed!"
echo "Results are in ssinghal/test_results/"

# Generate simple summary
cat > ssinghal/test_results/generate_summary.py << 'EOF'
#!/usr/bin/env python3
import os
import glob
from datetime import datetime

def analyze_test_results():
    results_dir = "ssinghal/test_results"

    print("="*50)
    print("OPERATOR TESTS SUMMARY")
    print("="*50)
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
        with open(result_file, 'r') as f:
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
    print("="*50)
    print("FINAL SUMMARY:")
    print("="*50)
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
EOF

python3 ssinghal/test_results/generate_summary.py

echo ""
echo "Operator test execution complete!"
echo "Check ssinghal/test_results/ for detailed results"
