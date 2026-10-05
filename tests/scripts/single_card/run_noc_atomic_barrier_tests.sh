#!/bin/bash

set -euo pipefail

test_file="tests/ttnn/unit_tests/operations/test_noc_atomic_barriers_single_device.py"
mapfile -t test_cases < <(pytest --collect-only -q "${test_file}" | grep "^${test_file}::")

if [[ ${#test_cases[@]} -eq 0 ]]; then
    echo "No NoC atomic barrier tests were collected" >&2
    exit 1
fi

for test_case in "${test_cases[@]}"; do
    TT_METAL_NOC_DEBUG_DUMP=1 pytest -xv "${test_case}"
done
