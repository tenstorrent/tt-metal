#!/usr/bin/env bash
# Runs the tests listed in smoke_tests.txt, failing first if any listed name matches no test.
# Extra arguments (for example --gtest_output=xml:...) are passed through to the test binary.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
list_file="${script_dir}/smoke_tests.txt"
test_binary="${TTML_TESTS_BINARY:-build/tt-train/tests/ttml_tests}"

mapfile -t tests < <(sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' "${list_file}")
if [[ ${#tests[@]} -eq 0 ]]; then
    echo "error: no tests found in ${list_file}" >&2
    exit 1
fi
filter="$(IFS=:; echo "${tests[*]}")"

registered="$("${test_binary}" --gtest_list_tests --gtest_filter="${filter}" |
    awk '/^[^ ]/ { suite = $1; next } { print suite $1 }')"
missing="$(comm -23 <(printf '%s\n' "${tests[@]}" | sort -u) <(sort -u <<< "${registered}"))"
if [[ -n "${missing}" ]]; then
    echo "error: ${list_file} lists tests that do not exist in ${test_binary}:" >&2
    echo "${missing}" >&2
    exit 1
fi

exec "${test_binary}" --gtest_filter="${filter}" "$@"
