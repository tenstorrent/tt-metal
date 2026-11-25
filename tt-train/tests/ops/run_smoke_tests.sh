#!/usr/bin/env bash
# Runs the tests listed in smoke_tests.txt, failing first if any listed name matches no test.
# Extra arguments (for example --gtest_output=xml:...) are passed through to the test binary.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
list_file="${script_dir}/smoke_tests.txt"
if [[ -n "${TTML_TESTS_BINARY:-}" ]]; then
    test_binary="${TTML_TESTS_BINARY}"
elif [[ -x "build/tt-train/tests/ttml_tests" ]]; then
    test_binary="build/tt-train/tests/ttml_tests"
else
    test_binary="/usr/bin/ttml_tests"
fi

# Read one test name per line from the list file. The sed expressions, in order:
#   s/#.*//                strip comments (from '#' to end of line)
#   s/^[[:space:]]*//      strip leading whitespace
#   s/[[:space:]]*$//      strip trailing whitespace
#   /^$/d                  drop lines left empty
mapfile -t tests < <(sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' "${list_file}")
if [[ ${#tests[@]} -eq 0 ]]; then
    echo "error: no tests found in ${list_file}" >&2
    exit 1
fi
# Join the names with ':' to form a gtest filter, e.g. "SuiteA.Test1:SuiteB.Test2".
filter="$(IFS=:; echo "${tests[*]}")"

# --gtest_list_tests prints each suite unindented with a trailing '.' ("SuiteA.") followed by
# its tests indented on the lines below ("  Test1"). awk rebuilds full names:
#   /^[^ ]/ { suite = $1; next }   check for the header, which is a line not starting with a space
#   { print suite $1 }             the other lines are tests so print them like "SuiteA." + "Test1"
# Using $1 drops the trailing "# GetParam() = ..." annotation gtest adds to parameterized tests.
registered="$("${test_binary}" --gtest_list_tests --gtest_filter="${filter}" |
    awk '/^[^ ]/ { suite = $1; next } { print suite $1 }')"
# comm -23 prints lines only in the first (sorted) input: listed names that gtest did not report.
missing="$(comm -23 <(printf '%s\n' "${tests[@]}" | sort -u) <(sort -u <<< "${registered}"))"
if [[ -n "${missing}" ]]; then
    echo "error: ${list_file} lists tests that do not exist in ${test_binary}:" >&2
    echo "${missing}" >&2
    exit 1
fi

exec "${test_binary}" --gtest_filter="${filter}" "$@"
