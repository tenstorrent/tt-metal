#!/bin/bash

set -euo pipefail

# delete all .gcda files in the build directory (.build/default)
# start with a given directory
# for each test in the directory
# - run the test
# - generate a coverage report for the test
# - save the coverage report in the build directory
# - copy the coverage report to the coverage_reports directory
# - delete the .gcda files for the test
# generate a summary report of all the coverage reports

# ------------------------------
# Configuration & helpers
# ------------------------------

# Allow overriding toolchain version; default matches coverage_report.sh
: "${CLANG_VERSION:=17}"

# Allow overriding build directory (instrumented build). Prefer build_Release, fall back to .build/default.
if [[ -z "${BUILD_DIR:-}" ]]; then
    if [[ -d "build_Release" ]]; then
        BUILD_DIR="build_Release"
    elif [[ -d ".build/default" ]]; then
        BUILD_DIR=".build/default"
    else
        echo "ERROR: Could not determine build directory. Set BUILD_DIR env var." >&2
        exit 1
    fi
fi

# Where to look for tests. If an argument is provided and is an executable file,
# run only that test. Otherwise, treat it as a directory to enumerate tests.
TEST_TARGET="${1:-}"
TESTS=()
if [[ -n "${TEST_TARGET}" && -f "${TEST_TARGET}" && -x "${TEST_TARGET}" ]]; then
    TESTS+=("${TEST_TARGET}")
else
    TEST_DIR="${TEST_TARGET}"
    if [[ -z "${TEST_DIR}" ]]; then
        if [[ -d "${BUILD_DIR}/bin" ]]; then
            TEST_DIR="${BUILD_DIR}/bin"
        else
            TEST_DIR="${BUILD_DIR}"
        fi
    fi
fi

# Destination for final reports
TT_METAL_HOME_DEFAULT="$(cd "$(dirname "$0")" && pwd)"
: "${TT_METAL_HOME:=${TT_METAL_HOME_DEFAULT}}"
REPORTS_ROOT="${TT_METAL_HOME}/coverage_reports"

mkdir -p "${REPORTS_ROOT}"

echo "INFO: Build directory: ${BUILD_DIR}"
if [[ ${#TESTS[@]} -eq 1 ]]; then
    echo "INFO: Single test:     ${TESTS[0]}"
else
    echo "INFO: Tests directory: ${TEST_DIR}"
fi
echo "INFO: Reports root:    ${REPORTS_ROOT}"
echo "INFO: Using Clang ${CLANG_VERSION}"

# Create llvm-cov wrapper once in build dir (same approach as coverage_report.sh)
pushd "${BUILD_DIR}" >/dev/null
LLVM_COV_PATH="$(which "llvm-cov-${CLANG_VERSION}" || true)"
if [[ -z "${LLVM_COV_PATH}" ]]; then
    echo "ERROR: llvm-cov-${CLANG_VERSION} not found in PATH" >&2
    exit 1
fi
WRAPPER_SCRIPT_NAME="llvm-gcov.sh"
cat > "${WRAPPER_SCRIPT_NAME}" <<EOF
#!/bin/bash
exec "${LLVM_COV_PATH}" gcov "\$@"
EOF
chmod +x "${WRAPPER_SCRIPT_NAME}"
popd >/dev/null

# Absolute path to llvm-cov wrapper to avoid relative path issues in geninfo
WRAPPER_ABS="$(cd "${BUILD_DIR}" && pwd)/${WRAPPER_SCRIPT_NAME}"

# Helper: filter unwanted paths from an .info file
filter_info_file() {
    local in_file="$1"; shift
    local out_file="$1"; shift
    lcov --remove "${in_file}" '*/.cpmcache/*' '*/_deps/*' '*/tests/*' -o "${out_file}" >/dev/null
}

# Function: purge all .gcda (fresh coverage for each test)
purge_gcda() {
    if [[ -d "${BUILD_DIR}" ]]; then
        # Delete all .gcda files under the build directory
        find "${BUILD_DIR}" -type f -name '*.gcda' -delete || true
    fi
}

# Initial cleanup per requirements
echo "INFO: Purging existing .gcda files under ${BUILD_DIR}"
purge_gcda

if [[ ${#TESTS[@]} -eq 0 ]]; then
    # Enumerate tests: executables in TEST_DIR
    mapfile -t TESTS < <(find "${TEST_DIR}" -maxdepth 1 -type f -executable | sort)
    # If no executables, try Python tests (*.py)
    if [[ ${#TESTS[@]} -eq 0 ]]; then
        mapfile -t PY_TESTS < <(find "${TEST_DIR}" -maxdepth 1 -type f -name '*.py' | sort)
        if [[ ${#PY_TESTS[@]} -eq 0 ]]; then
            echo "ERROR: No executable tests or Python test files found in ${TEST_DIR}" >&2
            exit 1
        fi
        TESTS+=("${PY_TESTS[@]}")
    fi
fi

echo "INFO: Found ${#TESTS[@]} test(s)"

# Run each test and generate per-test coverage
PER_TEST_INFOS=()
for test_path in "${TESTS[@]}"; do
    test_name="$(basename "${test_path}")"
    echo "\n=== Running test: ${test_name} ==="

    # Fresh counters for this test
    purge_gcda

    # Run the test (non-interactive)
    if [[ "${test_path}" == *.py ]]; then
        python -m pytest "${test_path}" || { echo "ERROR: Test failed: ${test_name}" >&2; exit 1; }
    else
        "${test_path}" || { echo "ERROR: Test failed: ${test_name}" >&2; exit 1; }
    fi

    # Capture coverage for this test into build dir
    pushd "${BUILD_DIR}" >/dev/null
    per_test_info="${test_name}.info"
    lcov \
        --gcov-tool "${WRAPPER_ABS}" \
        --capture \
        --directory "." \
        --output-file "${per_test_info}"
    popd >/dev/null

    # Save HTML per-test report in build dir
    pushd "${BUILD_DIR}" >/dev/null
    per_test_html_dir="coverage_${test_name}"
    # Exclude unwanted paths before generating HTML
    per_test_info_filtered="${test_name}.filtered.info"
    filter_info_file "${per_test_info}" "${per_test_info_filtered}"
    genhtml "${per_test_info_filtered}" --output-directory "${per_test_html_dir}" >/dev/null
    popd >/dev/null

    # Copy per-test report to $TT_METAL_HOME/coverage_reports
    dest_dir="${REPORTS_ROOT}/${test_name}"
    rm -rf "${dest_dir}"
    mkdir -p "${dest_dir}"
    cp -r "${BUILD_DIR}/${per_test_html_dir}"/* "${dest_dir}/"
    cp "${BUILD_DIR}/${per_test_info_filtered}" "${dest_dir}/" || true

    PER_TEST_INFOS+=("${BUILD_DIR}/${per_test_info_filtered}")

    # Delete .gcda files for the test (isolate next run)
    purge_gcda
done

# # Generate a summary report of all the coverage reports
# echo "\n=== Generating merged summary (with baseline) ==="

# # Require BUILD_DIR for baseline so we include all instrumented files
# if [[ -z "${BUILD_DIR:-}" ]]; then
#     echo "ERROR: BUILD_DIR must be set to generate a baseline (e.g., export BUILD_DIR=/home/ubuntu/tt-metal/build_Release)" >&2
#     exit 1
# fi

# # Collect all .info files from $TT_METAL_HOME/coverage_reports (per-test reports)
# mapfile -t INFO_FILES < <(find "${REPORTS_ROOT}" -maxdepth 2 -type f -name '*.info' ! -name 'summary.info' | sort)
# if [[ ${#INFO_FILES[@]} -eq 0 ]]; then
#     echo "ERROR: No .info files found in ${REPORTS_ROOT} to merge" >&2
#     exit 1
# fi

# # Temporary workspace
# TMP_SUMMARY_DIR="$(mktemp -d)"
# summary_info="${TMP_SUMMARY_DIR}/summary.info"
# rm -f "${summary_info}"

# # Create baseline: includes all instrumented files in the build (even if unexecuted)
# baseline_raw="${TMP_SUMMARY_DIR}/baseline.raw.info"
# lcov \
#     --gcov-tool "${WRAPPER_ABS}" \
#     --capture \
#     --initial \
#     --directory "${BUILD_DIR}" \
#     --output-file "${baseline_raw}"

# # Optional: keep only tt_metal files in baseline, and drop .cpmcache
# baseline_filtered="${TMP_SUMMARY_DIR}/baseline.filtered.info"
# lcov --extract "${baseline_raw}" "${TT_METAL_HOME}/tt_metal/*" -o "${baseline_filtered}" >/dev/null
# lcov --remove  "${baseline_filtered}" '*/.cpmcache/*' '*/_deps/*' '*/tests/*' -o "${baseline_filtered}" >/dev/null

# # Merge baseline + all per-test infos
# tmp_merge="${TMP_SUMMARY_DIR}/merged.tmp.info"
# cp "${baseline_filtered}" "${tmp_merge}"
# for info in "${INFO_FILES[@]}"; do
#     lcov -a "${tmp_merge}" -a "${info}" -o "${tmp_merge}.next" >/dev/null
#     mv "${tmp_merge}.next" "${tmp_merge}"
# done
# mv "${tmp_merge}" "${summary_info}"

# # Generate HTML
# summary_html_dir="${TMP_SUMMARY_DIR}/coverage_summary"
# rm -rf "${summary_html_dir}"
# genhtml "${summary_info}" --output-directory "${summary_html_dir}" >/dev/null

# # Copy summary to $TT_METAL_HOME/coverage_reports/summary
# rm -rf "${REPORTS_ROOT}/summary"
# mkdir -p "${REPORTS_ROOT}/summary"
# cp -r "${summary_html_dir}"/* "${REPORTS_ROOT}/summary/"
# cp "${summary_info}" "${REPORTS_ROOT}/summary/" || true

# echo "\n✅ Done. Per-test reports + baseline summary available under: ${REPORTS_ROOT}"
