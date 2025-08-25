#!/bin/bash
#
# This script automates the process of generating a C++ code coverage report
# for a project built with Clang, using lcov and llvm-cov.
#
# Place this script in your C++ build directory and run it from there.
#

# --- Configuration ---
# Set the version of clang you are using (e.g., 17)
CLANG_VERSION=17

# Set the name for the output report directory
REPORT_DIR="coverage_report"
# --- End of Configuration ---


# Exit immediately if a command fails
set -e

cd build_Release

echo "INFO: Using Clang version ${CLANG_VERSION}"

# --- Step 1: Create the llvm-cov wrapper script ---
# lcov doesn't know how to call llvm-cov correctly, so we create a simple
# wrapper that adds the required 'gcov' subcommand.

WRAPPER_SCRIPT_NAME="llvm-gcov.sh"
LLVM_COV_PATH=$(which "llvm-cov-${CLANG_VERSION}")

if [ -z "$LLVM_COV_PATH" ]; then
    echo "ERROR: llvm-cov-${CLANG_VERSION} not found in your PATH."
    echo "Please install clang-${CLANG_VERSION} and its tools or adjust the CLANG_VERSION variable."
    exit 1
fi

echo "INFO: Creating wrapper script for lcov..."
cat <<EOF > "${WRAPPER_SCRIPT_NAME}"
#!/bin/bash
# This script calls llvm-cov with the required 'gcov' subcommand.
# "\$@" passes along all arguments lcov gives it.
exec "${LLVM_COV_PATH}" gcov "\$@"
EOF

# Make the wrapper script executable
chmod +x "${WRAPPER_SCRIPT_NAME}"
echo "INFO: Wrapper script created successfully."


# --- Step 2: Run lcov to capture coverage data ---
echo "INFO: Capturing coverage data with lcov..."
lcov \
    --gcov-tool "$(pwd)/${WRAPPER_SCRIPT_NAME}" \
    --capture \
    --directory "." \
    --output-file coverage.info

echo "INFO: Coverage data saved to coverage.info."


# --- Step 3: Generate the HTML report ---
echo "INFO: Generating HTML report..."
genhtml coverage.info --output-directory "${REPORT_DIR}"


# --- Done ---
echo ""
echo "✅ Success! Coverage report generated."
echo "   You can now open the report by running:"
echo "   xdg-open ${REPORT_DIR}/index.html"
