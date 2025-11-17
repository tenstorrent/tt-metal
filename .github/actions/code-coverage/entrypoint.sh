#!/bin/bash

set -eo pipefail

# Default values
COVERAGE_DIR="coverage"
KERNEL_NAMES_FILE="generated/watcher/kernel_names.txt"
SOURCE_DIR="."
CPP_OBJECTS=""
ENABLE_CPP_COVERAGE="true"
ENABLE_PYTHON_COVERAGE="true"
ENABLE_KERNEL_COVERAGE="true"
HTML_OUTPUT_DIR="coverage/html"
LLVM_PROFDATA_PATH="llvm-profdata"
LLVM_COV_PATH="llvm-cov"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage-dir)
            COVERAGE_DIR="$2"
            shift 2
            ;;
        --kernel-names-file)
            KERNEL_NAMES_FILE="$2"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --cpp-objects)
            CPP_OBJECTS="$2"
            shift 2
            ;;
        --enable-cpp-coverage)
            ENABLE_CPP_COVERAGE="$2"
            shift 2
            ;;
        --enable-python-coverage)
            ENABLE_PYTHON_COVERAGE="$2"
            shift 2
            ;;
        --enable-kernel-coverage)
            ENABLE_KERNEL_COVERAGE="$2"
            shift 2
            ;;
        --html-output-dir)
            HTML_OUTPUT_DIR="$2"
            shift 2
            ;;
        --llvm-profdata-path)
            LLVM_PROFDATA_PATH="$2"
            shift 2
            ;;
        --llvm-cov-path)
            LLVM_COV_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --coverage-dir DIR              Directory containing coverage data (default: coverage)"
            echo "  --kernel-names-file FILE        Path to kernel_names.txt (default: generated/watcher/kernel_names.txt)"
            echo "  --source-dir DIR                Repository root directory (default: .)"
            echo "  --cpp-objects OBJECTS           Space/newline-separated C++ objects/binaries"
            echo "  --enable-cpp-coverage BOOL       Enable C++ coverage (default: true)"
            echo "  --enable-python-coverage BOOL    Enable Python coverage (default: true)"
            echo "  --enable-kernel-coverage BOOL    Enable kernel coverage (default: true)"
            echo "  --html-output-dir DIR            HTML report output directory (default: coverage/html)"
            echo "  --llvm-profdata-path PATH       Path to llvm-profdata (default: llvm-profdata)"
            echo "  --llvm-cov-path PATH            Path to llvm-cov (default: llvm-cov)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Convert relative paths to absolute
SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
if [[ "$COVERAGE_DIR" == /* ]]; then
    # Already absolute
    COVERAGE_DIR="$COVERAGE_DIR"
else
    # Relative to current directory
    COVERAGE_DIR="$(cd "$(dirname "$COVERAGE_DIR")" 2>/dev/null && pwd)/$(basename "$COVERAGE_DIR")" || COVERAGE_DIR="$(pwd)/$COVERAGE_DIR"
fi
if [[ "$KERNEL_NAMES_FILE" == /* ]]; then
    # Already absolute
    KERNEL_NAMES_FILE="$KERNEL_NAMES_FILE"
else
    # Relative to source dir
    KERNEL_NAMES_FILE="$SOURCE_DIR/$KERNEL_NAMES_FILE"
fi

echo "=========================================="
echo "Code Coverage Report Generator"
echo "=========================================="
echo "Coverage directory: $COVERAGE_DIR"
echo "Source directory: $SOURCE_DIR"
echo "Kernel names file: $KERNEL_NAMES_FILE"
echo "HTML output: $HTML_OUTPUT_DIR"
echo ""

# Find LLVM tool with version suffix
find_llvm_tool() {
    local tool_base="$1"
    local default_path="$2"

    # First try the provided path
    if command -v "$default_path" &> /dev/null; then
        echo "$default_path"
        return 0
    fi

    # Try versioned names (llvm-profdata-17, llvm-profdata-18, etc.)
    for version in 17 18 19 16 15; do
        local versioned_tool="${tool_base}-${version}"
        if command -v "$versioned_tool" &> /dev/null; then
            echo "$versioned_tool"
            return 0
        fi
    done

    # Try in common LLVM installation directories
    for version in 17 18 19 16 15; do
        local alt_path="/usr/lib/llvm-${version}/bin/${tool_base}"
        if [ -f "$alt_path" ] && [ -x "$alt_path" ]; then
            echo "$alt_path"
            return 0
        fi
    done

    return 1
}

echo "Checking required tools..."
if [ "$ENABLE_CPP_COVERAGE" = "true" ]; then
    # Find llvm-profdata
    FOUND_PROFDATA=$(find_llvm_tool "llvm-profdata" "$LLVM_PROFDATA_PATH")
    if [ -n "$FOUND_PROFDATA" ]; then
        LLVM_PROFDATA_PATH="$FOUND_PROFDATA"
        echo "✓ Found llvm-profdata: $LLVM_PROFDATA_PATH"
    else
        echo "ERROR: Required tool 'llvm-profdata' not found in PATH"
        echo "  Searched for: llvm-profdata, llvm-profdata-17, llvm-profdata-18, etc."
        echo "  Try installing: apt-get install llvm-17-tools (or llvm-18-tools)"
        exit 1
    fi

    # Find llvm-cov
    FOUND_COV=$(find_llvm_tool "llvm-cov" "$LLVM_COV_PATH")
    if [ -n "$FOUND_COV" ]; then
        LLVM_COV_PATH="$FOUND_COV"
        echo "✓ Found llvm-cov: $LLVM_COV_PATH"
    else
        echo "ERROR: Required tool 'llvm-cov' not found in PATH"
        echo "  Searched for: llvm-cov, llvm-cov-17, llvm-cov-18, etc."
        echo "  Try installing: apt-get install llvm-17-tools (or llvm-18-tools)"
        exit 1
    fi
fi

if [ "$ENABLE_PYTHON_COVERAGE" = "true" ]; then
    if ! python3 -c "import coverage" 2>/dev/null; then
        echo "ERROR: Python 'coverage' module not found. Install with: pip install coverage"
        exit 1
    fi
fi

if ! command -v genhtml &> /dev/null; then
    echo "ERROR: 'genhtml' not found. Install lcov package (e.g., apt-get install lcov)"
    exit 1
fi

echo "✓ All required tools found"
echo ""

# Create output directories
mkdir -p "$COVERAGE_DIR"
mkdir -p "$HTML_OUTPUT_DIR"

# Step 1: Collect C++ Coverage
CPP_COVERAGE_FILE="$COVERAGE_DIR/cpp_coverage.info"
if [ "$ENABLE_CPP_COVERAGE" = "true" ]; then
    echo "Step 1: Collecting C++ coverage..."

    # Find all profraw files
    PROFRAW_FILES=("$COVERAGE_DIR"/*.profraw)
    if [ ! -e "${PROFRAW_FILES[0]}" ]; then
        echo "  ⚠ No .profraw files found in $COVERAGE_DIR"
        echo "  Creating empty C++ coverage file..."
        touch "$CPP_COVERAGE_FILE"
    else
        echo "  Found ${#PROFRAW_FILES[@]} .profraw file(s)"

        # Merge profraw files into a temporary .profdata file
        # Note: .profdata files are intermediate - we'll clean them up after generating .info files
        PROFDATA_FILE="$COVERAGE_DIR/coverage.profdata"
        echo "  Merging profraw files..."
        "$LLVM_PROFDATA_PATH" merge -sparse "${PROFRAW_FILES[@]}" -o "$PROFDATA_FILE"

        # Export to LCOV
        if [ -z "$CPP_OBJECTS" ]; then
            echo "  ⚠ No C++ objects specified. Skipping LCOV export."
            echo "  Use --cpp-objects to specify binaries/libraries for coverage."
            touch "$CPP_COVERAGE_FILE"
        else
            echo "  Exporting to LCOV format..."
            # Split CPP_OBJECTS by whitespace or newlines
            IFS=$'\n' read -d '' -r -a OBJECT_ARRAY <<< "$(echo "$CPP_OBJECTS" | tr ' ' '\n' | grep -v '^$')" || true

            # Convert relative paths to absolute
            ABS_OBJECTS=()
            for obj in "${OBJECT_ARRAY[@]}"; do
                if [[ "$obj" == /* ]]; then
                    ABS_OBJECTS+=("$obj")
                else
                    ABS_OBJECTS+=("$SOURCE_DIR/$obj")
                fi
            done

            # Export coverage for each object
            TEMP_FILES=()
            for obj in "${ABS_OBJECTS[@]}"; do
                if [ ! -f "$obj" ] && [ ! -d "$obj" ]; then
                    echo "  ⚠ Warning: Object not found: $obj"
                    continue
                fi
                TEMP_FILE="$COVERAGE_DIR/cpp_temp_$(basename "$obj").info"
                TEMP_FILES+=("$TEMP_FILE")
                echo "  Processing: $obj"
                "$LLVM_COV_PATH" export -format=lcov \
                    -instr-profile="$PROFDATA_FILE" \
                    "$obj" > "$TEMP_FILE" 2>/dev/null || {
                    echo "  ⚠ Warning: Failed to export coverage for $obj"
                    touch "$TEMP_FILE"
                }
            done

            # Merge all C++ coverage files
            if [ ${#TEMP_FILES[@]} -gt 0 ]; then
                python3 "$SCRIPT_DIR/merge_coverage.py" "${TEMP_FILES[@]}" > "$CPP_COVERAGE_FILE"
                rm -f "${TEMP_FILES[@]}"
            else
                touch "$CPP_COVERAGE_FILE"
            fi
        fi

        # Clean up intermediate .profdata file after generating .info files
        # (keep .profraw files as they might be useful for debugging)
        if [ -f "$PROFDATA_FILE" ]; then
            rm -f "$PROFDATA_FILE"
        fi
    fi
    echo "  ✓ C++ coverage collected: $CPP_COVERAGE_FILE"
else
    echo "Step 1: Skipping C++ coverage (disabled)"
    touch "$CPP_COVERAGE_FILE"
fi
echo ""

# Step 2: Collect Python Coverage
PYTHON_COVERAGE_FILE="$COVERAGE_DIR/python_coverage.info"
if [ "$ENABLE_PYTHON_COVERAGE" = "true" ]; then
    echo "Step 2: Collecting Python coverage..."

    # Check for .coverage file
    if [ -f "$COVERAGE_DIR/.coverage" ]; then
        COVERAGE_FILE="$COVERAGE_DIR/.coverage"
    elif [ -f ".coverage" ]; then
        COVERAGE_FILE=".coverage"
    else
        echo "  ⚠ No .coverage file found. Skipping Python coverage."
        touch "$PYTHON_COVERAGE_FILE"
    fi

    if [ -n "$COVERAGE_FILE" ] && [ -f "$COVERAGE_FILE" ]; then
        echo "  Found coverage file: $COVERAGE_FILE"
        echo "  Converting to LCOV format..."
        cd "$SOURCE_DIR"
        coverage lcov -o "$PYTHON_COVERAGE_FILE" --data-file="$COVERAGE_FILE" || {
            echo "  ⚠ Warning: Failed to convert Python coverage to LCOV"
            touch "$PYTHON_COVERAGE_FILE"
        }
        cd - > /dev/null
    fi
    echo "  ✓ Python coverage collected: $PYTHON_COVERAGE_FILE"
else
    echo "Step 2: Skipping Python coverage (disabled)"
    touch "$PYTHON_COVERAGE_FILE"
fi
echo ""

# Step 3: Generate Kernel Coverage
KERNEL_COVERAGE_FILE="$COVERAGE_DIR/kernel_coverage.info"
if [ "$ENABLE_KERNEL_COVERAGE" = "true" ]; then
    echo "Step 3: Generating kernel coverage..."

    if [ ! -f "$KERNEL_NAMES_FILE" ]; then
        echo "  ⚠ Kernel names file not found: $KERNEL_NAMES_FILE"
        echo "  Skipping kernel coverage."
        touch "$KERNEL_COVERAGE_FILE"
    else
        echo "  Parsing kernel names file: $KERNEL_NAMES_FILE"
        python3 "$SCRIPT_DIR/generate_kernel_coverage.py" \
            --kernel-names-file "$KERNEL_NAMES_FILE" \
            --source-dir "$SOURCE_DIR" \
            --output "$KERNEL_COVERAGE_FILE" || {
            echo "  ⚠ Warning: Failed to generate kernel coverage"
            touch "$KERNEL_COVERAGE_FILE"
        }
        echo "  ✓ Kernel coverage generated: $KERNEL_COVERAGE_FILE"
    fi
else
    echo "Step 3: Skipping kernel coverage (disabled)"
    touch "$KERNEL_COVERAGE_FILE"
fi
echo ""

# Step 4: Generate Zero-Coverage for All Files
ZERO_COVERAGE_FILE="$COVERAGE_DIR/zero_coverage.info"
echo "Step 4: Generating zero-coverage entries for all source files..."
echo "  This ensures all files appear in the report, even with 0% coverage..."
python3 "$SCRIPT_DIR/generate_zero_coverage.py" \
    --source-dir "$SOURCE_DIR" \
    --output "$ZERO_COVERAGE_FILE" || {
    echo "  ⚠ Warning: Failed to generate zero-coverage (continuing anyway)"
    touch "$ZERO_COVERAGE_FILE"
}
echo "  ✓ Zero-coverage generated: $ZERO_COVERAGE_FILE"
echo ""

# Step 5: Merge All Coverage
echo "Step 5: Merging all coverage files..."
MERGED_COVERAGE_FILE="$COVERAGE_DIR/merged_coverage.info"

COVERAGE_FILES=()
[ -f "$CPP_COVERAGE_FILE" ] && COVERAGE_FILES+=("$CPP_COVERAGE_FILE")
[ -f "$PYTHON_COVERAGE_FILE" ] && COVERAGE_FILES+=("$PYTHON_COVERAGE_FILE")
[ -f "$KERNEL_COVERAGE_FILE" ] && COVERAGE_FILES+=("$KERNEL_COVERAGE_FILE")
[ -f "$ZERO_COVERAGE_FILE" ] && COVERAGE_FILES+=("$ZERO_COVERAGE_FILE")

if [ ${#COVERAGE_FILES[@]} -eq 0 ]; then
    echo "  ⚠ No coverage files to merge!"
    exit 1
fi

echo "  Merging ${#COVERAGE_FILES[@]} coverage file(s)..."
python3 "$SCRIPT_DIR/merge_coverage.py" "${COVERAGE_FILES[@]}" > "$MERGED_COVERAGE_FILE"
echo "  ✓ Merged coverage: $MERGED_COVERAGE_FILE"
echo ""

# Step 6: Generate HTML Report
echo "Step 6: Generating HTML report..."
echo "  Output directory: $HTML_OUTPUT_DIR"
genhtml -o "$HTML_OUTPUT_DIR" "$MERGED_COVERAGE_FILE" --ignore-errors source || {
    echo "  ⚠ Warning: genhtml encountered errors but continuing..."
}

if [ -f "$HTML_OUTPUT_DIR/index.html" ]; then
    echo "  ✓ HTML report generated: $HTML_OUTPUT_DIR/index.html"
else
    echo "  ⚠ Warning: HTML report may not have been generated correctly"
fi
echo ""

echo "=========================================="
echo "Coverage report generation complete!"
echo "=========================================="
echo "HTML Report: $HTML_OUTPUT_DIR/index.html"
echo "Merged LCOV: $MERGED_COVERAGE_FILE"
echo ""
echo "Open the HTML report in your browser to view coverage details."
