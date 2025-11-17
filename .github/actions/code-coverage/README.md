# Code Coverage Report Generator

A unified code coverage tool that combines C++, Python, and kernel code coverage into a single HTML report.

## Overview

This GitHub Action generates comprehensive code coverage reports by:
1. **C++ Coverage**: Collects coverage from LLVM profraw files using `llvm-profdata` and `llvm-cov`
2. **Python Coverage**: Converts Python coverage data (`.coverage` files) to LCOV format
3. **Kernel Coverage**: Parses `generated/watcher/kernel_names.txt` and generates synthetic coverage (100% per file)

All coverage data is merged into a single LCOV file and converted to an HTML report using `genhtml`.

## Prerequisites

### Required Tools

- **llvm-profdata** and **llvm-cov**: For C++ coverage (usually part of LLVM/Clang installation)
- **Python coverage**: `pip install coverage`
- **genhtml**: Part of the `lcov` package (`apt-get install lcov` or `brew install lcov`)

### Build Requirements

For C++ coverage to work, code must be compiled with coverage flags:
- `-fprofile-instr-generate` (for profiling)
- `-fcoverage-mapping` (for source mapping)

In tt-metal, use the `ASanCoverage` build type:
```bash
./build_metal.sh --build-type ASanCoverage --build-tests
```

## Usage

### As GitHub Action

```yaml
- name: Build with Coverage
  run: |
    ./build_metal.sh --build-type ASanCoverage --build-tests

- name: Run Tests with Coverage
  env:
    LLVM_PROFILE_FILE: coverage/%p.profraw
    TT_METAL_WATCHER_APPEND: 1
  run: |
    coverage run -m pytest tests/
    ./build_ASanCoverage/test/tt_metal/unit_tests

- name: Generate Coverage Report
  uses: ./.github/actions/code-coverage
  with:
    coverage-dir: coverage
    cpp-objects: |
      ./build_ASanCoverage/lib/libtt_metal.so
      ./build_ASanCoverage/lib/libtt_stl.so
      ./build_ASanCoverage/test/tt_metal/unit_tests
    html-output-dir: coverage/html
```

### Local Testing (Standalone)

The scripts can be run directly for local testing:

```bash
# Make sure you're in the repository root
cd /path/to/tt-metal

# Run the entrypoint script directly
./.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --kernel-names-file generated/watcher/kernel_names.txt \
  --source-dir . \
  --cpp-objects "./build_ASanCoverage/lib/libtt_metal.so ./build_ASanCoverage/lib/libtt_stl.so" \
  --html-output-dir coverage/html
```

### Input Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `coverage-dir` | Directory containing coverage data files | `coverage` |
| `kernel-names-file` | Path to kernel_names.txt (relative to repo root) | `generated/watcher/kernel_names.txt` |
| `source-dir` | Repository root directory | `.` |
| `cpp-objects` | Space/newline-separated list of C++ binaries/libraries | `` |
| `enable-cpp-coverage` | Enable C++ coverage collection | `true` |
| `enable-python-coverage` | Enable Python coverage collection | `true` |
| `enable-kernel-coverage` | Enable kernel coverage | `true` |
| `html-output-dir` | HTML report output directory | `coverage/html` |
| `llvm-profdata-path` | Path to llvm-profdata binary | `llvm-profdata` |
| `llvm-cov-path` | Path to llvm-cov binary | `llvm-cov` |

## Local Testing Guide

### Step 1: Build with Coverage

```bash
# Build with ASanCoverage build type
./build_metal.sh --build-type ASanCoverage --build-tests
```

### Step 2: Run Tests with Coverage Collection

```bash
# Create coverage directory
mkdir -p coverage

# Set environment variables
export LLVM_PROFILE_FILE="coverage/%p.profraw"
export TT_METAL_WATCHER_APPEND=1

# Run Python tests with coverage
coverage run -m pytest tests/tt_eager/python_api_testing/unit_testing/ -xvvv

# Run C++ tests (profraw files will be generated automatically)
./build_ASanCoverage/test/tt_metal/unit_tests --gtest_filter="*"
```

### Step 3: Generate Coverage Report

```bash
# Run the coverage action locally
./.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --source-dir . \
  --cpp-objects "./build_ASanCoverage/lib/libtt_metal.so ./build_ASanCoverage/lib/libtt_stl.so ./build_ASanCoverage/test/tt_metal/unit_tests" \
  --html-output-dir coverage/html
```

### Step 4: View Report

Open `coverage/html/index.html` in your browser.

## Example Workflow

Here's a complete example workflow for local testing:

```bash
#!/bin/bash
set -e

# 1. Build with coverage
echo "Building with coverage..."
./build_metal.sh --build-type ASanCoverage --build-tests

# 2. Setup environment
export LLVM_PROFILE_FILE="coverage/%p.profraw"
export TT_METAL_WATCHER_APPEND=1
mkdir -p coverage

# 3. Run tests
echo "Running tests..."
coverage run -m pytest tests/tt_eager/python_api_testing/unit_testing/ -xvvv || true
./build_ASanCoverage/test/tt_metal/unit_tests --gtest_filter="*" || true

# 4. Generate report
echo "Generating coverage report..."
./.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --source-dir . \
  --cpp-objects "./build_ASanCoverage/lib/libtt_metal.so ./build_ASanCoverage/lib/libtt_stl.so ./build_ASanCoverage/test/tt_metal/unit_tests" \
  --html-output-dir coverage/html

# 5. Open report
echo "Coverage report generated at: coverage/html/index.html"
```

## Output Files

The action generates several files in the coverage directory:

- `cpp_coverage.info`: C++ coverage in LCOV format
- `python_coverage.info`: Python coverage in LCOV format
- `kernel_coverage.info`: Kernel coverage in LCOV format
- `merged_coverage.info`: Merged coverage from all sources
- `coverage.profdata`: Merged LLVM profiling data
- `html/`: Directory containing HTML report (open `index.html`)

## Troubleshooting

### No .profraw files found

**Problem**: C++ coverage not being collected.

**Solutions**:
- Ensure `LLVM_PROFILE_FILE` is set before running tests
- Verify code was built with `ASanCoverage` build type
- Check that tests actually execute (some may be skipped)

### No .coverage file found

**Problem**: Python coverage not being collected.

**Solutions**:
- Run tests with `coverage run -m pytest` instead of just `pytest`
- Check that `.coverage` file exists in coverage directory or repo root

### Kernel names file not found

**Problem**: `generated/watcher/kernel_names.txt` doesn't exist.

**Solutions**:
- Ensure `TT_METAL_WATCHER_APPEND=1` is set before running tests
- Check that tests actually execute kernel code
- Verify the path is correct (default: `generated/watcher/kernel_names.txt`)

### genhtml errors

**Problem**: HTML report generation fails.

**Solutions**:
- Check that `lcov` package is installed
- Verify merged LCOV file is valid
- Some errors may be warnings and can be ignored (coverage report may still be generated)

### Missing C++ objects

**Problem**: C++ coverage shows no data.

**Solutions**:
- Specify C++ objects/binaries using `--cpp-objects`
- Ensure objects are built with coverage flags
- Use absolute paths or paths relative to `--source-dir`

## Architecture

The action consists of:

1. **entrypoint.sh**: Main orchestration script that:
   - Collects C++ coverage from profraw files
   - Collects Python coverage from .coverage files
   - Generates kernel coverage from kernel_names.txt
   - Merges all coverage
   - Generates HTML report

2. **generate_kernel_coverage.py**: Parses kernel_names.txt and generates synthetic LCOV entries marking all lines as executed.

3. **merge_coverage.py**: Merges multiple LCOV files, summing execution counts for duplicate lines.

## Limitations

- **Kernel coverage**: Currently marks entire files as 100% covered (no line-level granularity)
- **Performance**: ASanCoverage build type includes AddressSanitizer, which adds runtime overhead
- **Path resolution**: Requires consistent path handling between build and coverage collection

## Future Enhancements

- Support for pure coverage build type (without ASan)
- Incremental coverage tracking
- Coverage diff reports
- Coverage threshold enforcement
- Parallel processing for large codebases
