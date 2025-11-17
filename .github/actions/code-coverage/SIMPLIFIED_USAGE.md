# Simplified Code Coverage Usage

## Quick Start

Use the `run_coverage.sh` script to automatically run a test and generate a coverage report:

```bash
# For C++ binaries
.github/actions/code-coverage/run_coverage.sh ./build_ASanCoverage/test/tt_metal/test_add_two_ints

# For Python tests
.github/actions/code-coverage/run_coverage.sh tests/ttnn/unit_tests/operations/matmul/test_matmul.py

# For Python tests with specific test cases
.github/actions/code-coverage/run_coverage.sh "tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul"
```

That's it! The script will:
1. Detect whether it's a C++ binary or Python test
2. Set up the coverage environment
3. Run the test with coverage instrumentation
4. Generate an HTML report at `coverage/html/index.html`

## What's Different?

### All Files Included
The coverage report now includes **ALL** source files in the repository, even if they have 0% coverage. This gives you a complete picture of what's covered and what's not.

### Single Command
No more manual steps:
- ❌ Old: Build → Setup env → Run test → Generate report
- ✅ New: Just run `run_coverage.sh <test>`

## Advanced Usage

### Custom Coverage Directory
```bash
COVERAGE_DIR=my_coverage .github/actions/code-coverage/run_coverage.sh <test>
```

### Custom HTML Output
```bash
COVERAGE_HTML_DIR=my_html .github/actions/code-coverage/run_coverage.sh <test>
```

### Multiple Tests
Run multiple tests separately, then merge manually using `entrypoint.sh`:

```bash
# Run first test
.github/actions/code-coverage/run_coverage.sh test1

# Run second test (coverage will accumulate)
.github/actions/code-coverage/run_coverage.sh test2

# The final report will include coverage from both tests
```

## How It Works

1. **Zero-Coverage Generation**: Creates LCOV entries for all source files with 0% coverage
2. **Test Execution**: Runs your test with coverage instrumentation
3. **Coverage Collection**: Collects C++, Python, and kernel coverage
4. **Merging**: Merges all coverage data (including zero-coverage)
5. **HTML Generation**: Creates the final HTML report

## Troubleshooting

### "Test failed but continuing..."
The script will still generate a coverage report even if the test fails. Check the test output for details.

### Missing files in report
If a file is missing from the report, check:
- Is it excluded by patterns in `generate_zero_coverage.py`? (build/, .cpmcache/, third_party/, etc.)
- Is it a valid C++ source file? (.cpp, .hpp, .h, .cc, .cxx, .c)

### Report shows 0% for everything
Make sure:
- You built with `ASanCoverage` build type
- The test actually ran (check for .profraw files in `coverage/`)
- Environment variables were set correctly (check `setup_coverage_env.sh` output)
