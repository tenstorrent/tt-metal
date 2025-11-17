# Quick Start Guide

## Local Testing (5 minutes)

### 1. Build with Coverage

```bash
./build_metal.sh --build-type ASanCoverage --build-tests
```

### 2. Setup Coverage Environment

```bash
# This installs missing dependencies and sets up environment variables
source .github/actions/code-coverage/setup_coverage_env.sh
```

### 3. Run Tests (collect coverage data)

**For C++ Binaries:**
```bash
# Run your C++ test binary
./build_ASanCoverage/test/tt_metal/test_add_two_ints
# This will generate .profraw files in coverage/
```

**For Python Tests:**
```bash
# Run Python tests with coverage
coverage run -m pytest tests/tt_eager/python_api_testing/unit_testing/ -xvvv -k "test_add" || true
```

**For Both:**
```bash
# Run C++ tests
./build_ASanCoverage/test/tt_metal/unit_tests --gtest_filter="*Add*" || true

# Run Python tests
coverage run -m pytest tests/tt_eager/python_api_testing/unit_testing/ -xvvv -k "test_add" || true
```

**Important**: `ASanCoverage` builds require AddressSanitizer runtime libraries. If you get errors like `undefined symbol: __asan_option_detect_stack_use_after_return`, make sure:
- `LD_LIBRARY_PATH` includes your build directory: `export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH}"`
- The ASan library is available (usually in `/usr/lib/x86_64-linux-gnu/` or similar)

### 4. Generate Report

**For C++ Binary Only:**
```bash
./.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --source-dir . \
  --cpp-objects "./build_ASanCoverage/test/tt_metal/test_add_two_ints ./build_ASanCoverage/lib/libtt_metal.so ./build_ASanCoverage/lib/libtt_stl.so" \
  --enable-python-coverage false \
  --html-output-dir coverage/html
```

**For Python Tests Only:**
```bash
./.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --source-dir . \
  --enable-cpp-coverage false \
  --html-output-dir coverage/html
```

**For Both C++ and Python:**
```bash
./.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --source-dir . \
  --cpp-objects "./build_ASanCoverage/lib/libtt_metal.so ./build_ASanCoverage/lib/libtt_stl.so ./build_ASanCoverage/test/tt_metal/unit_tests" \
  --html-output-dir coverage/html
```

**Or use the test script (auto-detects everything):**
```bash
./.github/actions/code-coverage/test_local.sh
```

### 5. View Report

```bash
# Open in browser
xdg-open coverage/html/index.html  # Linux
open coverage/html/index.html      # macOS
```

## What You'll See

- **C++ Coverage**: Shows which C++ lines were executed
- **Python Coverage**: Shows which Python lines were executed
- **Kernel Coverage**: Shows which kernel files were used (marked as 100% covered)

All combined into a single HTML report with file-by-file breakdown.

## Troubleshooting

**No coverage data?**
- Make sure you built with `ASanCoverage`
- Check that `LLVM_PROFILE_FILE` was set before running tests
- Verify `.profraw` files exist in `coverage/` directory

**Missing tools?**
```bash
# Install lcov
sudo apt-get install lcov  # Ubuntu/Debian
brew install lcov          # macOS

# Install Python coverage
pip install coverage
```

**Need help?**
See `README.md` for detailed documentation.
