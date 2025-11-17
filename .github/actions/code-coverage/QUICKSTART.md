# Quick Start Guide

## Local Testing (5 minutes)

### 1. Build with Coverage

```bash
./build_metal.sh --build-type ASanCoverage --build-tests
```

### 2. Run Tests (collect coverage data)

```bash
# Setup environment
export LLVM_PROFILE_FILE="coverage/%p.profraw"
export TT_METAL_WATCHER_APPEND=1
mkdir -p coverage

# Run some tests
coverage run -m pytest tests/tt_eager/python_api_testing/unit_testing/ -xvvv -k "test_add" || true
./build_ASanCoverage/test/tt_metal/unit_tests --gtest_filter="*Add*" || true
```

### 3. Generate Report

**Option A: Use the test script (easiest)**
```bash
./.github/actions/code-coverage/test_local.sh
```

**Option B: Run manually**
```bash
./.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --source-dir . \
  --cpp-objects "./build_ASanCoverage/lib/libtt_metal.so ./build_ASanCoverage/lib/libtt_stl.so ./build_ASanCoverage/test/tt_metal/unit_tests" \
  --html-output-dir coverage/html
```

### 4. View Report

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
