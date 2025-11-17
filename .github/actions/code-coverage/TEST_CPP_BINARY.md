# Testing C++ Binaries with Coverage

## Step-by-Step Process

### Step 1: Build with Coverage (if not already done)

```bash
./build_metal.sh --build-type ASanCoverage --build-tests
```

This creates:
- `build_ASanCoverage/` directory with your binaries
- `build` symlink pointing to `build_ASanCoverage`

### Step 2: Setup Coverage Environment

```bash
# Source the setup script (installs dependencies and sets environment variables)
source .github/actions/code-coverage/setup_coverage_env.sh
```

This will:
- Install missing tools (llvm-profdata, llvm-cov, genhtml, Python coverage)
- Set `LLVM_PROFILE_FILE` for C++ coverage collection
- Set `TT_METAL_WATCHER_APPEND=1` for kernel tracking
- Set `LD_PRELOAD` and `LD_LIBRARY_PATH` for ASan runtime

**Important**: After sourcing, you'll see `LD_PRELOAD` is set. This is needed for running the C++ binary, but we'll handle it carefully.

### Step 3: Run Your C++ Test Binary

**Option A: Use the helper script (recommended - handles LD_PRELOAD correctly):**
```bash
.github/actions/code-coverage/run_cpp_test.sh ./build_ASanCoverage/test/tt_metal/test_add_two_ints
```

**Option B: Run manually (make sure LD_PRELOAD is NOT set):**
```bash
# Make sure you're in the repo root
cd /tt-metal

# IMPORTANT: Unset LD_PRELOAD for C++ binaries (they have ASan statically linked)
unset LD_PRELOAD

# Set LD_LIBRARY_PATH to find libraries
export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH}"

# Run the test binary
# The LLVM_PROFILE_FILE environment variable will cause .profraw files to be generated
./build_ASanCoverage/test/tt_metal/test_add_two_ints
```

**What happens:**
- The test runs and generates coverage data
- `.profraw` files are created in the `coverage/` directory (as specified by `LLVM_PROFILE_FILE`)
- If the test uses kernels, `generated/watcher/kernel_names.txt` will be updated

### Step 4: Generate Coverage Report

```bash
# Run the coverage report generator
.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --source-dir . \
  --cpp-objects "./build_ASanCoverage/test/tt_metal/test_add_two_ints ./build_ASanCoverage/lib/libtt_metal.so ./build_ASanCoverage/lib/libtt_stl.so" \
  --enable-python-coverage false \
  --html-output-dir coverage/html
```

**What this does:**
- Collects C++ coverage from `.profraw` files
- Generates kernel coverage from `generated/watcher/kernel_names.txt` (if it exists)
- Merges everything into a single HTML report

### Step 5: View the Report

```bash
# Open the HTML report
xdg-open coverage/html/index.html  # Linux
# or
open coverage/html/index.html      # macOS
```

---

## Complete Example (All Steps Together)

```bash
# 1. Build (if needed)
./build_metal.sh --build-type ASanCoverage --build-tests

# 2. Setup environment
source .github/actions/code-coverage/setup_coverage_env.sh

# 3. Run your C++ test (using helper script - handles LD_PRELOAD correctly)
.github/actions/code-coverage/run_cpp_test.sh ./build_ASanCoverage/test/tt_metal/test_add_two_ints

# OR run manually (make sure to unset LD_PRELOAD first):
# unset LD_PRELOAD
# export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH}"
# ./build_ASanCoverage/test/tt_metal/test_add_two_ints

# 4. Generate report
.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --source-dir . \
  --cpp-objects "./build_ASanCoverage/test/tt_metal/test_add_two_ints ./build_ASanCoverage/lib/libtt_metal.so ./build_ASanCoverage/lib/libtt_stl.so" \
  --enable-python-coverage false \
  --html-output-dir coverage/html

# 5. View report
xdg-open coverage/html/index.html
```

---

## Troubleshooting

### If you get "incompatible ASan runtimes" errors:

**This means LD_PRELOAD is set but the binary has ASan statically linked.**

**Solution:** Unset LD_PRELOAD for C++ binaries:
```bash
unset LD_PRELOAD
export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH}"
./build_ASanCoverage/test/tt_metal/test_add_two_ints
```

Or use the helper script which handles this automatically:
```bash
.github/actions/code-coverage/run_cpp_test.sh ./build_ASanCoverage/test/tt_metal/test_add_two_ints
```

### If you get "undefined symbol" errors when running Python tests:

For Python tests, you DO need LD_PRELOAD:
```bash
export LD_PRELOAD="/usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.asan-x86_64.so"
export LD_LIBRARY_PATH="$(pwd)/build/lib:/usr/lib/llvm-17/lib/clang/17/lib/linux:${LD_LIBRARY_PATH}"
coverage run -m pytest tests/...
```

### If no .profraw files are generated:

Check that `LLVM_PROFILE_FILE` is set:
```bash
echo $LLVM_PROFILE_FILE
# Should show: coverage/%p.profraw
```

### If the report shows no coverage:

Make sure you specified the correct binary in `--cpp-objects`:
```bash
# Verify the binary exists
ls -la ./build_ASanCoverage/test/tt_metal/test_add_two_ints

# Make sure you include both the test binary AND the libraries it uses
```

---

## Quick Reference

**Environment Variables Set by `setup_coverage_env.sh`:**
- `LLVM_PROFILE_FILE=coverage/%p.profraw` - Where C++ coverage data goes
- `TT_METAL_WATCHER_APPEND=1` - Track kernel usage
- `LD_PRELOAD` - Preload ASan runtime (for Docker/ASanCoverage builds)
- `LD_LIBRARY_PATH` - Find C++ libraries and ASan runtime

**Files Generated:**
- `coverage/*.profraw` - C++ coverage data (one per process)
- `coverage/coverage.profdata` - Merged C++ coverage
- `coverage/cpp_coverage.info` - C++ coverage in LCOV format
- `coverage/kernel_coverage.info` - Kernel coverage in LCOV format
- `coverage/merged_coverage.info` - Combined coverage
- `coverage/html/` - HTML report directory
