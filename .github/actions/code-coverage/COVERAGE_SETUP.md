# Coverage Setup Guide

## Question 1: How Python Tests Find C++ Binaries

### The `build` Symlink Mechanism

When you run `build_metal.sh`, it creates a **symlink** from `build` to the actual build directory:

```bash
# From build_metal.sh line 255
ln -nsf $build_dir build
```

So if you build with `--build-type ASanCoverage`, it creates:
- `build_ASanCoverage/` (actual build directory)
- `build` → `build_ASanCoverage` (symlink)

### How Python Tests Use It

Python tests find C++ libraries through:

1. **LD_LIBRARY_PATH**: Many test scripts set:
   ```bash
   export LD_LIBRARY_PATH=$(pwd)/build/lib
   ```
   This uses the `build` symlink, so it automatically points to the correct build directory.

2. **RPATH in shared libraries**: The C++ libraries are built with RPATH settings that include `build/lib`, so they can find dependencies automatically.

3. **TT_METAL_HOME/build/lib**: Some code (like tt-train) explicitly looks for libraries in `TT_METAL_HOME/build/lib`, which also uses the symlink.

### Ensuring Coverage Builds Are Used

**To guarantee Python tests use `build_ASanCoverage`:**

1. **Build with ASanCoverage** (this creates/updates the symlink):
   ```bash
   ./build_metal.sh --build-type ASanCoverage --build-tests
   ```

2. **Verify the symlink**:
   ```bash
   ls -la build
   # Should show: build -> build_ASanCoverage
   ```

3. **Run tests** - they'll automatically use the correct build:
   ```bash
   export LLVM_PROFILE_FILE="coverage/%p.profraw"
   export TT_METAL_WATCHER_APPEND=1
   coverage run -m pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py
   ```

### Important Notes

- **Only one `build` symlink exists** - it always points to the **last build directory** you created
- If you build multiple times with different build types, the symlink will point to the **last one**
- **Solution**: Always build with `ASanCoverage` before running coverage tests, or manually update the symlink:
  ```bash
  ln -nsf build_ASanCoverage build
  ```

### Checking Which Build Python Is Using

You can verify which libraries Python is loading:

```python
import ctypes
import os

# Check LD_LIBRARY_PATH
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "not set"))

# Try to find the library
lib_path = os.path.join(os.getcwd(), "build", "lib", "libtt_metal.so")
print("Library path:", lib_path)
print("Exists:", os.path.exists(lib_path))

# Check symlink target
if os.path.islink("build"):
    print("build symlink points to:", os.readlink("build"))
```

---

## Question 2: Is `coverage` Part of Python?

**No, `coverage` is NOT part of Python's standard library.** You need to install it separately.

### Installation

```bash
# Using pip
pip install coverage

# Or pip3
pip3 install coverage

# Or in a virtual environment
python3 -m pip install coverage
```

### Verify Installation

```bash
python3 -c "import coverage; print(coverage.__version__)"
```

### Why It's Not Standard

The `coverage` module is a third-party tool maintained by Ned Batchelder. While it's widely used, it's not included in Python's standard library because:
- It's a development/testing tool, not core functionality
- It has its own dependencies
- It's updated independently of Python releases

### In CI/Workflows

If you're running in CI, you may need to install it:

```yaml
- name: Install coverage
  run: pip install coverage
```

Or add it to your `requirements.txt` or `requirements-dev.txt` if you have one.

---

## Complete Coverage Workflow

Here's the complete workflow to ensure everything works:

```bash
# 1. Install coverage (if not already installed)
pip install coverage

# 2. Build with coverage flags
./build_metal.sh --build-type ASanCoverage --build-tests

# 3. Verify build symlink
ls -la build  # Should point to build_ASanCoverage

# 4. Setup environment
export LLVM_PROFILE_FILE="coverage/%p.profraw"
export TT_METAL_WATCHER_APPEND=1
export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH}"
mkdir -p coverage

# ASanCoverage builds require AddressSanitizer runtime - ensure it's available
# The ASan library should be in standard paths, but if you get undefined symbol errors,
# you may need to add it explicitly (usually not needed):
# export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# 5. Run Python tests with coverage
coverage run -m pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py

# 6. Run C++ tests (they'll generate profraw files automatically)
./build/test/tt_metal/unit_tests --gtest_filter="*"

# 7. Generate coverage report
./.github/actions/code-coverage/entrypoint.sh \
  --coverage-dir coverage \
  --source-dir . \
  --cpp-objects "./build/lib/libtt_metal.so ./build/lib/libtt_stl.so ./build/test/tt_metal/unit_tests" \
  --html-output-dir coverage/html
```

---

## Troubleshooting

### Python can't find C++ libraries

**Problem**: Import errors or library not found errors.

**Solution**:
```bash
# Check symlink
ls -la build

# Set LD_LIBRARY_PATH explicitly
export LD_LIBRARY_PATH=$(pwd)/build/lib:$LD_LIBRARY_PATH

# Verify libraries exist
ls -la build/lib/libtt_metal.so
ls -la build/lib/_ttnncpp.so
```

### Coverage module not found

**Problem**: `ModuleNotFoundError: No module named 'coverage'`

**Solution**:
```bash
pip install coverage
# Or
pip3 install coverage
```

### Wrong build directory being used

**Problem**: Tests are using libraries from a different build type.

**Solution**:
```bash
# Rebuild with correct type (updates symlink)
./build_metal.sh --build-type ASanCoverage --build-tests

# Or manually update symlink
ln -nsf build_ASanCoverage build
```
