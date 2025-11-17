# Troubleshooting Guide

## Common Issues and Solutions

### Issue: `undefined symbol: __asan_option_detect_stack_use_after_return`

**Error Message:**
```
ImportError: /tt-metal/build_ASanCoverage/lib/libtt_stl.so: undefined symbol: __asan_option_detect_stack_use_after_return
```

**Cause:**
The `ASanCoverage` build type includes AddressSanitizer, which requires the ASan runtime library (`libasan.so`) to be available at runtime. The library isn't being found by the dynamic linker.

**Solution:**

1. **Use LD_PRELOAD to preload the Clang ASan runtime (REQUIRED for Docker containers):**
   ```bash
   # For Clang-17 builds (most common)
   export LD_PRELOAD="/usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.asan-x86_64.so"
   export LD_LIBRARY_PATH="$(pwd)/build/lib:/usr/lib/llvm-17/lib/clang/17/lib/linux:${LD_LIBRARY_PATH}"

   # Or if using a different Clang version, find it:
   # find /usr/lib/llvm-* -name "libclang_rt.asan-x86_64.so" 2>/dev/null
   ```

   **Why?**
   - Clang uses its own ASan runtime (`libclang_rt.asan-x86_64.so`), not the system `libasan.so`
   - In Docker containers, the ASan runtime must be **preloaded** using `LD_PRELOAD` to ensure it's loaded before your libraries
   - `LD_LIBRARY_PATH` alone isn't sufficient - the runtime must be explicitly preloaded

2. **Verify ASan library exists:**
   ```bash
   # Find ASan library
   find /usr -name "libasan*.so*" 2>/dev/null

   # Should show something like:
   # /usr/lib/x86_64-linux-gnu/libasan.so.8
   ```

3. **If ASan library is missing, install it:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libasan8  # or libasan6, depending on your compiler version

   # Or install the full compiler toolchain
   sudo apt-get install clang-17 libclang-17-dev
   ```

4. **Verify the symlink is correct:**
   ```bash
   ls -la build
   # Should show: build -> build_ASanCoverage
   ```

5. **Complete working example (especially for Docker containers):**
   ```bash
   # Build
   ./build_metal.sh --build-type ASanCoverage --build-tests

   # Setup environment (IMPORTANT: LD_PRELOAD is required in Docker!)
   export LLVM_PROFILE_FILE="coverage/%p.profraw"
   export TT_METAL_WATCHER_APPEND=1
   export LD_PRELOAD="/usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.asan-x86_64.so"
   export LD_LIBRARY_PATH="$(pwd)/build/lib:/usr/lib/llvm-17/lib/clang/17/lib/linux:${LD_LIBRARY_PATH}"
   mkdir -p coverage

   # Run tests
   coverage run -m pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py
   ```

---

### Issue: Python can't import ttnn or other modules

**Error Message:**
```
ImportError: cannot open shared object file: No such file or directory
```

**Cause:**
Python can't find the C++ shared libraries.

**Solution:**

1. **Set LD_LIBRARY_PATH:**
   ```bash
   export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH}"
   ```

2. **Verify libraries exist:**
   ```bash
   ls -la build/lib/libtt_metal.so
   ls -la build/lib/_ttnncpp.so
   ```

3. **Check symlink:**
   ```bash
   ls -la build
   # Should point to build_ASanCoverage
   ```

---

### Issue: No .profraw files generated

**Cause:**
C++ coverage data isn't being collected.

**Solution:**

1. **Verify LLVM_PROFILE_FILE is set:**
   ```bash
   echo $LLVM_PROFILE_FILE
   # Should show: coverage/%p.profraw
   ```

2. **Verify code was built with coverage flags:**
   ```bash
   # Check if build was done with ASanCoverage
   ls -la build
   # Should show: build -> build_ASanCoverage

   # Check if libraries have coverage symbols
   nm build/lib/libtt_metal.so | grep __llvm_profile
   # Should show symbols if coverage is enabled
   ```

3. **Run tests and check for profraw files:**
   ```bash
   export LLVM_PROFILE_FILE="coverage/%p.profraw"
   ./build/test/tt_metal/unit_tests --gtest_filter="*"
   ls -la coverage/*.profraw
   ```

---

### Issue: No .coverage file for Python

**Cause:**
Python tests weren't run with `coverage run`.

**Solution:**

1. **Use coverage run:**
   ```bash
   # Wrong:
   pytest tests/...

   # Correct:
   coverage run -m pytest tests/...
   ```

2. **Check for .coverage file:**
   ```bash
   ls -la .coverage
   # Or
   ls -la coverage/.coverage
   ```

---

### Issue: Kernel names file not found

**Error Message:**
```
Warning: Kernel names file not found: generated/watcher/kernel_names.txt
```

**Cause:**
Tests didn't execute kernel code, or `TT_METAL_WATCHER_APPEND` wasn't set.

**Solution:**

1. **Set TT_METAL_WATCHER_APPEND before running tests:**
   ```bash
   export TT_METAL_WATCHER_APPEND=1
   ```

2. **Run tests that actually use kernels:**
   ```bash
   # Tests that use device code
   coverage run -m pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py
   ```

3. **Check if file was created:**
   ```bash
   ls -la generated/watcher/kernel_names.txt
   ```

---

### Issue: genhtml errors or warnings

**Error Message:**
```
genhtml: ERROR: ... no data found
```

**Cause:**
LCOV file is empty or invalid.

**Solution:**

1. **Check if coverage files were generated:**
   ```bash
   ls -la coverage/*.info
   ```

2. **Check file contents:**
   ```bash
   head coverage/merged_coverage.info
   # Should show LCOV format data
   ```

3. **Verify all components are enabled:**
   ```bash
   # Make sure you're collecting all types of coverage
   # Check that cpp, python, and kernel coverage files exist
   ls -la coverage/cpp_coverage.info
   ls -la coverage/python_coverage.info
   ls -la coverage/kernel_coverage.info
   ```

---

### Issue: Wrong build directory being used

**Cause:**
The `build` symlink points to the wrong directory.

**Solution:**

1. **Rebuild with correct type:**
   ```bash
   ./build_metal.sh --build-type ASanCoverage --build-tests
   ```

2. **Or manually update symlink:**
   ```bash
   ln -nsf build_ASanCoverage build
   ```

3. **Verify:**
   ```bash
   ls -la build
   # Should show: build -> build_ASanCoverage
   ```

---

## Quick Diagnostic Commands

Run these to diagnose issues:

```bash
# Check build symlink
echo "Build symlink:" && ls -la build

# Check libraries exist
echo "Libraries:" && ls -la build/lib/*.so | head -5

# Check environment variables
echo "LLVM_PROFILE_FILE: $LLVM_PROFILE_FILE"
echo "TT_METAL_WATCHER_APPEND: $TT_METAL_WATCHER_APPEND"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check coverage files
echo "Coverage files:" && ls -la coverage/ 2>/dev/null || echo "No coverage directory"

# Check ASan library
echo "ASan library:" && find /usr -name "libasan*.so*" 2>/dev/null | head -1

# Test Python import
python3 -c "import sys; sys.path.insert(0, '.'); import ttnn; print('ttnn imported successfully')" 2>&1
```

---

## Still Having Issues?

1. **Check the full error message** - it often contains clues
2. **Verify all prerequisites** are installed (see README.md)
3. **Check file permissions** - make sure files are readable
4. **Try a clean build** - sometimes stale build artifacts cause issues:
   ```bash
   rm -rf build_ASanCoverage build coverage
   ./build_metal.sh --build-type ASanCoverage --build-tests
   ```
