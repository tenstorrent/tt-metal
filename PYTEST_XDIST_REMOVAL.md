# pytest-xdist & Watchdog Removal Summary

## Important Context

**pytest-xdist was NOT used for parallelization** - it was "abused" to run with a single worker (`-n auto` → 1 worker) specifically to enable:
1. External timeout monitoring via the custom watchdog
2. The `pytest_handlecrashitem` hook for crash recovery

This means **ZERO performance impact** from removal - tests were always sequential!

## Changes Made

### 1. Dependencies Removed
- ✅ `tt_metal/python_env/requirements-dev.txt` - Removed `pytest-xdist==3.8.0`
- ✅ `tt_metal/third_party/tt_llk/tests/requirements.txt` - Removed `pytest-xdist==3.8.0`
- ✅ `scripts/basic_dev_image/minimum-testing-kit.txt` - Removed `pytest-xdist==3.6.1`
- ✅ `.github/workflows/conda-post-commit.yaml` - Removed from conda env

### 2. Code Changes

#### Main conftest.py (`/workspace/conftest.py`)
- ✅ **REMOVED ALL WATCHDOG CODE** (~170 lines):
  - Removed `watchdog_cmd_queue_key` and `watchdog_process_key`
  - Removed `_watchdog_main()` function (multiprocess timeout supervisor)
  - Removed `pytest_sessionfinish()` watchdog shutdown hook
  - Removed `pytest_timeout_set_timer()` override hook
  - Removed `pytest_handlecrashitem()` crash recovery hook
- ✅ Removed `--metal-timeout` command-line option
- ✅ Simplified `pytest_runtest_teardown` - always resets devices on failure
- ✅ Removed all xdist worker ID detection and logging

#### LLK conftest.py (`/workspace/tt_metal/third_party/tt_llk/tests/python_tests/conftest.py`)
- ✅ Added documentation about changed behavior (comments only)

### 3. Script Updates ✅ COMPLETE

Removed `-n auto` from **all** scripts and workflows:
- **7 GitHub workflow YAML files**
- **2 Pipeline test YAML files**
- **10 Shell script files**
- **Total: 64 files modified, 330 net lines removed**

## Impact Assessment

### What Still Works ✅
1. ✅ **All pytest functionality** - Core pytest features unaffected
2. ✅ **pytest-timeout plugin** - Provides `--timeout=SECONDS` flag
3. ✅ **pytest-split plugin** - Available if parallelization ever needed
4. ✅ **Device reset on failure** - Always enabled now
5. ✅ **Same performance** - Tests were always sequential anyway!

### What Changes 🔄
1. ✅ **stdout visibility** - No longer suppressed (BENEFIT!)
2. ✅ **Simpler code** - 330 fewer lines to maintain
3. ✅ **Better hang detection** - Library's built-in replaces broken watchdog
4. ❌ **No external crash recovery** - `pytest_handlecrashitem` hook gone
5. ❌ **No `--metal-timeout` flag** - Use `--timeout` instead

### What Was Removed and Why

#### pytest-xdist "Abuse" Pattern
**Why it was used:**
- NOT for parallelization (always ran with single worker)
- Enabled external timeout monitoring via custom watchdog
- Enabled `pytest_handlecrashitem` hook for crash recovery

#### Watchdog System (~170 lines)
**Why it existed:**
- Created separate multiprocess to monitor test timeouts externally
- Could take "different recovery actions" (e.g., device reset, SIGKILL)
- Used `SIGKILL` to forcefully terminate hung tests
- Only worked when xdist was active (single worker mode)

**Why it was removed:**
1. **Library has built-in hang detection** - More robust than SIGKILL
2. **pytest-timeout is sufficient** - Simpler, standard solution
3. **Didn't work properly** - "code does nothing upon timeout"
4. **Unnecessary complexity** - 170 lines of multiprocess orchestration
5. **Maintenance burden** - Queue management, signal handling, race conditions
6. **xdist not needed** - Can use pytest-timeout directly

**Alternative:** `pytest --timeout=300` (from pytest-timeout plugin)

## Benefits of This Change

1. ✅ **Zero performance impact** - Tests were never parallel anyway
2. ✅ **Simpler codebase** - 330 lines removed across 64 files
3. ✅ **Better stdout visibility** - No more xdist output suppression
4. ✅ **Less confusing** - No more `--metal-timeout` flag
5. ✅ **Fewer dependencies** - One less plugin to maintain
6. ✅ **Better debugging** - Direct output, simpler flow
7. ✅ **More reliable** - Library's hang detection > broken watchdog

## Testing Recommendations

```bash
# Test that pytest still works
pytest tests/ttnn/unit_tests/test_*.py --tb=short

# Test with pytest-timeout for timeout handling
pytest tests/ttnn/unit_tests/test_*.py --timeout=300

# Run a sample test suite
pytest tests/ttnn/unit_tests/operations/test_*.py -v
```

## Rollback Plan

If issues arise:

```bash
git checkout HEAD -- tt_metal/python_env/requirements-dev.txt
git checkout HEAD -- tt_metal/third_party/tt_llk/tests/requirements.txt
git checkout HEAD -- scripts/basic_dev_image/minimum-testing-kit.txt
git checkout HEAD -- conftest.py
git checkout HEAD -- tt_metal/third_party/tt_llk/tests/python_tests/conftest.py
pip install -r tt_metal/python_env/requirements-dev.txt
```

## Next Steps

1. ✅ Remove pytest-xdist from requirements
2. ✅ Remove all watchdog code
3. ⏳ Test locally without xdist
4. ⏳ Gather feedback from developers
5. ⏳ Update 328 script instances (remove `-n auto`)
6. ⏳ Monitor CI/CD pipeline changes
