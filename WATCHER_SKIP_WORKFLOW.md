# Watcher Skip Workflow

## Overview
This document describes the workflow for processing test failures when running with `TT_METAL_WATCHER=1` enabled. The workflow is split between a **Main Thread** (runs tests, detects failures) and an **Agent** (analyzes failures, adds skips, documents issues).

## Configuration

| Setting | Value |
|---------|-------|
| Pipeline | Blackhole Post-Commit (`blackhole-post-commit.yaml`) |
| Master Issue | https://github.com/tenstorrent/tt-metal/issues/27840 |
| Branch Pattern | `dstoiljkovic/bh_post_commit_skips_<group_number>` |

## Test Groups (Branches)

| Group # | Job Name | Branch |
|---------|----------|--------|
| 1 | run-profiler-regression | `dstoiljkovic/bh_post_commit_skips_1` |
| 2 | umd-unit-tests | `dstoiljkovic/bh_post_commit_skips_2` |
| 3 | models-unit-tests | `dstoiljkovic/bh_post_commit_skips_3` |
| 4 | blackhole-demo-tests | `dstoiljkovic/bh_post_commit_skips_4` |
| 5 | ttnn-unit-tests | `dstoiljkovic/bh_post_commit_skips_5` |
| 6 | ops-unit-tests | `dstoiljkovic/bh_post_commit_skips_6` |
| 7 | metalium-smoke-tests | `dstoiljkovic/bh_post_commit_skips_7` |
| 8 | ttnn-smoke-tests | `dstoiljkovic/bh_post_commit_skips_8` |
| 9 | tt-cnn-unit-tests | `dstoiljkovic/bh_post_commit_skips_9` |
| 10 | blackhole-20-cores-tests | `dstoiljkovic/bh_post_commit_skips_10` |
| 11 | blackhole-multi-card-post-commit-tests | `dstoiljkovic/bh_post_commit_skips_11` |

---

## Build and Environment Setup

### Building the Project
```bash
./build_metal.sh --build-tests
```

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md) sections:
- "Running post-commit regressions"
- "Running Googletest (gtest) C++ tests"

### Watcher Documentation

**Primary documentation:**
- Online docs: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html
- Local docs: `docs/source/tt-metalium/tools/watcher.rst`
- Debugging guide: `tech_reports/Debugging/Kernel_Debugging_Tips.md`
- Contributing guide: `CONTRIBUTING.md` (section "Using watcher")

**Key watcher concepts:**
- Enable with `TT_METAL_WATCHER=1` environment variable
- Watcher log location: `generated/watcher/watcher.log`
- Detects: NOC errors, bad alignment, race conditions, invalid addresses
- Useful for hang debugging - check waypoints in watcher log

**Example watcher error output:**
```
Always | WARNING  | Watcher detected NOC error and stopped device: bad alignment in NOC transaction.
Always | WARNING  | Device 0 worker core(x= 0,y= 0) virtual(x= 1,y= 1): brisc using noc0 tried to access DRAM core w/ physical coords (x=0,y=11) DRAM[addr=0x00003820,len=102400], misaligned with local L1[addr=0x00064010]
Always | INFO     | Last waypoint: NARW,   W,   W,   W,   W
Always | INFO     | While running kernels:
Always | INFO     |  brisc : tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp
```

---

## Issue Assignment Reference

### Finding Code Owners

The CODEOWNERS file is located at `.github/CODEOWNERS`. Use it to determine who should be assigned to issues.

**How to find the owner for a file:**
```bash
# Check CODEOWNERS for a specific path
grep -E "^[^#].*<path_pattern>" .github/CODEOWNERS
```

**Key ownership patterns:**
| Path Pattern | Owners |
|--------------|--------|
| `tt_metal/hw/ckernels/` | LLK team (@rtawfik01, @rdjogoTT, @nvelickovicTT, etc.) |
| `tt_metal/impl/dispatch/` | Runtime team (@jbaumanTT, @nhuang-tt, @tt-asaigal, @tt-aho) |
| `tt_metal/fabric/` | Fabric team (@ubcheema, @aliuTT, @aagarwalTT, @SeanNijjar, etc.) |
| `tt_metal/distributed/` | Distributed team (@cfjchu, @aliuTT, @tt-asaigal, @jbaumanTT) |
| `ttnn/cpp/ttnn/operations/` | TTNN ops team |
| `tests/ttnn/` | Test owners vary by subdirectory |

### Using Git History for Assignment

**Check recent contributors to the failing test file:**
```bash
git log --oneline -10 <test_file_path>
git log --format="%an" -10 <test_file_path> | sort | uniq -c | sort -rn
```

**Check who last modified the relevant kernel/op:**
```bash
git blame <kernel_file_path> | head -20
```

**Find the author of the test:**
```bash
git log --diff-filter=A --format="%an <%ae>" -- <test_file_path>
```

### Assignment Priority
1. Check CODEOWNERS for the failing test file path
2. Check CODEOWNERS for the kernel/op mentioned in watcher output
3. Check git history for recent contributors to the test
4. Check git blame for the kernel code mentioned in the error
5. Default to the team owning the parent directory

---

## Tests to Skip Without Issue

Some tests are **not rational to run with watcher enabled** and should be skipped without creating or linking to an issue. These tests fall into categories where watcher overhead fundamentally interferes with the test's purpose.

### Categories of Tests to Skip Without Issue

| Category | Reason | Example Test Groups |
|----------|--------|---------------------|
| **Profiler/Performance regression** | Watcher adds overhead that invalidates performance measurements | `run-profiler-regression` |
| **Timing-sensitive benchmarks** | Watcher instrumentation affects timing accuracy | Performance microbenchmarks |
| **Stress tests with tight timeouts** | Watcher overhead may cause spurious timeout failures | High-throughput stress tests |
| **Kernel OOM / Code size exceeded** | Watcher increases kernel code size, causing L1 memory overflow - no actionable info | Tests with large/complex kernels |

### Skip Syntax (No Issue Required)

#### Python - Performance/Profiler tests:
```python
from models.common.utility_functions import skip_with_watcher

@skip_with_watcher("Profiler test not rational with watcher overhead")
def test_profiler_regression():
    ...
```

#### Python - Kernel OOM due to watcher code size:
```python
from models.common.utility_functions import skip_with_watcher

@skip_with_watcher("Kernel OOM with watcher enabled due to code size increase")
def test_large_kernel():
    ...
```

#### C++ - Performance/Profiler tests:
```cpp
#include "tests/tt_metal/test_utils/env_vars.hpp"

TEST(ProfilerSuite, PerformanceRegression) {
    SKIP_FOR_WATCHER();  // Profiler test not rational with watcher overhead
    // ... rest of test
}
```

#### C++ - Kernel OOM due to watcher code size:
```cpp
#include "tests/tt_metal/test_utils/env_vars.hpp"

TEST(LargeKernelSuite, ComplexKernel) {
    SKIP_FOR_WATCHER();  // Kernel OOM with watcher enabled due to code size increase
    // ... rest of test
}
```

### How to Identify These Tests

**Profiler/Performance tests:**
1. **Test group name** contains: `profiler`, `perf`, `performance`, `benchmark`, `regression` (performance context)
2. **Test file path** is under: `tests/tt_metal/perf_microbenchmark/`, profiler directories
3. **Test measures**: latency, throughput, cycle counts, or other timing metrics
4. **Test has tight timeouts** that would be affected by watcher overhead

**Kernel OOM / Code size exceeded:**
1. **Error message** contains: `Kernel OOM`, `code size exceeded`, `L1 memory overflow`, `out of memory`, `text section too large`
2. **Watcher output** shows memory allocation failures related to code size
3. **Test uses large/complex kernels** that are already near L1 limits

### Agent Behavior for These Tests

When the agent encounters a failure in a test that falls into these categories:
1. **Do NOT search for existing issues**
2. **Do NOT create issue content**
3. **Add skip with descriptive reason** (no issue number)
4. **Commit message**: `Skip [test_name] with watcher enabled (not rational for profiler/perf test)`

---

## Main Thread Responsibilities

The main thread orchestrates the overall workflow and runs the tests.

### Setup Phase
1. **Create branch** for the test group being processed
   ```bash
   git checkout -b dstoiljkovic/bh_post_commit_skips_<group_number>
   ```

2. **Set environment** for watcher-enabled testing
   ```bash
   export TT_METAL_WATCHER=1
   ```

### Execution Phase
3. **Run tests** for the specific test group
   - Execute the test command from the pipeline (e.g., pytest command from `ttnn-tests.yaml`)
   - Capture terminal output (stdout/stderr)

4. **On test failure**:
   - Capture the full terminal output including watcher errors
   - Extract the failing test information:
     - Test file path
     - Test function name
     - Parameter combination (if parameterized)
   - **Invoke the Agent** with this information

5. **After Agent completes**:
   - Agent will have added the skip and committed
   - **Re-run the same test command** to continue processing remaining tests
   - Repeat until all tests in the group pass (or are skipped)

### Completion Phase
6. **When all tests pass** (or are skipped):
   - Report summary of skips added
   - Branch is ready for review

---

## Agent Responsibilities

The agent is invoked as a hook when a test fails. It receives the failure context and handles the skip logic.

### Input (received from Main Thread)
- Test command that failed
- Terminal output (including watcher output)
- Current working directory and branch
- Failing test details (file, function, parameters)

### Agent Workflow

1. **Parse the failure**
   - Extract the test file path and test function name
   - Extract the specific parameter combination that failed
   - Identify the watcher error from terminal output

2. **Analyze watcher output**
   - Look for error patterns: invalid NOC addresses, alignment issues, race conditions, runtime build errors
   - Identify the kernel/op involved
   - Note the specific error message

3. **Search for existing issues**
   - Query sub-issues of master issue #27840
   - Match based on: error pattern, kernel name, op name
   - Use: `gh issue list --search "is:issue is:open" --json number,title,body -L 100 | jq '.[] | select(.body | contains("#27840"))'`

4. **Determine skip type**
   - **Entire test fails**: Use `@skip_with_watcher()` (Python) or `SKIP_FOR_WATCHER()` (C++)
   - **Specific parameters fail**: Use `if is_watcher_enabled()` check with parameter conditions

5. **Add the appropriate skip** to the test file

6. **Determine issue assignee** (for new issues)
   - Check `.github/CODEOWNERS` for the test file path
   - Check `.github/CODEOWNERS` for the kernel/op mentioned in watcher output
   - Use `git log` to find recent contributors
   - Use `git blame` on the kernel file if identified

7. **Prepare issue content** (for manual review)
   - Create issue content in `/tmp/watcher_issues/` directory
   - Follow the template below
   - If matching issue found, note the issue number
   - Include suggested assignee based on CODEOWNERS/git history

8. **Commit the skip**
   - Commit message format: `Skip [test_name] with watcher enabled (pending issue)` or `Skip [test_name] with watcher enabled (#ISSUE_NUMBER)` if issue exists

9. **Report back to Main Thread**
   - What skip was added
   - Whether existing issue was found or new issue content was prepared
   - Suggested assignee for new issue
   - Ready for next test run

---

## Skip Syntax Reference

### Python Tests

#### Entire test fails - use decorator:
```python
from models.common.utility_functions import skip_with_watcher

@skip_with_watcher("GitHub Issue #XXXX")
def test_something():
    ...
```

#### Specific parameters fail - use conditional check inside test:
```python
from models.common.utility_functions import is_watcher_enabled

def test_something(param1, param2, param3):
    # Skip specific parameter combination, github issue #XXXX
    if param1 == failing_val1 and param2 == failing_val2 and is_watcher_enabled():
        pytest.skip("Test case fails with watcher enabled")

    # ... rest of test
```

### C++/GTest Tests

#### Entire test fails - use macro at start:
```cpp
#include "tests/tt_metal/test_utils/env_vars.hpp"

TEST(TestSuite, TestName) {
    SKIP_FOR_WATCHER();  // GitHub Issue #XXXX
    // ... rest of test
}
```

#### Specific parameters fail - use conditional check with macro:
```cpp
#include "tests/tt_metal/test_utils/env_vars.hpp"

TEST_P(TestFixture, TestName) {
    auto params = GetParam();
    // Skip specific parameter combination, github issue #XXXX
    if (params.x == failing_value && params.y == other_value) {
        SKIP_FOR_WATCHER();
    }
    // ... rest of test
}
```

---

## Issue Template (for manual review)

Save to `/tmp/watcher_issues/issue_<timestamp>.md`:

```markdown
---
title: "[WATCHER] <kernel/op name>: <brief error description>"
labels: bug, watcher
assignees: <suggested assignee from CODEOWNERS/git history>
---

## Component / Area
<ops|kernels|dataflow|runtime - based on error>

## Issue Type
<from watcher output: Race Condition, Invalid Address, Runtime Crash, etc.>

## Observed
When running with `TT_METAL_WATCHER=1`:
<paste relevant watcher output>

## Expected
Test should pass with watcher enabled.

## Steps to Reproduce
```bash
TT_METAL_WATCHER=1 pytest <test_path>::<test_name>[<params>] -xvs
```

## Test Details
- **File**: <test file path>
- **Test**: <test function name>
- **Parameters**: <failing parameter combination>

## Watcher Output
```
<full watcher error output>
```

## System Details
- **Hardware**: Blackhole
- **Pipeline**: BH Post-Commit

## Related
- Parent issue: #27840

## Suggested Assignee
Based on CODEOWNERS/git history: @<username>
```
