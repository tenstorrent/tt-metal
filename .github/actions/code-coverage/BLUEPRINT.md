# Code Coverage Tool Blueprint for Tenstorrent Repositories

## Executive Summary

**Feasibility: ✅ HIGHLY FEASIBLE**

This document outlines a comprehensive blueprint for implementing a unified code coverage tool that combines:
- **C++ coverage** via LLVM's coverage tools (llvm-cov/llvm-profdata) → LCOV format
- **Python coverage** via Python's `coverage` library → LCOV format
- **Kernel code coverage** via parsing `generated/watcher/kernel_names.txt` → synthetic LCOV entries

The tool will be implemented as a reusable GitHub Action that can work across multiple Tenstorrent repositories.

---

## 1. Feasibility Analysis

### 1.1 Existing Infrastructure ✅

**C++ Coverage:**
- ✅ LLVM coverage infrastructure already exists (see `.github/workflows/smoke.yaml`)
- ✅ Uses `-fprofile-instr-generate -fcoverage-mapping` flags (ASanCoverage build type)
- ✅ `llvm-profdata` and `llvm-cov` tools are available
- ✅ Can export to LCOV format: `llvm-cov export -format=lcov`

**Python Coverage:**
- ✅ Python's `coverage` library supports LCOV export: `coverage lcov`
- ✅ Pytest integration is straightforward: `coverage run -m pytest`
- ✅ Can merge multiple coverage runs

**Kernel Tracking:**
- ✅ Watcher mechanism is well-established
- ✅ `TT_METAL_WATCHER_APPEND=1` ensures kernel names are appended
- ✅ File format is simple: `ID: path/to/kernel.cpp`
- ✅ Paths are relative to repository root

### 1.2 Technical Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Multiple test runs** | Use coverage append modes; merge at the end |
| **C++ binaries need coverage flags** | Detect build type; require coverage-enabled builds |
| **Kernel files marked as 100% covered** | Generate synthetic LCOV entries with all lines marked as executed |
| **Combining LCOV files** | Use `lcov --add-tracefile` or Python script to merge |
| **HTML report generation** | Use `genhtml` (from lcov package) on merged LCOV file |
| **Cross-repo compatibility** | Make action configurable with input parameters |

### 1.3 Requirements Checklist

- ✅ Set `TT_METAL_WATCHER_APPEND=1` before tests
- ✅ Build C++ code with coverage flags (`-fprofile-instr-generate -fcoverage-mapping`)
- ✅ Run Python tests with `coverage run`
- ✅ Run C++ tests with `LLVM_PROFILE_FILE` set
- ✅ Parse `kernel_names.txt` and generate synthetic coverage
- ✅ Merge all coverage data into single LCOV file
- ✅ Generate HTML report
- ✅ Support multiple test executions

---

## 2. Architecture Overview

### 2.1 Component Structure

```
.github/actions/code-coverage/
├── action.yml                    # GitHub Action definition
├── entrypoint.sh                 # Main entrypoint script
├── merge_coverage.py            # Python script for merging coverage
├── generate_kernel_coverage.py   # Generate synthetic LCOV for kernels
├── README.md                     # Usage documentation
└── BLUEPRINT.md                 # This file
```

### 2.2 Data Flow

```
┌─────────────────┐
│  Test Execution │
└────────┬────────┘
         │
         ├──> C++ Tests ──────> *.profraw files
         │                      (LLVM coverage)
         │
         ├──> Python Tests ───> .coverage file
         │                      (Python coverage)
         │
         └──> Kernel Execution ─> kernel_names.txt
                                   (Watcher output)
         │
         ▼
┌─────────────────┐
│ Coverage Merge  │
│   & Processing   │
└────────┬────────┘
         │
         ├──> llvm-profdata merge ──> coverage.profdata
         ├──> llvm-cov export ──────> cpp_coverage.info (LCOV)
         ├──> coverage lcov ────────> python_coverage.info (LCOV)
         └──> generate_kernel_coverage.py ─> kernel_coverage.info (LCOV)
         │
         ▼
┌─────────────────┐
│  Merge LCOV     │
│  Files          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate HTML  │
│  Report         │
└─────────────────┘
```

---

## 3. Implementation Blueprint

### 3.1 GitHub Action Inputs

```yaml
inputs:
  coverage-dir:
    description: 'Directory to store coverage data'
    required: false
    default: 'coverage'

  kernel-names-file:
    description: 'Path to kernel_names.txt file'
    required: false
    default: 'generated/watcher/kernel_names.txt'

  source-dir:
    description: 'Root directory of source code'
    required: false
    default: '.'

  cpp-objects:
    description: 'Space-separated list of C++ objects/binaries for coverage'
    required: false
    default: ''

  enable-cpp-coverage:
    description: 'Enable C++ coverage collection'
    required: false
    default: 'true'

  enable-python-coverage:
    description: 'Enable Python coverage collection'
    required: false
    default: 'true'

  enable-kernel-coverage:
    description: 'Enable kernel coverage (via kernel_names.txt)'
    required: false
    default: 'true'

  html-output-dir:
    description: 'Directory for HTML report output'
    required: false
    default: 'coverage/html'

  llvm-profdata-path:
    description: 'Path to llvm-profdata binary'
    required: false
    default: 'llvm-profdata'

  llvm-cov-path:
    description: 'Path to llvm-cov binary'
    required: false
    default: 'llvm-cov'
```

### 3.2 Action Workflow Steps

#### Step 1: Setup
- Create coverage directory
- Verify required tools are installed (`llvm-profdata`, `llvm-cov`, `coverage`, `genhtml`)
- Set `TT_METAL_WATCHER_APPEND=1` environment variable

#### Step 2: Collect C++ Coverage
- Find all `*.profraw` files in coverage directory
- Merge using `llvm-profdata merge`
- Export to LCOV using `llvm-cov export` for each binary/object
- Combine into single `cpp_coverage.info`

#### Step 3: Collect Python Coverage
- Check for `.coverage` file(s)
- Convert to LCOV: `coverage lcov -o python_coverage.info`
- Handle multiple `.coverage` files if present

#### Step 4: Generate Kernel Coverage
- Parse `kernel_names.txt`
- Extract unique kernel file paths (skip "blank" entries)
- For each kernel file:
  - Count total lines
  - Generate LCOV entry marking all lines as executed
  - Format: `DA:<line_number>,1` for each line

#### Step 5: Merge All Coverage
- Use Python script to merge LCOV files:
  - `cpp_coverage.info`
  - `python_coverage.info`
  - `kernel_coverage.info`
- Handle duplicate file entries (sum line hits)
- Output: `merged_coverage.info`

#### Step 6: Generate HTML Report
- Run `genhtml -o <html-output-dir> merged_coverage.info`
- Optionally upload as artifact

### 3.3 Key Implementation Details

#### 3.3.1 Kernel Coverage Generation

**LCOV Format Requirements:**
```
TN:
SF:<absolute_path_to_file>
DA:<line_number>,<execution_count>
DA:<line_number>,<execution_count>
...
end_of_record
```

**Algorithm:**
1. Parse `kernel_names.txt` line by line
2. Extract file path after `: ` (e.g., `"tt_metal/impl/dispatch/kernels/cq_prefetch.cpp"`)
3. Skip entries that are "blank"
4. For each unique file:
   - Read the source file
   - Count executable lines (skip comments, blank lines, preprocessor directives)
   - Generate `DA:<line>,1` for each executable line
   - Write LCOV record

**Edge Cases:**
- File doesn't exist → skip with warning
- Empty file → skip
- Duplicate entries → only process once
- Relative vs absolute paths → convert to absolute based on `source-dir`

#### 3.3.2 LCOV Merging

**Challenges:**
- Multiple LCOV files may have overlapping source files
- Need to sum execution counts for same line
- Need to preserve file structure

**Solution:**
- Parse each LCOV file into a data structure:
  ```python
  {
    "file_path": {
      "lines": {line_num: execution_count, ...},
      "functions": {...},
      "branches": {...}
    }
  }
  ```
- Merge by summing execution counts
- Regenerate LCOV format

#### 3.3.3 C++ Coverage Collection

**Requirements:**
- Must be built with coverage flags (`-fprofile-instr-generate -fcoverage-mapping`)
- Must set `LLVM_PROFILE_FILE` environment variable before running tests
- Example: `LLVM_PROFILE_FILE="coverage/%p.profraw"`

**Binary/Object Discovery:**
- User provides list via `cpp-objects` input
- Or auto-detect common patterns: `libtt_metal.so`, `libtt_stl.so`, test binaries
- For each binary, run: `llvm-cov export -format=lcov -instr-profile=<profdata> <binary>`

---

## 4. Usage Examples

### 4.1 Basic Usage in Workflow

```yaml
- name: Run Tests with Coverage
  env:
    TT_METAL_WATCHER_APPEND: 1
    LLVM_PROFILE_FILE: coverage/%p.profraw
  run: |
    # Run your tests here
    pytest tests/
    ./build/test/tt_metal/unit_tests

- name: Generate Coverage Report
  uses: ./.github/actions/code-coverage
  with:
    coverage-dir: coverage
    cpp-objects: |
      ./build/lib/libtt_metal.so
      ./build/lib/libtt_stl.so
      ./build/test/tt_metal/unit_tests
```

### 4.2 Python-Only Coverage

```yaml
- name: Run Python Tests with Coverage
  run: |
    coverage run -m pytest tests/

- name: Generate Coverage Report
  uses: ./.github/actions/code-coverage
  with:
    enable-cpp-coverage: false
    enable-kernel-coverage: false
```

### 4.3 Full Coverage (C++ + Python + Kernels)

```yaml
- name: Setup Coverage Environment
  run: |
    mkdir -p coverage
    export TT_METAL_WATCHER_APPEND=1
    export LLVM_PROFILE_FILE=coverage/%p.profraw

- name: Run All Tests
  run: |
    coverage run -m pytest tests/
    ./build/test/tt_metal/unit_tests

- name: Generate Coverage Report
  uses: ./.github/actions/code-coverage
  with:
    coverage-dir: coverage
    cpp-objects: |
      ./build/lib/libtt_metal.so
      ./build/lib/libtt_stl.so
      ./build/test/tt_metal/unit_tests
    html-output-dir: coverage/html

- name: Upload Coverage Report
  uses: actions/upload-artifact@v4
  with:
    name: coverage-report
    path: coverage/html
```

---

## 5. Dependencies & Prerequisites

### 5.1 Required Tools

**System Packages:**
- `llvm` (or `llvm-17`, `llvm-18`, etc.) - for `llvm-profdata` and `llvm-cov`
- `lcov` - for `genhtml` (HTML report generation)
- `python3` - for Python scripts
- `python3-coverage` - Python coverage library

**Python Packages:**
- `coverage` - Python coverage library

### 5.2 Build Requirements

**C++ Code:**
- Must be compiled with:
  - `-fprofile-instr-generate` (for profiling)
  - `-fcoverage-mapping` (for source mapping)

**CMake Integration:**
- Use `ASanCoverage` build type (already exists)
- Or add custom coverage build type

---

## 6. Testing Strategy

### 6.1 Unit Tests

- Test kernel file parsing logic
- Test LCOV generation for kernel files
- Test LCOV merging logic
- Test edge cases (missing files, empty files, etc.)

### 6.2 Integration Tests

- Test with sample `kernel_names.txt`
- Test with real C++ profraw files
- Test with Python coverage files
- Test full workflow end-to-end

### 6.3 Validation

- Verify HTML report contains all expected files
- Verify kernel files show 100% coverage
- Verify C++ and Python coverage is accurate
- Test with multiple test runs (append mode)

---

## 7. Potential Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| **Kernel files not found** | Use path resolution; warn but continue |
| **Large kernel_names.txt** | Process incrementally; use efficient parsing |
| **Missing coverage tools** | Check in setup; provide clear error messages |
| **LCOV format inconsistencies** | Use robust parser; handle edge cases |
| **Performance with many files** | Parallel processing where possible |
| **Path resolution issues** | Normalize paths; support both relative/absolute |

---

## 8. Future Enhancements

1. **Incremental Coverage**: Track coverage changes between commits
2. **Coverage Badges**: Generate coverage percentage badges
3. **Coverage Diff**: Show what changed in PRs
4. **Parallel Processing**: Speed up large coverage merges
5. **Coverage Thresholds**: Fail CI if coverage drops below threshold
6. **Multiple Repository Support**: Aggregate coverage across repos
7. **Kernel Line-Level Coverage**: If/when hardware supports it

---

## 9. Implementation Phases

### Phase 1: Core Functionality (MVP)
- ✅ Basic C++ coverage collection
- ✅ Basic Python coverage collection
- ✅ Kernel coverage generation
- ✅ LCOV merging
- ✅ HTML report generation

### Phase 2: Robustness
- ✅ Error handling
- ✅ Path resolution
- ✅ Edge case handling
- ✅ Documentation

### Phase 3: Optimization
- ✅ Performance improvements
- ✅ Parallel processing
- ✅ Caching

### Phase 4: Advanced Features
- ✅ Coverage diffs
- ✅ Thresholds
- ✅ Badges

---

## 10. Conclusion

This code coverage tool is **highly feasible** and can be implemented using existing infrastructure. The main components are:

1. **LLVM coverage tools** (already in use)
2. **Python coverage library** (standard tool)
3. **Kernel name parsing** (straightforward file parsing)
4. **LCOV format** (well-documented standard)

The implementation should be straightforward, with the main complexity in:
- Robust LCOV merging logic
- Handling edge cases in kernel file parsing
- Path resolution across different repository structures

**Recommended Approach:**
1. Start with MVP (Phase 1)
2. Test with real workflows
3. Iterate based on feedback
4. Add advanced features as needed

**Estimated Effort:**
- Phase 1 (MVP): 2-3 days
- Phase 2 (Robustness): 1-2 days
- Phase 3 (Optimization): 1-2 days
- Phase 4 (Advanced): Ongoing

---

## Appendix A: LCOV Format Reference

```
TN:                    # Test name (optional)
SF:<file_path>         # Source file
FN:<line>,<function>   # Function definition
FNDA:<count>,<function> # Function execution count
FNF:<total_functions>  # Total functions
FNH:<hit_functions>    # Hit functions
DA:<line>,<count>      # Line data (line number, execution count)
LF:<total_lines>       # Total lines
LH:<hit_lines>         # Hit lines
BRDA:<line>,<block>,<branch>,<taken> # Branch data
BRF:<total_branches>   # Total branches
BRH:<hit_branches>     # Hit branches
end_of_record          # End of file record
```

## Appendix B: Example kernel_names.txt Parsing

**Input:**
```
0: blank
1: tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
2: tt_metal/impl/dispatch/kernels/cq_dispatch.cpp
7: tt_metal/impl/dispatch/kernels/cq_prefetch.cpp  # Duplicate
```

**Output (LCOV snippet):**
```
SF:/repo/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
DA:1,1
DA:2,1
...
end_of_record
SF:/repo/tt_metal/impl/dispatch/kernels/cq_dispatch.cpp
DA:1,1
DA:2,1
...
end_of_record
```

Note: Duplicate entries are handled by only processing unique files once.
