---
name: nuke-op
description: Remove a TTNN operation from the codebase for agent evaluation. Deletes C++ sources, kernels, tests, cleans build/bind/Python references, verifies the build still succeeds, and runs a smoke test on a surviving operation. Args = <category> <operation>.
argument-hint: "<category> <operation>"
---

# Nuke TTNN Operation

Remove a TTNN operation cleanly from the codebase so agent systems can be evaluated on recreating it from scratch.

## Input Parsing

Extract from the user's arguments:
1. **Category**: The operation category path (e.g., `normalization`, `eltwise/unary`, `reduction`, `data_movement`)
2. **Operation**: The operation directory name within that category (e.g., `softmax`, `concat`, `groupnorm`)

Both are required. If missing, ask the user.

## Name Variants

Operations sometimes have inconsistent naming between their directory name and their test/reference names. For example:
- Directory: `groupnorm` but tests named `group_norm` or `group_norm`
- Directory: `batch_norm` but tests named `batchnorm`

**The script uses the operation name as a substring match for test file discovery.** If the operation has naming variants, you MUST run the script multiple times — once per variant. The script is idempotent for the op directory deletion (skips if already gone) and additive for test deletion.

### How to detect variants

Before running, check for variants:
1. Look at the operation directory name (e.g., `groupnorm`)
2. Consider the obvious alternative: with/without underscores (`group_norm`)
3. Quick-check: `find tests/ -iname "*groupnorm*" -name "*.py" | head -5` and `find tests/ -iname "*group_norm*" -name "*.py" | head -5`
4. If both return results, you need two runs

### Example multi-run

```bash
./scripts/nuke_op.sh normalization groupnorm           # deletes op dir + tests named "groupnorm"
./scripts/nuke_op.sh normalization group_norm           # op dir already gone, catches "group_norm" tests
```

---

## Execution Flow

```
Step 1: Pre-flight checks
Step 2: Dry run
Step 3: Run nuke script (possibly multiple times for name variants)
Step 4: LLM review of modified files
Step 5: Build verification (agent, background)
Step 6: Smoke test a surviving operation (agent, background)
Step 7: Summary
```

---

## Step 1: Pre-flight Checks

1. **Operation exists**: Check that `ttnn/cpp/ttnn/operations/{category}/{operation}/` is a directory
2. **Clean git state**: Run `git status --short` — warn if there are uncommitted changes
3. **Detect name variants**: Run the variant detection described above
4. Present findings and confirm before proceeding

---

## Step 2: Dry Run

Run with `--dry-run` first to show what will be affected:

```bash
./scripts/nuke_op.sh {category} {operation} --dry-run
```

This lists:
- The op directory to delete
- Files to modify (CMake, nanobind, Python)
- All test files that will be deleted

If name variants were detected, dry-run each variant. Present the combined output.

---

## Step 3: Run Nuke Script

Execute the real deletion. For each name variant:

```bash
./scripts/nuke_op.sh {category} {variant}
```

The script:
- Backs up everything to `/tmp/nuked_ops/{category}/{variant}/`
- Deletes the operation directory (kernels, program factories, device code, pybinds, headers)
- Strips references from CMakeLists.txt, nanobind file, and Python golden/config file
- Finds and deletes all `.py` test files under `tests/` and `models/` whose filename contains the operation name
- Backs up deleted test files before removing them
- Skips YAML configs and sweep_framework files

Capture script output and verify exit code 0 for each run.

---

## Step 4: LLM Review of Modified Files

After all script runs complete, review the modified registration files:

1. **Read** `ttnn/cpp/ttnn/operations/{category}/CMakeLists.txt`
   - No dangling references to the deleted operation
   - Valid CMake syntax (no orphaned list entries)
   - Other operations in the category are untouched

2. **Read** `ttnn/cpp/ttnn/operations/{category}/{category_leaf}_nanobind.cpp`
   - No `#include` or `bind_*` calls for the deleted operation
   - File still structurally valid

3. **Read** `ttnn/ttnn/operations/{category_leaf}.py`
   - No golden functions, config aliases, or imports for the deleted operation
   - Surviving code is syntactically valid

Fix any issues with the Edit tool before proceeding.

---

## Step 5: Build Verification

Launch a background agent to rebuild and verify:

```
Agent: general-purpose (run in background)
Prompt: |
  Build the tt-metal project and report whether it succeeds.

  Run: ./build_metal.sh

  If the build FAILS:
  - Capture the FULL error output
  - Identify which file(s) failed and why
  - Report: BUILD FAILED with error details

  If the build SUCCEEDS:
  - Report: BUILD PASSED

  Do NOT fix any errors. Just report.
```

---

## Step 6: Smoke Test

After build succeeds, test a surviving operation from the same category.

1. List remaining operations in `ttnn/cpp/ttnn/operations/{category}/`
2. Find a test for one of them under `tests/`
3. Run it:

```
Agent: general-purpose (run in background)
Prompt: |
  Run a smoke test to verify the build is not broken.

  Run: scripts/tt-test.sh --dev {test_file_path} -x --timeout=120

  Report:
  - SMOKE TEST PASSED if at least 1 test passes
  - SMOKE TEST FAILED with the error if tests fail
  - SMOKE TEST HANG if output stops for >60 seconds (kill: pkill -9 -f pytest && tt-smi -r)

  Do NOT fix anything. Just report.
```

---

## Step 7: Summary

```
## Nuke Report: {category}/{operation}

### Deleted
- Operation directory: {path} ({N} files)
- Test files: {N} files deleted

### Modified (references cleaned)
- CMakeLists.txt: {N} lines removed
- Nanobind file: {N} lines removed
- Python file: {N} lines removed

### Verification
- Build: PASSED / FAILED
- Smoke test: PASSED / FAILED / SKIPPED

### Name variants used
- {variant1}, {variant2}, ...

### Backup
- /tmp/nuked_ops/{category}/{operation}/

### Restore
git checkout -- .
```

---

## If Build Fails

Common causes after a nuke:

1. **Other C++ files include deleted headers** — search:
   ```
   Grep: #include.*{operation}
   Path: ttnn/cpp/
   ```
2. **CMakeLists syntax errors** — sed may leave orphaned entries
3. **Python imports from the deleted operation** — search:
   ```
   Grep: from.*{operation}|import.*{operation}
   Path: ttnn/ttnn/
   ```

Fix the issues and re-run the build. Do not give up after one attempt.

---

## What Gets Deleted vs Kept

| Content | Action |
|---------|--------|
| `ttnn/cpp/ttnn/operations/{cat}/{op}/` (kernels, factories, device code, pybinds, headers) | DELETE |
| Test `.py` files with op name in filename (under `tests/`, `models/`) | DELETE |
| CMakeLists.txt lines referencing the op | EDIT (remove lines) |
| Nanobind registration lines | EDIT (remove lines) |
| Python golden functions, config aliases, imports | EDIT (remove blocks) |
| YAML sweep configs | KEEP |
| `sweep_framework/` test files | KEEP |
| Mixed test files (op name only in content, not filename) | KEEP (may break at runtime) |
