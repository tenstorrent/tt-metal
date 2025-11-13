# External Contributor Testing Guide

This guide explains how to run workflows to verify that external contributor changes haven't broken anything in the tt-metal repository.

## Overview

Testing external contributions requires a two-step process:

1. **Mirror the fork branch** - Create a local branch in the tt-metal repository from the contributor's fork
2. **Run appropriate test workflows** - Execute testing workflows based on what code was changed

---

## Step 1: Mirror Fork Branch

Before running any tests, you must first create a mirror branch from the contributor's fork.

### Running the Mirror Workflow

1. Navigate to **Actions** → **Mirror Fork Branch to Origin**
2. Click **Run workflow**
3. **⚠️ Select the branch**: Use `fvranicTT/fix-mirroring-timeout`
   - Click the "Use workflow from" dropdown
   - Select `fvranicTT/fix-mirroring-timeout` (NOT main - main branch has known issues)
4. Provide the required input:
   - **Source**: `<fork_owner>:<branch_name>`
     - **💡 How to find it**: Go to the PR page and look at the top where it says "User wants to merge X commits into tenstorrent:main from `fork_owner:branch_name`"
     - Click the **two squares icon** (📋) next to the branch name to copy it to clipboard
     - Example from [PR #32122](https://github.com/tenstorrent/tt-metal/pull/32122): `jasondavies:where-loadmacro`
   - **Target branch** (optional): Leave empty to auto-generate `mirror/<fork_owner>/<branch_name>`

### What Happens Next

The workflow will:

- Fetch the branch from the contributor's fork
- Create/update a mirror branch in the tt-metal repository (e.g., `mirror/johndoe/fix-memory-leak`)
- Post a comment on the associated PR with the mirror branch name

### ⚠️ IMPORTANT: Save the Mirror Branch Name

**The mirror branch name (starts with `mirror/...`) is critical!** You will need this exact branch name to run all test workflows in Step 2.

Write it down or keep the PR comment visible - you'll be selecting this branch from the branch dropdown when running each test workflow.

---

## Step 2: Run Testing Workflows

Based on what architecture code was changed in the PR, you need to run different combinations of test workflows.

### Determining Which Workflows to Run

First, identify which architecture(s) are affected by the changes:

#### How to Check for Architecture-Specific Changes

1. **Go to the PR** and click the **"Files changed"** tab
2. **Look for changes** in these directories:

   **A) Under `tt_metal/hw/ckernels/`:**
   - If files under `wormhole_b0/` are changed → **Wormhole affected**
   - If files under `blackhole/` are changed → **Blackhole affected**
   - If files under both directories are changed → **Both architectures affected**

   **B) Under `tt_metal/third_party/tt_llk/`:**
   - Look at the changed file paths in the file list
   - If you see `wormhole` or `wormhole_b0` in the paths → **Wormhole affected**
   - If you see `blackhole` in the paths → **Blackhole affected**
   - If you see both → **Both architectures affected**

**Example**: In [PR #32122](https://github.com/tenstorrent/tt-metal/pull/32122), if you see changes in `tt_metal/hw/ckernels/wormhole_b0/` or `tt_metal/third_party/tt_llk/...wormhole_b0/...`, then Wormhole tests are needed.

#### Decision Matrix

| Changes Include | Architectures Affected |
|----------------|----------------------|
| Files in `tt_metal/hw/ckernels/wormhole_b0/` | Wormhole only |
| Files in `tt_metal/hw/ckernels/blackhole/` | Blackhole only |
| Files in `tt_metal/third_party/tt_llk/` with `wormhole` in paths | Wormhole only |
| Files in `tt_metal/third_party/tt_llk/` with `blackhole` in paths | Blackhole only |
| Files in both `wormhole_b0/` and `blackhole/` directories | Both architectures |
| Common/shared code or unsure | Both architectures (to be safe) |

### Workflow Matrix

Based on the affected architecture(s), run the following workflows:

#### For Wormhole Changes

Run these **2 workflows**:

1. **All Post-Commit Workflows**
   - Tests: TTNN, Models, C++, TT-CNN, Profiler, T3000, Fabric

2. **TT-Metal L2 Nightly** (with C++ tests enabled)
   - Must enable: `Run APC C++ tests`
   - Tests: L2 nightly tests, C++ unit tests, tutorials

#### For Blackhole Changes

Run these **2 workflows**:

1. **Blackhole Post-Commit**
   - Tests: C++, Models, TTNN, TT-CNN, Demo tests, 20-core tests

2. **TT-Metal L2 Nightly** (with C++ tests enabled)
   - Must enable: `Run APC C++ tests`
   - Tests: L2 nightly tests, C++ unit tests, tutorials

#### For Both Wormhole and Blackhole Changes

Run **all 3 workflows**:

1. **All Post-Commit Workflows** (for Wormhole)
2. **Blackhole Post-Commit** (for Blackhole)
3. **TT-Metal L2 Nightly** (with C++ tests enabled) (for both)

#### 💡 Pro Tip: Run L2 Nightly First

**TT-Metal L2 Nightly with C++ tests will fail faster** if there are issues (~30-60 minutes vs 2-4 hours for post-commit workflows). Consider running it first:

1. Start with **TT-Metal L2 Nightly** (with C++ tests enabled)
2. Wait for it to complete or fail
3. Only if it passes, then run the appropriate **Post-Commit workflows**

This saves time by catching errors early before running longer test suites.

---

## Step 3: Running the Test Workflows

All workflows are run from the GitHub Actions tab in your browser. For each required workflow:

### How to Run a Workflow

1. **Open the Actions tab** in the tt-metal GitHub repository
2. **Find the workflow** in the left sidebar:
   - "All post-commit tests" (for Wormhole)
   - "Blackhole post-commit tests" (for Blackhole)
   - "Nightly tt-metal L2 tests" (for L2 Nightly)
3. **Click "Run workflow"** button (top right)
4. **⚠️ CRITICAL: Select the branch**
   - Click the "Use workflow from" dropdown
   - **Select the mirror branch** from Step 1 (e.g., `mirror/johndoe/fix-memory-leak`)
   - ❌ DO NOT use `main` or any other branch!
5. **Configure inputs** (if needed):
   - **For L2 Nightly ONLY**: Check the box for `Run APC C++ tests`
   - All other workflows: Leave defaults as-is
6. **Click "Run workflow"** (green button)

### Example: Running All Post-Commit Workflows

```text
1. Go to: Actions tab → "All post-commit tests"
2. Click: "Run workflow" button
3. Select branch: mirror/johndoe/fix-memory-leak
4. Leave all inputs as default
5. Click: "Run workflow"
```

### Example: Running L2 Nightly with C++ Tests

```text
1. Go to: Actions tab → "Nightly tt-metal L2 tests"
2. Click: "Run workflow" button
3. Select branch: mirror/johndoe/fix-memory-leak
4. Check: ✅ Run APC C++ tests
5. Leave other inputs as default
6. Click: "Run workflow"
```

---

## Step 4: Monitoring Test Results

After triggering the workflows, monitor them from the browser:

1. **Watch progress** in the Actions tab:
   - Each workflow run will appear at the top of the Actions list
   - Click on a run to see detailed progress and logs
   - Workflows typically take 2-6 hours to complete

2. **Check for completion**:
   - ✅ Green checkmark = All tests passed
   - ❌ Red X = Some tests failed
   - 🟡 Yellow dot = Tests still running

3. **Review failures** (if any):
   - Click into the failed workflow run
   - Examine the failed job logs
   - Determine if the failure is related to the contributor's changes
   - Request fixes from the contributor if their changes caused the failure

---

## Quick Reference Cheat Sheet

```text
1. MIRROR THE FORK (Actions tab → "Mirror Fork Branch to Origin"):
   Use workflow from: fvranicTT/fix-mirroring-timeout (NOT main!)
   Input: <fork_owner>:<branch_name>
   Result: Creates mirror/<fork_owner>/<branch_name>
   ⚠️  SAVE THIS BRANCH NAME - you'll need it for all test workflows!

2. RUN TEST WORKFLOWS (Actions tab → Select workflow → Run workflow):
   ⚠️  ALWAYS select the mirror branch in "Use workflow from" dropdown!

   For Wormhole changes:
   → Run "All post-commit tests" on mirror branch
   → Run "Nightly tt-metal L2 tests" on mirror branch (check run_APC_cpp_tests)

   For Blackhole changes:
   → Run "Blackhole post-commit tests" on mirror branch
   → Run "Nightly tt-metal L2 tests" on mirror branch (check "Run APC C++ tests")

   For Both architectures:
   → Run "All post-commit tests" on mirror branch
   → Run "Blackhole post-commit tests" on mirror branch
   → Run "Nightly tt-metal L2 tests" on mirror branch (check "Run APC C++ tests")

3. MONITOR RESULTS (Actions tab):
   → Watch workflow runs until completion (2-4 hours)
   → Review any failures and determine if caused by contributor changes
```

---

## Common Issues and Tips

### Issue: Mirror branch not found in workflow dropdown

- **Solution**: Wait a few seconds and refresh the page. The branch may need time to sync.

### Issue: Workflow fails immediately

- **Solution**: Check if the mirror branch is up to date with the contributor's fork. You may need to re-run the mirror workflow.

### Issue: Tests pass locally but fail in CI

- **Solution**: Ensure the contributor's branch is rebased on the latest `main` before mirroring.

### Tip: Partial Testing

If you're confident about the scope of changes, you can:

- Run only the affected architecture's post-commit workflow
- Skip L2 Nightly for minor changes (not recommended for external contributions)

---

## Additional Resources

- [Contributing Guide](/CONTRIBUTING.md)
- [CI/CD Best Practices](/contributing/BestPractices.md)

---

**Last Updated**: November 2025
