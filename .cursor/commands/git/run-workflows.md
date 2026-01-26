# Run Workflows

## Overview
Analyze branch changes to determine affected build products, trace which tests/scripts use them, identify the workflows that run those tests, then launch the workflows and post results to the PR.

## Steps
1. **Get changed files**
   - Run `git diff --name-only origin/main...HEAD` to list all changed files
   - Group files by type: C++/headers, Python, CMake, configs, etc.

2. **Decide whether to use all-post-commit-workflows.yaml**

   `all-post-commit-workflows.yaml` encapsulates these workflows:
   - `ttnn-post-commit.yaml`
   - `models-post-commit.yaml`
   - `tt-train-post-commit.yaml`
   - `run-profiler-regression.yaml`
   - `t3000-fast-tests-impl.yaml`
   - `ops-post-commit.yaml`

   **Run `all-post-commit-workflows.yaml` instead of the above individual workflows when:**
   - `conftest.py` changed (affects all pytest-based tests)
   - Multiple workflow files changed (3+)
   - Core test infrastructure changed (`tests/scripts/run_tests.py`, test utilities)
   - Changes span 3+ major areas (e.g., tt_metal + ttnn + models + tests/scripts)
   - CMake build system core files changed
   - Submodule references changed

   ```bash
   gh workflow run all-post-commit-workflows.yaml --ref $(git branch --show-current)
   ```

   **Important:** Still continue analysis for workflows NOT covered by all-post-commit, such as:
   - Galaxy tests (`galaxy-*.yaml`)
   - T3000 demo/integration/perf/unit tests (`t3000-demo-tests.yaml`, `t3000-integration-tests.yaml`, etc.)
   - Single-card demo tests (`single-card-demo-tests.yaml`)
   - Blackhole tests (`blackhole-*.yaml`)
   - Nightly tests (`*-nightly-*.yaml`)
   - Code analysis (`code-analysis.yaml`)

3. **Determine affected build products** (only if changes are targeted)
   - For C++/header changes: identify which libraries/targets they compile into
     - Check `CMakeLists.txt` files to find which target includes the changed file
     - Trace library dependencies (e.g., `tt_metal` library, `ttnn` module, device libs)
   - For Python changes: identify which modules/packages are affected
     - Check imports and module structure
   - For CMake changes: consider all dependent build products potentially affected

4. **Find tests/scripts that use affected products**
   - Search for test files that import or link against the affected products
   - Look in `tests/` directory for pytest files, C++ test binaries
   - Check `tests/scripts/` for test automation scripts that use the products
   - Search workflow files for direct invocations of affected binaries/modules

5. **Trace scripts to workflows**
   - Search `.github/workflows/` for workflows that:
     - Run the identified test files (grep for test paths, pytest markers)
     - Execute the identified scripts
     - Build and run the affected targets
   - Check workflow `run:` steps and `uses:` actions for matches
   - Note: workflows may call reusable workflows (`*-impl.yaml`) - trace those too

6. **Launch recommended workflows**
   - Use `gh workflow run <workflow.yaml> --ref <branch>` for each identified workflow
   - Wait briefly and capture run IDs with:
     ```bash
     gh run list --workflow=<workflow.yaml> --branch=<branch> --limit=1 --json databaseId,url
     ```

7. **Post PR comment with triggered runs**
   - Get PR number: `gh pr view --json number -q .number`
   - Post formatted comment with workflow links using `gh pr comment`

## Analysis Approach

When analyzing, think through this chain:

```
Changed File (e.g., tt_metal/impl/device/device.cpp)
    ↓
Build Product (e.g., libtt_metal.so, _ttnn.so Python extension)
    ↓
Test/Script that uses it (e.g., tests/tt_metal/test_device.py, test_*.cpp)
    ↓
Workflow that runs those tests (e.g., cpp-post-commit.yaml, build-and-unit-tests.yaml)
```

### Key places to search for tracing:

- **CMakeLists.txt files**: Find which targets include changed source files
- **Python imports**: `grep -r "from ttnn" tests/` or `grep -r "import tt_metal" tests/`
- **Workflow run steps**: Search for test invocations, pytest calls, binary executions
- **Workflow job names**: Often hint at what they test (e.g., `ttnn-unit-tests`, `cpp-tests`)

## Example Commands

```bash
# Get changed files
git diff --name-only origin/main...HEAD

# Find which CMake target a file belongs to
grep -r "device.cpp" --include="CMakeLists.txt"

# Find tests importing a module
grep -rl "from ttnn.operations" tests/

# Find workflows running specific tests
grep -rl "pytest tests/ttnn" .github/workflows/

# Find workflows that reference a script or test pattern
grep -rl "test_device" .github/workflows/

# Trigger workflow
gh workflow run build-and-unit-tests.yaml --ref $(git branch --show-current)

# Get latest run URL
sleep 5  # Wait for run to register
gh run list --workflow=build-and-unit-tests.yaml --branch=$(git branch --show-current) --limit=1 --json url -q '.[0].url'

# Post PR comment
gh pr comment --body "## 🚀 Triggered Workflows

Based on changes affecting **[affected products]**, launched:

| Workflow | Run |
|----------|-----|
| workflow-name | [View](url) |

<details>
<summary>Analysis</summary>

**Changed files:** X files
**Affected products:** list
**Tests/scripts using products:** list

</details>

*Triggered at: $(date -u +%Y-%m-%dT%H:%M:%SZ)*"
```

## Notes
- Skip `all-static-checks.yaml` - runs automatically on PRs
- Focus on workflows with `workflow_dispatch` that provide targeted testing
- Some workflows are wrappers calling `*-impl.yaml` - trigger the wrapper, not the impl
