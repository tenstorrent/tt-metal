# ND Bisect Scripts for N300 Tests

Generic scripts to run non-deterministic (ND) bisect workflows for any N300 test from the single-card-demo-tests workflow.

## Quick Start

```bash
# Find first commit that caused ND failure, starting from a known-bad commit (auto-determines retry count on CI)
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f28

# Same, but with a pre-determined retry count (faster - skips retry determination)
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f28 --retries 30

# Run ND bisect starting from HEAD
./.github/scripts/nd_bisect_n300.sh run_bert_func
```

## Usage

### Basic Usage

```bash
./.github/scripts/nd_bisect_n300.sh <test_function> [options]
```

### Required Arguments

- `test_function`: Name of the test function from `run_single_card_demo_tests.sh`
  - Examples: `run_bert_func`, `run_resnet_func`, `run_falcon7b_func`, etc.
  - Must match pattern: `run_*_func`, `run_*_perf`, or `run_*_demo`

### Options

- `--bad-commit <sha>`: Known-bad commit to start the search from (default: `HEAD`)
- `--retries <num>`: Number of retries per commit (auto-determined on CI if not provided)
- `--timeout <minutes>`: Timeout per test run (default: `60`)
- `--runner-label <label>`: Runner label (default: `N300`)
- `--tracy`: Enable Tracy profiling (default: `true`)
- `--no-tracy`: Disable Tracy profiling
- `--no-search`: Disable search mode (requires `--commit-range`)
- `--commit-range <good,bad>`: Commit range for bisect (required if `--no-search`)

## Examples

### Find first bad commit with auto retry count (recommended)

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f28
```

This will:
1. Dispatch the workflow to CI
2. CI will run the test on `51fc518f28` repeatedly until it fails
3. CI sets retry count to 3× the number of attempts needed
4. CI searches backward to find a good commit (exponential backoff: 1 day, 2 days, etc.)
5. CI bisects between the good and bad commits to find the first bad commit

### With pre-determined retry count (faster)

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f28 --retries 30
```

Skips retry determination step - useful if you already know how flaky the test is.

### Use commit range instead of search mode

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func \
  --no-search \
  --commit-range abc123,def456 \
  --retries 30
```

### Different timeout

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f28 --timeout 90
```

### Disable Tracy

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f28 --no-tracy
```

## How It Works

1. **Workflow Dispatch**:
   - Dispatches the `bisect-dispatch.yaml` workflow to CI
   - Uses search mode by default (finds failure boundary automatically)
   - Always attempts to download artifacts first (falls back to building if needed)

2. **Retry Count Determination** (on CI, if not provided):
   - CI checks out the target commit
   - CI builds/downloads artifacts for that commit
   - CI runs the test repeatedly until it fails
   - CI sets retries to 3× the number of attempts needed

3. **Search for Good Commit** (on CI):
   - CI tests commits going back in time (1 day, 2 days, 4 days, etc.)
   - CI stops when it finds a commit that passes all retries
   - This becomes the "good commit" for bisect

4. **Git Bisect** (on CI):
   - CI runs git bisect between good and bad commits
   - Each commit is tested with the determined retry count
   - CI identifies the first commit that introduced the failure

## Available Test Functions

All test functions from `tests/scripts/single_card/run_single_card_demo_tests.sh` are available. Common ones include:

- `run_bert_func`
- `run_resnet_func`
- `run_falcon7b_func`
- `run_llama3_func`
- `run_vgg_func`
- `run_distilbert_func`
- `run_mnist_func`
- `run_squeezebert_func`
- `run_efficientnet_b0_func`
- `run_stable_diffusion_func`
- And many more...

See `tests/scripts/single_card/run_single_card_demo_tests.sh` for the complete list.

## Requirements

- `gh` CLI installed and authenticated
- Git repository with the test code
- Access to dispatch GitHub Actions workflows

## Notes

- The script works both locally and in CI environments
- Artifact downloads are always attempted first to avoid rebuilding
- Search mode uses exponential backoff (1 day, 2 days, 4 days, etc.) to find failure boundaries
- ND mode is always enabled for non-deterministic failure detection

## Script Stability During Bisect

The bisect workflow is designed to ensure the test scripts remain stable even as git checks out different commits:

1. **Bisect orchestration scripts** (`tt_bisect.sh`, `test_single_commit.sh`) are copied to `./build_bisect/` before bisect starts and run from there
2. **Test command scripts** are stored in `/tmp/` (outside the repo) so git checkouts don't affect them
3. **Build scripts** (`create_venv.sh`, `build_metal.sh`) intentionally use each commit's version to ensure proper builds

This means you can safely bisect across commits that may have changed the bisect infrastructure itself.
