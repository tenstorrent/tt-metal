# ND Bisect Scripts for N300 Tests

Generic scripts to run non-deterministic (ND) bisect workflows for any N300 test from the single-card-demo-tests workflow.

## Quick Start

```bash
# Find first commit that caused ND failure, starting from a known-bad commit
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f284972f46f32bb1ad77c1e6f535c6a2e --retries 30

# Run ND bisect starting from HEAD (auto-determines retry count)
./.github/scripts/nd_bisect_n300.sh run_bert_func

# Run with pre-determined retry count from HEAD
./.github/scripts/nd_bisect_n300.sh run_resnet_func --retries 30
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
- `--target-commit <sha>`: Commit to test for failure count determination (default: same as `--bad-commit`)
- `--retries <num>`: Number of retries per commit (auto-determined if not provided)
- `--timeout <minutes>`: Timeout per test run (default: `60`)
- `--runner-label <label>`: Runner label (default: `N300`)
- `--tracy`: Enable Tracy profiling (default: `true`)
- `--no-tracy`: Disable Tracy profiling
- `--no-search`: Disable search mode (requires `--commit-range`)
- `--commit-range <good,bad>`: Commit range for bisect (required if `--no-search`)

## Examples

### Find first bad commit starting from known-bad commit (recommended)

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f284972f46f32bb1ad77c1e6f535c6a2e --retries 30
```

This will:
1. Start the search from commit `51fc518f...` (not HEAD)
2. Search backward to find a good commit (exponential backoff: 1 day, 2 days, 4 days, etc.)
3. Bisect between the good commit and `51fc518f...` to find the first bad commit

### Auto-determine retry count from specific commit

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f284972f46f32bb1ad77c1e6f535c6a2e
```

This will:
1. Test the specified commit locally until it fails to determine retry count
2. Set retries to 3× the number of attempts needed
3. Dispatch the bisect workflow starting from that commit

### Start from HEAD with pre-determined retry count

```bash
./.github/scripts/nd_bisect_n300.sh run_resnet_func --retries 30
```

### Use commit range instead of search mode

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func \
  --no-search \
  --commit-range abc123,def456 \
  --retries 30
```

### Different timeout

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f --retries 30 --timeout 90
```

### Disable Tracy

```bash
./.github/scripts/nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f --retries 30 --no-tracy
```

## How It Works

1. **Retry Count Determination** (if not provided):
   - Checks out the target commit (default: HEAD)
   - Runs the test repeatedly until it fails
   - Counts the number of attempts
   - Sets retries to 3× that count

2. **Workflow Dispatch**:
   - Dispatches the `bisect-dispatch.yaml` workflow
   - Uses search mode by default (finds failure boundary automatically)
   - Always attempts to download artifacts first (falls back to building if needed)

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
- For local testing: N300 hardware and proper environment setup
- For retry count determination: The test must be able to run locally

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
