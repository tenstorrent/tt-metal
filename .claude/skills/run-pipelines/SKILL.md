---
name: run-pipelines
description: Run CI/CD pipelines for pull requests. Use when asked to "run pipelines", "run standard pipelines", "run APC", "run L2 tests", "run demo tests", or any pipeline/workflow-related request.
---

# CI/CD Pipeline Runner

## Quick Aliases

| Alias | Full Name | Workflow |
|-------|-----------|----------|
| **APC** | All Post Commit | `all-post-commit-workflows.yaml` |
| **BH APC** | Blackhole Post Commit | `blackhole-post-commit.yaml` |
| **L2** | L2 Nightly Tests | `tt-metal-l2-nightly.yaml` |
| **Device Perf** | Device Perf Regressions | `perf-device-models.yaml` |
| **Model Perf** | Model Perf Tests | `perf-models.yaml` |
| **Frequent** | Frequent Model/TTNN Tests | `fast-dispatch-frequent-tests.yaml` |

## Standard Pipelines

When asked to "run standard pipelines", run ALL of these:
1. L2 Nightly (convs only): `-f additional_test_categories='conv'`
2. APC (All Post Commit)
3. BH APC (Blackhole Post Commit)
4. Device Perf
5. Model Perf
6. Frequent Tests

## Discovering Unknown Pipelines

When the user asks for a pipeline not in the aliases above:

### 1. List all available workflows
```bash
gh workflow list
```

### 2. Search workflow files by keyword
```bash
ls .github/workflows/ | grep -i "<keyword>"
```

### 3. Read workflow file to understand inputs
```bash
# Check what inputs a workflow accepts
head -100 .github/workflows/<workflow-file>.yaml
```

### 4. View workflow details
```bash
gh workflow view <workflow-name-or-file>
```

Use these commands to find and understand any pipeline the user mentions, even if it's not predefined.

## Commands

### Get current branch
```bash
git branch --show-current
```

### Trigger a pipeline
```bash
gh workflow run <workflow-file> --ref <branch> [options]
```

### Common trigger examples
```bash
# APC - All Post Commit
gh workflow run all-post-commit-workflows.yaml --ref <branch>

# BH APC - Blackhole Post Commit
gh workflow run blackhole-post-commit.yaml --ref <branch>

# L2 Nightly with specific test categories
gh workflow run tt-metal-l2-nightly.yaml --ref <branch> \
  -f additional_test_categories='conv,pool,sdxl'

# Device Perf
gh workflow run perf-device-models.yaml --ref <branch>

# Model Perf
gh workflow run perf-models.yaml --ref <branch>

# Frequent Tests
gh workflow run fast-dispatch-frequent-tests.yaml --ref <branch>
```

### Input flag types
- `-f key=value` - string/JSON input
- `--bool key=true` - boolean input
- `--ref <branch>` - target branch

## L2 Test Categories

Available categories for L2 Nightly (`additional_test_categories`):
- `conv` - Convolution tests
- `pool` - Pooling tests
- `sdxl` - Stable Diffusion XL tests
- `matmul` - Matrix multiplication tests
- `train` - Training tests
- `sdpa` - Scaled dot-product attention tests
- `bos` - BOS tests (P150b only)
- `eltwise` - Element-wise operation tests
- `transformers` - Transformer tests
- `moreh` - Moreh tests
- `data_movement` - Data movement tests
- `fused` - Fused operation tests
- `docs_examples` - Documentation example tests
- `experimental` - Experimental tests
- `misc` - Miscellaneous tests
- `ops_docs_check` - Ops docs check (Wormhole only)

Example: "run L2 sdxl and pool tests" becomes:
```bash
gh workflow run tt-metal-l2-nightly.yaml --ref <branch> \
  -f additional_test_categories='sdxl,pool'
```

## Getting Run Links

After triggering, wait a few seconds then:

```bash
# List recent runs for a workflow
gh run list --workflow=<workflow-file> --limit=3

# Get URL for most recent run
gh run list --workflow=<workflow-file> --limit=1 --json url --jq '.[0].url'

# Get multiple run URLs (after triggering multiple pipelines)
gh run list --limit=10 --json workflowName,url,createdAt --jq '.[] | "\(.workflowName): \(.url)"'
```

## Workflow

1. Get current branch: `git branch --show-current`
2. **IMPORTANT: Push local commits to remote before triggering pipelines!**
   ```bash
   git push --force-with-lease
   ```
   Pipelines run on the remote branch, not local changes. Always push first.
3. If pipeline unknown, discover it using `gh workflow list` or searching `.github/workflows/`
4. Read workflow file to understand required inputs if needed
5. Trigger pipelines with `--ref <branch>`
6. Wait ~5 seconds for runs to register
7. Collect run URLs using `gh run list`
8. Format links for PR comment

## PR Comment Format

```markdown
## CI Runs
- [Pipeline Name](url)
- [Another Pipeline](url)
```
