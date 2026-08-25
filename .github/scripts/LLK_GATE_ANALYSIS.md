# LLK PR Gate Trigger Analysis

This document explains how to calculate what percentage of commits to main trigger the LLK PR gate.

## Background

The LLK PR gate runs conditionally: it only executes when commits modify LLK-specific code. This means not every merge to main triggers the expensive LLK test suite.

Understanding this ratio is important for:
- Budgeting CI resources (how many LLK gates actually run per month?)
- Arguing for increased test scope (if only 10% of merges trigger the gate, we can afford more thorough testing)
- Capacity planning for dedicated runners

## What Triggers the LLK Gate

The LLK PR gate runs when a commit modifies any of these paths:

| Trigger | File Pattern |
|---------|------|
| **llk-wormhole-changed** | `tt_metal/tt-llk/tt_llk_wormhole_b0/**` or `tt_metal/hw/ckernels/wormhole_b0/**` |
| **llk-blackhole-changed** | `tt_metal/tt-llk/tt_llk_blackhole/**` or `tt_metal/hw/ckernels/blackhole/**` |
| **llk-common-changed** | `tt_metal/tt-llk/common/**` |
| **llk-sfpi-changed** | `tt_metal/sfpi-info.sh` or `tt_metal/sfpi-version` |
| **llk-unit-tests-changed** | `tests/tt_metal/tt_metal/llk/**` |
| **llk-ci-changed** | LLK CI files (`.github/workflows/llk-*.yaml`, `tests/pipeline_reorg/llk_*.yaml`, etc.) |

These patterns are defined in [.github/scripts/utils/find-changed-files.sh](.github/scripts/utils/find-changed-files.sh).

## How to Run the Analysis

```bash
# Analyze July 2026
.github/scripts/analyze-llk-gate-triggers.py --since 2026-07-01 --until 2026-08-01

# Analyze the past 30 days
.github/scripts/analyze-llk-gate-triggers.py --since "30 days ago"

# Show which files triggered each gate run
.github/scripts/analyze-llk-gate-triggers.py --since 2026-07-01 --until 2026-08-01 --verbose
```

## Historical Results

### Two Metrics: Direct Trigger vs. Cascade Effect

There are two important metrics for understanding LLK test execution frequency:

1. **Direct Trigger**: How many commits directly modified LLK code?
2. **Cascade Effect**: How many times does the LLK suite actually run, considering that queued LLK commits cause all subsequent commits to inherit their changes?

#### Analysis by Month

| Month | Total Commits | Direct Triggers | Cascade Runs |
|-------|:-------------:|:----------------:|:------------:|
| May 2026   | — | — | — |
| June 2026  | — | — | — |
| **July 2026**  | **165** | **17 (10.3%)** | **165 (100%)** |

**Notes:**
- May and June data unavailable (commits not present in current main history)
- July 2026 covers July 8–14 (commit history range on main)

### Key Findings

#### Direct Trigger Metric
- Only **17 out of 165 commits (10.3%)** directly modified LLK-specific code
- The remaining **148 commits (89.7%)** modified code in ops, models, ttnn, or other areas
- These non-LLK commits would NOT trigger the LLK gate if run independently

#### Cascade Effect Metric
- Due to merge queue cascading, **165 out of 165 commits (100%)** end up running the LLK test suite
- **Average**: Each LLK commit causes **9.7** subsequent commits to inherit and run its LLK changes
- **Multiplier**: The cascade effect magnifies the trigger count by **9.7x**

### What This Means

When an LLK commit is queued in the merge queue:
1. It triggers the LLK tests immediately
2. All subsequent commits inherit those LLK changes **while waiting for the LLK PR to land**
3. Each of those commits also runs the full LLK test suite
4. This continues until the next LLK commit lands and replaces the inherited changes

**Result:** In July, even though only 10.3% of commits directly touched LLK code, 100% of commits ended up running the LLK suite due to this cascade behavior.

### Budget Impact

The cascade effect is the real metric for resource budgeting:
- **Not just 17 LLK test runs per month**, but effectively **165 runs**
- This represents significant CI resource utilization
- Understanding this cascade behavior is essential for justifying LLK test capacity and scope

## Usage for Budget Arguments

### Direct Trigger Argument
When requesting more LLK test capacity:

> "Only 10% of merges directly modify LLK code.
> We can afford more thorough testing without burdening non-LLK commits."

### Cascade Effect Argument
When understanding the true resource impact:

> "Due to merge queue cascading, LLK commits cause 100% of subsequent commits
> to run the LLK test suite while queued. This 10x multiplier means we're
> investing significant CI resources. We should ensure that investment is
> well-scoped and effective."

## Related Files

- [.github/workflows/merge-gate.yaml](.github/workflows/merge-gate.yaml) - Merge gate workflow with LLK job conditions
- [tests/pipeline_reorg/llk_merge_gate_tests.yaml](../../tests/pipeline_reorg/llk_merge_gate_tests.yaml) - LLK test configuration
- [.github/actions/find-changed-files/action.yml](.github/actions/find-changed-files/action.yml) - Action that determines which gates run
