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

## Example Results

From July 2026 analysis:

```
Total commits:     165
Triggered LLK gate: 17
Percentage:        10.3%
```

This means only **10.3%** of commits that landed on main in July actually triggered the LLK PR gate.
The other 89.7% modified code in ops, models, ttnn, or other areas.

## Usage for Budget Arguments

When requesting more LLK test capacity or arguing for increased test scope:

> "In the past month, only X% of merges to main triggered the LLK PR gate.
> This means we can afford more comprehensive testing on those X% of merges
> without impacting overall CI throughput or merge latency."

## Related Files

- [.github/workflows/merge-gate.yaml](.github/workflows/merge-gate.yaml) - Merge gate workflow with LLK job conditions
- [tests/pipeline_reorg/llk_merge_gate_tests.yaml](../../tests/pipeline_reorg/llk_merge_gate_tests.yaml) - LLK test configuration
- [.github/actions/find-changed-files/action.yml](.github/actions/find-changed-files/action.yml) - Action that determines which gates run
