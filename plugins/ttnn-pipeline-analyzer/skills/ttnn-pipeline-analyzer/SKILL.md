---
name: ttnn-pipeline-analyzer
description: Analyze TTNN reader/compute/writer overlap, blocking points, and pipeline bottlenecks from code artifacts.
argument-hint: "<program_factory_or_analysis_path> [--output <analysis_path>]"
---

# TTNN Pipeline Analyzer (Codex)

Use this role for performance and pipeline-behavior analysis, not for implementing new operations.

## Inputs

- Program factory path or existing operation analysis path
- Optional output path

## Workflow

1. Build CB inventory: producer, consumer, capacity, block size, lifetime.
2. Trace kernel loops and identify all `cb_wait_front` / `cb_reserve_back` blocking points.
3. Determine whether overlap is possible given CB capacity and access cadence.
4. Construct per-kernel execution timeline and idle intervals.
5. Classify bottlenecks and propose concrete optimization candidates.

## Output Contract

Write a report with:
- CB inventory table
- Blocking-point map
- Overlap verdicts (possible/impossible/conditional)
- Timeline sketch
- Ranked optimization opportunities

## Legacy Mapping

This skill is the Codex-native runtime replacement for:
- `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-pipeline-analyzer.md`


## User Controls

Before execution, read `references/user-config.md` and apply user preferences for scope, depth, and output style.
