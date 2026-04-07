---
name: ttnn-operation-analyzer
description: Analyze a TTNN operation program factory and kernels in depth, then write a reusable architecture analysis for downstream design work.
argument-hint: "<program_factory_path> [--role <input_stage|compute_core|output_stage>] [--output <analysis_path>]"
---

# TTNN Operation Analyzer (Codex)

Use this role to produce a deep implementation analysis of an existing TTNN operation.

## Inputs

- Program factory path (required)
- Optional role focus:
  - `input_stage`
  - `compute_core`
  - `output_stage`
- Optional output path; default: `<factory_dir>/<factory_stem>_analysis.md`

## Workflow

1. Read the program factory and enumerate all kernel sources and key helpers/APIs.
2. Identify work-unit granularity, loop structure, and per-core work distribution.
3. Trace end-to-end dataflow (reader, compute, writer), CB ownership, and synchronization points.
4. Document tensor requirements (layout, memory layout, dtypes, sharding assumptions).
5. Extract compile-time/runtime argument mapping and index calculations.
6. If a role focus is provided, prioritize that section and de-emphasize unrelated detail.

## Output Contract

Write a markdown report containing:
- Overview and supported variants
- Work-unit definition
- Dataflow table
- Circular buffer table
- Core distribution strategy
- Kernel responsibilities
- Argument mapping (compile-time/runtime)
- Risks or unknowns that may affect derivative implementation

## Quality Bar

- Include file:line references for non-obvious claims.
- Prefer helper-level understanding first, then raw API details.
- Flag assumptions explicitly.

## Legacy Mapping

This skill is the Codex-native runtime replacement for:
- `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-operation-analyzer.md`


## User Controls

Before execution, read `references/user-config.md` and apply user preferences for scope, depth, and output style.
