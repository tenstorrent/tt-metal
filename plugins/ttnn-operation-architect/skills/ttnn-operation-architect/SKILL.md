---
name: ttnn-operation-architect
description: Turn operation requirements plus reference analyses into a high-level TTNN architecture and design journal.
argument-hint: "<operation_path> <requirements> <analysis_paths...>"
---

# TTNN Operation Architect (Codex)

Use this role after reference analyses are available and before detailed engineering.

## Inputs

- Operation path: `ttnn/ttnn/operations/{op_name}`
- Structured requirements:
  - operation name, math definition, I/O tensor rules, parameters
- One or more analyzer reports

## Workflow

1. Classify operation type (compute, data_movement, fused, CCL, reduction, etc.).
2. Define high-level phase flow (for example `tilize -> compute -> untilize`).
3. Map each mechanism to concrete helpers first; use raw APIs only for explicit helper gaps.
4. Assign reader/compute/writer responsibilities.
5. Propose rough CB layout (purpose-level only, no final page counts).
6. Propose work distribution strategy and expected invariants.
7. Record key decisions with provenance in a design journal.

## Output Contract

Produce:
- `{op_path}/architecture.md`
- `{op_path}/design_journal.jsonl`

`architecture.md` must include:
- Goal and math
- Phase plan
- Helper/API mapping with file:line pointers, including explicit rationale for any `NO HELPER` fallback
- Kernel role split
- Rough CB map
- Open questions for engineering

`design_journal.jsonl` should include machine-readable entries such as:
- `helper_found`
- `api_reference`
- `mechanism_choice`
- `risk`

## Legacy Mapping

This skill is the Codex-native runtime replacement for:
- `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-operation-architect.md`


## User Controls

Before execution, read `references/user-config.md` and apply user preferences for scope, depth, and output style.
