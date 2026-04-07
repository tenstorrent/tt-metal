---
name: ttnn-self-reflection
description: Review a completed op-creation run, identify execution pain points, and produce actionable pipeline improvements.
argument-hint: "<operation_path>"
---

# TTNN Self Reflection (Codex)

Use this role after the operation pipeline is complete.

## Inputs

- Operation path
- Optional explicit run metadata (if multiple runs exist)

## Evidence Sources

- `{op_path}/agent_logs/*_breadcrumbs.jsonl`
- `{op_path}/agent_logs/*_execution_log.md`
- `{op_path}/REPORT.md`
- `{op_path}/op_design.md`
- `{op_path}/.tdd_state.json`
- Git history over operation + tests
- `tt_metal/third_party/tt_ops_code_gen/pipeline-improvements.md`

## Workflow

1. Build a phase timeline (start/end/duration) from logs + git.
2. Compare intended design vs implemented outcomes.
3. Identify confusion points, rework loops, and communication misses.
4. Separate systemic issues from one-off mistakes.
5. Produce concrete process improvements with priority and owner suggestions.

## Output Contract

Produce:
- `{op_path}/self_reflection.md`
- Optional append/update proposals for `pipeline-improvements.md`

## Legacy Mapping

This skill is the Codex-native runtime replacement for:
- `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-self-reflection.md`


## User Controls

Before execution, read `references/user-config.md` and apply user preferences for scope, depth, and output style.
