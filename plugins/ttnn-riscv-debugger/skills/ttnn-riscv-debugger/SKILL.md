---
name: ttnn-riscv-debugger
description: Single-hypothesis debugging workflow for TTNN kernel failures, hangs, and synchronization bugs.
argument-hint: "<journal_json_path_or_inline_json> <symptom> [--analysis <analysis_path>]"
---

# TTNN RISCV Debugger (Codex)

Use this role when kernels hang, crash, or produce wrong output and a structured debug loop is required.

## Inputs

- Debug journal JSON
- Symptom description
- Optional operation analysis file
- Optional failing test command/log snippet

## Method

Each iteration must do exactly one cycle:
1. Observe (new evidence)
2. Form one falsifiable hypothesis
3. Run one targeted experiment
4. Update confidence and next steps

## Output Contract

Return a proposal containing:
- `add_observations`
- `add_hypotheses`
- `add_experiments`
- `add_conclusions` (if supported)
- `add_next_steps`

## Guardrails

- Do not run parallel root-cause branches in one step.
- Prefer high-signal falsifiers over broad shotgun edits.
- Persist journal updates so the session can resume safely.

## Legacy Mapping

This skill is the Codex-native runtime replacement for:
- `tt_metal/third_party/tt_ops_code_gen/agents/ttnn-riscv-debugger.md`


## User Controls

Before execution, read `references/user-config.md` and apply user preferences for scope, depth, and output style.
