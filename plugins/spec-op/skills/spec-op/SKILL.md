---
name: spec-op
description: Create a structured operation specification for a new TTNN operation. Produces op_spec.md consumed by create-op pipeline. Use 'auto' keyword for simplest defaults.
argument-hint: "[auto] <operation requirements>"
---

# Spec Op (Codex User Skill)

This skill mirrors the legacy workflow skill of the same name while standardizing execution for Codex.

## Source of Truth

- `tt_metal/third_party/tt_ops_code_gen/skills/spec-op/SKILL.md`

## Purpose

Create a structured operation specification for a new TTNN operation. Produces op_spec.md consumed by create-op pipeline. Use 'auto' keyword for simplest defaults.

## Codex Execution Contract

1. Read and follow the source skill at `tt_metal/third_party/tt_ops_code_gen/skills/spec-op/SKILL.md` for full workflow details and output contracts.
2. Apply overrides from `references/user-config.md` before executing.
3. Use Codex-native tools/scripts for execution, not Claude-specific runtime assumptions.
4. For multi-role orchestration, prefer `tt_metal/third_party/tt_ops_code_gen/scripts/start_ttnn_role_sequence.sh` and the `start_ttnn_*.sh` launchers.
5. Keep role runs sequential unless the source skill explicitly requires parallelism for that task class.

## Output

Produce the same artifact types and intent as the source skill, with Codex-compatible execution and reporting.
