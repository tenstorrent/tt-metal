---
name: prompt-optimizer
description: >-
  Analyze a draft prompt, identify missing context, map to project skills and
  commands, and produce a ready-to-paste optimized prompt. Advisory-only skill:
  it improves prompts but does not execute the task.
---

# Prompt Optimizer

Use this when a user asks to improve, rewrite, or optimize a prompt.

## Trigger Signals

- "optimize this prompt"
- "rewrite my prompt"
- "help me ask this better"
- "how should I prompt Cursor for this"
- "improve prompt quality"

Do not trigger when the user clearly wants direct execution ("just do it").

## Behavior

Advisory only. Do not run tools or implement code while this skill is active.
Output:
- diagnosis of the draft prompt
- missing context checklist
- recommended commands/skills
- a full optimized prompt
- a compact optimized prompt

## Analysis Pipeline

1. **Project detection**
   - Prefer `AGENTS.md`, then `.cursor/rules/`, then project docs to infer conventions.
   - Infer stack from files like `package.json`, `go.mod`, `pyproject.toml`, `Cargo.toml`.
2. **Intent classification**
   - Feature, bug fix, refactor, research, test, review, docs, infra, design.
3. **Scope estimate**
   - TRIVIAL, LOW, MEDIUM, HIGH, EPIC based on breadth and cross-module impact.
4. **Component mapping**
   - Recommend relevant project commands and skills by intent/scope.
5. **Gap check**
   - Identify missing constraints (acceptance criteria, security, tests, boundaries).
6. **Prompt synthesis**
   - Produce copy-paste-ready prompts with clear workflow and done criteria.

## Missing Context Checklist

- Tech stack and framework
- Target files/modules
- Acceptance criteria
- Test expectations
- Security constraints
- Performance constraints
- Explicit non-goals/scope boundaries

If 3 or more critical items are missing, ask up to 3 clarification questions first.

## Output Format

1. **Prompt diagnosis**
2. **Recommended components** (command/skill + purpose)
3. **Optimized prompt (full)**
4. **Optimized prompt (quick)**
5. **Why this is better**

## Quality Bar

The optimized prompt should:
- define context and expected output shape
- include verification steps
- avoid ambiguous verbs ("improve", "fix") without acceptance criteria
- state what not to do
