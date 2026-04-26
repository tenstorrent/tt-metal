---
name: search-first
description: >-
  Research-before-building workflow. Check existing project code, existing
  dependencies, and proven external solutions before implementing new custom
  logic.
---

# Search First

Use this skill before writing net-new utilities or abstractions.

## Core Principle

Prefer:
1. Existing code in this repo
2. Existing dependency capabilities
3. Well-maintained external packages
4. Custom implementation only when needed

## Workflow

1. **Local discovery first**
   - Search repo for similar logic/tests/patterns.
2. **Dependency capability check**
   - Verify whether currently installed libraries already solve the need.
3. **External options**
   - Identify top candidates and compare maintenance, fit, and complexity.
4. **Decision**
   - Adopt, wrap, compose, or build custom.
5. **Implement minimally**
   - Keep custom code thin when adopting external tools.

## Decision Matrix

- Exact maintained fit -> adopt
- Partial fit with strong foundation -> wrap/extend
- Several narrow tools compose well -> compose
- No suitable option -> build custom

## Anti-Patterns

- Rebuilding common functionality without checking existing options
- Adding large dependencies for tiny needs
- Ignoring established project conventions while introducing new tools
