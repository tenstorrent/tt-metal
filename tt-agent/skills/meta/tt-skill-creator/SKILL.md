---
name: tt-skill-creator
description: "Create and improve skills for the tt-agent system. Use when a TT developer wants to write a new tt-agent skill, improve an existing one, or verify a skill follows tt-agent conventions. Invokes /skill-creator for base mechanics then applies Tenstorrent-specific guidelines from tt-guidelines.md."
---

# TT Skill Creator

## Purpose

Creates high-quality skills for the tt-agent system. Wraps `/skill-creator`
(which handles generic skill mechanics: format, frontmatter, progressive load
tables, description optimization, evals) and adds TT-specific guidance on top.

## When to Invoke

Use when asked to:
- Create a new tt-agent skill (kernel, op, model, workflow, or meta)
- Improve or review an existing tt-agent skill
- Verify that a skill follows tt-agent conventions

## How This Skill Works

1. **Invoke `/skill-creator`** for all base mechanics: SKILL.md format, YAML
   frontmatter structure, progressive load table layout, description wording,
   and eval design.

2. **Load `tt-guidelines.md`** and apply TT-specific rules on top of whatever
   `/skill-creator` produces.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Any skill creation or review | `tt-guidelines.md` |
| Base skill format, frontmatter, evals | Invoke `/skill-creator` |

## Output

A skill directory containing:
- `SKILL.md` with valid YAML frontmatter (`name` matches directory, `description`
  is a single rich sentence optimized for triggering)
- Sub-files for domain knowledge, loaded progressively
- Placed in the correct layer directory under `tt-agent/skills/`
