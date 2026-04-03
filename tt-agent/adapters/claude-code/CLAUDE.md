# tt-agent — Claude Code Adapter

This is the Claude Code entrypoint for tt-agent. When tt-agent is installed,
this file replaces or extends the root CLAUDE.md.

## Persona

Load `tt-agent/persona.md` at the start of every session. Individual skills
may override when their domain requires it.

## Skills

See `tt-agent/skills/` for available skills. Each skill declares its layer via
`metadata.layer` in frontmatter (orchestration, workflow, tool, or meta).

## Codebase Research

When you need to understand how something works in a TT codebase — APIs, test
patterns, build steps, op structure, model architecture — use `/tt:learn`.
tt-learn follows `knowledge/references/` as starting pointers, uses deepwiki for
semantic search, and writes dated context notes to `~/.tt-agent/notes/`. These
notes are reusable across sessions. Subagents doing research should also follow
the tt-learn protocol (use references, write notes, use deepwiki).

**Check notes before researching.** If `~/.tt-agent/notes/context-<topic>.md`
exists and is recent, read it instead of re-researching.

## Knowledge

- `knowledge/hardware/` — stable silicon facts (always valid to read directly)
- `knowledge/references/` — curated pointers to canonical examples (starting points for tt-learn)
- `knowledge/recipes/` — per-repo execution patterns (build, test, env)
- `knowledge/profiling/` — profiling patterns and methodologies
- `knowledge/debugging/` — crash patterns and debug guides
- For volatile info (APIs, patterns): use `/tt:learn`, never hardcode

## Notes

Shared blackboard at `~/.tt-agent/notes`.
Always include: date, repo name, commit hash, sources read.

## Quality Bar

PCC > 0.999 vs PyTorch reference. CB sizing fits L1. Tile alignment correct.
