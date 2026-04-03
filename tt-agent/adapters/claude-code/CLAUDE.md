# tt-agent — Claude Code Adapter

This is the Claude Code entrypoint for tt-agent. When tt-agent is installed,
this file replaces or extends the root CLAUDE.md.

## Persona

Load `tt-agent/persona.md` at the start of every session. Individual skills
may override when their domain requires it.

## Skills

See `tt-agent/skills/` for available skills. Each skill declares its layer via
`metadata.layer` in frontmatter (orchestration, workflow, tool, or meta).

## Knowledge

- `tt-agent/knowledge/hardware/` — stable silicon facts
- `tt-agent/knowledge/references/` — curated pointers to canonical examples
- `tt-agent/knowledge/recipes/` — per-repo execution patterns (build, test, env)
- `tt-agent/knowledge/profiling/` — profiling patterns and methodologies
- `tt-agent/knowledge/debugging/` — crash patterns and debug guides
- For volatile info (APIs, patterns): use `tt-learn`, never hardcode

## Notes

Shared blackboard at `~/.tt-agent/notes`.
Always include: date, repo name, commit hash, sources read.

## Quality Bar

PCC > 0.999 vs PyTorch reference. CB sizing fits L1. Tile alignment correct.
