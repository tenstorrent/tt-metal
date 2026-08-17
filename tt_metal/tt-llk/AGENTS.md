# TT-LLK Codex Adapter

The canonical TT-LLK assistant instructions remain in `.claude/CLAUDE.md`.
Before doing any work in this directory or a descendant, read that file in full
and follow it as project guidance. Do not duplicate its contents here.

Apply these host mappings when reading the canonical instructions and linked
skills:

- `Claude` or `Claude Code` means Codex.
- The Claude `Skill` tool means invoke or read the matching Codex skill exposed
  under `.agents/skills`.
- `Read(...)`, `Glob`, `Grep`, `Bash`, and `Edit` mean the equivalent Codex
  filesystem, search, shell, and patch capabilities.
- The Claude `Agent` tool means spawn the matching custom Codex agent from
  `.codex/agents`, keep orchestration in the parent, and collect its result.
- A Claude `Workflow` means a Codex multi-agent fan-out followed by parent-side
  synthesis, subject to the available concurrency limit.
- A `Cursor Canvas` report means an evidence-backed Markdown report, using a
  native visualization when one is available and materially useful.
- Paths under `.claude/` are intentional canonical paths and remain valid.

Codex-specific files live under `.codex/`. The `.agents/skills` tree is only
the repository skill-discovery index; each entry links to `.codex/skills`.

The Codex skills and agents are adapters over `.claude`; update the canonical
Claude files when shared LLK behavior changes.
