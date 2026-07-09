# Cursor project agents

`skills/` and the `dg-*.md` files under `commands/` are the Cursor-native
DiffusionGemma agent bundle.

They are intentionally independent from the Claude Code bundle at:

`models/experimental/diffusion_gemma/.agent/`

Shared model facts, stage gates, paths, and current performance conclusions
must be updated in both bundles. Platform orchestration is allowed to differ:

- Cursor uses the `Subagent` tool and `serial-cursor` fallback;
- Claude uses its Task/Agent and model-tier conventions;
- Cursor skills must not reference Claude `project-memory`;
- both bundles follow the `dg-*` command rule to commit and push stage-owned
  changes after a clean stage review; invoking the command authorizes both
  actions.

Shared validation scripts remain canonical under
`models/experimental/diffusion_gemma/.agent/scripts/`. Cursor skills and
commands intentionally call those scripts rather than duplicating them.

Cursor discovers project skills from `.cursor/skills/` and commands from
`.cursor/commands/`. Do not replace these directories with `.claude/` symlinks;
doing so reintroduces platform-specific tool semantics and duplicate discovery.
