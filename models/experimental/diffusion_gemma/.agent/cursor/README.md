# Cursor project agents

`skills/` and the `dg-*.md` files under `commands/` are the Cursor-native
DiffusionGemma agent bundle.

## Where this lives, and why it moved

This bundle used to live in a tracked `.cursor/` directory at the **repo root** — 39 files. That
broke the rule that all DiffusionGemma work stays inside
`models/experimental/diffusion_gemma/`, so on 2026-07-30 it moved here, next to the Claude Code
bundle it mirrors.

Cursor still only discovers skills from a root `.cursor/skills/` and commands from a root
`.cursor/commands/`, so those are now **untracked symlinks** into this directory, created by:

```bash
bash models/experimental/diffusion_gemma/.agent/scripts/install_agent_bundles.sh
```

That script links both bundles (`.claude/` and `.cursor/`) and adds the link paths to
`.git/info/exclude`, which is a local file — so the repo root gains no tracked files at all. Root
`.claude/` already worked exactly this way; this just makes it reproducible and extends it. Run the
script after a fresh clone, and re-run it after adding a skill or command (it is idempotent).

Because Cursor reads the bundle *through* those links, every `.cursor/...` path written inside these
skills stays correct — do not rewrite them to `.agent/cursor/...`.

The root `.cursor/commands/` and `.cursor/rules/` files that upstream owns are untouched; only
`.cursor/skills/` and `.cursor/commands/dg-*.md` belong to DiffusionGemma. Do not point
`.cursor/skills` at `.claude/skills`: that reintroduces Claude tool vocabulary and duplicate
discovery.

## Relationship to the Claude Code bundle

The two bundles are intentionally independent in **platform orchestration** and identical in
**content**. Shared model facts, stage gates, paths, and current performance conclusions must be
updated in both:

- Cursor uses the `Subagent` tool and `serial-cursor` fallback;
- Claude uses its Task/Agent and model-tier conventions;
- Cursor skills must not reference Claude `project-memory`;
- both bundles follow the `dg-*` command rule to commit and push stage-owned
  changes after a clean stage review; invoking the command authorizes both
  actions.

Drift has run in **both** directions before, so a one-way resync loses work. When the 2026-07-30
audit compared the trees it found three platform-neutral hardenings that existed only on the Cursor
side and had to be ported back into `../skills/` first: the `stage-review` rule that a serial review
can never return `clean-pass` or unlock commit+push, the `autodebug` serial fallback plus its
verify-before-implement ordering, and the `tt-device-usage` rule never to reboot a shared host
without explicit approval. Check both directions before syncing.

## What is a symlink and what is a real copy

Thirteen skills were byte-identical between the two bundles, so under `skills/` they are now
**symlinks** into `../../skills/<name>`. That removed ~5,900 duplicated lines and makes the
"identical content" invariant structural instead of aspirational: a platform-neutral skill cannot
silently diverge.

The remaining eight are **real copies**, because they legitimately carry Cursor vocabulary
(`Subagent`, `readonly=true`, `serial-cursor`): `autodebug`, `autofix`, `autotriage`, `beautify`,
`diffusion-gemma`, `forge-functional-decoder`, `stage-review`, `tt-device-usage`. If an edit to one
of these is *not* platform-specific, make it in `../skills/` too.

Shared validation scripts remain canonical under
`models/experimental/diffusion_gemma/.agent/scripts/`. Cursor skills and
commands intentionally call those scripts rather than duplicating them.
