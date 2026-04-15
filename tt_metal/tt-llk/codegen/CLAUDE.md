# LLK CodeGen

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`). **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git.

---

## Architecture-Based Orchestrators

Each target architecture has its own orchestrator and agent playbooks under `codegen/agents/{arch}/`:

| Architecture | Orchestrator | Agents |
|--------------|-------------|--------|
| **quasar** | `codegen/agents/quasar/orchestrator.md` | `codegen/agents/quasar/llk-*.md` |
| **blackhole** | `codegen/agents/blackhole/orchestrator.md` | `codegen/agents/blackhole/llk-*.md` |

When a user asks to **"generate {kernel} for {target_arch}"**, read and follow `codegen/agents/{target_arch}/orchestrator.md`.

If the target architecture is not specified, default to **quasar**.
