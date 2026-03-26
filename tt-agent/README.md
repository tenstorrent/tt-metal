# tt-agent

Agentic tooling for Tenstorrent hardware development. Enables AI agents (Claude Code,
Codex, and others) to autonomously design, write, test, profile, debug, and optimize
TT kernels, operators, and models.

## Prerequisites

- `tt-metal` cloned and built
- [tt-device-mcp](https://github.com/tenstorrent/tt-device-mcp) installed
- deepwiki-mcp configured in your agent environment

## Install (Claude Code)

Add to your root `CLAUDE.md` or ensure it references `tt-agent/adapters/claude-code/CLAUDE.md`.

## Quick Start

- "Design a new eltwise op for Wormhole B0"  → tt-designer
- "Optimize the attention block in my model"  → tt-iterator
- "This CI test is failing, fix it"           → tt-ci-fixer
- "Review this kernel for correctness"        → tt-code-review

## Learn More

- [DESIGN.md](DESIGN.md) — why things are built the way they are
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add skills and adapters
- [Spec](../docs/superpowers/specs/2026-03-26-tt-agent-design.md) — full design spec
