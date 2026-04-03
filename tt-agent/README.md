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

- "Debug why this test is failing"              → tt-debugger
- "Optimize the attention block throughput"     → tt-optimizer
- "Write tests for this new eltwise op"         → tt-tester
- "Just run this pytest on device"              → tt-run
- "I need to understand how CCL works"          → tt-learn

## Architecture

```
skills/             — how to accomplish tasks (procedural instructions)
  orchestrator/     — routes requests to the right skill
  run/              — workspace detection, build, execute (tool layer)
  optimizer/        — profile → analyze → optimize → verify (workflow)
  debugger/         — reproduce → diagnose → fix → verify (workflow)
  tester/           — design → execute → stress (workflow)
  skill-creator/    — create and validate new skills (meta)
  learn/            — research live codebases on demand (meta)

knowledge/          — stable facts and patterns
  hardware/         — silicon-stable facts (Tensix architecture, CB model)
  references/       — curated pointers to canonical code examples
  recipes/          — per-repo execution patterns (build, test, env)
    tt-metal/       — build, test, env for tt-metal
    vllm/           — build, server lifecycle, benchmark, test, env
  profiling/        — bottleneck patterns, roofline analysis, trace setup
  debugging/        — crash patterns, watcher interpretation

~/.tt-agent/notes   — shared blackboard (findings, plans, experiments)
```

## Contributing

Three ways to contribute — each independent of the others:

| Who | Contributes to | Guide |
|---|---|---|
| Repo engineer | `knowledge/recipes/<repo>/` | Add build/test/env files for your repo |
| Domain expert | `knowledge/profiling/`, `debugging/` | Add patterns and methodologies |
| Agent team | `skills/` | Build workflow logic via tt-skill-creator |

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Learn More

- [DESIGN.md](DESIGN.md) — why things are built the way they are
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add skills, recipes, and domain knowledge
