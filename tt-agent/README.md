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
- "Review my changes before I merge"            → tt-code-review

## Architecture

```
skills/             — how to accomplish tasks (procedural instructions)
  orchestrator/     — routes requests to the right skill (orchestration)
  run/              — workspace detection, build, execute (tool)
  profiler/         — device profiling + bottleneck interpretation (tool)
  optimizer/        — profile → analyze → optimize → verify (workflow)
  skill-creator/    — create and validate new skills (meta)
  learn/            — research live codebases on demand (meta)
  code-review/      — strict, challenging parallel review (meta)

knowledge/          — stable facts and topic knowledge
  hardware/         — silicon-stable facts (Tensix architecture, CB model)
  matmul.md         — matmul refs + memory/compute/debug/K-blocking/numerical/structural
  ccl.md            — CCL refs + configuration/numerical/tuning
  kernels.md        — kernel references (grows with contributions)
  models.md         — model references
  operators.md      — operator references
  sharding.md       — sharding references
  recipes/          — per-repo execution patterns (build, test, env)
    tt-metal/       — build, test, env, profiler for tt-metal
    vllm/           — build, server lifecycle, benchmark for vLLM

~/.tt-agent/notes   — shared blackboard (findings, plans, experiments)
```

## Contributing

Three ways to contribute — each independent of the others:

| Who | Contributes to | Guide |
|---|---|---|
| Repo engineer | `knowledge/recipes/<repo>/` | Add build/test/env files for your repo |
| Topic contributor | `knowledge/<topic>.md` | Add references, patterns, traps per topic |
| Agent team | `skills/` | Build workflow logic via tt-skill-creator |

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Learn More

- [DESIGN.md](DESIGN.md) — why things are built the way they are
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add skills, recipes, and domain knowledge
