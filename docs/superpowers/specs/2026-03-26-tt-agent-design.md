# tt-agent Design Spec

**Date**: 2026-03-26
**Status**: Draft

---

## What Is tt-agent

tt-agent is a platform-agnostic agentic tooling system for developing on Tenstorrent hardware and the TT software stack. It packages skills, knowledge, and platform adapters into a single repository that enables AI agents (Claude Code, Codex, and others) to autonomously write, test, profile, debug, and optimize TT kernels, operators, and models.

It is inspired by Karpathy's autoresearch pattern: rather than one-shot code generation, agents iterate — hypothesize, implement, run on hardware, analyze results, and converge on optimal solutions.

---

## Core Principles

### Skills vs Knowledge vs Notes

These are three distinct things that must not be conflated:

- **Skills** — how to accomplish a task. Procedural instructions for the agent. Examples: how to profile a kernel, how to iterate toward a perf target, how to review code with TT standards.
- **Knowledge** — stable, hardware-level facts that don't change between software releases. Tensix core model, NOC topology, tile granularity, CB producer/consumer pattern. Written once, rarely updated. Plus curated references: pointers to canonical examples and key documents in the codebase, organized by topic, for developers and agents to use as a starting point.
- **Notes** — shared, evolving findings produced during work. Context briefs from tt-learn, experiment logs from tt-iterator, profiler findings, plans. Written by agents and humans alike; shared across sessions and team members.

The key insight: the TT software stack evolves rapidly. API signatures, op implementations, sharding patterns — these change with every PR. **Volatile knowledge is never written down.** Instead, the `tt-learn` skill researches the live codebase on demand and writes its findings to `notes/`. Knowledge files in `knowledge/hardware/` are reserved for hardware invariants that are stable across software releases. Knowledge files in `knowledge/references/` are curated pointers to canonical examples — they change slowly (examples don't move as often as implementations) and serve as the onboarding reading list for new developers and the starting point for tt-learn searches.

### Platform-Agnostic Content

tt-agent is not a Claude Code plugin. It owns the full stack: skills and knowledge are authored once in platform-agnostic markdown, and platform adapters (Claude Code, Codex, and others) package that content into the format each platform expects. The same content ships to all platforms.

### Shared Blackboard (notes/)

Agents and developers share a common workspace called `notes/` at the project root. This is a blackboard: any agent can write findings, any agent in any future session can read them, and developers can review, edit, and annotate them. A note from a profiling run last week is still valid context for an optimization session today. Notes include: tt-learn context briefs, experiment logs, profiling results, per-task plans and status.

---

## Repository Structure

tt-agent lives inside `tt-metal` for now. The skills reference tt-metal paths (API headers, programming examples, operator source), so co-location is correct — the agent needs tt-metal regardless. When the time comes, `git subtree split` on `tt-agent/` yields a clean independent repo with full history.

```
tt-metal/
  CLAUDE.md                          # pointer only → tt-agent/adapters/claude-code/CLAUDE.md
  notes/                             # shared blackboard: findings, contexts, experiments
  tt-agent/
    tt-agent.yaml                    # manifest: name, version, mcp dependencies
    README.md                        # what this is, how to install, quick start
    DESIGN.md                        # why things are built the way they are
    CONTRIBUTING.md                  # how to add skills, knowledge, adapters
    adapters/
      claude-code/
        CLAUDE.md                    # real Claude Code entrypoint
      codex/
        AGENTS.md
    skills/
      orchestration/
        tt-orchestrator/             # routes, plans, decomposes, invokes workflows
      workflows/
        tt-iterator/                 # iterate toward a perf/correctness goal
        tt-ci-fixer/                 # replicate CI failure, iterate to fix
        tt-bisect/                   # find commit that introduced regression
      tools/
        tt-device/                   # build, run on hardware
        tt-profiler/                 # profile, expose bottlenecks
        tt-tester/                   # test quality, expose weaknesses
        tt-debugger/                 # debug kernels, hangs
        tt-designer/                 # design phase: TT brainstorm + perf estimation + data-movement planning
        tt-code-review/              # review with TT standards
      meta/
        tt-skill-creator/            # help write new TT skills
        tt-learn/                    # research codebase via deepwiki-mcp, write findings to notes/
    knowledge/
      hardware/                      # stable hardware invariants (silicon facts)
        tensix-architecture.md       # cores, FPU, SFPU, L1, NOC topology
        circular-buffer-model.md     # CB producer/consumer coordination primitive
        quirks.md                    # hardware gotchas: no malloc, 32-bit, tile sizes
      references/                    # curated pointers to canonical examples, per topic
        operators.md                 # canonical op examples, program factory patterns
        kernels.md                   # reader/compute/writer, kernel fusion examples
        sharding.md                  # sharding strategies, tensor sharding deep dive
        matmul.md                    # matmul perf analysis, matrix engine docs
        ccl.md                       # ethernet guide, CCL op patterns
        models.md                    # model bringup examples, inference patterns
```

---

## Skill Hierarchy

Skills are organized into four layers, each with a distinct character.

### Orchestration layer

**`tt-orchestrator`** — the entry point for any request. Analyzes the request, determines scope, decomposes into sub-tasks, routes to the appropriate workflow or tool skill, verifies results, and iterates. Knows when to invoke `tt-iterator` (optimization), `tt-ci-fixer` (failure), `tt-bisect` (regression), or dispatch directly to tool skills for narrow tasks.

### Workflow layer

Workflow skills own an autonomous loop. Each is given a concrete goal, runs until it converges or escalates, and leaves a complete record in `notes/`. They are thin — the loop structure is simple — but high-impact because they operate for extended periods delegating to the tool layer.

All three workflow skills share the same base loop:
1. Define goal and success criteria
2. Form hypothesis
3. Delegate implementation to appropriate tool skill
4. Run on hardware via tt-device
5. Analyze results
6. Log experiment to notes/
7. Converged? → report best result. No → form next hypothesis → go to 3.

What differs between them is triggering context and convergence criteria:

| Skill | Triggered by | Success when |
|---|---|---|
| `tt-iterator` | optimization goal | perf target + PCC met |
| `tt-ci-fixer` | failing CI run | all tests green |
| `tt-bisect` | regression report | first bad commit identified |

`tt-iterator` is the primary autoresearch skill. It uses hardware roofline analysis learned via `tt-learn` (starting from `knowledge/references/matmul.md` as a guide to canonical sources) to reason about why a result is suboptimal and what to try next. The search is guided by understanding (compute-bound vs memory-bound vs NOC-bound), not random.

### Tool layer

Concrete, single-purpose skills. Each does one thing well. Invoked by workflow skills and by tt-orchestrator directly for narrow tasks.

- **`tt-device`** — build tt-metal, run programs and tests on TT hardware (wraps tt-device-mcp)
- **`tt-profiler`** — invoke Tracy or device profiler, interpret output, identify bottlenecks
- **`tt-tester`** — write and run tests, evaluate coverage, expose quality issues
- **`tt-debugger`** — debug device kernels, trace hangs, interpret RISC-V core state
- **`tt-designer`** — the design phase in one skill. Wraps `/superpowers:brainstorm` with TT-specific grilling (tile alignment, L1 budget, arithmetic intensity, PCC targets), performance estimation (roofline analysis, bandwidth math, CCL latency), and data-movement planning (sharding across cores/chips, CCL strategy selection, compute-communication overlap). Pairs with `tt-code-review` as bookends around implementation — designer before writing code, code-review after. Sub-files: `brainstorm.md`, `estimator.md`, `data-movement.md`.
- **`tt-code-review`** — review code with TT-specific standards and good practices: CB sizing vs L1, tile alignment, NOC conventions, program cache hash, sharding validity, PCC test coverage

### Meta layer

- **`tt-skill-creator`** — helps TT developers write new skills for tt-agent. Builds on top of the `/skill-creator` superpowers skill (which handles generic skill mechanics: format, frontmatter, progressive load tables, evals) and adds TT-specific guidance on top: what makes a good TT skill, how to decide what belongs in a skill vs `knowledge/` vs left to `tt-learn`, the "point to code not inline APIs" principle, how to structure references, TT coding standards and hardware abstraction levels to respect, and team conventions for the tt-agent system.
- **`tt-learn`** — research a topic in the live codebase via deepwiki-mcp, write a dated context brief to `notes/`. Cross-cutting utility invoked by any skill at any time. Consults `knowledge/references/` as starting points before doing a broader search.

---

## Knowledge: Hardware Invariants and References

`knowledge/hardware/` contains three files of stable, silicon-derived facts: the Tensix architecture (five RISC-V cores, FPU, SFPU, L1, NOC), the circular buffer coordination model, and hardware quirks (no dynamic allocation, 32-bit RISC-V, tile size constants). These are written once and updated only when the hardware changes.

`knowledge/references/` contains curated pointers to canonical examples in the codebase, one file per topic. Each entry is a path and a one-line description — no content inlined. References are the onboarding reading list for new developers and the starting point for `tt-learn` searches. They change slowly (a good operator example stays a good operator example even as its internals evolve) and are easy to maintain when they do go stale (update a path, not a content block).

Example entry in `knowledge/references/operators.md`:
```
## Canonical simple eltwise op
ttnn/cpp/ttnn/operations/eltwise/binary/device/
Best example of the full device operation pattern: validate, compute_output_specs,
program factory, runtime args.
```

---

## tt-learn and the Notes Blackboard

`tt-learn` is the bridge between agents and the live codebase. When a workflow or tool skill needs to understand something volatile (current matmul implementation, CCL patterns, sharding conventions in use), it invokes `tt-learn` with a topic. `tt-learn` uses **deepwiki-mcp** to query the codebase — reading source files, tech reports, test files, and programming examples — then writes a structured context brief to `notes/`. `knowledge/references/` files serve as the starting point, steering deepwiki-mcp toward canonical sources before it does a broader search.

Example invocation: `tt-learn("matmul sharding strategies")` consults `knowledge/references/matmul.md` for starting points, queries deepwiki-mcp across `ttnn/cpp/ttnn/operations/matmul/`, `tech_reports/tensor_sharding/`, and related tests, then writes `notes/context-matmul.md`.

Context brief format:
```markdown
# Context: [topic]
Generated: 2026-03-26
tt-metal commit: abc1234
Sources read: [list of files]

[structured findings]
```

Notes persist across sessions and are shared among team members. They are explicitly dated and include the tt-metal commit at time of generation so readers can judge freshness. An agent encountering an existing note checks the date and commit before deciding whether to use it or regenerate.

The notes/ directory is the team's shared, growing, codebase-derived knowledge base — not a replacement for documentation, but a practical supplement that's always grounded in the actual code.

---

## MCP Dependencies

tt-agent declares two MCP dependencies:

- **[tt-device-mcp](https://github.com/tenstorrent/tt-device-mcp)** — hardware execution: running kernels and tests on actual TT devices, returning structured results. Used by `tt-device`.
- **deepwiki-mcp** — codebase research: semantic search and reading across source files, tech reports, and documentation. Used by `tt-learn`.

Everything else (build, profile, file I/O) is achieved via CLI and Bash.

Declared in `tt-agent.yaml`:
```yaml
name: tt-agent
version: 0.1.0
description: Agentic tooling for Tenstorrent hardware development
mcps:
  - name: tt-device
    source: https://github.com/tenstorrent/tt-device-mcp
  - name: deepwiki
    source: deepwiki-mcp
```

---

## Platform Adapters

Each adapter in `adapters/` packages the tt-agent content for a specific agentic platform.

**`adapters/claude-code/CLAUDE.md`** — the real Claude Code entrypoint (the root `CLAUDE.md` is a one-line pointer to this file). Contains: what tt-agent is, the skill layer structure, how to invoke skills, key conventions.

**`adapters/codex/AGENTS.md`** — equivalent entrypoint for Codex.

Adding a new platform means adding a new subdirectory under `adapters/` and writing one entrypoint file that references the same `skills/` and `knowledge/` content. The content itself does not change.

---

## Meta Documentation

Three documents at the tt-agent root serve distinct audiences:

**`README.md`** — for someone who just discovered the repo. What it is, prerequisites, install instructions per platform, two quick-start examples. Links to DESIGN.md and CONTRIBUTING.md.

**`DESIGN.md`** — for someone who wants to understand the decisions. Every non-obvious architectural choice is recorded here with rationale and date. This document preserves the intent behind the system so future agents and developers can continue in the same direction without re-litigating settled questions. Key decisions documented: co-location in tt-metal, own-the-stack vs superpowers, skills/knowledge/notes split, volatile knowledge via tt-learn + deepwiki-mcp, notes/ as shared blackboard, workflow layer as thin base loop, two MCP dependencies (tt-device-mcp + deepwiki-mcp), tt-designer as unified design-phase skill, tt-learn in meta/ as cross-cutting utility, extraction path.

**`CONTRIBUTING.md`** — for someone who wants to extend the system. How to write a new skill (SKILL.md format, progressive load table), what belongs in knowledge/ vs learned via tt-learn, how to add a platform adapter, the "point to code not inline APIs" rule and why, PR conventions.

---

## Distribution Path

**Now**: git clone `tt-metal`, the skills are immediately available. Platform adapter files point Claude Code / Codex at the right entrypoint.

**Near-term**: install script (`adapters/install.sh`) that sets up the right entrypoint file for the user's platform automatically.

**Long-term**: published to Claude Code plugin marketplace, Codex plugin registry, and other agentic platform marketplaces as versioned packages. The tt-agent.yaml manifest is already structured for this.

If tt-agent outgrows tt-metal (used with other TT repos, or by the OSS community independently), it extracts to `tenstorrent/tt-agent` via `git subtree split` with full history preserved.

---

## What Is Explicitly Deferred

- **New kernel authoring** — writing net-new fused kernels autonomously is a hard, open problem. Not in scope for the initial system. The workflow layer and existing tool skills provide the foundation; this will be revisited when the iteration loop is proven out.
- **Additional workflow skills** — the three initial workflows (iterator, ci-fixer, bisect) cover the most common autonomous loops. Others will be identified through usage.
- **Marketplace publishing** — the content and manifest are designed for it; the publishing pipeline comes after the content is stable.
- **Additional platform adapters** — Claude Code and Codex first; others added as needed.
