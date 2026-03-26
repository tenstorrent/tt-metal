# tt-agent Design Decisions

This document records the non-obvious architectural decisions behind tt-agent and their
rationale. Future agents and developers: read this before changing anything structural.
Each decision is dated so you can judge whether it is still current.

---

## 2026-03-26: Co-location in tt-metal

tt-agent lives inside `tt-metal/tt-agent/` rather than a separate repo.

**Why:** Skills reference tt-metal paths deeply (API headers, programming examples,
operator source). The agent needs tt-metal checked out regardless. Co-location is
correct, not a compromise.

**Extraction path:** When tt-agent outgrows tt-metal, `git subtree split --prefix=tt-agent`
yields a clean repo with full history.

---

## 2026-03-26: Own the full stack (not built on superpowers)

tt-agent does not use the superpowers plugin framework as its base.

**Why:** Skills and knowledge must be platform-agnostic — authored once, delivered
to Claude Code, Codex, and future platforms. Owning the stack means we control the
format, versioning, and distribution. Adapters in `adapters/` handle per-platform packaging.

---

## 2026-03-26: Skills vs Knowledge vs Notes

Three content types that must not be conflated:

- **Skills** (`skills/`) — how to accomplish a task. Procedural instructions.
- **Knowledge** (`knowledge/`) — stable hardware invariants (silicon facts) + curated
  references (pointers to canonical code examples). Never volatile APIs.
- **Notes** (`notes/`) — shared blackboard. Findings written by agents and humans,
  shared across sessions and team members.

**Why the split:** The TT software stack evolves rapidly. Inlining API signatures or
implementation patterns into static files creates lies. Volatile knowledge is always
learned fresh from source via `tt-learn`.

---

## 2026-03-26: Volatile knowledge via tt-learn + deepwiki-mcp

API signatures, op implementations, sharding patterns, CCL usage — never written down.

**Why:** These change with every PR. The `tt-learn` skill researches the live codebase
via deepwiki-mcp on demand, using `knowledge/references/` as starting points. Findings
are written to `notes/` with a commit hash, so readers can judge freshness.

---

## 2026-03-26: notes/ as shared blackboard

The `notes/` directory at the repo root is the team's shared, evolving knowledge cache.
Not session memory — notes persist across sessions and are shared between developers
and multiple agent sessions. Named "notes" (not "workspace", "memory", or "context").

---

## 2026-03-26: Two MCP dependencies only

tt-device-mcp (hardware execution) and deepwiki-mcp (codebase research). Everything
else via CLI and Bash.

**Why:** MCPs add configuration burden. Only add one when CLI genuinely cannot do the job.
Hardware execution requires a persistent device connection (MCP). Semantic codebase search
is deepwiki-mcp's purpose. Everything else (build, profile invocation, file I/O) is CLI.

---

## 2026-03-26: Skill layers

Four layers visible in the filesystem:
- `orchestration/` — routes, plans, decomposes
- `workflows/` — autonomous loops (iterate until converged)
- `tools/` — single-purpose capabilities invoked during execution
- `meta/` — system-level utilities: extend the system (tt-skill-creator) and learn from it (tt-learn)

**Workflow layer is intentionally thin.** tt-iterator, tt-ci-fixer, tt-bisect share the
same base loop (hypothesize → implement → run → analyze → next hypothesis). What differs
is triggering context and convergence criteria.

---

## 2026-03-26: tt-designer as unified design-phase skill

`tt-designer` in `tools/` combines TT-specific brainstorming, performance estimation
(roofline, arithmetic intensity), and data-movement planning (CCL strategy) into one skill.

**Why unified:** These are not separate invocations — planning a TT implementation naturally
covers all three. Wraps `/superpowers:brainstorm` and adds TT hardware constraint grilling.
Bookend to `tt-code-review`: designer before writing code, code-review after.

---

## 2026-03-26: Two languages, one stack

The TT software stack spans two distinct language ecosystems:

- **C++** — kernels (bare-metal RISC-V), ttnn operators (host-side program factories),
  and the tt-metal runtime. Low-level, hardware-aware, compiled per-processor.
- **Python** — models (composing ttnn operators), training scripts, and tooling.
  High-level, dynamic, PyTorch-adjacent.

tt-agent must cover expertise at both levels. A kernel developer and a model developer
have different mental models, different failure modes, different debug workflows, and
different quality bars. Skills must not assume one language or the other — they must
be explicit about which level they operate at.

**Why this matters for skill design:** A skill like `tt-debugger` needs different
procedures for a C++ kernel hang vs a Python model producing wrong outputs. A skill
like `tt-tester` writes C++ unit tests for operators and Python pytests for models.
Skills should state their target level explicitly, and `tt-orchestrator` should route
based on which level the request targets.

---

## 2026-03-26: tt-skill-creator first, then use it to build everything

All skills after tt-skill-creator are built using tt-skill-creator itself.

**Why:** Validates the tool works. Every subsequent skill is both a deliverable and a
test of tt-skill-creator's quality. "Use what you build."
