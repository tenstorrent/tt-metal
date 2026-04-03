# tt-agent Design Decisions

This document records the non-obvious architectural decisions behind tt-agent and their
rationale. Future agents and developers: read this before changing anything structural.
Each decision is dated so you can judge whether it is still current.

---

## Co-location in tt-metal

tt-agent lives inside `tt-metal/tt-agent/` rather than a separate repo.

**Why:** Skills originated from tt-metal's deep hardware stack (API headers, programming
examples, operator source). Co-location is correct for the primary repo, not a compromise.
See "Multi-repo readiness" for how skills generalize beyond tt-metal.

**Extraction path:** When tt-agent outgrows tt-metal, `git subtree split --prefix=tt-agent`
yields a clean repo with full history.

---

## Multi-repo readiness

tt-agent starts in tt-metal but must work across all Tenstorrent repos (vLLM,
tt-inference-server, tt-shield, etc.) without structural changes.

**Architectural constraints:**

1. **Skills never hardcode repo identity.** Detect context from the working directory
   (git remote, file structure), not from string-matching repo names.
2. **knowledge/references/ are tt-metal hints, not requirements.** Skills must work
   (via deepwiki + local search) even without them. They accelerate research in
   tt-metal; they are not load-bearing.
3. **Deepwiki is the repo-agnostic backbone.** Local Grep/Read works in whatever repo
   you're in. Deepwiki can search any TT repo by name.
4. **Notes are repo-tagged.** Context notes include which repo was researched, so
   findings from different repos don't get confused.

**Why now:** Decisions made while only targeting tt-metal easily bake in assumptions
(hardcoded paths, tt-metal-specific references as required inputs) that are expensive
to undo later. These constraints cost nothing to follow now.

---

## Own the full stack (not built on superpowers)

tt-agent does not use the superpowers plugin framework as its base.

**Why:** Skills and knowledge must be platform-agnostic — authored once, delivered
to Claude Code, Codex, and future platforms. Owning the stack means we control the
format, versioning, and distribution. Adapters in `adapters/` handle per-platform packaging.

---

## Skills vs Knowledge vs Notes

Three content types that must not be conflated:

- **Skills** (`skills/`) — how to accomplish a task. Procedural instructions.
- **Knowledge** (`knowledge/`) — stable facts, patterns, and execution recipes:
  - `hardware/` — silicon-stable facts (Tensix architecture, NOC topology, tile granularity)
  - `references/` — curated pointers to canonical code examples (path + one-line description)
  - `recipes/<repo>/` — per-repo execution patterns (build, test, env, server lifecycle).
    Plain markdown, no frontmatter, ≤60 lines per file. Each repo directory has an `index.md`.
  - `<domain>/` (e.g., `profiling/`, `debugging/`) — domain expertise contributed by
    subject-matter experts. Patterns, methodologies, interpretation guides — not procedures.
- **Notes** (configured via `notes_path` in `tt-agent.yaml`) — shared blackboard.
  Findings written by agents and humans, shared across sessions and team members.

**Why the split:** The TT software stack evolves rapidly. Inlining API signatures or
implementation patterns into static files creates lies. Volatile knowledge is always
learned fresh from source via `tt-learn`.

**Who contributes what:**

| Who | Contributes to | Without touching |
|---|---|---|
| Repo engineer | `knowledge/recipes/<repo>/` | Skills or domain knowledge |
| Domain expert | `knowledge/<domain>/` | Skills or recipes |
| Agent team | `skills/` | Wires knowledge in via phase tables |

---

## Volatile knowledge via tt-learn + deepwiki-mcp

API signatures, op implementations, sharding patterns, CCL usage — never written down.

**Why:** These change with every PR. The `tt-learn` skill researches the live codebase
via deepwiki-mcp on demand, using `knowledge/references/` as starting points. Findings
are written to the notes directory with a commit hash, so readers can judge freshness.

---

## Notes live outside tt-metal

Notes are stored at `~/.tt-agent/notes`, outside the tt-metal repo.

**Why:** tt-metal is a large open-source repo with many contributors. Notes are scoped
to small teams working on shared problems — not a repo-wide resource. Putting notes in
the repo root would pollute the working tree for everyone, and committing them would
pollute history.

**What didn't change:** Notes are still the shared blackboard. They persist across
sessions and follow the same naming conventions (`context-<topic>.md`,
`experiments-<task>.md`, etc.). Always dated with repo name and commit hash.

---

## Two MCP dependencies only

tt-device-mcp (hardware execution) and deepwiki-mcp (codebase research). Everything
else via CLI and Bash.

**Why:** MCPs add configuration burden. Only add one when CLI genuinely cannot do the job.
Hardware execution requires a persistent device connection (MCP). Semantic codebase search
is deepwiki-mcp's purpose. Everything else (build, profile invocation, file I/O) is CLI.

---

## Skill layers

Four layers, declared via `metadata.layer` in YAML frontmatter. Skills are flat under
`skills/<name>/` — layers are metadata, not directories.

| Layer | Purpose | Examples |
|---|---|---|
| `orchestration` | Routes work to other skills | tt-orchestrator |
| `workflow` | Runs until a goal is met, has phases | tt-optimizer, tt-debugger, tt-tester |
| `tool` | Does one concrete thing | tt-run |
| `meta` | Builds or introspects the system | tt-skill-creator, tt-learn |

**Workflow skills declare phases.** Each phase specifies what knowledge to load (Loads)
and what it produces (Produces). After each phase, the agent summarizes in 3-5 lines
and moves on — loaded knowledge is consumed, not carried forward. The repo isn't known
until after workspace detection, so repo-specific recipes can't be loaded eagerly.

**Structural invariant:** Every file referenced in a Loads column must exist on disk.
Enforced by `test_skill_frontmatter.py`.

---

## tt-run: shared execution engine

tt-run is a tool-layer skill that handles workspace detection, recipe loading, build
orchestration, MCP routing (Bash vs tt-device-mcp), and job lifecycle. It is both
directly invocable ("just run this pytest") and loaded by workflow skills as their
execution engine.

**Why:** Every workflow skill needs to detect the workspace, build, and execute. Factoring
this into a shared tool-layer skill avoids duplicating execution logic across
tt-optimizer, tt-debugger, and tt-tester.

**MCP routing rule:** "Device commands go through MCP, not Bash" is a safety invariant
in CLAUDE.md, not gated behind skill loading.

---

## tt-designer as unified design-phase skill

tt-designer combines TT-specific brainstorming, performance estimation (roofline,
arithmetic intensity), and data-movement planning (CCL strategy) into one skill.

**Why unified:** These are not separate invocations — planning a TT implementation naturally
covers all three. Wraps `/superpowers:brainstorm` and adds TT hardware constraint grilling.
Bookend to `tt-code-review`: designer before writing code, code-review after.

---

## Two languages, one stack

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

## Shared persona, overridable per skill

A default persona is defined in `tt-agent/persona.md`. The adapter entrypoint
references it; skills can load it from their progressive load tables.

**Why:** The developer using tt-agent is experienced. Every skill benefits from the
same baseline: precise, critical, no filler. A shared file avoids duplication and
makes the persona easy to evolve in one place.

**Override path:** If a specific skill needs a different voice (e.g., a teaching-oriented
onboarding skill), it overrides the persona in its own SKILL.md.

---

## tt-skill-creator design-first process

tt-skill-creator's primary value is the design alignment phase, not the writing phase.
Before any skill content is written, the agent interrogates the developer with structured
questions — one at a time — to establish scope, boundaries, failure modes, and dependencies.
The goal is to expose flaws in the spec before writing begins.

**Why:** Skills that skip the design phase end up over-scoped, overlapping with other skills,
or encoding assumptions that break. The interrogation protocol (propose worst interpretation,
summarize understanding, ask what's wrong) catches these issues early.

All skills after tt-skill-creator are built using tt-skill-creator itself. Validates the
tool works — every subsequent skill is both a deliverable and a test of its quality.
