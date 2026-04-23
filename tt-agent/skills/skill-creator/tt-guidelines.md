# TT Skill Guidelines

TT-specific constraints for tt-agent skills. Applies on top of `/skill-creator` base
mechanics — do not duplicate what `/skill-creator` already covers (frontmatter format,
progressive load table layout, description optimization, evals).

---

## Layer Placement

Every skill belongs to exactly one layer, declared via `metadata.layer` in YAML frontmatter.
Skills are flat under `skills/<name>/` — layers are metadata, not directories.

| Layer | Frontmatter value | Decision rule |
|---|---|---|
| Orchestration | `metadata: { layer: orchestration }` | Routes work to other skills? |
| Workflow | `metadata: { layer: workflow }` | Runs until a goal is met? |
| Tool | `metadata: { layer: tool }` | Does one concrete thing tied to a pipeline (build, run, profile, debug)? |
| Meta | `metadata: { layer: meta }` | Cross-cutting utility, or builds/introspects tt-agent? |

Tool vs Meta: both "do one concrete thing". The distinguishing question is whether the
skill is **pipeline-bound** (domain tool: `tt-run` for execution, `tt-profiler` for
profiling) or **cross-cutting** (invoked by any skill at any time, produces a notes
artifact: `tt-learn`, `tt-code-review`). Cross-cutting utilities go in Meta.

---

## Content Placement

| Content type | Where it goes | Rule |
|---|---|---|
| Procedural instructions | Skill (SKILL.md / sub-files) | Steps, decision trees, patterns |
| Hardware invariants | `knowledge/hardware/` | Silicon facts only. Survives software releases. |
| Canonical example pointers | `knowledge/references/` | Path + one-line description. No inlined content. |
| Per-repo execution patterns | `knowledge/recipes/<repo>/` | Build, test, env, server lifecycle. Plain markdown, ≤60 lines. |
| Domain expertise | `knowledge/<domain>/` | Profiling patterns, debug guides. Plain markdown, ≤80 lines. |
| Volatile info (APIs, patterns) | Nowhere — use `tt-learn` | Agent reads fresh from source on demand |
| Work products (findings, logs) | `~/.tt-agent/notes` | Dated, includes repo name and commit hash |

**The cardinal rule: never inline volatile content.** Point to source files. Describe
patterns and intent, not API signatures.

Wrong:
```
noc_async_read(uint32_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size)
```

Right:
```
For NOC read/write API: tt_metal/hw/inc/api/dataflow/dataflow_api.h
```

---

## Workflow Skills

Must define explicit convergence criteria:

```markdown
## Convergence Criteria
- **Success:** [condition — e.g., "PCC > 0.999 AND throughput ≥ target"]
- **Local optimum:** [e.g., "5 iterations with < 5% improvement"]
- **Escalate:** [what to report when stuck]
```

Must declare phases with a phase table. Each phase specifies what knowledge to load
and what it produces:

```markdown
| Phase | Loads | Produces |
|---|---|---|
| Prepare | `skills/run/workspace-detect.md` | Workspace context |
| Build | `knowledge/recipes/<repo>/build.md` (via tt-run) | Built artifacts |
| ...   | ... | ... |
```

Rules for phase tables:
- Every file referenced in Loads must exist on disk (enforced by tests)
- Repo-specific recipes use `<repo>` placeholder — resolved at runtime after workspace detection
- After each phase, summarize in 3-5 lines and move on. Loaded knowledge is consumed, not carried forward.
- Persistent findings go to `~/.tt-agent/notes/`

---

## Quality Bar

Any skill involving code generation must verify:
- PCC > 0.999 vs PyTorch reference (default bar for all ops)
- Lower thresholds (e.g., 0.99) acceptable only with explicit justification:
  end-to-end model accuracy after many layers, or ops using intentional
  approximation (fast GELU, LoFi math modes)
- Hardware-aware correctness: CB sizing fits L1, tile alignment, NOC conventions
- Output matches patterns found in tt-metal, not invented conventions

---

## Notes Naming

Each skill owns the naming of the notes it produces. Declare the exact filename
pattern in the skill's `SKILL.md` (or the sub-file that writes the note) — not
here. Keeping the inventory out of this file prevents drift.

### Convention

Note filenames follow `<kind>-<scope>.md`, with a timestamp suffix when multiple
instances can coexist in a single session:

- `<kind>` is what the file *is* (`context`, `plan`, `findings-review`,
  `profile`, `experiments`, …). Dash-separated words, all lowercase.
- `<scope>` is what the file *is about* — topic, task, target, or scope slug.
- Append `-<YYYY-MM-DD-HHMMSS>` when overwrite-on-rerun is unacceptable
  (e.g., review findings: one per session, never clobbered).

All notes include, in their body, the date, the repo name, and the commit hash
at time of writing.

---

## Developer-Rule Conflict Protocol

A developer's personal rules (global CLAUDE.md, project CLAUDE.md, etc.) may
conflict with what a tt-agent skill needs to do. Personal rules must not
silently override skill behavior, but skills must not silently override
personal rules either. The skill surfaces the conflict.

**Applies to any skill that does something autonomous** — commits, creates
branches, spawns workspaces, deletes files, modifies shared state, runs
long-lived background work.

**Protocol — run at the start of the skill, before any state-changing work:**

1. **State plainly** what the skill will do autonomously, in one or two
   sentences. Name the specific actions (e.g., "commit every iteration",
   "create N new workspaces", "modify source files in place").
2. **Detect conflicts**: check the developer's CLAUDE.md files for rules
   that forbid or constrain those actions. Common conflicts: commit rules,
   push rules, file-deletion rules, parallel-execution rules.
3. **Surface the conflict**: quote the rule, explain what the skill needs,
   ask the developer to override or adjust scope.
4. **Wait for explicit confirmation** before proceeding. Silence is not
   consent.

The developer's response is session-scoped — it authorizes this one
invocation, not future ones. A skill that re-enters for a new target
re-runs the preflight.

**What this is not:** a blanket exemption. `git push`-level prohibitions
and clearly destructive rules remain in force. The protocol surfaces
friction between autonomy and personal rules; it does not erase the rules.

## Self-Check

Before finalizing any skill:
- [ ] Correct layer in `metadata.layer`
- [ ] No inlined API signatures — all volatile content points to source
- [ ] Workflow skills define convergence criteria
- [ ] Workflow skills have a phase table with Loads and Produces columns
- [ ] All files referenced in Loads columns exist on disk
- [ ] Autonomous skills declare and run the Developer-Rule Conflict Protocol
- [ ] `pytest tt-agent/tests/test_skill_frontmatter.py` passes
