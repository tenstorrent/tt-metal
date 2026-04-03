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
| Tool | `metadata: { layer: tool }` | Does one concrete thing? |
| Meta | `metadata: { layer: meta }` | Builds or introspects tt-agent? |

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

| Type | Filename pattern |
|---|---|
| Context brief | `context-<topic>.md` |
| Experiment log | `experiments-<task>.md` |
| Plan | `plan-<task>.md` |
| Profile | `profile-<workload>.md` |
| Debug findings | `findings-debug-<target>.md` |
| Perf findings | `findings-perf-<target>.md` |

---

## Self-Check

Before finalizing any skill:
- [ ] Correct layer in `metadata.layer`
- [ ] No inlined API signatures — all volatile content points to source
- [ ] Workflow skills define convergence criteria
- [ ] Workflow skills have a phase table with Loads and Produces columns
- [ ] All files referenced in Loads columns exist on disk
- [ ] `pytest tt-agent/tests/test_skill_frontmatter.py` passes
