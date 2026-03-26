# TT Skill Guidelines

TT-specific rules for building skills in the tt-agent system. Load this alongside
`/skill-creator` when creating or reviewing any tt-agent skill.

---

## 1. Skill Layer Placement

Every skill belongs to exactly one layer. Place SKILL.md in the correct subdirectory:

| Layer | Path | Character |
|---|---|---|
| Orchestration | `skills/orchestration/` | Routes, plans, decomposes requests |
| Workflows | `skills/workflows/` | Autonomous loops — runs until convergence |
| Tools | `skills/tools/` | Single-purpose, invoked during execution |
| Meta | `skills/meta/` | System-level: extend or learn from the system |

**Decision rule:**
- Does it route work to other skills? → orchestration
- Does it run a loop until a goal is met? → workflows
- Does it do one concrete thing (profile, test, review, design)? → tools
- Does it help build or understand tt-agent itself? → meta

---

## 2. Skills vs Knowledge vs Notes

When writing a skill, be precise about where content goes:

**In the skill (SKILL.md or sub-files):**
Procedural instructions — how to accomplish a task. Steps, decision trees, patterns.

**In `knowledge/hardware/`:**
Silicon-stable facts only: Tensix architecture, NOC topology, tile granularity, CB model,
hardware quirks. If it could change in a software release, it does not go here.

**In `knowledge/references/`:**
Curated pointers to canonical examples, one file per topic. Format: path + one-line
description. No content inlined. Update paths when examples move; never add volatile info.

**Left to `tt-learn`:**
Everything volatile — API signatures, function names, op implementations, current patterns
in the codebase, sharding strategies in use. **Never write these down in a skill.** The
agent reads them fresh from source via `tt-learn` when needed.

**Written to `notes/`:**
Findings produced during work — context briefs, experiment logs, profiler results.
Skills write to `notes/`; they reference `knowledge/` for stable facts.

---

## 3. "Point to Code, Not Inline APIs"

The most common mistake: inlining API documentation that will go stale.

**Wrong — do not do this:**
```
To read from DRAM, use:
  noc_async_read(uint32_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size)
```

**Right:**
```
For NOC read/write API, see:
  tt_metal/hw/inc/api/dataflow/dataflow_api.h
```

Sub-files should describe *patterns and intent*, not *API signatures*. The agent
reads the actual header when it needs the exact signature.

---

## 4. Progressive Load Pattern

Every SKILL.md must have a progressive load table. Only list sub-files that exist.
Do not create sub-files for content that fits in SKILL.md itself.

```markdown
## Progressive Load Table

| Sub-task | Load |
|---|---|
| [specific sub-task description] | `sub-file.md` |
```

**Size rules:**
- SKILL.md ≤ 150 lines. If it grows beyond that, move domain content to sub-files.
- Sub-files: 150–250 lines each. One clear topic per file.
- Total context per task: SKILL.md + 1-2 sub-files + source files as needed.

---

## 5. YAML Frontmatter Requirements

Every SKILL.md must start with:

```yaml
---
name: <directory-name>        # must match directory name exactly
description: "<rich single sentence optimized for triggering>"
---
```

**Description guidelines (from /skill-creator):**
- Single sentence, no period at end
- Starts with what it does, includes when to use it
- Specific enough to trigger correctly, not so broad it triggers on everything
- Include key synonyms and trigger phrases

Run `pytest tt-agent/tests/test_skill_frontmatter.py` to validate.

---

## 6. Workflow Skills Must Define Convergence

If creating a skill in `skills/workflows/`, it must explicitly define:

```markdown
## Convergence Criteria
- **Success:** [explicit condition — e.g., "PCC > 0.999 AND throughput ≥ target"]
- **Local optimum:** [e.g., "5 iterations with < 5% improvement"]
- **Escalate:** [what the agent reports when it cannot converge]
```

---

## 7. TT Quality Bar

Any skill that involves code generation must include a step that verifies:
- **PCC > 0.999** vs PyTorch reference for numerical correctness
- **Hardware-aware correctness**: CB sizing fits L1, tile alignment, NOC conventions
- Output matches patterns found in tt-metal (not invented conventions)

Reference: `tt-agent/knowledge/hardware/` for invariants. Use `tt-learn` for current
patterns in the codebase.

---

## 8. Notes Naming Convention

When a skill writes to `notes/`, use consistent naming:

| Type | Pattern | Example |
|---|---|---|
| Context brief (tt-learn output) | `context-<topic>.md` | `context-matmul.md` |
| Experiment log | `experiments-<task>.md` | `experiments-mlp-opt.md` |
| Per-task plan | `plan-<task>.md` | `plan-fuse-attention.md` |
| Profiler finding | `profile-<workload>.md` | `profile-llama-decode.md` |

Each note must include: topic, date, tt-metal commit hash, sources read.

---

## 9. tt-learn Integration

When a skill needs volatile information (current API, implementation patterns, op
details), it should invoke `tt-learn` rather than encoding the information inline.

Pattern in sub-files:
```markdown
For current [topic] patterns, invoke `tt-learn("[topic]")`.
Starting points: `knowledge/references/[relevant-file].md`
```

---

## 10. Self-Check Before Finalizing a Skill

Before handing off a new skill, verify:
- [ ] SKILL.md starts with valid YAML frontmatter
- [ ] `name` matches directory name
- [ ] `description` triggers correctly (test with /skill-creator eval guidance)
- [ ] Progressive load table is present and all referenced files exist
- [ ] No API signatures inlined — all volatile content points to source
- [ ] Skill is in the correct layer directory
- [ ] `pytest tt-agent/tests/test_skill_frontmatter.py` passes
