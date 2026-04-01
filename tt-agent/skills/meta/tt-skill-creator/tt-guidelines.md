# TT Skill Guidelines

TT-specific constraints for tt-agent skills. Applies on top of `/skill-creator` base
mechanics — do not duplicate what `/skill-creator` already covers (frontmatter format,
progressive load table layout, description optimization, evals).

---

## Layer Placement

Every skill belongs to exactly one layer:

| Layer | Path | Decision rule |
|---|---|---|
| `skills/orchestration/` | Routes, plans, decomposes | Routes work to other skills? |
| `skills/workflows/` | Autonomous loops | Runs until a goal is met? |
| `skills/tools/` | Single-purpose capabilities | Does one concrete thing? |
| `skills/meta/` | System-level utilities | Builds or introspects tt-agent? |

---

## Content Placement

| Content type | Where it goes | Rule |
|---|---|---|
| Procedural instructions | Skill (SKILL.md / sub-files) | Steps, decision trees, patterns |
| Hardware invariants | `tt-agent/knowledge/hardware/` | Silicon facts only. Survives software releases. |
| Canonical example pointers | `tt-agent/knowledge/references/` | Path + one-line description. No inlined content. |
| Volatile info (APIs, patterns) | Nowhere — use `tt-learn` | Agent reads fresh from source on demand |
| Work products (findings, logs) | `~/.tt-agent/notes` | Dated, includes tt-metal commit hash |

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

---

## Self-Check

Before finalizing any skill:
- [ ] Correct layer directory
- [ ] No inlined API signatures — all volatile content points to source
- [ ] Workflow skills define convergence criteria
- [ ] `pytest tt-agent/tests/test_skill_frontmatter.py` passes
