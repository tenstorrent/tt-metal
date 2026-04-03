# Decomposer: Orchestration Patterns for TT Development

Guidance for breaking high-level requests into sequenced skill dispatches.
Load this when the orchestrator reaches step 3 (Decompose).

---

## Build New vs Optimize Existing

The first decision changes the entire task shape.

**Build new** — the thing does not exist yet:
- Start with `tt-designer` to produce a spec (data flow, CB layout, grid shape)
- Then `tt-run` to build and compile
- Then `tt-tester` to verify PCC > 0.999
- Then `tt-code-review` before merge

**Optimize existing** — the thing works but is too slow or too inaccurate:
- Start with `tt-optimizer` — never guess bottlenecks
- Optimizer profiles, analyzes, applies targeted improvements, and verifies
- Profile again to confirm the improvement is real

**Fix failure** — something that worked has broken:
- If CI failure or wrong outputs: `tt-debugger` (reproduces, diagnoses, fixes, verifies)
- If regression in perf or accuracy: `tt-bisect` to find the commit, then `tt-debugger`

---

## Common Decomposition Patterns

### Single kernel (new)

```
1. tt-designer   — design CB layout, NOC strategy, tile flow
2. tt-run        — compile and run minimal smoke test
3. tt-tester     — write pytest, verify PCC > 0.999
4. tt-code-review
```

### Single ttnn op (new)

```
1. tt-designer   — design host-side op: grid, shard strategy, CB config
2. tt-run        — compile op
3. tt-tester     — parametric tests: dtypes, shapes, edge cases
4. tt-code-review
```

For volatile details on op structure and canonical examples, invoke
`tt-learn("ttnn op structure")`. Starting point:
`tt-agent/knowledge/references/operators.md`

### Full model (new or port)

```
1. tt-designer   — layer map, weight layout, KV-cache plan, multi-device strategy
2. Per-layer:    — tt-designer → tt-run → tt-tester loop
3. Integration:  — tt-tester with end-to-end accuracy test
4. tt-optimizer  — identify slow layers, optimize
5. tt-code-review
```

---

## The Optimization Workflow

`tt-optimizer` handles the full optimization loop internally via phases:

1. **Prepare** — detect workspace, identify target
2. **Build** — compile via tt-run
3. **Profile** — capture traces, identify hot kernels
4. **Analyze** — apply bottleneck patterns, roofline analysis
5. **Optimize** — targeted code changes
6. **Verify** — confirm improvement, check correctness

For hardware invariants that constrain optimization (L1 capacity, NOC bandwidth,
tile granularity), see `tt-agent/knowledge/hardware/`.

---

## Iteration Budget

| Phase | Max iterations before escalating |
|---|---|
| Build new (single kernel/op) | 5 compile+test cycles |
| Optimization loop | 10 profile+optimize cycles |
| Debug / CI fix | 5 hypothesis+test cycles |

Escalate means: write current STATUS.md with hypothesis, findings, and blockers,
then surface to the developer for guidance.

---

## Notes Written by Orchestrator

Always create PLAN.md before starting dispatch:

```markdown
# Plan: <task name>
Date: YYYY-MM-DD
Commit: <git hash>
Skills: <list to be invoked>

## Sub-tasks
1. [ ] ...
2. [ ] ...
```

Update STATUS.md after each skill completes:

```markdown
# Status: <task name>
Date: YYYY-MM-DD
Step: <current step>
Completed: [list]
Blocked: [if any]
Next: <next skill to invoke>
```
