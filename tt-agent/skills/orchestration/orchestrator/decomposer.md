# Decomposer: Orchestration Patterns for TT Development

Guidance for breaking high-level requests into sequenced skill dispatches.
Load this when the orchestrator reaches step 3 (Decompose).

---

## Build New vs Optimize Existing

The first decision changes the entire task shape.

**Build new** — the thing does not exist yet:
- Start with `tt-designer` to produce a spec (data flow, CB layout, grid shape)
- Then `tt-device` to scaffold and compile
- Then `tt-tester` to verify PCC > 0.999
- Then `tt-code-review` before merge

**Optimize existing** — the thing works but is too slow or too inaccurate:
- Start with `tt-profiler` — never guess bottlenecks
- Then `tt-iterator` to apply targeted improvements
- Then `tt-tester` to confirm correctness is preserved
- Profile again to confirm the improvement is real

**Fix failure** — something that worked has broken:
- If CI: `tt-ci-fixer` first (reads logs, narrows failure, patches)
- If regression in perf or accuracy: `tt-bisect` to find the commit, then debug
- If wrong outputs on device: `tt-debugger`

---

## Common Decomposition Patterns

### Single kernel (new)

```
1. tt-designer   — design CB layout, NOC strategy, tile flow
2. tt-device     — compile and run minimal smoke test
3. tt-tester     — write pytest, verify PCC > 0.999
4. tt-code-review
```

### Single ttnn op (new)

```
1. tt-designer   — design host-side op: grid, shard strategy, CB config
2. tt-device     — compile op
3. tt-tester     — parametric tests: dtypes, shapes, edge cases
4. tt-code-review
```

For volatile details on op structure and canonical examples, invoke
`tt-learn("ttnn op structure")`. Starting point:
`tt-agent/knowledge/references/ops.md`

### Full model (new or port)

```
1. tt-designer   — layer map, weight layout, KV-cache plan, multi-device strategy
2. Per-layer:    — tt-designer → tt-device → tt-tester loop
3. Integration:  — tt-tester with end-to-end accuracy test
4. tt-profiler   — identify slow layers
5. tt-iterator   — optimize hot layers
6. tt-code-review
```

---

## The Optimization Workflow

When dispatching `tt-iterator`, always set it up with:

1. **Profile first** (`tt-profiler`): identify the hot kernel or op. Do not optimize
   without data. Write findings to `~/.tt-agent/notes/profile-<workload>.md`.

2. **Extract** the isolated kernel or op into a standalone test. This makes iteration
   fast and prevents regressions elsewhere.

3. **Isolate** the specific bottleneck: is it compute-bound, bandwidth-bound, or
   CB-stall-bound? The profiler output (op cycles, NOC utilization) tells you which.

4. **Iterate** with `tt-iterator`: apply one change at a time, measure, keep if better.
   Log each attempt to `~/.tt-agent/notes/experiments-<task>.md`.

For hardware invariants that constrain optimization (L1 capacity, NOC bandwidth,
tile granularity), see `tt-agent/knowledge/hardware/`.

---

## Iteration Budget

| Phase | Max iterations before escalating |
|---|---|
| Build new (single kernel/op) | 5 compile+test cycles |
| Optimization loop | 10 profiler+iterate cycles |
| CI fix | 3 patch attempts |
| Debug | 5 hypothesis+test cycles |

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
