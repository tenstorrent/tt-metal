# Benchmark: establishing and comparing a baseline

Profiling says where time goes inside a run; benchmarking says whether a change
moved the number you care about. Do the baseline **first** — a campaign without
a fixed "before" produces numbers nobody can check.

## The primary metric

**Warm device time for the component under optimization**, at a fixed shape and
mesh.

| Metric | Scope |
|---|---|
| Per-step denoise time | The pipeline-level headline |
| Per-component device time | VAE encode/decode, text encode, export — reported separately, they scale and optimize differently |
| End-to-end wall clock, peak L1/DRAM | Secondary; the latter only when memory is a constraint |

For a tiled or chunked component the baseline is a **per-invocation time plus a
work-unit count**, so full-clip time is a multiplication rather than another
measurement:

```
768P/5s = encode_clip_tile × 28 tiles × 8 clips + decode_invocation × 28 × 7
```

Keep roundtrip quality and perf baselines in the same file but deliberately
separated — they answer different questions and drift apart.
`tests/models/wan2_2/test_performance_wan.py` is the in-tree `expected_metrics`
harness to copy.

## Fix these before measuring anything

A benchmark is only comparable if the whole configuration is pinned. Record
every one of these with the number:

| Pinned | Why |
|---|---|
| **Mesh shape** | A `(2,4)` number is not comparable to a `(4,8)`. Use the id convention: `bh_2x4sp1tp0`, `bh_4x8sp1tp0_ring` |
| **Parallel config** | `sp`/`tp` axis assignment changes everything downstream |
| **Input shape** | Production shape, from the model's real schedule |
| **Seed** | For anything whose output feeds a quality gate |
| **dtype per component** | Must match the reference (`../tt-dit-performance/optimization-levers.md`) |
| **Commit SHA** | The baseline is worthless without it |
| **Warm-window method** | Signpost / tail-N / iteration-2 |
| **Trace on or off** | Traced and untraced numbers are different metrics, not better and worse versions of one |

## The gate: is this the path you think it is?

Every result must come from the configuration you intend to ship. Each of these
silently produces a valid-looking number for the wrong thing:

| Trap | Check |
|---|---|
| **Cold, not warm** | First iteration populates the program cache. Iterate ≥ 2×, measure the second |
| **Weight upload / activation prep counted** | One-time construction cost, often the largest thing in an unfiltered measurement. Signature: a run of `TilizeWithValPadding`/`Untilize` at the head of the capture, before the first real compute op |
| **Conv3d blocking fallback** | Grep the log — a fallback warning means the tuned table missed and you're benchmarking a conservative default |
| **A fast path not engaged** | `existing-fast-paths.md` § "Classifying a hot spot". Benchmarking an unfused path then "optimizing" it back to the fused one is not a win |
| **Wrong parallel config** | Confirm the factors took effect rather than silently degenerating to 1 |
| **The optimization did not engage** | Env flag left off, fused op fell back, blocking table missed. `../tt-dit-performance/SKILL.md` § "No speedup is accepted on wall clock alone" |

A number from a run that tripped any of these is invalid. Fix and rerun before
optimization proceeds.

## Recording a baseline

Prefer an in-tree performance test over an ad-hoc script — it becomes the
regression gate for free. Follow
`tests/models/wan2_2/test_performance_wan.py`: an `expected_metrics` dict keyed
by mesh shape and resolution, with a slack factor.

```python
BH_4X8_EXPECTED_METRICS_SLACK = 1.10

expected_metrics = {"encoder": 0.1, "denoising": 163.0, "vae": 18.2, "total": 192.0}
```

Set bars **generously**. They exist to catch a regression or a pathology, not to
pin a tuned number — a tight bar on an untuned component is a flaky test.

Then write the journal entry (`../shared/journal-protocol.md`):

```markdown
## Baseline — <component> @ <mesh> <input shape>
**<date>** · `<sha>` · warm via <method> · trace <on|off>

| Metric | Value |
|---|---|
| <component> per invocation | <ms> |
| work units @ <config> | <n> × <m> |
| projected <config> | <s> |

**Command:** <exact pytest invocation>
**Gate:** <quality metric at this commit>
**Top ops:** <from the profile>
**Frozen output:** <path to the baseline decode — image / video / audio>
```

**Freeze a baseline output** alongside the numbers. A chain of individually
PCC-passing optimizations can still degrade what a user sees, and tile seams and
temporal flicker are precisely what whole-tensor PCC hides
(`../tt-dit-add-model/testing-and-accuracy.md` § "Artifact rubric").

## Comparing

Same command, same everything, new SHA. Report:

```
<component>: <before> → <after> (<Δ%>) · quality <before> → <after> · <bound class>
```

Two rules that have burned this tree:

| Rule | Evidence |
|---|---|
| **Never judge a per-op change by whole-model wall clock** | One case concluded an SDPA config was a regression (0.652 → 0.876 s/wave); the same code spans 0.34–0.99 s/wave, so it was noise and was later reversed. Untraced wall clock has ~3× variance — measure the op, under the profiler |
| **Report the quality metric alongside every performance number** | A speedup without its gate is not a result. `tt-dit-performance` aborts on quality regression and can only do that if the number is there |

## Merge checklist

Before a performance change lands:

- [ ] Baseline recorded with command, mesh shape, input shape, SHA, warm-window method
- [ ] New measurement taken the same way, on the same window method
- [ ] Quality gate green at the production shape, at the new commit
- [ ] **Profile shows the change actually engaged** — not inferred from the timing
- [ ] Frozen baseline output re-rendered and compared, if the change touched
      sharding, normalization or precision
- [ ] A representative profile saved, with the op-to-op gap median **and** mean
- [ ] Hot spot classified against `existing-fast-paths.md` — confirmed genuinely
      new work, not a fast path that was disabled
- [ ] `expected_metrics` updated if the component has a perf test
- [ ] Journal entry written, including the whole sweep curve if a sweep was run
- [ ] Anything unproven left behind an env flag, with the flag's default stated
