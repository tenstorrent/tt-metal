# `plan.md` template

Written once in Phase 2, before the first round. Its job is to make the loop
**safe to leave alone**: the stop gates say when to finish, this says what the
loop may do in the meantime.

Copy the block below into `<campaign-root>/plan.md` and fill it in.

---

```markdown
# Campaign: <model> <component> on <mesh>

## Goal

<One sentence. The user-visible outcome, not the mechanism.>
e.g. "MiniMax-H3 visual VAE decode under 3 s per invocation at 768P/5s on a 4x8
Blackhole Galaxy, with roundtrip quality unchanged."

## Acceptance criteria

Observable, each with the command that proves it.

| # | Criterion | Proof command | Status |
|---|---|---|---|
| 1 | Warm device time <= 3000 ms at the production shape | `pytest .../test_performance_...py -k <id>` | open |
| 2 | Quality gate green at the production shape | `pytest .../test_..._vae.py -k <id>` | open |
| 3 | Frozen output shows no new artifact (seams, flicker, banding) | compare against `artifacts/round-0/frozen/` | open |
| 4 | Every landed change has profile evidence it engaged | `ledgers/optimizations.md` | open |

State criteria as behaviour a human can verify, not as internal structure.
"SDPA uses chunk 192" is not a criterion; "decode is under 3 s and quality holds"
is.

## Working point (fixed)

Mesh · production shape(s) · parallel config · dtype per component · seed.
**Changing any of these after Phase 1 invalidates the campaign.**

## Path boundaries

### Upper bound — the most the loop may do unattended

<e.g. multiple patches per component; parallel-config changes; blocking and
chunk sweeps; math-fidelity changes; fusion using ops that already exist;
adding a parameter to an existing ttnn op; repeated profiling.>

### Lower bound — the least that counts as a round

One profile-backed change plus revalidation — unless the round's evidence proves
no change is warranted, in which case the evidence itself is the round.

### Can use

Existing fused ttnn ops · config and blocking tables · parallel-config changes ·
math fidelity · layout changes · trace where shapes are predictable · new
parameters on existing ops · focused tests and microbenchmarks · tracy and
tt-perf-report.

### Cannot use

- Changing the fixed shape, mesh or target after seeing results
- Weakening or skipping a quality gate to land a speedup
- Claiming a win from a run that failed the validity gate
- Kernel work on an op outside the warm-window top 3
- Skipping the Phase 0.5 prior-art gate
- `--no-verify`, force-push, or rewriting a landed round's history
- Dropping below the reference dtype without a perceptual-quality gate
- Device commands outside `../shared/device-hangs.md` (notably `tt-smi -r`)

## Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | Fixed baseline recorded | Phase 1 complete, frozen output saved |
| 2 | <first target, e.g. largest hot spot addressed> | <criterion> |
| 3 | Target met or ceiling proven | A stop gate fires |

## Known constraints

<Hardware, branch, dependency, or scheduling constraints that bound the work —
e.g. "audio VAE work is blocked on PR #NNNN", "Galaxy available weekdays only".>
```

---

## Why path boundaries matter more than the stop gates

The stop gates decide when the campaign is *over*. The boundaries decide what
happens in the hours between rounds when nobody is watching. An unattended loop
without a `Cannot use` list will eventually weaken a gate to make a number move,
because that is the locally rational step — and it will be recorded as a win.

The `Cannot use` list above encodes failures this codebase has actually seen:
measurements taken from invalid runs, kernel effort spent on ops that were not
the bottleneck, and a device reset command that took a machine down.
