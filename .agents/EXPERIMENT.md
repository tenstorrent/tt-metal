# skillexp — isolating the contribution of graph-fusing and shard-advise

This branch family is a 2×2 ablation over the autoport bringup pipeline. It answers one question:
**how much of the optimize stage's speedup comes from `$graph-fusing`, and how much from
`$shard-advise`?**

Everything outside the two factors is held constant: same tt-metal commit, same skills, same
prompts, same functional decoder per model, same measurement convention.

## Branches

| Branch | fusing | shard-advise |
|---|---|---|
| `mvasiljevic/qb2/skillexp/base` | — | — | (shared parent; stage 01 runs from here) |
| `mvasiljevic/qb2/skillexp/nofuse-advise` | no | yes |
| `mvasiljevic/qb2/skillexp/nofuse-noadvise` | no | no |
| `mvasiljevic/qb2/skillexp/fuse-advise` | yes | yes |
| `mvasiljevic/qb2/skillexp/fuse-noadvise` | yes | no |

The four arm branches differ from `base` **only** in the factor edits listed below. Nothing else.

## The factors

**fusing = no.** `$graph-fusing` is unreachable, not merely unrequested: the `01b-fused-decoder`
stage prompt is deleted, the `graph-fusing` skill directory is deleted, and every `$graph-fusing`
reference is removed from `optimize/SKILL.md` (orientation step 1, the pre-tuning paragraph, and the
evidence checklist item). Leaving the skill on disk while only skipping stage 01b would not isolate
the factor — `$optimize` step 1 tells the agent to run it anyway.

**fusing = yes.** Stage `01b-fused-decoder` runs between 01 and 02, and `$optimize` keeps its
"fix topology before tuning knobs" step.

**shard-advise = no.** The `shard-advise` skill directory is deleted, `OPT-015` and every
`$shard-advise` reference is removed from `optimize/SKILL.md`, the REQUIRED shard-advise bullet is
removed from `02-optimized-decoder.txt`, and `02-optimized-decoder.check.sh` (the hard gate that
fails the stage without `report.json` + `final_ir.mlir`) is deleted. What remains is the
pre-advisor-tool sharding guidance: the agent still owns on-device sharding, still chains L1-resident
ops on one shard spec (OPT-003), still sweeps DRAM-sharded decode matmuls (OPT-004) — it just derives
the layout itself from the perf report instead of querying a compiler tool. This arm needs no tt-mlir
checkout at all.

**shard-advise = yes.** Unchanged from `base`: OPT-015 seeds the dense-path L1 layout from a real
`ttnn-advise capture`, and the runner-side gate enforces that the advisor actually ran this pass.

## Held constant across all four arms

Applied on `base`, so identical in every arm:

- **`DECODE_BATCH`** — traced decode is validated and measured at both batch 1 and the serving decode
  batch, and the optimize stage runs its layout/matmul candidate search at `DECODE_BATCH`. Candidate
  legality is batch-shaped: several decode matmul families (DRAM-sharded weights in particular) are
  only legal at a decode-shaped `M`, so a batch-1-only decode contract would delete them from the
  search space and make the shard-advise arms indistinguishable from the no-advise arms for reasons
  that have nothing to do with the advisor.
- **functional-decoder stays untuned** — no hand-picked shard specs, matmul program configs, or
  per-core grids in the runtime forward, and framework-default compute-kernel config unless a
  recorded PCC failure forced otherwise. Only ops that need an L1-sharded input to run at all (paged
  cache update, decode SDPA, decode head-concat) get a minimal workload-derived layout. A
  pre-optimized functional decoder shrinks and hides the delta the experiment is trying to measure.

These three items were backported from `forge-functional-decoder`, where they were learned during the
forge-seeded optimize runs. They are generic (nothing about them depends on a forge emit), and the
plain `functional-decoder` skill did not have them.

## Reading the results

Per model and arm: `models/autoports/<model>/doc/optimized_decoder/README.md` (before/after traced
decode at batch 1 and `DECODE_BATCH`, chosen and rejected configs) plus `work_log.md`. The fused arms
additionally have `doc/fused_decoder/`. The shared baseline is
`doc/functional_decoder/` — identical across all four arms for a given model, by construction.

Full run plan, machine split, and exact commands: `docs/skillexp-run-plan.md`.
