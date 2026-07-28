# skillexp — isolating the contribution of graph-fusing and shard-advise

This branch family is a 2×2 ablation over the autoport bringup pipeline. It answers one question:
**how much of the optimize stage's speedup comes from `$graph-fusing`, and how much from
`$shard-advise`?**

Everything outside the two factors is held constant: same tt-metal commit, same skills, same
prompts, same functional decoder per model, same measurement convention.

## Branches

| Branch | `$graph-fusing` | `$shard-advise` |
|---|---|---|
| `mvasiljevic/qb2/skillexp/base` | shared parent — stage 01 runs from here | |
| `mvasiljevic/qb2/skillexp/fuse-advise` | yes | yes |
| `mvasiljevic/qb2/skillexp/fuse-noadvise` | yes | no |
| `mvasiljevic/qb2/skillexp/nofuse-advise` | no | yes |
| `mvasiljevic/qb2/skillexp/nofuse-noadvise` | no | no |

Each arm branch differs from `base` **only** by its factor edits. Nothing outside `.agents/`
differs at all, so one tt-metal build serves all four arms — never rebuild between arms.

## The factors

**Fusing is exactly one thing: does stage `01b-fused-decoder` run.** `$graph-fusing` was
de-referenced from `optimize/SKILL.md` on *every* arm, including the fuse arms (orientation step 1,
the "before knob tuning first use `$graph-fusing`" paragraph, and the evidence-checklist item).
Those three were a second, weaker channel for the same work, added for multichip topology testing
and worth nothing on a single chip where the 01b goal already owns the fusing contract. Leaving them
would blur the ablation in both directions: a "no fusing" arm still reading *run `$graph-fusing`
first* is not a clean control, and a "with fusing" arm would get the work counted twice.

- **fusing = yes** — `.agents/skills/graph-fusing/` and `01b-fused-decoder.txt` are present; the
  stage runs between 01 and 02.
- **fusing = no** — both are deleted. Unreachable, not merely unrequested.

**shard-advise = no.** The `shard-advise` skill directory is deleted, `OPT-015` and every
`$shard-advise` reference is removed from `optimize/SKILL.md`, the REQUIRED shard-advise bullet is
removed from `02-optimized-decoder.txt`, and `02-optimized-decoder.check.sh` (the hard gate that
fails the stage without `report.json` + `final_ir.mlir`) is deleted. What remains is the
pre-advisor-tool sharding guidance: the agent still owns on-device sharding, still chains L1-resident
ops on one shard spec (OPT-003), still sweeps DRAM-sharded decode matmuls (OPT-004), still has the
"decode activations width-sharded in L1" checklist item — it just derives the layout itself from the
perf report instead of querying a compiler tool. **This arm needs no tt-mlir checkout at all.**

**shard-advise = yes.** OPT-015 seeds the dense-path L1 layout from a real `ttnn-advise capture`, and
the runner-side gate enforces that the advisor actually ran this pass. Pinned to tt-mlir
`mvasiljevic/shard-advisor-dram-sharding` @ `618cd4e75d` — the DS-integrated advisor with the
`kNumDRAMBanks` Blackhole fix; see `skills/shard-advise/SETUP.md`.

Note the asymmetry this creates, and do not read it as bias: a hard gate can fail and trigger one
remediation goal, so the advise arms can receive an extra agent turn the no-advise arms cannot. That
is inherent to shipping the advisor as a gated tool and is part of what is being measured.

## Held constant across all four arms

Applied on `base`:

- **`DECODE_BATCH`** — traced decode is validated and measured at both batch 1 and the serving decode
  batch, and stage 02 sweeps layout/matmul candidates at both. Primary target stays batch-1 traced
  decode, matching `$optimize`. The reason to pin both is that advised and measured shard params
  differ by batch (a sub-tile activation yields `per_core_M = 1`, a full-tile one does not), so a
  config tuned at one batch is not evidence for the other — and an arm that reported only one batch
  would not be comparable with an arm that reported the other.

  Note for anyone reading an older version of this file: DRAM-sharded decode matmuls are **not**
  batch-32-only. tt-metal's constraint is `TT_FATAL(M == 1, "currently only support in0 tensor height
  of tile height")` in `matmul_device_operation.cpp`, where `M` is in tiles off the padded shape — so
  batch 1 pads to one tile row and passes, and batch 64 is what gets refused. The advisor used to
  carry a hardcoded `M % 32 == 0 && M / 32 == 1` that admitted exactly batch 32; tt-mlir
  `28bc859f9e` removed it, and the pinned `618cd4e75d` has no M gate at all. Batch 1 measured DS as
  the *fastest* option on QB2 (Qwen2.5-Coder DS-40c, 2.1503 ms).
- **functional-decoder stays untuned** — no hand-picked shard specs, matmul program configs, or
  per-core grids in the runtime forward, and framework-default compute-kernel config unless a
  recorded PCC failure forced otherwise. Only ops that need an L1-sharded input to run at all (paged
  cache update, decode SDPA, decode head-concat) get a minimal workload-derived layout. A
  pre-optimized functional decoder shrinks and hides the delta the experiment is trying to measure.

  These two were backported from `forge-functional-decoder`, where they were learned on the
  forge-seeded optimize runs. They are generic — nothing about them depends on a forge emit — and the
  plain `functional-decoder` skill did not have them.

- **the context/seq-len contract stays strict** — the forge-only goals (`prompts/forge_goals/`) and
  the forge functional-decoder skills are deleted, so their static-shape scope reduction ("a small
  and a larger `seq_len`", `decode_status = pending_emitted_decode_version`, "context below
  advertised") cannot reach a plain run. `check_context_contract.py` keeps the strict upstream gate:
  a context below HF-advertised without DRAM evidence is a critical failure. Supported context sets
  the KV-cache and L1 footprint, so an arm quietly shipping a short-context decoder would be
  optimizing a different problem than the others.

## Reading the results

Per model and arm: `models/autoports/<model>/doc/optimized_decoder/README.md` (before/after traced
decode at batch 1 and `DECODE_BATCH`, chosen and rejected configs) plus `work_log.md`. The fuse arms
additionally have `doc/fused_decoder/`. The shared baseline is `doc/functional_decoder/` — identical
across all four arms for a given model, by construction.

Full run plan, machine split, exact commands and handoff protocol: `.agents/RUN-PLAN.md`.
