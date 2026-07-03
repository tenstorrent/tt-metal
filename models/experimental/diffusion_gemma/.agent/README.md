# DiffusionGemma Bring-up Agents

This directory is a **DiffusionGemma-specific** adaptation of the autonomous model-bringup
harness. It points AI coding agents at the in-repo DiffusionGemma module
(`models/experimental/diffusion_gemma/`) and drives it through the remaining bring-up work —
correctness, long-context fit, precision, perf, and block-granular vLLM serving on QB2.

It was forked from a generic *autoregressive* LLM autoport pipeline. The generic skills still
live in `skills/`, but they are now overridden for the **text-diffusion** path. The keystone is
`skills/diffusion-gemma/SKILL.md` (`$diffusion-gemma`): every stage loads it first, and it is the
authority that maps each generic autoregressive assumption onto the diffusion path.

```text
models/experimental/diffusion_gemma/.agent/
  prompts/model_bringup_diffusion_gemma/   # the ten DiffusionGemma stage goals, one prompt each
  skills/diffusion-gemma/                   # $diffusion-gemma — shared model context + hard rules
  skills/                                   # the generic TTNN skills, overridden for diffusion
  scripts/multigoal                         # the runner that chains the stages together
```

## This is not a greenfield bring-up

The DiffusionGemma module is **already substantially built and has RUN on QB2**. Foundation, the
KV-phase machine, bidirectional masked SDPA (incl. the >32768 long-prompt path), on-device canvas
sampling, and the decode-loop control flow are built and validated; the first real-26B multi-block
prompt→text run passed on QB2. The stages below are therefore **validate / harden / optimize /
integrate the existing code**, ordered by the issue-map roadmap — not "author a decoder from
scratch". Read `models/experimental/diffusion_gemma/plan.md` Part 0 for live status.

## The two hard rules (read before anything)

1. **Never edit the shared Gemma-4 backbone.** The backbone is `models/demos/gemma4/`, reused
   as-is. Every DiffusionGemma fix stays inside `models/experimental/diffusion_gemma/`. The gate
   `prompts/model_bringup_diffusion_gemma/check_no_shared_gemma4_edits.sh` enforces this (set
   `DG_BASE_REF` to the ref the work branched from — `main` on a main-based dev branch).
2. **Correctness is judged on diffusion decisions, not teacher-forcing top-k.** No AIME24 top-1/5.
   Validate entropy values, Gumbel-max argmax agreement, and accept/renoise agreement vs the torch
   reference, with the torch run's Gumbel noise injected for token-exact comparison. See
   `$diffusion-gemma`.

## The stages

| Stage | Delivers | Issues |
|---|---|---|
| 01 backbone-parity | The reused gemma4 backbone validated against the DiffusionGemma checkpoint (weight mapping + causal PCC on QB2) | #47461 #47468 |
| 02 diffusion-delta | The net-new device pieces validated: bidirectional attention, three-phase KV, self-conditioning, canvas sampling | #47462 #47474 #47472 |
| 03 denoise-loop | The discrete-diffusion decode loop proven correct and trace-safe (fixed step budget, on-device cutoff mask) | #47463 |
| 04 block-diffusion-run | The end-to-end prompt→text multi-block RUN hardened on QB2 (RUN-first; degenerate output OK) | #47464 |
| 05 decision-fidelity | The #48291 decision-fidelity bar measured and decided (engineering vs product-accept) | #48291 |
| 06 qb2-longcontext | The full 256K memory budget + long-context fit on the (1,4) mesh, incl. canvas scratch + non-causal mask | #47487 |
| 07 datatype-sweep | The fastest precision config that preserves the diffusion decisions | #47465 #47475 |
| 08 optimize-perf | The denoise-step / block path optimized and traced | #47465 |
| 09 vllm-integration | Block-granular serving through the tenstorrent/vllm TT plugin | #47466 #47488 |
| 10 release-ci | The release handoff + tiered models-CI wiring | tti-release #47489 |

`$diffusion-gemma` is loaded by every stage. `$tt-device-usage` is loaded by every
hardware-facing stage. The `$skill` references inside each prompt attach the generic skill whose
knowledge that stage reuses; that skill's "DiffusionGemma adaptation" section states the overrides.

## How the generic skills map

- **Reused as-is** (model-agnostic TTNN/infra): `autodebug`, `autotriage`, `models-ci`, `tt-lang`,
  `tt-device-usage`, and the bulk of `optimize` / `tt-enable-tracing` TTNN knowledge.
- **Lightly noted** (reused process, with a short `## DiffusionGemma note` that re-points scope and
  examples to the diffusion path): `autofix`, `code_quality_review`, `beautify`, `qualitative-check`.
- **Adapted** (a "DiffusionGemma adaptation" section overrides the autoregressive assumptions):
  `functional-decoder`, `full-model`, `datatype-sweep`, `multichip`, `optimize`,
  `tt-enable-tracing`, `vllm-integration`, `tti-release`, `stage-review`.
- **Not applicable**: `forge-functional-decoder` — the backbone is the hand-written gemma4 code,
  not a tt-forge emit; this pipeline does not use it.

## Context-length contract

The target is the HF-advertised 262144 (256K) context for `prompt + generated`; that is a capacity
limit, not a requirement that prompts be 256K tokens. Do not quietly reduce it. A reduction is
acceptable only for a hard physical device limit (QB2 DRAM), with a byte calculation or a failed
capacity probe as evidence — and the budget must include the per-step canvas K/V scratch and the
non-causal long-context mask buffers, not just weights + KV. The artifact is
`models/experimental/diffusion_gemma/doc/context_contract.json`; stages create/verify it.

Each stage leaves evidence under `models/experimental/diffusion_gemma/doc/<stage>/`: `README.md`
(results), `work_log.md` (the journey), and the backing artifacts.

## Verification gates

After a stage goal completes, the runner runs a sibling check script (e.g.
`04-block-diffusion-run.check.sh`) if present. Exit codes: `0` pass, `1` advisory (one remediation
attempt, then continue), `2` critical (one remediation attempt, then stop later stages), `3` checker
error. `check_no_shared_gemma4_edits.sh` is the reusable backbone-isolation gate; the RUN stage
check calls it. Add diffusion-appropriate gates per stage (e.g. a decision-agreement floor) rather
than an autoregressive degeneracy check.

## Running it

```bash
python models/experimental/diffusion_gemma/.agent/scripts/multigoal \
  --replace MODEL_DIR=models/experimental/diffusion_gemma \
  models/experimental/diffusion_gemma/.agent/prompts/model_bringup_diffusion_gemma/*.txt
```

Stages are model-specific (no `HF_MODEL` substitution needed). The DiffusionGemma checkpoint is
resolved via `DG_CKPT` (as the device tests do). Use `--start-index N` to resume from stage N,
`--dry-run` to validate the sequence. The 4000-char post-substitution objective cap still applies;
`scripts/check_agent_prompt_lengths.py` measures it.

## Extending it

- **New knowledge** goes in a skill; keep model facts and hard rules in `$diffusion-gemma` so every
  stage inherits them. Prefer a check over prose — enforcement generalizes.
- **New gate**: add `<prompt-stem>.check.sh`, scope it to `models/experimental/diffusion_gemma`,
  and follow the exit-code convention. Calibrate any threshold against known-good and known-bad
  artifacts before trusting it.
