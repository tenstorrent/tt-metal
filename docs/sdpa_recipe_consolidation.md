# SDPA recipe consolidation (task document)

Goal: replace the legacy streaming and non-streaming SDPA compute paths with a
small set of named recipes (A FAST, B COMPENSATED, C BALANCED, D ACCURATE,
E LOW_PRECISION). DiT variants go first: noncausal SDPA, joint SDPA, ring joint
SDPA and exp ring joint SDPA. Once these are done, follow-on work ports the
remaining variants (causal, decode, caches, MLA, ...) and deletes the legacy
kernels.

## Contract

- **The numerical implementation of A-E is the contract.** Each recipe fixes its
  matmul fidelities, destination/state formats, exponential, correction and
  normalization arithmetic, and rounding points. The accuracy envelope is what
  that implementation yields at given chunk sizes and input regimes; it is
  measured and documented, not the definition.
- Blocking is an execution detail. It may change reduction order (for example
  the first PV row group of each Q chunk) and so rounding, but not the recipe's
  arithmetic choices. Q256/K512/D128 stays bit-identical to the frozen digests.
- Inputs: SDPA consumes the tensors it is given. It does not prepare, round or
  verify inputs; see [recommended inputs](sdpa_recipe_inputs.md).

## Plan and status

| # | Task | Status |
| --- | --- | --- |
| 1 | Contract + input recommendations docs | done (5cffae00) |
| 2 | Generic kernel geometry: any tile-aligned Q/K/D within L1; Q256/K512/D128 fast path unchanged | dense + joint done and validated (sweep of 25 geometries x 7 variants x 3 regimes green on bh-38; frozen digests unchanged; ring 897 / exp 131 unchanged); ring/exp geometry in progress (`cglagovich/sdpa-recipe-ring-geometry`) |
| 3 | Op-selected blocking and grid; `program_config` becomes an optional override | code done on `cglagovich/sdpa-recipe-auto-blocking` @ 4ad8b204 (host tests pass); device runs and perf table queued |
| 4 | Recipe-owned program factories for ring and exp ring (no `#ifdef` forks in legacy kernels) | done (`cglagovich/sdpa-recipe-ring-factories` @ 4b65adf7, merged); legacy kernels byte-identical in 204/206 configs, ring/exp/continuation/mesh green, perf unchanged |
| 5 | FAST on the shared recipe loop (bit-identical to A's frozen digests) | planned |
| 6 | DiT gaps: masks, device-tensor logical lengths, exp ring geometry | masks done (`cglagovich/sdpa-recipe-masks` @ f365ffc8, merged; 131 mask tests, unmasked digests unchanged); lengths/exp geometry after task 4 |
| 7 | Parity gates, then default flip for the four ops; drop model compute configs and tuning tables | planned |
| 8 | Restack into reviewable PRs | planned |

Parity gates (task 7), per op: legacy test suites pass with the recipe default;
accuracy no worse than legacy on the qualification inputs; trace-wall time at
least legacy's at the models' tuned shapes; Galaxy / larger rings and
real-checkpoint quality qualified.

## Starting point

`cglagovich/sdpa-dit-adoption` (062cac92): whitelisted geometry (Q128-320 in
32-row steps, K256/384/512, D64/128/256; exp ring K512/D128), model opt-in for
11 DiT models, FAST parity with legacy. This branch is the reference and test
source; the consolidation lands on `cglagovich/sdpa-recipes-consolidate`.

## Task 2 notes

- Dense/joint bounds: Q chunk 32-1024 rows (recurrent-state arrays hold 32 tile
  rows), any tile-aligned K chunk, any tile-aligned D. L1 fit is the only other
  limit (`check_recipe_l1_fit`).
- QK/PV subblock widths are the largest of 4/2/1 dividing the K chunk and D
  tiles, passed as `SDPA_RECIPE_QK_W` / `SDPA_RECIPE_PV_W`.
- Granularity defines follow legacy SDPA's rules; the recipe loop does not read
  them, so they cannot move the frozen digests.
- Geometries outside the previously qualified set build paired BF16 recipes with
  pack-only `-Os` (kernel config buffer), like odd Q chunks already did.
- Ring and exp ring keep their current limits until task 4 moves them onto
  recipe-owned factories; `recipe_q_tiles` / `recipe_k_tiles` remain their checks.
- Test: `test_sdpa_recipe_geometry.py`. At K512/D128, Q384 fits L1 only for 3 variants and
  Q512/Q1024 for none; Q1024 is exercised at K128/D64 (3 variants fit).

## Task 2 findings

- One-tile BF16 Q chunks hung: the first PV group assumed two tile rows (fixed in 352a6f49).
- LOW_PRECISION hung whenever a QK or PV subblock was one tile wide (odd K chunks such as
  K32/K96/K352, all KV formats; the blocking agent's D64 E_bfp8 Q64/K352 report). LoFi no-MOP
  matmuls record the reuse-side source clear in the replay image, and `mm_no_mop_reinit_short`
  does not re-record it, so a 2x1 QK replay reused by a 2x4 PV matmul cleared the wrong source.
  Fixed in 3fda4059 (`recipe_mm_reinit`); qualified geometries keep their instruction stream.
  The legacy streaming kernel (`compute_streaming.hpp`) uses the same reinit, so legacy SDPA with
  a LoFi compute config and a one-tile-wide QK subblock is likely affected too (inferred from
  code, not reproduced).

## Task 6 notes (masks)

- Additive `attn_mask` `[1|B, 1|H, Sq, Sk]`, BF16/BFP8/BFP4 (FP32 for C/D), on CB 15.
  Added onto packed QK scores by L1 accumulation before the max; recipe arithmetic unchanged.
- The op pre-scales the mask by 1/scale like legacy; C/D do it in FP32.
- Key-padding mask == truncated K bit-identically (49 cases). Fully masked rows are finite (as legacy).
- Masked runs cost 1.3-2.4x unmasked (per-call pre-scale, dense mask reads, no reduce overlap).
  Open: key-padding fast path; pre-scale once.
- Joint SDPA has no `attn_mask` argument.

## Decisions log

- 2026-09-24: contract is the A-E numerical implementation (user).
- 2026-09-24: SDPA documents recommended dtypes/rounding and never prepares
  inputs; E callers own preparation (user).

## Open questions

- `inputs_prepared` is a caller acknowledgment SDPA cannot verify. Keep it as an
  explicit opt-in for E, or drop it and rely on documentation?
