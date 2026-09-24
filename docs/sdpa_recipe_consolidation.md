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
| 2 | Generic kernel geometry: any tile-aligned Q/K/D within L1; Q256/K512/D128 fast path unchanged | dense + joint implemented (b3c7d066), validating on bh-32; ring/exp follow task 4 |
| 3 | Op-selected blocking and grid; `program_config` becomes an optional override | in progress (`cglagovich/sdpa-recipe-auto-blocking`) |
| 4 | Recipe-owned program factories for ring and exp ring (no `#ifdef` forks in legacy kernels) | in progress (`cglagovich/sdpa-recipe-ring-factories`) |
| 5 | FAST on the shared recipe loop (bit-identical to A's frozen digests) | planned |
| 6 | DiT gaps: masks, device-tensor logical lengths, exp ring geometry | masks in progress (`cglagovich/sdpa-recipe-masks`; Ideogram4, LTX-2); lengths/exp geometry after task 4 |
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
- Test: `test_sdpa_recipe_geometry.py`.

## Decisions log

- 2026-09-24: contract is the A-E numerical implementation (user).
- 2026-09-24: SDPA documents recommended dtypes/rounding and never prepares
  inputs; E callers own preparation (user).

## Open questions

- `inputs_prepared` is a caller acknowledgment SDPA cannot verify. Keep it as an
  explicit opt-in for E, or drop it and rely on documentation?
