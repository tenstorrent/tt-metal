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
| 1 | Contract + input recommendations docs | in progress |
| 2 | Generic kernel geometry: any tile-aligned Q/K/D within L1; Q256/K512/D128 fast path unchanged | planned |
| 3 | Op-selected blocking and grid; `program_config` becomes an optional override | planned |
| 4 | Recipe-owned program factories for ring and exp ring (no `#ifdef` forks in legacy kernels) | planned |
| 5 | FAST on the shared recipe loop (bit-identical to A's frozen digests) | planned |
| 6 | DiT gaps: masks, device-tensor logical lengths, exp ring geometry | planned |
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

## Decisions log

- 2026-09-24: contract is the A-E numerical implementation (user).
- 2026-09-24: SDPA documents recommended dtypes/rounding and never prepares
  inputs; E callers own preparation (user).

## Open questions

- `inputs_prepared` is a caller acknowledgment SDPA cannot verify. Keep it as an
  explicit opt-in for E, or drop it and rely on documentation?
