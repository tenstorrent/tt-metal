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
| 2 | Generic kernel geometry: any tile-aligned Q/K/D within L1; Q256/K512/D128 fast path unchanged | dense + joint done and validated (sweep of 25 geometries x 7 variants x 3 regimes green on bh-38; frozen digests unchanged; ring 897 / exp 131 unchanged); ring/exp geometry done (`cglagovich/sdpa-recipe-ring-geometry` @ cfd70ad9, merged): any tile-aligned Q 32-1024/K/D within L1 for B-E; 98 prior-qualified ring/exp digests unchanged |
| 3 | Op-selected blocking and grid; `program_config` becomes an optional override | done (`cglagovich/sdpa-recipe-auto-blocking` @ 4ad8b204, merged); device checks green; auto/tuned trace time geomean 0.89 over 67 DiT cases (worst 1.10, H3 exp ring A) |
| 4 | Recipe-owned program factories for ring and exp ring (no `#ifdef` forks in legacy kernels) | done (`cglagovich/sdpa-recipe-ring-factories` @ 4b65adf7, merged); legacy kernels byte-identical in 204/206 configs, ring/exp/continuation/mesh green, perf unchanged |
| 5 | FAST on the shared recipe loop (bit-identical to A's frozen digests) | planned |
| 6 | DiT gaps: masks, device-tensor logical lengths, exp ring geometry | masks done (`cglagovich/sdpa-recipe-masks` @ f365ffc8, merged); device-tensor `logical_n`/`logical_l` (ring) and `logical_n` (exp ring) done with ring geometry; exp ring geometry done |
| 7 | No global default flip (user, 2026-09-24): every SDPA-variant call in `models/tt_dit` passes an explicit recipe; drop its compute configs and chunk tuning tables; per-model accuracy/speed parity gates vs its legacy setup | in progress |
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

## Merged-head validation (bh-38, b41e8cc8 + 2f05777f)

| Suite | Result |
| --- | --- |
| Dense geometry sweep (25 geometries, 7 variants, 3 regimes) + constant V | green (L1 skips only) |
| Dense regression (recipes, accuracy, Q/K chunks, head dims, joint, tails, GQA, preparation) | 922 passed |
| Frozen digests (recipes, accuracy, joint, tails) | 405 passed |
| Ring / continuation+mesh / exp ring | 714 / 183 / 131 passed |
| Masks | 131 passed |
| Blocking | 195 passed |
| DiT model host / model smoke / Ideogram4+LTX | 113 / 58 / 36 passed (one stale LTX test rewritten) |

## Task 7 status (`cglagovich/sdpa-dit-explicit-recipes`, validated on bh-38 @ a1e56fac)

- Every denoiser SDPA call in `models/tt_dit` selects a recipe on Blackhole (default FAST via
  `sdpa_precision_default`); Blackhole chunk tables removed; Wormhole keeps legacy configs.
- Left on legacy: D512 VAEs (L1), encoders (causal / decode / windowed / cu_seqlens / FP32 inputs).
- Parity harness (op level, DiT shapes, FAST vs tuned legacy HiFi2 / BF16 dest / exact exp):
  speed geomean 0.74x over 34 denoiser cases; error 0.85-1.05x legacy. Slower than 1.03x:
  LTX text cross 1.42x and A2V cross 1.50x (short-K chooser, being fixed), LTX V2A ring cross
  1.12x, H3 exp ring 4x32 1.11x.
- Speedup sources: joint shapes run the streaming recipe path instead of the legacy
  non-streaming joint kernel; FAST's approximate exp vs legacy exact exp; op-selected chunks.
  Ring (same legacy kernel for FAST) gains only from exp mode and chunks. Not yet decomposed.
- Tests: host 140, smoke 58, parity 148 passed.
- Open: Wormhole legacy path untested with these PRs; Galaxy meshes untested.

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

## Fixes after the DiT parity run (`cglagovich/sdpa-recipe-fixes-sizes-shortk` @ 351b534b, merged)

- exp ring B/E overflowed the kernel config buffer at the **default** worker L1 (70656 B) even at
  qualified geometries; earlier exp ring suites ran at `worker_l1_size=1344544` (~187 KB config
  buffer) and missed it. BF16-dest B/E exp ring builds now use pack+unpack -Os when the buffer is
  under 96 KiB; large-buffer builds are unchanged. Regression test:
  `test_sdpa_recipe_exp_ring_config_buffer.py` (default L1).
- Blocking cost model: per-core fill/drain and per-Q-chunk first-K/V fill terms; per-half subblock
  penalties; dense B/E odd-Q penalty. Calibration pick/best geomean 1.032 -> 1.012. Blocking perf
  auto/tuned geomean 0.886, worst 1.01 (was 1.20). h3 exp ring 4x32 FAST 1.10x -> 0.99x legacy.
- Remaining: short-K dense recipes (LTX text / A2V cross) cost ~30 us more than legacy at any
  blocking (1.35-1.41x); needs kernel/dataflow work. The exp ring K ranking at default L1 misranks
  non-512 K (h3 B/E 1.09-1.23x vs K512).

## Ring / exp ring geometry notes

- The single list of supported ring/exp geometry is `recipe_geometry_rejection`
  (`validate_recipe_geometry` for the op, the blocking chooser for auto chunks).
- FAST (A) keeps the legacy ring kernels and their qualified set (Q 128-320, K 256/384/512,
  D 64/128/256; exp K512/D128).
- Fixed: the ring recipe writer read `logical_n` from the state-tensor common arg (hang with
  device-tensor lengths); the exp ring writer used the reduce-scaler CB for the length.

## Task 6 notes (masks)

- Additive `attn_mask` `[1|B, 1|H, Sq, Sk]`, BF16/BFP8/BFP4 (FP32 for C/D), on CB 15.
  Added onto packed QK scores by L1 accumulation before the max; recipe arithmetic unchanged.
- The op pre-scales the mask by 1/scale like legacy; C/D do it in FP32.
- Key-padding mask == truncated K bit-identically (49 cases). Fully masked rows are finite (as legacy).
- Masked runs cost 1.3-2.4x unmasked (per-call pre-scale, dense mask reads, no reduce overlap).
  Open: key-padding fast path; pre-scale once.
- Joint SDPA has no `attn_mask` argument.

## Task 3 notes (op-selected blocking)

- `q_chunk_size`/`k_chunk_size` default to 0 (op-selected); explicit values are honored.
- Cost model: per-variant roofline fitted to one 8192x8192 shape; Q256/K512 preferred within 5%.
- Open: the chooser does not budget the mask CB (masked auto blocking relies on run_recipe's
  L1 check); the FAST legacy-ring L1 mirror is conservative (rejects Wan's tuned Q288/K512).

## Decisions log

- 2026-09-24: contract is the A-E numerical implementation (user).
- 2026-09-24: SDPA documents recommended dtypes/rounding and never prepares
  inputs; E callers own preparation (user).
- 2026-09-25: DiT denoiser attention defaults to FAST (user). VAE attentions keep BALANCED
  (legacy used FP32 dest; FAST there is ~3x legacy error on peaked softmax).
- 2026-09-24: no implicit default flip for SDPA callers in general (too many callers); instead
  every SDPA-variant invocation under `models/tt_dit` selects a recipe explicitly (user).

## Open questions

- `inputs_prepared` is a caller acknowledgment SDPA cannot verify. Keep it as an
  explicit opt-in for E, or drop it and rely on documentation?
