# Device Gumbel noise is not IID across canvas positions (#48291)

**Status 2026-07-25: CONFIRMED on device (P150x4, QB2). Root-caused to `ttnn.rand`. No DG-local
layout or seeding workaround fixes it. `DG_VLLM_GUMBEL_MODE=host` is the only IID option today,
and the shipped default (`device`) is the WORST of all device arms on the functional metric.**

## The blind spot this closes

`sample_gumbel_noise_with_permuted_vocab` (`tt/sampling.py`) exists to keep the vocab axis off
`ttnn.rand`'s innermost axis, because QB2's rand shows last-dimension correlation. It collapses
every non-vocab axis into one trailing axis and draws `ttnn.rand((vocab, inner))` — and for the
production logits shape `(1, 1, 256, vocab)` that `inner` **is the 256 canvas positions**. The
known last-dim correlation was therefore never removed; it was *relocated onto the canvas-position
axis*, which is the axis the diffusion decisions are taken along.

The only gate on this path, `tests/test_device_canvas_sampling_dist.py`, cannot see it: it
averages over a sample axis into per-position marginals, and correlation *between* positions
leaves every marginal correct. Independence across positions was never tested.

New gate: `tests/test_device_gumbel_position_correlation.py` (fails today, by design).

## Metrics

Two, both calibrated against a host torch-Gumbel IID control drawn at the same shape:

* **exact-duplicate rows** — how many of the 256 position rows are byte-identical to another;
* **flat-logit winner multiplicity** — with flat logits the winner at each position is the argmax
  of that position's noise row. Over a vocab this large, 256 IID winners essentially never
  collide, so `distinct_winners` ≈ 256 and `max_mult` ≈ 2. This is the *functional* metric: it is
  exactly the "synchronized same-token burst" texture, and it is worst where the logits are
  flattest — the deep-block regime where degeneration is observed.

## Results (canvas 256, vocab 16384, one seed; vocab 262144 identical where measured)

| arm | unique rows | distinct winners | max_mult | max abs r | mean abs r |
| --- | --- | --- | --- | --- | --- |
| host torch Gumbel (IID reference) | 256/256 | **255/256** | **2** | 0.035 (4.5σ) | 0.0062 |
| **`permuted` — the shipped default** | 192/256 | **119/256** | **11** | **1.00000** | 0.0249 |
| `chunked`, chunk=1024 (serving default size) | 160/256 | 156/256 | 6 | 1.00000 | 0.0209 |
| `chunked`, chunk=2048 | 160/256 | 155/256 | 6 | 1.00000 | 0.0186 |
| `plain` (vocab innermost, diagnostic) | 160/256 | 157/256 | 4 | 1.00000 | 0.0175 |

`max abs r = 1.00000` is not "highly correlated" — it means whole noise rows are **identical**.

## Root cause: `ttnn.rand` reuses 24 of every 32 row streams

The duplication is exactly periodic and the prediction was confirmed element-for-element:

* the duplicate set is **exactly** `{i : i % 32 >= 24}` — 8 of every 32 rows, 64 of 256;
* every duplicate row equals row `i - 24`, i.e. within each 32-row TILE, rows 24..31 repeat
  rows 0..7;
* the pattern is **independent of the other axis extent** — identical at vocab 16384 and 262144;
* it is present in the **raw** `ttnn.rand((vocab, 256))` output, before DG's permute/reshape, so
  DG's layout code is exonerated. The defect is in the op.

The vocab-innermost layout has the same class of defect with a different constant (offset 17,
96 of 256 duplicated), so it is a property of `ttnn.rand`'s row-stream assignment, not of which
semantic axis happens to be innermost. Worth an upstream issue.

## Two DG-local workarounds tested — both insufficient

Both remove **all** exact duplication and still fail the functional metric:

| candidate | unique rows | distinct winners | max_mult |
| --- | --- | --- | --- |
| A: 11 × `rand((vocab, 24))`, distinct seed per chunk | 256/256 | 156/256 | 10 |
| B: one `rand((vocab, 352))`, keep columns with `col % 32 < 24` | 256/256 | 167/256 | 5 |
| (host IID, for scale) | 256/256 | 255/256 | 2 |

So there is a **second, deeper defect beyond stream reuse**: even with 256 distinct rows, the
rows remain correlated in value, and their argmaxes still coincide far more often than IID
allows. Chunking along vocab with distinct seeds (`chunked`, already implemented and wired)
does not fix it either. No cheap layout or seeding change reaches IID.

## Tried in ttnn and REVERTED: the per-core seed was not the cause

`rand_program_factory.cpp` seeds core `i` with `seed + i + device_seed_offset` — consecutive
integers straight into per-core PRNGs, which is poor practice and was the obvious suspect. It was
implemented (a SplitMix64 finalizer applied to the same linear index, so the documented sharding
law and `test_rand_mesh_shard_matches_single_device` both still hold), built, and measured
against the reverted baseline on the same device:

| metric (64 tile-rows, width 16384) | without the mix | with the mix | host IID |
| --- | --- | --- | --- |
| exact duplicate row pairs | 0 | 0 | 0 |
| max abs r | 0.04423 (5.7σ) | 0.05046 (6.5σ) | 0.02780 (3.6σ) |
| distinct argmax winners | 63/64 | 64/64 | 64/64 |

Indistinguishable, and the canvas-position metrics did not move either (permuted 119 -> 120
distinct winners; the `{i : i % 32 >= 24}` duplication was byte-for-byte unchanged). **Cores are
already independent** — `init_prng_seed` evidently scrambles internally — so the change was pure
churn on a shared op that alters the output of every `ttnn.rand` caller, and it was reverted
rather than shipped.

What that experiment *does* buy is attribution: the defect is **inside a single core's tile**, in
the SFPU PRNG path `compute_uniform.cpp` -> `ckernel_sfpu_rand.h` (8 SFPU draws per face, 4 faces
per tile), along the tile's WIDTH axis. Fixing it means changing the SFPU sequence, which affects
every consumer of `rand`/`randn`/`uniform`, so it needs SFPU/hardware knowledge and its own
validation rather than a guess from here.

Filed as a ttnn-side regression instead:
`tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py` — two xfail(strict)
tests pinning the two width-axis properties, plus a passing cross-tile control that is exactly
the arm which rules out the seeding hypothesis. strict=True means it flips to a failure the day
the op is fixed, which is the signal to delete the xfail.

## What this means for the open decisions

1. **The owed sub-40 host-vs-device re-gate is not bookkeeping.** `DG_VLLM_GUMBEL_MODE` defaulted
   to `device` on the strength of W4 *marginal* validation. The distributions differ structurally
   along the decision axis, so the re-gate is measuring a real change, and the mechanism now
   predicts its sign and its shape: cross-position synchronized same-token bursts, worst in deep
   blocks where the logits flatten.
2. **The docstring claim that the permuted path "avoids that correlation" is measured false** and
   has been corrected in `tt/sampling.py`.
3. **`host` is the only IID arm.** It costs the per-step host RNG and the replicated PCIe DMA that
   `device` was introduced to remove, so this is a real correctness-vs-throughput decision, not a
   free fix — but it should be made with these numbers in hand, and it is a single-variable arm
   under the one-knob-per-arm rule.

## Reproduce

```bash
# the permanent gate (fails today)
DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_gumbel_position_correlation.py -s
# the production 262144 geometry
DG_RUN_DEVICE=1 DG_GUMBEL_CORR_FULL_VOCAB=1 pytest \
  models/experimental/diffusion_gemma/tests/test_device_gumbel_position_correlation.py -s -k production_vocab
```

Root-cause and workaround scans: `probe_gumbel_dup_structure.py`, `probe_gumbel_tile_fix.py`,
`probe_gumbel_chunked_arm.py` (same directory).
