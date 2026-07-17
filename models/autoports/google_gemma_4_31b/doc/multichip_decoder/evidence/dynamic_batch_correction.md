# Post-Stage-04 dynamic-batch correction

## Starting evidence

Stage 11's full 32-request IFEval workload exposed a latent Stage 04 decoder
bug at active batch 19. The original Stage 04 implementation commit
`683adda7a3d12cc060df9ab3a36f1fd506eef234` required the active users to form
one rectangular `CoreRange`. On the target P150 11x10 runtime worker grid that
predicate has no solution for active batches 13, 17, 19, 23, 26, 29, and 31,
so `max()` raised before head concatenation.

The original Stage 04 batch-32 regression did not cover dynamic partial
batches. The prior clean reviews are superseded on this one point by the
downstream runtime evidence. The complete source diagnosis is retained at
`../../tti_release/autofix/autofix_dynamic_batch/AUTODEBUG.md`.

## Verified fix

Local commit `97a16e1c982a27fbc2f4e27b65dbd6b077f9e34f` added the shared
`tt/decode_head_grid.py` planner and wired it into `tt/multichip_decoder.py`.
Factorable batches retain the original rectangular grid. Non-factorable
batches use an exact row-wise multi-range `CoreRangeSet`, with the matching
full-worker-grid `sub_core_grids` contract passed to
`nlp_concat_heads_decode`. No user padding, cache padding, context reduction,
or host data fallback was introduced.

The `$autofix` report is retained at
`../../tti_release/autofix/autofix_dynamic_batch/FIX_RESULT.md`.

## Host regression

Command rerun on 2026-07-17:

```bash
pytest -q --confcutdir=models/autoports/google_gemma_4_31b/tests \
  models/autoports/google_gemma_4_31b/tests/test_decode_head_grid.py
```

Result: `146 passed in 0.17s`.

The suite exhaustively checks batches 1-32 on 11x10 and 14x10 grids, exact
irregular ranges, core counts and bounds, subcore selection, invalid capacity,
and AST wiring for the multichip, optimized-single-chip, and fused decoders.

Bound source hashes:

- `tests/test_decode_head_grid.py`:
  `2f98b1e3688292fdff1da37f92f3bf628f7f5e2bd9fbf88879343a4bc205b896`
- `tt/decode_head_grid.py`:
  `63ee6f54ba10bbf625255f2dcf41a87b005c84881bdae306b641544126d31ec8`

## Target-mesh runtime validation

The repaired P150x4 server prepared and replayed decode traces for every
previously unsupported active batch. The tracked
`readiness_vllm/server.log` records `Gemma 4 decode traces ready` for active
batches 13, 17, 19, 23, 26, 29, and 31. Its SHA-256 is
`54bcf3b6f65654590b86a5a6dab2c430c61438cb23aed96655b868838485b264`.

The post-fix release then completed all 541 IFEval requests and all 17
benchmark points with zero failed requests, including explicit concurrency-13
and concurrency-26 runs. The tracked authoritative workflow log is
`../../tti_release/tti_release_final6.log`, SHA-256
`b479faf832817336a23a496d6044ac940256d19fa0753005a7407101c9982df3`.
`../../tti_release/RUN_NOTES.md` records the same dynamic-batch coverage.

This is stronger than a synthetic mesh-only probe: it exercises trace capture
and replay, local attention heads, head concatenation, cache ownership, and
the serving decoder with real checkpoint weights on the target mesh.

## Final status

Fixed. The cumulative repository state contains the correction and regression
tests, the full advertised context remains unchanged, and no Stage 04
performance topology was weakened. The Stage 11 release remains blocked only
on unrelated mandatory Meta reference/threshold evidence.
