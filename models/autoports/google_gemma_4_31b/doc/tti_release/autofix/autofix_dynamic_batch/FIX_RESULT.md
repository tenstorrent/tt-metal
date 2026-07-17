# AutoFix Report: Gemma 4 31B dynamic decode batch grid

## Starting Evidence

- Source: `AUTODEBUG.md` in this directory.
- Original failure: the full 32-request IFEval workload reached active batch
  19 and `MultichipDecoder._decode_attention_tp` raised
  `ValueError: max() arg is an empty sequence` while requiring one exact
  rectangular shard grid.
- Corrected runtime evidence reports an 11x10 logical worker grid. On that
  grid, the same divisor rule is undefined for active batches
  13, 17, 19, 23, 26, 29, and 31. The originally analyzed 14x10 geometry is
  undefined for 17, 19, 23, 29, and 31.

## Hypothesis Experiment

- Hypothesis: dynamic decode needs the Qwen-proven pair of (a) an exact
  row-wise multi-range shard grid when no rectangle fits and (b) a full worker
  grid `sub_core_grids` argument for the concat-heads subcore program.
- Focused experiment: construct grids for every batch 1..32 on both 11x10 and
  14x10 using a CPU fake of the TTNN core-range types, then independently run
  the same matrix with the installed real TTNN core-range implementation.
- Result: verified. Every grid contains exactly one core per active user and
  remains in bounds. The real TTNN probe selected irregular sets
  `[13, 17, 19, 23, 26, 29, 31]` on 11x10 and
  `[17, 19, 23, 29, 31]` on 14x10.
- Verdict: **verified and fixed**.

## Fix

- Added `tt/decode_head_grid.py` as the shared autoport-local implementation.
- Rectangular batches preserve the previous largest-fitting-divisor geometry.
- Non-factorable batches use
  `ttnn.num_cores_to_corerangeset(..., row_wise=True)` without padding.
- Multi-range inputs receive a full physical worker-grid `sub_core_grids`;
  rectangular inputs continue to pass `None` and use the original concat
  program path.
- Wired the same computed `core_grid` and matching subcore selection through:
  - `tt/multichip_decoder.py`
  - `tt/optimized_decoder.py`
  - `tt/fused_decoder.py`
- Added `tests/test_decode_head_grid.py` with exhaustive core count/bounds,
  exact 11x10 and 14x10 irregular ranges, preserved rectangular geometry,
  subcore behavior, invalid-capacity checks, and AST verification of all three
  concat call sites.

## Verification

- Focused CPU-only suite:

  ```text
  pytest -q --confcutdir=models/autoports/google_gemma_4_31b/tests \
    models/autoports/google_gemma_4_31b/tests/test_decode_head_grid.py
  146 passed in 0.18s
  ```

- Real TTNN host-only range probe: all batches 1..32 passed on both 11x10 and
  14x10; exact irregular sets matched the expected lists above.
- Broader CPU-only autoport contract suite:

  ```text
  pytest -q --confcutdir=models/autoports/google_gemma_4_31b/tests \
    models/autoports/google_gemma_4_31b/tests/test_mlp_dtype_geometry.py \
    models/autoports/google_gemma_4_31b/tests/test_vllm_adapter_contract.py \
    -k 'not max_context_gate'
  28 passed, 1 deselected
  ```

  The one deselected test requires the repository-root `expect_error` fixture;
  its max-context behavior is unrelated to this grid change.
- A broader static run also produced 32 passes and one existing unrelated
  failure in `test_cache_update_dtype_contract.py`: its source assertion looks
  for `if config.cache_position_modulo is None:` in the current baseline
  decoder, where that text is already absent. No cache-update code was changed.
- `python3 -m py_compile` passed for all five changed/new Python files.
- Black reports the two new files unchanged. The existing full
  `fused_decoder.py` has three pre-existing formatting deltas outside this
  patch; the other changed decoder files are unchanged by Black.
- Scoped `git diff --check` passed. Repository-wide `git diff --check` remains
  noisy only because the pre-existing generated `readiness_vllm/server.log`
  contains CRLF/trailing whitespace.

## Final Status

**Fixed with CPU/host evidence.** No hardware command, live-server action, or
commit was performed by this fix agent. Hardware validation of the irregular
concat program and the staggered 32-request workload remains with the parent
release stage.
