# AutoFix Report: non-greedy traced sampling RNG

## Starting Evidence

- Source report: `AUTODEBUG_NON_GREEDY_SAMPLING.md`.
- Original defect: `Sampling1D._sample_topk()` puts `ttnn.manual_seed()` and
  `ttnn.sampling()` in the sampling trace. Gemma left a real seed in the
  persistent seed tensor, so every replay reset the PRNG and consumed the same
  random quantile.
- Scope was limited to the Stage 06 generator, its focused tests, and this
  report. Shared sampling and vLLM code were not changed.

## Hypothesis Experiments

- Hypothesis: leaving a real seed in the TP4 sampler trace repeats one draw;
  seeding once at the request boundary and replacing the trace-bound seed with
  `UINT32_MAX` lets `ttnn.sampling()` advance device PRNG state.
- Focused experiment: Gemma-physical `(1,4)` TP4 `Sampling1D`, vocab 262144,
  batch 1, max-top-k 32, FP32 broadcast candidate gather, constant logits,
  `k=32`, `p=1`, `temperature=1`, and 12 trace replays.
- Command:

  ```bash
  LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
  GEMMA4_31B_FULL_MODEL_RUN_NON_GREEDY_RNG=1 \
  pytest -q -s \
    models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_tp4_non_greedy_sampling_rng_trace_replay
  ```

- Baseline/A-B result before the generator fix: passed in 6.05 seconds. The
  fixed-real-seed trace returned one repeated token for all 12 replays. The
  seed-once/sentinel trace returned more than one token, and reseeding with 17
  reproduced the exact 12-token sequence.
- Verdict: verified.

## Fix

- `Gemma4Generator` now accepts explicit request seeds, with an entropy-derived
  default, and maps inactive fixed slots to the skip sentinel.
- `_initialize_non_greedy_rng()` copies real request seeds once, runs
  `ttnn.manual_seed()` once outside trace capture, synchronizes that request
  boundary, then copies `UINT32_MAX` into the persistent `Sampling1D` seed
  tensor.
- Non-greedy eager first-token sampling initializes its own sampler at the
  request boundary. The traced decode phase uses a distinct deterministic
  phase seed, avoiding reuse of the first-token quantile.
- Fresh and already-captured non-greedy request paths both reinitialize at the
  request boundary. Semantic greedy, force-argmax, explicit host-sampling
  compatibility, fixed-slot token feedback, page-table ownership, and trace
  keying are unchanged.
- `decode_next_token_traced()` contains no seed update, host copy, readback, or
  synchronization. Counters expose exactly two seed-buffer host copies and one
  seed initialization per non-greedy request boundary, with zero per-token seed
  traffic.

## Verification

- Delivered-helper TP4 verification, same command as above: `1 passed` in
  4.41 seconds. It additionally checked that the persistent seed tensor was
  `UINT32_MAX` before and after replay and that two request initializations
  produced four boundary copies total.
- Static full-model contract:

  ```bash
  LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
  pytest -q models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py
  ```

  Result: `22 passed` in 7.04 seconds.
- Syntax/diff checks:

  ```bash
  python -m py_compile \
    models/autoports/google_gemma_4_31b/tt/generator.py \
    models/autoports/google_gemma_4_31b/tests/test_full_model.py
  git diff --check -- \
    models/autoports/google_gemma_4_31b/tt/generator.py \
    models/autoports/google_gemma_4_31b/tests/test_full_model.py
  ```

  Result: passed.

## Final Status

- Fixed with focused TP4 hardware evidence.
- Alternating semantic-greedy and sampled parameters remain distinct in the
  existing sampling trace key; the contract test proves a key mismatch releases
  both traces before recapture. The greedy trace does not touch sampler seeds.
- Remaining risk: the focused probe isolates the exact sampler and trace
  lifecycle without loading Gemma weights. A later all-layer sampled qualitative
  run would add end-to-end distribution evidence, but it is not needed to prove
  or repair this RNG-state defect.
