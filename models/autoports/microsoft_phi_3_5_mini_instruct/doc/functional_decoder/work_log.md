# Work log

Date: 2026-07-28 UTC

- Started from repo HEAD `b9e6c242a34011e3daeebab9207fbb5b79750f39`.
- Implemented the correctness-first Phi-3.5 dense decoder layer, paged
  prefill/decode, LongRoPE, real-weight loading, and trace-safe on-device
  current positions.
- Correctness tests cover the only meaningful target layer kind, permuted page
  tables, page/tile boundaries, long non-aligned context, real weights,
  full-context decode, deterministic repeat inputs, and traced batch 1/32.
- AutoDebug isolated the capture-execution versus steady-replay state effect.
  AutoFix proved replay/eager equivalence, fixed Blackhole batch-32
  head-concat with the required 8x4 one-core-per-user rectangle, and repaired
  long prefill with bounded standard-SDPA chunks and TTNN-only offset masks.
- Tracy command:
  `python -m tracy -r -p -v -m pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/functional_decoder_perf.py`
- Correctness command:
  `pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py`
- Watcher command and exact artifacts are recorded in `README.md`.
- No optimized-decoder, multichip, full-model, or vLLM work was started.
- Initial independent stage review returned `more-work-needed`; its findings
  were addressed with batch-2 prefill PCC, nonzero-position traced reference
  PCC at batch 1/32, long-RoPE trace PCC, exact 131072 prefill, nonzero 32769
  boundary PCC, HF layer/rotary controls, real-weight statistics, and refreshed
  completion documentation.
- Final rereview verdict and local stage commit SHA are appended after review.
- Exact-final-runtime correctness: `functional_decoder_tests_final_runtime.log`
  plus the corrected distinct-position batch-32 oracle rerun
  `trace_distinct_positions_b32.log`.
- Exact-final-runtime watcher: `watcher_final/watcher_console.log` (3 passed)
  and `watcher_final/generated/watcher/watcher.log`.
- Exact-final-runtime Tracy: `tracy_final/profile_console.log` and
  `tracy_final/ops.csv`; derived human-readable/CSV reports share that folder.
- Terminal independent stage-review verdict: `clean-pass`.
- Stage implementation/evidence commit: `27507afd469`.
