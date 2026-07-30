# AutoFix Report

## Starting Evidence

- Source: `AUTODEBUG.md`, hypotheses H1 and H5.
- Original failing check:
  `test_decode_trace_replay_is_deterministic[mesh_device0-1]`.
- Scope was limited to the Phi functional-decoder test and this report; the
  runtime implementation was not changed.

## Hypothesis Experiments

- Hypothesis: H1, the strict equality failure is harmless BF16 numerical drift.
  Experiment: print `torch.equal`, mismatch count, maximum/mean absolute
  difference, finiteness, and PCC for every pair in capture/R1/R2/R3; compare
  three synchronized eager forwards using fresh identical caches.
  Result: refuted at batch 1. Capture versus each replay differed in 3071
  elements, max/mean absolute difference 4.484375/0.8538217, and PCC
  -0.0012316. R1/R2/R3 were mutually bitwise equal, finite, and PCC 1.0. The
  three fresh-cache eager results were also mutually bitwise equal and PCC
  1.0. R1 and eager E1 were bitwise equal and PCC 1.0.
  Verdict: refuted. This is not small BF16 drift.
  Verification:
  `timeout 900 pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_decode_trace_replay_is_deterministic -k '1 and not 32'`
  passes after gating the steady-state replay outputs.

- Hypothesis: H5, the capture execution has a one-time capture-state effect and
  should not be treated as a steady-state replay result.
  Experiment: the same full pairwise matrix plus the fresh-cache eager control.
  Result: verified at batch 1. Capture differs identically from R1/R2/R3, while
  all three replays agree bitwise with each other and with eager E1.
  Verdict: verified.
  Fix: test-only gate now requires R1/R2/R3 bitwise equality, three fresh-cache
  eager results to be bitwise equal, and R1 to be bitwise equal to eager E1.
  Capture metrics remain printed as diagnostics. No runtime change was made.

- Hypothesis coverage at serving batch 32.
  Experiment:
  `timeout 900 pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_decode_trace_replay_is_deterministic -k '32'`
  Result: the eager compile pass fails before trace capture in
  `ttnn.experimental.nlp_concat_heads_decode` with `RuntimeError: bad optional
  access`; therefore no batch-32 H1/H5 matrix can be collected from the current
  runtime.
  Verdict: still uncertain at batch 32; this is a separate runtime blocker, not
  evidence for either H1 or H5.

## Final Status

- H1 refuted and H5 verified for batch 1.
- Batch-1 steady-state trace determinism is proven by exact replay/replay and
  replay/eager equality.
- Batch 32 remains blocked before trace capture by decode head-concat. The
  runtime was deliberately not modified in this focused experiment.

## Batch-32 Decode Head-Concat Follow-Up

- Hypothesis: H2/layout, the height-sharded tensor passed from paged SDPA
  through `to_memory_config` to `nlp_concat_heads_decode` uses a non-rectangular
  core set at batch 32. The Blackhole compute grid is 13 cores wide, so
  `num_cores_to_corerangeset(32, ..., row_wise=True)` describes 13+13+6 cores;
  the decode concat kernel requires one rectangular range.
  Experiment: compare the Phi layout builder with the working Gemma4 and
  GPT-OSS decode paths, replace only the core-set construction with the widest
  exact rectangle that fits the runtime grid, and probe the resulting tensor
  immediately before concat.
  Result: verified. The focused probe recorded logical and padded shape
  `[1,32,32,96]`, HEIGHT_SHARDED L1 memory, shard shape `[32,96]`, ROW_MAJOR
  orientation, and the single rectangular core range `(0,0)..(7,3)` (8x4).
  `nlp_concat_heads_decode` then ran successfully.
  Verdict: verified.
  Evidence artifacts:
  `/tmp/phi_batch32_concat_shape_probe.log` and
  `/tmp/phi_batch32_concat_fix.log`.
  Fix: `_decode_concat_memory_config` now derives the widest exact rectangular
  factor from `(batch, device grid)` and constructs one `CoreRangeSet`. This
  preserves the minimal one-core-per-user workload layout and existing
  `[TILE_SIZE, head_dim]` shard; it adds no tuned grid, program config, or
  compute-kernel policy.
  Verification:
  `timeout 900 pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_decode_trace_replay_is_deterministic`
  passed for batch 1 and batch 32 (`2 passed`). At both batches, R1/R2/R3 were
  mutually bitwise equal and R1 was bitwise equal to the fresh-cache eager
  control. Final log:
  `/tmp/phi_trace_determinism_b1_b32_final.log`.

## Updated Final Status

- Fixed: batch-32 decode now reaches trace capture/replay and proves exact
  steady-state determinism.
- Batch 1 remains passing; the layout correction introduces no regression.
- The capture-execution tensor remains different from steady-state replay at
  both batches, consistently with the separately verified H5 capture-state
  effect; replay/replay and replay/eager gates pass exactly.

## Blackhole Long-Prefill Follow-Up

- Starting evidence:
  `long_prefill_32769.log` showed the real 32769-token model path failing at
  the Python binding for paged chunked SDPA. The exposed signature named a
  Wormhole-only compute config even when that optional argument was omitted.
- Hypothesis experiments:
  an explicit Wormhole config and positional `chunk_start_idx` both reproduced
  the binding failure. Conversely, a minimal Blackhole mesh call succeeded, so
  generic Blackhole/mesh incompatibility was refuted; the failure depends on
  the real invocation shape/state.
- Fix:
  for sequences over 32768, the model-local functional path now runs ordinary
  SDPA on query chunks no larger than 32768. The first prefix uses causal SDPA
  against matching prefix K/V; subsequent chunks use full K/V plus a TTNN-only
  absolute-position causal mask. Paged K/V cache population and the permuted
  page table remain exercised. No host fallback, runtime program config,
  compute-kernel override, or core-grid tuning was introduced.
- Verification:
  `timeout 420 pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_paged_prefill_crosses_nonchunked_sdpa_limit`
  passed on Blackhole at sequence 32769. Durable log:
  `long_prefill_standard_sdpa_mask_pass.log`.

## Final Long-Prefill Status

- Fixed with a proven model-local workaround.
- The 32769-token non-aligned boundary executes fully on device and returns the
  expected zero-weight result.

## Nonzero Long-Prefill Numerical Follow-Up

- Starting evidence:
  `review_findings_tests.log` recorded last-token PCC `0.992089152011556`
  against the synthetic nonzero reference at sequence 32769, below the
  functional threshold `0.995`. The zero-weight execution test passed.
- Hypothesis:
  the framework-default HiFi2/approximate compute policy loses enough accuracy
  in the masked tail SDPA reduction over 32769 keys to cross the functional
  PCC gate.
- Experiment:
  change only the masked long-SDPA launch to Blackhole HiFi4 with approximate
  math disabled and FP32 destination accumulation enabled. All projections,
  the initial 32768-token causal SDPA, cache operations, and the rest of the
  decoder retained framework defaults.
- Result:
  PCC improved from `0.992089152011556` to `0.9953599974129062`, clearing the
  `0.995` acceptance gate.
- Verdict:
  verified. The precision exception is isolated to the long-context masked
  SDPA whose default policy demonstrably failed PCC; no program config, grid,
  sharding, dtype, or host fallback changed.
- Refuted controls:
  explicitly padding K/V and the additive mask from 32769 to 32800 while
  retaining the default compute policy produced the exact original PCC
  `0.992089152011556`, refuting leakage from the 31 physical padding rows
  (`long_prefill_nonzero_explicit_padding.log`). HiFi4 alone while retaining
  approximate math and BF16 destination accumulation produced PCC
  `0.991978339521613`, so fidelity alone is not the fix
  (`long_prefill_nonzero_hifi4_only.log`). Only the recorded combined
  high-accuracy policy cleared the gate.
- Verification:
  `timeout 420 pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_paged_prefill_nonzero_chunk_boundary_last_token_pcc`
  passed on Blackhole. Durable passing log:
  `long_prefill_nonzero_hifi4.log`.

## Updated Long-Prefill Status

- Fixed: both the zero-weight 32769 execution boundary and the nonzero
  last-token PCC gate pass.
- Remaining risk: the measured margin is `0.00036` above the threshold, so the
  exact deterministic seed remains a required regression test.
