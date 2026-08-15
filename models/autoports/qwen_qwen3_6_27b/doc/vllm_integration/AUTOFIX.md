# AutoFix Report

## Starting Evidence

- Source reports: `AUTODEBUG.md` and `SEEDING_REDIAGNOSIS.md` in this directory.
- Original failure artifact: `readiness_vllm/sampling_tests.log` (`68 passed`, `4 failed`, `1 skipped`). The failures were `mixed_params_batch`, `same_seeds_reproduce_across_batches`, and `specific_seed_reproducible` for seeds 42 and 0.
- The failure artifact predates the current adapter and seed-state repairs. This investigation used source inspection and host-only tests; it did not open TT devices or launch a server.

## Hypothesis Experiments

- Hypothesis: the failed run ignored request seeds and used the sampler's construction-time slot-index seeds.
  Experiment: inspect the failed artifact and the request-seed path from vLLM `slot_sampling_params()` through `generator_vllm.py`, `SamplingGenerator`, and `SeedManager`.
  Result: the old diagnosis explains the failure pattern, but is stale against the current tree. Current prefill binds compact request-order seeds to `empty_slots`; decode consumes slot-ordered seeds, preserves state across `slot_remap`, aligns explicit-seed counters to absolute positions, and derives each device seed from `(request seed, token counter)` rather than physical slot.
  Verdict: verified for the logged pre-repair run; repaired in current source.
  Evidence artifacts: `SEEDING_REDIAGNOSIS.md`, `tt/generator_vllm.py`, `models/common/sampling/generator.py`, and vLLM plugin `input_batch.py::slot_sampling_params()`.
  Fix: no additional production edit was justified from stale runtime evidence.
  Verification: the host-only command below passes.

- Hypothesis: the repaired seed stream can still vary when an identical request occupies a different batch slot or is remapped during batch condensation.
  Experiment: add focused host-only tests that compare four derived device seeds for the same request seed in slots 0 and 2, and compare the next seed before/after moving a live request from slot 1 to slot 2.
  Result: both streams are identical.
  Verdict: refuted.
  Evidence artifact: `models/common/tests/test_sampling.py`.
  Fix: added durable regressions `test_seed_manager_request_stream_is_independent_of_batch_slot` and `test_seed_manager_slot_remap_preserves_request_stream`.
  Verification: `pytest -q models/common/tests/test_sampling.py -k 'seed_manager or seed_counter or broadcast_sampling_params'` -> `5 passed, 9 deselected`.

## Final Status

- Source-level request-seed/slot repair: verified by inspection and host-only focused tests.
- A first repaired-source serving rerun improved the suite to `69 passed`, `3 failed`, `1 skipped`, but still failed `mixed_params_batch`, `same_seeds_reproduce_across_batches`, and `specific_seed_reproducible[999]`. This fresh evidence refuted slot independence alone as sufficient.
- Remaining runtime gate: rerun the three failed sampling node IDs after the stale-position fix below, then rerun the full sampling profile if they pass. No hardware experiment was run by this AutoFix agent.

## Fresh-Evidence Repair

- Hypothesis: the adapter rewinds an explicit request's RNG stream from stale host positions during async decode.
  Experiment: trace the decode seed lifecycle through `generator_vllm.py`, vLLM request-state slot construction/remapping, and token-out trace setup. Compare it with the async serving contract, which permits token/current-position inputs to be stale while on-device token feedback owns the next step.
  Result: every decode submission called `align_seed_counters_to_positions(seed_values, active_seed_slots, start_values)`. Repeating a stale `start_pos` therefore resets an already-advanced request counter to an old value. Whether and how often that happens depends on overlap and batch scheduling, predicting the observed request-reordering sensitivity. The request seed itself is already initialized at prefill, `reset_seed_from_slots_if_needed` preserves an unchanged stream, and `apply_slot_remap` moves its counter with request state; position realignment is neither necessary nor safe.
  Verdict: verified source-level async contract violation.
  Evidence artifacts: `tt/generator_vllm.py`, `models/common/sampling/generator.py`, and the refreshed `readiness_vllm/sampling_tests.log`.
  Fix: removed per-decode counter alignment from stale `start_pos`; retain seed reset only when the slot's requested seed actually changes.
  Verification: added `test_unchanged_decode_seed_does_not_rewind_stream_on_stale_input`. `pytest -q models/common/tests/test_sampling.py -k 'seed_manager or seed_counter or unchanged_decode_seed or broadcast_sampling_params'` -> `6 passed, 9 deselected`; `python -m py_compile models/autoports/qwen_qwen3_6_27b/tt/generator_vllm.py` passes.
  Remaining uncertainty: a targeted live-server rerun must prove the TT trace/scheduler path now preserves seeded outputs across shuffled batches.

## Fresh Re-diagnosis After Live A/B

- Experiment: rerun the full live sampling suite with per-decode position alignment removed.
  Result: regressed from the best repaired-source result (`69 passed`, `3 failed`, `1 skipped`) to `66 passed`, `6 failed`, `1 skipped`; additional seed cases and structured-output JSON truncation failed.
  Verdict: the proposed removal was refuted and has been reverted. Position recovery is required for at least request re-admission/trace lifecycle correctness. The stale-input regression test remains useful evidence that unchanged seed state itself must not be reset; it is not evidence for removing position recovery.
  Fix disposition: restored `align_seed_counters_to_positions(...)` in `tt/generator_vllm.py`, returning production code to the best 69/3 state.

- Hypothesis: position recovery must be monotonic. Re-admitted/restored requests need their counter advanced to the absolute decode position, while an async submission carrying an older host position must never lower an already-advanced counter.
  Experiment: initialize an explicit-seed slot with counter 12, then apply current position alignment using stale position 8.
  Result: current code lowers the counter to 9. This proves the two observed needs are not mutually exclusive: retaining alignment repairs lifecycle gaps, but assignment-based alignment can rewind live state.
  Verdict: verified source-level hazard; live effect still uncertain.
  Evidence artifact: the initial characterization experiment was converted to the durable desired-contract test `test_seed_counter_position_alignment_does_not_rewind_on_stale_position` in `models/common/tests/test_sampling.py`.
  Fix: after the active server shut down, changed alignment to `max(existing_counter, position + offset)` rather than unconditional assignment. This preserves the alignment that improved structured/seed behavior while preventing stale-position rewinds.
  Verification command: `pytest -q models/common/tests/test_sampling.py -k 'seed_manager or seed_counter or unchanged_decode_seed or broadcast_sampling_params'` -> `7 passed, 9 deselected`. `python -m py_compile models/common/sampling/generator.py models/autoports/qwen_qwen3_6_27b/tt/generator_vllm.py` also passes.
  Runtime status: host/source verified; targeted live sampling rerun remains required before declaring the three serving failures closed.

## Monotonic A/B Refutation And Instrumented Next Step

- Experiment: rerun the full live suite with monotonic counter alignment (`max(existing, position + offset)`).
  Result: `67 passed`, `5 failed`, `1 skipped`, worse than the best `69/3` state and with additional seeded failures.
  Verdict: refuted as a serving fix. The shared counter assignment and its adapter call were restored to the best-known state. The host characterization test intentionally records that stale positions can rewind current counters; it no longer claims monotonic behavior is the fix.

- Fresh diagnosis status: source-only reasoning cannot distinguish device seed tensor ordering/races from scheduler/trace lifecycle using pass/fail text alone. Both speculative counter-policy changes made runtime results worse. The next experiment must record the actual seed lifecycle from prefill through first decode and later replay.
  Instrumentation: `TT_SAMPLING_DEBUG=1` now enables the existing `SamplingDBG` logs for any model, not only Llama-3.3-70B. Counter alignment additionally records active slots, positions, requested seeds, and counters before/after. Existing logs already cover prefill seed reset, slot remap, generated device-seed vectors, parameter reset, and direct-versus-traced sampling selection.
  Focused experiment: launch only the failing seeded node IDs with `TT_SAMPLING_DEBUG=1`, preserve `server.log`, and compare one request seed across its two runs. Check (1) compact prefill seed -> physical slot binding, (2) first device seed, (3) remap source/destination, (4) decode position/counter, (5) generated device seed per emitted token, and (6) whether trace warm/capture occurs between the intended seed write and real replay. Do not apply another behavioral fix until those logs show the first divergent lifecycle event.
  Host verification: `pytest -q models/common/tests/test_sampling.py -k 'seed_manager or seed_counter or unchanged_decode_seed or broadcast_sampling_params'` -> `7 passed, 9 deselected`; `python -m py_compile models/common/sampling/generator.py models/common/sampling/tt_sampling.py` passes.

## Primary Benchmark OOM

- Original failure: primary `vllm bench serve` workload `128 input / 128 output / 1 request / concurrency 1` reported `completed=0`, so the readiness runner correctly stopped before CI.
  Experiment: correlate `vllm_benchmark.log`, `vllm_result.json`, and the server traceback at the benchmark timestamp.
  Result: this was not a timeout, prompt-format error, max-token rejection, or result parser bug. EngineCore died during 128-token linear-attention prefill at `ttnn.concat` with an allocator OOM: requested 805,306,368 bytes total (100,663,296 bytes/bank), with 141,336,512 bytes/bank free and only a 62,914,560-byte largest block. The benchmark client consequently received no valid streaming chunk and correctly recorded zero completions.
  Verdict: verified serving KV-pool/workspace sizing bug.

- Capacity hypothesis: reducing the KV pool could make room for both simultaneous 805,306,368-byte concat outputs.
  Experiment: reduce `MAX_TOKENS_ALL_USERS` from 1,726,400 to 1,658,400 and rerun live traffic.
  Result: refuted as a completion fix. The first concat allocated, but the second still failed with 122,514,496 bytes/bank free and a 61,341,696-byte largest block. Pool reduction changed the failure boundary without eliminating the two-workspace peak.
  Fix disposition: restored `MAX_TOKENS_ALL_USERS=1,726,400` and its 2,190-page aggregate allocation, preserving the largest feasible concurrency. The lifetime repair below addresses the actual simultaneous peak without reducing serving capacity. The public per-request `max_model_len=262144` remains unchanged.

## Targeted Sampling Lifecycle Result

- Experiment: run only the three remaining sampling failures with `TT_SAMPLING_DEBUG=1` and inspect the first lifecycle event.
  Result: the failing requests produced no seeded `SamplingDBG` lifecycle at all. Their `top_k=50` or penalties exceed the dedicated device sampler contract, so the plugin intentionally selected explicit host sampling. The only device-sampling logs belonged to the runner's unseeded health request. This proves the earlier device seed/counter hypotheses were investigating a path these failures did not execute.
  Verdict: device seed lifecycle refuted as the cause of these three failures.

- Hypothesis: host compatibility decode loses model-owned request state when scheduler rows change.
  Experiment: inspect `async_decode.submit_decode` and the Qwen adapter's `sampling_params is None` branch. The plugin forwarded `slot_remap` only inside `perform_device_sampling`; the host branch did not call `remap_decode_slots`. Thus linear-attention recurrent/conv state stayed attached to the old physical row after shuffled batches, condensation, or row reuse.
  Verdict: verified source contract bug matching shuffled-batch and repeated-run failures.
  Fix: forward `slot_remap` for both host and device sampling in the TT plugin; in Qwen's explicit host compatibility branch apply `gen.remap_decode_slots(remap)` before eager decode. On-device traced sampling remains unchanged.
  Verification: `test_host_sampling_compatibility_preserves_slot_remap`; the adapter contract suite passes `7 passed` with compilation and diff checks clean. Live targeted rerun remains required.

## Second-Concat OOM Re-diagnosis

- Experiment: inspect the post-pool-reduction allocator failure at `00:16:59`.
  Result: the reduced pool allowed `previous_transform` (the first 805,306,368-byte concat) to allocate. The failure moved exactly to `previous_bias`, the second equal-sized concat: 122,514,496 bytes/bank free, 61,341,696-byte largest block. This refutes the assumption that capacity for one concat was sufficient, while confirming the pool reduction created enough room for the first.
  Verdict: verified overlapping temporary-lifetime bug.
  Fix: compute the transform matmul and deallocate `previous_transform` before allocating `previous_bias`; then compute the bias matmul. The two shifted tensors are independent inputs to separate matmuls, while `old_transform` and `old_bias` remain live until both results are complete, so the algebra is unchanged. Restore the original KV pool because only one concat workspace is now live at a time and the reduced-pool A/B did not complete the workload.
  Verification: `test_linear_scan_does_not_hold_both_concat_workspaces` asserts the lifetime ordering. Adapter contract suite passes `7 passed`; Python compilation and diff checks pass. Hardware benchmark rerun remains required.

## Fragmented First-Concat OOM

- Fresh evidence: the full sampling gate passed `72 passed`, `1 skipped`, but the following benchmark failed on the first `previous_transform` concat. Allocator state was 3,935,323,968 bytes/bank allocated, 137,017,408 free, and a largest free block of only 62,914,560 bytes versus the required contiguous 100,663,296 bytes/bank.
  Verdict: the lifetime reorder removed the simultaneous peak, but repeated serving fragmented DRAM enough that a late dynamic 805,306,368-byte concat allocation is still unsafe. Aggregate free bytes are sufficient; contiguous placement is not.

- Hypothesis: reserve one shared scan concat output before vLLM cache allocation and reuse it for both shifted transform and shifted bias.
  Experiment/source contract: `ttnn.concat` accepts `output_tensor`. Qwen's local TP4 scan shapes are identical for transform and bias (`batch * 12`, 64, 128, 128), and layers execute serially. One model-owned BF16 tensor therefore serves all 48 linear layers and both concat roles; no per-layer multiplication is needed.
  Result: refuted by the live reduced-pool run. Startup and cache allocation succeeded, but the first scratch-backed concat terminated at `concat.cpp:272` with `TT_FATAL optional output tensor currently unsupported` (`!optional_output_tensor.has_value()`). The Python signature exposes `output_tensor`, but this backend implementation rejects it; persistent scratch reuse is therefore invalid.
  Fix disposition: remove the model-owned scratch allocation and all concat `output_tensor` calls. Retain the sequential lifetime repair, which deallocates `previous_transform` before allocating `previous_bias`. Retain the evidence-backed pool reduction: the earlier full-pool benchmark's later `transform[:, :-distance]` slice requested 792,723,456 bytes total (99,090,432 bytes/bank) with 149,901,504 bytes/bank free but only a 62,914,560-byte largest block. The initial reduction of exactly 116 800-token pages (92,800 tokens) allowed that dynamic slice in a live rerun. Its following first concat then requested 100,663,296 bytes/bank with a 99,090,432-byte largest block, a shortfall of 1,572,864 bytes/bank. A two-page trial raised free memory from 252,436,672 to 254,177,472 bytes/bank exactly as predicted (`2 * 870,400`), but the largest block remained 99,090,432 bytes because `transform[:, :-1]` is still live when concat requests its output. This refutes treating the 1,572,864-byte scalar shortfall as the capacity requirement. Before those slices, inferred free memory is approximately `254,177,472 + 100,663,296 = 354,840,768` bytes/bank; the next experiment must preserve a second, concat-sized contiguous region while the slice is live. At 870,400 bytes per 800-token page, `ceil(100,663,296 / 870,400) = 116` further pages. Reduce by 234 pages total (116 initial + 2 diagnostic + 116 simultaneous-live reserve), or 187,200 tokens: `MAX_TOKENS_ALL_USERS=1,539,200`; with 32 lookahead pages the aggregate allocation is 1,956 pages (1,564,800 tokens, 5.96 times the per-request context). This is the largest evidence-backed next experiment, not yet a proven final capacity, and preserves the per-request context contract of 262,144.
  Verification: `test_linear_scan_does_not_hold_both_concat_workspaces` asserts the transform workspace is deallocated before the bias concat and forbids the rejected scratch/output-tensor path. Adapter contract suite, `py_compile`, and `git diff --check` are the host gates.
  Follow-up result: the 234-page live experiment refuted cache reduction as the control knob. The concat OOM was unchanged (100,663,296 bytes/bank required; 99,090,432-byte largest block), while free memory increased only from 254,177,472 to 254,476,480 bytes/bank despite removing another 92,800 advertised tokens. Device cache allocation is therefore essentially independent of this advertised pool in the measured path. Restore the largest original `MAX_TOKENS_ALL_USERS=1,726,400` and 2,190 aggregate pages.

- Hypothesis: reduce the Qwen3.6 linear-prefill recurrent-scan chunk from 64 to 32 tokens, attacking the allocation shape rather than advertised cache capacity.
  Semantics: `_linear_attention_prefill` processes chunks in sequence order, and each `_linear_attention_prefill_chunk` reads the recurrent cache as its initial state and writes its final state back before the next chunk. The affine recurrence is associative within each chunk; changing the boundary only adds an equivalent recurrent handoff and does not reset or reorder state. `_BalancedSequenceConcat` preserves output token order. Existing long-state tests compare chunked recurrence against token-by-token recurrence across chunk boundaries.
  Allocation bound: scan tensors scale linearly with chunk length. The 64-token concat was 805,306,368 bytes total (100,663,296 bytes/bank); at 32 tokens it is approximately 402,653,184 bytes total (50,331,648 bytes/bank), below the measured 62,914,560-byte largest block even in the fragmented full-pool run. Transform slices are likewise approximately halved.
  Fix: set Qwen3.6 `LINEAR_PREFILL_CHUNK_SIZE=32`, retain sequential concat lifetime deallocation, and restore the original serving pool.
  Verification: adapter contracts assert the 32-token chunk and workspace bound; long-state host tests validate recurrence and boundary semantics. A live benchmark rerun remains required to prove the smaller device allocations complete under serving fragmentation.

- Live 32-token result: the scan no longer OOMed, validating the smaller allocation shape, but prefill failed earlier at `multichip_decoder.py:804`. `_conv_state_selectors` still came from a generator loop hardcoded to 64-token chunks, so reshaping a 68-wide selector as `(1, batch, 1, sequence + 4)` for a 32-token chunk (width 36) failed volume validation.
  Root cause: `LINEAR_PREFILL_CHUNK_SIZE` controlled the decoder recurrence loop, while generator mask/selector construction and streaming-model metadata indexing independently used literal 64. Changing only the recurrence chunk split the metadata contract.
  Fix: import and use `LINEAR_PREFILL_CHUNK_SIZE` in `generator.py` for mask/selector chunk construction and in `model.py` for streaming metadata slice indices. Each selector is now exactly `(batch, chunk_len + 4)`, including a final non-aligned chunk, matching the decoder reshape `(1, batch, 1, sequence + 4)`. Sequence masks use the same `chunk_len`.
  Verification: the adapter source contract ties construction, indexing, and consumption to the shared constant and shape expressions. The long-state recurrence test continues to cover non-aligned chunk boundaries and token-by-token state equivalence. Live serving rerun remains required.

## Fixed-Capacity Serving Decode Overhead

- Hypothesis: the production max-num-seqs-32 trace performs avoidable 32-slot model decode work for a single active request.
  Experiment: compare the existing production artifact against a fresh server with only `--max-num-seqs 1` changed. Both arms use max-model-len 262144, trace mode all, on-device split sampling, the selected precision, `FABRIC_1D_RING`, a 200 MB trace region, and the same greedy 128-input/128-output/one-request/concurrency-one benchmark. The sampler remains padded to its canonical 32 rows, so its shape is controlled across arms. Artifacts: `readiness_vllm/capacity_ab/maxseq32/` and `maxseq1/`.
  Result: max-num-seqs 32 measured TTFT 111,416.0 ms, mean TPOT 251.6556 ms, ITL P50/P99 243.9158/245.8212 ms, and 3.9737 t/s/u. Max-num-seqs 1 measured TTFT 4,138.6 ms, mean TPOT 70.7335 ms, ITL P50/P99 55.8614/57.5023 ms, and 14.1376 t/s/u. The 3.56x decode gain verifies fixed inactive-slot model execution as the dominant overhead. Cache allocation changed only from 2,190 to 2,159 blocks; max context did not change.
  Verdict: verified.
  Fix: make max-num-seqs 1 the headline primary serving configuration. Keep max-num-seqs 32 as the explicit secondary CI/capacity profile. This avoids speculative multi-trace state remapping, preserves canonical on-device split sampling and full 262,144 context, and raises primary vLLM serving above the 6.96 t/s/u teacher-forcing lower bound.
  Verification: the max-num-seqs-1 runner completed successfully and shut down cleanly. Process audit found no `vllm.entrypoints`, `EngineCore`, or `run_vllm_server` owner; `tt-smi -ls --local` showed all four p300c devices healthy. Root primary JSON/log artifacts now contain the max-num-seqs-1 result; prior max-num-seqs-32 CI JSON/log artifacts remain unchanged.
