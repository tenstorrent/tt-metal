# AutoFix Report

## Starting Evidence

- Source diagnosis: `doc/full_model/AUTODEBUG.md`, written before implementation edits.
- Original failing evidence: `doc/full_model/logs/reduced_split_trace.log`.
- Original command constructed `build_generator(num_layers=4, max_context=128, batch=1)` on TP4 and called `generate([1,2,3,4,5], 3)`.
- Original terminal failure: `ttnn.sampling` values/indices logical-shape mismatch.

## Hypothesis Experiments

- Hypothesis: the common sampler's canonical 32-row index/parameter tensors were paired with batch-1 decode logits.
  Experiment: pad only the decode terminal rows to 32 before final norm/LM head, retaining the real TP4 sharded LM head and normal local-top-k/all-gather sampler.
  Result: the original values/indices mismatch disappeared. The next validation was reached: preallocated sampler output rank 2 was invalid.
  Verdict: verified.
  Evidence artifact: `logs/autofix_split_trace_model_pad.log`.
  Fix: `Qwen36Model.terminal_forward(..., pad_decode_rows=True)` pads decode rows to the common sampler's fixed-slot shape; prefill remains logically unpadded.
  Verification: the focused run passed local top-k, gathered values/indices, and their equality validation.

- Hypothesis: direct token feedback requires the sampler's canonical rank-4 `[1,1,1,32]` output tensor, with the model consuming active slots from that same persistent tensor.
  Experiment: allocate the trace token as `[1,1,1,max_sampling_batch]`, pass it as `tt_out_tok`, and slice/reshape the active model batch on device at decode entry.
  Result: the complete reduced TP4 `generate([1,2,3,4,5], 3)` returned `GENERATE_OK [220, 220, 220]`; two model and sampler trace replays completed with zero host token, position, or page-table refreshes.
  Verdict: verified.
  Evidence artifact: `logs/autofix_split_trace_rank4.log`.
  Fix: rank-4 persistent feedback allocation in `tt/generator.py`; device-side active-slot selection in `tt/model.py`.
  Verification: trace counters were `replays=2`, `token_host_refreshes=0`, `position_host_refreshes=0`, `page_table_refreshes=0`, `readbacks=3`.

- Hypothesis: the LM-head/logits topology itself required a custom sampler.
  Experiment: retain the standard `models/common/sampling` implementation and the existing TP4 column-parallel LM head throughout both experiments.
  Result: the reduced generator completed after only fixed-row and feedback-buffer contract repairs.
  Verdict: refuted.
  Fix: none; no custom sampler was introduced.

- Hypothesis: `SamplingGenerator.capture_trace` redundantly precompiled the sampler after the model trace was live, allocating temporary buffers that could invalidate model-trace addresses.
  Experiment: retain the exact sampler warmup before model capture, then call `capture_trace(..., skip_precompile=True)` so sampling temporaries created for capture are owned by the trace allocator rather than allocated in ordinary mode after a live trace exists.
  Result: the allocator warning disappeared, sampler capture/replay completed with 694/694 program-cache hits, and the reduced token-feedback run returned `GENERATE_OK [220, 220, 11]` with two trace replays and no host token/position/page-table refreshes. The warning-bearing control returned `[220, 220, 220]`; the changed third token after stable allocation is also consistent with non-stale feedback/position state.
  Verdict: verified.
  Evidence artifacts: warning-bearing control `logs/autofix_split_trace_rank4.log`; fixed run `logs/autofix_split_trace_stable_alloc.log`.
  Fix: pass `skip_precompile=True` when capturing the separate sampling trace, after the already-existing exact sampler warmup and model-trace capture.
  Verification: no `unsafe due to the existence of an active trace`, program-cache miss, trace-capture write, or runtime failure appears in the fixed log.

- Hypothesis: batch-2 full attention fails because `nlp_concat_heads_decode` returns the canonical logical 32-user tile while the split QKV gate retains logical shape `[1,1,2,1536]`; B=1 broadcasts but B=2 cannot subtile-broadcast to 32.
  Experiment: pad only the gate's decode-user dimension to 32 before sigmoid/multiply, leaving TP4 projection, attention, caches, collectives, dtype/fidelity, and residual contracts unchanged. Run the exact reduced B2 mixed-slot capture with tokens `[220,11]`, positions `[5,3]`, and active mask `[1,0]`, then replay model and force-argmax sampler traces once.
  Result: the full-attention multiply, complete four-layer capture, and replay pass. Persistent positions are `[7,3]`: the active slot advances through capture plus replay, while the inactive slot remains exactly 3. Tokens after replay are `[220,220]`.
  Verdict: verified.
  Evidence artifacts: `logs/autofix_mixed_b2_gate_pad.log` (first capture passes; probe ends only on a misspelled diagnostic helper), `logs/autofix_mixed_b2_replay.log` (correct direct persistent-tensor read), and final warning-free `logs/autofix_mixed_b2_warm_restore.log`.
  Fix: pad `gate` to the decode tile's 32 logical rows at the full-attention concat/gate boundary in `tt/multichip_decoder.py`.
  Verification: B2 mixed capture/replay passes; B1 reduced generation regression passes in `logs/autofix_mixed_b1_regression.log` with two trace replays and zero host token/position/page-table refreshes.

- Hypothesis: the remaining allocator warning after B2 capture was first-use compilation/allocation of cache restore copies after traces became live.
  Experiment: warm every exact backup-to-cache `ttnn.copy` immediately after backup allocation, before model warmup and either trace capture; retain the same post-capture restores.
  Result: the warning-bearing B2 control logged the warning immediately before return from capture. The warmed-copy run completes capture and replay without the allocator warning, program-cache miss, trace write, or runtime failure.
  Verdict: verified.
  Evidence artifacts: control `logs/autofix_mixed_b2_replay.log`; fixed `logs/autofix_mixed_b2_warm_restore.log`.
  Fix: exact restore-copy warmup plus synchronization before trace capture in `tt/generator.py`.
  Verification: final B2 result `B2_MIXED_WARM_COPY_OK [220,220] [7,3]`; final B1 result `B1_REGRESSION_OK [220,220,220]`. Readback occurs only after replay synchronization for evidence and is not inside either live trace.

## Final Status

- Fixed for the reported reduced split-sampling failure, batch-2 mixed-slot full-attention gate mismatch, and trace-lifetime/address-stability warnings.
- Proof command: the bounded TP4 Python reproducer recorded in `logs/autofix_split_trace_stable_alloc.log`.
- The fixed proof has 100% program-cache hits and no allocator-safety, program-cache-miss, trace-write, or runtime warning in the split sampling path.
- Remaining risk: B32 runtime coverage and full-layer/full-context evidence remain separate stage gates; this AutoFix intentionally used the reduced four-layer B2 and B1 reproducers.

## Stage-review remediation pass

- Hypothesis: the context finding is a real public-contract contradiction, not merely schema wording.
  Experiment: compare `Qwen36Generator.MAX_PREFILL_TOKENS`, public validation, the 192511 pass/194559 physical failure evidence, and every full-model context field.
  Result: verified. Public prompt capacity cannot truthfully be 262144; only decode-cache allocation/absolute positions reach it.
  Fix: advertise 192511 as public supported context and keep 262144 in separately named decode-cache fields; remove stale notes claiming otherwise. Add a static equality regression.
  Verification: `test_context_contract_matches_public_prefill_limit` passes and both JSON documents parse.

- Hypothesis: low-level sampled serving required private trace state and a host-logits boundary.
  Experiment: inspect all public generator methods and trace/sampler entry points.
  Result: verified. Only private capture/replay exposed common sampling; public `decode_forward` returned full host logits.
  Fix: add public `setup_token_out_decode` and `token_out_decode_step` with explicit token/position/page-table/cache/active-mask/`SamplingParams` state. Replay returns the persistent device token by default; readback is opt-in and never reads logits. High-level greedy generation now delegates its replay to this public step.
  Verification: signature regression passes; reduced hardware verification is explicitly queued for greedy and non-greedy modes.

- Hypothesis: documentation overstated layer-0 mixed-prompt evidence as a full-wrapper layer-stack test.
  Experiment: inspect `tests/mixed_prompt_state.py` and cited log against wrapper boundaries.
  Result: verified.
  Fix: correct README and add `tests/full_model_mixed_slots.py`, which instantiates the reduced wrapper with both real layer kinds, mixed S65/S63 prefill, inactive row, terminal sampler, traced feedback, and output formatting.
  Verification: script compiles; hardware command is recorded in `work_log.md`.

- Hypothesis: qualitative metadata was stale/vague rather than evidence of a completed shared suite.
  Experiment: enumerate qualitative artifacts and inspect `tests/full_model_qualitative.py` plus the shared prompt source.
  Result: the prompt-correct runner exists, but no shared-suite output exists. The review finding is verified and cannot be closed statically.
  Fix: metadata now names source, runner, expected artifact, and exact completion gate; work log records the serialized TP4 command. No output or verdict was fabricated.

- Hypothesis: the reported performance workload's malformed continuation came from plain encoding/truncation rather than the model trace.
  Experiment: inspect `tests/full_model_perf.py` prompt construction versus the chat-template correctness contract.
  Result: verified; it plain-encoded text.
  Fix: render the performance prompt through `apply_chat_template`, add `--num-layers` for the required reduced profiler, and document exact Tracy/`tt-perf-report` commands and cost accounting.
  Verification: profiler harness compiles; hardware profiling remains required.

Static verification command:

```text
python -m py_compile tt/generator.py tests/full_model_perf.py tests/full_model_mixed_slots.py
pytest -q tests/test_full_model_public_contract.py
# 2 passed
```

Remaining stage blockers requiring serialized hardware: shared HF/TT qualitative suite plus human review; public non-greedy top-k/top-p trace replay; reduced B2 full-wrapper run; reduced profiler, sampler comparison and layer-stack lower bound; Watcher/runtime-integrity; and exact partial-slot cache reset/reuse. Whole-model reset remains valid, but partial reset is not claimed because safely clearing paged blocks plus recurrent/conv rows needs an on-device implementation and probe.

## B2 full-wrapper LM-head repair

- Hypothesis: `_project_lm_head_tile` used sequence height 32 for a width-sharded tensor whose physical matrix height is `batch * sequence_tile = 64` at B2.
  Experiment: run the public reduced B2 S65/S63 wrapper probe through the real terminal. The original log fails at interleaved-to-sharded with `Shard height 32 must match physical height 64`.
  Result: verified. Simply setting shard/program M to 64 was refuted by the retained DRAM-sharded program's exact one-tile-M contract (`currently only support in0 tensor height of tile height`).
  Verdict: verified with the initial proposed flattened-M repair refuted.
  Fix: preserve the selected 32-row DRAM-sharded LM-head program and weights. For B>1, slice each fixed slot on device as `[1,1,32,H]`, project it with the unchanged optimized path, and concatenate outputs on the device batch axis. There is no host fallback or policy change.
  Verification: `logs/full_model_mixed_slots_lm_head_fix_v4.log` passes `FULL_MODEL_MIXED_SLOTS_OK [12,220] [66,63]` through mixed S65/S63 prefill, both real layer kinds, terminal, public non-greedy common sampler (`temperature=.8, top_k=5, top_p=.9`), traced feedback, and inactive-slot position preservation. The log's sampler reset records `force_argmax=False`.

- Regression: `logs/lm_head_b1_regression.log` passes canonical force-argmax B1 generation with `[220,220,220]`, two replays, and zero host token/position/page-table refreshes.

Final status for this bug: fixed with B2 non-greedy public-path and B1 greedy evidence. B32 execution remains a broader stage capability gate, not part of this focused failure.

## Reduced profiler AutoFix

- Hypothesis: profiler overflow was caused by capturing construction, prefill, compile, trace capture, and 16 replays instead of the requested steady-state reduced replay.
  Experiment: add `--profile-only-decode`; call `ReadDeviceProfiler` after capture, signpost one replay, synchronize/read immediately, and use Tracy `--dump-device-data-mid-run`.
  Result: partially verified. The measured replay is 7.61 ms in v2, but markers accumulated before the first mid-run dump had already overflowed on some cores. Full Tracy correlation remained invalid.
  Verdict: verified cause, first mitigation insufficient.

- Hypothesis: a true one-layer-per-kind model would reduce construction/compile marker volume enough for a valid raw capture.
  Experiment: add explicit `layer_indices=[0,3]` support (debug/profiling only), run one 32-token prefill, capture, flush, then one canonical force-argmax model+sampler replay between `FULL_MODEL_DECODE` signposts.
  Result: raw v3 capture succeeds with no `Profiler DRAM buffers were full` warning, 100% program-cache hits, and one token-out replay measured at 5.659 ms (176.70 t/s/u). Artifacts: `artifacts/profile_reduced_v3/perf.json`, `.logs/cpp_device_perf_report.csv`, `.logs/profile_log_device.csv`, and `logs/profile_reduced_v3.log`.
  Verdict: verified for overflow removal and valid reduced runtime capture.

- Hypothesis: standard Tracy postprocessing plus `tt-perf-report` can turn the valid raw capture into named operation rows.
  Experiment: allow wrapper postprocess, retry process-logs-only, then generate a device-only report and invoke `tt-perf-report`.
  Result: refuted in the current tool path. The wrapper exceeded the bounded command while exporting the 28.7 MB host trace. `--process-logs-only -o` incorrectly resolves `.logs/.logs/tracy_profile_log_host.tracy`. Manual `tracy-csvexport` succeeds (454 MB times CSV), but normal `process_ops_logs.py` then fails `KeyError: 3` while matching mid-run trace IDs across devices (`logs/profile_reduced_v3_correlate.log`). Device-only processing succeeds and writes `reports/ops_perf_results.csv`, but that schema omits `DEVICE ID` and all host-correlated OP names; `tt-perf-report` rejects it with `missing the 'DEVICE ID' column`. The raw C++ CSV has device IDs but blank OP names, so it cannot distinguish decoder, LM head, all-gather, and argmax.
  Verdict: exact tooling blocker; AutoFix failed to produce the required named `tt-perf-report` without fabricating correlation.

The prior profiler blocker was subsequently fixed in `tools/tracy/process_ops_logs.py`:
mid-run partial host captures may contain a trace END without its earlier BEGIN;
the parser now ignores that uncorrelatable END while retaining strict mismatch
checks when a BEGIN is present. A new `-r` reduced capture produced a valid
four-device named report and `tt-perf-report` exit 0.

That report proved the original force-argmax sampler dominated (1.319 ms
ArgMax + 0.831 ms all-gather + 0.093 ms untilize). The matched common
candidate-gather alternative was worse (9.697 ms TopK). AutoFix therefore
changed the force-argmax contract to slice sampler logits to active fixed slots,
perform semantic global greedy, and pad/copy only the tiny sampled-token result
to the canonical 32-slot feedback buffer. Exact reduced tokens remain
`[220,220,220]`. Final named profile: 4.264 ms wall, 3.402 ms named device ops,
sampler all-gather 0.830 ms (24.4%), LM head 1.198 ms (35.2%); sampler no longer
dominates. Full 64-layer result is 17.5168 t/s/u.

Final profiler status: fixed and proven. Artifacts are
`profile_reduced_v4`, `profile_candidate_greedy`, and
`profile_active_row_greedy`.
