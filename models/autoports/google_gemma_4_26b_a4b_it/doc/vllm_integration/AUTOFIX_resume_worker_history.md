# AutoFix: reconcile worker history after async preemption

## Source and contract

The scheduler can discard outstanding async output tokens during preemption. The worker's completed-decode handler independently appends each finalized token to `CachedRequestState.output_token_ids`. Before this fix, `apply_cached_req_state_update` replaced resumed blocks/computed position but retained every worker output. Consequently, a request whose worker held four outputs while the scheduler retained three could replay the fourth, report an incorrect output count, and shift the restored original-prompt boundary.

The installed vLLM 0.26 `Scheduler._make_cached_request_data` provides `num_output_tokens` and, when the request was not scheduled on the previous step, `all_token_ids`. Normal decode counts can include output placeholders and lag/ahead async work; they cannot be used to trim ordinary live decode history. On resumed prefill, the TT scheduler has reset those placeholders, making retained count and any full token-ID payload authoritative.

A second ordering issue made late trimming unsafe: `build_model_input` performed `_update_states` before its layout-change drain. Pending output application after reconciliation could reintroduce the discarded suffix. Existing outer callers often drain earlier, but this method itself did not guarantee the order. Finally, a resumed request could still occupy a persistent row; the old branch updated only computed length and appended new block IDs, leaving token lengths/prefix and a freed block mapping in place.

## Negative controls

Added `vllm/plugins/vllm-tt-plugin/tests/test_resumed_request_history.py` and ran before implementation:

```sh
# From vllm/plugins/vllm-tt-plugin:
python -m pytest tests/test_resumed_request_history.py -q --disable-warnings --tb=short
```

Result: **9 failed, 2 passed**. Eight failures cover front-packed/lane batches, resumed row present/absent, and authoritative full IDs present/absent. The ninth proves that a pending decode was not drained before `_update_states`. Both normal-decode controls passed, preserving four worker outputs when scheduler metadata retained only three.

Evidence: `readiness_vllm/resume_worker_history_before.log`.

## Repair

- The shared cached-state helper now receives the scheduler output count and optional full IDs. On resume only, it validates and replaces the retained suffix from full IDs when available, or truncates the worker suffix to the scheduler count. Missing retained tokens without a full payload are rejected. The output list is updated in place so persistent batches and logits processors keep the correct shared reference.
- Both normal and lane callers pass this metadata. Ordinary cached decode does not reconcile output history.
- A resumed request still present in the front-packed batch is reloaded at its existing row with `InputBatch.add_request`, replacing token data/logical lengths/block mapping from reconciled state. Lane mode already removes/re-adds resumed rows. Merely appending a new block list is no longer used for this front-packed resume case.
- Normal `build_model_input` explicitly drains pending output before `_update_states` when resumed IDs exist. The lane entry point also explicitly requires the pre-update drain on resume, regardless of an overlap prediction. Existing post-layout drains remain idempotent; normal stable decode is unchanged.

This repairs the source of counts used by the model's resumed-prefill sampler fix; it does not reduce context, preemption, capacity, or async capability.

## Validation

```sh
# From vllm/plugins/vllm-tt-plugin:
python -m pytest tests/test_resumed_request_history.py tests/test_hybrid_cache_contracts.py -q --disable-warnings --tb=short
# From repository root:
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q --disable-warnings --tb=short
```

Results: **37 passed in 5.99 s** (12 retained-history tests and 25 existing hybrid/cache tests), plus **78 passed in 33.93 s** sampling-contract tests. Durable logs are `readiness_vllm/resume_worker_history_after.log` and `resume_worker_sampling_regressions.log`.

The final tests also replace a wrong worker-retained token using authoritative IDs, preserve list identity, verify block-table replacement and the actual count producer, prove that a second drain cannot reappend discarded tokens, and exercise the actual lane entry point with an intentionally optimistic overlap prediction.

The new test file passes Black. Modified runtime lines were formatted without rewriting pre-existing unrelated assert-format differences in the plugin. Python-only changes require no C++ build. These are CPU contract tests; server/device validation remains owned by the parent agent and must load the final source.
