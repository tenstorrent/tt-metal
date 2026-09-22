# Bounded owned-prefill trace candidate

Historical design proposal, subsequently implemented and validated. Final
acceptance is [prefill_contract_full.json](prefill_contract_full.json),
[watcher_prefill_final.json](watcher_prefill_final.json), and
[after_full_final.json](after_full_final.json). Proposal-time commands and
unverified statements below describe the earlier authoring checkpoint. The
applied patch is retained as [prefill_trace.patch](prefill_trace.patch).

Source-only candidate, 2026-09-13. No implementation file was changed and no device was opened. Patch: `/tmp/qwen38-prefill-trace.patch`. Candidate copies: `/tmp/qwen38-prefill-trace.{model,generator}.candidate.py`. Prepared against the applied deferred-history generator and the current `in0_block_w=5` LM head.

## Scope and resulting behavior

`generate(..., trace_prefill=True)` enables a single cached prefill graph for logical prompt lengths 1 through 4096. The limit bounds trace storage to one existing prefill chunk; it does not reject a prompt. Lengths above 4096 use the existing full-context chunk loop, and `trace_prefill=False` is the eager-prefill comparison control. Decode remains split model/sample traces in either case. First-token sampling remains the existing common sampler outside the prefill graph, with the normal first-token read included in TTFT. The final history transfer stays included in decode elapsed time.

The cached graph is owned exclusively by high-level generation: B=1, slot 0, prefix position 0, final-token logits. Public `prefill_forward` retains independent output ownership and its current mixed-slot, continuation, all-logits, nonaligned-length and chunk behavior. Calling it discards the owned prefill preparation and all live traces before producing its independent outputs. This is necessary because outputs allocated after a live prefill trace can occupy that trace's scratch and remain alive until the caller releases them.

The key includes cache identity, batch, capacity, physical-page count, bound table identity and shape, slot, prefix position, logical length and final/all-logits mode. Tokens are refreshed in place. Positions are immutable for that key. The captured graph slices the bound table on every replay, so an in-place page-table change is observed on device. New keys evict all three traces before allocating new persistent buffers; no shape-indexed trace dictionary is introduced.

## Lifetime and capture order

1. Request setup allocates cache/table/positions/tokens, optional history, and common sampler state as now. A new prefill key releases every live trace.
2. Upload persistent prefill tokens and absolute positions. Run the real prefill once over those exact tensors; it advances request cache state once and warms the graph. Pad logits to the sampler's 32 physical rows, clone that output into a persistent anchor, and run the exact copy-to-anchor operation once to warm it. No model layer is rerun just for prefill warmup.
3. First-token sampling and its read use this anchor. This also warms common sampler/CCL work before any trace exists.
4. At the first decode, retain the current mutable-state backup/warm/restore logic. Capture the model trace, retaining its output allocation; then capture the sampling/history trace as now.
5. Only then record prefill and copy its transient output into the earlier anchor. Capture does not execute the graph, so it must not be replayed here: replay would overwrite the already advanced decode cache with the prefix. There is no new prefill backup/restore and no extra KV warmup.
6. Future requests with the same key reset cache normally, copy prompt tokens into their original allocation, submit the prefill trace nonblocking, sample, and read the first token. Capture-created prefill temporaries are released before either earlier decode trace replays.

The prefill anchor predates the model trace; the model output allocation predates the prefill trace. Both therefore remain protected by the allocator during each other's capture. `keep_prefill=True` is used only when decode history recording or active-slot specialization changes: all trace IDs are released together, but the warmed prefill inputs/output survive until the new pair and prefill graph are captured in the same order. Cache/history reallocation, parameter-branch changes, new prompt keys, public prefill and close discard preparation as well. The optional model positions argument validates its device-format metadata; callers must supply the absolute range matching `start_pos`, as the generator does.

Fresh G=1 calls compute prefill eagerly and do not create a decode pair solely to capture prefill. A later G>1 request can capture the pending preparation; G=1 can reuse an already existing prefill trace. G=0 remains the earlier no-work return. No dtype, decoder policy, fallback-audit or allocation-tracker setting is changed. Capture failures propagate; there is no exception-driven eager fallback.

## Counters and test consequences

- Existing `trace_captures` continues to count model/sample captures only, preserving the split-decode measurements.
- Added `prefill_trace_captures`, `prefill_replays`, `prefill_eager_calls`, `prefill_token_refreshes` and `last_perf.prefill_trace_eligible` identify actual replay, cold preparation and intentional eager routing.
- Consecutive warmed requests with one key should add exactly one prefill replay and one token refresh, zero prefill captures/eager calls, and zero decode captures. Prefill never records a token-history row.
- A changed prompt length now deliberately recaptures decode, including a length seen earlier but evicted by the single-entry cache. `tests/full_model_shapes.py:45` currently asserts that *every ever-seen length* preserves decode captures. Its functional length/output/position checks remain valid, but trace reuse should be checked for consecutive same-key requests. Keep long-prompt eager reuse coverage separately. This is a bounded cache policy, not a relaxation of valid inputs.

## Focused hardware acceptance, before adoption

1. **Ordering and lifetime smoke:** reduced layers [0,3], preallocate sufficient cache/history; run S33/G8 once and two consecutive repeats. Retain default allocation tracking. Wrap `_prefill_trace_step` with the existing `device_only` guard plus forbidden upload/read/sync functions and reject program-cache misses during capture. Assert three distinct trace IDs exist, only the prefill ID replays at TTFT, and all three IDs plus per-device addresses of prepared tokens/positions/output and decode logits persist across repeats. Confirm the first cold decode still produces the immediate eager oracle's exact complete tokens: this detects accidentally executing prefill at capture time.
2. **Fresh input and sampling:** build eager oracles with `trace_prefill=False` first, then warm and replay the same shape with two different prompts and return to the first prompt. Compare every generated token to the appropriate oracle, including fixed-seed sampled -> greedy -> sampled transitions, callbacks, immediate and deferred history delivery, G1 and G0. Eager-oracle calls intentionally evict the cache, so do not place them between measured warm trace requests. Guard warmed decode separately, leaving the permitted first/final host reads outside the loop guards.
3. **Boundary and bounded reuse:** S1,31,32,33,127,128,129,4095,4096 each followed by a same-key repeat; S4097 and S8193 must return full outputs and final positions via the old chunk path with `prefill_trace_eligible=False`, no prefill replay, and at most two live decode traces. Repeat a short key after a long request and verify a single new prefill graph, with no retained list of old traces. Full context remains covered by the existing advertised-context contract.
4. **Explicit state and table:** leave public mixed B32/nonaligned/continuation/all-logits contract checks intact, run them after an owned generation has live traces, and confirm public calls discard owned preparation. For changed table contents, a focused controlled `_prefill_for_generate` probe can reset a B1 owned cache, permute physical page IDs in the existing table buffer, run the same key, and compare raw cache-page placement or logits to the public eager oracle under that same mapping. Do not infer a table-change check merely from changed prompt outputs. Restore table contents in place and compare again.
5. **Measurement:** after reduced correctness passes, repeat the exact full-model benchmark with the same eager/deferred settings and prompt. Separate cold preparation/capture from consecutive warm TTFT. Compare all tokens against the untraced-prefill control; report measured TTFT and decode throughput separately. First-token synchronization and final history read must remain timed as before. The supplied ~83ms eager TTFT and reduced ~3062us device/~3518us gaps motivate the candidate, but are not measurements of this patch.

## Source support

- `.agents/skills/tt-enable-tracing/SKILL.md`: exact graph warmup, stable device inputs, no host upload/read/sync in capture, split model and sampling traces.
- `models/tt_transformers/tt/generator.py:601` (`_prepare_trace_prefill`) and `:663` (`_record_trace_prefill`): official separation of compile/persistent input allocation from recording, with explicit cross-trace scratch lifetime rationale.
- `tt_metal/impl/allocator/trace_allocation_tracker.cpp:117`: allocations after an active trace are conservatively marked unsafe; no allocation-tracker bypass is warranted.
- `tt_metal/distributed/mesh_device.cpp:1396` (`begin_mesh_trace`) and `:1421` (`end_mesh_trace`): trace command recording and registration; captured commands are distinct from replay execution.
- `models/autoports/qwen_qwen3_8_27b/tt/model.py:231`: explicit positions upload is the host-originating work removed from the optional device-input prefill path; existing slot/cache scatter logic is unchanged.
- Existing exact TP4 history probe's capture-at-cursor-one test independently established that recording does not execute a state update.

## Host checks actually run

`python_env/bin/python -m py_compile /tmp/qwen38-prefill-trace.model.candidate.py /tmp/qwen38-prefill-trace.generator.candidate.py`

`python_env/bin/black --check --config pyproject.toml --target-version py312 /tmp/qwen38-prefill-trace.model.candidate.py /tmp/qwen38-prefill-trace.generator.candidate.py`

`git apply --check /tmp/qwen38-prefill-trace.patch`

All pass. Device correctness, exact CCL/program-cache reuse, trace memory consumption and performance remain unverified until the coordinator's hardware probe.
