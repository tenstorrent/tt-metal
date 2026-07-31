# AutoDebug Report: vLLM Greedy Nondeterminism

## Starting Evidence

- Diagnosis-only pass; no implementation code was edited.
- Current failing command:
  `python -m models.common.readiness_check.run_vllm_server --model-dir models/autoports/google/gemma-4-12B --hf-model google/gemma-4-12B --mesh-device T3K --max-num-seqs 1 --max-model-len 4096 --sampling-profile smoke --server-timeout 1800 --tt-config '{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING"}'`
- Artifacts inspected:
  `models/autoports/google/gemma-4-12B/readiness_vllm/sampling_tests.log`,
  `models/autoports/google/gemma-4-12B/readiness_vllm/server.log`.
- Smoke result: `test_top1_is_greedy` and `test_min_p` pass. `TestBatchIsolation::test_mixed_params_batch` fails even with `max_batch_size=1`: prompt `Count: `, `temperature=0`, run 1 returns `12 | Time:`, run 2 returns `125\n\nSize`.
- The chat logprobs failure is unrelated for this diagnosis: server log shows missing tokenizer chat template.
- Server log confirms `sample_on_device_mode='all'`, `trace_mode` defaulting to `all`, `async_scheduling=False`, decode warmup succeeds, and vLLM-owned KV cache is allocated.
- The optimized full-model baseline only proves traced replay at a fixed position/token: `capture token 499, replay token 499, match true`. It does not prove vLLM-style repeated decode replay with changing sampling profile and increasing `start_pos`.

## Top Hypotheses

### 1. Outer decode trace key is too coarse and replays the wrong sampling graph

`Gemma412BGenerator._decode_traces` is keyed only by `sample_on_device`. During traced warmup, `WarmupForwardMixin` runs non-greedy penalty/logprob configs before greedy. The first traced `sample_on_device=True` decode can capture a random/penalty/logprob graph, then greedy requests replay that same full-model trace. `SamplingGenerator.reset_trace()` reacts to force-argmax changes, but it only clears the internal sampling trace state; the outer full-model trace in `generator.py` remains cached.

Smallest verify/refute experiment:
- Rerun the same smoke command with tracing disabled:
  `--tt-config '{"trace_mode":"none","sample_on_device_mode":"all","trace_region_size":100000000,"fabric_config":"FABRIC_1D_RING"}'`.
- If the two greedy `Count: ` runs become stable, the bug is in trace capture/replay, not KV cache allocation or vLLM scheduling.
- A narrower component check is to run the adapter warmup sequence with traced decode, then issue two identical greedy direct adapter decode sequences and inspect whether the first captured `_decode_traces[True]` survives sampling mode changes.

Likely fix boundary:
- Fix in the autoport trace boundary, not the vLLM scheduler. Key or invalidate `Gemma412BGenerator` outer traces by the effective sampling graph: force-argmax, penalties active, logprobs active, sample-on-device, and any other graph-shaping sampling mode. Do not rely on changing warmup to greedy-only; serving can still alternate sampling profiles.

### 2. Trace replay captures a Python `token_index` branch that is not refreshed

`decode_forward_traced()` refreshes token tensors, `position_idx`, `position_idx_cache`, and page table before replay, but the Python scalar `token_index` used during capture is fixed in the traced graph. In this model the short `Count: ` repro probably stays below the sliding-window branch and uses 2D RoPE embeddings, so this is lower probability for the current failure. It remains a vLLM trace contract gap because normal serving increments `start_pos` each step, while the optimized baseline replayed the same position.

Smallest verify/refute experiment:
- Direct full-model script on 1 or 2 layers: prefill `Count: `, capture traced decode at position `p`, replay at `p+1`, and compare against a non-traced decode or a fresh trace captured at `p+1` with the same KV/page table.
- Repeat with a start position just below and just above `sliding_window=1024` to test the Python `decode_position >= sliding_window` branch.

Likely fix boundary:
- Either avoid graph-shaping Python scalars in traced decode, or key/rebuild the outer trace by token-position branch class. The device position tensors should continue to carry the actual `cur_pos` for paged update/attention.

### 3. Request-level sampler reset is not coupled to the outer trace lifecycle

The plugin marks `reset_batch=True` when the persistent batch changes, and for this smoke `async_scheduling=False`, so async result reordering is unlikely. However, `reset_batch`, `slot_remap`, seed refresh, and sampling parameter resets are applied before calling `decode_forward`; none of those state changes invalidate the outer full-model trace. A stale trace can therefore ignore a correct request reset and keep replaying the graph captured under a previous warmup/request state.

Smallest verify/refute experiment:
- Add temporary trace-only logging (or a local probe script) for: request id, `reset_batch`, `slot_remap`, temperature/top_k/top_p, force-argmax flag, penalties/logprobs flags, and outer trace id before every decode. Confirm whether both `Count: ` requests use the same outer trace id despite a request reset.
- As an A/B, run with `--tt-config '{"trace_mode":"decode_only","sample_on_device_mode":"all",...}'` and with `trace_mode":"none"`. If only traced modes fail, async/scheduler ordering is refuted for this repro.

Likely fix boundary:
- Keep vLLM-owned KV cache and plugin request bookkeeping intact. Couple adapter sampling-state changes to the generator trace cache, either by passing an explicit trace signature from `generator_vllm.py` or by making `Gemma412BGenerator.decode_forward_traced()` derive a complete signature from `self.model.sampling`.

## Current Verdict

Most likely cause: stale outer full-model decode trace reuse across sampling profiles and request resets. The cache ownership path appears healthy: vLLM allocates per-layer KV cache, page tables are passed through, and the sync decode path is used for this smoke. The next engineering step should be a trace-mode A/B run; if `trace_mode=none` makes the greedy isolation test deterministic, implement the trace-key/invalidation fix at the autoport generator boundary.
