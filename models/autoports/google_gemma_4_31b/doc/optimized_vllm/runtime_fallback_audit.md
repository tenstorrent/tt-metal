# Runtime fallback audit

## Measured path

The measured path is the TT vLLM plugin with
`TT_GEMMA4_TEXT_VER=gemma4_31b_autoport`, `sample_on_device_mode=all`, trace
mode `all`, and async scheduling. Its steady decode sequence is:

1. `TTAsyncDecodeController.submit_decode()` proves unchanged scheduler state
   with `can_use_steady_decode_fast_path()` and passes the explicit reuse flag.
2. `Gemma4ForCausalLM.decode_forward(..., read_from_device=False)` calls
   `Gemma4Generator.decode_next_token_traced()` without inspecting host token,
   position, or page-table contents.
3. The model trace and greedy TP4 sampler trace execute nonblocking. The sampler
   writes the selected token into the persistent model trace token tensor.
4. The device output is returned across the async boundary; the plugin owns the
   one deferred contract read.

## Findings

- No host greedy/top-1 `argmax` is present in the adapter decode implementation.
- No full-logits readback is used by device-greedy decode.
- No generic eager or force-argmax sampler is selected by semantic greedy.
- No host sampled-token read/write feedback loop is present.
- No standalone KV-cache assumption exists; serving requires caller-owned
  vLLM cache handles and hybrid per-layer page tables.
- No per-token `torch`, `ttnn.from_torch`, `ttnn.to_torch`, tilize/untilize,
  reshard, blocking read, or page-table equality check occurs in the proven
  steady branch.
- Token, RoPE position, cache position, page tables, sampler output, and sampler
  work tensors persist on device across trace replays.
- Scheduler-state changes deliberately leave the steady branch and refresh or
  recapture. Those transition operations are correctness work, not steady-token
  fallback.
- A scheduler `new_block_ids` update is an explicit page-table-change signal:
  it drains prior overlap and performs one persistent table refresh without
  setting `reset_batch` or recapturing traces.
- Each distinct persistent page-table source/target copy program is prewarmed
  before trace registration, so the first 64-token boundary does not allocate a
  device program while traces are live.
- Host tensor conversion functions remain for prefill, API output processing,
  explicit stochastic/structured-output compatibility tests, and reset or
  dynamic-batch transitions. Neither benchmark selects those as its decode
  sampling path.

The fresh complete server log contains zero unsafe allocator warnings, zero
traceback/error matches, and records nonblocking model/sampler trace IDs for
batch 1, 4, 8, and 32. Sampling, a long batch-1 request crossing page-table
boundaries, primary/CI benchmarks, and shutdown all completed.

Evidence: `evidence/adapter_contract.xml`,
`evidence/plugin_lane_contract.xml`, `after/sampling_tests.log`,
`after/logit_determinism.json`, and `evidence/final_server.log`.
