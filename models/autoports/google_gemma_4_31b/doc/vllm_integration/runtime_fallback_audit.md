# Stage 09 runtime fallback and cleanup audit

## Measured performance path

The primary and CI benchmarks use `temperature=0.0`, no penalties, no
logprobs, and no structured-output constraints. The plugin's model-scoped
`sample_on_device_policy="greedy_only"` therefore selects device sampling.
The adapter delegates first decode to
`Gemma4Generator.prepare_token_out_decode(..., pad_to_max_batch=False)` and
steady decode to `decode_next_token_traced()`. The sampler trace writes its
token directly into the persistent input consumed by the next model trace.

Source audit of `Gemma4ForCausalLM.decode_forward()` confirms that this path
contains none of the forbidden feedback/fallback operations:

- no `argmax` or generic top-k greedy fallback;
- no `read_sampled_token()`;
- no `write_teacher_forced_token()`;
- no full-vocabulary logits conversion/readback;
- no Python token readback/writeback feedback loop.

One sampled token must be returned asynchronously to vLLM for scheduler/API
state. That required D2H result is an output boundary, not model token
feedback: steady model input and position advance entirely on device.

## Optional host compatibility

`GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT=1` is explicit and exists for the shared
stochastic, penalty, allowlist, structured-output, and logprob tests. When the
batch requires those semantics, `sampling_params is None` enters eager logits
decode and vLLM's host sampler. The branch synchronizes/releases any canonical
decode traces before page-table conversion and uses the same vLLM-owned cache.
It is never selected by either benchmark profile and does not replace the
traced device-greedy performance path.

## Cache and trace ownership

- vLLM allocates and owns the hybrid KV cache and page tables.
- Serving constructs `Gemma4Generator(..., allocate_standalone_cache=False)`;
  absent external handles fail at the adapter boundary.
- Dynamic decode trims only the front-packed active prefix and keys the
  cooperating traces by physical active batch. Shared plugin wire padding and
  advertised `max_num_seqs=32` remain intact.
- Trace teardown drains CQ0 before releasing model/sampler trace programs or
  bound buffers. Batch-shaped page-table allocation happens only after that
  release.
- The inherited allocator warning during second sampler-trace registration is
  classified with controls in `anomaly_ledger.md`; it is not a transition-time
  allocation.

## Final cleanup

The final runner terminated the API server and EngineCore cleanly. A subsequent
process audit found no live vLLM, API-server, runner, or EngineCore process; the
only matches were historical PID-1 zombie EngineCore records, which cannot hold
devices. `tt-smi -s` then reported all four P150b boards with healthy DRAM,
PCIe Gen5 x16, advancing heartbeats, and normal temperatures. The final log has
one inherited allocator warning during cooperating trace setup and no runtime
fallback on either benchmark path.

The audited final server configuration was `max_model_len=113280`,
`max_num_seqs=32`, 64-token vLLM blocks, `sample_on_device_mode=all`,
`trace_mode=all`, a 256 MiB trace region, `FABRIC_1D`, model warmup, and async
scheduling on a `1x4` P150b mesh. The primary 127-actual-input/128-output
single-user benchmark and secondary 99-actual-input/100-output/32-request burst
both used the device-greedy traced path. The optional host compatibility path
was exercised only by shared tests requiring host-only sampling semantics.

Firmware 19.9 emitted the expected warning that 19.5 is the latest fully tested
Blackhole bundle. No firmware recovery, eager serving fallback, or alternate
model path occurred. Process cleanup left no live vLLM/API/runner/EngineCore
device holder; historical EngineCore records adopted by PID 1 were zombies and
cannot retain device resources.
