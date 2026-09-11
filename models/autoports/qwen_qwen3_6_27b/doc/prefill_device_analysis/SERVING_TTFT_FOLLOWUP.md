# Serving TTFT follow-up: request-boundary decode capture

## Finding and evidence scope

The native prefill speedup has not removed the main serving request-boundary
cost. The adapter releases its model and sampler traces for every prefill and
rebuilds them on the first decode. Matching HTTP controls show that requesting
a second output token adds **520.660 ms** to first streamed-text latency. This
strongly implicates first-decode setup and its placement in the delivery path;
the entire delta has not yet been attributed with server span measurements.

This follow-up now includes an instance-only runtime prototype and three
bounded four-layer device runs. The prototype remains **experimental and
unselected**: correctness passed, but allocation tracking dominates timing and
full-model/HTTP improvement is unmeasured. See
[SERVING_TRACE_REUSE_RESULTS.md](SERVING_TRACE_REUSE_RESULTS.md) for both the
initial timing regression and its refinement. No production defaults, server
configuration or sampler RNG behavior were changed by this experiment.

| HTTP output limit | Median TTFT, three warmed requests | Difference from output limit 1 |
| --- | ---: | ---: |
| 1 | 330.309 ms | — |
| 2 | 850.969 ms | +520.660 ms |
| 8 | 868.426 ms | +538.117 ms |

Source: [serving_ttft_controls.json](artifacts/serving_ttft_controls.json).
Requests use `/v1/completions`, 128 repetitions of token ID 1000, no prefix
caching, and first **nonempty streamed text** as the endpoint. Each output
length has one discarded warmup and three samples. These are lifecycle/perf
controls, not qualitative evidence: their repeated-word prompt and output are
deliberately synthetic. The result does not prove exactly when the server
internally sampled the first token.

The parent investigation reports about 119 ms for native full-model B1 S128
prefill, about 1299 ms for S4096, and about 891 ms actual S128/output-252/C1
serving TTFT. The serving launch still allocates 32 slots. Those measurements
have different boundaries and are not interchangeable. Even reducing the HTTP
output-2 TTFT by the full 521 ms control delta would leave about 330 ms, well
above a 60 ms target. Trace reuse is the next hypothesis, not completion of the
serving performance objective.

## Exact source path

Paths below are relative to `models/autoports/qwen_qwen3_6_27b` unless otherwise
specified. Line references describe the inspected working tree.

1. `tt/generator_vllm.py:273` maps the compact prefill request into fixed slots.
   At line 291 it calls `gen.reset_slots(slots)`, then resets sampler request
   state, refreshes the generator-owned page table and runs prefill. Line 313
   sets `_decode_ready=False`. Prefill sampling is eager at line 318, followed
   by a token readback; logits are released before returning to the runner.
2. `tt/generator.py:814` unconditionally calls `_release_traces()` before the
   slot reset. `_release_traces` fences outstanding work, releases model and
   sampler traces, deallocates persistent token/position/mask tensors and cache
   backups, and clears captured output aliases (lines 821–866).
3. The plugin's `/tmp/qwen-ci-vllm-plugin/src/vllm_tt_plugin/async_decode.py:233`
   computes `reload_inputs=True` when the resident decode chain is invalid or
   its layout changes. Versioned flags are passed to `decode_forward` at lines
   823–867. The newly added contract therefore correctly requests authoritative
   inputs on a transition; it does not require rebuilding an identical graph.
4. `tt/generator_vllm.py:430` currently translates that reload into
   `setup_token_out_decode` at line 450. Setup in `tt/generator.py:547` releases
   traces again, resolves cache/page ownership, refreshes sampler parameters,
   calls `_capture_token_out_trace`, then writes the real token and position.
5. `_capture_token_out_trace` at line 420 allocates stable inputs; clones cache
   state; warms restore copies; executes full decode, position advance and
   sampling for warmup; captures model and sampler separately; and restores
   cache state (lines 466–536). There are real synchronization and data-copy
   costs in addition to Python graph construction.

**The serving path already disables full-attention cache backups.**
`generator_vllm.py:74–78` sets `trace_backup_attention_cache=False`; capture then
backs up the 48 linear layers' conv/recurrent state, not the large vLLM K/V pool.
Do not present removal of an attention-cache clone as a new fix. The selected
32-slot configuration still clones all allocated linear slots even for C1.

The plugin invokes `model.decode_forward` synchronously while submitting the
first async decode. Its model runner constructs the next step before returning
the async wrapper (`model_runner.py:1697–1714`). This provides a concrete place
where capture can block engine progress. Source plus HTTP controls support the
hypothesis, but a timestamp at first-output publication is still required to
prove the exact delivery dependency.

## State and ownership ledger

| Boundary | Existing behavior | Reuse requirement |
| --- | --- | --- |
| Token and position | `_seed_token_out_trace`, generator line 661, copies host values into persistent captured tensors | Keep allocations; call only on authoritative `reload_inputs`, never from stale steady-decode host values |
| Active masks | `_trace_active_mask` advances positions; `_trace_active_state_mask` protects inactive caches | Refresh both in place when supporting a changed active set; initial prototype requires the identical active slot |
| Page table | `refresh_page_table`, line 800, copies changed host contents into `_page_table` | Same allocated shape/layout and generator-owned buffer; new contents are data, not a graph change; external ownership or changed shape recaptures |
| KV binding | `_check_state`, line 133, can bind a new cache collection | Verify every underlying cache allocation before any reuse; Python container equality alone is insufficient |
| Slot reset | Model line 372 masks linear caches in place; K/V remains scheduler-owned | Drain previous request before mutation; preserve `_slots_requiring_prefill` rejection until prefill completes |
| Fused conv | Model lines 341–370 temporarily borrow composite layout | `multichip_decoder.py:598` copies back into the original window and restores its object; check that original address before replay |
| Narrowed prefill | Model lines 543–640 temporarily substitute one-slot cache rows and batch-index metadata | Final copy restores original full-slot cache addresses and batch geometry; no replay while the view is active |
| Slot remap | Model lines 408–431 gather rows, then copy into original linear caches | Data transition can be compatible with stable addresses, but permutation-specific slice programs need warmup; defer nonidentity remap in initial prototype |
| Sampler | Its trace requires exact captured logits/output object identities (`models/common/sampling/generator.py:302`) | Keep captured model logits and token output objects; preserve current parameter, history and seed lifecycle |
| Sampling parameters | Ordinary K/P/temperature updates copy into persistent tensors; force-argmax transition resets sampler traces | Initially require identical unseeded, no-penalty/no-logprob sampling contract; broader modes need separate graph keys and lifecycle validation |
| Program-owned buffers | First use can allocate lasting DRAM/L1 metadata even if user tensors later die | Exact prefill/reset/sampler/refresh shape envelope must be warmed before capture; no unverified surviving post-capture allocation before replay |

The final row is required by the earlier [AUTOFIX](AUTOFIX.md): retaining inputs
and outputs alone did not prevent a recurrence trace from corrupting later
eager work. Stable cache addresses do not prove the rest of the allocation graph
is safe. This follow-up does not propose global post-operation hooks or monkey
patching `ttnn.deallocate` for a serving model.

## Minimal guarded prototype

The first runtime experiment should cover repeated **C1, same physical/logical
prefill shape, same allocated batch, same active slot, same unseeded sampling
contract**. Token IDs and page-table contents may differ. Keep native GDN and
the current sampler implementation unchanged.

1. Save a contract after the existing first successful capture: model/mesh and
   precision identity, exact prefill geometry and modes, active slot, sampling
   contract, model/sampler trace handles, all persistent trace I/O identities
   and buffer addresses/specs, cache bindings, and page-table ownership/spec.
   Record the prefill/reset/sampler program envelope warmed before capture.
2. At the next prefill entry, before reset or cache rebinding, compare that
   contract. On a mismatch release normally **before** running the new shape.
   On a match drain pending work and perform the current linear reset in place,
   retaining trace handles and all persistent trace storage. Leave
   `_decode_ready=False`: request inputs still require an explicit reload.
3. Execute eager prefill and the existing eager first-token sampler. Retain the
   existing request seed/history operations. Finish and release request
   metadata, narrowed cache rows and prefill logits. Check original cache
   ownership was restored. If program-cache entries changed or any surviving
   post-capture allocation remains unexplained, release the traces before any
   replay and use normal setup. A cache-count match alone is insufficient.
4. On explicit first-decode reload, with the same captured owners still alive,
   refresh page-table contents and use `_seed_token_out_trace` to refresh token
   and position. The initial same-slot restriction keeps both captured active
   masks valid. Skip decode warmup, cache snapshots and both captures. Let the
   adapter perform its current authoritative history reset and seed update in
   their existing order, then replay model and sampler.
5. Preserve the normal release path for teardown, failure, cache rebind,
   nonidentity remap, shape change, sampler-graph change or unsupported mode.
   An identity remap could eventually become a no-op, but avoid changing the
   plugin's seed-slot remap behavior as part of this performance experiment.

The reviewable host candidate is
[serving_trace_reuse_plan.py](serving_trace_reuse_plan.py). It returns explicit
preserve/refresh/recapture/reject decisions and requires observed allocation
evidence before allowing replay. The policy itself imports no TTNN. It is now
used by [serving_trace_reuse_runtime.py](serving_trace_reuse_runtime.py), which
patches only the supplied adapter/generator instances and restores their methods
on exit. [validate_serving_trace_reuse.py](validate_serving_trace_reuse.py)
compares that adapter against forced recapture. Neither is a production default;
the reduced results do not establish a full-model or HTTP TTFT reduction.

Host validation command:

```bash
python3 models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/validate_serving_trace_reuse_plan.py
```

The checks exercise cold setup, request/slot/cache/page-owner changes,
sampling-mode boundaries, new cached programs, missing allocation evidence,
lost sampler trace, stale scheduler inputs and decode before required prefill.
Results: [serving_trace_reuse_host_validation.json](artifacts/serving_trace_reuse_host_validation.json).

## Next measurements and acceptance

- Add host spans around adapter reset, native prefill, eager prefill sampling,
  decode setup, cache backup/restore, model capture, sampler capture, first
  runner result publication and first HTTP chunk. Count captures and input
  refreshes per request. Existing `QWEN36_DECODE_LOG_SETUP` can count setup;
  `QWEN36_DECODE_PROFILE` adds synchronizations and should be a diagnostic run,
  separate from final HTTP timing. No serving Tracy/device-profiler run is needed.
- Compare current setup against guarded reuse with the same max-output 1/2/8
  HTTP matrix and S128/output-252/C1 benchmark. The hypothesis predicts the
  output-2 minus output-1 gap falls when warm requests no longer capture.
- Before a performance claim, compare the same two different real prompts
  sequentially against forced recapture. Verify prefill tokens, subsequent
  decode tokens, token/position feedback and cache ownership; verify new
  page-table contents with the same shape. Same prompt repeated alone does not
  test stale inputs.
- Extend only after that passes: a different active slot, changed length and
  ragged tail, cache-page allocation changes, abort/restart, C>1 and remap. All
  unsupported transitions must recapture cleanly. Seeded/sampled/penalty/logprob
  modes remain separate until their current quality investigation is resolved.
- Separately time fixed-slot reset and narrow-view splice costs, eager
  prefill sampling, logits expansion back to 32 rows and scheduler delivery.
  Removing capture alone cannot close the remaining 330 ms control floor.

## What the pipeline missed

The subsequent [slot-logits scatter probe](SLOT_LOGITS_SCATTER.md) identified a
second concrete cost outside B1 model timing. Restoring the 32 sampler slots
used BF8 `zeros`, which packs and uploads padded host data. A device zero fill
and repeated-row concat reduced the isolated scatter from 176–185 ms to
0.5–1.0 ms with exact outputs across four ranks and three active-slot positions.
The coordinating agent integrated that change; full-serving validation remains
separate from the static result.

The earlier generator speedup measured the native model computation, while
serving rebuilds a second graph after prefill and before the first decode.
There was no request-lifecycle capture count or output-limit TTFT control to
expose that cost. Fixed 32-slot linear-state snapshots and conv layout round
trips also sit outside a B1 layer timing; the source comment that request-boundary
conversions "cost nothing that matters" is not supported for TTFT. Warmup hooks
in `generator_vllm.py:545–551` are no-ops, so server startup does not absorb the
exact first-request capture setup.

The explicit plugin reload fix correctly separates stale steady-state inputs
from authoritative transitions. The remaining abstraction conflates
**reload request data** with **destroy and recapture the graph**. Improving that
boundary requires verified allocation ownership, not removing the reload signal.
The guarded four-layer runtime now passes repeated-request cache and token
comparisons with zero surviving unsafe allocations, but production ownership
integration without debug tracking, full-model validation, retained-memory
accounting and actual HTTP improvement remain open. The coordinating agent
retains responsibility for promotion and the serving objective.
