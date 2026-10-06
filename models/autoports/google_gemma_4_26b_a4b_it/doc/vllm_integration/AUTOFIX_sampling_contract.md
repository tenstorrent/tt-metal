# Stage 09 optional host sampling repair

Date: 2026-09-27. Status: **implemented and host-verified; host-mode device/server validation remains with the main stage**.

Starting diagnosis: [AUTODEBUG_sampling_contract.md](AUTODEBUG_sampling_contract.md). The verified failure was the adapter refusing the plugin's host-sampling submissions, which intentionally omit `sampling_params` for TP4 logprobs and host-only parameters. The earlier 26-test suite proved that boundary and the vLLM logits contract before implementation.

## Retained changes

- `tt/generator_vllm.py` accepts host fallback only when `GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1`. Missing, `0`, and `true` retain explicit refusal. Non-`None` sampling parameters always use the existing device sampler, including when the option is enabled.
- Host prefill reuses canonical last-token logits and the existing `_read_logits` TP composition. Host decode requests `return_logits=True` from the canonical generator, uploads scheduler tokens/positions every step, and returns `[slots,1,vocab]`. Prefill preserves the main stage's compact page-table row repair: it does not forward `empty_slots` into the generator's attention row index.
- Adapter mode changes release existing traces and invalidate the device parameter signature. Returning from host to device sampling reconfigures the sampler and forces an input refresh. The existing prompt/output history handoff is retained. Torch logits pass directly through synchronous/deferred read helpers; asynchronous Torch reads return an empty event list.
- `tt/generator.py` adds optional `return_logits=False` to low-level decode. Logits mode captures/replays the same model forward while omitting sampler capture, sampler replay, seed advancement, host argmax, and low-level logits readback. The adapter owns the explicit readback. Mode changes force a new trace binding, and standalone generation cannot reuse a logits-only trace. The selected precision configuration and model operations are unchanged.
- The temporary `GEMMA4_AUTOPORT_VERIFY_ASYNC` capability gate was removed after the main stage reported nine live reduced asynchronous requests matching the synchronous token reference exactly. `supports_async_decode=True` is now enabled at the main stage's instruction. This host test task did not run or independently certify that device experiment; it does not make host fallback eligible for steady device-feedback sampling.

## Hypothesis experiments

| Hypothesis | Discriminating check | Result |
| --- | --- | --- |
| The plugin omits sampling parameters for host mode and the default adapter must refuse it | Actual prefill/decode submission into the adapter with a fake generator; absent and non-`1` flags | Verified: refusal occurs before model work. |
| Explicit compatibility should return full logits, not already selected tokens | Actual plugin submit/read/finalize using distinguishable full-vocabulary synthetic logits and a fake generator | Verified `[B,1,V]`, complete values, correct row order, compact prefill tables, no sampler call. |
| Host logits should reach the installed host sampler with correct logprob semantics | Actual `_get_output_tokens` and vLLM0.26 `Sampler` on CPU | Correct token IDs, sampled-token logprobs, top-3 shape, and padded-row exclusion. |
| Host/device transitions require trace and sampling reinitialization | Device → steady device → host → device sequence | Three mode releases; device sampling config rebuilt; feedback enabled only for the steady device call. |
| Logits-only trace capture must exclude sampler work and seeds | Actual generator `_capture`, `_forward`, and `_replay` with CPU tensor stand-ins for TTNN primitives | Positions restored after capture; zero seed increments/capture/sampling/readback in logits mode; normal device counters retained in token mode. |
| Repeated host decode requires current scheduler state | Actual `decode_forward` with traced execution stubbed, distinct next-step tokens/positions, then mode switch | Host steps refresh tokens/positions; same-mode trace retained; mode switch recaptures; no low-level host readback. |

## Verification

The final suite passed **38 tests, 16 warnings in 18.27 seconds**, using the installed vLLM0.26 wheel and source TT plugin:

```bash
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q --disable-warnings --tb=short
python -m black --check --target-version py310 models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py
git diff --check
```

All checks passed. The final pytest transcript is copied to `readiness_vllm/sampling_host_contract_tests.log`. No C++/CMake changed, so no build was required. This repair task did not open a TT device or launch a server.

Remaining proof belongs to the device stage: optional-mode reduced serving, shared full sampling/host-only/logprob tests, mode switches with real traces and allocator updates, and unchanged device-performance counters. Compatibility requests intentionally read full logits; default performance requests must continue reporting zero full-logit readbacks and use the canonical device sampler.
