# Generator serving contract AutoFix

Diagnosis: `AUTODEBUG.md`, sections1–4. The shared generator accepted public
`temperature` but passed it directly to a kernel which multiplies logits by
inverse temperature. Its public low-level decode always uploaded host tokens and
positions, incompatible with the plugin's resident-input commands. A parameter
reload also reset RNG counters, and recurrent remapping did not move dormant
sampling state.

Repairs in `tt/generator.py`:

- Preserve the existing common Sampling1D and model/sample traces. `sample_prefill`
  delegates terminal prefill selection to that same sample trace, also used by
  standalone generation.
- Convert public temperature to inverse temperature once. Keep the old
  `set_sampling_params(..., seed=1234)` default behavior; `reset_seed=False`
  updates only k/p/inverse-temperature, preserving host initial seeds and live
  device counters. `reset_sampling_state(seed=..., offsets=...)` explicitly
  reconstructs counters at a drained authoritative scheduling boundary. The
  adapter owns the offset convention and must not derive it from stale host
  positions on resident steps.
- Factor authoritative/resident decode into one replay/output path. Resident
  calls never inspect host tokens/positions, retain active-row metadata, and
  independently refresh a changed page table. Prefill, reset, and remapping
  invalidate resident metadata until an authoritative decode submission.
- Preallocate serving-only recurrent/sampler gather indices and scratch before
  trace capture. Remapping moves convolution, FP32 recurrence, k/p/temp and live
  seed counters once, including a dormant sampler. Reset clears only selected
  recurrent slots. Warmup exercises these operators before any capture. No
  attention page ownership or sampling implementation changes were introduced.

Allocation lifetime: request-state gather scratch and indices are allocated by
construction and survive all traces. Each gathered state is copied back before
scratch is reused for another layer. State tensors retain their original
addresses. Reset's temporary `where` result is immediately copied back and not
retained. Both operations are warmed before trace capture; acceptance still
requires representative device replay with tracking enabled. No allocation
acknowledgment or broad corruptible scope was added.

Validation run (host only, no TT device opened):

```sh
docker exec qwen38-functional-build \
  bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh \
  python -m pytest models/autoports/qwen_qwen3_8_27b/tests/test_serving_generator_contract.py -q
```

Six tests passed in2.04s; exact artifact:
`readiness_vllm/generator_host_tests.log`. Cases cover T=.5/1/2 reciprocal
uploads, parameter-only preservation of live RNG/initial seeds, explicit
seed+offset reset, reserved RNG sentinel rejection, poisoned stale host inputs
through resident and page-only decode, and cyclic recurrent/dormant-sampler
remapping. Replay tests stub the device operations and establish host control
flow only; they do not establish TT kernel correctness.

`tests/probe_serving_generator_contract.py` is a separate device probe for the
stage owner to run serially. It uses real layers0/3, B2, context512, canonical
sampling and all prepared buckets. It compares authoritative and resident token
outputs, positions, and every cache field, including a consumed page-table change
across32 tokens. It poisons stale host inputs and checks refresh counters.
It is reduced integration evidence, not full-model accuracy/performance.
The subagent did not run this device probe because the parent owns hardware
serialization. Device acceptance, exact FP32 reset/remap behavior, and tracker
acceptance remain unverified until that result exists.

Remaining scope: prefix snapshots/restoration, plugin continuation ownership,
adapter async event/read lifetime, seeded request-offset translation, chunked
prefill and all-layer serving gates belong to their respective stage work.
No performance improvement or completed serving gate is claimed here.

## Device follow-up run by stage owner

`generator_device_contract.json` passes the reduced real0/3-layer comparison:
tokens, positions and all cache fields are exact; resident token/position
refresh deltas are zero and the consumed-page mapping changes once.

The first adapter device check passed those same async/page assertions but found
that cyclic state remapping copied stale scratch. `ttnn.gather(...,out=scratch)`
for a non-last dimension transforms optional output into a separate allocation
(`gather.cpp:377–400`); the return tensor, not original scratch, is authoritative.
Both recurrent and sampling remaps now copy the returned tensor, immediately
release the temporary, and warm the exact path before capture. The host regression
now deliberately returns storage different from `out`; six tests pass in2.01s.

`adapter_device_contract.json` then passes exact sampled tokens, positions and all
state fields, page-only growth, zero steady token/position refreshes, four retained
traces without recapture, a step-N async read consumed after submissionN+1, and
exact live-state remap plus single-slot reset preserving its peer. Host logits
reads are zero. These are reduced targeted adapter checks, not all-layer server
or scheduler-driven page-allocation evidence.

### Peer RNG preservation during partial-batch terminal prefill

A terminal prefill group previously reset and advanced counters for every fixed-batch slot, including concurrent peers that did not produce an output token. `reset_sampling_state(slots=...)` now changes only selected live counters and host base seeds. `sample_prefill(active_slots=...)` still replays the canonical split sampling trace, then restores untouched counters from a preallocated device backup. Native UINT32 `where` performs exact selection without float conversion or host state readback. Mask/update/output tensors and backup are allocated during serving generator construction; both merge routes are warmed before captures. Standalone calls without slot arguments retain their existing behavior.

Host validation: `tests/test_serving_generator_contract.py` — 7 passed (2.22 seconds). The extended `tests/probe_serving_generator_contract.py` checks all mesh replicas with counters spanning the UINT32 domain, subset reset, alternating slot samples and stable capture count. Device verification is delegated to the parent for serialized execution; this entry does not claim that pending device check passed.

### Mixed-feature seeded reproducibility diagnosis

`sampling_tests.log` records `test_mixed_params_batch` failing with seed42/temperature2 (`'1\n#'` versus `'21\n'`). The source-level cause supported by routing evidence is mixed RNG ownership: the test includes penalty requests, the adapter advertises `supports_device_penalties=False`, and plugin `model_runner.py::check_perform_device_sampling` selects host sampling for the whole submitted batch when penalties are present. Restricting top-k to32 does not prevent this transition.

`full_b32_events.jsonl` events0–18 show the first mixed run entirely on host (sampling replay count0, host-logits reads0→13). Events19–21 show the second run admitting a length3 request alone and taking one canonical device sample. Events22–37 then show penalty companions arriving and host-logits reads13→26 while the sampling replay count stays1. These are different random streams, not evidence of request-slot-dependent device RNG. `manual_seed` selects each core's supplied seed by user ID; `rand_tile_init(seed)` uses stream ID0, and the Wormhole SFPU initializes directly from that seed. Sampling reads the same random-tile lane offsets0/2 for every user core.

`tests/run_sampling_route_controls.py` separates three reordered device-only batches (no penalties, explicit bounded top-k, zero host-read delta required) from two reordered host logprobs controls at identical first-token prefixes. It writes raw responses and logprob comparisons. This script was prepared without issuing HTTP requests; execution is delegated to the parent. Explicit `seeded_sampling_policy=host` can stabilize the shared compatibility suite; performance retains canonical device sampling with host compatibility disabled.
