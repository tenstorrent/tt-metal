# Gemma4 TTFT source diagnosis

Scope: isolated AutoDebug source inspection only, 2026-09-28. No model execution,
hardware use, serving measurements, or implementation edits. Performance benefit
below is a hypothesis pending the parent's matched serving baseline/experiment.

## Supported findings

- `tt/generator.py:256` uploads page tables and exact logical token rows, runs
  `model.prefill_forward` once per request row, and concatenates last-token logits.
  It has no prefill trace. `sample_prefill` at line 282 explicitly disables the
  canonical sampler's trace. The adapter calls both at
  `tt/generator_vllm.py:208`, then reads only sampled tokens on the device path.
- `tt/model.py:217` embeds the full logical sequence and invokes every decoder,
  but already slices the final hidden row before final norm/head for token-out.
  Thus skipping intermediate output heads from the Qwen inspiration is not an
  immediate Gemma optimization: Gemma already performs one final head per row.
- `configure_sampling` (`tt/generator.py:142`) releases traces unless explicitly
  reused. Decode `_bind` (line 348) always releases traces and creates input,
  position, table and output buffers. `_capture` (line 377) warms model and sampler
  before capture, restoring tokens, positions and seeds. Simply adding a prefill
  trace that survives `_bind` would violate that existing allocation discipline.
- `_release_trace` resets *all* canonical sampler trace states. The common sampler
  keys traces by penalties/logprobs/argmax/bucket, validates input/output Python
  tensor identity, and captures rather than replays on first `sample(trace=True)`.
  See `models/common/sampling/generator.py:155,189,289,343,365,430`. Flipping
  `enable_trace=False` to true alone neither creates a stable input nor guarantees
  a computed first output. Explicit capture then replay is required on a miss.
- Gemma uses persistent device seeds itself; prefill does not increment them,
  while decode `_forward` does. Adapter resumed-prefill handling restores seed
  offsets and partitions prompt/generated tokens for penalties (lines 189–207).
  Sampler `precompile` suppresses token counting; capture includes counting for
  subsequent replay. Warmup must not introduce an extra real draw/count.

## Proposed first experiment

Implement an opt-in, **one-entry generator-owned B1 prefill trace**, initially
for an exact logical length at or below the native 1024-token chunk, slot zero,
last-token logits, and full-prompt start zero. This is an acceleration eligibility
rule, not a new public length restriction. Preserve eager behavior for other
lengths, batches, slots, compatibility logits, and unsupported signatures.

The entry owns token input, persistent per-layer page-table inputs, a padded
32-row sampler-logit staging tensor, and first-token output. Its signature includes
exact length, slot, layer selection/policy, cache tensor identities (including
underlying views/bindings), table shapes/specifications, and sampling graph mode.
Page-table *contents* and token values are refreshed in place, not included as
constants. Keep the external cache alive by reference; never clear or replace it.

Use the existing model for the prefill graph and the common sampler for sampling.
Capture the model-to-persistent-logits copy as part of prefill. Capture a separate
first-token sampling graph against that stable input; either give the canonical
sampler a distinct namespace and explicit lifecycle, or capture its eager public
`sample(..., enable_trace=False)` call in a generator-owned trace. The latter
avoids colliding with decode's identity-keyed sampler cache, but all warmup and
penalty semantics still apply. Do not include `_Gemma4SamplingGenerator`'s public
decode-token copy unless the first-token output is intentionally bound to it.

Recommended capture sequence follows c901b7d879ca: the first request prepares
persistent prefill inputs/output and runs eager prefill/sample; decode bind/warmup
then completes before capture of decode, prefill and first-token sampler graphs.
Prepared prefill buffers must survive that bind; trace IDs need not. A subsequent
matching request refreshes inputs and replays. If capturing on the first prefill
instead, invalidate those traces before decode allocation/warmup and explicitly
recapture after decode preparation; otherwise repeated requests will never gain
reuse. Measure first-use setup separately from warm replay.

After the B1 lifetime experiment is sound, permit exact nonaligned lengths within
the bound (e.g. 33, 127, 129, 1023), then investigate longer chunks or generic
eager-prefill-to-persistent-sampling staging. A single-entry cache bounds memory
and naturally falls back on changing signatures; avoid a trace per arbitrary
length without a capacity/eviction policy.

## Lifetime and correctness requirements

1. Release conflicting traces before any new shape compiles or persistent buffer
   is allocated, and before external-cache/table-shape replacement. Retained
   public outputs must not alias scratch overwritten by another trace replay.
   Do not treat `mark_corruptible` as proof that live user data may be overwritten.
2. Keep per-layer hybrid tables distinct: the adapter requires scheduler tables
   for sliding/full groups, and cache views can share physical allocations across
   different logical geometries (`generator_vllm.py:98–127`). Refresh every table
   actually captured. Preserve compact prefill rows; `empty_slots` are not the
   prefill page-table row map (`generator_vllm.py:213`).
3. Exact length affects slicing, attention masks, cache fill and the selected
   last token. Do not round logical prompts to a trace bucket by padding without
   an explicit valid-length contract through the graph.
4. Long prefill at `tt/multichip_decoder.py:1177` performs Python chunk assembly,
   sliding-tail lifecycle and partial-tile handling, with 1024-token chunks.
   Leave that existing path intact initially. Its persistent tail buffers need a
   separate lifetime audit before replaying long traces.
5. Preserve request seed refresh, resumed output offsets, prompt/output penalty
   partition and exactly one output-token count per actual sample. Sampling mode
   changes can reset common traces; stale generator IDs must not survive them.
6. Reset/teardown and capture exceptions must release every new trace before
   dropping its backing buffers. Reset must retain external cache contents.
   Keep asynchronous decode/read ownership and nonblocking execution unchanged.

## Inspiration commits assessed

| Commit | Applicable lesson |
| --- | --- |
| `c901b7d879ca` | Closest pattern: bounded owned B1 prefill, stable sampler staging, capture after decode warmup, generic eager fallback and independent public outputs. |
| `fac11633c959` | Optional deployment warmup and counters can distinguish compilation from steady state. Warmup must be workload-independent and reported, not hidden from first-use TTFT. |
| `585e04e5c8fe` | Equal-length grouped prefill is separately gated and preserves fallback. Gemma's flattened embedding and per-user attention mean this is a larger follow-up, not a direct port. |
| `42dbce6bf08b` | Grouped final heads may help a future batched path. Gemma already slices its final token before its head and has no repeated generator-chunk head to remove. |
| `22bc096710f3` | Occupancy-selected decode shapes require explicit state/slot mapping. This is separate from first-token tracing. |
| `df19975ffac2` | Resident state avoids repeated gather/scatter but needs flush/suspend on ownership changes. Qwen recurrent-state machinery does not directly apply to Gemma attention caches. |

## Validation for the implementation owner

Compare eager and traced outputs on the same cache/table geometry with changed
tokens, changed page IDs, external cache replacement, exact nonaligned lengths,
multiple slots/batches using fallback, and lengths beyond the fast-path bound.
Exercise greedy, seeded top-k/top-p, each penalty, sampling-mode changes, resumed
prefill and repeated teardown/reset. Check fresh then reused traces and decode
after prefill, not only isolated prefill replay. Existing starting points:
`tests/check_vllm_batch_cache.py`, `check_vllm_resumed_prefill.py`,
`check_vllm_seed_lifecycle.py`, `check_full_prompt_lengths.py`,
`test_vllm_prefill_slots.py`, and `test_vllm_sampling_contract.py`.

Add counters for prefill eager/capture/replay, sampling eager/capture/replay,
signature invalidation and table refresh. Accept performance only from matched
real-server TTFT/throughput/quality evidence with cold/warm status recorded.
