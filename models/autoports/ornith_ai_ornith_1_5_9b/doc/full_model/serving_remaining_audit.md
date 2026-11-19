# Serving contract reconciliation

This audit reconciles all 114 rows of `serving_contract.json` against the pinned
`tenstorrent/vllm-tt-plugin` commit
`042a94863ad27d88207bcf71abdf67122c6f4c27`. It supersedes the stale missing-feature
claims in `serving_gap_audit.md` and the previous version of this file. It does
not claim that a vLLM adapter, server or plugin test suite was run. The matrix
records source hashes at inspection; concurrent source changes require a final
hash refresh and matching evidence review.

At this revision: **76 supported boundaries, 0 missing rows, 38 documented
optional/single-group/text-only exclusions**. This closes the interface matrix,
not every full-model stage gate. Final maximum-context allocation, execution and measured-budget evidence now passes in `capacity_full_final.json`.

## Evidence now accepted

| Artifact, relative to this model directory | Established scope |
|---|---|
| `doc/full_model/batch32_full.json` | Full32 layers, B32/context512, mixed lengths33..64, greedy prefill matches host argmax, selected solo slots0/16/31, reversed request order, inactive row1, resident feedback/positions and zero host input uploads, explicit-feedback control, 597 unchanged programs. |
| `doc/full_model/capacity_full_final.json` | Full32 layers, B1/context262144: validated allocation from eight `(shape,dtype,tensor_idx)` triples with8192 blocks per layer; public nonaligned262143 and full262144-token prefill; finite terminal decode at262143; 820 unchanged programs; unchanged persistent allocation; measured262144-token budget with1GiB/device reserve. This is capacity/finite-output evidence, not long-context HF accuracy. |
| `readiness/prefill_resume_check.json`; `doc/full_model/logs/accepted_prefill_resume.log` | Production full32 resumed prefill at length513/starts32,68,512; top1 agreement1.0; missing-history controls fail as expected. This is the production artifact, not merely the earlier normalized-adapter experiment. |
| `doc/full_model/serving_prefill_probe.json` | Reduced real layers0/3, B3/context1024, mixed/reordered prompts and starts, cache/logit agreement, scoped reset, invalid-row atomicity, implicit horizon33→68→513 exact match, stable614 programs. |
| `doc/full_model/external_cache.json` | Reduced real layers0/3, B3/context256, global24 and arbitrary global11 pools, noncontiguous scheduler IDs, public prefill/decode equivalence, inactive peers, caller backing identities, authoritative changed/unchanged reloads, invalid page rejection and scoped causal-only clear. |
| `doc/full_model/serving_decode_probe_final2.json` | Reduced real layers0/3, B3/context256: atomic rejection before state/remap mutation, two pending outputs, observational/idempotent readback, forced same-table reload, stale inputs unread, resident remap with token/position/page coherence, dormant/inactive state, writable independent host logits. Source assertions are in `tests/full_model_serving_decode_probe.py`. |
| `doc/full_model/sampled_trace_probe.json` | Frozen real reduced-model logits: same seed repeats, seed42 versus43 differs including first token, greedy matches host argmax, seed advances in common sampling trace. It does not test unseeded admission or generated-history reconstruction. |
| `doc/full_model/sampling_state_probe.json` | Synthetic full32 sampler buffers: permutation, scoped reset/snapshot and inactive seed/count preservation. Not numerical penalty semantics by itself. |

Earlier `serving_decode_probe.json` does not contain the final atomic/remap
checks; rows now cite `serving_decode_probe_final2.json`. Earlier diagnostic
full-layer resume artifacts must not substitute for the production readiness
artifact above.

## Additional accepted evidence

- `doc/full_model/cache_ownership_full.json`: all32 layers, B3/context256,
  external29-block pool. Constructor injection performs no redundant model-cache
  allocation; a later binding retains and accounts for standalone storage.
  Invalid bindings reject atomically. One-argument release targets serving storage,
  preserves peers and is idempotent. Close drains pending transfers, readback
  remains observational after close, repeated close is safe, and caller backing
  remains allocated. Actual eight-layer BF8/BF16 cache specs and a measured
  **768-token configured budget** pass. This is not a maximum-context budget claim.
- `doc/full_model/sampling_transition2.json`: reduced real layers0/3 B3/context512.
  Seeded resume including the sampled prefill token matches uninterrupted next
  tokens, counts and seeds; sampled↔greedy reload preserves stream position;
  scoped reset preserves peers. All three BF16 penalty-oracle greedy winners
  match the common trace and differ from the unadjusted winners. Eight unseeded
  admissions vary their recorded streams/first tokens. Programs remain520.
  This supersedes failed `sampling_transition.json` for these gates.
- `doc/full_model/watcher_trace.json` and `logs/watcher_device.log`: watcher B1
  cross-request replay,35 model and35 sampling replays, no token/page host uploads.
  Watcher elapsed time is **not throughput evidence**.
- `doc/full_model/trace_alloc_tracking/positive_control.log`: allocation-detector
  positive control. This establishes detector sensitivity separately from model
  correctness or performance.

## Final interface-row closure

`plan_cache_specs` now validates all eight `(shape,dtype,tensor_idx)` triples,
requires one physical block count and exact optimized geometry/precision,
retains tensor identity labels and rejects unsupported cross-layer aliases.
Its returned kwargs feed the already device-validated external allocator.
The serving audit ran:

```text
python models/autoports/ornith_ai_ornith_1_5_9b/tests/full_model_cache_contract_test.py
Ran 11 tests in 0.033s — OK
```

These pure-host tests include full-context and arbitrary-pool plans, malformed
triples, dtype/geometry/alias rejection, measured budgets and ownership; a test
also verifies no TTNN/device import. The actual full32 allocation-from-triples
boundary at maximum context now passes in `doc/full_model/capacity_full_final.json`: eight independent BF8 layer pools, each with shape `[8192,1,32,256]`, feed the public full-context execution and terminal decode.

The documented uniform `allocate_kv_cache` fallback is unselected because the
later adapter will expose the per-layer allocator. Likewise, the verified single
group0 specs do not produce `page_tables_per_layer`; pinned docs send that argument
only for multiple groups. These two rows are justified `not_supported` conditional
alternatives, not unfinished implementations or ignored supplied arguments.

`doc/full_model/prefill_generation.json` passes all nine atomic invalid-admission
cases, including unsupported prefill trace=True, raw sampler physical32/logical
1/3/32 controls, sampled B3 prefill and B3 native fixed-pool generation preserving
inactive peers. Programs remain472; seven model/sample replays require zero token
host refreshes. `enable_trace` is now a supported **decode_only** contract:
prepared eager prefill accepts False; captured decode requires True. No captured
prefill or eager decode is advertised.

`doc/full_model/runtime_fallback_probe.json` passes reduced real-model tripwires
for host argmax/full-logit reads on the device token-output path and host-to-device
uploads during seven decode replays. Eight device tokens equal the explicit
host-greedy compatibility completion. This is a boundary audit, not full-model
accuracy, quality or throughput evidence.

## Final capacity closure and stage boundary

`doc/full_model/capacity_full_final.json` passes logical prefill lengths262143 and262144, terminal decode position262143, actual full32 allocation from per-layer triples, and a validated total token budget262144. Its1GiB/device reserve follows preparation of live traces and all persistent resources. Programs remain820 and allocator usage remains unchanged across long prefill. `doc/context_contract.json` now records `full_model.status: capacity_passed`, measured weight/KV/staging/trace/other-persistent bytes, and no advertised-context reduction. This closes the previously pending final-capacity boundary; it does not claim max-context HF accuracy or B32 simultaneous maximum-context residency.

The passed constructor-ownership, sampling-transition, prefill-generation and runtime-fallback artifacts listed above close their respective follow-up gates. Closing the interface matrix and these runtime gates is not a substitute for the broader full-model stage's accuracy, qualitative-control, performance and independent review requirements. This document does not claim stage-review acceptance.

Prefill now validates supplied sampling controls and request metadata before
preparation/admission. Named high-level sampler kwargs now merge explicit
`temperature=0, top_k=1, top_p=1` defaults. These source fixes and passing prefill-generation validation supersede the
earlier audit's defect descriptions.

## Authoritative declaration mapping

`serving_contract.json.capability_mapping` now lists **every22 pinned
model-capability key**, its typed selected value and exact documented default.
The mapping is repository-local metadata for a later adapter, not executable
vLLM integration. It requires chunked prefill, on-device sampling, async decode
and device penalties; stochastic device top_k<=32; one output token per row;
FABRIC_1D enum kwargs; version1 reload commands; and only the configured full
fixed-slot batch capacity. Optional defaults remain explicit. No extra plugin
capability keys are invented. Python enum/empty-tuple serialization is explained.

Automatic cross-request prefix caching stays disabled for `recurrent_state`.
Passing same-request readiness `prefix_cache_hit` does not supply another
request's24 recurrent/conv states or aligned tails. Speculation, device grammar,
block/adaptive outputs, lane-DP, compact/selective readbacks and stage inputs/
outputs remain documented optional exclusions. Text-only unknown multimodal or
mRoPE kwargs reject rather than disappearing silently.

Host-only min_p, bad words, logit bias, allowed IDs, min_tokens, custom processors,
structured output and TP4 logprobs use the validated writable independent
full-vocabulary host boundary. The pinned plugin owns their processing. This
does not claim its HTTP tests ran. Ordinary one-token models need no independent
`note_state_slots_moved` callback; decode owns the single state permutation.
Legacy `reset_batch` is rejected under version1.

## Inventory and evidence discipline

All114 rows retain pinned source/test references and concrete generator mappings.
Request-test inventory includes seed/temperature/top-k/penalty transitions,
unseeded and first-token variety, logprobs, host filters, structured slot reuse,
chunked needles and request isolation. Spec tests map to documented exclusion.
Capability/table lists were checked against the pinned markdown tables.

The latest matrix records hashes for production files and consumed artifacts.
Refresh those after remaining repairs. Source existence does not establish new
hardware assertions; a newer failed result never becomes a passing artifact by
citation. No single-chip, replicated-model, host-state or measured-path host-logit
fallback follows from these interface mappings.
