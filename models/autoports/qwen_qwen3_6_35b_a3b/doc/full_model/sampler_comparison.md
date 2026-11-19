# Common sampler comparison

Recommend `models/common/sampling.SamplingGenerator` / `TTSampling` for this model, with a generator-owned trace wrapper or a narrowly verified shared fix for the capture issues below. Reject `Sampling1D` as the primary serving sampler because it leaves penalties, request RNG management, slot remapping and trace ownership to the caller. The selected implementation now has a traced terminal-contract benchmark on the retained 1x4 Blackhole mesh; full-model quality and performance gates remain separate.

## Sources and provenance

- Current checkout: tt-metal `121ab5df0f494787236fff2b371e422e6edb599d`; inspected `models/common/sampling/{generator.py,tt_sampling.py,tt_penalties.py,tt_log_probs.py,vocab_padding.py,README.md}`. Source takes precedence where README is stale.
- `python_env/bin/python -m pip show tt-transformers` reports no installed package. No environment changes were made.
- Read-only sibling checkout `/home/ttuser/qwen36-35b-a3b/tt-transformers-contract` is detached at `c95935d8934f136833f02f306f32deb8a1136177`. This is the exact runtime dependency pin in repo `models/autoports/aleph_alpha_kolibri_1_bf16/requirements.txt` and the QB2 demo requirements. It is a comparison source pin, not a claim that Qwen already depends on it. `tests/scripts/tt_transformers_ref.txt` instead pins `b0d3e9861751848c7e58c56ff64f34abe2fc7de3` and explicitly says it is a CI test input rather than a runtime dependency. Inspected `src/tt_transformers/modules/sampling/sampling_1d.py` and its vocabulary padding contract.

## Contract comparison

| Contract | Common SamplingGenerator / TTSampling | Sampling1D at pinned source |
|---|---|---|
| State ownership | Owns persistent sampling params, seed manager, penalties, logprobs and keyed trace states. Caller explicitly orders reset/remap/seed advance. | Declarative config with lazy persistent masks/offsets/seeds; `k`, `p`, `temp`, optional seed tensors supplied per call. No request-state manager. |
| Public sample | `sample(logits, enable_trace=True, tt_out_tok=..., skip_precompile=..., count_tokens=..., grammar_bitmask=...)` | `decode_forward(logits, k=..., p=..., temp=..., seeds=..., tt_out_tok=..., enable_log_probs=...)`; returns `(tokens, logprobs_or_none)`. |
| Trace ownership | `precompile`, `capture_trace`, keyed nonblocking replay; exact input/output object identity checked on reuse. Explicit request seed currently bypasses internal tracing. | No capture or replay API. Caller must resolve every lazy buffer, warm variants, capture and retain traces, and restore state. |
| Persistent feedback | Passes `tt_out_tok` as `ttnn.sampling(output_tensor=...)` or argmax output. | Same persistent output capability. |
| Greedy | Local top-32 candidates + candidate all-gather + sampling k=1/p=0/temp=1; optional full-vocab gather + argmax. | Same two strategies; force-argmax selected by all k/p/temp None and config allow_force_argmax=True. |
| Top-k/top-p | Per-row persistent tensors, normally max_top_k=32; vocab-sharded logits. | Same candidate strategy with caller-owned tensors. |
| Seeds | Slot-aware explicit request streams, counters and negative-seed hashing; remap/invalidation/reset helpers. Unseeded hardware PRNG cannot be physically remapped and is reset on command. | `manual_seed` consumes persistent/default or caller tensor; no matched per-request counter/reset/remap semantics. Caller must implement these. |
| Penalties | Common TTPenalties applies repetition before presence/frequency; output counts update inside sampling graph. | None in Sampling1D; caller would compose another module and manage all history. |
| Logprobs | Sampled-token logprobs, optional top-K modes and rank payload support. Hardware/mesh limitations still apply. | Top-k path sampled-token logprob only; argmax returns None. Calculator returns None on 1x1/1x2; documented target is T3K 1x8. No top-K alternatives API here. |
| Grammar | Optional persistent grammar mask and separate trace mode. | No grammar API. |
| Mesh/CCL | 1D and configured 2D sampling gather axis, sampling_dp params; supports persistent `line_all_gather` buffer keys. | Intended for 1D, so fits 1x4 physically; 2D defaults use axis 0 for top-k and axis 1 for argmax and offsets use max(mesh dimensions). Do not generalize this to arbitrary 2D layouts. |
| Row padding | Rounds physical sampler max batch to >=32 and a tile multiple; live-row argmax optimization. | Caller config defaults max_batch_size=32 and must provide matching physical param/output tensors. |
| Vocab padding | Real versus padded vocabulary distinct; invalid columns masked before selection; global offsets use padded width. | Same requirement, but config calls padded width `vocab_size` and real width `valid_vocab_size`. |
| Lifecycle | Mode keys include penalties/logprobs/force_argmax/grammar/bucket; reset_trace releases all. Current reset_sampling_params preserves mode traces. | No trace lifecycle; `release` frees lazy buffers/logprob calculator. |

The selected decoder uses true 1x4 TP/EP with the accepted replicated BF16 inter-layer residuals. The terminal norm/LM head must produce contiguous, correctly indexed vocabulary shards for this sampler without changing the decoder's residual policy. A localized terminal gather/reduction may be necessary and must be measured; it must preserve the accepted residual policy.

## Integration risks identified before implementation

1. **Capture scope:** `SamplingGenerator.capture_trace` currently wraps the whole sampling capture in `trace_allocation_tracker.corruptible_allocation_scope`. This conflicts with the requested tracing acceptance contract. Do not inherit that exemption as passing evidence. Use a narrowly scoped generator-owned wrapper around the common sampling device graph, or fix shared capture to acknowledge only exact returned backing allocations after capture. Enumerate aliases, all live trace consumers and replay order. Precompile all modes first, including penalty updates, then capture with no late program allocation. Include tracker positive control and program-cache allocations.
2. **Explicit seeds:** `sample` computes `use_internal_trace = enable_trace and not seed_manager.has_active_request_seed()`. An explicit seed therefore takes the eager path even if the caller requested tracing. Do not count that call as traced sampling. A safe wrapper can trace the common device graph over the stable seed tensor while caller-authoritative seed updates happen outside capture; prove no lazy allocation, stale seed, double advance or host token feedback. If per-token seed advancement is moved on device, preserve the request stream and remap semantics and test them. Greedy steady state must have no host token/position feedback or seed upload loop.
3. **Mode lifetime:** The common README still says force-argmax changes invalidate traces, but current source caches `_TraceKey(penalties_on, log_probs_on, force_argmax, grammar_on, bucket)`. Use the source behavior. Prepare every supported mode before capture and prove greedy -> sampled -> greedy reuses the first trace. Do not call `reset_trace` at request reset or parameter changes.
4. **Semantically greedy tiled candidates:** Keep `max_top_k=32` physical candidates per shard and set k=1, p=0, temp=1. For 1x4 this gathers 128 candidates, not a physical width-1 tensor. Benchmark this against force-argmax under identical logits/feedback/layout. A generic k=32 stochastic run is not a greedy comparison. Both samplers convert logits to BF16 in the top-k path; final accuracy and exact greedy token semantics must be checked against that boundary.
5. **Tie behavior:** Both implementations correct the lowest global ID among gathered equal maxima but document a limitation if more than 32 maxima tie within a shard: unstable local top-k may drop the true lowest ID. All returned tied maxima remain argmax values; exact HF lowest-index agreement needs an explicit tie contract/test, and changing to stable top-k must be measured. Padded vocabulary columns must be masked, including all-negative real logits.
6. **Topology and output contract:** Use a 1x4-compatible CCL implementation and correctly padded vocab offsets. Stable decode token input must have the exact dtype/layout/shape accepted by sampler output and embedding input. Repair that layout boundary if needed; do not add a host cast/argmax/token reconstruction loop.
7. **1x4 logprob compatibility:** Pinned plugin routes all logprob requests on four devices to host logits; it permits device logprobs only on 8/32 devices and top-K device alternatives only for gpt_oss. Maintain explicit compatibility logits readback. This does not justify using host sampling in measured token-out decode.
8. **Request and slots:** Precompile/reset/remap calls may mutate penalty and seed state. Restore request state after warmup and initialize only incoming slots at prefill. Apply slot remaps exactly once even during host compatibility mode, then rebuild invalidated device params/history on the next commanded device transition.

Both common implementations fit a 1D vocab-sharded terminal graph. The richer common implementation minimizes new serving-state code; a generator trace wrapper reusing it is permitted by the full-model contract and is not a new custom sampling algorithm. Do not write a third sampler merely to avoid these trace-state requirements. If terminal sampling dominates token-out time, profile its real graph and fix LM-head/candidate layout before accepting full-model performance.

## Integration evidence

The generator now owns capture around the common device graph, without the
common capture exemption or its explicit-seed eager bypass. Persistent tokens,
positions, parameters and per-slot RNG state are prepared before capture.
`generator_replay_probe/final.json` covers nonaligned/mixed prefills, repeated
and changed page tables, greedy/sampled parameter changes and per-slot seed
isolation. `serving_commands_probe/result.json` covers remap, reset, async
output lifetimes and caller-owned logits. `reduced_positive.json` confirms the
allocation tracker's unsafe-buffer rejection. Full40 teacher gates and the
six-prompt free-running comparison are recorded in `teacher_accuracy.json`,
`batched_teacher/result.json` and `qualitative/tt_qualitative_verdict.json`.
The final performance and reduced component profile are reported in README.md.
The risks above explain design choices; they are not current open findings.
The common sampler's documented >32-equal-maxima local tie limitation remains:
it selects a genuine maximum, but exact HF lowest-index identity is not promised
for that pathological tie set. The tested cross-shard ties pass lowest-ID checks.

## Traced greedy strategy measurement

`tests/run_sampler_contract_probe.py` compares the same real TP4 terminal logits,
persistent 32-row token output, mask/seed feedback operations, and 100 replay
iterations. Tracking and program-cache allocation checking are enabled.
`sampler_contract_probe.json` and `logs/sampler_contract_shape.log` record all
three fixtures: real terminal, all-negative unique maximum, and cross-shard
ties. Every strategy selected the true maximum and lowest tied ID.

| Common strategy | Real terminal ms/replay | Decision |
|---|---:|---|
| Split greedy, physical local top-32, k=1/p=0 | 0.4883 | Selected |
| Split greedy with power-of-two vocabulary padding | 0.5218 | Rejected, slower |
| Full-vocabulary gather plus force-argmax | 2.6647 | Rejected, slower |

The selected sampler is about 2% of the accepted 24.9935 ms layer-stack lower
bound; final token-out profiling must still verify the whole terminal path.
An initial probe-only global/local vocabulary reshape error was repaired before
this comparison; the generator already used the local shard width correctly.
