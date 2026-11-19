# Serving transition repair investigation

Contract: immediate and two-queued async reads must produce identical tokens,
logical inputs, final device state and RNG state for both mixed histories.
Allocation tracking remains enabled with program-cache checks; watcher is an
additional correctness control. No tolerance relaxation, host sampling fallback,
or blocking replay is permitted as a repair.

AutoDebug report: AUTODEBUG.md. Candidate traced copies fail with watcher;
eager-copy control also fails at the same first A515 output. Traced copies
without watcher pass both histories exactly. No copy lifetime/precision or
host snapshot ownership violation was found. Cause remains unresolved.

Priority hypotheses and discriminating checks:
1. Cross-run reset/capture history: immediate-twice and queued-first controls
   preserve all exact oracles while changing only read order. Independent CPU
   reviewer inspects reset coverage. No production change yet.
2. Watcher instrumentation or polling exposes a shared model defect: compare
   watcher unset, 10-second and long polling intervals, preserve each watcher log.
3. First model-state divergence around compressed group128: compare snapshots
   before the first changed output, then localize the first layer/intermediate.
4. Async DMA/submission interaction: only after equal pre-step state is proven,
   separate read issue from its wait and keep sequence-tagged snapshots.

Diagnostics added: tests/probe_state_copy_control.py permits eager transfer
control; tests/probe_vllm_async_transitions.py adds read-order modes. Default
acceptance mode and all exact assertions remain intact. These are diagnostic
changes, not a production fix. Current full-model performance comparison shows
no meaningful gain from copy traces; measured evidence is preserved under
readiness_vllm/stage10_copy_candidate.

Status: investigation active; watcher failure is not waived or declared fixed.

Immediate-twice watcher control also fails, first at B position133 (sequence11),
10659 versus97178. Its second run matches the no-watcher reference; first run
has a different B trajectory. Original watcher controls first differed at A515.
This excludes async overlap and a simple always-second-run failure as necessary
causes. Cross-process trajectories also differ. Commands/results:
`probe_state_copy_control --read-order immediate-twice` with watcher10,
`traced_copy_watcher_immediate_twice.json` and matching process/watcher logs.
No watcher assertion or tracker error; clean device closure, no reset.
Next control keeps instrumentation but moves polling interval to3600 seconds.

The2x2 matrix is complete: eager copies without watcher also pass both histories
with exact tokens, ledger, final state and RNG. Watcher3600 still fails, with
first B token difference at index3 and A at index9. This excludes periodic
10-second host polling; compiled instrumentation/timing remains a hypothesis.
Independent reset audit found all request-state fields covered and complete
prior reads drained. Persistent execution scratch survives by design; no
proven read-before-write yet. Multiplication reset would retain NaNs, so the
next state diagnostic records finiteness rather than changing reset blindly.

`tests/probe_vllm_state_determinism.py` observes immediate-twice through A520/B135.
It compares all live per-layer state, valid feedback/RNG/membership and sampled
logits, preserving the original oracle. Unfinished initial-prefill feedback
and entropy-seeded sampler state are excluded until a first sample makes them
valid; cache state is always compared. Added host reads make this diagnostic
unsuitable for async or performance acceptance. First differing field may be
saved locally; no live device allocation is added.

Watcher10 with ONLY TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1 passes both exact
histories (traced_copy_watcher_without_noc_sanitizer.json). This is a diagnostic
feature isolation, never a production workaround or watcher acceptance.

State-localization v1 (all watcher features enabled) fails: seq0..5 exact;
first unequal state at seq6, B singleton prefill128→129, before joint decode.
Actual layer3(new row128) K/V and raw_history differ; all values finite;
compressed keys equal because128 does not close a group. Earlier layer0/1
conv/recurrent states equal. Logits differ but sampled B token still matches.
See watcher_state_determinism_v1.state.json and saved first-field tensor.
This localizes the observed divergence before sampling/readback and disproves
A515 scorer-boundary as the earliest cause in this run. Earlier layer recurrent
state equality does not prove equal later projection/residual/MLP outputs.
Next diagnostic adds persistent singleton intermediate copies during warmup
and capture, compares them only at singleton-prefill boundaries.

Singleton intermediate v1 first history passes, but second history exposes
corruption at A final prefill512→513. First unequal intermediate is layer1 MLP
block rank2 only:64 adjacent features1216..1279 (tiles38/39), including values
up to4.25e37 where the other execution is about0.01. Layer1 inputs, attention
residual, MLP mixed input and injection match. Other ranks' MLP blocks match.
Downstream layer3 inputs differ first onrank2, then collective mixing spreads
it. This is concrete corruption, not a tolerance issue. See
watcher_singleton_intermediates_v1.state.json and first_difference.pt.

A standalone watcher-enabled ccl.sum→BF16cast→persistentcopy traced stress
passes1000iterations on rank-distinct BF16 inputs, every rank exact.
Command: tests.probe_exact_rank_sum_stress --iterations1000; artifact
exact_rank_sum_stress_v1.json. This excludes neither interleaved-kernel state
nor model-context lifetime/collective races. No production repair yet.

Native reduction counterfactual: monkeypatch only multichip_decoder.exact_rank_sum
to the original accurate FP32 TTNN H-sum; retain fullwatcher, copytraces, async
oracle,precisionconfig andbothhistories. native_rank_sum_watcher_v1.json passes
all exacttokens/ledger/finalstate/RNG. This supports custom-path involvement,
but the replacement also changes timing/allocations, so it is not yet a root
cause. Repeating immediate-twice; custom-path pregather/gather/FP32 taps pending.
A second small stress with LoFi BFP4 matmul immediately before gather/sum passes
1000iterations, allranks identical andstable (exact_rank_sum_matmul_stress_v1).

The native FP32 control also passes immediate-twice under full watcher
(native_rank_sum_watcher_immediate_twice.json), both complete histories.
Independent LLK audit found no confirmed compute-API violation. Candidate
discriminators are hoisting the repeated copy initializer and, separately,
full DST synchronization; neither is a repair until exact hardware checks pass.
Collective-tap v1 did not execute hardware: first invocation lacked the required
program-cache tracking environment; a retry refused its existing artifact.
The correctly configured fresh v2 run is the relevant experiment.

Collective-tap v2 finds ONLY the saved gathered BF16 tensor unequal at seq6,
rank3, remote rank slice1,64 features1216..1279. Inputs and subsequent FP32
reducer/BF16 result/residual are exact; original token/state oracle passes.
Thus reducer arithmetic is not established as the cause. The tap may observe
incomplete gather or copy corruption. Alternating-input standalone gather tap
plus custom reducer passes1000 iterations (alternating_gather_stress_v1.json).
Independent source audit identified a candidate completion-route mismatch in
multicast all_gather: even-ring payloads alternate routes, completion credits
follow only the original route. The observed antipodal rank1→rank3 mismatch
is consistent with an alternate-route packet arriving after completion.
A diagnostic disables alternate payload routing without removing watcher.

Matching-route diagnostic passes fullwatcher original immediate/two-queued
async oracle, both histories, exact tokens/ledger/finalstate/RNG
(matched_route_watcher_async_v1.json). It changes only the two multicast
kernels to keep payload routing equal to the existing completion routing.
No host/plugin changes or synchronization added. This supports the identified
completion-route defect. Production repair will preserve alternating routes
and acknowledge both route sets, leaving startup barrier unchanged.

Production candidate changes only all_gather multicast reader/writer kernels.
Payload alternation is retained; done credits cover original and alternate
route sets, with 2*(N-1) receive count when alternating. Header storage is
reused only after flushing its first local send; no additional header pool
allocation. Startup semantics currently unchanged. JIT and1000 alternating
input/gather-tap/custom-reducer iterations pass fullwatcher
(dual_route_gather_stress_v1.json). Integrated async and tap checks pending.

Epoch audit: model calls omit persistent_output_tensor, so each next launch
waits for all ranks at the separate startup barrier. A rank still waiting for
current done credits cannot contribute next startup, preventing premature
next done credits. The generic persistent_output_tensor branch suppresses
startup and has a pre-existing cross-epoch credit-count hazard. This model
does not exercise it; broader core repair scope is being assessed explicitly.

Final core repair also makes multicast startup unconditional in the host
factory, protecting preallocated-output completion epochs. Incremental build
and runtime/Python installation passed; see collective_repair_build.json.
The correct Python API keyword is output_tensor; persistent-stress v1 was
a setup TypeError, v2 passes1000 alternating-input replays under fullwatcher.
Ordinary dual-route async and intermediate-tap checks both pass. All gathered
BF16, reducer FP32, BF16 outputs and residual taps compare exactly in both
histories through singleton-prefill boundaries. Final combined-source async
validation and full-model timing are running.

Root cause: multicast all_gather completed before all alternate-route payloads
arrived on even rings. The fast custom reducer exposed the race; replacing it
with a slower native reduction masked it. The reducer, adapter, datatype policy
and async boundary require no repair. Watcher sanitization is retained.

Final combined-source validation completed: `final_candidate_async_watcher_v1.json`
passes both strict histories with full watcher and allocation tracking (80.30 s),
including tokens, slot ledger, persistent feedback and final device/RNG state.
`full_model_comparison_v2.json` passes all 48 layers, B1, C262144, 128 input /
128 generated tokens; resident traced decode is 102.191080 ms / 9.785590 t/s/u.
The v1 full-model attempt failed before hardware setup because the vLLM
environment lacks the standalone qwen4_exp AutoConfig registration; the native
python_env rerun fixes environment selection, with no model-code workaround.
Local repair commit: `572c94156bc4e594f0d2159a8e21808b6ea34ad7`; combined
unused-prompt-history optimization commit: `be261ee9a013f3af35dd932512d355274286c840`.
AutoFix succeeded; final full-serving acceptance remains separate work.
