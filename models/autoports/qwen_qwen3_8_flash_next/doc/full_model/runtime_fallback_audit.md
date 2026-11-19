# Runtime fallback audit — current production revision

Final closure: [independent stage review round1](stage_review_round1.md) returned
**clean-pass**, with no required work or further hardware runs. The audit below
retains its pre-review provenance; pending-review statements are superseded by
that verdict. Local checkpoint records are in [commits.json](commits.json).

Independent CPU-only refresh, 2026-11-14, including current watcher, full48 B1
accuracy/resume and clean performance, full48 B2/B32 integration, and reduced
seed-state/current sampled-geometry evidence. No device work by this reviewer,
production edits or gate relaxation. Four generator-equivalent serving rows
were reconciled separately after full48 B32v2 passed. The earlier audit is preserved below as a
**historical snapshot**; its v4/v6 evidence and pending-work statements describe
those revisions. This refresh supersedes its current-source status assertions.

Source inspection found no single-chip, replicated-model, host-model, eager
request-time decode, or less-optimized cache fallback. This is a scoped source
finding corroborated by current trace/lifecycle evidence, not an assertion that
independent stage review has passed. Current full48 B1 accuracy/resume and clean token-out
performance pass. Full48 B32v2 passes accuracy, state and trace gates, and final
reduced C262144 terminal attribution is complete. Once-only maximum-context
execution also passes; acceptance gates are complete pending fresh independent
review and local commit. Historical cross-active-width numerical failures remain FAILED,
and the current forced43 PCC failure has an owner-controlled limitation
disposition that remains subject to independent review.
No pending run is counted as a pass here.

## Current runtime and host boundaries

| Surface and exact source | Audited behavior | Current evidence and limit |
| --- | --- | --- |
| `tt/model.py::QwenModel.__init__/embed/forward_chunk/decode_batch/logits` | Constructor rejects meshes other than 1x4. Torch/Safetensors weight conversion is setup. Runtime embeddings, all decoder layers, final hyperconnection mixer and vocabulary-sharded DRAM LM head use device operations. No `.cpu()`, `to_torch`, host HF forward, exception-selected alternate model, or tensor-parallel reduction to one chip exists in these forwards. | Current `acceptance_accuracy_resume_final_v1.json` validates full48 B1: prefill97/100/100 and traced teacher96/100/100 top1/5/100, with resumed-prefix gates passed. v7/watcher_v2 and full48 B2 cover the selected serving source. Earlier `full_accuracy_v2.json` remains historical evidence. |
| `tt/multichip_decoder.py::MultichipDecoder`, `tt/model.py::FinalMixer` | TP4 attention, EP4 native experts, selected projections and residual representation remain inherited. Rank r retains one complete 2560-feature hyperconnection stream between layers. Embedding/terminal gathers and selected collectives do not create four independent full-model replicas. The selected BF16 paged K/V, BF16 indexer and FP32 recurrence are retained. | No branch switches to single-chip, replicated, host execution, or BFP8 paged-cache candidate. Numerical and policy evidence remains separately scoped; this audit does not select a new dtype/fidelity policy. |
| `tt/generator.py::_capture/_replay_decode`, `tt/factored_decode.py` | Configured B1 uses one whole-model decode trace. Configured B>1 uses a bounded set: one clear trace, per-layer/per-slot attention traces, per-layer/power-of-two MoE traces and per-slot terminal traces. Model attention/cache work executes only active real slots. Device gather repeats the last valid MoE row into physical padding and scatters valid results only. Singleton uses inherited FP32 reduction; multi-row buckets use inherited native BF16. | `_capture` explicitly rejects B>1. v7 exercises 15 model segments plus six prefill/one sampling trace; full48 B2 retains 195 model segments plus six prefill/one sampling trace. No per-active-subset capture family, eviction or eager fallback. Reduced B32 corrected-seed diagnostics pass the larger fixed-slot schedule and active-width transitions. Full48 B32v1 captured the bounded set but failed final state under mismatched seed histories; corrected full48 B32v2 passes HF/logits/tokens/final-state and bounded-trace gates with1,857 model segments plus six prefill/one sampling trace. |
| `FactoredDecode.set_slots/replay` and `tt/indexed_rows.py` | Persistent row buffers and UINT32 membership metadata precede capture. On unchanged sorted membership, `set_slots` returns without a device upload and `replay` reuses the same cached tuple of trace IDs. It enqueues those retained segments; it does not reconstruct the model graph, gather activations on the host, or rebuild the dispatch plan each token. Membership changes rebuild the small host dispatch tuple and upload indexed-row metadata once. | This is not a claim of zero Python work: the retained segment loop still dispatches from Python. All fixed shapes are warmed before capture. Current reduced/B2 tests and reduced B32 corrected-seed control retain IDs/program counts and verify metadata uploads/cached-plan reuse. Full48 B32v2 independently passes:137 unchanged-membership replays,136 cached-plan identity reuses,204 native-ID checks and207 program-cache checks;65 request metadata uploads correspond to membership transitions. Its final-state gate passes independently of allocation evidence. |
| `tt/generator.py::_sample`, `models/common/sampling/generator.py`, `models/common/sampling/tt_sampling.py` | The model writes persistent BF16 TILE vocabulary-sharded `[1,1,32,62080]` logits. A separate generator-owned trace runs the selected common sampler with `tt_out_tok=self.decode_tokens`, then restores inactive tokens and seed sentinel. Keys contain semantic sampler flags rather than active subsets. No host argmax or full-logit read is in `_sample`. | Normal split greedy uses logical k=1/p=0/temp=1 with physical local k=32; public temperature zero is normalized by the common formatter. Selected `allow_force_argmax=False`; optional force-argmax is device-only and rejected for production by measured cost, not used as a fallback. `sampling_contract_audit.md` records both common sampler comparison and greedy unique/tied-max evidence. Current full48 clean token-out performance passes at7.206985 t/s/u. Completed reduced eager C262144 attribution measures split0.429263ms versus force2.886563ms merged kernel sums, with both returning the same actual greedy token; split is3.673% of model-plus-split kernel sum. This is not full48 trace/component timing. |
| `tt/token_feedback.py::DeviceTokenFeedback`, `tt/device_seed.py::DeviceSeedState` | Device dataflow selects current versus saved UINT32 tokens using the persistent active mask. It changes no distribution or logits. Seed progression executes in the sampling trace. Active mask uploads occur only on membership changes; remap invalidates membership so the next transition refreshes it. | v7 verifies dormant-token preservation and resident feedback with zero steady token/position/page-table refresh counters. Full48 B2 verifies exact token feedback/order. Reduced B32 corrected-seed control passes resident sampled steps and changed active membership; full48 B32v1 resident feedback passes independently of its failed final-state gate. Corrected full48 B32v2 passes four resident sampled steps, including changed active membership, coherent positions on all12 QSA layers/all fixed slots on rank0, unchanged inactive feedback and zero token/position/page refreshes. Its position read scope is rank0; the separate B1 clean run covers all4 ranks. |
| `tt/generator.py::generate` | Default loop dispatches model then sampling, reads sampled IDs for returned output, and uses the persistent device token for the next model step. The Python integer is not copied back. Request initialization and reset occur before the loop. | Output readback synchronizes because this high-level API returns tokens; that cost belongs in token-out performance. `next_input` and `host_sampling=True` are explicit separate branches that intentionally upload feedback. No readback/writeback loop exists in the default path. Current clean performance records127 decode steps with zero steady input/position/page refreshes, resets, captures or releases and coherent positions on all12 QSA layers/all4 ranks. |
| `tt/generator.py::_table_plan/_bind/decode_forward` | Authoritative reload validates submitted cache/slots/positions/tables before mutation. `reload_inputs=False` ignores stale host token/position values except the active-row sign; device tokens/positions remain resident. `_bind` compares mapping values before upload. Within an unchanged page, `_table_plan` returns before constructing full shadow tables; a newly live page or explicit mapping/remap transition validates retained host ownership, including dormant slots. | v7 proves changed mapping writes the new physical page while the old page remains unchanged, plus unchanged-table replay. Page-boundary host ownership validation is not a per-token device page rebuild and does not upload unchanged tables. This audit does not claim the entire scheduling method is host-free. |
| `tt/generator.py::_read_logits/read_decode_output/process_decode_output_host/prefill_forward` | Full logits cross to host only through requested prefill/accuracy outputs, explicit raw-logit decode (`sample_on_device=False`), diagnostic reads, or explicit host compatibility. Async read snapshots the submitted tensor to HOST; host processing formats that snapshot without sampling or advancing state. The low-level method's default is raw logits, so a caller must explicitly select device sampling for token-out. | v7 validates reverse-order delayed token/logit snapshots, correct physical-to-submitted row formatting and independent CPU results. Default high-level generation selects `_sample` and never reads full logits. Raw-logit calls are compatibility/diagnostic surfaces, not evidence of measured on-device token-out. |
| `tt/generator.py::set_sampling_params/generate`, `tt/host_sampling.py` | Device parameter validation rejects unbounded/top-k>32 stochastic sampling, penalties and logprobs rather than silently clamp the requested distribution or switch modes. `host_sampling=True` explicitly invokes `HostSampler`, full vocabulary readback and token upload. The pinned plugin may own host processors via the explicit raw-logit API. | v7 proves three-step host sampling, unchanged device RNG and return to the traced device route. Compatibility sampling and teacher-forcing reference uploads must remain excluded from on-device token-out measurements. |

Common split greedy is semantically greedy: k=1 chooses a maximal logit;
physical k=32 exists for the tiled candidate contract, not stochastic top-32
sampling. `sampler_ccl_smoke_v1.json` covers unique maxima and the recorded
65-way ties on all four output ranks. No new custom probability sampler was
introduced: `DeviceSeedState` and `DeviceTokenFeedback` implement request seed
state and inactive-token ownership around the selected common sampler. The
rejected `Sampling1D` contract and force-argmax measurements remain in
`sampling_contract_audit.md`; reduced timing is not full48 throughput.

## Cache, request transitions and lifecycle

`QwenGenerator` distinguishes owned private caches, constructor-borrowed private
caches, and setup-bound opaque `SharedKVPool`. Allocation validates per-layer
shape/dtype/tensor_idx and requires empty slots before freeing owned private
K/V. It rejects replacement of externally bound/borrowed storage. Once mutation
starts, an allocation failure poisons the instance; there is no recovery to a
private/host/smaller cache. Deferred state cannot execute until bound.
`SharedKVPool.freeze()` prevents rebinding after preparation; each bound cache
retains the pool owner. Shared K/V aliases are preserved rather than duplicated.

`SlotStateMover` moves private recurrence, convolutions, PLE/indexer state and
page tables through fixed-address scratch, never shared K/V. Token and 64-bit
seed state migrate separately on explicit scheduler remap. Canonical B>1
prefill uses scratch with the current request's copied private state and page
tables, borrowing that request's pool pages without independent concurrent
ownership. Its six dyadic trace keys are independent of scheduler slot. Public
prefill splits logical spans at absolute boundaries, masks/slices physical
padding and copies each logical last row to its sampler slot before workspace
reuse; no public alignment restriction is introduced.

`PrefillCheckpoints` retains compact device snapshots for eligible odd-length
continuations. Restoration replays the prior tail with the new suffix using the
selected native path, then slices out the old row. It does not move recurrence
or logits to CPU. Decode consumption, request reset and close invalidate only
appropriate checkpoint metadata; remap permutes snapshot owners outside traces.
The fixed per-slot checkpoint buffers and scratch are included in capacity
planning and are not assumed free.

`_reset_cache` clears private recurrence/conv/PLE/indexer/position, and private
K/V where applicable. Shared K/V bytes and scheduler mappings are retained;
the host mapping shadow is invalidated so a fresh request must supply an
authoritative mapping. `release_request(slot)` resets only that slot's private
model/checkpoint and sampling seed/output state. Neither reset nor release
retires model/sampling/prefill traces or closes the mesh. `close()` invalidates
checkpoints, releases retained captures once, clears trace dictionaries and
invalidates the cached factored dispatch plan. It leaves the caller-owned mesh
open. Explicit subsequent reuse can recapture already-warmed variants; this is
an unload boundary, not request-time eviction. Failed capture ends/releases its
partial command stream without replay, then propagates the original error.

Current `serving_probe_v7.json` passes all 16 reduced checks with tracking and
program-cache allocations enabled, 2,180 stable programs, 22 retained captures,
zero request-time releases and clean caller mesh close (49.394 seconds is
instrumented test wall time). It covers reset/reuse, active/dormant migration,
seeded heterogeneous parameters, failed capture cleanup, immutable readback,
explicit host mode, and a tracker positive control that rejects an unsafe live
allocation before a final clean replay. Its all-inactive call returns sentinels
without model or sampler work. The separately executed `serving_watcher_v2.json`
also passes all16 checks on the same production generator, with2,180 stable
program entries,22 captures, clean caller mesh close and72.266s instrumented
wall time. It corroborates the current segmented/workspace path; the earlier
watcher_v1 remains historical.

Current `prefill_factored_full48_v1.json` passes full48 B2 shared-pool mixed and
reversed logits, resumed 68→257 and 31→265 exactly, and matched-seed decode order
with exact logits/tokens/private and logical KV state/feedback. It has 2,182
stable programs, 202 retained captures and zero request-time releases. These
facts establish current all-layer B2 integration. Full48 B1 HF accuracy and
clean latency are established separately below. This B2 result does not prove
full48 B32, maximum-context execution or universal cross-active-width equality.

`acceptance_accuracy_resume_final_v1.json` passes current full48 B1 at
configured context262144: prefill top1/5/100 is97/100/100% over100 predictions;
traced teacher forcing is96/100/100% over100 predictions. The refreshed
`readiness/prefill_resume_check.json` passes starts32/68/256 at logical length257,
top1 agreement1 and minimum PCC0.9999656772; both missing-history controls reject
for the longer suffixes. Full configured allocation is not max-context execution.

`acceptance_perf_final_v1.json` passes clean current full48 B1 with tracking,
watcher and profiler disabled. Public prompt128/generate128 TTFT is7038.672ms;
127 token-out decode steps average138.754269ms or7.206985 t/s/u. Retained trace
IDs and all12 QSA positions on all4 ranks are verified; steady decode has zero
token/position/page refreshes, resets, captures and releases. Caller token reads
are included. Its separate repeated-fixed-token model-only window is7.238372
t/s/u, explicitly **not teacher forcing**. The accuracy run's teacher-forcing
decode6.716107 t/s/u uses reference uploads, a different prompt and tracking;
it is separately labeled and is not a clean sampler cost comparison.

`batch32_hf_factored_v1.json` remains FAILED: both2704-prediction runs pass
97.818/100/100% HF top1/5/100, with exact same-history logits/tokens, dormant
state and resident feedback, but all32 final combined state hashes differed.
The harness supplied scalar seed0, which seeds only lane0; other requests had
fresh unseeded entropy and migration reassigned the seeded identity. The
production scalar-seed contract was preserved. Actual setter/formatter/decode
CPU tests verify the correction to explicit per-request seed lists.

`batch32_seed_state_reduced_v1.json` now passes that corrected control on real
layers0/1/3, configured B32/C512, in137.131738s (process exit0). Matched-history
logits/tokens/final state are exact; per-field mismatch maps are empty, dormant
and all-inactive checks pass, and resident sampling feedback passes. It explicitly
has `acceptance_eligible=false` and `hf_accuracy_evaluated=false`.

Corrected full48 `batch32_hf_factored_v2.json` now independently passes in
1296.826s (exit0), `acceptance_eligible=true`. Baseline and migration each score
2645/2704 top1 and2704/2704 top5/top100; the per-phase prefill and decode groups
also pass unchanged HF bars. Same-history logits/tokens/final request state are
exact, and final/dormant per-field mismatch maps are empty. All-inactive noop,
resident feedback, stable pool addresses and bounded trace gates pass. It retains
1,864 total captures with zero releases and2,179 unchanged programs. These are
correlated histories from one AIME24 reference, not2,704 independent prompts,
and the instrumented wall time is not serving throughput. All three allocator
snapshots match v1 exactly. Passingv2 does not rewrite failedv1 or claim a pass
for the separate raw cross-active-width PCC comparator.

`current_geometry_sampler_v1.json` exits0 with188/188 diagnostic checks passed,
62 actual device-sampled predictions semantically greedy and in matched HF
top5/top100, exact repeated/migrated outputs, coherent feedback/positions and
2,179 unchanged programs. The62 rows include exact repeats of31 overlapping
history/position rows, not62 independent examples. Survivor44 actually samples
81760, the unique TT maximum and a matched HF strict-rank4 token. CPU tensor
rescoring confirms the direct metrics. Full-logit diagnostic reads do not feed
the next input. Current forced43 still FAILS the separately recorded frozen
TT-self PCC comparison (whole0.990041606/minimum0.988163811), although its
TT-self top1/5/100 are now all1. Current42 and survivor44 pass that comparator.

`final_capacity_reconciliation.md` distinguishes measured allocation from
calculated plugin-expanded pools; it does not turn short-context execution
into maximum-context proof. Actual maximum-context evidence is now separately
recorded in `acceptance_max_context_final_v1.json` and `max_context_attempt.json`:
nonaligned262143-token public prefill, one decode and two sampled outputs consume
262144 context, with8196 prefill/1 model/2 sampling replays. All12 QSA layers on
all4 ranks reach262144 and final active logits are finite on all4. The JSONs pass
and the log contains phase/driver completion and clean device close. The outer
monitor was interrupted; no numeric process exit is claimed. Final post-finiteness
allocated/free/largest bytes per bank are4099126528/173214848/165250368,8192
bytes more allocated than the earlier memory_after snapshot.

`serving_contract.json` records42 supported generator equivalents and21
unsupported declarations, none missing. The capacity row links actual execution
and calculated8194/545-block worker pool budgets, not unobserved allocations.
The owner geometry disposition remains linked; no adapter/server test is claimed.

## Completed terminal attribution and rank scope

`terminal_final_analysis.md/json` and `performance_boundary_audit.md` close the
current reduced C262144 operation-profile work: layers0/1/3, B1, one warmed
**eager** iteration, no live traces and a very short live history. Independently
recomputed coverage matches21488 host/device/final operations and six signposts,
with positive preserved durations and equal region counts on all four ranks.
The measured head retains BF16/HiFi4/FP32 DEST and accurate math; no head or
sampler advice remains. Split versus force greedy is0.429263 versus2.886563ms
merged kernel sums, both actual token243975. This supports the existing sampler
selection and does not show sampler dominance.

Embedding/gather0.036766ms, final mixer0.372330ms, head/layout0.668237ms and row
assembly0.045470ms are named reduced eager kernel costs. Existing full-capacity
QSA scoring is2.387150ms; causal selection remains0.010177ms on this short
history. These are neither full48 critical-path measurements nor maximum-position
costs, and cannot be multiplied into advertised throughput. Some BinaryNg shape
annotations are inconsistent; the analysis preserves this limitation and makes
no FLOP/bandwidth claims from them. Clean full48 traced138.754269ms token-out
remains the authoritative measured workload boundary.

`raw_artifacts.json` records the final raw `terminal_final_tracy_v1/` directory
retained uncompressed in persistent workspace storage. Final coverage/reports
and their hashes are in `terminal_final_perf_v1/` and the analysis JSON; failed
historical captures remain retained. No profiler run was performed by this audit.

Supported serving rows were checked against their actual scopes. B32v2's resident
position diagnostic reads every QSA layer and fixed slot on **rank0**; it does
not establish B32 position telemetry on all four ranks. The separate clean B1
run validates all12 QSA positions on all4 ranks. The serving ledger's B32 rows
claim scoped resident feedback/coherence, not all-rank B32 observation; no
unsupported all-rank claim or serving-status change was found or introduced.
The capacity API row now has actual maximum-context evidence with separately
calculated worker expansion. Fresh independent review and local commit remain;
no plugin adapter/server execution is implied.

## Remaining findings and provenance

No additional executable fallback defect was identified in this refresh.
Terminology matters: historical `shared_pool_generator_v5` compares one-active
n1 and two-active n2 calls **inside a configured-B2 TT generator**, plus a
B2-history singleton survivor. It is not a separately configured-B1 versus
configured-B2 wrapper comparison. Original HF controls do use actual B1/B2
tensor geometries. Neither historical nor current reduced geometry probes
establish equivalence between separately configured TT wrappers.

Historical forced43 PCC/top5 and survivor PCC failures remain raw FAILED
artifacts. The current true sampler/HF checks and same-history controls pass,
but the current forced43 TT-self PCC diagnostic still fails. The proposal in
`batch_geometry_disposition_proposal.md` distinguishes inherited active-width
rounding sensitivity from required same-history batch-position reproducibility,
state ownership and direct HF accuracy. The owner disposition is now recorded
in `batch_geometry_disposition.md` and the scoped entry in
`../eval_readiness/limitations.json`: current forced43 TT and matched original
HF both fail the frozen PCC gate, while current TT top1/top5 agree across the
tested active-width histories and direct HF accuracy bars pass. It retains the
inherited precision policy and raw FAILED artifacts, and limits reproducibility
claims to identical histories and batch-position relocation. Independent stage
review must assess this disposition. It is not a blanket raw-gate pass; this
fallback audit supplies no waiver, stage pass, precision change or new numerical
threshold.

The diagnostic argmax/topk tie-policy inconsistency is now repaired with
consistent direct argmax scoring. `shared_pool_hf_control_v3_rescore.json`
reproduces all144 raw scores and records only the permitted top1/derived-verdict
changes (55 raw failures,40 corrected failures), preserving old artifacts and
thresholds. Independent direct TT-to-HF checks and the actual sampler probe
close the former ranked-ID-versus-sampled-token evidence gap. They do not make
the current frozen PCC comparison pass. This scorer correction is test code,
not a production sampling change.

The prior dispatch/native-kernel audit below retains its scope: optional TTNN
golden comparison executes host diagnostics, the legacy
`throw_exception_on_fallback` field alone is not enforcement evidence, the
model-selected native MoE stochastic-rounding flag is not a Python expert
fallback, and the Blackhole floating-reduction repair has broader native
scope. No additional native change was made in this refresh. Current watcher_v2
execution is separately recorded above, not inferred from v7. The production
source hashes below remain unchanged; the cache-pool hash includes the already
reviewed docstring-only scratch-ownership correction, with unchanged executable
AST. Final reduced terminal attribution is complete with the boundaries below.
Maximum-context execution passes. Fresh independent stage-review, including
the owner's geometry disposition, and local commit remain separate work.

Current-source SHA256 manifest (repo-relative paths):

```json
{
  "models/autoports/qwen_qwen3_8_flash_next/tt/model.py": "bc46bb7e0428947cb29d7fc312b3fe6c3fd147eb88c4e6fa847f13755c10352a",
  "models/autoports/qwen_qwen3_8_flash_next/tt/generator.py": "f539343185c824cbc73b820c241b03d9d953fc73e0808fa421be0e21db4abc63",
  "models/autoports/qwen_qwen3_8_flash_next/tt/factored_decode.py": "2337245025e8d94a7b31f3585f4dbe034802af181208a2387397917fcb7c8312",
  "models/autoports/qwen_qwen3_8_flash_next/tt/indexed_rows.py": "c0d4a3e458918fa83c732f30df70e2b3079b83f014fd81afa8357040457b44a9",
  "models/autoports/qwen_qwen3_8_flash_next/tt/token_feedback.py": "e83579a7aba1b9a779c3ceba5652a9a3b9818fea171bcb87e403f542bb7525d1",
  "models/autoports/qwen_qwen3_8_flash_next/tt/device_seed.py": "b410a3b251164fa1a17e49496dc3b59e889cb25cd0621f9fe0147fc2a3c475d0",
  "models/autoports/qwen_qwen3_8_flash_next/tt/cache_pool.py": "3ef966dbfe58404b77586c8e8268e8f53f50a5601356aa80d58ee8b17c0384c6",
  "models/autoports/qwen_qwen3_8_flash_next/tt/slot_state.py": "e8cf8296e75ab25ee4049677dc5fc29cb97e282c7056635ede6e080f97b17274",
  "models/autoports/qwen_qwen3_8_flash_next/tt/prefill_checkpoint.py": "1642a2750d8cc34b8f15bdda3b7f5701ba02b309d27bca03a6328d7e0818a11d",
  "models/autoports/qwen_qwen3_8_flash_next/tt/host_sampling.py": "26a4a764599c784155b0125025fc2aa6c27635e8eaaf35e48d4e48a949bc4db8",
  "models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py": "67be6f8c0082a39d9848cdb8b7c3af44b5af543c9559a20ad2a6c2e1a1f76359",
  "models/common/sampling/generator.py": "6b3ff7cd106b64d20d4726622393821beb2bbf0c56bfd4cacbca9468d3f5b83d",
  "models/common/sampling/tt_sampling.py": "7ae3725caa32d8143963117b42035101608f46e6954d17cb0a8c33a9dcb67afb"
}
```

---

# Historical snapshot: audit before production factored v7 validation

The following text is preserved for provenance. “Current”, “latest” and
“pending” below refer to that earlier audit revision, not this refresh.


Source inspection covers `tt/model.py`, `tt/generator.py`,
`tt/multichip_decoder.py`, `tt/device_seed.py`, `tt/slot_state.py`,
`tt/cache_pool.py`, `tt/host_sampling.py`, `tt/prefill_checkpoint.py`,
`tt/token_feedback.py`, `tt/factored_decode.py`, `tt/indexed_rows.py`, common sampling, TTNN dispatch, and the native
MoE/reduction changes described below. This audit distinguishes current source
from earlier device evidence. `serving_probe_v6.json/log` passes16 reduced
tracked checks after checkpoint, token-feedback selector and dedicated-prefill
workspace integration, with2177 program entries unchanged and10 traces released
once. It predates the bounded factored production decode integration, whose
hardware validation remains pending. `prefill_checkpoint_full48_v1.json/log` passes all48 shared-pool B2 resume
31/68, mixed/reversed logits and matched submission order exactly, with2182
programs unchanged; its source hash f7c458a3 predates the workspace edit. Full48
workspace validation, refreshed HF accuracy, all48 B32/shared geometry,
qualitative, clean performance and max-context execution remain incomplete.

Current-source changes and limits:

- Compact checkpoints preserve per-request recurrent/history state before a
  final singleton at an even position. Continuation may restore and replay that
  last prefix token with new rows using the existing native multi-row path.
  No dtype/fidelity change or host model computation is introduced. The old
  prefix row is sliced out of returned continuation logits.
- `DeviceTokenFeedback` is an exact UINT32 dataflow select into the existing
  token buffer. The persistent active mask also drives seed progression;
  sampling keys contain semantic flags, not active subsets. Component tracked
  proof covers14 cases/all4 ranks; servingv6 now passes the reduced integrated
  request lifecycle, including dormant feedback and exact changed-mask reuse. No probability or sampled-token math is
  implemented by this helper.
- Current prefill uses one dedicated persistent output, separate from the
  sampler canvas. `_write_prefill_last` copies each final row into its physical
  sampler slot, then drops the temporary slice before another replay. This
  removes per-slot full-vocabulary workspaces. Servingv6 passes reduced tracked
  mixed-row/sampled-prefill/async/output-lifetime validation. All48 workspace
  confirmation remains required; the pre-workspace checkpoint pass cannot
  establish that proof.
- Current source reserves `_capture` for standalone B1. Configured batches
  above one use `FactoredDecode`: fixed per-slot attention traces, power-of-two
  MoE bucket traces, and fixed per-slot terminal traces. Runtime slot metadata
  changes on membership transitions; replay plans retain the same device
  buffers and traces. Bucket padding repeats a valid MoE activation row and
  discards padded outputs; it never adds dummy attention/cache rows. Singleton
  MoE keeps the accepted FP32 path, other buckets retain native BF16. No eager
  fallback, per-subset trace eviction or replicated model is introduced.
  Reduced candidate evidence does not establish production all48 integration
  or memory fit. Current production workspace ownership, all supported active
  widths, singleton behavior, fixed-slot isolation, retained trace lifetime
  and clean latency remain hardware gates. `memory_forecast_current.md`
  records the bank-normalized constraints.

| Boundary | Implemented behavior | Evidence / remaining proof |
| --- | --- | --- |
| Weight loading | Safetensors/Torch conversion is constructor work. Real layer weights remain TP4/EP4; terminal embedding/head are sharded. | Earlier full accuracyv2 records measured full262144 allocation; new head/checkpoints/workspaces require fresh measurement; full-context execution remains pending. |
| Model forward | Embedding, decoder stack, final mixer and LM head call TTNN operations. No runtime Torch model, `.cpu()`, `.to_torch()`, catch-and-retry alternate model, or single-chip branch. | Earlier all48 tracked HF accuracy/resume passes in full_accuracy_v2; current wrapper/checkpoint revision needs refresh; all48 B32 and token-out acceptance remain. |
| Greedy sampling | Model trace fills persistent vocabulary-sharded logits. Separate common sampling trace writes persistent `decode_tokens` via `tt_out_tok`. No host argmax in this path. | Common split greedy selected over force-argmax by reduced equality/timing/profile evidence in sampling_contract_audit.md; final full-model performance remains. |
| Token feedback | Default generation reads tokens for caller output, but next decode reads the same persistent device tensor. B1 whole-model and B>1 factored device traces consume its physical slots. The returned Python token never reconstructs default decode input. | servingv6 passes resident feedback with the new selection helper before factored integration; steady token/position/page refresh deltas are all0. Current factored production replay remains to be validated. |
| Positions | QSA device position advances in decoder; host cache.position is scheduler metadata. Request transitions may explicitly upload an authoritative position. | servingv4 verifies stale-host positions, independent reloads and coherent device positions on all4 ranks. |
| Page tables | Scheduler transitions update changed tables only. `_table_plan` validates all live/dormant ownership; unchanged within-page replay skips reconstruction. A new page boundary may validate host metadata without uploading unchanged device tables. | servingv4 proves changed/unchanged page writes. Poolv5 proves exact reduced same-geometry lifecycle and stable addresses; its cross-geometry numerical gate fails; all48 B2 pre-workspace checkpoint revision passes resume31/68 in prefill_checkpoint_full48_v1; current workspace/all48 B32 remain unproven. |
| Compatibility logits | Prefill checks and explicit `sample_on_device=False` expose full logits. `read_decode_output` snapshots to HOST; `process_decode_output_host` formats only that snapshot. | servingv4 passes reverse-order immutable token/logit snapshots and writable independent full-vocabulary CPU results. |
| Compatibility sampling | Explicit `host_sampling=True` uses HostSampler; unsupported device features raise rather than silently selecting it. This mode reads full logits and uploads selected tokens intentionally. | servingv4 passes explicit three-token host mode, independent helper match, unchanged device RNG and restored traced device route; CPU helper tests cover individual features. |
| Teacher forcing | Readiness `next_input` intentionally uploads the supplied reference token. Predicted output still traverses the common device sampler. This is distinguished from free token-out performance. | Earlier-revision all48 prefill/teacher accuracy97/100/100 passes; current integrated accuracy and final performance labels remain. |
| Reset/reuse | Private recurrence, convolution, PLE history and QSA indexer state reset at request boundaries. Shared KV bytes remain pool-owned; private states and page tables move at fixed addresses. Request seeds/output state reset independently. Traces remain until explicit unload/close. | servingv4 proves seeded release/reuse, active/dormant migration and trace retention; poolv5 adds exact reduced shared-state lifecycle. |
| Failure behavior | Invalid metadata is rejected before reset/upload. No exception selects a slower model or host sampling. Failed captures close/release without replaying partial commands. | servingv4 closes/releases failed capture and successfully replays subsequent prefill/decode. |

## Dispatch and host boundaries

`ttnn/ttnn/decorators.py::FastOperation.__call__` invokes its registered function
and rethrows failures; it does not catch a device failure and invoke Torch.
`Operation` likewise rethrows execution failures. Its optional comparison mode
can execute a host golden reference for diagnostics, so that mode must be
recorded separately from normal/performance execution. The explicit
`get_fallback_function` utility exists and calls a golden implementation; neither
the model/generator nor the selected common sampling path invokes it.
`ttnn/api/ttnn/device_operation.hpp::launch` validates tensors, creates device
outputs and dispatches the native operation; this path has no alternate host
implementation selected after an exception. These are scoped source findings,
not a claim that every TTNN operation or optional debug mode is host-free.

The observed startup settings `enable_fast_runtime_mode=true` and
`throw_exception_on_fallback=false` do not prove either presence or absence of a
fallback. `tests/probe_full_model.py --strict-fallback` is parsed and sets the
latter config field, but **the runtime has no consumer of that field**: the
`ttnn/` search finds its declaration in `ttnn/api/ttnn/config.hpp` and a legacy
tutorial. Reporting the flag as true is not strict-fallback evidence.

The permitted host boundaries in `tt/generator.py` are explicit:

- `_read_tokens` returns sampled IDs for caller output. `_sample` writes
  `decode_tokens` inside its sampling trace. B1 `_decode_body` and B>1 factored
  attention traces consume those same persistent slots. Default generation
  never uploads its returned token.
- `_read_logits` and `read_decode_output` perform requested host reads;
  `process_decode_output_host` only formats that submission's host snapshot.
  `sample_on_device=False` provides raw vocabulary logits for plugin-owned
  penalties, logprobs, grammar and custom processors without an internal draw.
- `generate(host_sampling=True)` deliberately reads logits, runs the CPU
  `HostSampler`, and uploads feedback. Unsupported device parameters raise;
  no exception silently selects this mode. `next_input` is the separate,
  explicit teacher-forcing feedback boundary.
- CPU prompt/page-table/position/sampling metadata validation, chunk input
  construction, reset metadata and empty/inactive-row sentinels are API work.
  They do not execute model tensor arithmetic on the host. All-inactive calls
  do not run a model or sampler. This audit does not describe the entire Python
  generator as host-free.

`QwenModel.embed/forward_chunk/decode_batch/logits` use native TTNN/custom device
operations. Decoder construction rejects meshes other than 1x4. No exception or
runtime branch switches to a single-chip model, replicated model, CPU decoder,
or lower-precision cache. The inherited terminal gather and
per-layer collectives retain their selected layout policy.

## Native rounding changes

The model's `MeshMoe` now passes `stochastic_rounding=False` to native
`moe_fused_swiglu`. The public binding, operation arguments, reflected cache-key
attributes and program factory carry that boolean to the compile-time
`MOE_FPU_STOCHASTIC_ROUNDING` define. The device kernel always performs normal
hardware startup; the false variant omits the subsequent FPU stochastic-rounding
enable/reseed calls and retains deterministic rounding. It is still the native
fused SwiGLU kernel, not a Python custom expert implementation. Tensor formats,
LoFi expert fidelity, BF16 DEST, packer configuration, routing, dispatch and
combine strategy remain unchanged. The default argument and fallback compile
define remain true for other callers sharing this kernel. Thus this arithmetic
change is selected by this model, not applied silently to every native MoE user.

The separate Blackhole LLK change in
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h`
repairs lane alignment and floating-add association in
`perform_reduce_row_face_pair`. It affects native float SUM/AVG row reductions
using this implementation, including the model's residual normalization; it is
**not model-scoped**. Compile-time `RowFold::FloatAdd` guards leave integer-add
and min/max branches unchanged. This is device arithmetic repair, not a host
fallback. The scope requires the native row-reduction regression suite as well
as resumed-prefill and refreshed all-layer HF accuracy. The device owner's
`rms_native_regression.log` records10 passed and normal device close;
`rms_repair_review.md` describes its scope. The earlier all48 revision also
passes in `full_accuracy_v2.json`: prefill and traced teacher forcing each report
97/100/100% top1/5/100. Installed `readiness/prefill_resume_check.json` passes
starts32/68/256 of length257, top1 agreement1.0 and minimum PCC0.9999657; both
missing-history controls reject at starts32/68. This establishes numerical gates for that source revision, not the later
head/checkpoint/workspace revisions, final performance or max-context execution.

The accepted n=1 feature-parallel FP32 MoE reduction and n>1 native BF16 grouped
reduction remain distinct. Poolv5 proves uninterrupted versus migrated B2 and
matching private/pool lifecycle outputs/state exact, but later forced-token
B1/B2 logits fail the frozen accuracy gate. That known failure remains work;
correct state migration does not excuse it. See `shared_kv_pool_audit.md`.

## Cache and reset ownership

`allocate_kv_cache_per_layer` validates the entire setup before mutation and
only binds before preparation. Deferred caches cannot execute until bound.
When replacing generator-owned private K/V, it releases those allocations before
creating the global pool; an allocation/bind failure leaves an explicitly
unusable instance. Borrowed storage is not freed through that replacement path.
`SharedKVPool` retains allocation ownership and aliases; `freeze` forbids later
rebinding. Neither request reset nor slot migration clears/copies shared K/V.

`SlotStateMover` copies private tensors through preallocated scratch at fixed
addresses, including dormant slots and page tables. Token feedback and full64-bit
seed state move through their separate device helpers. `_reset_cache` clears
recurrence/conv/PLE/indexer/position, preserves shared page mappings, and invalidates
the host mapping shadow so a new request supplies authoritative tables.
`release_request` resets the selected slot's private model state/checkpoint and sampling output/seed ownership; it preserves shared KV storage and other slots.
`close` releases captures and leaves the caller's mesh ownership intact.
`PrefillCheckpoints` owns one compact device snapshot per configured request:
GDN conv/recurrent, PLE conv/token history and QSA raw history/position. It omits
KV, compressed index keys and page tables. The eligible even-position singleton
cannot write a compressed key (writes occur only at position mod4==3); replay
overwrites its last KV row before causal attention reads it. Current validated
page ownership remains authoritative. Reset, decode consumption and close
invalidate checkpoints; slot remap permutes checkpoint buffer references and
metadata outside captured operations without freeing any buffer or allocating
migration scratch. Cache position and saved tail token must match before
restore. All48 pre-workspace resumed starts31/68 and matched-seed submission
order pass exactly in `prefill_checkpoint_full48_v1.json`; the historical
start31 failure in `prefill_signatures_full48_v2.json` remains preserved, not
an unresolved checkpoint result. Servingv6 adds reduced integrated workspace
and lifecycle coverage. These source properties still require full48 current-revision shared/private
release/reuse/remap and changed-page hardware checks; snapshot allocation is
charged in capacity planning, not treated as free.

## Separate watcher run

Run this only under the serialized device owner after the ordinary reduced
serving gate passes and after restoring the accepted mesh descriptor. Keep
`TT_METAL_WATCHER_DISABLE_*`, `TT_METAL_WATCHER_DUMP_ALL` and
`TT_METAL_WATCHER_SKIP_LOGGING` unset. No Tracy wrapper or profiler accompanies it.

```bash
env -u TT_METAL_DEVICE_PROFILER -u TT_METAL_TRACE_PROFILER -u TT_METAL_STREAMING_PROFILER \
  TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_8_flash_next/doc/full_model/serving_watcher \
  TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 3600 python_env/bin/python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_serving \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/serving_watcher.json \
  > models/autoports/qwen_qwen3_8_flash_next/doc/full_model/serving_watcher.log 2>&1
```

`TT_METAL_WATCHER=10` means a ten-second polling interval, per
`tt_metal/llrt/rtoptions.cpp`; device instrumentation is enabled for the run.
The probe includes real layers0/1/3, mixed/tail/resumed prefill, traced decode and
common sampling, persistent feedback, metadata reload, active/dormant migration,
reset/reuse, host compatibility and trace cleanup. Its JSON must report every
check passed and `caller_closed_mesh=true`. Inspect watcher attach/detach records
for the target devices. The existing `TT_METAL_LOGS_PATH` setting places
`generated/watcher/watcher.log`, `kernel_names.txt` and `kernel_elf_paths.txt`
inside this run's `doc/full_model/serving_watcher/` directory; preserve those
files alongside the probe JSON/log. No watcher assertion or
sanitizer failure may be dismissed as a numerical tolerance issue.

Watcher corroborates device execution and memory/protocol integrity; it does
not prove absence of host logits or numerical accuracy by itself. The source
boundaries, trace/counter assertions and independent accuracy checks supply those
parts of the audit. The device owner ran this scope successfully as `serving_watcher_v1.json/log`;
the CPU-only audit author did not access devices. The artifact passes all16
checks, records281.706s JSON wall time (external process about299s), unchanged
10905 program entries and clean caller mesh close. The dedicated watcher log
has no ERROR/ASSERT/CORRUPT hits. Raw evidence is retained under
`serving_watcher/generated/` (about48MB); it is separate from profiler runs.

## Remaining evidence

`serving_contract.json` has63 unique rows:37 supported,21 declared unsupported
and5 missing. Supported rows now cite the exact reduced serving checks and
host boundary scope; they do not claim plugin/server execution. Remaining rows:

| Row | Work still required |
| --- | --- |
| get_max_tokens_all_users | Measured capacity reconciliation and the single maximum-context prefill/decode execution. |
| get_kv_cache_spec | Verify all48 actual layer specs and12 QSA bound geometries in the prepared B32 shared probe. |
| allocate_kv_cache_per_layer | Exercise all12 full-model pools/B32; retain exact reduced lifecycle proof and repair the known cross-geometry numerical failure. |
| decode_forward | Repair later-token B1/B2 accuracy failure, then full48 B32 acceptance. |
| request_isolation | Resolve cross-batch numerical dependence; same-B2 seeded order/migration already passes. |

The all48 B2 ragged-prefix gate now passes in `ragged_prefix_dependency_v1.json/log`: lengths300/668, starts283/68,24 predictions per row,100% top1 agreement and minimum PCC0.9999663. Tracking is enabled;10908 program entries remain unchanged and close is clean. This proves numerical retained-prefix behavior; semantic recall remains unassessed and requires matched HF controls before any semantic investigation.

Separately required stage gates remain qualitative review, current-revision
validation of the selected sampler, bounded factored decode and shared workspace,
and isolated teacher-forcing versus token-out performance,
maximum-context execution, review and local commit. Full262144 allocation is
not maximum-context execution. Native row-reduction regression10 cases and
earlier-revision all48 HF/resume pass; current integrated revision requires refresh.

### Historical tracked serving result (v4)

`serving_probe_v4.json/log` passes16/16 checks with allocation accounting and
program-cache allocation tracking enabled. It records191.47s wall time
(181.69s preparation; external process about199s),20 captures,88 prefill,
54 model and59 sampling replays. Program entries stay10905 before/after;
request operations release0 traces, then explicit unload releases all20 once
and the caller closes the mesh. Its intentional live-allocation control rejects
before replay, then clean replay succeeds. Snapshot reads are test diagnostics,
so these timings are not token-out performance claims.

The v3 failure retained temporary last-logit slices across sampler replay.
Deleting `device_rows` after the persistent input copy fixes Python ownership;
v4 passes the real sampled-prefill branch and all later lifecycle checks under
the unchanged tracker. `sampled_prefill_lifetime_audit.md` and its four CPU
regression tests retain the source diagnosis and negative control. The v3 log
lacks an allocation traceback, so exact historical buffer-ID attribution is
still inferred; the repaired execution path is now hardware-verified.

### Prepared all-layer shared-pool batch check

`tests/probe_full_model_batch.py --shared-kv` now constructs the same all48 model
with deferred K/V and binds all12 actual QSA pairs through
`allocate_kv_cache_per_layer` before any preparation/capture. Each layer receives
its own pool and a per-layer permutation of disjoint global pages across fixed
slots. At B32/context128 each pool is `[128,1,32,256]`, BF16 TILE, four pages per
slot;64 temporary warmup pages per pool fit within128. All32 request caches and
the recurrence/indexer scratch bind the same12 pool pairs, with FP32 recurrent
state preserved. The probe records the actual all48 `cache_spec`, verifies every
slot/layer/rank page mapping, and checks final pool addresses unchanged.

The existing solo/mixed-prefill, frozen decode accuracy and device-feedback
assertions are unchanged. Shared mode additionally tests an all-inactive no-op,
alternating inactive rows, all inactive token/seed rows and all device positions,
plus an exact complete48-layer dormant-cache snapshot including that request's
owned global KV pages. Default private mode keeps its existing execution path.

Historical source arithmetic for B32/context128, per rank (predates head padding
and compact checkpoints; superseded for planning by capacity_plan.json and
memory_forecast_current.md):

| Item | Bytes |
| --- | ---: |
| Known model/generator state before global K/V, including33 private-state copies | 29,568,320,000 |
| Twelve BF16 K/V pools | 50,331,648 (48 MiB) |
| Listed resident total | 29,618,651,648 |
| Separate unmeasured runtime/trace allowance | 2,147,483,648 |
| Headroom against inherited34,178,731,008-byte allocatable bound after allowance | 2,412,595,712 |

This is a planning footprint for the probe's4096 aggregate token positions,
not a memory measurement or the pinned worker's output/null-padding allocation.
It retains per-slot private recurrence/indexer state and defers all K/V, avoiding
a private-plus-pool setup peak. The JSON records real memory views after load,
binding, preparation and the final workload; those measurements remain required.

CPU checks passed12 layouts (B2/8/16/32 at C35/128/129), all per-pool page IDs
unique/in range, distinct per-layer maps, exact BF16 pool byte arithmetic, and
four invalid-input cases. Syntax/formatting pass. No hardware execution was
performed for this optional mode by its author. Ready serialized command:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
timeout 7200 python_env/bin/python -m \
models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_batch \
--shared-kv --batch-size 32 --capacity 128 \
--output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch32_shared.json
```

DeviceSeedState implements only the common request-seed hash and progression.
It does not implement a custom probability distribution or sampled-token draw.
Its full64-bit arithmetic passed a seed-only device probe. Actual sampled-token
integration exposed a first-capture binary-loading gap, repaired by eager initial
program dispatch with zero update/active masks. The actual common sampler plus
generator trace wrapper passes42 checks,6 captures and41 replays in
`seeded_sampling_device_v2.json`; no custom distribution was introduced.


### Latest integrated reduced serving and all48 checkpoint evidence

`serving_probe_v6.json/log` passes all16 checks,43.9047s instrumented wall time,
2177 stable program entries,10 captures,132 prefill/54 model/59 sampling replays,
and0 request-time releases. Resident token/position/page refresh deltas remain0.
It covers the integrated checkpoint, runtime-mask token preservation and shared
prefill-output wrapper: inactive/dormant feedback, changed and unchanged page
writes, cross-request reuse, heterogeneous seeds, independent reload/reset,
explicit host compatibility and return to device sampling, delayed immutable
snapshots, failed-capture cleanup, tracker positive control and final replay.
This is reduced real0/1/3 B2/context128 evidence, not all48 performance.

`prefill_checkpoint_full48_v1.json/log` passes exact all48 B2 shared-pool mixed
and reversed logits, resumed68→257 and31→265 (top1=1, maxabs=0), and matched-reset
decode submission order with exact complete request state/token feedback and
reused traces. All six prefill signatures remain retained, no request-time
release or late program compilation occurs,2182 programs remain stable, and
wall time is269.265s. The generator hash is
`f7c458a3fae43f481cca514acb9699efbe3e8991d25ac811f34823ebded16675`, before the
single-output workspace edit archived in `prefill_workspace_source_baseline.json`.
It resolves the recorded odd-tail continuation failure for that revision but
is not current-workspace HF accuracy, B32 capacity, final performance or stage
acceptance. No missing serving-contract row is promoted by this audit update.
