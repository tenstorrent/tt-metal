# Frozen-source full-model runtime contract audit

CPU-only inspection of the frozen stage7 integration and existing final artifacts. No production/test/native source was edited and no device was opened. Source identity is pinned in [runtime_contract_source_hashes.json](runtime_contract_source_hashes.json); independent semantic-key checks are in `prefill_semantic_key_audit.json`. This is a source/evidence audit, not a new hardware run or stage-review verdict. This refresh reconciles current source bytes with completed prefill/teacher, B32, bounded qualitative, all40 runtime/performance and reduced native-profile evidence. Long-context smoke/census also passes; independent stage review remains pending.

**No concrete correctness defect was found in the selected frozen path.** Canonical split model/sampling tracing, persistent token feedback and device positions remain present. The experimental explicit persistent-broadcast sampler is separately rejected by a native partial-packet transfer defect; it was never selected. Final serving and resident-performance/replay gates pass at the scopes below. Reduced native profiling and bounded prefill attribution are complete; context qualification passes; independent review remains pending.

The audited checkout HEAD is `ece17d228671dbc1375e45c6e73f90156707aa30`; the worktree source hashes, rather than HEAD alone, identify this uncommitted stage. Exactly two audited byte hashes changed since the previous audit: `tt/mesh_narrow_scan.py` is now `a125c7c516e75eb7d1e4518457a937daec6e6b93cd356ea1d2007499ccb26a05`, and `tt/mesh_direct_recurrent_sum.py` is now `a30a709df55c198ab4c3fa451d0495c3fc7db7f990133e4128a4922a61588eff`. These are the reported formatting refresh; this audit independently pins the current bytes and verifies that final prefill/teacher provenance already contains those exact hashes. All 16 audited sources covered by each accuracy sidecar match; B32's narrower sidecar covers three audited sources, all matching; the frozen final source manifest covers 19 of 20 audited sources, all matching. Missing coverage is listed explicitly in the hash JSON. `final/source_provenance.json` still has `passed:false` and an empty job list, so it is used only for source identity, never as an aggregate completion verdict.

## Canonical token-out and sampler path

| Contract | Exact source anchors | Finding |
|---|---|---|
| Persistent token/logit state | `tt/generator.py:64-85` | Positions are one persistent INT32 scalar/slot; tokens are a persistent UINT32 `[1,1,1,32]`; sampler-ready BF16 logits are persistent `[1,1,32,padded_vocab]`, vocabulary-sharded on mesh dimension3. Prior/incoming token buffers and32-row active mask are also persistent. |
| Terminal output to sampler | `tt/generator.py:444-461`; `tt/model.py:189-204`; `tt/lm_head.py` | Model terminal produces local vocabulary shards. Generator maps logical rows to physical request slots, pads inactive physical rows, and copies into persistent sampler logits. New head geometry changes GEMM storage/program configuration; it does not gather full vocabulary to host. |
| Canonical token-out | `tt/generator.py:463-470`; `models/common/sampling/generator.py:413-444` | `_run_sampling(...,tt_out_tok=self.tokens)` writes the same persistent token tensor consumed by the next model trace. Active-row merge restores inactive token bits; seed advancement is device-side and masked. |
| Separate stable captures | `tt/generator.py:472-524` | All eager programs, copy/readout interfaces and persistent state are prepared before capture. CQ0 captures one model trace (`model` or `model_fixed`) and a separate sampling trace. Shared buffers impose model→sampler→next-model data dependency on the same CQ. |
| Nonblocking resident replay | `tt/generator.py:560`; `tt/generator.py:568-624` | `decode_forward(...,refresh_inputs=False,read_tokens=False)` queues model and sampling `execute_trace(...,blocking=False)` and returns resident tokens. No tensor readback/upload occurs on the selected active resident branch. |
| Greedy plus top-k/top-p | `tt/generator.py:189-241`; `models/common/sampling/tt_sampling.py:340-364,962-1048,1247-1502` | Greedy maps to k1,p0,temp1 with physical maximum top-k32. Device sampling supports bounded top-k≤32 and per-row top-p/temperature; unsupported unbounded or larger stochastic top-k is explicitly rejected unless caller selects host compatibility. |

Here “split sampling” means the canonical generator's separately captured common sampling path, preserving resident logits and `tt_out_tok` feedback. It is not a claim that every native operation has a `sampling_split` name. The common implementation performs local candidate selection, candidate value/index collectives, device tie adjustment, device seeding and `ttnn.sampling(output_tensor=tt_out_tok)`.

No model configuration enables force-argmax: the generator passes no `SAMPLING_AG_CONFIG`, and common sampling defaults `_allow_force_argmax_sampling=False` (`tt_sampling.py:354-364`). Its force-argmax full-vocabulary gather branch exists at `tt_sampling.py:1138-1245` but is not selected by this constructor. The normal path gathers top-k candidates (`tt_sampling.py:1360-1412`), not the full vocabulary. The misleading common comment “host-side tie-break” names Python-dispatched TTNN tensor operations; `_adjust_values_for_tiebreak` itself uses device reductions/comparisons/arithmetic and has no `to_torch`/host-logit calculation.

Final sampler acceptance and clean comparisons pass as recorded below. The [native decode audit](final_profile_audit.md) validates the selected31-op split sampler: local `TopkRoutePrep`/`TopkLargeIndices`/`TopkRouteFinish`, two top32 candidate gathers and device sampling; no generic `TopKDeviceOperation`, argmax op or full-vocabulary gather occurs in this selected decode window. This does not describe eager prefill, whose expert router uses generic TopK. Clean sampler control0.487714543ms is2.28% of all40 queued21.416868244ms, not dominant; the reduced two-layer kernel-work share25.63% is a different scope. The common sampler documents a lowest-token-ID tie limitation when a shard has more than32 equal maxima (`tt_sampling.py:981-986`); returning any maximum is semantically greedy, while exact lowest-ID behavior must not be overstated. Existing cross-shard tie controls cover their explicit fixtures only.

## Device positions, RoPE, cache and fixed slots

- `tt/model.py:268-283` derives RoPE table indices from persistent device positions, looks up persistent cosine/sine tables with device embeddings and supplies the same current-position tensor to attention. Inactive slots use safe RoPE indices and current position−1 through `tt/slot_state.py:202-210`.
- `tt/model.py:285-313` captures the fixed physical slot list once. After all selected layers and terminal work, positions advance with `ttnn.plus_one` for B1 or masked `advance_position` for fixed-batch execution. The latter reads activity on device (`tt/slot_state.py:212-221`); it does not turn positions into host integers.
- Cache binding is explicit at generator construction and validated (`tt/generator.py:60-63`, `tt/cache_contract.py`). Resident decode refuses a different cache identity or unprepared execution signature (`tt/generator.py:586-588`). Input refresh is an explicit admission/update operation, not an implicit per-token action.
- Changed-only page tables remain implemented at `tt/generator.py:275-292`: host-page snapshots are compared and only changed rows copy into persistent page-table tensors. With both page arguments absent the method returns immediately, so measured resident loops perform no host page comparison or tensor operation. Explicit changed-page submissions have a host control boundary by design.
- `tt/generator.py:261-272` changes activity only when the selected slot tuple changes. `tt/model.py:298-303` masks hidden rows before and between layers. Linear recurrence computes temporary output for masked execution and commits only active state (`tt/multichip_decoder.py:524-538`); the convolution reader accepts a device activity input (`:182-237`). Full attention uses current-position−1 for inactive cache/SDPA work and zeroes inactive attention outputs before subsequent arithmetic (`:1788-1836`).
- Sampling restores inactive tokens and advances only active seeds (`tt/generator.py:463-470`, `tt/slot_state.py:247-254`). Preserving seeds/positions/cache/token bytes requires the fresh inactive-slot replay gates; the source path has not replaced masks with replicated active rows.
- Public serving commands retain explicit page, position, prompt-length, batch/slot and output-lifetime state through `tt/serving.py`, `tt/serving_state.py`, and `tt/prefill_validation.py`. The audit does not broaden support beyond `tt/serving_capabilities.py` or infer that every detailed serving snapshot checks every layer.

Resident execution trusts the caller's admitted context budget. Refreshed positions and `generate` lengths are checked (`tt/generator.py:594-606,741-744`); no per-token host bounds read is added to resident replay. This is consistent with a serving scheduler owning admission/lifetime, not an automatic promise that replay beyond configured context is valid.

## Nonaligned and mixed prefill

`tt/prefill_validation.py:48-159` validates each row's absolute logical end/start, unique fixed slot and page ownership; it does not demand tile/page/chunk alignment. `tt/generator.py:381-424` iterates `(slot,start,end)` independently, clears state only for a fresh start0 request, processes the exact logical suffix and records each resulting end position. Mixed prompts therefore remain explicit serial per-slot work with shared bounded device workspaces.

`tt/prefill.py:26-40` preserves arbitrary linear-attention tails via64-row chunks and scalar full-attention progression until a legal aligned chunk is available. `:195-218` copies only valid token/activation rows and rounds MoE physical rows to32 while committing only logical rows. `:219-234` similarly bounds all-logits chunks and returns the final logical row. `tt/prefill_transport.py` remains the valid-row movement boundary. A logical257 or32769 prompt is not rounded into additional live cache tokens.

Prefill deliberately has host token admission and optional host all-logits output. It is eager and serial (`BoundedPrefill._busy`, `tt/prefill.py:164-193`); generation decode remains traced. These boundaries must not be misreported as a decode fallback or as traced prefill. The shared residual stream is still selected replicated BF16; `tt/prefill.py:294-295` explicitly refuses a different residual/MoE policy.

## Semantic prefill preparation key

The change at `tt/prefill.py:238-272` replaces Python object-address-bearing compute repr with semantic values. Current C++ `ComputeKernelConfig` contains exactly six fields at `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp:19-26`: math fidelity, approximate mode, FP32 destination accumulation, packer L1 accumulation, full-destination synchronization and throttle level. Wormhole/Blackhole names are aliases of the same unified type (`:37-39`). The key explicitly includes all six for `compute`, `expert_compute` and `grouped_compute`, retaining the original scalar switches, policy and sparse-geometry entries. Layer index is intentionally excluded; weight addresses are runtime bindings.

A CPU-only execution of the current `_layer_key` proved separately allocated equivalent configs/different layer indices share a key, while independently changing each of the six exposed fields produces a distinct key (`prefill_semantic_key_audit.json`). No device was opened. Different constructor scalar flags—including `full_model_optimized`, layer kind, CCL/layout switches—remain in the scalar switch tuple. `_cache_signature` separately checks shape, padded shape, dtype, layout and memory configuration (`tt/prefill.py:154-162`); every layer's `prepare_moe` still runs before representative warming (`:290-300`).

This proves the new serialization retains all current **ComputeKernelConfig** differences; it does not turn the key into a general validator for arbitrary future tensor/helper mutation. Existing non-scalar custom override dictionaries/helper types outside its named fields are not a new supported mutation surface. In the frozen selected model the constructor uses the same accepted architecture/policy for all layers of each kind and supplies no per-layer precision override. Changing custom policy/config after preparation remains unsupported. Final all40 boundary cases64/255/256/257/1025/2049 retain9590 programs and unchanged trace IDs; all40 mixed/resumed replay retains9632 programs. Those measured cases supply runtime evidence beyond semantic equivalence. This does not establish every future signature or every future context/signature.

## Optimized helper dispatch and CCL

The full model explicitly constructs every selected decoder with `full_model_optimized=True` (`tt/model.py:80-89`). Decoder setup selects production `mesh_cached_scan`, `mesh_direct_recurrent_sum`, `mesh_product_recurrent_sum` and `mesh_state_slab_recurrence` (`tt/multichip_decoder.py:1190-1203`). No experimental probe import or automatic host fallback was found in these production modules.

- Cached coefficient scan selects only packed input `[4,1,512,3136]` and local output `[2,1,512,2048]` (`tt/mesh_cached_scan.py:143-151`). Router/K1024/other supported geometry dispatches to `mesh_narrow_scan`; its unsupported cases return the original device helper. These are explicit device-kernel geometry choices, not Torch conversion. Cache padding is unspecified outside the consumed coefficient column; no-clear poisoning/product exactness evidence must remain linked to its acceptance.
- The full state-slab native guard runs first (`tt/multichip_decoder.py:388-394`), with explicit product-sum/direct/native fallback. Product-sum is guarded by concrete FP32 tiled state/column geometry; unselected geometries use the existing TTNN products and direct/native ordered sum (`tt/multichip_decoder.py:367-438`). Device fallback preserves prior arithmetic. Final reduced native metadata confirms supported B1 cached K512 input/output scans, narrow32 router and FP32 state-slab dispatch. Product/direct-sum remain guarded fallback helpers; do not claim they also execute in the supported slab case or that the same decode helper fires for every prefill row shape.
- Decoder all-reduce scratch and semaphores remain shared persistent workspace allocations (`tt/multichip_decoder.py:804-818,1094-1131`), passed into the selected direct AR (`:1270-1305`). Logical-row pad/slice removal does not change selected BF16/FP32 CCL arithmetic or mesh topology.
- `SamplerCCL.line_all_gather` still accepts but does not consume `buffer_key` (`tt/generator.py:38-50`). It uses shared cycling semaphores and captured temporary gather storage; **explicit sampler persistent output buffers remain disabled**, independently of the enabled decoder persistent scratch. The candidate-local ownership hook and barrier-preserving broadcast alternative were actually exercised, not dismissed from timing. [Watched v3 raw diagnostics](persistent_sampler_l1_watched_v3.diagnostics.json) show 3072/4096 mismatches per rank before sampling or trace capture: each local 32-column tile is correct, remote 96 columns remain poison. One 2048-byte tile cannot fill the native 4096-byte scatter packet, and `FabricScatterWriter::send` never flushes that tail before signaling completion. [Exact rejection/repro](sampler_persistent_broadcast_rejection.md) and [ownership/barrier audit](sampler_persistent_audit.md) retain the evidence. There is no per-operation packet-size API; global 2048-byte payload conflicts with selected FP32 collective pages. Width64 padding/compaction at one link and the narrow native one-tile packet specialization remain untried/unapplied, with no claimed speedup. This is a rejected experimental geometry, not a defect in the selected split graph or a claim that sampler persistence is enabled.

## Runtime fallback and host-boundary audit

The selected queued segment calls only `decode_forward(None,[0],refresh_inputs=False,read_tokens=False)` with nonempty active slots. Its branches are metadata checks, no-op absent-page binding, nonblocking model replay and nonblocking sampling replay. `_sample`'s Torch sampling branch (`tt/generator.py:527-559`) is reachable only through explicit `host_sampling=True`; normal constructors/defaults and quality recipes selectFalse. `return_logits=True`, `_read_tokens`, input refresh and prefill all-logits callbacks are explicit diagnostic/caller boundaries excluded from queued token-out measurement.

`tests/runtime_audit.py:15-32` rejects Torch dispatch and TTNN host conversion inside `device_only`. Current performance runner wraps queued replay and asserts zero readback, token/position/page/activity refresh and recapture deltas (`tests/run_full_model_performance.py:504-555`). Caller-observed delivery is a separate segment that deliberately reads one token per step outside that guard (`:596-637`). Traced teacher forcing is separate again: HF inputs are uploaded outside its guarded model replay, sampling does not run (`:156-244`). TTNN's global `throw_exception_on_fallback` flag alone is not the audit; actual guards, counters and native graph/profiler evidence are required.

No host tensor compute/conversion was found in the measured resident source path or new production kernels. Final all40 queued replay now passes the actual device-only guard with zero token/position/page/activity refreshes, readbacks or recaptures; reduced native evidence separately confirms the captured split graph. This runtime conclusion comes from final/performance.json, not from reduced timing or teacher accuracy with deliberate host I/O.

## Completed final evidence inspected in this refresh

| Gate/artifacts | Verified result and exact scope |
|---|---|
| [Prefill accuracy](final/prefill_accuracy.json), [sidecar](final/prefill_accuracy.json.provenance.json) | `passed:true`; all40 layers on1x4, context512, installed `run_prefill_check`, pinned AIME24 chat reference. One100-position row: top1=98%, top5=100%, top100=100%. Both requested top-k thresholds pass. Allocation tracking1/program-cache skip0. |
| [Teacher accuracy](final/teacher_accuracy.json), [sidecar](final/teacher_accuracy.json.provenance.json) | `passed:true`; all40/context512, installed `run_teacher_forcing`, same chat reference and100 positions: top1=98%, top5=100%, top100=100%. The artifact's46.406t/s/u is traced teacher-forcing diagnostic timing, not clean token-out throughput and not substituted for the final performance gate. |
| [B32 smoke](final/b32.json), [sidecar](final/b32.json.provenance.json) | All40, batch32, context1024, traced generation with explicit permuted page tables and fixed slots. Checks all31 inactive slots' convolution/recurrent tensors on mesh chip0 plus token/seed/position preservation; reordered returned tokens and final positions `[7,6,...,6]` pass. Program-cache growth0, recaptures0. This smoke deliberately includes2 token readbacks and1 activity refresh in its two-step resident delta; it is not a no-host-boundary performance measurement and does not prove all-rank KV immutability. |
| [Six-case qualitative review](final/qualitative/verdict.json), [paired text](final/qualitative/paired_review.md), [degeneracy check](final/qualitative/degenerate_output_check.json) | Six canonical chat prompts, all40/context1024,128 greedy tokens each, default thinking/presence penalty0. Read the12 paired continuations; they stay relevant and coherent within the cap. Degeneracy findings empty/exit0. Independently checked all18 verdict-referenced artifact hashes and byte equality of all six current TT texts to `doc/full_model/qualitative/tt_greedy/qXX/tt_completion.txt`. This is a bounded no-observed-regression smoke, not completed-task correctness or broad HF equivalence; both models spend the cap in reasoning, and q03's doubled “thinking” appears on both sides. |
| [AIME autoregressive result](final/aime_autoregressive/result.json), [verdict](final/aime_autoregressive/verdict.json) | Exact161-token AIME24 chat prompt, all40 layers,100 greedy tokens, `host_sampling:false`. Read both text files and verified the verdict's three artifact hashes. Both give coherent problem setup and stop mid-reasoning; they differ from token0. No solved-answer or exact-HF-token claim.99 model/100 sampling replays and100 explicit observation readbacks are diagnostic generation evidence, not resident throughput. |
| [Tracker positive control](final/positive_control.json) | Reduced layers0/3 intentionally allocate a live device buffer after capture; `positive_control_rejected:true` records tracker rejection. This validates the detection mechanism only, not all40 replay correctness. |

The generation result files retain their initial `pending reading` markers; the separate verdict JSONs and actual paired text supply the bounded review outcome. No new model execution was performed for this audit.

## Final runtime acceptance refresh

The following finished runs supersede earlier pending labels in this source audit:

| Artifact | Measured contract |
|---|---|
| `final/performance.json` | All40 clean batch1:127 nonblocking model/sampling replay pairs; zero token, position, page, activity refreshes, token readbacks or recaptures inside the queued interval. Two traces and8440programs remain stable. Device positions advance128→255. Device-only guard passes.21.416868ms/token includes split sampling and device feedback. |
| `final/replay_all40.json` | All40 B3, Watcher10, tracking1/skip0:17cases/32checkpoints; fresh and resumed nonaligned prompts, changed unused pages, mixed versus solo, inactive middle slot, sampled seed/parameter isolation.9632programs stable, captures2/releases2. Diagnostic observations are explicit and excluded from resident timing. |
| `final/serving_commands_all40_v3.json` | Nine serving checkpoints pass, including unsorted admission, asynchronous output ownership, simultaneous dormant remapping (selected recurrent state checked on all four chips),30changed page tables across10full-attention layers×3slots, zero unchanged updates, live RNG parameter-only updates and reset without reload. The two prior failures were harness count/shadowing errors, retained in the manifest; production source did not change. |
| `final/prefill_boundaries.json` | Logical64/255/256/257/1025/2049 at context4096 each return a first sampled token, then two resident feedback steps with zero refresh/readback/recapture; positions end at logical length+2. All9590programs and both trace IDs remain stable. |
| `final/sampler_acceptance.json`, `final/sampler_timing.json` | Final semantically greedy strategies pass real-terminal, negative-maximum and cross-shard tie fixtures. Selected split0.487649ms; padded split0.521800ms; force-argmax2.669194ms. These controlled component numbers are separate from all40 queued timing. |
| `final/queued_layer0.json`, `final/queued_layer3.json` | Four-rank eager/traced output and all-cache bitwise comparisons:96checks per kind. Five200-replay samples use resident selected residual layout with no per-replay transfer/sync and stable cache. Weighted stack estimate20.668310ms. |

The measured resident path has no host fallback or per-token host boundary.
Host logits, explicit admissions, diagnostic reads and caller-observed delivery
remain separate supported operations. Final reduced native operation attribution
is complete at its two-layer scope. The separate32769/context65536 smoke passes with two resident replay pairs,
zero refresh/readback/recapture and stable9590-program cache; context_memory_audit.md
records its unchanged endpoint census. Source identities are recorded in
`final/measurement_source_provenance.json` and
`runtime_contract_source_hashes.json`; the only source change between the
original correctness queue and final measurements is the serving test harness.

## Completed native and prefill evidence reconciliation

[final_profile_audit.md](final_profile_audit.md) records validated reduced
layers0/3 B1/context1024 native profiling: sixteen replays/component,199 model
and31 sampling operations per rank/replay,29440 validated execution rows and680
rank/program metadata entries. All fifteen recorded source hashes match the
clean all40 performance artifact and current files. Watcher is disabled for
this profile; its evidence does not replace the separate all40 Watcher run.
The optional WASM-copy warning refers to a different default directory; the
retained raw host trace/device report converted successfully. Complete advice
and CSV tables remain under final/profile_reduced.

Native operands/configs establish BFP8/HiFi2 FP32-destination reader3/block4
LM head, BF16-activation/BFP8-weight HiFi2 ordinary projections, BFP8/LoFi sparse active experts,
FP32/HiFi4 sensitive router/recurrence and BF16 paged SDPA/KV. Rank2 kernel work
is1.266784ms model and0.436511ms sampler, versus instrumented fenced reduced
wall time2.015464ms; these are not all40 latency or additive independent-control
costs. Full token-out is3.62% above the current queued-layer engineering estimate
20.668310ms before terminal allowance; the raw runner's24.99350229ms field
is a labeled historical estimate. The clean all40 useful-byte floor passes at
10.1001% of its stated analytical model, not measured physical bus utilization.

BF16 and FP32 all-reduce native inputs include the selected persistent scratch.
The FP32-input collective's BF16 dtype attribute describes its output, not a
pre-collective narrowing. Decoder workspace remains serial and persistent.
Sampler native metadata independently confirms barrier=true and
using_persistent_buffers=false for both candidate gathers. The experimental
broadcast's exact unflushed2048-byte tail in4096-byte packets remains a measured
rejection; padding/compaction and a native one-tile specialization remain
untried, with no enabled sampler persistence claim.

One bounded metadata limitation remains: embedding gather program metadata
retains64 rows from a reused program, while the preceding captured tilize
producer has logical1/padded32 rows. Width512→2048 and timing are clear; cached
height must not be used to infer runtime decode payload. The audit explicitly
preserves this inconsistency rather than misreporting a64-row decode contract.
It is not an observed numerical-policy/output failure.

Every final report advice family has a disposition in the native audit.
Inherited sparse input-L1/subblock losses retain their precise prior search
scope. Isolated selected-policy FP32-router L1/subblock controls now pass with no
kept gain (router_advice_comparison.md); broader sparse/head combinations remain
untried. Generic fidelity advice does
not authorize changing the selected expert/router/projection policy. No
exhaustive optimization claim follows from this audit.

[Prefill performance analysis](prefill_performance_analysis.md) completes the
bounded CPU attribution separately from runtime correctness. Clean all40 TTFT
boundary samples at64/255/256/257/1025/2049 differ−0.030% to+3.428% from the
baseline; one timed sample/length supplies no statistical or causal improvement
claim. Their nonaligned tokens, two resident feedback steps and program-cache
stability pass. The reduced128-token native prefill window has12468 eager and
124 traced rows;73.964–74.084ms eager and0.428–0.442ms traced kernel work per
rank. Its15761.827ms host TTFT includes profiler drains, not clean latency.
Only668 eager rows have entries in the decode-oriented native inventory, so
this is not exhaustive prefill fidelity/kernel identification.

The measured source/shape distinctions matter: two64-token embedding pieces,
one128-token SDPA chunk, a large physical-Y64 custom group consistent with
chunk64 recurrence, and128 per-token K1024 packed-input scans. The1024 reduction
axis is not a1024-token logical chunk. Individual inline roles remain unresolved
in the detailed CSV. Setup-only SDPA32..256 buckets are prepared variants, not
individually timed request lengths. Finer nonaligned per-chunk kernel attribution
and matched before/after prefill kernel capture remain unmeasured. None of this
reduces the advertised context. The independent long-context smoke now passes
as recorded in context_memory_audit.md.

## State-slab refresh: exact dispatch, preparation and batch positions

No additional concrete correctness defect was found. The previously observed batched linear `position=None` failure is fixed at `tt/multichip_decoder.py:1699` by the same conditional activity lookup already used by single-row `_attention_residual`. No positions or masks are invented: `None` and `{}` pass no activity; an existing activity tensor is forwarded. Parent-owned production B3 layer0, prompt129/16decode steps passed after this repair. Clean layer0 exactness passed with mean0.6029883754ms versus0.63254126ms for candidate1. These layer results do not replace full-model gates.

`mesh_state_slab_recurrence.StateSlabRecurrence.supports_native` accepts only canonical local Q/K FP32 `[1,1,4,128]`, V BF16 `[1,1,8,128]`, beta/decay FP32 `[1,1,8]`, and state FP32 `[1,8,128,128]`; all require canonical padded dimensions, TILE layout and interleaved L1/DRAM storage. Optional output state must match the same complete state contract. Absent decay or another unsupported input returns to the selected device recurrence, which retains its own guards. The engine is installed only for optimized linear layers on the constructor's required1x4 mesh. Production kernels are literal strings, with no probe imports, source rewriting or runtime host conversion. Native Q/K permutations and explicit `round_state` forwarding are unchanged.

The SFPU kernel preserves BF16 rounding only at state consumption and separately rounds each FP32 multiply/subtract/add. FP32 state commits, ordered16-then8 contractions, Q scaling, HiFi4/FP32 destination mode and terminal row geometry match the qualified component. Each of32workers owns one128x32state slab and exclusively reads/writes its state pages. Dedicated CB26 has one pack producer, and explicit unpack source reconfiguration surrounds the BF16 V copy. The local CB total is116736bytes/worker; these are program-local buffers, not persistent tensor allocations. Scalar replication uses bitwise shifts/OR. Padded terminal rows remain unspecified and are outside the logical eight-head tensor. No kernel comment was found to promise valid padded rows, different head counts or unsupported batched state tensors. The `state_commit` geometry describes the selected persistent-DRAM/temporary-L1 usage; the explicit guard also safely accepts an interleaved L1 output supplied by a caller.

`generator.prepare` eagerly executes an unmasked singleton before capture and, for capacity>1, the complete fixed masked row set (`tt/generator.py:477-488`). This warms rounded state consumption with persistent DRAM output and rounded state consumption with temporary L1 output plus masked commit. Activity is device data, so inactive values reuse the same prepared program. Bounded prefill prepares all1..64linear row counts and separately runs the public-length1 singleton (`tt/prefill.py:310-327`); that singleton selects decode recurrence with `cache_boundary=False`, warming unrounded consumption with persistent state. Multirow prefill uses its unchanged chunk recurrence. The selected generator never requests unrounded temporary masked state; component tests nevertheless qualified that fourth combination. Prepare remains before capture, and full-model cache-count checks remain the runtime evidence that no variant was missed.

The new helper runs inside the model trace and contains only TTNN device allocations/dispatch and metadata checks during eager/capture construction. Resident token-out replays already-captured model and sampling programs without reconstructing this helper or querying state on the host. It adds no per-token host readback, synchronization, uploads, sampling replacement or position advance. Fixed-slot temporary output still reaches the existing device `copy_if` in `_linear_attention`; the kernel never writes inactive persistent state when output_state is absent. Convolution masks, paged attention, device RoPE, split token feedback and changed-only page submission remain unchanged.

No weight, activation, KV-cache, CCL dtype/fidelity setting, rejection ledger or residual policy was modified by the slab/None-position changes. The existing constructor default is replicated BF16 residual (`tt/multichip_decoder.py:844`), which bounded prefill requires; this refresh preserves that already-selected policy rather than introducing a new replicated stream. Cache allocation/layout and advertised context remain unchanged; `doc/context_contract.json` appends the completed stage7 smoke. Component exactness covers four128step variants with12288rank comparisons each; B1 layer0/34 and production B3 integration results, final all40 accuracy/trace gates and separate reduced0/3 native dispatch evidence establish the applicable scope. There is no all40 native profile.

CPU guard and dispatch execution evidence is in `state_slab_production_cpu.json` and `state_slab_dispatch_cpu.json`; literal kernel/descriptor equivalence and hooks are also recorded there and in the slab audit. The refreshed source hash map includes the production helper. This audit combines CPU source inspection with the explicitly scoped final runtime evidence above. Native dominance/configuration audit is complete at the scopes above; long-context capacity passes; independent stage review remains pending. Bounded qualitative evidence does not establish broad quality.


Final optional router probe tracking1/skip0/Watcher10 passes all three compared
geometries after releasing a probe-local Python loop alias. That bug was outside
the frozen implementation/tests and never affected selected full-model tracing.
No tracker exemption was added. The completed source-backed native accounting
is in `native_accounting_derivation.md`; all operation sums remain distinct from
critical-path spans and clean all40 wall latency.
