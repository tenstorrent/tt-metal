# Stage7 optimization checklist and remaining evidence

Current acceptance disposition (2026-11-16): the [owner decision](owner_decision_2026_11_16.json) accepts the measured decode speed as a run-specific performance limitation and prohibits further tuning for this floor. Numerical floor failure and round-1 findings below are historical evidence; this amendment supersedes their acceptance-blocking disposition only. Independent [round-2 review](stage_review_round2.md) returned `clean-pass`; stage 7 is complete under the amended owner contract. Checkpoint sources are `checkpoint_sources_v5.json`; measured sources remain v4.

CPU evidence audit from2026-11-14 after compaction rollback, retained for the2026-11-16 owner-amended closure. The owner has confirmed the4h wall-cap Stop Rule at15:09:21UTC. Current source keeps
native causal scoring, head20/3/2, exact residual write/local rank sum, RHS
hyperconnection norms and R16 Q1 GR-up/GDN-output. The original GR-down
slice/concat remains selected after native compaction failed integrated mixed
prefill. [Source/dispatch table](operation_topology_audit.md#current-exact-kernel-selection-and-evidence-boundary)
binds each selected helper, exact fallback and component evidence.

[Final resident v1](final_resident_v1.json) measures full48 B1 at103.064960ms,
9.702619t/s/u and4705.950625ms TTFT, with all128 caller-control tokens equal
to baseline. The resident window has127 model/sampler replays and zero
per-token reads, synchronizations, uploads, captures or releases. [Restored-source serving Watcher v2](final_serving_watcher_v2.json)
passes all16 reduced checks. Full48 [final validation v2](final_validation_v2.json)
completed PASS at14:24:33UTC: AIME24 chat-template prefill and teacher forcing
both top5/top100=1;257-token resumed/nonaligned controls,32769-token context,
full-body runtime audit and autoregression pass. [Fresh qualitative verdict](final_validation_v2/qualitative/verdict.json)
is within the matched-control margin after independent label-hidden review.
[All48 B32 HF/control v1](final_batch32_hf_v1.json) also passes: two2704-prediction
runs each top1=.978180473, top5/top100=1, exact matched-history logits/tokens and
final request-state hashes after slot migration, dormant/inactive preservation
and resident sampled feedback. This C512 test uses correlated continuations of
one203-token AIME prompt; it is not32 independent prompts or a throughput test.

[Measured sources v4](final_sources_v4.json) supersedes [v3](final_sources_v3.json)
(coverage-only router-header/shared-source additions) and [v2](final_sources_v2.json).
After B32, v4 adds only standard32x32/untransposed-tile rejection in the causal
scorer validator. Supported-path kernels/math are unchanged;25 CPU metadata
checks and [final code hooks v3](final_code_hooks_v3.log) pass. Earlier B1 validation and B32 HF
reports keep their original hashes and precede this guard change. Final resident, [component traces](component_trace_v2.json) and [decode profile](terminal_profile_analysis_v2.json)
match all72 model and3 shared hashes in v4. [Full48 mixed/resumed prefill](final_prefill_signatures_v1.json)
passes on v4. [Prefill-profile analysis](prefill_profile_analysis_v1.md) and semantic advice review are complete; [B32 clean throughput](batch32_resident_v1.json) passes at 0.390840t/s/u (12.506878aggregate), with all384 slot/layer positions and four-rank page tables checked. Historical round1 returned `more-work-needed` for the floor alone; the2026-11-16 owner accepts that performance limitation, with independent round-2 `clean-pass`. The final strict FUNCTIONAL fraction6.223772% still
fails10%; validation's stage_acceptance_claim remains false.

The explicit user contract preserves the accepted decoder dtype/fidelity,
BF16 KV/activation/CCL policy, rejection ledger and rank-owned residual
stream. No new precision frontier is authorized here. Inherited expert
BFP4/LoFi is the previously accepted physical-fit policy, not a new stage7
selection. Stage8 owns any full-model datatype frontier.

## Current measured candidates

| Candidate | Evidence and disposition | Remaining scope |
|---|---|---|
| Native causal score/selector | [causal_packed_v4.json](causal_packed_v4.json):36 real-operand cases, Q1/3/17/32 and positions0 through near maximum; live FP32 scores and ordered selected IDs exact. Selected by `MeshQSA._selected_ids(..., packed_decode=True)`. [Recovery/fix record](AUTOFIX.md) and [upload repair](AUTOFIX_upload_v1.md) retain failures. | Component evidence has tracking disabled. It does not replace final integrated allocation safety, model accuracy or long-context smoke. |
| Fixed-policy head geometry | [Initial sweep](head_sweep_v2.json), [CSV](head_sweep_v2.csv) selected20/3/2. Completed larger-block [v2 JSON/provenance](head_larger_blocks_v2.json), [CSV](head_larger_blocks_v2.csv), [log](head_larger_blocks_v2.log) and [v3 JSON/provenance](head_larger_blocks_v3.json), [CSV](head_larger_blocks_v3.csv), [log](head_larger_blocks_v3.log) pass every Q1/Q32 candidate with unchanged BF16/HiFi4/FP32 DEST/packer-off and exact common greedy tokens. All measured blocks5/8/10 are slower than20/3/2; selected geometry retained. | [Complete choices, timings and L1 diagnosis](head_geometry_followup_v1.md). Real weights/synthetic activations; acceptance_eligible=false. [v1 failure](head_larger_blocks_v1.json), [CSV](head_larger_blocks_v1.csv), [log](head_larger_blocks_v1.log): c10/b8 physical104640B overlap; adapted40/80-core block10 controls pass. Unexecuted20/b10 is a source capacity estimate, not a measured rejection. |
| One-reader head control | [head_sweep_v1.json](head_sweep_v1.json), [CSV](head_sweep_v1.csv):8/1/1 fails during compile/allocation: static DFB region3108800bytes exceeds1572864bytes/core. | Exact implementation/L1 blocker for this geometry, not evidence that all one-reader decompositions are impossible. |
| Sampler async barrier gather | [sampling_ccl_v1.json](sampling_ccl_v1.json):three clean64-replay windows; synchronous0.474610–0.474808ms, async0.480787–0.481158ms. [Tracked counterpart](sampling_ccl_tracked_v1.json) passes with no allocation exemptions. **Rejected as slower**; synchronous selected sampling CCL retained. | Candidate deliberately does not pass `persistent_output_buffer`: that factory option disables the inter-invocation barrier. Do not claim an unmeasured persistent-output variant passed or failed. |
| Before full model | [baseline_resident.json](baseline_resident.json), [driver snapshot](baseline_resident_driver.py), [manifest](manifest.json):all48, B1,128in/128out, C262144; resident138.618793ms/token,7.214029t/s/u, TTFT7035.653ms. | [selected_resident_v1.json](selected_resident_v1.json) completes the same-harness pairing:113.881454ms,8.781061t/s/u,4805.506ms TTFT; all128 tokens equal baseline. Public resident decode_forward is113.846141ms,8.783785t/s/u. Caller-read, logits-only and readiness teacher-forcing boundaries stay distinct. |
| Reduced serving/watcher | [Restored-source Watcher v2](final_serving_watcher_v2.json), [log](final_serving_watcher_v2.log): all16 reduced serving/state checks pass with tracking and program-cache allocations included after compaction rollback. | Reduced scope only; current [full48 v2](final_validation_v2.json) separately passes quality/runtime/context. All48 [B32 v1](final_batch32_hf_v1.json) separately passes HF/migration controls; final B1 resident/decode-profile refresh passes; B32 clean throughput passes at 0.390840 tokens/s/user (12.506878 aggregate); [prefill analysis](prefill_profile_analysis_v1.md) is complete. |

## Optimize checklist

“Inherited” means the exact earlier-stage evidence is retained; it does not
claim that an integrated stage7 run has already passed. The
[full-path topology audit](operation_topology_audit.md) contains tensor shapes,
padding, communication bytes, next-consumer layout and collective-family tables.

| Skill item | Evidence / decision / pending work |
|---|---|
| Real-shape warmed traced decode and prefill; context and nonaligned lengths | [Baseline](baseline_resident.json), earlier [causal/head selection](selected_resident_v1.json), [exact bundle](exact_integrated_resident_v1.json) supply all48 B1 128in/128out C262144 measurements. Current [v2 validation](final_validation_v2.json) passes32769 smoke and257-token nonaligned resume. [Final resident](final_resident_v1.json) refreshes31/32/33/128/1024 TTFT; [Reduced six-bucket prefill analysis](prefill_profile_analysis_v1.md) completes89,740 operations and semantic advice review; matched v4 sources. [Context contract](../context_contract.json) stays262144, BF16 page32 KV unchanged. |
| Topology audit before tuning; coherent collective families | [Stage7 topology](operation_topology_audit.md), [stage5 audit](../optimized_multichip_decoder/operation_topology_audit.md). Preserve entry-only embedding gather and zero inter-layer boundary conversion. |
| Rank against best correct path; reproduce final default | [Exact bundle](exact_integrated_resident_v1.json) improves resident latency113.881454→103.068541ms over causal/head-only selection, with128 unchanged caller-control tokens. [Serving v2](final_serving_watcher_v2.json) and current [full48 validation v2](final_validation_v2.json) pass. [Final resident](final_resident_v1.json) gives103.064960ms,9.702619t/s/u and4705.950625ms TTFT; matched-source decode profiling is complete; [B32 controls](final_batch32_hf_v1.json) pass in their correlated C512 scope; strict floor failed. [Correct split/argmax](terminal_clean_v1.json) retains device split sampling. |
| vLLM profiler restriction | Not applicable: this is full-model stage7; no adapter/server/integration is authorized. All48 profiling remains prohibited. |
| Shared qualitative suite | Current [v2 TT outputs](final_validation_v2/qualitative/tt_outputs.json):66 requests, six prompts, greedy plus10 sampled seeds each. Fresh independent label-hidden [comparison](final_validation_v2/qualitative/blind_comparison.json) has0/10 failures per prompt/backend; [verdict](final_validation_v2/qualitative/verdict.json) is within matched margin. [AIME direct review](aime_autoregressive_review_v2.json) finds coherent matched-control prefixes;128-token caps do not establish solved AIME answers or completed code. [Prior-candidate comparison](final_validation_v2/qualitative/prior_candidate_comparison.json) finds all66 logical/raw token rows and completions equal, independently of fresh blind scoring. |
| Reduced one-layer-per-kind profiling; separate watcher | Earlier [v1 recipe](run_terminal_final_v1.sh), [coverage](terminal_final_perf_v1/coverage.json), [analysis](terminal_profile_analysis_v1.md): layers0/1/3, C262144, natural128 prefix,72,520 reconciled keys and24 signposts. Current exact-source [v2 recipe](run_terminal_final_v2.sh), [coverage](terminal_final_perf_v2/coverage.json) and [analysis](terminal_profile_analysis_v2.json) complete68,024 reconciled operations with all expected signposts and audited advice. [Serving Watcher v2](final_serving_watcher_v2.json) is separate, never combined with profiler. |
| tt-perf-report with advice; resolve actionable rows | Current-source [tables](terminal_final_perf_v2/model_perf_report.txt), [CSV](terminal_final_perf_v2/ops_perf_results.csv) and [advice/provenance](terminal_profile_analysis_v2.json) drive the kept/rejected/blocker decisions below. [Prefill tables and semantic interpretation](prefill_profile_analysis_v1.md) are complete, with per-bucket scope and metadata caveats explicit. Remaining unmeasured ideas have explicit stop-rule handoff below; a table or proposal is not checklist completion. |
| Roofline/device/end-to-end reconciliation and perf_summary.json | Fixed useful weights+KV floor6.414528ms. Current [resident](final_resident_v1.json)103.064960ms gives6.223772%, still below10% (<=64.145280ms):38.919680ms must be removed. Current [profile](terminal_profile_analysis_v2.json)99.939984ms weighted device sum and [component estimate](component_trace_v2.json)102.507076ms stack+1.576551ms entry/terminal/sampler=104.083628ms match v4. These representative frozen-input estimates are not physical lower bounds or direct all48 sums. Resident is0.979% below the trace estimate, so there is no positive10% orchestration gap; [Bounded AutoFix](AUTOFIX_functional_floor.md) failed to restore the mandatory floor; this is not proof of physical impossibility. |
| Prefill/TTFT short, long, chunk boundaries; per-chunk kernel table | Measured128-token TTFT: [baseline](baseline_resident.json)7035.653ms, [causal/head](selected_resident_v1.json)4805.506ms, [exact bundle](exact_integrated_resident_v1.json)4700.293ms. [Final TTFT rows](final_resident_v1.json),31/32/33/128/1024 tokens:1845.539/1202.206/1304.506/4705.951/38102.021ms. These public generate boundaries include reset/setup, complete prefill, first device sample and caller read. [Six-bucket profile analysis](prefill_profile_analysis_v1.md) completes89,740 operations and advice review; all72 model+3 shared hashes match v4. Stage5 N32/N1024 remain layer-only evidence. Final-only measurements cannot manufacture before/after deltas. |
| Dominant matmul program geometry | Inherited [selected geometry](../optimized_multichip_decoder/dram_geometry_selected_v1.json) and role tables: [GR](../optimized_multichip_decoder/dram_geometry_v3_gr.csv), [GDN](../optimized_multichip_decoder/dram_geometry_v3_gdn.csv), [QSA](../optimized_multichip_decoder/dram_geometry_v3_qsa.csv), [shared](../optimized_multichip_decoder/dram_geometry_v3_shared.csv), [WO](../optimized_multichip_decoder/dram_geometry_v3_wo.csv). Stage7 head fixed-policy sweep and [completed larger-block controls](head_geometry_followup_v1.md) retain20/3/2; all measured block5/8/10 alternatives are slower, including feasible40/80-core block10 adaptations. Other small-matmul geometry work remains separate. |
| DRAM-sharded decode; Blackhole reader counts, OPT-015 | Decoder per-role defaults retained. Head1-reader blocker and2/3-reader controls are measured above; selected head20/3/2 includes input reshard, output restore/slice and sampler-canvas copy in timing. |
| Packed QKV versus split, OPT-001 | Inherited [QSA geometry](../optimized_multichip_decoder/dram_geometry_v3_qsa.json) and [precision/geometry](../optimized_multichip_decoder/dram_precision_b1_qsa.json) compare packed/split; selected packed policy unchanged. GDN exact native QKV and compact ZAB contracts retained. |
| SDPA/KV contract and reduced KV, OPT-002 | BF16/page32 retained; [real BFP8 KV trial](../optimized_multichip_decoder/real_bfp8_kv.json) passed short-layer PCC but regressed B1 by12.09%. No cache dtype/layout change, so stage7 uses32K–64K smoke, not maximum-context rerun. |
| Carry residual through norms/adds, OPT-003 | [Inherited contract](../optimized_multichip_decoder/inter_layer_residual_contract.json): one complete2560-feature BF16 TILE/DRAM stream/rank, no inter-layer CCL/conversion. Selected [exact writer](../../tt/exact_residual_write.py) and RHS [norm helper](../../tt/fused_decoder.py) preserve this layout; [source/evidence table](operation_topology_audit.md#current-exact-kernel-selection-and-evidence-boundary) documents gates and fallback. |
| Logical batch versus tile padding, OPT-005 | Persistent sampler32 rows, active slot masks, logical Q1..32 and bucketed MoE remain distinct. Current [all48 B32](final_batch32_hf_v1.json) passes HF, exact slot-migration history, dormant/all-inactive and resident sampled-feedback controls. C512 and one correlated AIME prompt are explicit limits; [fresh full48 B2 mixed/reversed/resumed prefill](final_prefill_signatures_v1.json) passes all logits exactly, starts68/31 through257/265, trace reuse, submission order and no late compilation. |
| Cumulative integrated sign-off, OPT-006 | [Current-source manifest v4](final_sources_v4.json), [exact resident](exact_integrated_resident_v1.json), [serving v2](final_serving_watcher_v2.json) and [full48 validation v2](final_validation_v2.json) pass their stated scopes. [B32 v1](final_batch32_hf_v1.json) passes HF/state controls. Final B1 resident and selected decode profile/component refresh pass; B32 throughput is complete; the strict floor is owner-accepted for this run and independent round-2 review returned `clean-pass`; prefill analysis is complete; v4 guard-only source delta is explicit above. |
| Row-parallel output decompositions, OPT-008 | Inherited [fused HiFi4 family](../optimized_multichip_decoder/fused_family_HiFi4.json), [fused LoFi family](../optimized_multichip_decoder/fused_family_LoFi.json), [whole hidden RS](../optimized_multichip_decoder/hidden_qsa_rs.json), [whole hidden AG-MM](../optimized_multichip_decoder/hidden_qsa_agmm.json). Both GR blocks/MoE consume adapted hidden layout; this is not immediate RS→AG restoration. Selected residual policy cannot be replaced here. |
| Persistent CCL buffers, OPT-009 | Decoder [async persistent](../optimized_multichip_decoder/async-persistent_l0.json) is slower than selected synchronous path. Stage7 sampler async-barrier candidate also slower, with exact clean/tracked evidence above. Existing trace-stable state remains enabled; do not force a rejected buffer/CCL policy merely to satisfy the checklist wording. |
| Packed/split gate-up, OPT-010 | Inherited [packed shared real-layer trial](../optimized_multichip_decoder/dram_packed_shared_l0_v2.json), [packed rows](../optimized_multichip_decoder/dram_packed_rows_shared_l0.json) and [shared geometry](../optimized_multichip_decoder/dram_geometry_v3_shared.json). Selected packed BFP8/HiFi4 remains. |
| Phase-specific activation shards, OPT-011 | [DRAM projection](../../tt/dram_projection.py) prepares Q1/Q2..32 policies before capture and restores BF16 DRAM. Head20/3/2 measured for Q1/Q32. [MeshExactProjection](../../tt/multichip_decoder.py) opts only GR-up/GDN-out into native Q1 R16/zero_once; Q>1 and long-prefill Q1 tails retain inherited implementation. GR-up exact widening is inside timed/captured work. |
| Precision crossed with geometry, OPT-014 | Inherited real role controls and [B1](../optimized_multichip_decoder/dram_precision_b1_qsa.csv)/[B32](../optimized_multichip_decoder/dram_precision_b32_qsa.csv) tables. Stage7 head geometry is locked to the selected BF16/HiFi4 policy. No new cross-policy search authorized. |
| Precision/fidelity one group at a time | Inherited decoder precision and physical-fit expert exception preserved. [Stage5 rejection ledger](../optimized_multichip_decoder/anomaly_ledger.md) and candidate CSV below remain authoritative. Generic BF16/BFP8 guidance does not override the user's selected-policy instruction. |
| Attention projection precision separately, OPT-007 | [Real attention BFP4](../optimized_multichip_decoder/real_qsa_bfp4.json) fails whole-output accuracy; [HiFi2](../optimized_multichip_decoder/real_dram_qsa_HiFi2_native.json)/[LoFi](../optimized_multichip_decoder/real_dram_qsa_LoFi_native.json) role evidence retained. No resurrection of faster rejected policy. |
| Real inputs for precision decisions, OPT-012 | Inherited [real fixtures](../optimized_multichip_decoder/real_prompt_fixture.json) retained; scorer inputs real, head/projection component inputs synthetic with real weights. Current [v2 HF AIME24 chat-template checks](final_validation_v2.json): prefill .97/1/1 and teacher forcing .96/1/1 top1/top5/top100, meeting required top5>=.98/top100=1. No new precision selection from synthetic controls. |
| Measured dtype/fidelity propagation, OPT-013 | Earlier [raw profile](terminal_profile_analysis_v1.json) proves selected GR FP32 weights and accurate SFPU sums, GDN BF16/FP32 products, BF16 KV/CCL, HiFi2 SDPA and HiFi4 head. Current [selected source](final_sources_v4.json) plus [component table](operation_topology_audit.md#current-exact-kernel-selection-and-evidence-boundary) retain these policies; native projection has separate multiply/sequential K fold/selected row reduce and explicit BF16 cast. [Current profiler](terminal_profile_analysis_v2.json) confirms selected native projection/writer/rank-sum kernels and unchanged precision; all72 model+3 shared hashes match v4. Inherited [42 policy rows](../optimized_multichip_decoder/measured_policy_rows.json) remain authoritative. |
| Full-model gap, LM head and sampling | Earlier [analysis](terminal_profile_analysis_v1.md) and [correct sampler traces](terminal_clean_v1.json) establish device split sampling, candidate-only gather, no generic TopK/full-vocab all-gather, split .481979ms versus force-argmax2.765952ms. Current resident full48 keeps persistent feedback/position/RoPE/page tables and no interior host boundary. [Current matched-source traces](component_trace_v2.json) estimate102.507076+1.576551=104.083628ms; measured resident103.064960ms is0.979% below it. [Current eager split](terminal_profile_analysis_v2.json) .426224ms is0.414% of resident, versus2.680506ms force-argmax; no generic TopK or full-vocabulary gather in selected split. |
| MoE routed active-expert path | E512,k10,I640 means hypothetical TP4 width160=5tiles; whole-expert EP4 retained. `deepseek_prefill→moe_fused_swiglu→combine` handles token routing, including batches; no dense all-expert or union-masked sparse runtime. Stage5 [final summary](../optimized_multichip_decoder/final_acceptance_summary.json) records diverse B32 routing and MoE shares. New precision search not authorized. |
| Final evidence and runtime audit | Current [full48 validation v2](final_validation_v2.json) runtime_audit.passed=true:12 prefill,2 decode,3 common sampling and8 complete-capture body calls, all successful; Torch/from_torch/to_torch positive controls reject, fallback throws enabled and bindings restored. Setup/input preparation/reference scoring/caller diagnostics excluded. B32 HF/control scope passes separately; final resident/decode profile pass; B32 throughput, review and strict floor remain open; prefill analysis is complete. |
| Device/profiler recovery | [Work log](work_log.md), [AutoFix](AUTOFIX.md), [upload diagnosis](upload_pinning_diagnosis.md) preserve exact faults/recovery. Commands remain bounded and serialized; no all48 profiler or concurrent benchmark regime. |

## Final-source terminal advice and current decisions

[Final analysis](terminal_profile_analysis_v2.md), [JSON/provenance](terminal_profile_analysis_v2.json),
[tables](terminal_final_perf_v2/model_perf_report.txt) and [CSV](terminal_final_perf_v2/ops_perf_results.csv)
reconcile68,024 host/device/final operations. All72 model and3 shared sources
match v4 across profiler, component traces and resident benchmark. Ordinary
operations use maximum-rank duration; collectives use rank mean. Layer
weights35/1/12 plus entry/final mixer/head/row assembly/split sampler give
99.939984ms weighted eager device work; raw per-rank totals95.322–97.415ms.
The representative frozen-input trace estimate is104.083628ms, versus
103.064960ms all48 resident. These are distinct regimes; neither eager sums
nor the representative estimate establishes a physical lower bound. Advice
has been semantically audited; original-source v1 reports remain historical.

| Advice / candidate | Disposition and exact remaining requirement |
|---|---|
| Trace high eager op-to-op gaps | Already kept in the measured model+sampler path. Persistent inputs, nonblocking replay and no per-token sync/readback are evidenced by selected resident counters and matched component traces. Eager profiler gaps are not an additional promised win. |
| LM head geometry |20/3/block2 kept; current head region approximately0.655ms; raw geometry confirms the selected configuration. BF16/HiFi4/FP32 DEST unchanged. [Larger-block follow-up](head_geometry_followup_v1.md): blocks5/8 and adapted40/80-core block10 pass Q1/Q32 but are slower in both projection and projection+split-greedy windows; [v2 CSV](head_larger_blocks_v2.csv), [v3 CSV](head_larger_blocks_v3.csv). v1 c10/b8 allocation failure was investigated and retried with feasible storage geometry, not treated as family-wide rejection. |
| Split versus force-argmax | Both correct on frozen real logits; force-argmax measured slower and rejected. Selected candidate gathers are32→128 columns, not vocabulary gathers. Current split merged kernels .426224ms are0.414% of103.064960ms token-out; force-argmax2.680506ms is slower. Historical clean trace comparisons remain separately labeled. |
| Synchronous versus sampler async-barrier CCL | Async measured slower in clean/tracked controls, synchronous kept. Unmeasured persistent-output-buffer variant is not silently classified as rejected. |
| Head8/1/1 reader configuration | Specific L1 blocker:3,108,800B static DFB exceeds1,572,864B/core. Other one-reader decompositions are untried, not disproven. |
| Generic HiFi2 / lower precision advice | Excluded by explicit user selected-policy constraint and inherited rejection ledger. This is a scope restriction, not a new benchmark rejection. Retain all accepted dtype/fidelity/KV/CCL/residual choices. |
| Input0 L1 / larger K blocks / wider output subblocks | Actionable, untried for remaining small PLE/shared/attention matmuls. Preserve fidelity; K32 cannot blindly use block_w2. Test only admissible geometry and verify real-output/full-model behavior. |
| Exact projection, norm, write and local rank-sum kernels | Exact component controls and combined [integrated resident v1](exact_integrated_resident_v1.json) now pass their scopes:103.068541ms,9.702282t/s/u with all128 caller-control tokens unchanged. Selected GR-up/GDN-output, writer, RHS norm and local rank-sum retain exact arithmetic/BF16 boundaries. Compaction mixed-prefill validation failed ([report](final_serving_watcher_v1.json)); its dispatch was reverted, prior exact-fusion candidate retained, and [restored-source serving v2](final_serving_watcher_v2.json) passes16 checks; [full48 v2 validation](final_validation_v2.json) passes accuracy/resume/context/runtime/qualitative scope. B32 HF/state controls, final resident and decode profile pass; B32 throughput passes; review finds only the failed mandatory floor; prefill analysis is complete; final useful-weight/KV fraction6.223772% still fails10%. |
| GR-down four slices plus concat | [Repaired compaction and evidence](compact_gr_down_proposal.md) tested as a candidate for eligible Blackhole1x4 BF16 [1,1,Q,384] valid_width80, Q1..32; gather unchanged. [Watcher v3](compact_gr_down_watcher_v3.json) passes allQ, exhaustive65536 patterns and changed traces with strict legacy canonicalization (exp0→+0, exp255→signedInf, other bits unchanged). [Clean v1](compact_gr_down_clean_v1.json): Q1 .040585→.006272ms; Q32 .040602→.013979ms. Historical [v1](compact_gr_down_watcher_v1.json)/[v2](compact_gr_down_watcher_v2.json) control failures retained.3.310132ms current legacy-path zero-cost ceiling is not a full-model gain. **Rejected at integrated numerical gate:** [serving Watcher v1](final_serving_watcher_v1.json) fails mixed-prefill logits row0 versus solo; clean close. Production dispatch reverted, prior exact candidate remains selected; helper retained as experimental reproduction; [reverted-source control v2](final_serving_watcher_v2.json) passes16 checks. |
| QSA post-dot reduction/selection and sparse KV gather | Untried handoff: exact score epilogue2.682108ms (whole score/selector3.777024ms) and sparse KV gather2.865048ms weighted zero-cost ceilings; not additive or predicted savings. Preserve exact head tree, ordered selected IDs, sentinel/tail semantics and persistent device-position trace behavior; no host bucket or recapture shortcut. |
| Generic/native op categorization | Raw kernel readback identifies exact SFPU projections and data movement; generic table labels do not prove utilization. Use per-kernel evidence, not invalid modeled FLOP percentages, to select the next experiment. |

Strict useful-weight/KV performance remains below10%; no overall pass is
declared. The owner confirmed the4h wall-cap Stop Rule at15:09:21UTC; untried
optimization ideas are handed off below. Correctness and required gates remain
mandatory; the stop does not waive the floor.

## Completed six-bucket prefill attribution

[Analysis and advice](prefill_profile_analysis_v1.md) and [JSON/provenance](prefill_profile_analysis_v1.json)
reconcile89,740 operations and12 signposts across logical1/2/4/8/16/32 rows.
One warmed reduced0/1/3-layer chunk per bucket yields eager device sums
8.177249/17.827547/22.579935/34.278169/53.380053/87.538737ms. These are
alternative shapes, not additive chunks or all48 TTFT. All72 model+3 shared
hashes match v4. Q32's ordered indexer24.091004ms identifies an exact-math
handoff target; generic reduction groups span roles and cannot be assigned to
GR-down alone. Some cached CSV shape descriptors show32[2] at later buckets;
driver/runtime logical counts and all-rank positions establish the actual
bucket, while the descriptor-reuse explanation remains unproven. Whole-body
markers do not support a validated layer-class partition, so the all48 device
prefill estimate remains unavailable. No fabricated48/3 scaling or subtraction
from public TTFT is used. Semantic review retains selected head/precision/CCL,
rejected compaction and tracing, with multirow/indexer/geometry work handed off.

## Multichip evidence checklist

| Requirement | Evidence / scope |
|---|---|
| Baseline and target mesh strategy | Completed full model at [stage6](../full_model/README.md), initial commit in [manifest](manifest.json). Four Blackhole chips,1x4 ring, TP4 attention/EP4 experts; no single-chip full-stack fit/performance claim. |
| Shape/config/shard/padding table | [Stage7 topology audit](operation_topology_audit.md) records embedding, residual/norm, every projection, KV/indexer, MoE, PLE, head, sampler, token/state boundaries and per-rank payloads. |
| Common-module comparison and exact mismatches | Same audit records hyperconnection/PLE/selected-QSA contracts, common sampler selected directly, TT_CCL semaphore ownership and inherited fused next-consumer trials. |
| Context recomputation / physical limit | [Context contract](../context_contract.json), [stage6 capacity](../full_model/final_capacity_reconciliation.md), [max execution](../full_model/acceptance_max_context_final_v1.json) retain262144. Current [v2 context smoke](final_validation_v2/context_smoke.json) passes32769 prompt+2 tokens, all QSA/rank positions32770 and extra replay32771, terminal logits finite;191643648B free/bank afterward. KV/layout unchanged, no maximum rerun or capability reduction. |
| Nonaligned logical sequences | Current [257-token resume](final_validation_v2/prefill_resume_check.json) passes starts32/68/256 with top1=1; history-free/fresh-suffix negative controls fail as expected where applicable. [Serving v2](final_serving_watcher_v2.json) passes mixed3/33 prompts, continuation/fixed slots; [context v2](final_validation_v2/context_smoke.json) passes32769. All48 [B32 v1](final_batch32_hf_v1.json) passes correlated prompt/history controls; [final TTFT length refresh](final_resident_v1.json) is complete; [fresh full48 B2 mixed/resumed prefill](final_prefill_signatures_v1.json) passes every check on v4. |
| Multi-chip versus single-chip PCC | Inherited [L0](../optimized_multichip_decoder/final_mesh_l0.json), [L1](../optimized_multichip_decoder/final_mesh_l1.json), [L3](../optimized_multichip_decoder/final_mesh_l3.json) retain single-chip references. Exact helpers compare selected TT arithmetic on all ranks; current [full48 HF AIME v2](final_validation_v2.json) passes required top-k gates. Current [all48 B32](final_batch32_hf_v1.json) gives2704 HF predictions/run, top1=.978180473 and top5/top100=1; no single-chip full-model fit or diverse32-prompt claim. |
| Paged cache, RMSNorm/head ownership, fixed slots | [Stage6 contract](../full_model/serving_contract.json) and [B32 pools](../full_model/batch32_hf_factored_v2.json) inherited. [Serving v2](final_serving_watcher_v2.json) passes16 state checks; current [all48 v2](final_validation_v2.json) passes resume/context/runtime audit. Local2560 RHS norms/vocab-sharded head preserve ownership. [All48 B32](final_batch32_hf_v1.json) passes shared-pool migration/dormant/inactive/resident-feedback controls at C512. Recurrent-state prefix-cache exclusion retained. |
| Warmed target-mesh traces, determinism/stress | [Exact resident](exact_integrated_resident_v1.json) has127 model+sampler replays, stable inputs and no interior refresh/read/sync. [Serving v2](final_serving_watcher_v2.json) passes changed token/position/page tables, sampling switches, inactive rows and request reuse. [Full48 v2](final_validation_v2.json) audits capture bodies; post-context extra model+sampler replay advances position without host feedback. [B32 v1](final_batch32_hf_v1.json) passes changed-slot/matched-history and resident sampled feedback controls. [Final resident](final_resident_v1.json) passes with zero interior read/sync/refresh/capture/release counters and127 model+sampler replays. |
| Runtime fallback audit | Current [full48 v2](final_validation_v2.json) runtime_audit passes guarded prefill/decode/common-sampling/complete-capture bodies, no failures, all three positive controls rejected and bindings restored. Replays execute the audited command streams; setup/reference/caller diagnostics remain outside scope. This is distinct from the reduced serving state battery. |
| Watcher separate from profiler | Current restored-source [Watcher v2](final_serving_watcher_v2.json), [log](final_serving_watcher_v2.log) passes16 reduced checks. Compaction [v1](final_serving_watcher_v1.json) failed and was reverted. Fresh reduced profiler remains separate; all48 profiling prohibited. |
| Single-chip/mesh speedup and efficiency | Inherited [stage4 comparison](../multichip_decoder/performance_summary.json), selected [stage5 summary](../optimized_multichip_decoder/performance_summary.json). Do not mix old C128 single-layer timings with C262144 full-model metrics to manufacture speedup/efficiency. |
| CCL, matmul and kernel utilization findings | Current [raw findings](terminal_profile_analysis_v2.md) attribute6.130895ms weighted CCL transport, distinct from local reduction/clone/fill. Keep synchronous collectives and selected precision; native rank sum changes only local arithmetic. Exact-source decode profiling is complete; matched72 model+3 shared hashes and raw runtime geometry retain the selected policy. [Head follow-up](head_geometry_followup_v1.md) rejects larger blocks by measured latency, not modeled utilization. |
| Expert replication / largest-batch behavior | EP4 whole experts retained; stage5 diverse B32 expert unions and grouped route path documented. Current [all48 B32](final_batch32_hf_v1.json) HF/state checks pass, including exact history after migration and inactive preservation. One correlated source prompt; no routing-diversity or serving-throughput claim beyond the artifact. |
| Independent review and local commit | Stage5 [clean review](../optimized_multichip_decoder/stage_review_round1.md), stage6 [clean review](../full_model/stage_review_round1.md) are inherited only. [Stage7 independent review](stage_review_round1.md) finds only the failed mandatory floor; bounded AutoFix failed. Local checkpoint SHAs are recorded in work_log.md and checkpoint_commits.log. Never push. |

## Inherited rejection ledger links

The full [candidate CSV](../optimized_multichip_decoder/candidate_summary.csv),
[anomaly ledger](../optimized_multichip_decoder/anomaly_ledger.md),
[stage5 checklist](../optimized_multichip_decoder/optimization_checklist.md) and
[stop-rule handoff](../optimized_multichip_decoder/stop_rule_handoff.md) retain
untried versus rejected distinctions. Additional exact records:

- BF16 CCL retained: [global async BFP8](../optimized_multichip_decoder/async-bfp8_l0.json), [MoE subgroup](../optimized_multichip_decoder/real_ccl8_moe.json), [QSA subgroup](../optimized_multichip_decoder/real_ccl8_qsa.json).
- BF16 activations retained: [shared](../optimized_multichip_decoder/real_activation8_shared_l0.json), [QSA](../optimized_multichip_decoder/real_activation8_qsa_l3.json), [WO](../optimized_multichip_decoder/real_activation8_wo_l3.json).
- Sensitive GDN/shared fidelity: [GDN HiFi2](../optimized_multichip_decoder/real_dram_gdn-compact_HiFi2_native.json), [GDN LoFi](../optimized_multichip_decoder/real_dram_gdn-compact_LoFi_native.json), [shared BFP4/LoFi](../optimized_multichip_decoder/real_dram_shared_LoFi_bfloat4_b.json). The latter is stage8 candidate evidence, not a stage7 default.
- Native AR: [persistent AR](../multichip_decoder/mesh_l0_persistent_ar.json), [residual contract](../multichip_decoder/residual_contract.json), [AutoFix](../multichip_decoder/AUTOFIX.md). Historical per-request misses are not silently promoted into a current whole-output failure.

## Evidence commands and remaining final gates

The commands below document completed resident, full48 validation, B32 HF/control and mixed-prefill runs; they are not requests for further experiments after the confirmed wall cap. Current terminal v2 profiling also completed. Preserve all existing artifacts; any separately authorized rerun requires fresh paths and serialization after sourcing
[environment.sh](environment.sh), with fresh output names. The examples preserve
all48 and keep profiling restricted to three representative layers.

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/environment.sh
stage7_doc=models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model

# Completed PASS: primary before/after counterpart, resident boundary, TTFT shapes.
TT_METAL_TRACE_ALLOC_TRACKING=0 timeout 5400 python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_full_model_resident \
  --public-resident --ttft-lengths 31,32,33,128,1024 \
  --output "$stage7_doc/final_resident_v1.json"

# All48 AIME prefill/teacher forcing, retained-state resume, autoregression,
# shared qualitative suite, nonaligned32769 context smoke. No maximum rerun.
# v2 completed PASS: preserve the existing report; any rerun needs a new output path.
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 7200 python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_full_model_validation \
  --accuracy --resume --autoregressive --qualitative --context-smoke \
  --output "$stage7_doc/final_validation_v2.json"

# Full48 B32 v1 completed PASS: preserve this report; reruns require a new path.
# Shared-pool HF/control evidence, retained traces and feedback.
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 7200 python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_batch_hf \
  --batch-size 32 --capacity 512 --output "$stage7_doc/final_batch32_hf_v1.json"

# Completed PASS on v4: full48 mixed/resumed prefill and submission contract.
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 3600 python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.probe_prefill_trace_signatures \
  --all-layers --shared-kv --max-batch-size 2 --check-decode-submission-order \
  --output "$stage7_doc/final_prefill_signatures_v1.json"

# Historical serving rerun template, not an instruction to start a new job.
# Completed restored-source Watcher evidence is final_serving_watcher_v2.json.
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 1800 python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_serving \
  --output "$stage7_doc/final_serving_probe_v1.json"

# v1 historical; v2 completed with matched final v4 sources and audited advice.
# The recipe refuses existing output paths. Device owner serializes this job.
# timeout 5400 bash "$stage7_doc/run_terminal_final_v2.sh"
```

The serving battery is a trace/state test; its name does not imply that it
enables the runtime fallback guard. The completed [stage7 validation runner](../../tests/run_optimized_full_model_validation.py) guards prefill, decode and sampling/capture bodies with host-conversion bans, TorchFunctionMode and throw-on-fallback, including positive controls. Its current [v2 report](final_validation_v2.json) shows all four body categories executed (12/2/3/8 successful calls), zero failures, positive-control rejection and restored bindings. This establishes complete-body audit for the [selected source](final_sources_v2.json); earlier v1 remains historical. Intentional correctness reads remain outside the guard. The causal component's guarded pass alone is insufficient.

The reduced profiler recipe currently measures **decode** attribution, not a
prefill profile. Final warmed prefill/bucket kernel attribution was collected separately by [run_prefill_profile_v1.sh](run_prefill_profile_v1.sh):89,740 operations over six buckets; [CPU analysis and semantic advice](prefill_profile_analysis_v1.md) are complete. Reuse inherited prefill evidence only for unchanged
components and state that scope. Matching-capacity layer baselines and the
weighted35/1/12 reconstruction commands are in
[performance_accounting_plan.md](performance_accounting_plan.md).

The installed stage7 hard gate was located and inspected at
[`07-optimized-full-model.check.sh`](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/prompts/model_bringup_multigoal/07-optimized-full-model.check.sh).
It runs only the model-scoped autoregressive degeneracy checker, with missing
artifacts critical. It does not prove performance, accuracy, trace safety,
context, qualitative quality or checklist completion. Run it after new stage7
outputs exist, and retain the direct new-artifact check from the validation
runner so inherited stage6 output cannot substitute for stage7 evidence.

```bash
stage7_skill_root=/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24
MODEL_DIR="$PWD/models/autoports/qwen_qwen3_8_flash_next" \
  timeout 120 bash "$stage7_skill_root/prompts/model_bringup_multigoal/07-optimized-full-model.check.sh" \
  > "$stage7_doc/stage07_check.log" 2>&1
```

Current disposition: bounded AutoFix did not meet the numerical floor; B32 clean throughput passed; the owner accepted the floor shortfall on2026-11-16. Preserve the source checkpoint and numerical evidence; retain the round-2 `clean-pass` and log the closure SHA. B1 resident/decode accounting is complete. Larger head blocks are measured, not untried. No new optimization experiments are authorized after the owner-confirmed cap; required gates remain mandatory.

## Not tried (stop rule)

Stage start11:09:21UTC; the owner confirmed the default4h wall cap at
15:09:21UTC and recorded it in the stage budget/work log. This is the stop
trigger; last-two-kept<3% is not established. The latest kept exact bundle
improved latency9.495%; failed compaction and slower head trials are not kept
candidates. The10% FUNCTIONAL floor remains failed and is not waived.

Expected gains below are unknown. Contained-operation zero-cost ceilings
come from the matched-source [v2 candidate budgets](terminal_profile_analysis_v2.json),
overlap and cannot be added or treated as achievable savings. The final gap
to the fixed64.145280ms target is38.919680ms, larger than any listed individual
ceiling. No graph inflation or precision-policy change enters the denominator.

| Untried extension | Expected gain / bound | Required constraint or source |
|---|---|---|
| Exact QSA score epilogue/head-reduce/ordered-selection fusion | Unknown; epilogue ceiling2.682108ms, whole score/selector3.777024ms | [Scorer](../../tt/causal_indexer_scores.py), [multichip decoder](../../tt/multichip_decoder.py); preserve head tree, ties, sentinels and device-owned positions. |
| Sparse selected-KV gather optimization | Unknown; family ceiling2.865048ms | [Fused decoder](../../tt/fused_decoder.py); preserve paging, selected-ID order and nonaligned/tail semantics. |
| Further exact QSA indexer / GDN QKV / recurrence scheduling | Unknown; respective family ceilings5.260452/4.764354/2.553868ms | [Ordered projection](../../tt/ordered_projection.py), [multichip decoder](../../tt/multichip_decoder.py); no reassociation, lower fidelity or precision search. |
| Different exact GR-down streaming schedule beyond rejected R2/R4/R8/R16 | Unknown; whole local-down ceiling11.455886ms, product/reduce subset10.354546ms | [Exact projection candidate](../../tt/exact_projection_q1.py); previous schedules slower; new design needs raw FP32 exactness and real integrated controls. |
| Small PLE/shared/attention matmul input-L1/block/subblock configurations | Unquantified; no defensible projected gain | [DRAM helper](../../tt/dram_projection.py), selected stage5 role tables; preserve fidelity and legal K divisibility. Larger head blocks and8/1/1 are measured/rejected or blocked. |
| Safe persistent-output sampler gather | Unknown; current entire eager split .426224ms is a contained-work ceiling, not a trace saving forecast | [Collective probe](../../tests/probe_sampling_collective_policy.py); factory persistent output disables the replay barrier. A safe distinct implementation needs changed-winner/tie replay validation; async-barrier candidate already slower. |
| Further prefill scheduling/chunk/layout optimization | Unquantified; [completed six-bucket analysis](prefill_profile_analysis_v1.md) identifies Q32 ordered indexer24.091004ms within the reduced body, not an all48 or achievable saving | [Generator](../../tt/generator.py); preserve six canonical traces, mixed/fixed/inactive slots, odd lengths and explicit state. Multirow exact projection/indexer scheduling must retain reduction trees; no full48 scaling of reduced kernel groups. |

Policy-rejected lower precision/KV/CCL and alternate replicated/hidden-sliced
residual layouts remain excluded by the user contract and inherited ledger.
Compaction is a measured integrated numerical rejection, not an untried item:
the minimal mixed-prefill reproduction also failed. Conditioned eager real-operand
comparisons passed, but do not resolve native-only serving-trace failure or
establish its root cause. The helper is experimental and dispatch remains
reverted. Reopening requires repaired full serving controls, not isolated speed.
