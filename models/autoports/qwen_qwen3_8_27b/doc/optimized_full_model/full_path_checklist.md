# Stage 7 full-path optimization checklist

The hardware, host, correctness, performance, context and qualitative evidence gates below are complete for the selected Qwen/Qwen3.8-27B TP4 implementation. **Independent [stage review](stage_review.md) returns clean-pass; checkpoint provenance is in [work_log.md](work_log.md).** This checklist is a source-and-artifact audit, not the independent completion verdict. Only this Markdown file was edited during its finalization; no device commands or implementation edits were performed by this investigator.

The Stage5 decoder policy and rejection ledger remain fixed. Stage7 selects four-chunk BFP8/HiFi2 head geometry, traced prefill with one bounded cache, deferred token delivery, and8192-byte fabric payload while preserving topology, links, residuals, KV/CCL dtypes and the262144 context contract.

Paths are relative to `models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model/` unless explicitly identified as repository paths. **F** denotes [`tracy/profile_final_buffers/`](tracy/profile_final_buffers/), the final reduced real layers0/3 plus embeddings, terminal path, sampling and history; it uses cache capacity256, page table `[1,8]` and history capacity127. [`perf_summary.json`](perf_summary.json) confirms its required runtime source/binary hashes match [`after_full_final.json`](after_full_final.json). **P** denotes the retained eager-prefill control [`tracy/before_prefill_trace/`](tracy/before_prefill_trace/). Full64 timings are unprofiled; reduced device intervals and expanded estimates retain their distinct scopes.

## Inherited selected policy and rejection ledger

| Contract / decision | Exact source and evidence | Disposition in this stage |
| --- | --- | --- |
| Selected decoder policy | [`../../tt/multichip_decoder.py`](../../tt/multichip_decoder.py), `from_state_dict`, seeded from `DEFAULT_POLICY` in [`../../tt/optimized_decoder.py`](../../tt/optimized_decoder.py); [`../optimized_multichip_decoder/final_default_policy.json`](../optimized_multichip_decoder/final_default_policy.json), [`cumulative_contract.md`](../optimized_multichip_decoder/cumulative_contract.md) | Preserve BF16 activations/residual/norms/CCL, BFP4/LoFi projections with FP32 accumulation, BFP8 KV, FP32 recurrence and BF16 convolution history. The full-attention kind selects its recorded short-prefill overrides in source; the standalone policy JSON illustrates the linear-kind values. |
| Final measured default | [`../optimized_multichip_decoder/after_review_l0.json`](../optimized_multichip_decoder/after_review_l0.json), [`after_review_l3.json`](../optimized_multichip_decoder/after_review_l3.json), [`after_review_stack.json`](../optimized_multichip_decoder/after_review_stack.json), [`README.md`](../optimized_multichip_decoder/README.md) | B1/S128 traced decode 0.421990/0.306304/0.704825 ms for linear/full/stack. These include the isolated replay/synchronization contract; they are not a full-stack device kernel floor. |
| Coherent topology alternatives | [`../optimized_multichip_decoder/collective_contracts.md`](../optimized_multichip_decoder/collective_contracts.md), [`topology_family_results.json`](../optimized_multichip_decoder/topology_family_results.json), [`topology_matrix.json`](../optimized_multichip_decoder/topology_matrix.json), [`topology_remaining_matrix.json`](../optimized_multichip_decoder/topology_remaining_matrix.json) | Native AR selected. Replicated RS/AG about 0.744–0.747 ms; carried hidden-1280 residual with fused norm about 0.775 ms; adapted row AGMM/MMRS and column AGMM lose. Sharded comparison gather is outside timing. These are genuine adapted-family rejections, not a reason to repeat a topology search. |
| CCL dtype, links, buffers and residual/core grids | Same collective ledger; [`candidate_summary.csv`](../optimized_multichip_decoder/candidate_summary.csv), [`coherent_ab_matrix.json`](../optimized_multichip_decoder/coherent_ab_matrix.json), [`inter_layer_contract.md`](../optimized_multichip_decoder/inter_layer_contract.md), [`inter_layer_profile_audit.json`](../optimized_multichip_decoder/inter_layer_profile_audit.json) | Preserve axis 1 Ring/two links, BF16, shared persistent AR workspace, 40-core B1 residual. Ring one/two links, Linear one/two links, persistent on/off, BF8 communication and 10/20/40/80 residual/core alternatives are indexed in the measured families. No layer-boundary gather/reshard/reduction is added. |
| Projection packing and precision | [`candidate_summary.csv`](../optimized_multichip_decoder/candidate_summary.csv), [`precision_matrix.json`](../optimized_multichip_decoder/precision_matrix.json), [`precision_remaining_matrix.json`](../optimized_multichip_decoder/precision_remaining_matrix.json), [`topology_family_results.json`](../optimized_multichip_decoder/topology_family_results.json) | Preserve packed QKVZBA/QKVG and packed gate/up. Adapted split attention, separate gate/up and fused SwiGLU controls lose. Attention/output/gate/down precision and fidelity comparisons are inherited, not reopened. |
| Matmul geometry and native contracts | [`matmul_geometry_search.csv`](../optimized_multichip_decoder/matmul_geometry_search.csv), [`projection_summary.csv`](../optimized_multichip_decoder/projection_summary.csv), [`final_matmul_rows.csv`](../optimized_multichip_decoder/final_matmul_rows.csv), [`native_fix_source_review.md`](../optimized_multichip_decoder/native_fix_source_review.md), [`AUTOFIX_dram_mesh.md`](../optimized_multichip_decoder/AUTOFIX_dram_mesh.md) | Attention/output/gate/down input cores 10/8/40/8; K blocks 16/6/4/17; readers per bank 2/2/3/2. Native descriptor placement and padding-only reader fixes are inherited. Output subblocks are factory-derived, not a missing Python override. |
| KV, SDPA, GDN and prefill | [`kv_matrix.json`](../optimized_multichip_decoder/kv_matrix.json), [`kv_remaining_matrix.json`](../optimized_multichip_decoder/kv_remaining_matrix.json), [`sdpa_matrix.json`](../optimized_multichip_decoder/sdpa_matrix.json), [`sdpa_confirmation_matrix.json`](../optimized_multichip_decoder/sdpa_confirmation_matrix.json), [`prefill_matrix.json`](../optimized_multichip_decoder/prefill_matrix.json), [`prefill_remaining_matrix.json`](../optimized_multichip_decoder/prefill_remaining_matrix.json), [`advice_closure.md`](../optimized_multichip_decoder/advice_closure.md) | Preserve BFP8 cache and explicit SDPA grids/chunks; BFP4 KV discrepancy and adapted monolithic/serial GDN results are documented. Preserve 4096 internal prefill chunks and the short 1D/long minimal-or-2D choice. Stage 5 caller-owned prefill-trace results do not establish generator-owned Stage 7 trace correctness. |
| Anomalies and final validation | [`anomaly_ledger.md`](../optimized_multichip_decoder/anomaly_ledger.md), [`final_validation_index.json`](../optimized_multichip_decoder/final_validation_index.json), [`stage_review.md`](../optimized_multichip_decoder/stage_review.md), [`performance_accounting.json`](../optimized_multichip_decoder/performance_accounting.json), [`work_log.md`](../optimized_multichip_decoder/work_log.md) | Preserve passing source/installed-library provenance, repaired watcher teardown, strict PCC/state gates and measured host-pool selection. Historical failed/adapted runs remain distinguishable from accepted results. |

## Full-model path coverage

| Boundary | Selected source / program contract | Final evidence and disposition |
| --- | --- | --- |
| Embeddings and RoPE | `QwenModel.embed/rope` in [`model.py`](../../tt/model.py): BF16 embedding weights partition hidden across four ranks; device lookups, one entry async hidden gather, axis1/Ring/two links; persistent decode RoPE indices | F device0: three embedding kernels3.209 us, entry gather13.876 us; entry-and-RoPE segment65.124 us including incoming gaps. Final benchmark and B32 contract show zero steady token/position/RoPE refreshes. Included in the final profile and full timing; no new embedding-specific layout sweep or rejection is claimed. |
| Decoder stack, CCL and residual | `QwenModel.prefill/decode` directly calls selected `MultichipDecoder`; B1 replicated hidden in40-core L1 WIDTH_SHARDED storage, B2–32 public DRAM boundaries; shared persistent AR workspace | F verifies BFP4/LoFi decoder, BF16 collective payload and four internal ARs for two layers (48.237 us kernels on device0); no added boundary gather. [`perf_summary.json`](perf_summary.json) confirms decoder source equals Stage5. [`contract_final_full_b32.json`](contract_final_full_b32.json) passes on final source. |
| Final norm / head input | `QwenModel.logits(decode=True)`:40-core grid10×4, shard `[32,128]`, RMSNorm blockH1/W4/subblockW4, non-inplace; then eight-core `[32,640]` head input | F device0 final norm row1398 is6.453 us on40 cores. Inherited norm selection remains in [`../full_model/performance.md`](../full_model/performance.md). The recorded-activation head probe does not independently measure an alternative norm grid. |
| LM-head geometry / numerics | `QwenModel.__init__/_dram_logits`: local vocab62080; BFP8 weights, BF16 activation/output, HiFi2/FP32; chunks16384/16384/16384/12928; eight DRAM banks, two readers, K block5, M1, bankN64/64/64/52 | F device0 confirms four BFP8/HiFi2/K5 matmuls,888.241 us kernels. Final full-model timing and readiness pass. Geometry controls and the real-activation LoFi rejection are recorded below. |
| Logits padding and layout | Four sharded head outputs convert to DRAM and concatenate locally; physical bank padding does not expose logical IDs beyond62080. Common sampler receives true vocabulary metadata and masks invalid IDs before TopK when needed | F includes every output conversion/concat; head controls preserve logical output and local greedy IDs. Power-of-two logits padding was measured slower; selected `pad_logits_to_power_of_2=False`. No full-vocab gather is present in selected token-out execution. |
| Canonical sampler | [`generator.py`](../../tt/generator.py) uses common `models/common/sampling/tt_sampling.py` `TTSampling`; physical local top32, small candidate gathers, semantic greedyk1/p0/T1; sampled k1–32/top-p preserved; `tt_out_tok=self.tokens` | F device0 large-index TopK is266.254 us on110 cores; route prep/TopK/finish total310.734 us. Only two `[1,1,32,32]` to `[1,1,32,128]` candidate gathers,28.680 us kernels; no generic TopK/ArgMax. Final readiness and qualitative gates below pass. Native gather routing comes from configured fabric; deprecated sampler link arguments do not prove useful two-link scheduling for tiny candidate tensors. |
| Device feedback and complete delivery | `_model_step/_sampling_step` update persistent token/position/RoPE/seed state; `_append_history` uses UINT32 indexed-fill/copy/cursor increment; generation performs a final history read, callbacks retain immediate output | [`after_full_final.json`](after_full_final.json):127 model/sample replays and history appends, one final history read, exact all128-token equality to immediate delivery. F history append is5.470 us kernels plus2.470 us gaps. Decode timing includes final delivery. Queued final-token checking remains outside its diagnostic timing. |
| Cache, slots and inactive users | `allocate_cache`, `_ensure_cache`, `bind_cache`, `prefill_forward`, `decode_forward`; ceil(capacity/32) mapping, BFP8 local KV, caller-owned page tables, FP32 recurrence/BF16 convolution history, inactive-state preservation | [`contract_final_full_b32.json`](contract_final_full_b32.json), exit0: all64 layers,32 slots, mixed31/33 prompts in slots31/0, continuation PCC0.999214470, exact physical-page remapping, inactive state, all32 history lanes, sampled/greedy switching and zero steady refreshes. F SDPA runtime K/V dtypes are BFLOAT8_B. B32 short context and B1 maximum context are separate coverage points. |
| Decode trace / host boundary | `_capture` owns model/sample and eligible prefill traces; snapshots/restores mutable state/history cursor; replay uses `blocking=False`; cache/mode/signature changes invalidate relevant storage | Final full B32 and prefill contracts guard device-only capture/replay and persistent identity; F marks all token-out rows as trace replay. Explicit readiness logits, teacher forcing and host-sampling compatibility retain intentional host work; none is an automatic runtime fallback. |
| Prefill tracing / dispatch | One retained trace for eligible single-request prompt up to4096; persistent token/position/output buffers and same bound cache/page table; public eager prefill remains independent; longer prompts use existing4096 chunks | [`prefill_contract_full.json`](prefill_contract_full.json), exit0, covers1/31/32/33/4095/4096/4097, exact eager/traced tokens, changed prompts/modes,50 guarded captures and34 device-only prefill calls, physical KV ownership and public output lifetime. Final full TTFT59.295 ms. F device0 prefill3049.896 us kernels +733.854 us gaps versus P3062.471+3517.695 us; these are labeled reduced before/final profiles. |
| Context and persistent resources | [`../context_contract.json`](../context_contract.json) `optimized_full_model.status=validated`; one trace cache,4MiB prefill reserve, request-sized UINT32 history high-water storage and scratch; no public alignment requirement | [`readiness_final.json`](readiness_final.json) executes full64 prompt262143 plus last-position decode, and prompt262144; current position262144. Conservative per-device total13,538,275,840B versus34,138,688,512B DRAM,20,600,412,672B headroom. Prefill-trace4096 bound limits trace retention, not supported context. |

## Final correctness, ownership and qualitative gates

| Gate | Exact evidence | Result / scope |
| --- | --- | --- |
| HF readiness / AIME-derived token agreement | [`readiness_final.json`](readiness_final.json), `.exit_status=0` | Prefill top1=99/100, top5=100/100; traced decode top1=98/100, top5=100/100; top100=100/100 in both. This is100-token agreement against the pinned chat-template HF reference, not an AIME answer-solving score. Readiness diagnostic throughput is separate from final warmed benchmark. |
| Full prefill identity, changed inputs and ownership | [`prefill_contract_full.json`](prefill_contract_full.json) | All seven logical-length boundaries pass; changed prompt/sampling modes, exact changed-table KV writes, public outputs surviving other requests, cache/cursor/trace restoration, cold/cachedG1 andG0 no-work checks pass. Timings under guards are not performance evidence. |
| Full32-user state contract | [`contract_final_full_b32.json`](contract_final_full_b32.json) |64 layers, mixed fixed slots, inactive state, page mapping, continuation, histories, no-fallback guard and zero steady refreshes pass. |
| Watcher and trace allocation tracking | [`watcher_prefill_final.json`](watcher_prefill_final.json), [`environment`](watcher_prefill_final.environment.json), `.exit_status=0` | Real layers0/3 on all four chips, S33, changed-input/mode/page/lifetime checks; watcher10, trace allocation tracking1, NOINLINE1/fabricO3, no ETH disablement, no profiler.21 guarded captures and19 device-only prefill calls. Full-model tests and representative watcher are distinct complementary runs. |
| Qualitative and degeneracy | [`qualitative_metrics.json`](qualitative_metrics.json), [`tt_qualitative.json`](tt_qualitative.json), [`tt_qualitative_extended.json`](tt_qualitative_extended.json), [`tt_qualitative_story_2048.json`](tt_qualitative_story_2048.json) | Checker exit0, no findings, six prompts complete with EOS at418/148/1700/380/145/337 tokens. Generated Fibonacci code tests pass inputs−1/0/1/2/8. The story extension preserves the first1024 tokens and completes at1700; TT2048 completion is not a matched-budget quality comparison with the1024-token HF control. |

## Stage 7 measured candidate decisions

Head microtraces use the same real head weights and recorded layer0/3 activation, 128 queued warmed replays, and include output conversions/concat. The geometry rows retain BFP8 weights, BF16 activations, HiFi2 and FP32 accumulation; the separate LoFi fidelity control below was rejected. They are component measurements, not full64 token-out gains.

| Chunk / block / readers | Head us | Evidence / decision |
| --- | ---: | --- |
| 8192 / 10 / 2 | 999.725 | [`head_baseline_fixed.json`](head_baseline_fixed.json), baseline |
| 4096 / 20 / 2 | 1014.728 | [`head_4096_block20.json`](head_4096_block20.json), correct but slower |
| 16384 / 5 / 2 | 949.427 | [`head_16384_block5.json`](head_16384_block5.json), selected; minimum reported local logit PCC0.999993205, all local greedy outputs equal |
| 8192 / 10 / 1, compatible padding | 1150.478 | [`head_8192_reader1_recovered.json`](head_8192_reader1_recovered.json), correct but slower |
| 16384 / 10 / 3, compatible padding | 1174.401 | [`head_16384_block10_reader3_compatible.json`](head_16384_block10_reader3_compatible.json), correct but slower |
| 16384 / 10 / 2 | No accepted timing | [`head_16384_block10_reader2.log`](head_16384_block10_reader2.log), 101760-byte L1 overlap; legal larger-chunk block5 alternative selected |

The initial reader1 PCC failure used an incompatible bank stride. Reader3 trials also required reader-dependent tail padding; those initial failures do not reject the corrected reader families. [`AUTODEBUG_head_geometry.md`](AUTODEBUG_head_geometry.md) and the compatible measurements close that distinction. [`AUTOTRIAGE_head_startup.md`](AUTOTRIAGE_head_startup.md) concerns an intervening initialization transfer stall before candidate execution and supplies no head-geometry verdict.

| Sampler control, same reduced S128/G128 harness | Queued token-out t/s | Decision |
| --- | ---: | --- |
| Split greedy, no power-of-two pad | 441.105 | [`before_reduced.json`](before_reduced.json), retained sampler contract |
| Split greedy, power-of-two pad | 433.405 | [`padded_split_reduced.json`](padded_split_reduced.json), token equality passes, slower |
| Common force argmax | 240.305 | [`argmax_reduced.json`](argmax_reduced.json), repeat/final-token checks pass, slower |

The retained unpadded sampler already reaches parallel large-index TopK on this Blackhole checkout. The profile and measured padded control supersede a generic assumption that non-power-of-two width necessarily means a slow one-core route.

The real-activation LoFi control [`head_16384_block5_lofi_report.json`](head_16384_block5_lofi_report.json) keeps chunk16384/block5/readers2 and BFP8 weights, but is **rejected**: its fourth rank has active-row logit PCC0.998926342 below the head probe's0.999 floor. Other ranks pass and all local greedy IDs agree; those facts do not waive the PCC failure. Its diagnostic942.765 us timing is not an accepted result. The selected head remains HiFi2; this local fidelity control does not change the inherited decoder policy or claim a full-model datatype frontier.

| Fabric payload comparison | Reduced deferred t/s | Full64 deferred t/s | Decision |
| --- | ---: | ---: | --- |
| Default4352 | 447.125 | 39.583 | [`prefill_trace_reduced_perf.json`](prefill_trace_reduced_perf.json), [`prefill_trace_full.json`](prefill_trace_full.json), retained controls |
|8192 | 450.484 | 40.381 | [`fabric8192_reduced.json`](fabric8192_reduced.json), [`fabric8192_full.json`](fabric8192_full.json), selected after exact eager/traced and immediate/deferred token comparisons |

The full candidate's deferred TTFT is59.725 ms. The final default run below reproduces40.385 t/s with59.295 ms TTFT. This packet-only comparison retains TP4 Ring, link count, all dtypes, residual layout and decoder program policy. The measured full decode improvement is about2.0% versus the4352 control; the reduced result alone was not used to select it.

## Final full-model timing and accounting

| S128/G128/B1 full64 measurement | Warmed TTFT ms | Decode t/s/user | Scope |
| --- | ---: | ---: | --- |
| [`before_full.json`](before_full.json), immediate | 86.312 | 39.433 | Original Stage7 full-delivery control |
| [`after_full_final.json`](after_full_final.json), immediate | 58.852 | 40.328 | Final selected default, immediate token delivery |
| Same final run, deferred | 59.295 | 40.385 | Primary complete-delivery result; final history read included, all tokens equal immediate |
| Same final run, queued | 58.594 | 40.394 | Final-token check outside timing; diagnostic, not a complete-delivery claim |

Final S203/G100 teacher forcing records77.233 ms TTFT and40.238 t/s/user with sampling/token delivery. It intentionally refreshes forced input tokens and reads outputs. [`after_terminal_full.json`](after_terminal_full.json) remains the earlier head/history-only intermediate control; its83.013 ms/39.589 t/s deferred values are not the final selected result.

[`perf_summary.json`](perf_summary.json) reconciles F, its same-run host signposts and the full unprofiled benchmark with explicit scope boundaries:

| Accounting item | Result | Interpretation |
| --- | --- | --- |
| Full complete-delivery decode |24.761666 ms/token | Direct full64 measurement;40.385 t/s/user |
| Reduced same-profile read lower bound |0.879850 ms/token |450,483,200B/rank minimum projection/head/KV reads at nominal512GB/s |
| Reduced same-profile token-out device span |2.274247–2.279497 ms | Each device independently, kernel plus incoming gaps; never sum the four devices |
| Matching reduced host signpost |2.712827 ms | Same combined token-out window; host minus device0.433330–0.438580 ms in this instrumented reduced run |
| Full required-read estimate |7.671370 ms/token mean | Mean3,927,741,222.3B/rank over active context129–255, includes stored projection/head tiles and unique required KV tiles only |
| Expanded full kernel estimate |23.756443–23.883777 ms |48 linear +16 full representative kernel contributions plus entry/terminal/sample/history once; not a measured all64 device interval |
| Expanded kernels plus profile gaps |26.444203–26.556487 ms |Instrumentation and non-additive execution mean this can exceed the unprofiled full latency; the difference is not negative host overhead |

The full estimate's decoder projection reads are3,586,129,920B/rank and selected head reads339,804,160B/rank. The lower bound excludes recurrence, embeddings/constants, activations, writes, extra KV reads, CCL and dispatch; nominal bandwidth is not an achievable full-model target. `decode_ms_per_token_device` is intentionally null because full-stack profiling is excluded. The0.878–1.005 ms full-latency-minus-expanded-kernels residual is a mixed-run estimate, not measured host overhead.

F's independently drained device0 model/sample windows are1.756760/0.498176 ms; use the direct combined2.274247 ms window for matching host accounting instead of adding them. The selected terminal segment is0.967417–0.984427 ms, sampling without history0.497283–0.511657 ms and history0.007901–0.007953 ms across devices. Subgroups overlap their parent and are not added twice.

Required runtime source and native-library hashes match between final profile and benchmark. One recorded `tests/run_full_model.py` runner entry differs; `all_recorded_sources_match=false` is retained, while `all_required_runtime_sources_match=true`. F explicitly records cache256/page-table `[1,8]`/history127 to match persistent geometry. Layer count, prompt data and exact context evolution still differ, as the accounting limitations state. Original profile/control artifacts and manifests are preserved.

The perf tool's eight-core DRAM matmul denominator remains misleading for two/three readers. F's parsed runtime attributes record16/24 native workers and actual BFP4/LoFi decoder, BFP8/HiFi2 head and BFP8 KV; no utilization/saturation claim uses the incorrect displayed core/FLOPs percentage. Stored bank/tile bytes are computed independently of logical-byte DRAM percentages.

## Native implementation and warning disposition

| Concern | Source / evidence | Final disposition |
| --- | --- | --- |
| DRAM matmul readers and blocks | `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`, `device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`; Stage5 native ledger and [`../full_model/AUTOFIX_dram_head.md`](../full_model/AUTOFIX_dram_head.md) | Existing coordinate placement, padding-only reader and packet-splitting repairs inherited. Stage7 changes Python head geometry; no new native repair/build is claimed. |
| Norm / TopK / sampler native routes | `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp`; `operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp`; `operations/reduction/topk/device/topk_route_prep_program_factory.cpp`; `operations/reduction/sampling/device/sampling_device_operation.cpp` | Final runtime rows verify the intended sharded norm, parallel large-index TopK and native sampling. No custom host sampler or generic TopK/ArgMax is selected. |
| Fabric4352/8192 advice | `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:39–70` computes `min(15232/page_size,4)*page_size` on Blackhole, hence8192 for2048-byte pages | Static advice was tested with compatible reduced and full controls;8192 selected and final default reproduced. This does not reopen the decoder CCL dtype/topology/residual frontier. |
| L1 semaphore advisory | `operations/ccl/all_gather/device/all_gather_multicast_factory.cpp:34–45` allocates L1_SMALL when reserved, otherwise warns about L1 fragmentation | A potential capacity advisory, not observed corruption or failed allocation. Selected head geometry, full B32/context and tracked watcher pass; no claim of a measured L1_SMALL optimization or required reservation change. |
| Allocation with live traces | `tt_metal/impl/allocator/allocator.cpp` warns of a possible lifetime conflict | Actual final tracked prefill/ownership and watcher results validate the exercised storage lifecycle. A generic untracked warning is neither proof of corruption nor the validation evidence; new persistent buffers/traces are governed by the model lifecycle. |
| Instrumented Ethernet kernel size | Earlier [`watcher_contract_b3.log`](watcher_contract_b3.log) fails before loading with29104B>26624B; successful fit controls and final watcher retain NOINLINE/fabricO3 | Final watcher exits0 with all checks retained. Failed JSON/logs are preserved and excluded from acceptance; no ETH disablement or overlapping profiler run. |

`configure_fabric()` in [`generator.py`](../../tt/generator.py) selects the supported router payload before opening the same1×4 Ring mesh:

```python
router = ttnn.FabricRouterConfig()
router.max_packet_payload_size_bytes = 8192
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router)
```

Binding is `ttnn/cpp/ttnn-nanobind/fabric.cpp:121–141`; `tt_metal/fabric/fabric_context.cpp:442–460` requires a positive L1-aligned payload no larger than15232B on Blackhole. The4352 static assertions constrain the default, not this override. Larger payload changes channel storage/credits, which is why full controls and final validation accompany source legality. The benchmark retains `--fabric-payload-bytes 4352` for the original control.

## Completion gates

- [x] Preserve selected Stage5 decoder policy and exact rejection ledger; final runtime rows and source hash verify the inheritance.
- [x] Select head geometry and canonical sampler from real-weight/activation controls; retain the explicit LoFi PCC rejection.
- [x] Validate deferred history, persistent feedback and complete delivery; final full64/B32 state and no-fallback checks pass.
- [x] Complete bounded generator prefill tracing with full-model logical tails, changed inputs/state/pages, independent public outputs, capture guards and separate tracked watcher evidence.
- [x] Complete reduced/full4352-versus8192 comparison and reproduce the selected8192 default without dtype/topology/residual/link changes.
- [x] Reproduce final full64 warmed TTFT/decode with immediate, deferred, queued and teacher-forcing scopes separated.
- [x] Produce final reduced advice reports on all devices and same-profile lower-bound/device/host accounting; preserve the intentionally unmeasured full-device interval and profiler denominator limitations.
- [x] Refresh readiness token agreement, qualitative completions, generated-code checks and degenerate-output metrics on final source.
- [x] Execute full262144 context and final B32/non-aligned contracts; update persistent prefill/history accounting with no capability reduction.
- [x] Independent Stage7 [stage-review](stage_review.md) returns clean-pass after the metadata finding was fixed and rechecked.
- [x] Local checkpoint/provenance recorded in [work_log.md](work_log.md); operator-owned dirty state excluded.

All listed final hardware/host result artifacts and the diagnostic LoFi report have exit0; the LoFi report's `accepted=false` still rejects that candidate numerically. No MoE, vLLM/serving, broad datatype search or decoder-policy switch is part of this audit.
