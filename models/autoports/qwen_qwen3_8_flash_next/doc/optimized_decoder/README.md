# Qwen3.8-Flash-Next optimized decoder

Stage3 implements `tt/optimized_decoder.py` on one Blackhole p300c chip. It preserves the fused decoder's request state, paged BF16 KV cache, arbitrary positive logical prefill lengths and sensitive arithmetic. Tests construct `OptimizedDecoder` and `OptimizedSparseMoe` explicitly. Independent round2 review is `clean-pass` (`stage_review_round2.md`); local checkpoint SHAs are recorded in the work log. No multichip, full-model or serving implementation is included.

## Final measured result

Clean warmed latency, real checkpoint,32logical prefill tokens then one logical decode token; milliseconds. Fused and optimized use the same harness with BFP8 expert weights. These are actual factory-default results after the GDN precision repair, not earlier faster-but-invalid candidates.

| Representative layer | Fused prefill | Optimized prefill | Fused traced decode | Optimized traced decode | Decode reduction |
|---|---:|---:|---:|---:|---:|
| GDN0 | 78.284 | 58.806 | 5.019 | 3.052 | 39.2% |
| PLE+GDN1 | 95.877 | 63.110 | 6.349 | 4.248 | 33.1% |
| QSA3 | 184.062 | 149.206 | 9.526 | 7.296 | 23.4% |

Sources: `final_perf_{fused,optimized}_{0,1,3}_32.json`. Each selected variant is bracketed by baseline rows in its process; the fused-factory rows provide the cross-stage baseline.64 replay samples (first2 excluded) and5 warmed prefill samples; state/input restoration and reference readback are outside timing. Allocation tracking and profiler are disabled for ranking. Boundary/long lengths33,129,1024 and complete search rows are in `final_performance_summary.json` and `candidate_search.csv`. Prefill is warmed eager decoder latency, not full-model TTFT. Decode includes trace submission and synchronization.

`candidate_search.csv` records local same-run comparison results, not global acceptance. In particular, historical BFP8/HiFi2 GDN candidates were later disqualified by real-weight gates. `acceptance_manifest.json` identifies the final correctness evidence; raw failed trials are not promoted by a passing local timing comparison.

[Full timing table](performance_results.md) includes all boundary/long measurements and final separate-expert controls. Repeating the closest geometry challenger gives overlapping results within0.0001ms at the median; the selected default reproduces the best correct geometry region.

## Chosen policy

`selected_decoder_policy.json` is the complete per-group policy. Ordinary weights remain BFP8. Expert gate/up uses40cores, Kblock20, subblock1; down uses40cores, Kblock10, subblock2; LoFi with FP32 destination accumulation. Compact expert input/intermediates/output stay in L1, while larger physical-prefill intermediates stay in DRAM. Only the routed expert union executes; decode is10/512 experts.

Packed GDN z/a/b uses BF16/HiFi4. BFP8 GDN and BF16/HiFi2 were faster but fail real-weight batch/continuation checks; AutoDebug/AutoFix isolated and repaired this before final measurements. GDN QKV/output, GR projections, router and ordered QSA indexer preserve inherited precise representations. QSA ordinary QKV/output, shared packed projections and PLE key/value are BFP8. DRAM-sharded dense roles use4input cores,2readers per bank, Kblock2; ordinary roles HiFi2, GDN HiFi4. L1 residual storage survives consuming operations. QSA uses native paged SDPA with Qchunk32/Kchunk128 and8x8grid.

Packed GDN, shared and QSA projections beat legal tuned separate component controls including movement; PLE key/value remain separate because packed is slower. `dense_component_summary.json` includes final BF16/HiFi4 GDN controls: packed88.46us for the selected readers2/block2 versus best separate97.28us. Separate experts, including native fused-SiLU gate, are compared under final BFP8/LoFi in `split_experts_final*.json`. Geometry and reader sweeps retain failed configurations and adapted retries.

The best legal separate experts measure3.0766ms and pass real HF65 (`split_final_hf.json`), versus packed default3.0518ms. Native fused-SiLU separate measures3.0759ms but fails real HF decodePCC0.992139 (`split_final_hf_activation.json`); it is not selected.

BFP4/LoFi gate/up and down were measured separately with real weights, correctness probes and profiler rows. They remain stage8 precision candidates as requested. No synthetic-only veto justifies the BF16 GDN exception: the failure is reproduced with real checkpoint weights.

## Correctness and context

The whole-output acceptance threshold remains0.995. Matched HF results (`final_accuracy_summary.json`):

| Kind / logical length | Fused prefill PCC | Optimized prefill PCC | Fused minimum decode PCC | Optimized minimum decode PCC |
|---|---:|---:|---:|---:|
| GDN /65 | .9999890 | .9999838 | .9999908 | .9999780 |
| PLE+GDN /33 | .9981529 | .9981104 | .9978529 | .9978223 |
| QSA /1,2,31,32,33,65 (minimum across cases) | .9998885 | .9996225 | .9995465 | .9994451 |

The small QSA delta follows BFP8 ordinary projections/shared weights; all cases pass.14/14 delivered synthetic tests pass (`final_test_suite.xml`), exercising logical lengths1,2,31,33,34,65,66 with remapped pages and changing-input repeated traces. Real-checkpoint versions of all seven GDN lengths pass (`final_v3_real_ci0.json`). Real GDN128-in/128-out stress passes repeated reset/run and runtime audit (`final_stress.json`). Continuation, lead-row/tail, tuple-batch GDN32/QSA32/PLE5 and watcher evidence are listed below. Program-cache allocation tracking is enabled for correctness; no skip-program-cache acceptance.

The predecessor's262144 context and prepared resident PLE4768 contract are preserved in `../context_contract.json`. All32768 smokes pass with final default resident. Prefix coverage is deliberately scoped: GDN recurrence, PLE lookup/hash plus recurrence, QSA attention. GDN/PLE check complete terminal public decode; QSA checks attention-only decode. These are not full-MoE32768-prefill claims. No KV dtype/layout change or new maximum-capacity claim.

PLE4768 literal whole-output PCC0.9980767, minimum whole row0.9951833 and all three strict decode steps pass. One residual-subtracted prefill row0.9948805 fails the legacy extra contribution-row gate; the raw `final_ple_capacity4768.json` remains failed. The independent matched HF packed-lookup control passes every TT row (minimum contribution0.9985477). `final_ple_capacity4768_contract.json` separately verifies the user's whole-output contract, diagnostic floor, strict decode, runtime audit and exact resets. This is an explained diagnostic, not a claim that unchanged literal HF fails.4767 also passes the original strict probe. The work log and limitations registry retain both the result and control identity.

QSA1024 timing similarly retains an extra per-row diagnostic failure: SDPA Kchunk128 changes two rows relative to the conservative Kchunk32 control. Saved-input independent full-HF comparison passes whole PCC0.999601, minimum whole row0.992421, minimum contribution row0.991969 and both decode inputs>=0.999804. All cache buffers and replay outputs are exact relative to their eager executions. Kchunk32 with the same experts removes the strict local row miss but is slower; expert HiFi4 does not remove it. `qsa_long_assessment.json` binds the original and diagnostic runs, and `qsa_long_hf_control.json` supplies the independent HF evidence. This is accepted under the unchanged user whole-output contract, not reclassified as canonical HF failure. Running the raw timing plan reproduces its stricter diagnostic exit1; the separate assessment is required for the final contract verdict.

## Profiler conclusions

Official advice-enabled tables and CSVs exist for both phases of all three kinds under `tracy/final_ops_layer{0,1,3}_{prefill,decode}/report/`; BFP4 rows are under `tracy/final_ops_{gate4,down4}/report/`. `final_per_op_summary.json` collects matmul rows, advice and latency totals. These verify actual dtype/fidelity, indexed active-expert counts and program/memory settings.

GDN expert decode gate/up and down fall from roughly702/1337us in predecessor BF16/HiFi4 reports to105.7/49.7us with BFP8/LoFi tuned geometry. Final prefill expert unions are104/158/187; decode union10. Generic exact-GR binary/reduction kernels now dominate prefill. No all-resident-expert or entire PLE-table bandwidth denominator is used. Physical32-row padding FLOPs are not logical-token throughput.

Decode profiler collection is uninterrupted; prefill is segmented to avoid profiler buffering overflow. Kernel sums and gaps are diagnostic; clean whole-call timing above is authoritative. Segmented prefill gaps are unmeasured and its drain overhead must not be treated as application prefill latency. `profiler_accounting.md` reconciles same-capture decode host/kernel/gap measurements separately from clean timing.

Advice tried includes sparse K blocks/core grids/subblocks, fidelity, L1 memory, DRAM sharding and readers, packed/separate projections, SDPA configs and residual movement. Width-sharded residual is slower than the selected L1 interleaved contract after adapting its next consumer. Reduced KV dtype, large physical-prefill chunks/2D configs, wider exact-GR kernels, full-prefill expert L1 intermediates, shared-down refinements and BFP4 attention are deferred under the stop rule, not rejected without evidence. See [checklist](checklist.md) and [work log](work_log.md).

## Reproduction and artifacts

From the checkout with the existing built `python_env` active:

```bash
python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_job \
  --name optimized_pytest --tracked --seconds 1800 -- \
  python -m pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_optimized_decoder.py
python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_plan \
  --plan models/autoports/qwen_qwen3_8_flash_next/doc/optimized_decoder/final_perf_plan.json
```

`run_optimized_job` serializes chip0 admission, uses the established p150 mesh descriptor, records exact command/environment/log, and bounds execution. Correctness uses `TT_METAL_TRACE_ALLOC_TRACKING=1`, including program cache. Watcher uses `TT_METAL_WATCHER=10` in separate runs. All exact HF/checkpoint/cache paths and command lines are in `*_provenance.json`; checkpoint snapshot812d96e10a0dd82b69d749e68693698cf6c36daa. No device reset was required.

Key acceptance artifacts:

- `final_test_suite.xml`, `final_v3_real_ci0.json`, `final_stress.json`.
- `final_v3_hf0.json`, `final_v3_hf1.json`, `final_hf3.json`, `matched_fused_hf{0,1,3}.json`.
- `final_v3_continuation{0,1}.json`, `final_continuation3.json`; `final_v3_batch0_32.json`, `final_v3_batch1_5.json`, `final_batch3_32.json`.
- `final_v3_watcher_continuation{0,1}.json`, `final_watcher_continuation3.json`, corresponding watcher directories/logs.
- `final_gdn_context.json`, `final_ple_context.json`, `final_qsa_context.json`; `final_ple_capacity4767.json`, `final_ple_capacity4768_contract.json` and its raw/control reports.
- `autofix_accuracy/AUTODEBUG.md`, `autofix_accuracy/AUTOFIX.md`; failed candidates remain explicitly failed.

`artifact_manifest.json` records compact committed evidence and local bulky raw artifacts. Large raw Tracy captures/ops CSVs, generated identical kernel sources and watcher streams stay at their exact local paths. Compact reports, complete minified copies of oversized HF JSONs, provenance, source and tests are checkpointed. No compression or artifact relocation changes the original raw evidence.

Focused review closure: `review_watcher_tail2_summary.json` verifies final-default Watcher coverage of public34 (physical32+2) for all three kinds. PLE retains a strict row5 diagnostic, reproduced by the fused predecessor; global PCC and strict decode pass. See the work log for exact values, controls and commands. Raw failed probe statuses remain unchanged.
