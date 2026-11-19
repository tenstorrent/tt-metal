# Stage-5 remaining requirements, resume1925 audit

2026-11-13. CPU-only independent reconciliation of
`pending_requirement_matrix_resume1815.md` against fusions18, head-major19,
KV18/20, native recurrence19, profiles20, and the prepared22 queues. This is an
inventory, not a stage-review verdict or completion claim. No device execution,
TTNN import, build, or source mutation was performed for this audit.

The ledger snapshot is 19:25:51 UTC. A final filesystem read also saw the first
22-series jobs progressing; those later observations are identified separately
below and are not retroactively charged to the 19:25 ledger.

## Stop rule and reference provenance

The original prompt requires **(whole-layer roofline >=70% AND each of the last
two kept improvements <3%) OR eight hours of occupied device time**. Neither
condition is established. Checklist completion, two small gains alone, a wall
clock cap, and a high individual-op bandwidth percentage are not substitutes.

`device_time_resume1925.json` records **8449.931 seconds / 2.347203 hours** of
confirmed conservative occupancy, counting TP4 once. The driver-open/close
union is 8793.770 seconds; the 16491.227-second wall span includes idle/host work.
The conservative ledger is approximately 5.653 hours below eight hours, not an
instruction to occupy devices without useful work. Refresh it after later jobs
and evaluate the actual whole-layer bounds and kept-improvement history.

The new B32S33 profiles remain well below 70%:

| Kind / candidate | Decode device sum, us | Instrumented host, ms | Whole-layer bound, ms | Bound / host |
|---|---:|---:|---:|---:|
| GDN control |557.877|0.605688|0.280352|46.29%|
| GDN hidden-norm fusion |557.491|0.606589|0.280352|46.22%|
| Full attention control |441.384|0.496473|0.195296|39.34%|
| Full attention QK/RoPE fusion |400.249|0.444501|0.195296|43.94%|

Sources: `perf/fusions20_{control,fused}_l{0,3}_b32_s33/{metrics,accounting}.json`.
These are profiled candidate measurements, not final uninstrumented performance.

`tests/run_multichip_decoder.py --baseline` produces the accepted **frozen TTNN
OptimizedDecoder reference**, including when inputs are recorded HF activations.
It does not produce an independent HF cache/output oracle. Preserve the accepted
PCC gates, but label their reference correctly. Actual TP4-versus-HF contract
gates are a separate final validation phase described below.

## Evidence that closes stale entries

| Family | New evidence and disposition | Remaining scope |
|---|---|---|
| Residual layout, collective placement, persistent buffers, packed/separate projections | Existing coherent families and `ag_adapted9` contain 32 passing whole-finish comparisons; AGMM latency rejection follows adaptations. Tiled-batch prefill preserves its lower-movement layout through the family and is slower: B32S33 GDN 9.642 vs 7.011 ms, attention 6.631 vs 4.047 ms; long GDN 65.751 vs 46.312 ms. | Preserve selected contracts and final provenance; final stack/owner validation. Do not repeat an old family merely because a duplicate manifest is absent. |
| Native CCL | Linear receiver/admission AutoFix, guard11/build3, 15 expected rejections on each 1D/2D mesh, valid numerical checks and fresh/rebound/queued traces; MMRS NoC repair already exercised. | Include native changes in source/binary provenance and stage-owned local commits. No first-API-error rejection remains for these families. |
| Advice and geometry | `advice5` 10/10, `mlp9` 10/10; `precision_geometry5` 30 rows (28 passing), actual lowered dtypes verified in four profiles / 64 observations. `geometry13_full_down` K2/K8/K17 complete at 0.471430/0.473212/0.474591 ms. K34 has four physical adaptations in `geometry15_adapt*`: smaller grids, disabled pipeline, and DRAM output still exceed static or live L1 limits. | The four names in `geometry3_full_tail_unrun.json` no longer identify missing experiments: replacement evidence exists. Reconcile advice to these attempts, not a new blind geometry sweep. |
| Prefill padding/chunking | `prefill_chunks7`: 28 jobs / 376 configurations. `prefill_alignment8`: cap512/2048/4096 adaptations complete; short K128 schedule repairs pathological padding/single-token behavior. Coherent tiled-batch comparison also complete. | Public nonalignment and final B1 prefill nonregression remain default-path gates. Do not revive the absent old prefill2 manifest rows as separate untried families. |
| Recurrence columns/pairs | `recurrence12` 20/20 whole-layer rows complete across regression and real-text fixtures; B1 columns approximately 0.4942 to 0.4762 ms, B32 gain <1%. Static wrapper has CPU descriptor/source equivalence, but that is not hardware validation. | New production wrapper and default dispatch still require widths22/gdn22 below. |
| QK norm + partial RoPE | `qkrope_wide17` all B1..32 passes exact, poisoned-padding, changed/rebound and trace gates. Standalone B1 42.307 to 18.617 us; B32 76.449 to 25.844 us. `fusions18` full-attention decode control/fused: B1S512 0.437504/0.411757, B32S33 0.470333/0.419523, B32S513 0.497227/0.446285, text B32S33 0.469897/0.419008 ms, unchanged matched PCC. | Candidate win is established; select final combination only after packed/dispatch comparisons and final-default checks. |
| Hidden-norm fusion | `fusions18` completes all 18 rows. GDN B1 decode 0.484328 to 0.480451 ms, while prefill 2.077089 to 2.203804 ms; B32 decode roughly neutral/slower. Standalone prior admission is slower than native. Profiles20 confirm two fewer generic ops but essentially unchanged whole GDN device time. | New dispatch admission is a meaningful adaptation, not already proved by the old candidate. Resolve B1 prefill cost and B32 regression before enabling it. |
| Head-major recurrence/norm/gate | `head_major_wide19`: 258 passing cases including all B1..32, BF16/FP32 state, normal/zero/scaled, queued/tracked traces. `head_major_perf19` complete-chain native/head/tile us: B1 63.262/72.178/70.417; B3 67.807/78.011/76.159; B17 103.662/113.335/111.756; B32 149.641/154.903/153.024. | Coherent lower-movement family is rejected on measured whole-chain latency, without an immediate restoration to the old readout contract. No mandatory repeat remains. |
| KV BFP4 | Allocator and batched fill really use BFP4; `kv18` has four unchanged-gate rejections (regression B1S512/B32S33/B32S513 and text B32S33), clean close and no eligible timing. Text output passes while cache state fails. | Keep BFP8. This is a measured representation-loss rejection, not an unadapted API failure. Do not claim a BFP4 context benefit or re-open the old hardcoded-allocator objection. |

`kv20_reference_provenance_correction.json` explicitly corrects the original
`pcc_hf` label without deleting hardware evidence. Packing-only PCC to accepted
TTNN BFP8 cache is 1.0 for BFP8 round-trip; BFP4 minimum per-rank PCC is
0.976607/0.977961 for prefill K/V and 0.976831/0.977965 for decode K/V. Both K and
V fail the unchanged 0.995 state bar, so asymmetric BFP4 alone does not rescue
it. This diagnosis is not an independent HF-cache quality result.

## Remaining meaningful experiments

At the 19:25 snapshot, `widths22.sh`, `gdn22.json`, and `attention22.json` were
prepared, not completed evidence. The final read during this audit observed
`packed_qkrope_wide22.json` **pass, 32 cases**, and
`hidden_norm_dispatch_wide22.json` **running, 54 recorded cases**. A running
partial report is not a pass. Remaining stages below had no completed reports
at that read. The parent owns serial execution and the STOP guard.

1. Finish dispatch hidden-norm all-batch exact/watcher coverage; run matched
   uninstrumented packed-QKVG and dispatch-norm timings at B1/3/17/32. Packed
   timing must cover split/norm/RoPE/V/gate placement together; incremental
   comparison is against the already validated split + fused-QK/RoPE family.
   Preserve separate eager allocation and warmed trace timing fields.
2. Run `column_runtime_wide22`: default 132 cases, all B1..32, grains1/2,
   BF16/FP32 persistence plus low-core cases, changed inputs and traces, exact
   generated/static/automatic-source comparison, 50 queued replays and the
   allocation positive control. CPU descriptor equivalence is insufficient.
3. Complete `gdn22`: **25 whole-layer jobs**, control/column1/column2/
   hidden_dispatch/column1_dispatch across regression B1S512, text B1S512,
   regression B32S33, text B32S33, and regression B32S513. Select grain and
   batch admission from whole-layer evidence; retain B1 prefill nonregression.
4. Complete `attention22`: **12 jobs**, fused-QK/RoPE control, packed, and
   packed+hidden_dispatch across regression B1S512/B32S33/B32S513 and text
   B32S33. Control already includes the K128 short 8x4 SDPA schedule. Add a
   bounded interaction control only if a material disagreement requires it
   (for example hidden_dispatch without packed); no blanket Cartesian sweep.
5. Close the native recurrence candidate's unresolved failed-gate disposition.
   `native_recurrent_perf19_v2` fails B10 after 702 repeated updates: native
   FP32-state output versus accepted custom BF16-state control reaches PCC
   **0.9897577**. At that checkpoint native versus its FP32 Torch oracle passes
   output/state at **0.9985421/0.9999159**. This distinguishes a state-trajectory
   mismatch from a demonstrated native FP32 mathematical bug; it does not waive
   the accepted comparison. B17/B32 were not reached. Earlier wide18/50-queue
   passes do not close this failure, and failed-case timings are ineligible.
   Inspect same-precision and real-text/whole-boundary controls, preserve the
   failure, and use AutoDebug/AutoFix for unresolved implementation failures.
   A measured, adapted whole-family rejection can close this candidate; a
   short passing synthetic run or first failure cannot.

   A later observation during this audit is material: the prepared raw-packed
   boundary adaptation is now attempted in `prepared_native_wide22.json`, not
   untried. It fails B1 `zero_state` eager per-head PCC at rank1/user0/head6,
   **0.9892163**; B1 normal and zero_qkv precede it and pass. Diagnose this
   separately from the 702-step B10 precision trajectory. No prepared-family
   wide/performance completion is established by that partial report.
6. Review the actual new profile advice against prior adaptations. Profiles20
   already exist for all four control/fused kind combinations, with ops.csv,
   prefill/decode tables and CSV, accounting, command/source and raw Tracy
   provenance. Decode advice is empty. Prefill advice still suggests L1
   input0/program grids and increasing the 90/110-core GDN MLP-down MMRS grid.
   Map these to advice5, joint-L1 trials, the 376 prefill configurations and
   CCL-reserved-core constraints; trial only material uncovered configurations
   on the selected family. An old generic advice string is not evidence of an
   untried family. Recompute selected-path profiles after integration.

Continue meaningful applicable optimization until the original stop condition
actually holds. Record any genuinely untried applicable item as a hand-off with
its reason; a note does not itself discharge the stop rule or a failed gate.
The model is dense, so the MoE execution/routing clauses are inapplicable.
Full-model assembly, LM-head/generation and vLLM remain outside this stage.

## Final default-path gates still required

| Gate | Exact remaining work |
|---|---|
| Integration and reference identity | Select the combined defaults after meaningful comparisons. Measure actual TP4 with no policy overrides. Pin the selected source, all imported helpers and native build; accepted baseline source pin is `082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`. Candidate figures do not become final numbers merely by copying them. |
| Six before/after performance rows | L0 GDN and L3 attention at B1S512, B32S33, B32S513, same harness/reference semantics: warmed prefill and traced warmed decode, unchanged output/state PCC gates, clean uninstrumented timing, B1 prefill nonregression. |
| Six final profiles | Same six default cases: tt-perf-report prefill/decode tables and CSV, raw ops/Tracy, source/binary/runtime provenance, actual lowered dtypes, advice disposition. Whole-layer accounting includes compulsory BF16 recurrent-state read/write (786432×B bytes per device), without inflating the bound with avoidable/redundant traffic. Separate profiled host/device accounting from performance runs. |
| Final validation | Execute the fresh-source 68-job plan in `validation_plan.md`: correctness20, HF contract2, stack4, resources2, context4, stress12, watcher24. Create a fresh prefix after final integration; do not reuse the preselection manifest or stale receipts. |
| Independent HF contract | The two actual TP4 HF jobs cover 12 aligned/nonaligned lengths, seven resumed prefixes and dropped-history negative controls, and independent list rows starting32/33 with tails33/17. Preserve output/contribution PCC>=0.995. These prefill/resume checks are distinct from frozen-TTNN decode/cache comparisons and do not claim independent HF decode coverage. |
| Stress/ownership and inter-layer contract | Changed inputs, positions/page mappings, trace allocation positive controls and 50/long queued coverage must match new risks. Final suite includes B1 128-step and B32 512-step recurrence/decode stress, independent reruns, 64 independent communication owners and the real five-layer TP4 chain. Native compact BF16 residual is `[B,S,1280]` prefill and `[1,B,1280]` decode, hidden-sharded across four ranks; preserve it directly across layers. Any surviving inter-layer collective/conversion requires an actual measured overall win. |
| Capacity/context | Keep BFP8 KV and inherited262144 logical maximum unless hard physical evidence dictates the largest feasible reduction. Reconcile final dtype/layout/sharding, persistent buffers, per-device memory and reserve into `doc/context_contract.json`. Run final32K context smoke and 64-owner/resource tests; use32K–64K as smoke, not proof of262144. Changed selected storage/capacity needs the corresponding physical maximum evidence. Rejected BFP4 does not alter the default contract. |
| Runtime fallback and watcher | Refresh source and runtime audit for final packed/dispatch/static-column helpers and native CCL; `runtime_audit_resume21.json` (129 reachable, zero violations with negative controls) is prior evidence. Final runtime sentinel, tracked traces and watcher-clean logs must bind to the selected source. Kernel-string construction is not tensor fallback; host numerical work in forward is. |
| Documentation/evidence gate | Update README and work_log with exact commands, final before/after PCC/performance, topology actions, coherent-family CCL findings, measured/adapted rejections, memory/inter-layer contract, limitations and artifacts. Assemble `final_manifest.json` and run `check_optimized_multichip_evidence.py` to create `final_gate.json`; prepared checker/plan is not a passing report. |
| Independent review and local commits | Run fresh stage-review against the original goal, repair and rereview findings; require clean-pass or pass-with-carried after three rounds under the skill. Commit stage-owned touched-repo changes, including native changes, locally; log SHAs, never push. No such verdict is supplied by this inventory. |

The five-layer stack proves boundary behavior and final-chain PCC, not a
64-layer numerical model. The 64-owner harness shares representative weights;
it proves independent resources, not full-model weight residency. Context
reservation is additional to actual layer/cache/IO and must be reconciled with
the inventory rather than presented as a complete stack allocation proof.
Final quality gates remain required even when a valid optimization stop rule
is reached.
