# Stage 05 work log

Starting commit: adcb0e8f21. Target: google/gemma-4-26B-A4B-it, four P300c ASICs, 1x4 Linear FABRIC_1D. Stage remains in progress. No full-model or serving work.

Startup: `timeout 60 tt-smi -ls --local` exited 0, PCI IDs 0..3 visible. Active environment python_env; installed model-bringup environment.py resolved enabled skill roots. Working tree initially clean. Read optimize, tt-device-usage, model-bringup startup and tech_reports/LLMs/llms.md. Baseline runtime SHA256 a12a913cf752b765338736dc71f71151ab972af1529a0098455755dd4f499255.

Baseline commands: `HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --layer {0,5} --length 4096 --steps 128 --trace --check-cache --duplicate-replays 8 --output <baseline_sliding/full.json>`, separate serial runs, corresponding .log files. Layer0 exited0. Layer5 running. Timing reports distinguish TP1 reference from TP4 measured target.

## Initial operation-topology audit

Prior native tables: ../multichip_decoder/profile_final_{sliding,full}/{decode,prefill}_table.txt.gz; runtime policy: ../multichip_decoder/selected_policy_audit.md. Prior measurements are baseline evidence, not this pass's optimized measurements.

| Sequence / movement | Existing contract | Candidate and action |
|---|---|---|
| Input norm → packed QKV | replicated BF16 residual; FP32 normalized input; BFP8 LoFi packed weights | Preserve packing; inspect attention BFP4 and same-policy geometry evidence before new trials |
| RoPE → BFP8 paged cache → SDPA | local TP heads, BF16 sharded sliding RoPE, FP32 full RoPE | Preserve logical batch1, tile padding and page ownership; compare explicit attention configs |
| Local WO → RS → AG → norm/residual | BF16 sliding/BFP8 full communication, DRAM collective | Prior coherent sharded/Ring/fused AGMM family consumed fractured residual and lost; inspect current tables and persistent output/intermediate reuse |
| Common norm → router → sparse active8 GU/down | indexed gate-selected execution, BFP4 weights/BFP8 activation, LoFi | Preserve sparse semantics; reuse final-policy geometry controls, inspect actionable advice |
| Shared packed GU → split/activation → down | BFP4 GU, BFP8 sliding/BFP4 full down | Preserve best packed/split policy; evaluate any missing precision-locked geometry/readers |
| Pair shared/routed → RS → AG → split → separate branch norms → fused tail | BFP8 sliding/BF16 full paired communication; independent norms | Persistent CCL buffers and movement reduction are initial targets |
| Layer output → next layer | replicated BF16 [1,1,S,2816], no inter-layer collective | Preserve direct handoff; any candidate must measure its consuming norms/projections, with harness-only gather excluded explicitly |

No candidate is yet accepted or rejected in Stage05. Prior reduced-movement rejection is whole-layer, not an immediate replication-only probe (final_policy_alternatives.json). Precision, layout, buffer lifetime and next-layer compatibility must remain coherent in new comparisons.

## First measured candidates

Baseline TP4 host medians: sliding 93454.934us prefill /650.322us traced decode; full79042.991/724.813us. Both passed original .995 output/cache PCC and eight duplicate replays.

Persistent RS intermediate/output plus AG output first failed on the AG overload keyword: mesh_device overload requires persistent_output_tensor, not persistent_output_buffer. Adapted call and reran: sliding649.094us/full715.31us decode, PCC unchanged. This is tentative; selected/default, stack/batch and Watcher validation still required. Buffers are shape/dtype keyed and warm-allocated before capture; no persistent allocation has yet been accepted into the context contract.

Packed QKV BFP4/LoFi sliding trial passes min outputPCC .9967397883 with cache-consuming128 advancing steps, decode646.615us. Its prefill retains the original projection weight. Full trial in progress. No policy accepted yet.

Independent prior-evidence review identifies missing current-policy split GU, attention BFP4 and activation trials, multi-reader DRAM candidates, shared prefill K17/L1 and compatible sharded MoE-BFP8/fused-MMRS combinations. Existing artifacts are controls, not proof these missing combinations lose.

## L1 residual source diagnosis

Fresh delegated AutoFix source inspection produced `AUTODEBUG_l1_residual.md` for the first-replay layer0 candidate corruption. Four-core norm dimensions and closure binding are consistent; no implementation cause is proven. Priority is same-invocation residual/tail pre-reshard/tail post-reshard/final-output comparisons with exact mismatch indices, then conditional norm and trace-lifetime probes. Warm outputs are currently unread, so eager correctness has not been established. No hardware or implementation changes were made by the diagnosis agent; the parent owns serial experiments.

Follow-up full-attention artifact shows a separate accuracy failure despite exact replicas across128 traced positions and duplicates: prefillPCC .999945, first decode .795709, minimum .536955 at step14/position4110, aggregate cachePCC .99997. The report now requires paired baseline-TP4 versus candidate boundary metrics as well as cross-rank equality, with actual-input CPU tail/norm oracles and decode-row cache localization. Both failure source hashes are recorded; no common root cause is assumed.

Boundary capture localized L1 residual corruption to the final mixedFP32/BF16 fused add, with norms and88→4 reshard passing. Source audit found BinaryNg FPU batching selects8 tiles despite FP32 half-DEST capacity4. Parent's homogeneousFP32 workaround passed full; the smaller supported `fast_and_approximate_mode=False` mixed-input control then passed full/sliding4096/128 with exact replicas and duplicates. Full minPCC .9994528616, sliding .9976130548; measured decode751.177/685.395us versus baseline724.813/650.322us. `AUTOFIX_l1_residual.md` records the repaired, measured residual family and the still-unpatched native batching issue; no rejection is based on the original bug alone.

## DRAM multi-reader mesh AutoFix

Fresh source-only diagnosis (`AUTODEBUG_dram_mesh.md`) and extracted-source host experiment verified a placement/API mismatch: reader1 returns early, while reader2/3 pass a full MeshDevice to the unit-device NOC-hop API. `matmul_utilities.cpp` now scores secondary readers on the same first local physical device used for primary-reader assignment, retaining mesh grid validation and the public API guard. Details and reproducible diagnostic archive are in `AUTOFIX_dram_mesh.md` and `dram_mesh_host_probe.tar.gz`. Required copilot build wrapper failed because Docker is unavailable; the complete changed matmul unity translation unit nevertheless compiled and `_ttnncpp.so` linked successfully using the existing Clang20/Ninja build. Formatting and host tests passed. Parent installed the temporary linked library only after hardware closed. `dram_qkv_r2_fixed_sliding.json/log` then passed the original real4096/128 trace/cache/replica case (min output PCC .997589630569692; cache .9999966556897151), normal closure. TP4 host decode median661.898us versus earlier reader1 control655.006us, so no speedup claim. Remaining reader/role matrix continues; heterogeneous per-device placement is outside this fix.

Durable hardware regression: `python_env/bin/python -m pytest -q tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py::test_matmul_in1_dram_sharded_worker_counts_mesh` passed **all 3 cases in 2.75s** under the parent's serialized hardware run (`dram_mesh_regression.log`), with normal device closure. Reader counts 1/2/3 use an explicit 1x4 Blackhole mesh, eight DRAM banks, four storage cores and a small 32x512 @ 512x1536 shape; every replica passes PCC >= .999 and relative Frobenius <= .02 against PyTorch. Black/static checks passed. Final source/test scope inspection required no further implementation edits. Docker remains unavailable; the successful native changed-TU compilation and linked-runtime validation above are the build evidence.


## Continued precision-locked and topology matrix

`run_candidates4.py` serially measures real4096/128 layer/cache correctness and traced warmed host latency for QKV and WO DRAM reader2/3 with BFP8/BFP4, plus shared GU/down readers1/2/3 and the cumulative repaired L1 residual. Original BFP4 three-reader QKV padding failed tiled-pad validation; setup now converts the already-quantized BFP4 weight to BF16, pads zero columns, then restores BFP4. The adapted sliding run passes (`fixed_dram_qkv_bfp4_r3_v2_sliding`). Padding/slicing overhead stays in whole-layer comparisons. This is not a first-error rejection.

`run_candidates5.py` adds cumulative plain controls, L1 collective placement with/without persistence, BFP4 QKV HiFi2 and N1/N4 controls, BFP4 full WO HiFi2/N2/N4/K16/32/64, BFP4 full WO reader1, and compatible BFP4 sharded AGMM/MMRS families. A further output-column WO AGMM candidate repacks real WO along N, gathers local attention heads and hands local704 output directly to distributed norm/residual. Prior QKV AGMM does not cover that decomposition. These scripts retain exact commands/return codes; no queued candidate is a completed result.

Full BFP4 QKV+WO with restore-free sharded/Ring/QKV-AGMM and BFP8 MoE payload reaches minimum outputPCC0.994962447, below0.995 (`sharded_precision_full`); MMRS variant is also measured. This is a combined precision/topology loss, not an API rejection. The same precision on replicated residual passes0.995366478. Higher-precision WO sharded controls are queued to isolate the loss. Replicated persistentL1 candidate passes identical PCC at702.634us full and644.093us sliding host decode; no final default claim yet.

## Final geometry and CCL controls

Full QKV N1 initially generated an8x12 grid outside the11x10 device. Retried as11x9 with96 active output tiles (`qkv_n1_bfp4_v2_full`), passing all gates at719.796us. Sliding N1/N4 and full N4 also pass and lose to N2. Full BFP4 WO N2/N4/K16/K32/K64 and HiFi2 pass; no material advantage over selected N1/K8/LoFi is demonstrated.

Output-column AGMM is implemented in the harness, with actual WO weights resharded onN, localN704, 11x1/perN2/K16, BF16 gathered attention heads and FP32 matmul output; no output RS or immediate gather is inside the sharded-residual candidate. Both kinds pass. Ordinary/persistent variants measure820.870/801.780us sliding and903.156/883.680us full, slower than replicated controls. Full QKV-only BFP4 sharded controls (BF8 WO) restore PCC0.996207/0.996111 with ordinary/MMRS output, isolating the failed combined reduced policy.

Packed versus three split BFP4 QKV matmuls is measured on the final L1/persistent family: split costs670.513us sliding/724.175us full, versus644.093/702.634us packed. Head/cache/output PCC is preserved. Repaired four-core residual plus L1 CCL is also measured without a DRAM boundary restore; it remains slower.

CCL worker1/2/4, buffer2 and chunk1/4 controls keep selected attention precision, K44 and persistentL1 placement. Chunk1 first exposed an overload mismatch: the explicit mesh-device AG overload does not expose chunks_per_sync. The adapted call uses the persistent_output_buffer overload with the same cluster_axis1 and input-owned mesh; source maps both overloads to the same primitive. Retest is required, not rejected on the first API error.

## Real batch32 gate changed the selected full-attention precision

Initial defaults (`final_*_v1`) reproduce643.013us sliding/700.908us full on4096/128 with8 duplicate replays. Sliding B32 passes. Full B32 fails minimum decodePCC0.992573184 despite exact replica/replay equality, unchanged prefixes/other slots, refreshed page ownership and a clean runtime audit. This is real checkpoint/recorded-text evidence, not synthetic rejection.

Focused same-cache/same-CCL B32 control: `python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.test_multichip_contracts --layer 5 --batch 32 --attention-precision qkv --output .../contracts_qkv4_full.json` passes (minimum prefill0.998907214, decode0.995505438). It retains BFP4 QKV and restores only WO to BFP8. Thus the combined reduced-weight policy is not accepted. Current defaults retain BFP8 WO for both layer kinds, BFP4 QKV, K44, persistentL1, worker1 sliding/worker2 full. An output-only BFP4 same-CCL headline control is measured before repeating final validation. Final full numbers must be remeasured; the initial700.908us is not the accepted default result.

## Persistent CCL pool source diagnosis

`AUTODEBUG_ccl_pool.md` traces the maximum-context prefill failure to all 30 layers' retained decode-L1 payload (37,614,720B/device under the prior precision), then audits caller-owned pooling down to Linear RS/AG completion. Persistent CCL deliberately disables startup barriers; safety depends on separate attention/MoE role entries and the intervening opposite-role allreduce, plus serial same-mesh/CQ execution. Per-layer semaphores stay private and the fused tail returns a fresh BF16 DRAM tensor. Accepted all-attention-BFP8 policy needs an actual three-key union of 2,104,960 B/device, verified by host arithmetic; bank rounding/CB fit remain hardware checks. The report specifies private/pooled same-kind and mixed-kind stack controls, retained-output/trace checks, actual-pool max-context reservation, and DRAM-intermediate/full-DRAM fallback experiments. No implementation or hardware work was performed by the diagnosis agent. Parent's new same-kind control failed at prefill/layer1 before decode; source maps length 33 to one padded 64-row chunk, so the pool should still be empty. Private control and actual pool-entry/mismatch-row evidence are required before attributing that separate symptom.

### Consecutive sliding-layer prefill control

`test_multichip_stack --layers 0 1 --steps 128` fails prefill/layer1 before decode, with both shared and private pools (`pool_shared_kinds`, `pool_private_kinds`). BF8 attention controls differ in2802 finite values on ranks2/3. A private BF16 attention control (`pool_private_bf16_kinds`) also fails, in2692 finite values on ranks1/2/3. All outputs remain finite. Length33 is internally padded to64; persistent decode buffers are inactive. These controls refute attributing the failure solely to shared pooling or BF8 attention payloads. AutoFix boundary localization is continuing. No stage acceptance or final default performance claim is made.

### AutoFix semaphore-grid repair

`prefill_boundaries_bf16` localizes first replica divergence to layer1 attention allreduce; inputs and normalization agree. The inherited CCLManager allocates semaphores only on8x8, while native Blackhole worker selection uses11x10. Four-worker AG reaches x8/x9 without allocated semaphores. One-worker AG restores exact replicas through128 stack replays (`prefill_ag_workers1`). Full-grid semaphore allocation with unchanged default geometry does likewise (`prefill_full_semaphore_grid`). Native captured-input RS/AG controls pass with both old/full grids, so the isolated CCL sequence alone does not reproduce compute-induced semaphore pollution.

The model now uses `_MeshCCLManager` with actual mesh worker-grid semaphore coverage. At actual4096-token consecutive0->1 prefill, old-grid/AG1 control has layer1 PCC0.89651049; full-grid/default geometry restores0.99975299 (`stack_full_grid_actual_context`). Decode numerical loss remains separate: selected QKV4/attention8 layer1 minimum0.98660524. Previous precision control on the same repaired geometry is running. No acceptance gate was weakened.

## Blackhole CCL semaphore grid correction

`AUTODEBUG_ccl_semaphore_grid.md` identifies the imported manager's 8x8 semaphore allocation versus native 11x10 worker selection. Four-worker AG puts backward workers at (8,0)/(9,0), outside allocated semaphore storage; their 64-row workload owns the second tile row, matching the first divergent attention output. Both private and pooled stacks fail, and BF16 also fails. Parent whole-layer full-grid controls retain native CCL geometry and restore exact replicas/replays; real-context layer1 prefill improves from about .8965 to .999753. Decode cumulative PCC remains below .995 and is a separate precision investigation. Native captured RS/AG replays pass with both full-grid and old 8x8 allocation, so they do not independently reproduce the corruption without intervening model compute. Added bounded `diagnose_prefill_stack.py` and `diagnose_prefill_ccl.py`; the parent owns all hardware runs. Durable host-only `tests/test_multichip_ccl_cpu.py` passes (one test), checking all 12 semaphore allocations cover native worker placements including x=8/9. Three changed Python probe/test files pass Black for Python3.10. Parent integrated model-local `_MeshCCLManager`; no shared helper changes were made by this diagnosis agent.

### Shared pool equivalence and capacity

`stack_shared_grid_actual_context` has identical outputSHA256 values to `stack_full_grid_actual_context` for every TP1/TP4 prefill and128 advancing traced decodes. All replicas/replays agree. This proves this pool reuse case, not the still-failing cumulative numerical gate; see pool_equivalence.json.

`capacity_pool_full_grid_sliding` passes262143-token prefill and final-position decode with PCC0.99989555/0.99940172 and cache minimum0.999999966. Before prefill it allocates the actual2104960B/device shared CCL payload union,30 full-grid private semaphore sets and19864223744B/device of other-resident DRAM. Context remains262144. This is a capacity reservation probe, not full-model execution. Full-attention capacity is running; final selected precision must update/revalidate the memory contract.

Full attention `capacity_pool_full_grid_full` also passes:262143 prefill/last-position decode PCC0.99994355/0.99971868, cache minimum0.99997097;19394461696B/device other-resident DRAM plus actual shared pool and30 semaphore sets. No context reduction.

### Cumulative precision controls

Expanded real4096/128 stack0->1 keeps the0.995 gate. Independent changes from QKV4/attention8: expertGate8 improves minimum to0.990188, expertActivation16 to0.987297, sharedGate8 to0.989331, WOHiFi4 to0.986903; nonepasses. QKV8+expertGate8+attention16 passes0.99653849. CheaperQKV8+sharedGate8+attention16 fails0.99409042; QKV8+expertGate8+attention8 fails0.99260450; QKV4+expertGate8+sharedGate8+attention16 fails0.99301254.

The integrated sliding default (QKV8,expertGate8,attention16) reproduces0.99653849 in `stack_selected_samekind`. `stack_routes.routes.json` exactly reproduces the failing candidate and shows router outputs match HF on each own residual. Some route membership changes follow attention drift; another badposition usesidenticalroutes butroutedbranchPCC0.986475. This supports model-visible combinedprecisionloss rather thananunfixed routerselectionbug.

Mixed0->5 expandedstack exposes full-attention QKV4 cumulativePCC0.973639 (`stack_selected_mixed`); same-mesh QKV8 control is running. Currentpolicies remainprovisional untilbothkindsandmixedstack pass.

### Correct the mixed-stack activation contract

The inherited mixedtest connected0 directlyto5 usingrecordedlayer0inputs. That skips fourreal layers; it is a syntheticcomposition, not actual layer5 input flow. Broad precisioncontrols there still failed despite aligning severalTP1groups. Per optimize OPT-012, this alone cannot veto policies that pass realtargetactivationevidence. Keepthosefailures asdiagnostics. The valid0->1 consecutive tests remainrequired: integratedselecteddefault passes4096/128 at0.99653849 and33/128 at0.99504973.

The stackharness nowdefaults toadjacent4->5, selects a fixturematching its firstlayer, validates metadata.layer, and recordsfixturehash plus real_model_adjacency. A boundedHFCPU fixturecapture fromsavedrawlayer0inputs throughHF0..3 is beingprepared; tokenIDs andprovenance preserved. Real4->5 stillmustpass theunchanged0.995gate beforeaccepting the mixedhandoff. NoTTNNfullmodelorservingwork is added.

### Final-policy geometry and capacity validation
The real adjacent0→1 accuracy gate selects sliding QKV BFP8 plus expert gate/up BFP8 and BF16 attention CCL. Real4→5 fixtures retain full QKV BFP4. Final-policy geometry matrix keeps QKV K44 and sparse gate N1/K44: selected652.532us, K22 652.972, K88 653.914, QKV N1 658.819/N4 656.604, expert K88 655.318, N2/K44 679.143, N2/K88 681.940, HiFi2 655.594, separate684.765, DRAM-reader2 661.681, four-core L1 residual676.634. All are warmed traced host-wall decode microseconds, all passing these single-layer gates. Accuracy-matched unoptimized baseline656.489us. The carried-shard Ring/AGMM rerun produced rank-local cache NaNs and is being independently diagnosed under AutoFix; no rejection from this first failure. Final-default validation launched with source-hash guarded command records. Updated planned per-device extra resident weights1,754,480,640bytes and shared CCL payload2,690,688bytes; final capacity proof remains pending.

Host formatting checks: Black and cached isort pass runtime and acceptance harnesses. Cached autoflake found an unused CollectiveBufferPool import in the batch contract harness; removed it, with contract rerun required under its new source hash. No dependency installs. Independent stage reviewer confirmed the full shared-down policy is BFP4 (sliding BFP8); corrected the comparison document. The33-token real0→1 minimum0.995049733 has limited margin and remains an explicit numerical risk.

### Verified fused gather K-block repair
`probe_sharded_kblock.py` changes only fused QKV K44→K22 on the exact failing final policy. All128 decode positions and per-rank caches now pass: minimum output PCC0.998281294, cache0.999979739, warmed host decode806.040us. Retained K44 only when fused_agmm=False and added local ready-slice divisibility validation in _GatherProjection. The exact before/after patch, hashes and unchanged-default evidence are in fused_guard_source_delta.patch and fused_guard_source_equivalence.json. No earlier artifact hash was rewritten. Six unwrapped current-policy fused-family comparisons remain scheduled.

### Watcher assertion and serialized recovery
The first final sliding Watcher run aborted(-6) in TP4. BRISC asserted at included fabric_connection_manager.hpp:119 (has_forward_connection), not minimal_default_writer.cpp literal119. NCRISC CRBW is a downstream wait. Initial tt-triage had noInspector aftertheprocessaborted; explicit--dev=all ARC/Ethernetchecks succeeded and showedhealthy links/heartbeats. Preserved watcher_failure/{run.log,watcher.log,triage-command.log,tt-triage-devices.txt}. No stale testprocessremained. Bounded tt-smi list/reset/list allreturned0, expectedfourASICsvisible, then1x4meshopen/close passed. AutoTriage sourceledger identifiesunconditionalconnectionlookup atanoutwardLinearendpoint inone-workerAG; proposedguardpreservesrequiredrouteassertions. NativepatchandoriginalWatcherreverificationpending.

All four final model Watcher commands now exit0 on the repaired kernel: sliding/full4096/128 with8duplicate replays and real0→1/4→5 stacktraces. Output/cache/replica/ownership checks pass; normalWatcher shutdown andfullsourcehashesare in final_watcher_summary.json. Exact native regression is running next, then separatefinalprofiles. Final hostdefault medians now sliding93346.919/653.344us andfull79004.977/701.861us(prefill/decode), fromruntime6938c6d5. No earliercandidate latency isusedasfinal.

### Independent-review finite-PCC gate repair
Reviewer identified Python min() couldignoreNaNs aftera finiteprefix. HistoricalfailedK44cachevector demonstrates thebug; outputPCC independentlyfailedthatcandidate. Added explicitfinite-and-threshold checks tocache/output andbatch/stack gates, plushostregression. `audit_finite_gates.py` executes theactualsourcehelperwithout importingTTNN and verifiesnonfinite/thresholdcases plus42acceptedartifactvectors; allacceptedfinalmeasurements remainfiniteandpassing. Exactpostprocessing-only sourcechanges andhashesare retainedin finite_gate_source_equivalence.json/.patch. No deviceexecution or measurements change; independentreviewwillcheckthefix.

### Final advice closure and evidence freeze

The final profiler advice audit identified four prefill producer-L1 roles not yet measured under the selected policy. `run_prefill_producer_matrix.py` tested QKV, output, router and shared producers against both layer-kind controls (five prefill samples). All10 cases pass. `run_prefill_producer_repeat.py` repeats marginal candidates with15samples (12 cases); `run_prefill_producer_combined.py` combines full QKV/output/router against30-sample control (two cases). Actual producer intercepts and QKV memory captures prove placements. Retain DRAM producers. Initial0.4–0.5% apparent gains do not repeat consistently: second full control78540.505us vs QKV78543.679, output78558.617, router78586.753. Sliding router92717.347 vs control92690.445 in first repeat reverses the initial gain; second repeat changes ordering again. Combined full placement78487.685 vs78535.423us control is only0.061%, within overlapping30-sample distributions, while decode702.372 vs701.530us. No reproducible material whole-layer gain is established. Shared-input L1 loses2.3–2.4%. All24 cases pass identical per-kind output PCC; source intercepts verify actual L1 producer placement. No API failure was used as rejection.

The final default remains runtime6938c6d512d19cf25c3c9d7fb5cee29f2b3f46d174426b715180527b16e02cd3. No later candidate changes its wiring. Both separate final profiles complete and match this runtime/native endpoint fix. `final_perf_findings.md` and `final_perf_summary.json` report whole-layer device latency, four-ASIC rooflines, native policy and advice disposition. Sliding device prefill/decode329194.676/710.513us; full339373.403/744.896us. These include device gaps; ordinary warmed host medians remain93346.919/653.344 and79004.977/701.861us. No prefill or sliding original-baseline speedup is claimed.

`final_validation_summary.json` enumerates18 accepted gates: exact workload, B32, actual adjacent0→1/4→5 at33/4096, nonaligned4095, aligned/nonaligned maximum262144 capacity, and separate Watcher layers/stacks. Durable host finite-PCC tests pass(two cases); full-grid semaphore CPU test passes(one); native endpoint Watcher regression passes(two); native DRAM mesh-reader regression passes(three). Source-equivalence records preserve original hashes for capacity/default-neutral fused geometry, endpoint lookup and host finite-gate edits.

`preserve_evidence.py` retains compressed converted perf CSVs/tables, native policy rows, window CSVs and compact logs with SHA256 manifest. Original raw native CSV/Tracy captures and tensors remain local, not committed. `write_packet.py` writes the required local packet from final default evidence only. Black(py310), cached isort, touched-line C++ formatting and git diff whitespace checks pass. Docker build wrapper was attempted but unavailable; the changed matmul translation unit and native library were built with the existing native toolchain, and the endpoint kernel JIT-compiled in successful Watcher runs. No dependencies installed; no push.

### Independent acceptance

The fresh xhigh stage reviewer returned **clean-pass**, with no required work, in `stage_review.md`. It independently checked all18 final gates,24 producer-placement controls, both native profile windows,40 profile provenance hashes, final packet identifiers/calculations and307 compressed archives. Acceptance status is now recorded in README, context contract and final validation/memory summaries. Stage05 is complete; full-model and vLLM work were not started. Local implementation/evidence commit and SHA checkpoint follow; no push.

Commit hooks reformatted only a nested conditional in the diagnostic `diagnose_prefill_stack.py`; AST equivalence verified. A raw triage text file was restored byte-for-byte from its archive and excluded from tracking, preserving its manifest hash. Required pre-commit hooks are rerun on the staged files. Commit-local Codex identity matches earlier stage commits; no global Git configuration changed.

### Local checkpoint

Implementation, repairs, regressions and complete Stage05 evidence committed locally as `dc254cbde8997f0ec8f8f42bccb5aaf1f0e8d69e`. All applicable repository pre-commit hooks pass, including Black, isort, autoflake, clang-format, include validation, large-file and whitespace checks. This follow-up documentation checkpoint records the implementation SHA; its own SHA is recorded in the local telemetry packet and final handoff. No push.
