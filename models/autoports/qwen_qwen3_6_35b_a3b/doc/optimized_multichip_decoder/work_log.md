# Optimized multichip decoder work log

Stage 5, Qwen/Qwen3.6-35B-A3B; starting checkpoint `5ebe6996` (implementation `679bb405`). Started 2026-11-13 21:31 UTC. Default optimization wall cap: four hours. Scope: existing multichip decoder only, target Blackhole p300c chips 0–3, 1x4 FABRIC_1D / Linear. All measurements serialized. No full-model or serving work.

## Startup and audit

Enabled tt-autodebug inventory confirmed in CODEX_HOME/config.toml. Environment uses `source python_env/bin/activate; source ../lane-env.sh`, which resolves installed package exports. Working tree initially clean. `fuser /dev/tenstorrent/*` found no owner. `timeout 60 tt-smi -ls --local` found all four chips. Baseline pre-commit hooks passed (`logs/baseline_hooks.log`). Device runs are bounded with the installed `scripts/wait_for` wrapper and timeout. The first benchmark also opens/closes the correctly configured mesh.

Baseline profiles read: previous stage `profile_findings{,_b1}.md`, raw/profile evidence under `doc/multichip_decoder/{tracy,perf_b32/tracy}`, and `mesh_plan.md`. B32 is dominated by per-row attention and movement, not expert compute. Before any knob tuning, the operation-topology audit is being written in `topology_audit.md` by an independent CPU-only audit agent.

## Measurements

Commands, source digests, environment and checkpoint are recorded in each JSON's `.provenance.json`. Fresh before controls use the prior accepted saved single-chip numerical baseline and current multichip code; saved single-chip latency is not a fresh speedup comparison. Natural inputs are 32 distinct prompts, eight changing decode steps. Clean timings use 60 trace replays, three prefill repetitions and three stress repetitions. All meshes include all four devices.

## Candidate plan

1. Batch the attention-output collective across independent requests before residual add. Preserve FP32 linear partials until all-reduce. Compare direct AR, RS+AG, and hidden-sharded RS with its next consumer; do not restore a replicated residual solely for measurement.
2. Batch full-attention normalization/projections while retaining per-request paged SDPA and positions. Keep active expert routing and compare cumulative correctness/perf.
3. Follow measured bottlenecks and current perf-report advice; retain accepted B1 behavior.

No optimization result or stage completion claimed yet.

## Batch collective adaptations

The first batch direct-AR candidate failed at the following full-attention norm: inherited multirow RMSNorm requested interleaved output for a sharded input (`logs/batch_ar_l3.log`). Adapted the consumer to retain the input width shards. The next attempt reached the row-output splitter, where the native arbitrary-row slice tried a physical shard height1 (`logs/batch_ar_l3_v2.log`). Adapted the list-return boundary to DRAM before slicing; this is charged to the candidate. Neither failure rejected the collective family.

Both failures closed mesh devices cleanly. After confirming no live device owners, bounded reset/list and correct-fabric open/close smokes ran: `logs/reset_01{,_smoke}.log`, `logs/reset_02{,_smoke}.log`. No other process killed, no locks cleared. Reset01 recovered all four chips on its first reset. The second is a separate failed-candidate recovery, not an incomplete-list retry.

## Precision and collective screens

`attention_bfp4_l3.json` uses real weights/natural B32 input and LoFi under batched replicated/direct-AR topology: prefill PCC0.99436135, minimum decode0.99296884, below unchanged0.995; traced5.039884ms. This is a moderate real-model precision loss, not catastrophic corruption. BFP4 is never selected as the default. `attention_lofi_l3.json` passes0.999926/0.999780 at5.059717ms versus5.073939ms HiFi2 under the same topology before shared-output conversion optimization.

Batch full coherent residual controls are `split_once_l3` (replicated/directAR4.937193ms), `batch_rs_ag_l3` (replicated RS+AG5.015001ms), `batch_sharded_l3` (hidden512 residual, next ordered norm5.831879ms), and `batch_sharded_l0` (packed linear, hidden51221.585338ms). The sharded paths keep their residual through the next consumer, with harness host composition outside timing. CCL BF8 passes under directAR4.956955ms, slower than BF16; FP32 linear partials remain FP32 when this experimental knob is used. Direct-AR scratch/semaphores remain persistent and shared.

The combined fused AGMM/MMRS candidate completed device execution but failed exact paired K-cache identity in two scalar cells; no gate was relaxed. The original extra sandbox preflight for AutoDebug failed because this host cannot create an unprivileged namespace. Retried the authorized inspection-only child with AUTODEBUG_SKIP_CHILD_SANDBOX=1. Source analysis identifies rank-relative fused K accumulation rather than distributed-norm statistics order. AutoFix validated the canonical separate-input-projection control with the rest of the fused/sharded family retained; see AUTOFIX.md. Reset03/list/mesh-smoke recovered healthy four-chip state; Reset04 followed the separate BFP4 PCC failure. No processes or locks were killed/cleared.

## Integrated selection

Selected batched attention reduction, batched full projections, packed ordered linear inputs and one batched untilize before row slicing. BF16 KV/CCL and HiFi2 attention remain selected. Repeat200 control HiFi2:4.938860ms decode/42.464985ms prefill; LoFi:4.927429/43.853983. The ~0.23% decode gain does not justify the ~3.27% prefill loss. Shared DRAM projection and reduced-CCL/KV variants did not improve the coherent best family. The final defaults are measured without constructor overrides.

The most recent kept movement candidates are the once-only row split on full attention (5.073940 to4.937193ms,2.70%) and its integration with packed linear attention (20.590145 to20.466604ms,0.60%). Both are below3% on the largest-batch decode target, satisfying optimize Stop Rule2. This stops further optional B32 tuning; correctness, performance-floor, context, watcher and independent-review gates remain mandatory. The earlier packing candidate also improves prefill substantially; that earlier gain is not used to claim saturation. See optimization_decisions.md for exact untried handoffs.

The K64 DRAM geometry exceeded physical L1 twice: first input8/output2 and then adapted input16/output1. Reset05 and Reset06/list/mesh-smoke followed the separate closed-device failures and recovered all four devices. Repaired fused output with BFP8 CCL was also compared coherently; it remains slower than the selected replicated residual. All rejected options keep their raw failure logs or passing JSON and command/source sidecars.

Source pre-commit hooks passed before the final validation battery (logs/final_source_hooks.log). The installed standalone Black version suggested a different style, but the repository-pinned Black23.10.1 passed without modifying source; repository hooks are authoritative. An intentional STOP sentinel paused the initial final battery after linear B32 to finish the added FUNCTIONAL qualification runner; no live hardware process was interrupted. Those files will be refreshed by the complete final battery.

## Mandatory FUNCTIONAL repair

The inherited stage4 evidence omitted a mesh-normalized128-in/128-out floor. Added an opt-in runner and strict stage5 gate; single-chip percentages are not accepted for the mesh. Both first runs pass whole-output PCC and deterministic changing-position trace replay, but fail10% useful-byte efficiency.

Layer0: 1.007534997ms, 5.347389%; prefillPCC0.999991977, minimumdecodePCC0.999158705. Theoretical per-rank logical traffic is computed in tests/multichip_functional.py at512GB/s, excluding padding/rereads/intermediates.

Layer3: 0.397953289ms, 8.655099%; prefillPCC0.999969297, minimumdecodePCC0.999777332. Theoretical per-rank logical traffic is computed in tests/multichip_functional.py at512GB/s, excluding padding/rereads/intermediates.

These are failed gates, not a waiver. A new isolated AutoDebug inspection is running, and fresh B1 profiler captures are under functional_debug/tracy. Candidate repairs must preserve exact selected-expert semantics and whole-output accuracy; larger-batch speedup does not satisfy this gate.

## Packed-storage correction

Source audit found the inherited replicated scan_attention_weight and scan_output_weight still resident when pack_linear=True. The packed branch now deletes both superseded attributes after creating its local combined weight. This releases2048*(6208+6144)*2=50,593,792bytes/chip compared with the first packed candidates. Local packed coefficients occupy2048*3136*2=12,845,056bytes/chip, exactly equal to the previous local split coefficient payload. Runtime arithmetic is unchanged; source-bound final runs and resident census are being refreshed. Repository-pinned hooks pass (logs/storage_hooks.log).

## Resumed FUNCTIONAL AutoFix (2026-11-14 01:12 UTC)

Continued the existing state; no restart. AutoDebug completed (`AUTODEBUG_functional.md`). Corrected the useful-byte model: linear router weights are BF16 even though router logits are FP32, and the output-norm gamma is L1-resident. The honest linear 10% ceiling is0.518283125ms; full attention0.3444325ms. Moved audit-context setup/teardown outside the blocking-replay timer; all device-only auditing remains enabled. This is a measurement correction, not a kernel speedup. Original measurements remain in functional_initial_l{0,3}.json.

Fresh corrected native norm+projection linear candidate:0.641690534ms,8.07684%, minimumPCC0.999129197. Adding native recurrence:0.615891656ms,8.41517%, minimumPCC0.998372623. Both fail the performance floor and remain experimental. Full input norm on direct-AR shards:0.372412604ms,9.24868%, minimumPCC0.999834106. Each uses128-in/128-out and384 changing-input replay samples, with exact eager/trace equality. No quality gate changed.

FP32 selector component passes real-router, one-ULP, ties, signed-zero, nonfinite and repeated-trace checks on all4ranks; median18.410us versus native54.384us. Compensated norm-tail component is bit-identical for all129 scalar rows on all4ranks. Integrated candidates are pending. Commands and outputs: `tests.probe_mesh_fp32_topk --output .../fp32_topk_component.json`; `tests.probe_mesh_norm_epilogue --output .../norm_epilogue_component.json`, with lane environment, timeout600 and trace allocation tracking.

All failed closed-device experiments are followed by bounded list/reset/list and FABRIC_1D1x4 open/close smoke through `tests/reset_multichip.sh`. Resume and each repair reset log live in logs/reset_*; no external owner was killed or locks cleared.

## Not tried (stop rule)

- A public batched residual interface carried across stacked mixed layers,
  eliminating row extraction entirely. The earlier profile's total row split
  was about200 us; the new remaining cost must come from the final profile.
- Canonical-accumulation fused AG-MM or pair-owner KV broadcast restoring exact
  duplicated KV equality; no speedup is claimed for this unimplemented repair.
- Shared packed-versus-separate local GU/scalar with fused AG and gathered-input
  reuse for router/experts; the grouped path's small share limits whole-step gain.
- HiFi2 B32 per-role geometry cross-products and residual-grid variants beyond
  inherited singleton controls; current LoFi results are not a precision-locked
  rejection. More explicit 2D prefill/chunk-boundary geometry sweeps remain.
- Grouped-expert core-grid tuning, fused combine+score reduction, and persistent
  dispatch/combine output alternatives. Active-expert execution must remain.
- Further BFP4 expert/shared candidates beyond inherited screens; final datatype
  selection belongs to stage8 and cannot change this stage's BFP8 default.
- BFP8 normalized activations throughout a compatible projection family, beyond
  measured BFP8 CCL and KV candidates; keep FP32-sensitive exceptions explicit.

LM head, sampling, token feedback, TTFT to generated text and qualitative text
checks are not implementable in this decoder-only stage and are reserved for
their full-model/serving stages. Decoder prefill is measured, not relabeled TTFT.


## FUNCTIONAL component integration

FP32 selector+projection-aligned input norm passes full attention128/128:0.341464075ms/10.08693%, minimumPCC0.999762756. Exact compensated norm+selector is numerically unchanged on layer0 (minPCC0.999158705), but0.759227388ms/6.82645% still fails. Native norm/projection/recurrence+selector reaches0.577010844ms/8.98221%, also below the floor. The sensitive layer34seed154128/192 control fails minwholePCC0.968853207 at step131 (first failurestep37), so the native family is not accepted. This is a real arithmetic regression, not an API rejection; individual components are being isolated. Native+SiLU once-per-page at seed154 gives0.561513518ms/9.23011%, minPCC0.997745464 on layer0 only, and cannot waive the layer34 failure.

### Layer34 diagnostic clarification

The current default itself fails the new layer34seed154128/192 comparison against the single-chip OptimizedDecoder (minPCC0.991685325). Selector-only reproduces the same minimum. Source audit confirms a preexisting router-policy difference: the single-chip decoder rounds logits to BF16 and uses partition-based tie handling; the mesh baseline selects FP32 logits. This is not established as an HF-controlled exception. Direct candidate-to-mesh comparisons still show additional native regressions: norm0.988462994, projection0.988639016, recurrence0.968895326. Thus native changes remain unaccepted; routing restoration is an explicit separate precision/semantics experiment. `layer34_default_candidate_comparison.json` records the matched whole-output comparisons.

## Four-hour optimization cap (2026-11-14 01:31 UTC)

Optional optimization is stopped under Stop Rule3 as well as the earlier Rule2. The currently queued experiments are AutoFix of the mandatory FUNCTIONAL/accuracy failures, not a new optional tuning sweep. The stop rule does not waive the 10% floor, whole-output PCC, final-default validation/profiles, watcher, context, review or commit requirements. All untried optional work is listed above. No stage acceptance is claimed.

Full attention's residual-layout-preserving candidate (FP32 selector,32-core input norm matching the existing residual) passes at0.341138911ms/10.09655%, minimumPCC0.999834106. This is preferred for further integration over the16-core first-norm alternative because it preserves the established inter-layer residual contract.

SiLU component eager checks pass all65,536 BF16Z codes, all four ranks and both output/debug-gate bit comparisons. Its first traced replay is correctly rejected by allocation tracking because input-copy program-cache allocations were not warmed before capture. The probe is repaired to warm every exact copy signature; no audit exemption is added. Original failure retained in silu_component.json/log; retry will use a fresh artifact name.

## AutoFix evidence consolidation (2026-11-14)

The compact hypothesis/result ledger is now [AUTOFIX_functional.md](AUTOFIX_functional.md), derived from `repair_*.json`, command/source sidecars and saved tensor comparisons. Earlier entries are chronological observations; the following classification supersedes their provisional attribution. Stage remains **INCOMPLETE**; no waiver or accepted final default.

Exact compensated norm epilogue + FP32 selector + tile-worker SiLU reaches0.7217527ms/7.18090% (`repair_exact_norm_silu_l0.json`). CPU comparison against `repair_corrected_default_l0.pt` found identical inputs, prefill, all **128** decode outputs, and all saved rank-wise convolution/recurrent states. The exact family still needs0.2034696ms (28.19%) less latency to reach the honest0.518283125ms ceiling. No measured small remaining change is known to supply it.

Native DRAM projection controls remain below the floor: default native0.5527317ms/9.37676%, HiFi2 0.5512432ms/9.40208%, LoFi0.5512291ms/9.40232%. Their layer0 minimum PCCs are0.999131380/0.999124661/0.996622928. LoFi's negligible time improvement does not close the gate. These candidates are additionally disqualified by the sensitive-layer arithmetic findings; layer0 success cannot waive them.

Full attention candidates pass128/128: projection-aligned norm+selector0.341464075ms/10.08693%; residual-aligned norm+selector0.341138911ms/10.09655%. Both are experimental and close to the threshold. The latter preserves the established residual layout. Fresh integrated-default qualification is still required; full-attention success does not satisfy the linear floor.

Sensitive layer34seed154S128/D192 default and selector-only outputs are bit-equal (`selector_layer34_comparison.json`) while both miss the single-chip baseline at0.991685325. This is an existing mesh/baseline mismatch, not a new selector defect. Restoring the **complete** baseline router policy (BF16 logits, inherited partition tie handling and inherited normalized BF16 scores) improves but does not close it: `repair_baseline_router_l34.json` minimum0.992745431, first/minimum step41. Thus router precision/ties alone are not a sufficient diagnosis; no HF-controlled limitation is established.

Additional native arithmetic regressions remain real: matched candidate-to-default minima are norm0.988462994, projection0.988639016 and recurrence0.968895326 (`layer34_default_candidate_comparison.json`). Full baseline-router restoration with the native family still fails0.988486105, and with native recurrence0.968781130 (`repair_baseline_router_{native,recurrence}_l34.json`). Merely BF16-rounding the FP32 selector input also leaves0.988486105. These changes remain unaccepted. First-divergence residual/norm/router/expert capture is still needed for the default mismatch.

SiLU retry passed exhaustive BF16Z, all ranks and tracked replay with program-cache allocations included (`silu_component_retry.json`); once/page and tile-worker integrated tensors match (`silu_integrated_tensor_comparison.json`). The original allocation-tracking failure is preserved and was fixed by warming exact copy signatures, without an exemption.

Remaining justified but unmeasured kernel hypotheses include exact ordered-scan coefficient reuse and exact recurrence data-movement/fusion, each requiring numerical controls and complete-consumer timing. There is no supported prediction that either minimal change closes the28.19% exact-path gap. They are unresolved repair avenues, not claimed gains. Optional optimization Stop Rules2/3 have fired; the hard floor and accuracy investigations remain required work. AutoFix has not restored the full contract. Final-default validation/profiles, MoE attribution, strict gates, watcher/context, independent review and local checkpoint remain incomplete.

### 01:44 UTC current-default measurements and initial HF diagnostic interpretation (corrected below)

Integrated only exact selector/norm/SiLU helpers and the 32-core full-attention norm layout. Four final clean runs pass whole-output PCC. Linear B1 prefill52.603419/decode0.725101ms; B32 prefill1156.660697/decode16.162265ms. Full B1 prefill2.134202/decode0.346633ms; B32 prefill44.724137/decode4.941677ms. These are actual default results, not stage acceptance. SiLU component retry passes87 eager/261 changing-input trace checks with tracking enabled and program-cache allocations included.

CPU investigation found an existing matched HF oracle for layer34seed154S128D192: single-chip minimumHF PCC0.999667588; starting mesh minimumHF PCC0.991128479. Inputs match exactly. The initial interpretation called this an accuracy defect; that per-step gate interpretation is superseded by the contract correction below. Source-only AutoDebug launched for earliest divergence; a targeted eager intermediate capture is being prepared. Requested the validation driver stop between jobs using STOP; no in-flight device work is interrupted. Remaining validation will follow the repair.


## Contract correction and current-default snapshot (2026-11-14)

This entry supersedes earlier claims that default layer34seed154S128/D192 fails a mandatory accuracy gate or requires an HF waiver. The additional sensitivity stress uses aggregate whole-output PCC≥0.995 with per-step diagnostics floor0.99. Default and selector-only have matched-HF aggregate0.999702065 and diagnostic minimum0.991128479, meeting both criteria; all192 outputs are bit-equal between them. Restored baseline router policy also meets both (aggregate0.999841150/minimum0.992242348). Standalone hidden-state channel argmax is not token top-1; token agreement is N/A here. Existing stage4 mandatory per-forward fixtures and strict checks remain unchanged. Historical `passed:false` repair artifacts are preserved, not relabeled. [The CPU comparison script](../../tests/compare_layer34_hf.py) now records aggregate criteria and below0.99 diagnostic anomalies explicitly in [layer34_hf_comparison.json](layer34_hf_comparison.json), along with exact matched inputs and historical/current source provenance.

Native norm/projection/recurrence alternatives retain below0.99 diagnostics and remain unaccepted. Their passing aggregate is recorded, not used to dismiss anomalies. Native-to-default minima are0.988462994/0.988639016/0.968895326. First-divergence investigation applies to these experimental alternatives, not to a newly invented default per-step0.995 extended-stress gate. No HF limitation or gate waiver is asserted.

Current default enables exact FP32 selection, compensated norm tail, tile-worker SiLU and full-attention input normalization on the existing residual shards, alongside the previously selected packed/batched movement family. Native norm/projection/recurrence remain experimental. The four clean default timings are documented in [README.md](README.md): linear B1/B32 decode0.725101498/16.162264503ms and full B1/B32 decode0.346632998/4.941676998ms; all four fixtures pass existing PCC checks. These are source-bound measured current-default numbers, not stage acceptance.

Fresh current-default [functional_l0.json](functional_l0.json) fails the hard floor at0.719601805ms/7.20236%; [functional_l3.json](functional_l3.json) passes at0.340614331ms/10.11210%. Both pass recorded accuracy and replay checks. Linear remains0.201318680ms above the honest0.518283125ms ceiling (27.98% required latency reduction). Optional tuning Stop Rules do not waive this hard floor. Stage remains **INCOMPLETE** while current-default validation/profiles proceed and the experimental exact-recurrence repair is investigated separately. Required remaining artifacts include risk-matched validation, profiles/CSV/provenance and MoE shares, strict evidence gates, watcher/context, independent review and local stage-owned commit; no acceptance or push is claimed.


## Completed current-default validation subset (2026-11-14 02:07 UTC)

CPU-only consolidation reran the unchanged `Gate` methods `paired`, `natural`, `watcher`, `mixed`, `context` and `trace_positive_control` against the current artifacts. All passed: eight public boundary fixtures (layers 0/3, lengths 1/31/65/257), two S129/D128 tracked stress fixtures, two natural B32 fixtures, four watcher fixtures, mixed-stack/shared-cache ownership, both 32769-token context smokes and the allocation positive control. This was a validation-subset check, not a complete stage gate run or independent stage review. Each artifact's decoder provenance matches source `13b403518fb8b71dcf5b502bddeb9be79bcd6acf27b1bb51ab47c4ef902d952f`; original strict PCC thresholds/default-policy checks were retained.

Stress minimum decode PCC is 0.999768122 linear and 0.999834106 full. Natural B32 minimum decode PCC is 0.999904061/0.999854346, with 32 distinct teacher-forced prompt/token rows and activation provenance verified. Watcher logs contain respectively 2/4/1/2 completed dumps for linear B1/B32 and full B1/B32, with no failure signatures. The positive control rejects one unsafe device buffer, then passes clean replay on all four devices. Mixed stack verifies shared/disjoint page ownership, repeated eager/trace equality and runtime fallback audit. Context prefill is 32769 and allocated capacity 65536; only two subsequent decode steps are measured, so this is not a 65536-token filled-prefix claim. Updated context-contract evidence includes fresh census, selector persistent DRAM buffers and expanded SiLU L1 LUT; BF16 cache and inherited 262144 aggregate-token scope remain unchanged.

Linear FUNCTIONAL remains failed at 0.719601805 ms/token (7.20236%); full passes at 0.340614331 ms/token (10.11210%). Stage remains **INCOMPLETE**. Profiling is still in progress; completed validation does not substitute for current-default B1/B32 tables/CSV/provenance, advice experiments or MoE attribution. Complete evidence gates, independent review/fixes/rereview and local stage-owned commit remain outstanding.

The pending native FP32 all-reduce patch is preserved in `fp32_all_reduce_repair.patch` with original source hashes in `fp32_all_reduce_repair_patch_sources.json`. It restricts its new DEST/SFPU path to Float32 input plus FP32 destination accumulation and leaves the BF16 branch unchanged. CPU-only inspection found no concrete patch/probe defects; `git apply --check` and probe AST parsing passed, without a build or device claim. The patch is currently unapplied. If subsequently applied/selected, native factory/kernel and loaded-library identity must be recorded and affected measurements refreshed; unchanged Python decoder hashes alone cannot qualify old evidence for a new native execution path. Current profile evidence is retained as the control, not silently relabeled after rebuilding.

## Native FP32 collective AutoFix control (2026-11-14 02:08 UTC)

The final B32 full-attention capture has closed its devices; the remaining profile driver only postprocesses/reports. `fuser /dev/tenstorrent/*` returned1 (no owners). No live C++ source was changed during captures. Run the pre-repair control serially with `source python_env/bin/activate; source ../lane-env.sh`, profiler/watcher unset, then `timeout 300 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.probe_mesh_fp32_all_reduce --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/fp32_ar_before.json`; console is `logs/fp32_ar_before.log`. This is a component precision check, not a latency measurement concurrent with profiler CPU postprocessing. Native loaded-object identities and C++ source hashes are recorded in the result.


## Native build identity and AutoFix evidence (2026-11-14 02:12 UTC)

The FP32 all-reduce source repair was built successfully with target `ttnncpp -j2` (`logs/fp32_ar_build.log`). Runtime component `ttnn-runtime` was installed under staging prefix `build_Release/fp32_ar_install` (`logs/fp32_ar_install.log`), then the staged `_ttnncpp.so` was copied and atomically replaced into `build_Release/lib`. Atomic replacement preserved the old inode held by the still-running profile postprocessor; it did not mutate that process's mapped library. The component run records rebuilt `_ttnncpp.so` SHA256 `cf28f221fdf91a50f429e2383ab7e343c64bef2ecec3828f25fc1805d3c5bcb0`, inode30188024, plus actual mapped `_ttnn.so`/`libtt_metal.so` identities and both CCL source hashes.

The first revised CCL test (`fp32_ar_after.json`) fixed 7 of 8 FP32 cancellation tuples; the remaining tuple was a BF16 pack tie (actual1.0078125 versus host-oracle1.0). All20 BF16-control comparisons passed. That original failed artifact remains unchanged. The separately reviewed hardware-pack oracle run (`fp32_ar_after_pack_oracle.json`) now passes all40 eager/changing-trace comparisons; its explicit reference is serial FP32 rank order followed by the hardware BF16 pack rule, with rounding-source provenance retained. This component result is not a whole-decoder acceptance claim.

The exact recurrence component passes48/48 bit-equality comparisons (`exact_recurrence_component.json`). The combined native CCL/native norm/native DRAM projection/exact-recurrence layer34 candidate still has a diagnostic anomaly: matched-HF aggregate0.999559383, minimum0.987942862 at step174 (`repair_fp32_ar_native_exact_l34_hf.json`). Thus collective repair does not remove the upstream native-family anomaly. That candidate's timing overlapped CPU report postprocessing and is diagnostic, not a basis for ranking; isolated layer0 timing remains pending.

After the layer34 run closed its devices, shared `tests/provenance.py` was extended for subsequent runs: record both FP32 all-reduce C++ source hashes and each actual `/proc/self/maps` TT native-library path, mapped inode/device and SHA256. The helper imports no TTNN and opens no hardware. It verifies the opened file's inode/device against the mapped object; an atomic replacement cannot silently assign a new file hash to an older loaded mapping. If the old mapped inode cannot be opened, hash unavailability is explicit. Historical artifacts are not rewritten. CPU checks cover mapped identity/hash, duplicate mappings and replacement mismatch; this addition does not qualify measurements made against a different native build.


## MoE attribution retry and bounded AutoFix outcome (2026-11-14)

The final profile driver completed all four captures, tt-perf-report tables/CSVs and B1/B32 performance summaries, then the MoE attribution tool rejected the B1 selector boundary because production uses `MeshFP32TopK` instead of `TopKDeviceOperation`. Updated only the CPU summarizer to select this boundary when `decode_fp32_selector=true` and batch=1, requiring exactly one matching FP32 singleton256 input, preallocated FP32 values8/UINT32 IDs8, UINT32 output8, inline data-movement kernel and no compute kernel. Native TopK must be absent in that branch. Historical/B32 profiles still require exactly one native TopK and no matching FP32 selector. Existing router, collective, final-residual, output-split, four-rank and16-replay assertions remain. Duplicate/missing/wrong-policy/wrong-dtype selector checks reject ambiguous input.

CPU-only retry command (exit0):

```bash
timeout 300 python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_multichip_moe_share --doc models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder > models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_multichip_decoder/profile_logs/moe_share_retry.log 2>&1
```

[moe_share.json](moe_share.json) now records all four captures. Critical-rank MoE kernel-work shares: linear B1 **24.3591%** (0.165511688/0.679466438ms kernel sums), linear B32 **5.80355%** (0.903055563/15.560387875ms), full B1 **53.5805%** (0.160109063/0.298819375ms), full B32 **18.7749%** (0.822257750/4.379551438ms). These are sums of per-rank kernel durations, not attribution of clean elapsed decode latency. The source boundary includes router input conversion, selected/shared experts and their output collective, excluding postnorm, final residual add and public output splitting. Historical B1 layer0/layer3 regression reproduced every preexisting numerical summary field exactly; [regression log](profile_logs/moe_share_regression.log). Repository pre-commit and synthetic selector ambiguity/policy checks passed. All summary/regression CSV processing ran after the device-timing-done notification and completed before the restored-default timing; no timing result is inferred from this CPU work.

The additional AutoFix controls do not restore the hard contract. `repair_fp32_ar_exact_l0` measured **0.869184612ms /5.96287%**, slower than current default0.719601805ms, with accuracy/replay passing. `repair_fp32_ar_native_l0` measured **0.554099732ms /9.35361%**, minimumPCC0.999129693 on layer0, still below the10% floor. The corrected FP32-collective component passes40/40 cases and exact recurrence passes48/48 bit-exact component records, but their combined native norm/DRAM-projection sensitive-layer run still has HF aggregate0.999559383 and minimum0.987942862 at step174 ([HF comparison](repair_fp32_ar_native_exact_l34_hf.json)). Aggregate success does not waive the below0.99 diagnostic anomaly. Thus the documented bounded AutoFix attempt has failed to restore the mandatory contract; this does not claim every possible kernel design is exhausted. Current-default linear FUNCTIONAL remains7.20236%, requiring27.98% less latency to meet0.518283125ms. Optional Stop Rules do not waive that failure, and no stage-review approval/completion follows. Remaining repair handoff is precise: preserve arithmetic while reducing ordered-scan/recurrence movement, or identify the native step174 norm/projection-to-router divergence; neither is an evidenced accepted gain. Restore and verify pre-experiment C++/loaded-library identity before reusing final default evidence; that restoration is caller-owned and not asserted complete here.


### Within-replay MoE device span supplement

After the caller confirmed final restored-default device timing had completed and devices were closed, reran the same CPU-only summary command with output redirected to `profile_logs/moe_share_device_span_retry.log` (exit0). Added firmware-envelope fractions alongside unchanged kernel-work fields in [moe_share.json](moe_share.json). The CSV supplies `DEVICE FW START CYCLE`/`DEVICE FW END CYCLE`, but no absolute kernel start/end timestamps; kernel duration alone cannot reconstruct the kernel envelope. Each rank/replay therefore uses explicitly labelled firmware span `max(end)-min(start)` for the established MoE operation range and for the entire layer including output split, then divides summed MoE spans by summed layer spans across16 replays. This includes intervening gaps and possible overlap; it is not exclusive causal attribution or a fraction of clean host decode latency. There is no cross-device clock subtraction, and clock scale cancels within each fraction.

On the same kernel-work-critical ranks reported above, MoE device-firmware span shares are: layer0 B1 **24.5777%**, layer3 B1 **49.5671%**, layer0 B32 **5.3124%**, layer3 B32 **14.4146%**. All four ranks and16 replays are retained for every case. Existing kernel-work fractions remain exactly unchanged; synthetic overlapping/gapped intervals and invalid timestamp checks passed, and repository pre-commit passed. No hardware was accessed by this supplement.

## Restored default and final AutoFix outcome (2026-11-14 02:17 UTC)

The two CCL C++ edits were reversed with `git apply -R doc/optimized_multichip_decoder/fp32_all_reduce_repair.patch` (full model-relative path used), rebuilt with `cmake --build build_Release --target ttnncpp -j 2`, installed into `build_Release/fp32_ar_restore_install` with component `ttnn-runtime`, then atomically copied into `build_Release/lib/_ttnncpp.so`. The C++ source hashes exactly match the pre-experiment control; the relinked library has a new hash, explicitly recorded rather than claimed binary-identical. `fp32_ar_restored_pack_oracle.json` reproduces the original FP32-cancellation failure with all BF16 controls passing. The repair remains an unapplied, component-qualified patch.

Both kinds were rerun on the restored default with `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder --layer {0,3} --length 128 --decode-steps 128 --functional-sequence --functional-repeats 3 --repeats 30 --prefill-repeats 3 --stress-repeats 3 --output .../restored_functional_l{0,3}.json`, bounded by timeout1800, serial and profiler/watcher unset. Native provenance is in each sidecar. `native_restore_verification.json` proves prefill and all128 decode outputs are bit-equal to the earlier default for both kinds. Fresh linear FUNCTIONAL0.719504474ms/7.203334% still fails; full0.340836807ms/10.105496% passes.

The bounded AutoFix attempt has failed to restore the mandatory linear floor. Correct recurrence fusion is slower; the faster native family remains below the performance floor and retains an additional sensitive-layer diagnostic anomaly. No quality gate is waived. Further exact ordered-scan data-movement/kernel work is an unresolved engineering handoff, not a claimed speedup or proof that all conceivable implementations are exhausted.

Final Python pre-commit detected formatting-only changes in two files and import ordering in four files. The measured production module was restored exactly to SHA13b403518fb8b71dcf5b502bddeb9be79bcd6acf27b1bb51ab47c4ef902d952f; its formatter-only changes are preserved as `deferred_formatting.patch` to avoid silently changing recorded source identity. Other probe-only formatting was retained. All other invoked hooks passed. The first strict gate was affected by that transient formatting change (31/33, linear floor plus source-hash mismatch), preserved as `gate_summary_formatting_interference.json`; a fresh strict check follows restoration. This blocked checkpoint does not claim formatting acceptance or stage review pass.

Raw `silu_component_retry.json` remains unmodified on persistent storage. It exceeds the repo500KB file limit and is locally ignored like raw profiles; `silu_component_retry.summary.json` preserves status, controls, metadata and comparison counts in the local checkpoint. No raw artifact was compressed or deleted.

Final strict evidence command completed32/33: only `functional_l0` fails. `logs/final_gate_check.log` and `gate_summary.json` preserve the result; no threshold changed. Final checkpoint pre-commit passes with `SKIP=black,isort` solely to preserve measured production-source bytes; all other changed Python files separately passed all applicable hooks. Formatting acceptance is not claimed. Independent stage review is requested against this incomplete checkpoint after all runs/checks have finished.


## Review round1: targeted watcher remediation (2026-11-14)

[Independent round1](stage_review_round1.md) returned **more-work-needed**: P1 is the unchanged linear FUNCTIONAL floor after the failed bounded AutoFix; P2 requested watcher coverage of non-aligned/resumed prefill. For P2, completed serial runs with `TT_METAL_WATCHER=10`, allocation tracking enabled including program-cache allocations, profiler variables unset, device-ownership checks before each job, and timeout1800 per job. Exact executed commands are retained in the three provenance sidecars. The new `watcher_paths` group reproduces the same cases with explicit lengths:

```bash
source python_env/bin/activate
source ../lane-env.sh
unset TT_METAL_DEVICE_PROFILER TTNN_OP_PROFILER TT_METAL_PROFILER_TRACE_TRACKING
bash models/autoports/qwen_qwen3_6_35b_a3b/tests/run_optimized_multichip_validation.sh watcher_paths
```

`watcher_mixed_non_aligned.json` covers both layer kinds, prefixes33/35 and resumed lengths32/34 at nonzero starts, live decode trace/shared cache, disjoint pages, and four decode steps per continuation. `watcher_l0_s1.json` and `watcher_l3_s1.json` cover singleton prefill plus four decode steps. All three pass existing numerical, saved-state, replay, runtime-fallback and allocation checks. Watcher logs contain12/2/1 completed dumps respectively and no failure signatures; [watcher_paths_summary.json](watcher_paths_summary.json) binds the logs to provenance. Production decoder bytes remain SHA256 `13b403518fb8b71dcf5b502bddeb9be79bcd6acf27b1bb51ab47c4ef902d952f`; native source/runtime remain restored. No new performance claim is drawn from watcher runs.

The strict checker now enforces this subset as `watcher_non_aligned_paths`, reusing the unchanged mixed/paired correctness checks and watcher error/dump validation. Existing stage4 default validation remains unchanged. CPU fixture/rejection checks, shell syntax and all applicable pre-commit hooks on the three changed validation files pass (`logs/watcher_paths_precommit.log`). The previous32/33 result is preserved as `gate_summary_before_targeted_watcher.json` and its separate log. A fresh strict check follows; P1 has not been waived.

Fresh strict consolidation completed **33/34**, failing only `functional_l0`; `watcher_non_aligned_paths` passes. Command: `timeout 300 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.check_optimized_multichip_evidence > .../logs/final_gate_check.log 2>&1` (expected exit1 for the unchanged floor failure). No hardware or tests remain running. Fresh diff-only review round2 is requested for P2 closure and the still-open P1.


## Independent round2 and incomplete checkpoint (2026-11-14)

A fresh xhigh, fork-free diff-only reviewer independently checked round1 findings and the new artifacts. [Round2 report](stage_review_round2.md) returns **more-work-needed**: P2 is closed by targeted watcher evidence; P1 remains the mandatory linear FUNCTIONAL failure. The reviewer independently reproduced the7.203334% result and verified new watcher provenance against selected Python/native identities. No new blocking finding was identified. A third review without a repair would not change this failed hard gate; no passing verdict or carried waiver is claimed.

The goal remains unfulfilled after the documented bounded AutoFix failure. The concrete blocker is accepted linear decode0.719504474ms/token versus required≤0.518283125ms/token. Correct exact recurrence is slower; native norm/projection with repaired CCL still misses the floor and has an unresolved sensitive-layer numerical anomaly. Further progress requires a new verified kernel/data-movement repair that preserves accepted arithmetic, or resolution of the native family's first divergence followed by enough speedup; the current attempts supply neither. This is an engineering impasse for this run, not proof that every possible design is impossible. Existing thresholds and accepted defaults are preserved. Full-model/vLLM stages must not start from this incomplete state.

Stage-owned changes are checkpointed locally on `main` in this tt-metal checkout, with no push. Raw tensors/profiles/logs remain on persistent storage; compact evidence and reproduction code are committed. Black/isort changes for the measured production module remain deferred, as already documented; all other applicable hooks pass. Commit identity follows below.

Round2 minor documentation note fixed: README reproduction-group list now includes `watcher_paths`; no code or measurement changed.


Local incomplete checkpoint: repo `/home/ttuser/qwen36-35b-a3b/tt-metal`, branch `main`, commit **`8ef853637be7da3da63aa1aa00d945ded86437cb`** (`Checkpoint incomplete Qwen3.6 multichip decoder optimization`). Parent `5ebe699604445e7b30ea0813e15aa913efbd3188`. Commit hooks pass with the documented `SKIP=black,isort`; log `logs/incomplete_checkpoint_commit.log`. Production SHA remains exactly `13b403518fb8b71dcf5b502bddeb9be79bcd6acf27b1bb51ab47c4ef902d952f`. The staged `git diff --check` reported only blank-context marker spaces in the two preserved unified-patch artifacts; those spaces are patch syntax and were retained. Implementation/doc diffs and commit hooks otherwise pass. No push occurred; working tree was clean after this checkpoint. This following documentation-only commit records its identity and does not change acceptance, implementation or evidence. Goal disposition is **blocked after failed bounded AutoFix**, with P1 and the failed hard gate explicitly open; no completed-stage signal is authorized.


## Owner decision: accept stage5 linear performance limitation (2026-11-14)

The owner explicitly instructs: “Owner decision: accept the linear-attention decode speed (0.7195 ms/token vs the 0.5183 ms floor, 7.2% of roofline) as a known limitation. Record it in limitations with this decision and the evidence, finish the review and complete stage 5. No further tuning for this floor.” Source is the multigoal runner's `05-05-optimized-multichip-decoder.resume-2.message.txt`; verbatim repo copy is [owner_decision.md](owner_decision.md). This supersedes the prior blocked acceptance disposition only. The failed AutoFix result, raw failed flags, original10% target and rejected native candidates remain unchanged.

Added `doc/eval_readiness/limitations.json`, ID `stage5-linear-decode-performance`, explicitly performance-only and `original_model: not_applicable`. It binds the exact stage/layer/batch/mesh, final restored artifact, measurements, Python/native identities, decision and supporting evidence. It grants no correctness or future-stage exemption. No new tuning or hardware work was performed.

The CPU checker now uses `restored_functional_l{0,3}.json`, validates current native identities, recomputes the same sample means/roofline, and reports a measured full-attention pass plus the exact owner-accepted linear shortfall. The inherited paired checks still execute for PCC/state/trace/fallback/layout; stage4 retains strict raw pass enforcement. Historical failed summaries remain intact (`gate_summary_before_owner_acceptance.json`). New regression coverage rejects missing/changed decisions, changed measurements/builds, quality/state/replay/fallback failures and a full-attention failure. Six CPU tests including negative subcases pass; all applicable hooks on changed checkers/tests pass.

Commands: `python -m unittest models.autoports.qwen_qwen3_6_35b_a3b.tests.test_owner_performance_acceptance -v` (log `logs/owner_acceptance_unit.log`); `pre-commit run --files <three changed validation/test files>` (log `logs/owner_acceptance_precommit.log`); `timeout 300 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.check_optimized_multichip_evidence` (log `logs/owner_acceptance_gate_check.log`). Independent round3 will review only the changed acceptance contract and disposition of round2 findings after these checks finish.

## Carried to next stage

- The owner-accepted linear decode performance is a documented planning limitation:0.719504474ms/token,7.20333% of this per-rank roofline. Do not resume tuning for this stage5 floor. This does not waive any separate future-stage acceptance requirement.
- Preserve the measured residual/serial workspace contract when stacking layers: replicated BF16 hidden2048, no inter-layer CCL/reshard, shared persistent CCL storage consumed serially and retained until every dependent trace is released.
- Apply the routine production formatting patch when a later implementation change already requires fresh provenance and affected validation. It is formatting-only; stage5 retains exact measured bytes.
- Optional untried items remain under `Not tried (stop rule)` with their original scope; they are not earned rejections or required additional stage5 tuning. Native arithmetic/CCL experiments remain unselected and need their own qualification before promotion.

Updated full CPU stage acceptance completed **34/34**, exit0 (`logs/owner_acceptance_gate_check.log`):33 measured/check passes plus one scoped owner-accepted performance limitation. The linear detail still records `raw_floor_passed=false` and0.719504474ms/token. All commands and checks are finished before fresh round3 review; raw performance JSONs are byte-identical to the previous checkpoint.


## Stage5 acceptance after owner decision (2026-11-14)

Fresh independent [round3](stage_review_round3.md) returned **pass-with-carried**, with no Required Work. P1 closes by the explicit owner decision, not a speed repair; P2 remains closed by its previously reviewed watcher evidence. The reviewer independently recomputed sample means/fractions, verified raw artifacts and current native/Python identities, checked preservation of the old33/34 summary and unchanged non-FUNCTIONAL rows, reran all six CPU tests plus additional negative controls, and found no new correctness defect. Existing `Carried to next stage` notes remain the handoff; no further stage5 floor tuning is requested.

Stage5 is **COMPLETE under the owner-revised contract**: acceptance34/34 comprises33 measured/check passes plus the specifically accepted linear performance shortfall. The floor target and failed raw flags remain unchanged. Production default and all reported before/after performance remain byte-for-byte as measured. No hardware experiment, full-model/vLLM work, rejected-candidate promotion, or push occurred in this acceptance follow-up. The earlier blocked/incomplete entries and reviews are chronological evidence superseded only for stage acceptance by the owner decision and round3 verdict.

The passing stage-owned local checkpoint follows this review; its SHA is recorded below. Original implementation/evidence checkpoint8ef853637be7da3da63aa1aa00d945ded86437cb and SHA-log commitbdba2e3af7ae463d4666429a56b4042c3c2ce4ca remain in history.

Accepted stage5 checkpoint: repo `/home/ttuser/qwen36-35b-a3b/tt-metal`, branch `main`, SHA **`74014c9b12bf29e2be83968bd8bd06a4445993e2`**, subject `Accept stage 5 with owner-approved linear decode limitation`. All applicable commit hooks passed, including Black/isort for the changed checkers/tests (`logs/owner_acceptance_commit.log`); measured production was untouched. Working tree was clean after the acceptance commit; no push. This following documentation-only commit records the SHA.
