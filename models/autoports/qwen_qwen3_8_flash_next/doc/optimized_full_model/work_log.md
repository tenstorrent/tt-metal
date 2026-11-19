# Stage 7 work log

Started 2026-11-14 from completed stage6 checkpoint
`de432e04` on main. Target is the existing four p300c Blackhole devices,
1x4 TP4/EP4 with the p300_x2 descriptor and ring fabric. Device ownership
check found no holders; bounded `tt-smi -ls --local` lists all four devices.
Both required bringup and autodebug plugins are installed and enabled.

Preserve the inherited selected precision (including physical-fit expert BFP4),
BF16 KV, hyperconnection residual stream, fixed-slot serving contract and
262144 context. No datatype frontier or vLLM integration. Unrelated untracked
fused-decoder artifacts are excluded from this stage.

Initial evidence: stage6 clean 128/128 B1 TTFT 7038.672ms, token-out
138.754269ms (7.206985t/s/u), model-only 138.152610ms. Stage5 weighted
decoder estimate is110.300609ms, measured with C128 rather than C262144.
Reduced stage6 terminal profiling identifies capacity-dependent exact QSA
scoring as a leading gap candidate. This is a hypothesis pending a controlled
measurement, not an additive full-stack attribution.

Commands use `source models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/environment.sh`.
All hardware jobs are serialized and bounded. Fresh baseline command:
`TT_METAL_TRACE_ALLOC_TRACKING=0 timeout --signal=INT --kill-after=60 3600 python -u -m models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_acceptance --perf --capacity 262144 --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/baseline_perf.json`.

Independent CPU bottleneck audit delegated under the selected skills; no
device job or implementation edit delegated. Final stage review remains pending.

## Fresh baseline

Baseline clean run exited0 after252s with clean teardown. Warm B1 128/128
TTFT 7036.368957ms; caller-read token-out 138.780732ms (7.205611t/s/u); model-only
fixed-token trace 138.143508ms. The model-only metric is not teacher-forcing.
The second baseline uses the resident harness and31/32/33/128/1024 TTFT lengths.

Independent audit confirms selected scorer uses accurate SFPU sequential tile fold,
not the historical explicit half-tree. Candidate therefore emits per-head FP32 dots
using the selected sum order, then retains ReLU/head-sum/scale and native selector.
No short-prefix zero-score shortcut: even when all keys fit the budget, ordering
affects SDPA arithmetic. Exact ordered-ID comparisons are required.

The resident baseline completed its timed window at138.618793ms (7.214029t/s/u),
versus caller-read138.7927ms (7.204990t/s/u); full run still finishing TTFT rows.
Thus readback is not the leading gap.

Resident baseline exited0 after414s. Full result baseline_resident.json includes
zero interior readbacks/synchronizations/refreshes,127 model+127 sampling replays,
matching first/final token to the public loop and all48 QSA rank-position checks.
Warmed TTFT rows: [{"prompt_tokens": 31, "generated_tokens": 2, "batch": 1, "ttft_ms": 2485.757440328598, "scope": "Warmed public generate: reset/setup, complete prefill, first device sample and caller read"}, {"prompt_tokens": 32, "generated_tokens": 2, "batch": 1, "ttft_ms": 1783.9990411885083, "scope": "Warmed public generate: reset/setup, complete prefill, first device sample and caller read"}, {"prompt_tokens": 33, "generated_tokens": 2, "batch": 1, "ttft_ms": 1922.813924960792, "scope": "Warmed public generate: reset/setup, complete prefill, first device sample and caller read"}, {"prompt_tokens": 128, "generated_tokens": 128, "batch": 1, "ttft_ms": 7035.653050988913, "scope": "Public generate: reset/setup, complete prefill, first on-device sample and caller token read"}, {"prompt_tokens": 1024, "generated_tokens": 2, "batch": 1, "ttft_ms": 56733.39607100934, "scope": "Warmed public generate: reset/setup, complete prefill, first device sample and caller read"}]

## Experimental scorer stall (11:22 UTC)

First component probe `timeout --signal=INT --kill-after=60 600 python -u -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_causal_indexer_scores --rows 1 --starts 0,3,127,2051,32767,205504,3 --iterations 8 --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/causal_q1_v1.json` stalled after fabric initialization without case output. No production wiring. Live tt-triage attachment PID4106066 blocked in `tenstorrent_set_aggregat`; no device-state report obtained yet. Host GDB attachment PID4109420 exceeded its20s timeout and was killed (diagnostic process only), releasing ptrace; target remained alive under its original600s timeout. AutoDebug initial child sandbox preflight failed because unprivileged namespaces are unavailable. Retried source-only child with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under existing unrestricted workspace authorization; diagnosis pending. Added upload/warm/capture phase markers to the next probe version to distinguish input setup, baseline, and candidate phases.

At11:32UTC the diagnostic tt-triage process had exceeded its120s timeout by more than6min, blocked in the driver before creating any report. SIGKILL was queued for this run-owned diagnostic PID4106066 so it cannot resume into a subsequent recovery job. The model probe remains under its original timeout. Independent source audit found balanced CB counts at zero/one/multiple live groups and no proven deadlock. Probe exact-score requirement tightened from tolerance-only acceptance to bitwise equality plus ordered-ID equality.

The bounded probe ended via its configured SIGKILL grace at660s, exit137 (wait_for exit1). At11:33:30UTC both target and triage PIDs were gone and fuser showed no chip holders. Began serialized list/reset/list recovery1.log. AutoDebug found no source-proven stall cause; source-only report preserved as AUTODEBUG_causal_q1_v1.md. AutoFix begins with diagnostic localization plus strict exact-score acceptance, not a speculative kernel change. Kernel dmesg and /proc/stack were unavailable by host permissions.

Recovery1 list/reset/list exited0 in45s; all4 devices visible. recovery1_smoke.log exited0 in6s with all-rank arithmetic and cleanclose. Synthetic native raw probes raw_q1_v1.json (Q1 positions0/3/127) and raw_multi_v1.json (Q3/Q32, positions0/3/127/480 plusselector) passed bitwise dots, finite values, invalidzeros and orderedIDs on all4ranks, using128compressedkeys/oneworker.

Second full-capacity probe causal_all_v2.log stalls at INPUT_UPLOAD_BEGIN, before KEY_UPLOAD_DONE. This localizes failure before either scorer. CPU audit identifies fixturekeys alreadyBF16/contiguous, so .bfloat16().contiguous() retains memory-mapped file storage. TTNN borrows it; mesh4 counts4x16MiB=64MiB over32MiB pin threshold, invoking KMD FOLL_LONGTERM page pinning. Applied minimal .clone() to keys in both probes, plus rawfixturequeryclone. Candidate implementation unchanged. Await original600+30s timeout/recovery then test cloned upload as the discriminating AutoFix check.

Second mapped-key probe exited137 at630s via configured timeout grace; PID4147305 gone and allchip holders clear. recovery2.log runs bounded list/reset/list plus allrank arithmetic smoke. The saved causal_all_v2_host_state.json records247million majorfaults versus6.08s userCPU and198.95s systemCPU at11:39:31UTC, consistent with a host-kernel fault loop rather than a native scorer invocation; exact kernel PC remains unavailable.

CPU preparation: added per-component eager profiler signposts for fullweighted layer reconstruction; added optional public decode_forward resident benchmark separate from unchanged private-sequence baseline; prepared fixed-policy LMhead geometry sweep and sampler-only barrierasync comparison. No production modification yet. precommit harness_hooks_v4.log passed after formatter-only changes in v3.

Cloned upload fixed the probe: causal_all_v3 exited0 in18s, all27 exactscore/orderedIDcases passed. Q1pos1272.3966→1.0615ms; Q32pos12750.8292→6.4669ms; Q32nearmax53.7610→192.8828ms regression. Candidate remains experimental. A Q1 packed-head output variant is being implemented to reduce padded32MiB dot output and head transposes; a source audit is investigating RISC querybroadcast costs atlongprefix. Fixed-policy head_sweep_v1 began with11 geometrycandidates, preservingBF16/HiFi4/FP32DEST/packer-off.

Head sweep v1 rejected c8_r1_b1: static DFB3108800B exceeds1572864B L1 during warm; no timings claimed. Clean teardown and fuser clear. recovery3 list/reset/list plus mesh arithmetic exited0. Retry v2 excludes only proven single-reader allocation failure and retains2/3-reader geometries.

causal_packed_v4 passed36 exact score/ordered-ID cases Q1/3/17/32 and boundary/long positions. Hardware ROW broadcast removes scalar query expansion; Q1 packs four heads into one physical row tile and reuses keys. Q1pos1272.39953→0.318615ms;Q32pos12750.8308→4.06262ms;nearmax Q15.31008→3.40069,Q3253.7218→12.2649ms. Wired only MeshQSA selection; dtype/layout/selector unchanged. Head sweep v2 passed10geometriesQ1/Q32, preservedpolicy. Select20cores/3readers/block2: projection+greedy Q11.193880→1.183168ms,Q321.169561→1.160073ms. This synthetic-activation geometry result requires integrated real-activation validation.

Sampler-only barrierasync candidate passed changedwinners, ties,greedy/topk/topp and rankorder in clean and allocationtracked runs (sampling_ccl_v1.json,sampling_ccl_tracked_v1.json). Rejected for latency: sync0.47461–0.47481ms,async0.48079–0.48116ms. Retain existing synchronouscandidategathers with trace-residentbuffers; commonTT_CCLsemantics retained. No forceargmax/fullvocabgather introduced. Integrated reduced(0,1,3) B2 fixedslot mixedprompt serving probe now runs separately under watcher10+allocationtracking.

serving_watcher_v1 exited0 after78s; all16serving/trace/statechecks passed under Watcher10+tracking1(skipprogramcache0). PreservedWatcherlog. Integratedformat-onlyhooks v1appliedblack/isort before run;v2passed. Began selected_resident_v1 samebaselineharness plusseparatepublicresidentmeasurement and TTFT1/2/4/8/16/31/32/33/128/1024, all48C262144 cleantracking0.

selected_sources_v1.json records model/native/common-sampler source provenance at selected_resident_v1 launch. During its execution, causal_indexer_scores.py docstrings alone were corrected from experimental/notwired to integrated; executable code and kernels unchanged. No performance or correctness claim depends on the docstring edit.

selected_resident_v1 exited0 after363s with cleanclose. Sameharnessresident113.881454ms/8.781061t/s/u vs138.618793ms/7.214029; publicresident113.846141ms/8.783785;callerread8.774341. TTFT1284805.506msvs7035.653ms. Everytimedresidenthostread/sync/refresh/capture/releasecounterzero;127model+127samplerreplays, all12QSAlayersx4rankspos255, first/finaltokensmatchedpubliccontrol. TTFT31/32/33/1024=1902.735/1225.489/1342.396/38885.362ms; finalonlybucket1/2/4/8/16rowsrecorded separately.

All128greedytokens match baseline exactly in selected_resident_v1. Terminalclean_v1 exited0 after53s: reduced0/1/3C262144 splitgreedy0.481979ms vsforceargmax2.765952ms on samefrozenreallogits; splitselected. These samplertrace timings exclude model andhostread. Began final_validation_v1 all48tracking1 AIMEaccuracy/resume/autoreg/qualitative/32769context, with complete model/prefill/sampler/capture runtimeguard enabled. validation_hooks_v1 passed.

StrictFUNCTIONALaudit: preparedmajorweights+shortKV6.414528ms at512GB/s/rank;113.881454ms achieves5.6326%, belowmandatory10% (<=64.14528ms). Selectedgraphintermediate12.975296ms isNOTacceptabledenominator. functional_floor_audit_v1.json recordsfailedgate. AutoDebugsourceinvestigation forpolicy-preservingexactprojection/GR/CCLfusionactive; AutoFixrequired, noacceptanceclaim. FreshAIMEprefilltop1=.97/top5=1/top100=1;TF.96/1/1 and7.986209t/s/u trackedcallerboundary (final_validation_v1stillrunning).

12:20UTC CPU preparation: copied the unchanged stage6 blind-comparison packet builder into stage7 qualitative/; it will require all ten sampled seeds on each side before preparing an immutable label-hidden packet. Native local2560 residual-write and exact Q1 projection plans retain every selected rounding and reduction order. Larger fixed-policy head-block candidate geometry is recorded in head_larger_blocks_plan_v1.json; these rows are not device measurements. Current full48 validation remains the only device job.

Direct AIME128-token content review: both fresh TT and pinned HF prefixes coherently restate the9km/speed/coffee-time problem and begin the correct distance/speed setup; both reach the shared token cap before a solution. No text degeneration observed. This is not an AIME solved-answer score; see aime_autoregressive_review_v1.json and the separate top-k gates. Stage7 perf_summary.json now records paired clean metrics and explicit nulls for pending selected device/layer timings, retaining the failed strict FUNCTIONAL gate.

All66 fresh TT qualitative outputs completed; prepared immutable120-example matched sampled blind packet and dispatched a fresh label-hidden reviewer. The32769-token context smoke began in the same full48/tracked/runtime-guarded process. Installed isolated AutoDebug retry completed exit0; report saved as AUTODEBUG_functional_floor.md, root AUTODEBUG.md restored to HEAD. AUTOFIX_functional_floor.md records verified causes, prioritized exact-math experiments and promotion gates. New test-only candidate modules added during validation are not imported by the production path.

Fresh label-hidden qualitative review completed: all120 sampled examples scored under the frozen rubric before labels joined;0/60 HF and0/60 TT failures, each of six prompts within the20-point margin. Direct six-greedy review likewise found coherent matched behavior. Capped thinking/incomplete code and legitimate serialized EOS boundaries are explicitly limited, not complete-task success. See qualitative/verdict.json, blind_ratings.json, blind_judge_notes.md and blind_comparison.json.

12:48UTC CPU-only review while context smoke owns all4 devices: B32 resident harness now rejects any Watcher environment presence, checks stable program-cache counts across127 timed steps, and distinguishes Q1 per-slot attention/head from packed32 MoE and dyadic prefill. HF and mixed-prefill harness source manifests now include all model-local Python/native kernels;12 existing CPU checks passed. Fresh component timing harness and profile recipes are under independent alias/coverage review. Found/fixing missing hashlib import and incompatible historical six-marker validator; terminal profiling will stage a real128-token prefix so causal work matches the resident window.

Exact projection experiment now has an optional zero-once CB initialization variant. Conservative per-tile clearing remains default. Two query and two weight CB pages are initialized before first push; only identical regions are overwritten thereafter and compute never writes inputs. CPU simulation matched all physical bytes over5208 reuse states; hardware exactness/performance remain unmeasured. Exact residual, norm and rank-sum probes likewise remain test-only. No production executable change since selected full48 validation began.

12:53UTC final_validation_v1 exited0 after2757s with clean meshclose. AIME prefill97/100 top1,100/100 top5/top100; teacher forcing96/100 top1,100/100 top5/top100. Resume/autoregression/qualitative pass. Nonaligned32769-token smoke took1273.083s;1025 prefill replays,1 decode,2 split sampling replays, no new captures, all48 QSA/rank positions32770, finite terminal logits. Extra model+sampler feedback replay has no token/position/page refresh or readback and reaches32771. Complete runtime audit passed12 prefill,2 decode,3 common-sampling and8 complete-capture calls with zero failures and restored bindings; positive controls reject torch and host conversions. Context contract appended stage7 smoke/memory without changing262144 capability. Selected final free182426112B/bank. Terminal reduced profiler launched next, no overlapping device jobs.

12:58UTC terminal device profile passed(real128 prefix, measured position132, both greedy strategies select1292); mesh cleanclosed at12:55:48. CPU Tracy processing continues. With fuser clear, ran correctness-only projection smoke(v1/v2, skiptrace/no timings). v1 exits1 before native launch: TTNN Shape disallows slicing; converted to tuple before slice. v2 native kernels run/close cleanly; BF16 weights with BF16 andFP32 inputs pass, but first FP32-weight ramp fails raw/output bits. Source-only child investigating unpack/reconfiguration, no production promotion. No device hang/timeout observed; no reset needed for these deterministic validation/numerical failures with cleanclose. Writer tracked+Watcher10 probe follows serially; its timings are diagnostic, not clean performance, while CPU report processing continues. Experimental hooks v2 passed23files after formatting-only v1 changes. Installed stage07 check exits0; its degeneracy-only scope is not full-stage acceptance.

13:01UTC projection smoke v3 exits0 after8s: all8 shape/dtype combinations and changed operands pass rawFP32/finalBF16 equality after configuring both SrcA/SrcB to queryCB2 for ROW broadcast. BF16 broadcast uses SrcB; prior FP32-weight startup left that port in the wrong format. Native projection remains test-only pending real-role trace/performance. Terminal profile+all11tt-perf-report region tables/CSV/advice finish exit0 after389s; strict coverage72520 host/device/final operations and all24 markers pass. Clean matched-capacity component trace harness launched next. Writer tracked+Watcher10 passes Q1/3/17/32; norm tracked passes all3 real-weight groups with RHS andadd-rsqrt variants. Rank-sum v1 failed fixture padding assertion before native execution; repair pending.

13:12UTC exact kernel component results and harness failures recorded in AUTOFIX_functional_floor.md. Native writer/norm RHS and R16 GR-up/GDN-out pass exact component controls; minimal combined candidate being integrated. GR-down/router remain unchanged because native candidates are slower. Strict full-model performance floor still failed.

13:16UTC exact integrated reduced serving Watcher exited0 after58s, all16 state/trace checks pass, tracker includes program-cache allocations. Preserved watcher log and exact_integrated_sources_v1.json. All48 clean resident benchmark launched; no other device owner. Validation harness now stores qualitative results under the fresh run stem, preserving earlier qualitative outputs unchanged.

13:22UTC exact_integrated_resident_v1 exited0 after343s: all48 C262144 resident103.068541ms/token,9.702282t/s/u, TTFT4700.293363ms. All128 greedy tokens equal initial baseline and selected_v1;127 model+127 split-sampler replays, no interior refresh/read/sync/capture/release, persistent input IDs unchanged and all48 QSA/rank positions255. Strict roofline fraction6.2236% remains below10%;38.923261ms reduction still required. Larger fixed-policy head blocks now measured; compact-GR probe preparation isolated.

13:25UTC head_larger_blocks_v1 failed at c10_r3_b8 during warmup, cleancloseexit1. Source audit identifies same-program L1 storage/DFB overlap104640B; candidates retain only DRAM weights/configs, not prior L1 outputs. Adapted c20_r3_b8 retains estimated102208B margin; v2 reruns safe subset for actual correctness/timing. c20_r3_b10 estimated94400B overlap; adapted80-core storage geometry is a separate feasible candidate, not yet measured. Compact probe v1 failed old-baseline vs host raw-bit control before native kernel; original upload preserved bits. Diagnosing old slice/concat nonfinite behavior without weakening finite controls.

13:26UTC adapted larger-head runs v2/v3 both pass Q1/Q32 correctness/greedy with unchanged BF16/HiFi4/FP32DEST/packer-off. Blocks5/8 and adapted40/80-core block10 are slower than20/3/2, retained. v3 proves large-block family was adapted beyond first L1 error. CSV/log/provenance in head_larger_blocks_v2/v3; one-reader original and c10/b8 same-program physical collision remain scoped blockers, not blanket rejection.

13:30UTC exhaustive legacy compaction canonicalization exits0 after5s underWatcher/tracking, all65536BF16patterns +518Q1controls/rank. 509exceptions (signedzero/subnormals→+0;NaNs→signedInf). Native notyetintegrated; exact bit normalization repair in progress.

13:32UTC canonical compaction kernel mirrors exhaustive legacy mapping; actual C++ helper CPU-compiled and compared to all65536patterns plus65536paired-word controls, exact map SHA matches diagnostic. Production remains unwired. Strict watcher_v3 now tests allQ1..32 plus exhaustiveQ32 and518Q1controls/rank, no finite/nonfinite exemptions. Earlier native helper docstrings corrected after selected measurement; executable behavior unchanged by those documentation edits. pre-commit exact_integrated_hooks_v1 formatted only two Python files; final rerun pending after integration.

13:39UTC canonical compaction promoted after compact_gr_down_watcher_v3(allQ1..32/all65536encodings/changedreplays) and clean_v1 pass. Q1 .040585248→.006272341ms; Q32 .040602310→.013978861ms. Production dispatch guarded to Blackhole1x4 BF16 standard tile DRAM Q1..32 four80-valid/96-physical segments. All42 changed code hooks pass after isort-only repairs in v1 and rerun v2. Frozen final_sources_v1.json precedes integrated serving/full48 validation.

13:41UTC final_serving_watcher_v1 exits1 after50s: mixed-prefill logits row0 differs from solo. This is a real integrated contract failure after compaction; cleanclose and Watcher log preserved, no reset. Standalone exactness is insufficient. Compaction production dispatch reverted; prior exact candidate retained. CPU-only AutoFix investigation continues; no compaction performance is credited. final_sources_v1 belongs to rejected integration, not final selected code.

2026-11-14T13:42:08.310253+00:00 Restored-source final_serving_watcher_v2 exited0 after56s, all16checks pass. final_validation_v2 launched --accuracy --resume --autoregressive --qualitative --context-smoke with tracking1, Watcher/profiler off, timeout7200, wait_for7245. Selected production source final_sources_v2.json; exact_integrated_resident_v1 remains its matching executable-path performance (only formatting/docs/helper additions differ). No compaction gain credited.

2026-11-14T14:10:19.148571+00:00 Final-source AIME prefill97/100/100 and teacher forcing96/100/100 pass;99decode steps/11.262190589s=8.790474572t/s/u at tracked caller boundary. Resume257 passes. Direct AIME HF/TTprefixes equalpreviousoutput, coherent/capped, reviewv2saved. All66qualitative raw/logical/textoutputs equalpriorcandidate; freshblind120sample review0/60HF and0/60TT, allsixprompts within20pp. Labelsjoined onlyafterratingsfrozen; final_validation_v2/qualitative/verdict.json recordscaps/markers/meterlimits. Existing matched-HF cap limitation registry appended freshevidence. Installedstage07_check_v2 exits0(textscopeonly). Context32769smoke stillrunning; no overallpass claimed.

2026-11-14T14:25:19.331145+00:00 final_validation_v2 exits0 after2628s(wrapper), clean meshclose14:24:33. All requested full48phasespass. Context32769smoke1247.781030s,1025prefill/1model/2sampler replays, no newcaptures, all48QSA/rank entries32770; postdiagnostic extra model+sample reaches32771 withoutinputrefresh/read/sync. Terminalfiniteall4. Runtimeaudit12prefill/2decode/3sampling/8completecapture calls succeed,positivecontrols/bindingrestorationpass. Finalfree191643648B/bank,1533149184B/rank; context262144unchanged. Batch32HF launch: tracking1 timeout7200, --batch-size32 --capacity512 --output final_batch32_hf_v1.json; wait_for7245, no overlapping hardwarejob.

2026-11-14T14:35:27.921719+00:00 CPU evidence audit before final measurements: resident benchmark now snapshots all72 model Python/native/header files, three shared sampling/CCL sources and both harnesses before setup and after performance. Snapshot changes fail outside timing windows. `final_sources_v3.json` extends frozen v2 coverage with router_topk_algorithm.hpp and supplemental shared-source hashes; all prior model hashes remain unchanged. Terminal profiler also records shared tt_ccl.py. Self-audit found a missing custom-tile rejection in causal_indexer_scores; standard32x32 selected path remains supported. Minimal rejection-only patch and25 CPU metadata cases are prepared under causal_standard_tile_guard*, but production remains unchanged while B32HF runs.

2026-11-14T14:46:30.130388+00:00 final_batch32_hf_v1 exits0 after1246s with cleanclose. Both2704-prediction runs top1=2645/2704,top5/top100=2704/2704; exact history logits/tokens and finalstate across reordered requests/slotmigration, dormant/allinactive checks and residentsampledfeedback pass. Raw JSON/log retained; compact summary final_batch32_hf_summary_v1.json. Applied rejection-only causal standard-tile guard after close;25metadata checks and45file precommit final_code_hooks_v3 pass. final_sources_v4 records only this guard delta fromv3; all72model hashes match afterhooks. Finalclean resident launch: tracking0, timeout1200, benchmark_full_model_resident --perf --public-resident --ttft-lengths31,32,33,128,1024 --capacity262144 --output final_resident_v1.json; wait_for1245. No overlapping devicejob.

2026-11-14T14:52:36.017536+00:00 final_resident_v1 exits0 after358s, cleanclose14:51:59. Resident103.064959568ms,9.702618661t/s/u; TTFT4705.950625ms; caller9.692458200; public9.705529011. All128tokens equal originalbaseline;77sourcehashes unchanged. Componenttracev2 launchedtracking0, timeout600/wait645, no overlapping devicejob.

2026-11-14T14:53:38.156514+00:00 component_trace_v2 exits0 after58s cleanclose14:53:10. Representative35/1/12 stack102.507076256ms +entry/terminal/sampler1.576551423ms =104.083627679ms vsfullresident103.064959568ms; no positive >10–15% orchestration gap. These reduced frozen-input traces are a representative estimate, not an exact mathematical all48 lower bound. Splitgreedy0.481938393ms. Terminalprofilev2 launched via run_terminal_final_v2.sh timeout5400/wait5475, profiler1, Watcher/trackingoff.

2026-11-14T14:55:05.162386+00:00 Terminalprofile devicecapture passes and cleancloses14:54:47; confirmed no deviceholders before full48 mixed/resumedprefill launch. Tracy CPU postprocessing continues independently. final_prefill_signatures_v1 runs tracking1 --all-layers --shared-kv --max-batch-size2 --check-decode-submission-order, timeout3600/wait3645. No device overlap; no clean performance run overlaps profiler CPU processing.

2026-11-14T15:00:07.195053+00:00 terminal_final_driver_v2 exits0 after361s; strict coverage68024host/device/final keys,24markers, leafpartition passes. All11region tt-perf-report tables/CSV/advice complete under terminal_final_perf_v2. CPUanalysis assigned final_sources_v4 +component_trace_v2 +final_resident_v1. Full48 mixedprefill continues as soledevicejob.

2026-11-14T15:01:07.551878+00:00 final_prefill_signatures_v1 exits0 after326s, cleanclose15:00:28. All48B2 sharedKV undertracking passes mixed/reversedlogit equality, resumes68..257 and31..265 exact, retainedtraceIDs, no requesttime releases/latecompilation, matchedresetdecode submissionorder. Allchecks pass postguard sourcev4. Prefillprofilev1 launched timeout5400/wait5475; sixactualbuckets1/2/4/8/16/32, reduced0/1/3, tracking/Watcheroff, eager/no livecaptures.

2026-11-14T15:04:41.033447+00:00 Prefillprofile devicecapture completes allsixbuckets and cleancloses15:03:50; CPUTracyprocessing continues. Before stopruledeadline, launch isolated rejectedcompaction diagnostic --mode serving-traces, watcher10/tracking1, timeout600/wait645, outputcompaction_serving_repro_v1.json. Monkeypatchdispatch onlyinside diagnosticprocess; selectedproduction remainsv4 unchanged. Purpose: reproduce integratedmismatch, notpromote candidate orclaimtiming.

2026-11-14T15:05:25.194802+00:00 Finaldecode CPUaccounting in terminal_profile_analysis_v2.md/json: weighteddevicekernel sum99.939984ms, representativecomponentstack+terminal104.083628ms, full48resident103.064960ms. No >10%orchestrationgap. Splitgreedy0.426224ms(0.414%decode), notdominant; no selectedforceargmax/fullvocabgather/genericTopK. Largestremaining GRdown product/reduction10.354546ms withtestednativealternativesslower; QSAepilogue2.682108mszero-costceiling. All72model+3sharedsourcehashesmatchv4. Strictfloorstillmisses38.919680ms. Rawgeometry,dtype/fidelity,advice andattributionrangesrecorded; profiler fieldsupdatedinperf_summary.

2026-11-14T15:07:28.348791+00:00 compaction_serving_repro_v1 exits1 after80s, reproduces exact mixed-vs-solo mismatch; cleanclose15:05:59. Candidate remainsrejected, productionunmodified. Followed multichip failed-run reset rule after fuserclear: recovery4.sh bounded list/reset/list +1x4ringopen/close, exit0 after52s MESH_SMOKE_OK; no hungjob orotherprocess killed. Before15:09:21cap launch second diagnostic --mode eager-operands tracking1/Watcher10 timeout600/wait645 to discriminate captured-realoperand arithmetic from integrated trace/history behavior. Selectedsource stillv4, no candidate promotion.

2026-11-14T15:08:50.039997+00:00 compaction_eager_operands_v1 exits0 after57s, cleanclose15:08:24; realQ1/2/3/32 GRoperand arithmetic and finalstate/logits comparisons pass when both baseline/native are evaluated. This does not waive native-only serving-trace failure: instrumentation/baseline-before-native may condition LLKstate or alterlifetime. CPUdiagnosis requested; compaction remainsrejected and no selectedsource change.

2026-11-14T15:09:45.060711+00:00 STOP RULE FIRED: user confirms stage time/token cap. No new optimization experiments or optional devicejobs. Selectedv4 retained. Prefillprofile exits0 after486s with89740reconciledoperations/all6buckets; CPUanalysis finishes evidence. Sole remaining required devicegate batch32_resident_v1 alreadylaunched tracking0 timeout7200/wait7245 afterTracyCPUfinished andfuserclear. Finish this once, independentstage-review, evidence andlocalcheckpoint; failedFUNCTIONALfloor isnotwaived bycap.

2026-11-14T15:10:42.326455+00:00 BoundedAutoFix outcome: failed torestore mandatoryFUNCTIONALfloor afteractual exact-kernel/head/CCL candidates andfinaldefaultreproduction. AUTOFIX_functional_floor.md records103.064959568ms vs64.145280ms required,38.919679568ms deficit; no hiddenorchestration/samplergap andno theoreticaldenominatorinflation. This is notphysicalimpossibilityproof. Capdoesnotwaivegate; stageunaccepted pendingfinalevidence/review/checkpoint. Compactiondiagnosisv2 scopes native-onlyrow1finite mismatch vsconditionedeagerpass, no rootcauseclaim.

2026-11-14T15:11:42.073391+00:00 PrefillCPUanalysis complete: prefill_profile_analysis_v1.md/json, all89740operations/all6buckets/all72model+3sharedsourcesmatchv4. ReducedQ1/2/4/8/16/32 devicesums8.177249/17.827547/22.579935/34.278169/53.380053/87.538737ms; Q32orderedindexer24.091004mslargestidentifiedkernel. Adviceandrawlogical-shapemetadata caveatdocumented. Full48deviceprefillestimate intentionallynull; actualfull48TTFT4705.950625msseparate.


## Not tried (stop rule)

The user-confirmed four-hour cap ended optimization at15:09:21UTC from the
11:09:21UTC stage start; the owner recorded the stop at15:09:45UTC. No new
optimization experiment is authorized. Last-two-kept<3% is not the trigger:
the latest kept exact bundle improved latency9.495%, while slower geometry
and failed compaction trials are not kept candidates.

Final [resident measurement](final_resident_v1.json) is103.064959568ms/token,
9.702618661t/s/u, with4705.950625ms TTFT and128 caller-control tokens equal to
baseline. [Matched-source component traces](component_trace_v2.json) estimate
102.507076256ms decoder stack +1.576551423ms entry/terminal/sampler =
104.083627679ms; no positive10–15% orchestration gap remains. These are
representative frozen-input traces, not a physical all48 lower bound.
[Bounded AutoFix](AUTOFIX_functional_floor.md) failed to restore the mandatory
FUNCTIONAL floor:6.223772% versus10%, requiring latency<=64.145280ms, a
remaining38.919679568ms reduction. The cap does not waive this failure or prove
physical impossibility; the stage remains unaccepted.

The following expected gains are unknown. Listed bounds are contained-operation
zero-cost ceilings from [final decode analysis/provenance](terminal_profile_analysis_v2.json),
with its [tables](terminal_final_perf_v2/model_perf_report.txt) and
[CSV](terminal_final_perf_v2/ops_perf_results.csv). They overlap and must not be
summed or advertised as achievable savings. All72 model+3 shared hashes match
[final sources v4](final_sources_v4.json); no graph inflation, dtype search or
changed residual policy contributes to the useful-weight/KV denominator.

| Not tried | Bound and next constraints |
|---|---|
| Exact QSA score epilogue/head reduction/ordered-selection fusion | Epilogue2.682108ms; containing score/selector family3.777024ms. [Scorer](../../tt/causal_indexer_scores.py) and [selection](../../tt/multichip_decoder.py): preserve exact head tree, ties, sentinels and device-owned positions. |
| Selected sparse-KV gather optimization | Family2.865048ms. [Fused decoder](../../tt/fused_decoder.py): preserve page mapping, selected order, awkward tails and trace inputs. |
| Exact QSA indexer / GDN QKV / recurrence scheduling | Respective families5.260452/4.764354/2.553868ms. [Ordered projection](../../tt/ordered_projection.py), [multichip decoder](../../tt/multichip_decoder.py): retain selected arithmetic order/casts and fidelity. |
| Different exact GR-down streaming schedule | Whole local down11.455886ms, product/reduce subset10.354546ms. [Native experiment](../../tt/exact_projection_q1.py): earlier R2/R4/R8/R16 schedules lost; new scheduling needs raw FP32 equality, real-weight performance and integrated controls. |
| Small PLE/shared/attention matmul L1/block/subblock configurations | No defensible gain estimate. [DRAM helper](../../tt/dram_projection.py) and inherited role tables require legal K geometry and unchanged fidelity. Larger head blocks are already measured slower; specific L1 failures are retained. |
| Safe persistent-output sampler gather | No gain forecast; current entire eager split .426224ms is only a contained-work ceiling. [Probe](../../tests/probe_sampling_collective_policy.py): current persistent-output factory disables the replay barrier; any new design needs changed-winner/tie replay correctness and buffer ownership proof. Existing async-barrier trial was slower. |
| Multirow exact projection/indexer row-load reuse and further prefill scheduling | Unknown gains. [Completed prefill analysis](prefill_profile_analysis_v1.md) identifies Q32 ordered indexer24.091004ms in the reduced body, not full48 or free savings. [Generator](../../tt/generator.py) must retain six buckets, mixed/fixed/inactive slots, arbitrary valid lengths and explicit state. |

The completed [prefill JSON/provenance](prefill_profile_analysis_v1.json) covers
89,740 operations, six buckets and semantic advice review. Its alternative
reduced eager sums8.177249/17.827547/22.579935/34.278169/53.380053/87.538737ms
cannot be added, scaled48/3 or subtracted from public TTFT. Whole-body markers
lack a validated layer-class partition, so full48 device-prefill accounting is
unavailable. Some CSV descriptors retain32[2]; actual runtime logical counts
and all-rank positions establish the bucket, while descriptor reuse remains
an unproven explanation.

Compaction is a measured integrated numerical rejection, not an untried item.
[Native-only serving reproduction](compaction_serving_repro_v1.json) fails;
[conditioned eager operands](compaction_eager_operands_v1.json) pass without
resolving trace/lifetime/state behavior. Dispatch remains reverted. Isolated
speed and canonical-bit controls do not justify promotion. Inherited rejected
precision/KV/CCL policies and replicated/hidden-sliced residual alternatives
remain excluded by the user contract. B32 clean throughput is the already-running
remaining device gate; final review, evidence reconciliation and local commit
remain required. No successful stage or future gate is claimed here.


2026-11-14T15:21:48.829436+00:00 Final required device gate batch32_resident_v1 exits0 after726s, cleanclose15:21:19. All48/C256,32 active slots,125..128 natural prefix lengths,127 resident public decode steps:2558.592232ms/step,0.390839927t/s/u,12.506877649aggregate. Q1 per-slot attention/head and packed32MoE;384 slot/layer positions verified across4ranks.127model+127sampling replays,205359segments=127x1617;0metadata/hostreads/sync/refresh/capture/release inside timing;2150programcache entries before/after and4816persistentbindings stable. All72model hashes matchv4. No B32 before/after speedup claim. No hardware jobs remain; review and local checkpoint only.

Before/after accuracy comparison: completed stage6 `../full_model/acceptance_accuracy_resume_final_v1.json` has prefill97/100/100% and teacher forcing96/100/100% top1/top5/top100, each100 predictions. Final `final_validation_v2.json` is identical on these counts. Stage6 tracked teacher decode6.716107181t/s/u versus final8.790474572; separate from clean resident token-out timing.


## Final review and capped disposition

2026-11-14T15:29:49.699009+00:00 Fresh xhigh independent stage-review round1 writes `stage_review_round1.md`: `more-work-needed`, sole P1 is the mandatory FUNCTIONAL floor (6.2237719074% vs10%;103.064959568ms vs64.145280ms required). No additional correctness blocker or missing required device gate identified. All final device runs finished once. Bounded AutoFix already failed this same gate; no successful repair or physical-impossibility claim. No redundant review or unchanged gate rerun is requested. Stage acceptance remains false.

Review documentation notes were resolved: final prefill accounting/B32 completion and context-contract remediation status now agree with the completed evidence. Earlier timestamped work-log entries are historical; this final disposition supersedes their pending states. These are docs-only changes; model/native/shared/harness sources are unchanged.

## Carried to next stage

These handoff notes do not authorize advancing this unaccepted stage or starting vLLM. The seven untried optimization families and nonadditive ceilings remain under `Not tried (stop rule)`. B32 lacks a matched prior performance baseline and independent long-answer workload; per-slot Q1 attention/head remains a future scheduling candidate. Full48 prefill device attribution lacks a validated layer-class partition. Qualitative conclusions remain limited to the frozen comparative rubric and128-token cap; matched HF truncation is retained in the limitation registry. Experimental compaction remains disabled until native-only integrated trace/state failure is resolved. The failed mandatory floor is Required Work for an authorized continuation, not a carried nonblocking waiver.

## Local checkpoint policy

Commit only stage-owned `tt/`, `tests/`, context/limitation metadata and selected optimized-full-model evidence. Unrelated `doc/fused_decoder` files remain untouched. Source/test changes use normal repository hooks; the immutable evidence commit skips only mutating whitespace/EOF/Python-format hooks so measured snapshots, generated check JSON and raw perf tables retain their original bytes. All45 implementation/test files passed repository hooks before final measurements. Large raw JSON/log/Tracy/opsCSV and exact completion text remain at the documented workspace paths; `artifact_inventory.json` and compact summaries enumerate evidence. No push. Actual source/evidence SHAs are recorded below and in `checkpoint_commits.log`.

Local implementation checkpoint: repo `/home/ttuser/qwen38-flash/tt-metal`, branch `main`, SHA `9799a08439f6405e224ca30db7526d317fd6f86b`. Normal repository hooks pass in `source_commit_hooks_v2.log`. First staged commit hook applied clang-format to10new native files, despite earlier unstaged --files checks reporting pass; that first hook attempt did not commit (`source_commit_hooks.log`). The complete diff was inspected: whitespace/line wrapping and braces around existing single return/continue/assignment/function-call statements only; no declarations move scope and no operands/operators/order/math/layout/config change. `checkpoint_sources_v5.json` records all72model+3shared hashes and each measured-v4→checkpoint-v5 delta; non-whitespace/non-brace token text is identical. This is not a binary-equivalence proof. No hardware rerun after cap and no rewrite of measured provenance. Stage review remains the independent review of measured v4 behavior; style-only changes do not require another review round under the skill.

The evidence checkpoint records this implementation SHA and frozen artifacts. Its own SHA is written after commit to local `checkpoint_commits.log` (avoiding a self-referential commit hash in this file) and reported to the user. No push.


## Owner decision and resumed closure — 2026-11-16

Owner decision: accept the current decode speed (about 9.7 tok/s/user vs the 15.6 FUNCTIONAL floor) as a known limitation for this run. Record it in limitations with this decision and the evidence, finish the review and complete stage 7. No further tuning for this floor.

Authority: owner message delivered by the multigoal runner in this conversation; preserved verbatim in `owner_decision_2026_11_16.json`. Original goal copy: `/home/ttuser/qwen38-flash/multigoal_logs/07-07-optimized-full-model.prompt.txt` (not the source of this later amendment). Added the same scoped decision, unchanged metrics and evidence to `../eval_readiness/limitations.json`. This supersedes the Nov14 floor-blocked disposition and Required Work handoff only for this run. Numerical floor pass remains false; no hardware or tuning is authorized for this closure, and no datatype sweep/vLLM work begins.

Implementation checkpoint `9799a08439f6405e224ca30db7526d317fd6f86b`; evidence checkpoint `3e05e46cc7096fa27dd86019844d058e60603788`; branch `main`, repository `/home/ttuser/qwen38-flash/tt-metal`. Current source hashes will be checked against checkpoint v5; previous measurement hashes and round1 verdict remain unchanged. Fresh independent round2 reviews only prior findings, owner amendment, docs and the documented commit-format delta.


## Final closure under owner amendment — 2026-11-16

Independent fresh xhigh round-2 review: `stage_review_round2.md`, verdict **clean-pass**, no Required Work. The reviewer independently rederived 9.702618661 tokens/s/user and the failed 15.589611582 numerical floor, checked all 72 model plus 3 shared checkpoint hashes, and inspected all ten measured-to-checkpoint formatter diffs. Stage 7 is complete under the owner-amended acceptance contract; the original numerical floor remains false. This final disposition supersedes earlier pending/unaccepted/floor-Required-Work language without rewriting historical measurements or the round-1 report.

Closure-only checks: standard-library JSON/hash/evidence audit (`closure_audit_2026_11_16.json`), exact Git-blob recovery and formatter-diff verification (`checkpoint_clang_format_delta.patch`, `checkpoint_format_delta_verification.json`), `git diff --check`, and repository commit hooks. Seven retained device-gate artifacts pass their recorded scopes; no hardware rerun, tuning, datatype frontier or vLLM work occurred. The runner will perform its stage check after goal completion; no new runner result is claimed here.

### Carried to next stage — final disposition

The seven untried families remain in `Not tried (stop rule)` with nonadditive ceilings and constraints. B32 has no matched prior throughput baseline or independent long-answer workload; full48 prefill device attribution lacks a validated layer-class partition; qualitative evidence remains capped; rejected GR-down compaction stays disabled. These are disclosed hand-off notes. The current decode floor shortfall is an explicitly owner-accepted performance limitation for this run, not an instruction to tune further. Future stage scope requires its own authorization; this closure starts neither datatype sweep nor vLLM.

Local closure changes are documentation/evidence only. Implementation and evidence SHAs remain `9799a08439f6405e224ca30db7526d317fd6f86b` and `3e05e46cc7096fa27dd86019844d058e60603788`, on `main` in `/home/ttuser/qwen38-flash/tt-metal`. The closure checkpoint SHA will be recorded immediately after commit. No push.

Local closure checkpoint: `e6c1b7bac7dd1b040d49e38e6449d8f33e493871` on `main`, repository `/home/ttuser/qwen38-flash/tt-metal`. All applicable normal commit hooks passed (`owner_closure_commit_hooks_v2.log`); 15 stage-owned documentation/evidence files committed, no implementation changes and no unrelated files included. This receipt records the closure SHA; its own commit SHA is recorded in local `checkpoint_commits.log` after commit. No push.
