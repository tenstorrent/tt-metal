# Optimized vLLM work log

Stage 10 started from clean stage 9 checkpoint `5e2dca2e`. The installed 0.1.24 model-bringup and enabled 0.1.8 AutoDebug packages are used. No push is authorized. Device jobs are serialized on the 1x4 Blackhole mesh; initial fuser found no owners and bounded tt-smi listed all four chips.

## Contract and baseline

Preserve selected baseline_bfp8 policy, checkpoint thinking/sampling defaults, context262144, capacity32, valid nonaligned requests, canonical split sampling and resident async decode. No profiler. Prior stage owner decisions remain scoped as recorded in `../vllm_integration/owner_decisions_20261117.md`.

Baseline command:

```bash
QWEN_VLLM_AUDIT=0 TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 PYTEST_ADDOPTS='-p vllm' timeout 14400 bash models/autoports/qwen_qwen3_6_35b_a3b/tests/run_vllm_server.sh models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_baseline_server 32 2048 --stages serve
```

The launcher delegates to the shared run_vllm_server, uses context262144, meshP300x2, trace region805306368, FABRIC_1D, decode_only traces, async scheduling, page32, and all-device sampling. Benchmark evidence is pending. Historical primary128/128/1 TPOT53.798ms is a reference, not a new measurement.

## Work remaining

Same-harness baseline/final primary and CI burst; three scheduler budgets and eval timing/load; measured optimization candidates and stop-rule accounting; affected trace/async/stale-input checks; qualitative and sampling; final gate, independent stage review and local commits.

## Candidate preparation

Independent CPU topology audit: `topology_audit.md`. B4 is the first candidate: retain B2/B32, prepare B4 before all captures, expose existing plugin bucket mechanism only. B4 is estimated to fit768MiB; runtime capture must prove this. More buckets are not assumed to fit. No plugin changes proposed.

The exact bucket-state probe now accepts `--four-row-bucket`, compares2/4/32 against fixed32 using unchanged real layer0/3 arithmetic, and covers resident stale-input sentinels, two output leases, active/dormant state, greedy/seeded output, and2→4→32→4→2 transitions. Host comparator tests pass4 (`bucket_host_tests.log`); source precommit passes (`bucket_precommit.log`). No device test run yet.

Fresh primary baseline128/128/1 at concurrency1: TTFT3035.459ms, TPOT53.796971ms, ITL P50/P9953.773261/54.651661ms, output throughput12.971173tokens/s, TPOT-derived18.588407tokens/s/user. CI and prefill collection still running; raw and normalized output under `../../readiness_vllm/optimized_before/`.

Baseline collector completed PASS: `readiness_vllm/optimized_before/performance_execution.json`. Commands are expanded in that artifact. Canonical benchmark/prefill burst259device/0host; exact-token matchedB2HTTP381device/0host; no runtime errors. Source model/generator/plugin unchanged from stage9; only stage test/glue files have changed.

2048 budget probe command (same baseline server):

```bash
timeout 7200 python -m readiness_check.run_long_context --part chunk --server-url http://localhost:8000 --model-dir models/autoports/qwen_qwen3_6_35b_a3b --server-log models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_baseline_server/server.log --max-num-batched-tokens 2048
```

It retains default31decode users,4K/32K prompt lengths and context262144. After it exits, the queued cleanup verifies idle state and the named runner/log pair before SIGINT; B4 reduced probe begins only after all owned processes and chip holders are absent. All commands are bounded. No reset has been needed.

## Additional baseline and functional-floor investigation

Before shutdown, queue the same expanded mixed lifecycle workload against the original2/32 server (`mixed_before.execution.json` records exact command and idle boundaries). This is performance-only audit0/tracking0; final same-workload latency will be separate from tracked correctness. New test coverage includes all B4 rows, native64 sampling, five-way concurrency to require B32, exact requested completion lengths and actual bucket selection/returns.

The historical useful-weight floor is21.63129875ms/token, derived from1,107,522,496useful bytes/rank and512GB/s/rank at10%. Fresh serving53.796970693ms does not meet that inherited single-user floor; padding/inactive rows must not inflate useful work. NativeB2 agreement is not a floor waiver. AutoDebug launched an isolated inspection of current groupedEP cost and minimal exact alternatives. Initial sandbox preflight failed before model launch because unprivileged namespaces are disabled; existing unrestricted execution authorization covers the explicit AUTODEBUG_SKIP_CHILD_SANDBOX=1 retry. Logs are autodebug_launcher.log and autodebug_retry.log. Prior tracked root report preserved verbatim as AUTODEBUG_previous_stage.md before the new report is produced. No device/profiling permission given to the investigator.

## B4 exactness and AutoFix route candidate

ReducedB4 PASS21cases across greedy/native32/native64: full active logits, all-chip states, tokens/positions/seeds exact vs fixed32;8601stable program entries;5captures/5releases. All40 B4 standalone PASS11full-vocabulary comparisons plus three96token serial trajectories;5captures/5releases. These controls use1024context as a correctness fixture, not a server context change. Current source hash later changes for the opt-in route candidate, so regenerate source-matched all40 evidence before HTTP comparison.

AutoDebug completed; final report copied to `AUTODEBUG_FLOOR.md` and previous root tracked report restored. It verifies the21.63129875ms useful-work floor and identifies synthetic/inactive rows entering grouped MoE counts as the first exactness-preserving candidate. AutoFix owns the opt-in model parameter `suppress_inactive_moe_routes=False`, persistent activity threading and model-owned route-ID mask. Off-rank skip IDs are coherent across counts/dispatch/post-combine, with the native ACK barrier unchanged. No plugin/vLLM change.

First route discriminator `optimized_route_probe/result.json` FAIL before model cases: six directUINT16mask tests passed, but FP32 router topk returnsUINT32 indices. The helper guard correctly rejected actual input. Source confirms topk_device_operation.cpp's FP32→UINT32 rule. Correct the helper's tile width/offsets and tests to the actualUINT32contract; retain failed artifact/log, rerun to fresh path. This is not evidence of an active-logit difference or a speedup.


## Route and solo-attention validation

Route probe v2 failed the allocation tracker because the independent reference generator allocated persistent inputs after the candidate capture. The tracker was kept enabled. Both diagnostic owners now warm before either captures; v3 passed all36 cases with stable8602 program entries, five candidate and three reference captures/releases. The six direct mask checks cover all32-active identity, holes, high slots and permutations. Inactive IDs use rank-local off-rank expert IDs coherently for counts, dispatch and combine; the ACK barrier remains unchanged.

The combined candidate additionally retains a model_solo0 trace only for authoritative active_slots=(0,). Its physical B2 arithmetic retains projections/router/norms/residual and skips only the inactive row's attention work. Immutable device zeros replace that row's partial outputs. All other active-slot patterns use ordinary2/4/32 traces. Reduced combined probe passed36 cases, full active vocabulary and all-chip recurrent/convolution/KV state exact against an independent unchanged reference;8604 stable program entries, six candidate and three reference captures/releases. `optimized_solo_probe/result.json` records source hashes and cleanup.

The adapter now opts into both flags when capacity>=2. No real server has yet run this change. Full40 unchanged and candidate runs are serialized in `full_control.log` and `full_candidate.log`, with saved full-vocabulary tensors for exact external comparison and three96-token trajectories. These1024-context correctness fixtures do not alter the262144 serving contract. Next: matched native/adapter measurement at full serving cache geometry, then real HTTP primary/CI on the original2048 scheduler budget. Matching native performance alone cannot waive the21.63129875ms FUNCTIONAL floor.


Full40 control and combined candidate completed PASS at07:30UTC. All11 full-vocabulary comparisons match exactly; an external comparison also proves identical96-token sequences for all three129/130/131-token fixtures. `full_candidate_comparison.json` hashes both artifacts. Candidate retained six traces in the unchanged768MiB reservation and released/closed cleanly. fuser found no device holders before the next job.

Matched full-context native/adapter timing command (tracking/auditing off):

```bash
QWEN_VLLM_AUDIT=0 QWEN_VLLM_REDUCED=0 TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 2400 python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/run_matched_b2_overhead.py --server-log models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_baseline_server/server.log --repetitions 2 --prompt-json models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/serving_overhead/baseline.prompt.json --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_native_candidate/result.json
```

The probe parses the baseline server's actual8224-page cache specs, allocates through the current adapter at context262144/capacity32, and compares native serial, adapter serial, and queued async2 with identical128/128 tokens. It is comparison evidence, not the primary HTTP benchmark. `native_candidate.log` captures execution. A source-only third candidate, integer route-plan fusion, is documented in `route_plan_design.md`; no implementation or performance claim yet.


## First measured combined candidate

`optimized_native_candidate/result.json` PASS: two measured runs after warmup each of native serial38.305837ms, adapter serial38.406699ms, queued async2 38.241890ms per decode token (128in/128out/1user,capacity32,context262144). All nine sequences, including warmups, are identical; six traces retained and cleanly released, mesh closed. `native_candidate_metrics.json` records source artifact hash and scope. This is about29% latency reduction from the repaired nativeB2 baseline, but still above21.63130ms FUNCTIONAL floor.

Real candidate server launched through the same model wrapper,32/2048, exact baseline configuration and environment; log `readiness_vllm/optimized_candidate_server/server.log`. Collector command:

```bash
timeout 2400 python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/run_final64_perf.py --server-log models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_candidate_server/server.log --output-dir models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_candidate_perf --execute
```

Primary128/128/1 completed: TTFT3036.463239ms, TPOT38.219612ms, ITL P50/P9938.204662/39.143082ms, output throughput16.221550tokens/s, decode26.164578tokens/s/user. The candidate reproduces native speed; CI and prefill burst remain in progress. This is intermediate evidence, not final optimized stage acceptance.

AutoFix continues with integer route-plan fusion (`route_plan_candidate.patch`, not yet applied) and a source-only grouped-kernel geometry audit. Active model source is frozen during HTTP collection. Host harness changes add opt-in flags only; precommit passes. No plugin/vLLM change, profiler, reset, or context reduction.


Intermediate collector completed: canonical primary/CI/prefill all passed,259device/0host decode steps; matched128/128 HTTP3requests381device/0host, exact baseline token sequences, no runtime errors, idle before/after. `intermediate_comparison.json` asserts identical normalized benchmark configs. CI100/100/32 output throughput24.26300tokens/s versus23.76859 before. Server shutdown PASS with six captures/releases and no remaining owned processes/device holders.

Applied opt-in route-plan and grouped-grid candidates after cleanup. Both default off/8x8 in model and remain disabled in adapter. Route helper PASS13eager cases plus5changed/unchanged input replays,1capture/1release, no cleanup errors, tracker1/skip0. The first reduced-model fusion attempt failed before any model cases: its independent fixed32 reference was incorrectly constructed with the solo specialization, which requiresB2. Removed that test-only reference flag; the fixed32 reference still uses the unfused masked route graph. Failed artifact preserved in `optimized_route_plan_model/result.json`. The grouped-grid microgate runs next, followed by the integrated reduced-model discriminator. Source precommit and19host tests pass.


Grouped4x8 microgate FAIL during native argument validation: columns>=rows>=2 is required (`moe_fused_swiglu_device_operation.cpp:218`);4x8 is unsupported. The source audit missed that guard. The failed artifact is retained, guard unchanged, candidate rejected, and its opt-in production plumbing will be removed after the currently running8x8 route-fusion discriminator completes. No performance or exactness result is claimed for4x8. Follow-up source inspection is limited to legal8x8 weight placement preserving arithmetic.


Route-plan reduced v2 PASS36cases with stable8605programentries. Removed unsupported4x8 production plumbing after this run; the archived patch and microgate preserve reproduction of the native validation rejection. The adapter now selects fused route planning along with route suppression and solo0, atcapacity>=2. Full40 source-matched control is running with `--four-row-bucket --route-mask --solo-attention --route-plan --serial-output-tokens96 --compare-full-logits readiness_vllm/optimized_full_control/logits.pt`; output `optimized_full_route_plan/standalone.json`, log `full_route_plan.log`. This new default is not yet serving-performance accepted.


Full40 fused route plan PASS11 full-vocabulary cases against the saved unchanged control; all three96-token serial trajectories externally identical. `full_route_plan_comparison.json` records hashes and scope. Six traces released cleanly. Matched full-context native/adapter timing launched to `optimized_native_route_plan/result.json` using the unchanged128/128 workload, baseline server config and8224 pages; source frozen for this run.


Matched route-plan measurement PASS: native36.061330ms, adapter serial36.212610ms and queued async2 36.001421ms/token (128in/128out/1user,capacity32,context262144,8224pages). Async latency improves5.8587% against the prior38.241890ms matched candidate; still above21.631299ms floor. All modes preserve identical tokens; six captures/releases, mesh closed, fuser reports no holders. `native_route_plan_metrics.json` records artifact hash and scope. HTTP acceptance for final accumulated candidate remains pending.

Applied `grouped_nd_weight_candidate.patch` as an opt-in model-owned startup placement candidate, defaultFalse; adapter remains interleaved. Gate/up use ND[32,64] shards; down and legal8x8 arithmetic unchanged. Python compilation passes. Fresh eager real-weight microgate is running under tracking1/skip0, output `optimized_nd_weight_micro/result.json`, log `nd_weight_micro.log`; it compares complete decoded weight values across all four chips and16 expert-output cases before any model/trace/performance adoption. No plugin/vLLM or native shared-op edit.


ND microgate PASS16cases and complete decoded weight values across128 GU matrices/chip×4chips. ND reduced integrated PASS36cases,8638stable programentries, six candidate and three reference traces retained/released. Precommit passes for the model/decoder placement changes. Adapter now selects ND forcapacity>=2 for the next candidate measurement; full40 exactness runs with `--ep-weight-nd-sharding` to `optimized_full_nd_weight/standalone.json`. Active TT source is frozen for this run.

ND host adapter/sampler checks PASS19 in5.08s (`nd_weight_host_tests.log`). Python shutdown prints nanobind binding-reference leak diagnostics; this host-only test process exits0 and opened no hardware. This is not evidence of a live serving/device resource leak; final runtime shutdown remains separately checked.


ND full40 PASS11 full-vocabulary comparisons and all three96-token sequences externally equal to unchanged interleaved control; six traces released. `full_nd_weight_comparison.json` preserves hashes. The next matched native/serial/async2 measurement uses exactly the same128/128 workload, original baseline served geometry and precision, with selected ND default recorded explicitly in `effective_decode_configuration`. Output `optimized_native_nd_weight/result.json`; log `native_nd_weight.log`. No HTTP latency claim yet for ND.


ND final native measurement PASS but no material gain: native36.000417ms, serial36.158956ms, async35.950710ms versus interleaved36.001421ms async (0.140857% improvement). Rejected startup-layout complexity; reversed ND production patch and removed adapter selection, preserving archived patch, flags and every correctness/timing result for reproduction. No benchmark was claimed to prove a bank-bandwidth improvement.

Applied the minimal dead-norm patch: in retained solo0 linear attention, skip inactive row1 before computing its unused singleton norm. No active arithmetic or physical B2 carrier changes. Fresh allocation positive control PASS under tracking1/skip0: deliberate post-capture hazard is rejected, then clean replay passes (`trace_positive_control.log`).

Applied opt-in conditional recurrent-state commit patch, defaultFalse. Existing state-slab reader and compute are unchanged; writer reads persistent activity and predicates only state stores, always draining state/output CBs. Unsupported metadata retains original staging/copy_if fallback. Rounded1 real/synthetic all-chip microgate is running at `optimized_state_commit_micro/rounded1.json`, log `state_commit_micro_rounded1.log`. No speed claim yet; production adapter remains flagFalse pending validation.


Conditional-state micro v1 failed before cases/captures at the KernelDescriptor Python constructor: defines requires a sequence of string pairs, not a dict. Corrected the argument to a list of pairs; retained failure artifact. V2 rounded1 and rounded0 each PASS96cases,2captures/2releases, no cleanup errors. Exact all-chip output/state bits and inactive-state preservation hold across changed and unchanged input/activity replays. Integrated reduced candidate (state commit plus dead norm; interleaved weights) now runs to `optimized_state_commit_model/result.json` under tracking1/skip0. The next row-extraction/masking patch remains archived until this run finishes.


Conditional-state+dead-norm reduced integration PASS36cases with8606stable programentries; source tracking enabled, all traces released. Applied default-off hidden-row extraction/mask helper after the job ended. Row micro v1 preserved exact old-vs-new warm0 output, then failed the CPU seed-bit oracle on4words beginning at signed zero. The seed-to-device/readback transport boundary needs explicit observation; no trace was captured. This failure is retained, not relabeled a kernel correctness pass. The repair will derive expected row bits from observed source snapshots while retaining complete original-vs-fused physical-padding equality and recording seed differences. Source precommit for all four affected production files passes.


Row micro v2 retained failure: exact replay upload preserves BF16 edge bits, while the old split chain canonicalizes them. Exhaustive transport diagnostic observed all65,536 patterns on each of4chips: exponent0 maps to positive zero; exponent255 strips mantissa preserving sign; all other words unchanged. The helper now applies this exact integer mapping. Exhaustive rerun `optimized_hidden_row_transport_v2/result.json` has zero old-versus-candidate physical-bit mismatches across all/even/none/return-all masks and unchanged input on every chip; `hidden_row_exhaustive_comparison.json` records the gate. The42-case trace oracle applies that independently measured rule to actual uploaded snapshots; strict old/new equality, source preservation and padding checks remain. Fresh v3 micro is running with tracking1/skip0. No model or performance acceptance claimed yet.


Row micro v3 failed before capture on a generic-op cache collision: B2 andB4 shared compile-time identity but needed10 versus14 runtime words. Added immutable bucket row count to the helper compile arguments/cache key. Fresh v4 PASS42cases,12captures/releases and84nonblockingreplays with stable program entries, exact all-chip padding, changed/unchanged source and activity, independent retained outputs and clean mesh close. Integrated state+row candidate now runs `run_vllm_bucket_probe --four-row-bucket --route-mask --solo-attention --route-plan --conditional-state-commit --hidden-row-mask` at `optimized_state_row_model/result.json`.


Combined state-commit+row-mask+dead-norm reduced integration PASS36cases with8609stable entries; candidate and reference traces released and device close logged. Adapter selects state/row fusions forcapacity>=2. Source precommit passed. Full40 standalone validation launched at `optimized_full_state_row/standalone.json`, with all kept flags, saved unchanged full-vocabulary control and three96-token serial trajectories. No performance acceptance until this gate and matched timing complete.


Full40 state/row/dead-norm candidate PASS11 full-vocabulary comparisons and all three96-token sequences exactly equal to the unchanged control; six traces released, device close logged. `full_state_row_comparison.json` records the external comparison. Matched native timing now runs at `optimized_native_state_row/result.json` with the original128/128 fixture, capacity32,context262144,8224served pages, selected precision, two repetitions plus warmup per mode; no profiler or host-logit performance path.


Matched state/row/dead-norm timing PASS: {"adapter_async2": {"mean_ms_per_decode": 33.5996254448899, "decode_tokens_per_second": 29.762236535648285, "mean_prefill_ms": 3469.5704039986595}, "adapter_serial": {"mean_ms_per_decode": 33.76740305901067, "decode_tokens_per_second": 29.61435909810526, "mean_prefill_ms": 3452.684573501756}, "native_serial": {"mean_ms_per_decode": 33.66448965352615, "decode_tokens_per_second": 29.70489112688082, "mean_prefill_ms": 3443.0540355097037}}. Combined async gain 6.671389903222269%; retained. Fullcontext262144,128/128/1,8224servedpages; all9token sequences identical;6captures/releases,meshclosed,noholders. FUNCTIONAL21.631299ms floor still fails. Next AutoFix discriminator is the archived default-off empty-expert mailbox schedule; no plugin/vLLM edit.


Applied `empty_expert_schedule_candidate.patch` and candidate-only `empty_expert_schedule_harness.patch` after matched state/row completed and fuser showed no holders. Native build command: `timeout1800 /usr/local/bin/cmake --build build_Release --target ttnn --parallel4`, log `empty_expert_build.log`. Ninja scheduled1210 dependency targets, so this is not a small incremental build. No device Python process is running during build/install. Adapter compaction remainsFalse. Full-logit source identity and matched timing provenance now include all native grouped-MoE sources, to bind future protocol evidence to exact source. No plugin/vLLM changes.


Native build PASS1210targets. Both `cmake --install build_Release --component ttnn-runtime` and `--component tt_pybinds` passed. Precommit of all touched tracked Python/native sources passes without edits. Fresh process microgate launched with `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 TT_METAL_WATCHER=10 timeout900 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_empty_expert_schedule_probe --output .../optimized_empty_expert_micro/result.json`; BFP8/default256x256 diagnostic geometry. Adapter compaction remainsFalse; no performance claim.


Empty-expert protocol micro PASS18eager/traced cases under Watcher10/tracker1/skip0, BFP8,256x256. Separate33..128 accepted-token-band micro also PASS18. Each captured/released2traces, no cleanup errors; all-empty, late63, cross31/32, mixed/ragged, dense, remapped/unchanged and active→empty→active cases preserve full output bits/sentinels on all4chips. Qwen2048x512 geometry micro is running next, still synthetic weights; no serving latency claim.


Qwen2048x512 native protocol micro PASS18cases under Watcher10/tracker1/skip0; three native configurations total54cases,6captures/releases, no cleanup errors. Integrated reduced compact schedule PASS36cases with8610stable programentries, including greedy/k8/k64, high-slot and bucket transitions. Adapter now selects compaction atcapacity>=2 for the next full40 gate; all lower-capacity/default callers remainFalse. Full40 source-matched validation runs at `optimized_full_empty_expert/standalone.json` against unchanged saved logits and serial96control. Native+Python sources are recorded. Performance acceptance remains pending.


Full40 compact expert schedule PASS11 full-vocabulary cases and all three96-token trajectories exactly equal to unchanged control;6captures/releases and clean device close. `full_empty_expert_comparison.json` records native and Python source identity. Matched128/128/1 native/adapter timing with full262144 context and8224served pages is running at `optimized_native_empty_expert/result.json`; no sampler, precision, scheduler or context change.


Compact expert schedule matched timing PASS but rejected: async33.365455ms versus33.599625ms, only0.696944% gain for128/128/1 at fullcontext262144. Native33.415628ms and serial33.516795ms; all9token sequences identical,6captures/releases,meshclosed. Reversed native/model/adapter and candidate harness patches; original native rebuild PASS5targets and both ttnn-runtime/tt_pybinds installs PASS. No native shared-op diff remains. Provenance coverage of native grouped-MoE sources remains in test tools. Next source-supported AutoFix candidates are packed per-token postreduce weights and native-Q/K state-slab reads; independent agents prepare disjoint default-off source and microprobes, no hardware jobs.


Prepared three default-off candidates: packed per-token postreduce scores, native Q/K state-slab reader, and native singlechunk valid_token_rows specialization. Q/K rounded1 persistent micro launched under tracker1/skip0 at optimized_native_qk_micro/rounded1.json; helper source frozen. Native postreduce source validated/formatted but not built while device job is running. Model reader opt-in requires existing conditional state-commit path; fallback and prefill unchanged. No adapter default changed yet.


Packed postreduce default0 micro PASS36cases,2captures/releases, Watcher10 and tracker1/skip0; output optimized_packed_postreduce_micro/result.json. Native valid_token_rows build PASS5targets and both runtime/pybind installs PASS. Native-Q/K microv1 rounded1persistent PASS112; strengthened v2 adds arbitrary FP32 edge transport and device-poisoned padding. V2 rounded0temporary PASS112 with2captures/releases, no cleanup errors; rounded1persistent is running. Model defaults remainoff for all three new candidates; adapter still uses preceding kept state/row path.


Native Q/K strengthened v2 rounded1persistent also PASS112,2captures/releases. Postreduce prefix Watcher10/tracker1 probes PASS for1/2/4/31 (72cases each) and32identity (36cases), total324cases/10captures+releases, no cleanup errors. Every prefix rejects four invalidAPIcases; shortened prefixes also ignore candidate-only stale nonfinite input/score tails and local IDs while preserving validprefix and explicitpositivezero outputtail. Combined reduced layers0/3 discriminator launched with all three options plus kept fusions at optimized_postreduce_qk_model/result.json; no adapter default changed yet. Native source frozen, no profiler.


Combined postreduce/QK reduced discriminator PASS36cases with8612stable programs, native32/native64 sampling and B2/B4/B32/solo0 transitions. Independent source review found no defect in CB17 tail-zero-before-c0 ordering, prefix-consistent CB counts, reflected cache attributes or len(route_slots)/offrank-padding invariant. Adapter now selects all three options forcapacity>=2 for the full40 exactness and subsequent matched measurement. Full40 output optimized_full_postreduce_qk/standalone.json compares unchanged fullvocabulary control and produces serial96trajectories. No new speed claim yet.


Full40 postreduce/QK PASS11full-vocabulary comparisons and externally exact three96-token trajectories versus unchanged control;6tracesreleased,meshclosed,noholders. full_postreduce_qk_comparison.json records current native/Python source identity. Host adapter/sampler checks PASS19 (binding-reference shutdown diagnostics recur in host-only process,exit0). Matched fullcontext262144/8224pages128in128out1user native/serial/async2 now runs at optimized_native_postreduce_qk/result.json with2measured+1warmup per mode, original baseline config. Source frozen, no profiler.


Matched postreduce/QK timing PASS:128/128/1,context262144,8224pages; native32.355652ms,serial32.479970ms,async32.296497ms. Gain3.878400% versus kept33.599625ms; retain combined exact dataflow change. All9token trajectories match,6captures/releases,meshclosed,noholders. Floor21.631299ms stillfails. Applied minimal native-Q/K zero-fill followup (model-owned Python kernel source, no native rebuild): replace8192volatile clears/worker with two NoC localzero requests plus completionbarrier before unchanged scatter. Rounded1persistent strengthened micro now running; production source frozen.


Q/K NoC zero-fill strengthened micro PASS112cases in each rounded1persistent and rounded0temporary mode,2captures/releases each, no cleanup errors. Only helper description metadata clarified after those micro runs; arithmetic/dataflow unchanged. Full40 source-matched control now runs at optimized_full_qk_zero_fill/standalone.json, identicalflags/logitscontrol/serial96. No native shared rebuild needed for model-owned generic kernel source.


Q/K zero-fill full40 PASS11full-vocabulary comparisons and externally exact raw129/130/131 serial96 trajectories,6tracesreleased and cleanmeshclose. full_qk_zero_fill_comparison.json records source. Final matched native/serial/async timing launched at optimized_native_qk_zero_fill/result.json with unchangedbaseline128/128/1fullcontextconfig; source frozen.


Optional-search Stop Rule reached at10:24:30.635446UTC (4h since stage start). No new optional candidates will start. The already-running zero-fill matched timing completes, then selectedsource realHTTPbefore/after measurement and cleanup. Latesttwo keptgains6.67139% and3.87840% do not trigger the alternative<3% rule; rejectedND/compact options do not count. Requiredfloor and servinggates remainunwaived. Checklist now carries untriedoptions as stop-rulehandoffnotes.


Final in-flight AutoFix zero-fill timing PASS: native31.506430ms,serial31.624722ms,async31.444986ms at128/128/1context262144/8224servedpages. Gain2.636544% versus32.296497ms; retain exact simpler zero-fill. All9token trajectories equal,6captures/releases,meshclosed,noholders. Floor21.631299ms remainsunmet. Real selectedsource server launched through unchanged sharedwrapper at optimized_selected_server,capacity32/budget2048, audit0/tracker0, baselineconfig; primary/CI/prefill collector follows readiness. This is final comparison evidence for incomplete floor repair, not stage acceptance.


AutoFix failure assessment: thirteen discriminating hypotheses improved matched128/128/1 async to31.444986ms but failed21.631299ms floor. Independent grouped_floor_audit agrees no supported remaining minimal repair accounts for9.813687ms gap; larger untried redesigns are not exhaustion proof. User-authorized AutoFix-failure stop condition applies; stage remains incomplete. Finish already-running selected HTTP collector and mixed-shape latency, then preserve WIP checkpoint and clean owned server. Final acceptance gates will be listed explicitly, never marked pass.


Selected real HTTP collector completed: canonical65requests PASS and matched3requests PASS;259device/0host and381device/0host sampling steps. Wrapper status is COLLECTED_PASS_PENDING_OWNER_REVIEW (not literal PASS); initial local extraction assertion expected the wrong wrapper spelling, then explicit job/artifact checks passed without rerunning or altering evidence. Primary128/128/1 TTFT3049.104202ms,TPOT31.425571ms,ITLp50/p99 31.418572/32.279581ms,throughput18.180097tokens/s,decode31.821220tokens/s/user. Baseline53.796971ms TPOT gives41.584869% reduction. CI100/100/32aggregate24.664962tokens/s; all32pass. Serial128/32/32prefillmedianTTFT51551.286127ms,after/before0.981098. Final comparison asserts same workload configs and initialized precision/adapter/layers/context/capacity. Untracked mixed18request lifecycle now runs with exactbaseline flags (4101nonalignedchunked prompt,topk40,96output).


Final untracked mixed lifecycle PASS18/18 HTTP200,96tokens each, every token array equals baseline including sampledtopk40.668device/0host sampling steps, no runtime errors. Prompts129..133 and4101; solo-first ITLmedian31.352537ms; chunkedthree69.353644ms with max51395.551139ms; three68.762503ms; five402.256901ms; four69.824451ms; sampled31.468314ms; solo-return31.375627ms. Rawchecks/metrics in mixed_after_comparison.json/mixed_after_metrics.json. The51.4s chunk interruption remains an eval-visible latency limitation, not a bound pass. Tracking/audit disabled; final tracked lifecycle/page-growth gate still incomplete. Ownedserver cleanup launched via stop_idle_server.py,runner3669879 only,afteridle.


Selectedserver cleanup PASS: no remaining ownedPIDs and fuserexit1/noholders; shutdown_record.json retains allshutdown events. Baseline andselectedactualserver_command arrays compare exactlyequal, recorded in final_comparison.json. Allproduction performance sources remainfrozen since full40/matched validation. Proceedingto localWIPcheckpoint, explicitlynotstageacceptance; independentstage-review notrun becausehardfloor andotheracceptancegatesremainincomplete.


Final checkpoint checks: pre-commit PASS209files after adding finalnewlines to9canonicalrawJSONfiles. precommit_normalization.json proves eachchange is exactlyone newline, parsedJSONunchanged; all measuredPython/native sourcehashes unchanged. gitdiff--checkPASS excluding archived.patch files whose single-space contextlines are required unifieddiffsyntax (alsoexcludedbyrepositoryprecommit). No sourceformattingorbehaviorchange occurred. Selectedshutdown retained6traces untilteardown and releasedall6; no livevLLM/readiness/API/EngineCoreprocesses ordeviceholders remain. Plugincheckoutclean at5d1785b76d17a8989c9495f6939abea6cef62f09; no vLLMpackageedit. PreparinglocalWIPcommit on main; neverpush.


Local checkpoint: tt-metal branch `main`, commit `08384c1007d1a1065955221284f6d2c197d85260` (`WIP qwen3.6 serving dataflow optimizations; functional floor unresolved`). Commit hooksPASS; no push. Plugin remainsclean at `5d1785b76d17a8989c9495f6939abea6cef62f09`; no stage-ownedplugin/vLLMcommit. A following documentation-only commit records this SHA/status. Finished `2026-11-17T10:46:32.556530+00:00` with stage incomplete after explicitAutoFixfailure, allownedservingprocessesstopped anddevicesreleased.


Continuation after WIPcheckpoint8fbe38e7: prior turn classifiedprogress (measured selectedserving/mixedtraffic,committedsource/evidence,cleaneddevices). Revalidatedcleanworktree/noholders. The goal remainsactive and independent acceptancework is available despite recorded AutoFixfloorfailure. No optional optimizationrestarted orfloorwaived. Launched trackedasyncserver at optimized_acceptance_async_2048, unchanged32/2048/262144config, QWEN_VLLM_AUDIT=1,TT_METAL_TRACE_ALLOC_TRACKING=1,SKIP_PROGRAM_CACHE=0. Next gates lifecycle4101/topk40/96, freshHTTPlogits against optimized_full_qk_zero_fill/standalone.json, and schedulerpagegrowth.


Tracked selectedsource lifecyclePASS23checks:18HTTP200requests/96tokens,nonaligned4101chunkcontinuationsinterleavedwithdecode,allB2/B4/B32/solo0retainedgraphs andnative32/native64samplers replayed,stable6traceIDs,no recapture/release,2overlappedoutputs,allocatorgrowth,page-onlyupdatespreservetoken/positioninputs. Artifact optimized_acceptance_async_2048/lifecycle_summary.json. Allocationtracking1/program-cache-skip0 confirmedinAPIparentlaunch; renamedEngineCore/proc environmentisnotreliable. Found/fixedmodel-ownedperformancecollectorhidden-environmentfalse-negative; live trackednegativecontrol nowrejectsproperly. No productionadapter/model/pluginedit. FreshHTTPlogitdiagnosticagainstsource-matching73filefull40baseline nowrunning; diagnosticlogprobsallowexplicit hostlogits, notperformanceevidence.


FreshHTTPlogitdiagnosticPASS against optimized_full_qk_zero_fill/standalone.json withall73production/sourcehashesmatching. Rawrequests in optimized_acceptance_async_2048/http_logits.json. Scheduler-growth6requeststwoadmissionorders (129/160,130/112,131/80) nowrunningontrackedasyncserver. Added summarize_growth.py tovalidate matching submissionIDs/page-only/unchangedupdates/async readbackdeltas; historicalStage9rawlog cannotbeusedforthisnewanalyzer because it lacks submission_id (read-onlycontrolKeyError, nohardwareeffect). Currentadapter emits IDs; newcurrentevidence required. Added --served-geometry toqualitativecontrolharness forcurrentadapterconstruction/8224servedpages/full262144context, no productioncodechange.


Acceptance continuation: async growth PASS13 checks over318 paired submissions, logical pages5..8,18 page-only updates and294 unchanged updates. Every decode submission has readback delta0 before the async boundary; max pending outputs2. The plugin permits the immutable device-backed DecodeOutput lease used here; async_contract_acceptance.md records the exact contract without claiming a raw Tensor return. Six shared HF greedy and sampled controls were reviewed in full; this is not a TT qualitative verdict.

Full canonical sampling (including --tt-chunked-prefill-budget=2048) is running against the unchanged selected source. Both solo recall tests pass, but test_prefills_sharing_a_step_each_recall_their_own_needle failed. Final traceback and outputs are pending suite completion; no cause or waiver inferred. AutoDebug bundled launcher sandbox preflight failed before starting a model because bwrap cannot create a namespace. Retried with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under existing workspace permissions, read-only source investigation, no target execution or live-server access. Logs autodebug_chunked_launcher.log and autodebug_chunked_launcher_retry.log; root AUTODEBUG.md backed up as AUTODEBUG_before_chunked_recall.md.

Performance collector provenance correction: renamed EngineCore /proc environment can be empty or partially reused. The collector now records the verified API parent environment for this renamed multiprocessing child, including TT_METAL_DEVICE_PROFILER. Negative control v2 correctly rejects the live tracked/audited server. Earlier benchmark metrics are unchanged; their explicit launch commands set audit0/tracker0, but older worker-environment artifact fields alone are weaker evidence. A fresh untracked positive control remains pending.


AutoDebug completed and was preserved as AUTODEBUG_CHUNKED_RECALL.md; prior root AUTODEBUG.md restored. Parent verified historical JUnit skipped all three recall tests, the selected attention planner handles ragged offsets via singleton/aligned blocks, and full-capacity slot remapping snapshots sources before destination writes. No supported state/offset bug found. Exact suite failure still pending.

Canonical raw HF control (same pinned checkpoint, BF16, raw prompts, greedy24, exact12 fixtures) completed:1 recall/11 misses. All12 read; failures stop in reasoning, passing case2-1 returns the expected phrase. This is raw stress evidence, not a chat-quality waiver or ten-seed model-card comparison. Original harness preserved as recall_controls_raw_v1.py; hashes/review in recall_raw_hf_review.json. CPU command: OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 timeout7200 nice -n10 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_chunked_recall_controls --backend hf --threads8 --output readiness_vllm/optimized_acceptance_recall/hf_serial.json (paths model-relative here).

Added separately labeled chat and model-card controls to the model-owned probe; canonical raw defaults unchanged. Existing plugin --tt-recall-reasoning-effort=low selects chat for exactly the recall tests; no shared plugin change needed. HF chat/default-thinking greedy1024 diagnostic is running across all12 fixtures to observe answer length. Native quality harness can append the same12 chat greedy1024 controls after the unchanged shared256 suite at actual served cache/context, to measure a native reasoning allowance before choosing the plugin test budget. No TT production source change.


Recall request-mode discriminator: first completed HF chat/default-thinking case0-0 recalled correctly after839 tokens. The CPU-only diagnostic was deliberately stopped and marked INTERRUPTED_DIAGNOSTIC; remaining cases are not passed. SIGINT did not end its ThreadPool worker, so SIGTERM targeted only the verified owned HF PID3875000; exit143. Full details in recall_hf_chat_interruption.json; no TT process was touched. Native and HF follow-up now use explicit nonthinking chat with the original24 cap, preserving the canonical short-answer assertion instead of increasing its allowance. The unchanged default-thinking shared qualitative suite remains required.

Nonthinking HF twelve-case greedy24 control is running; first cases pass. Model-owned test fixture has not changed while original full sampling is active. AutoFix isolated hypothesis review requested from serving_audit: assess whether existing chat recall mode plus explicit request override is justified and identify controls against masking a serving defect. This is not final stage-review. Tool precommit PASS after Black formatting; production Python/native source and selected precision remain unchanged.


HF explicit nonthinking chat controls PASS14: all12 concurrent-corpus answers are exactly their own phrase and EOS in10 tokens; canonical solo short and long also pass. `recall_chat_off_hf_all14_review.json` records reading all responses. The870454-byte twelve-case JSON was preserved losslessly as97309-byte `.json.gz` to satisfy the repository500KB limit, with exact-decompression and both hashes in recall_evidence_compression.json. `recall_fourteen_fixtures.json` retains all14 fixture definitions for native controls. The source auditor agrees this is a valid configured gate but not yet an explanation of original TT failure; detailed required discriminators in recall_request_mode_audit.md and recall_validation_plan.md. Original full suite remains running; its fixture is unchanged.


Original full profile completed in3838.02s:70passed,71skipped,1failed. Skips are70unsupported-speculation tests plus1missing-spec-artifact test. Exact concurrent failure:10/12 raw replies stop during reasoning at24 tokens; every failed response is byte-identical to its matching HF control. No wrong-needle or request-error evidence. `recall_original_failure_attribution.json` preserves all10 pairs and counts. Original fixture/command/hashes preserved before mutation.

Minimal model-owned test-configuration repair: checkpoint_default_test_requests.py adds the three exact recall test names to its explicit nonthinking chat override, for six named tests total. Existing plugin --tt-recall-reasoning-effort=low selects chat; --tt-recall-reasoning-token-budget=0 retains24cap. Prompts, seeds, concurrency, rounds and assertions unchanged; no production/plugin edit. PrecommitPASS. Focused all3canonical recall tests now run under run_probe name recall_chat_off, with request/response JSONL and JUnit, tracking/audit enabled at unchanged32/2048/context262144. Full final profile rerun remains required.


Acceptance continuation keeps production sources frozen. Updated optimization checklist now links completed final HTTP/native comparison, tracked lifecycle and async-growth evidence instead of stale pending labels. Added CPU-only summarize_recall.py to join actual request prompts and every logged chunk SHA256 to all14 preserved fixtures, verify contiguous partitions, ragged resumptions with intervening decode, six stable trace IDs and zero recapture/release. Focused recall remains active. Queued HTTP tail record: optimized_acceptance_async_2048/acceptance_tail.execution.json; queued cleanup/native record: optimized_acceptance_native/execution.json. Native only starts after HTTP completion, passing recall summary/qualitative collection and stop_idle_server PASS. Full configured sampling profile, opposite scheduling and eval gates are not yet passed.


Configured canonical recall finished:3tests PASS,0skips,1204.508s. All14 outputs read: exact own passphrase and stop,9..10tokens at unchanged24cap. Served prompt lengths:solo3117,long9296,concurrent3089/3115/3123/3149 across3rounds. Runtime audit58device/0host decode steps,noerrorlines. Shared default-thinking qualitative is now active; exact partition/trace summary follows tail. Pre-run read-only auditor found native harness could report collection PASS despite recall false; fixed before native starts: require14unique fixtures andall14recall/EOS,fail process otherwise. Protected mesh cleanup/save evenifgenerator.close fails. Native harness precommitPASS. Production source unchanged.


Shared default-thinking served suite completed12/12 at256cap; all12outputs read, coherent task-directed reasoning, no sustained collapse. Current HF prompt IDs and requestsettings verified in served_shared_quality_review.json; bounded smoke only, native controlpending. Degeneracycheckerexit0/no findings; checkpointdefault4modePASS; runtime3060device/0host decode steps.

Recall analysis initially failed due two CPU-tool assumptions, not device evidence: tokenizer.apply_chat_template(tokenize=True) returns BatchEncoding in this installed version; render then encode now matches native/HF control. Identical suffix chunks acrossrounds require association with an already-started contiguous prompt partition. Both errors/logs preserved. Corrected summaryPASS11checks: all14exactanswers/templatecounts/chunkdigests/contiguouspartitions, raggedstarts640/896/2688/2944,9interleavedcontinuations,6stabletraces/no recapture/release,58device/0host decodes.

Tracked asyncserver cleanupPASS: all6traces retained untilteardown/released, noownedPIDs/deviceholders. Native queue conservatively stopped before hardware because firstsummaryfailed; finalsamplingqueue also exited beforelaunch. Bothrecords preserved. Aftercorrectedsummary andprecommitPASS, directnative retry starts from execution_retry.json at optimized_acceptance_native, all40servedcache8224pages/context262144/capacity32, tracker1skip0. No production orsharedpluginchange.


Native shared256 suite completed12outputs; all12read. Sixgreedytexts matchservedexactly. All12served andall12native texts independently equaltheirrespectiveStage9 controls (shared_native_comparison.json). Sampled native/served texts differ justasbefore; matchingrequestparameters alone do not assert identicalRNG progression acrossAPIs. Savedlogcounteranalysis shared_sampler_widths.json showsall12servedrequests use256native32samplerreplays; proposed draw-width explanation was refuted, not claimed. BoundedqualityreviewPASS; no fullansweraccuracy orbroaderqualitywaiver. Native14recallcontrols andcleanup remainrunning.

### 2026-11-17 13:08 UTC — native controls complete, full configured sampling running

The current native control finished all 40 layers with actual served geometry (context262144, capacity32, 8224pages in ten cache pools). All fourteen recall outputs contain exactly their own phrase and EOS within9–10tokens at the original24cap. Six native traces were released and `fuser` reported no device holders. The six current greedy shared outputs equal served outputs exactly; all twelve native and all twelve served outputs separately equal their Stage9 controls. Six sampled cross-path differences therefore persist unchanged; equal recipe is not asserted to establish equal RNG progression. The proposed native32/native64 draw-width explanation was refuted by the actual twelve request records, each using native32. Shared review remains a bounded256-token reasoning smoke. Immutable native raw JSON is losslessly compressed with both hashes and verification in `optimized_acceptance_native/evidence_compression.json`.

Fresh tracked async server runner4107414 became ready at13:02:40UTC. Complete142-test sampling profile is running through `run_probe.py`, client4126115/pytest4126417, with the six documented request-mode overrides and no test selection. Original failure records are retained. Native and recall analyzer corrections affected evidence tooling only; production source remains unchanged.

### Eval batch32 necessary floor, CPU-only preparation

`eval_batch32_necessary_floor.json` derives a one-sided necessary floor, not actual route coverage. Pinned tokenizer renders all32installed eval prompts with thinking on; maximum74tokens. At upper position1074, extend the inherited `tests/multichip_functional.py::byte_model`: nonexpert weights once, expert weight term times32 (at most64unique resident experts/rank vsB1logical2/rank), state/cache traffic times32, plus shared head weights,32embedding rows and one terminal norm weight. The upper useful-byte total is10,745,186,240bytes/rank/batchstep. At512GB/s/rank and10%, necessary per-user rate is4.764924390924285t/s, equivalent209.86691875ms/step. Each step emits one token per active user; do not divide step rate by32. This does not assert all experts are touched, inflate the unchangedB1floor, count padding/CCL/redundant reads, or claim measured bandwidth. A future P4 rate below this conservative bound proves failure for every possible expert overlap within the inherited logical convention; a pass would not establish the stricter actual-work floor. Before use, verify final served prompt lengths<=74. No TT/profiler operation was used for the derivation.

### 2026-11-17 14:01 UTC — full configured sampling passed and cleaned

The fresh full profile completed in3380.02s:71passed,71skipped,zero failures, with all142test IDs exactly matching the original full run. Skip audit:70require speculative decoding; one requires speculative artifacts. No selection or assertion/cap relaxation. Six explicit request-mode overrides remain documented. All14fresh recall outputs were read: exact own phrase and normal stop in9–10tokens, prompts3117/9296 and3089/3115/3123/3149. Five bad-word constrained responses are exact by request/seed/content/stop versus both prior current-source and Stage9controls; odd constrained output is recorded separately in `bad_words_prior_control.json`, not accepted as a broad quality result.

Runtime log audit:1474device and988explicit host-logit decode steps in the full diagnostic API suite; not zero-host performance evidence. The primary/CI/native/shared-qualitative/async device-only evidence remains separate. No runtime/error markers. All six trace IDs remain stable across5157traffic events with no recapture/release. Owned cleanup releases6/6 and leaves no processes/deviceholders. `sampling_acceptance_summary.json` records counts, hashes, routes and cleanup. The serial continuation then started the tracked nonasync32/2048 server for exact growth comparison; no production/plugin change.

### 2026-11-17 14:07 UTC — opposite-scheduling acceptance passed

Tracked nonasync server runner98560/API98660 completed the six growth requests (129/160,130/112,131/80 in two admission orders). All output arrays exactly match the tracked async control. Runtime summary passes12checks over318submissions, logical pages5..8, changed/unchanged table behavior, slot remaps, all-device sampling and stable six traces. Maximum pending outputs1 versus2async; the async control also proves zero pre-boundary readbacks. `async_sync_acceptance.json` compares actual API argv from owned shutdown records: only async scheduling differs; adapter/layers/precision/context/capacity and allocated cache specs match. Cleanup released all six traces and left no owned processes/deviceholders. The continuation started the untracked audit0/tracker0 same32/2048 performance server, with the corrected identity collector queued after readiness.

### Eval setup correction before first measurement

The installed chunk probe defaults to10s background-decode warmup. Current serial32-user128-token prefill burst takes about100s to admit the last request, so10s would include initial background admissions rather than establish31decoders before the long prompt. Before any eval device/server/client launched, stopped only verified idleCPUqueue28867, preserved its cancelledprelaunch record and exactsource snapshot `continue_eval_sweep_v1.py.snapshot`, and restarted into fresh `optimized_acceptance_eval_v2`. All three budgets now use identical `--decode-warmup-s180`; request counts, lengths, output caps, ITLbound and fullcontext are unchanged. Final audit must verify31background128-token prefills completed before first long prefill. This corrects probe setup; no measured result was discarded.

### 2026-11-17 14:17 UTC — corrected-collector performance confirmation passed

Real TT-plugin same actual API argv and initialized adapter/layers/precision/context/capacity as baseline. Corrected identity proves audit0/tracker0 using nonempty inherited API-parent environment; profiler/watcher unset. All65canonicalrequests plus3matchedHTTPcontrols complete. Runtime259device/0host canonical and381device/0host matched; noerrors. Primary128/128/1: TTFT2969.747622ms,TPOT31.426704ms,ITLP50/P9931.431904/32.283667ms,end-to-end18.387105tokens/s,decode31.820072tokens/s/user. BaselineTPOT53.796971ms gives41.582762%reduction. Driftfromselected31.425571ms is0.003606%; primaryfloor21.631299ms stillfails. CI100/100/32:24.786966aggregate tokens/s,meanTPOT622.098996ms,TTFTmedian/P9956657.955235/87442.369569ms,ITL419.918015/877.595396ms. Prefill128/32/32 medianTTFT51639.491957ms,after/before0.982776667;serialpathonly. `confirmation_comparison.json` preserves everymetric andserverconfiguration equality; earlier`final_comparison.json`unchanged. Cleanup releases6/6traces,noownedprocesses/noholders. Evalv2queue then started budget512server; allthree budgets haveidentical180sbackground warmup.

### Final user cap — remaining required gates once

The user explicitly ended the optimization time/token budget: retain the best selected implementation, run only remaining required gates once, and record untried ideas as hand-off notes. Production stays frozen at the validated source. Completed sampling, mixed-shape, stale-input, async/sync, qualitative, native and same-config performance gates will not be repeated. Only the already-started three-budget eval sweep, long-context timing, 32-user eval load, installed stage check and independent review remain. Documentation corrections and local checkpointing follow. Non-blocking findings may carry; failed required gates remain labeled honestly. No new optional optimization or profiler collection is authorized by this continuation.

#### Carried to next stage

Untried larger changes: attention transport redesign; tiled combine/reduce fusion; row-aware router kernels; stack-compatible fractured residual/collective families; optional batched prefill with unchanged decode/trace retention and bit-equal batched-versus-solo proof. These are unmeasured candidates, not rejected possibilities or claims of impossibility. The existing checklist preserves the precise scope and prior evidence.

Current canonical benchmark, serial-prefill burst and shared qualitative artifacts promoted byte-for-byte from final acceptance runs, with previous canonical copies in `readiness_vllm/optimized_acceptance_canonical_archive` and hashes in `canonical_artifact_promotion.json`. Composite run summary indexes the separate current runs. No synthetic pass or batched-prefill serial_reference was inserted. Remaining-gate runner `finish_eval_once.py` and all modified Python tooling pass precommit.

### 2026-11-17 15:24 UTC — three-budget sweep complete, no qualified value

Each required budget ran once. With31 background128-token decoders and foreground4096/32768-token prompts (4outputs),180s admission warmup,capacity32/context262144:512 TTFT95.1399/772.6622s,worstITL25.6824s;2048 TTFT94.1334/736.9219s,worstITL92.4546s;4096 TTFT94.0028/742.5912s,worstITL186.7118s. All zero request errors; all31background rows admitted before long prompt; one unchanged six-trace identity set in each run; zero host decode routes. Every server cleaned with no holders. `optimized_acceptance_eval_v2/sweep_review.json` records the audit. None meets1000ms, so selected budget is null.

Started `finish_eval_once.py --budget2048 --diagnostic-no-qualified-budget` at15:23:56UTC (actual argv in execution record uses separated flags). The existing2048 production configuration is retained only for the required diagnostic P5/P4; it is not represented as a qualifying selection. Server fullcontext/capacity/mesh/precision are unchanged. P5 request timeout10800s/overall18000s accommodates measured long-prefill cost without shortening1K/8K/32K/128K probes. P4 remains32users×6000tokens with the eval profile and plain/eval comparison. Each runs once; failures preserved.

## Owner extension and P5 recovery — 2026-11-17 16:24 UTC

The owner loosened the stage wall cap to finish slow long-context gates and explicitly authorized cleanup of our own processes and device reset if needed. The interruption ended queue PID349928 and its `run_probe` wrapper, but the bounded P5 client PID362120 and server runner349939/API350001/EngineCore350356 survived. The 131072-token prefill continues. `recover_eval_final.py` waits for that exact surviving client, validates its fresh canonical result and HTTP idle boundary, then runs the remaining P4 gate once on the same server and closes the owned server. It does not repeat P5 or claim an unavailable orphan exit code. The recovery audit uses the entire dedicated P5 server log because the interrupted wrapper had not persisted its starting log offset. No device reset or model/source change was required.

Recovery command: source `../lane-env.sh` and `python_env/bin/activate`; `$TT_MODEL_BRINGUP_ROOT/scripts/wait_for --cmd 'timeout 25200 python models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_vllm/recover_eval_final.py' --file models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/optimized_acceptance_eval_final/recovery.log --timeout 25500 --tail 15`.

### 2026-11-17 16:33 UTC — P5 completed; P4 started

P5's surviving client completed all four context timings and all eight short-shape requests. Canonical result is fresh and copied into `optimized_acceptance_eval_final/long_context_timing.json`; the original interrupted execution record is preserved, and `records/p5_timing.recovery.json` records unknown orphan exit status, fresh result, two idle observations and the dedicated-server audit. With 1 user and32 output tokens, inputs1024/8192/32768/131072 measured TPOT31.50/31.86/32.37/34.75ms and TTFT22.4295/181.3235/722.5270/2932.6762s. No decode cliff. Three measured repetitions after warmup at127/128 input and4 output give medianTTFT3505.2/3044.2ms, ratio1.151PASS. All timing requests PASS; only chunk_size_sweep is advisory. P4 starts on the same idle2048 server with32users,6000output tokens,reasoning profile,plain/eval256-token comparison and the documented4.764924390924285t/s/user necessary batch floor.

### 2026-11-17 17:23 UTC — remaining hardware gates finished and cleaned

P4 completed32/32requests, each6000output tokens,maximum74prompt tokens. Total192000outputs/load2638.6s/wallaggregate72.8t/s. Plain/eval32user256output comparison medianTPOT421.53/421.20ms,ratio0.999PASS. First-to-last1000token bucket medianperuser2.36→2.28t/s; corresponding32×unroundedmedian estimates75.67→73.02t/s,0.965ratioPASS. Firstbucket2.36 fails conservative4.764924390924285necessaryfloor. Actualmaxprompt74validates its scope. P4 exit2 is preserved; newly measured failure hasfix_hours0, no fabricated time-box waiver. Allrequests/health/slots/KV/trace/preemption gates PASS. PeakKV0.7356,zero preemptions,health200,finalrunning/waiting/KVzero.

Installeddefaultforecast at32users:GPQA198×12000tokens8.98h(extrapolated),MMLU-Pro280×3000tokens3.12h,IFEval541×1500tokens2.99h,total15.08hbefore rowrounding versus14.4hplanningallowance. Eachindividualdefaultjobfits; combinedplanexceedsallowance. These areplanningestimates,not measuredreleasejobsoraccuracy; stage11mustuse itsrealplan,preferquestionsubsets overtruncatedreasoning.

P4audit6509device/0hostdecode,no errors. EntireP5/P4server6657device/0host,13411trafficevents,oneunchangedsixtraceidentityset,maxpending2,zero preboundaryreadbacks. Shutdownrecord proves6releases,noPIDs,noholders. Noreset. Nanobindshutdownreferencewarnings(2instances/19types/259functions)arepreserved;devicescloseandprocessesexit. FourretainedP4tailsread:threeinclude repeatedturnmarkers/post-answerdialogueunderignoreEOS; these tailscannotlocatefirstEOSandare notqualityacceptance. Sharedqualitativecontrolsremainseparate.

CPUcollector: `python models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_vllm/collect_eval_acceptance.py` from the repository root. Initialcollector incorrectly looked for per-trace release lines; corrected to use authoritative `shutdown_before_release`/`shutdown_after_release` events. No runtime rerun.

Installed `10-optimized-vllm.check.sh` ran once after all measurements: exit2. No degenerate output; contextcontract262144OK. Advisoryhistoricalnestedcontextcapstrings and optionalserial_referenceabsence remainclassified. P5chunkchoice and P4forecast advisory. The solecritical scriptedresult is P4functional_floor2.36vs4.76. `stage_check.json` and `stage_check.log` preserve invocation/output. Independent review follows; no passing verdict is claimed.

Final precommit checked175stage files. Only missing terminal newline in the installed P4 `answers_sample.json` output was corrected; JSON values and all measurements are unchanged. The follow-up hook run covers that file; other hooks passed. No new artifact compression, hashing or archiving was added.


## Review round 1 remediation — 2026-11-17 17:40 UTC

Independent `stage_review_round1.md` returns more-work-needed. RW1 finds fused row masking incorrectly enabled for prepared capacities 3 and 5–31. The minimal repair in `tt/model.py` enables that helper only for 2/4/32 rows; other capacities retain the existing generic extraction and per-layer activity mask. The selected capacity32 executable kernel path is unchanged. Host dispatch checks execute every prepared bucket at every capacity 1–32; all eight adapter tests pass. Full40 capacity8 real-plugin lifecycle validation is running with context262144, budget2048, async/device sampling, tracking1 and skip_program_cache0. Commands and ownership are in `../../readiness_vllm/optimized_capacity8_repair/execution.json`; original completed P5/P4 measurements are not repeated for this unaffected capacity32 path.

RW2 remains the unwaived single-user floor failure and its completed thirteen-hypothesis AutoFix failure. RW3 is being investigated independently through the AutoDebug launcher; report destination `p4_floor_repair/AUTODEBUG.md`. The child sandbox preflight failed because this host prohibits unprivileged namespaces; retried with the documented `AUTODEBUG_SKIP_CHILD_SANDBOX=1` under the existing unrestricted execution authorization. The child remains inspection-only. No previous B1 repair time is assigned to P4.

Forecast clarification: the default GPQA 12000-token answer forecast also exceeds measured KV capacity at 32 users: 32×(74+12000)=386368 logical tokens versus a263168-token pool. Its optimistic timing excludes potential preemption. Stage11 must size the actual release plan from measured lengths and available cache.

### RW1 focused validation complete

`run_capacity_repair.py` ran full40 capacity8/context262144/budget2048 through real vLLM TT-plugin serving. All18 requests returnedHTTP200 with96 outputtokens; prompt lengths129–133 and4101, greedy plus top_k64 sampling. All23 lifecycle checks pass, including six stable traces,95 replays of the restored8-row generic path, both sampler widths, chunk continuation with intervening decode, changed/unchanged page inputs and async overlap. All17 greedy arrays exactly equal the capacity32 control. Mixed-traffic ITL median69.4895ms/P99 122.3701ms/max52390.6619ms is diagnostic capacity8 evidence, not the primary throughput result or a1sITL pass. Cleanup passed with six releases and no processes/device holders.

The wrapper initially exited1 only in postprocessing: its sampled top_k64 fixture was compared with control top_k40. Both use native64, but sampled trajectories are not a matched comparison. Corrected the comparison to require matched greedy settings and arrays, preserving the original failed execution and all raw outputs. CPU-only `repair_acceptance.json` records the passing hardware checks and17 greedy comparisons; no hardware rerun. Evidence: `../../readiness_vllm/optimized_capacity8_repair/{execution.json,lifecycle_requests.json,lifecycle_summary.json,repair_acceptance.json,server/shutdown_record.json}`. Host command/result: `capacity_dispatch_host.log` and XML (8passed). Selected capacity32 code still chooses fused masks for2/4/32, so its completed acceptance measurements remain applicable; historical source hashes are kept as historical, not falsely matched to the repaired file.


### P4 AutoFix discriminators — 2026-11-17 18:08 UTC

Independent AutoDebug completed at `p4_floor_repair/AUTODEBUG.md`. Two isolated device discriminators now pass: prefix-balanced local combine,36cases acrossseq32/33/64 and all4ranks with exact metadata/output and trace reuse; batch32 compensated input norm,64 distinct recorded layer0 activations plus finite edge cases, exact downstream arithmetic/final physical outputs and tracked changed/unchanged replay. Both preserve completion barriers and the selected precision. Component timings are recorded in `p4_floor_repair/AUTOFIX.md`; none is serving acceptance.

The norm probe's first failure was a harness-oracle mismatch at the packed wide-input boundary: existing BF16 tilization maps signed zero/subnormals to positive zero. The corrected check verifies that exact independently measured map, then still requires every downstream arithmetic intermediate and final output to be bit-identical. The original failure remains at `readiness_vllm/p4_batch_norm_01/acceptance.json`; corrected acceptance and separate untracked timing are in `p4_batch_norm_02`.

Temporary combined candidate source is preserved in `p4_floor_repair/combined_candidate.patch`. The native factory is built/installed; the norm branch is limited to supported replicated B32 linear inputs. `run_p4_repair_candidate.py` runs tracked mixed-shape serving and17 exact greedy control comparisons, then a fresh tracking0 server measures32users/1000outputs with the same eval profile, budget2048 and context262144. The latter is a first-bucket repair discriminator, explicitly not a replacement for the canonical32×6000 P4. Canonical artifacts are untouched. This required-failure investigation does not restart optional optimization search.

Launch: `$TT_MODEL_BRINGUP_ROOT/scripts/wait_for --cmd 'timeout 7200 python models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_vllm/run_p4_repair_candidate.py' --file models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_vllm/p4_floor_repair/serving_candidate.log --timeout 7300 --tail 15`, with the lane environment and Python virtualenv. All three new probe/driver files pass precommit. Candidate disposition is pending; do not use original acceptance as proof of this temporary native/model implementation.


### P4 repair finished; retained implementation restored — 2026-11-17 18:35 UTC

The combined candidate passed23 mixed-shape lifecycle checks and17 exact greedy control arrays. A fresh untracked real-plugin server with actual API argv equal to canonical P4 completed32users×1000outputs, maximum74input. First-bucket2.56t/s/user still fails4.764924390924285. Plain/eval32×256medianTPOT388.33/388.00ms,ratio0.999. No engine/resource/trace/sampling failures:1509device/0hoststeps,999fully occupied B32 long-load steps,zero preboundaryreads,6captures/6releases,allcleanupPASS. The full retained32×6000 P4 remains2.36t/s/user and failed. No shortened diagnostic was promoted to canonical acceptance.

AutoFix disposition is failure to restore the floor, not an impossibility claim. The promising candidate patch remains in `p4_floor_repair/combined_candidate.patch`; complete candidate sampling, full-model accuracy,6000-token load and final selected-path gates were not run under the owner cap. Production norm/combine edits were reverted to HEAD; only RW1 capacity dispatch remains. Native rebuild and both install components pass, then36 restored component cases exactly match baseline with tracker/stale-input/fallback controls. No device reset was needed. `p4_floor_repair/serving_disposition.json` records the outcome and raw paths. Four saved diagnostic answer tails are empty; P4 persists content only, discarding separately collected reasoning. This is not a qualitative result or a TT-specific regression finding.

Counted P4 repair time is conservatively0.5786hours for18:00:00–18:34:43UTC, excluding earlier AutoDebug/host work and the B1 campaign. No2h waiver is claimed. Both limitations files preserve unwaived floor failures. The prior installed stage-check exit2 remains applicable because canonical results and the selected capacity32 executable path are unchanged. No unchanged expensive gate is repeated.

#### Carried to next stage — unadopted repair and untried ideas

The exact tested combined patch is a promising starting point, requiring complete acceptance before promotion. Remaining larger ideas are row-independent ordered projection batching with per-row carry, recurrent/state graph batching, duplicate residual packing removal and page-refresh upload reduction; their effect and correctness are unmeasured. These join the earlier optional geometry/transport/collective and batched-prefill notes. Neither floor is a non-blocking carried acceptance item; both remain unresolved required gates after AutoFix failure.

Final remediation precommit inspected214stage-owned files. Only terminal newlines in the capacity JUnit XML and diagnostic answer-tail JSON were normalized; XML/JSON values and measured results did not change. Raw original failure/status records remain intact. No new compression or hashes were introduced.


### Final independent review and incomplete checkpoint — 2026-11-17

[Review round2](stage_review_round2.md) returns **more-work-needed**. RW1 is closed by the capacity repair and focused evidence. RW2 (primary single-user floor) and RW3 (canonical batch32 floor) remain required and unwaived; both now have explicit scoped AutoFix failures. The reviewer found no new defect in the retained implementation and requests no unchanged expensive rerun. The original user stop condition permits ending after AutoFix fails, but this does not satisfy stage acceptance. No third review is used to relabel mandatory failed gates as non-blocking carried work.

The final documentation corrects the earlier P4 repair-time summary to0.5786h and marks completed runtime gates/cleanup as completed. Historical chronological pending entries remain historical. The combined prototype is not selected; its host ownership reproducer requires applying `p4_floor_repair/combined_candidate.patch`, which passes `git apply --check` against the restored tree. Untried ideas remain in the handoff above; their correctness and performance are not claimed.

Checkpoint scope is the stage-owned files under this autoport only, including the capacity dispatch repair, completed gate evidence, review and honest limitations. The native candidate library/source and model norm changes are restored and validated. Plugin commit remains5d1785b76d17a8989c9495f6939abea6cef62f09, clean and unchanged; installed vLLM0.31.0 is unchanged. No push is performed. The local checkpoint SHA is recorded in the following closeout entry.

### Local checkpoint recorded

Repository `tt-metal`, branch `main`, commit `5313d9b211e767489e88847a84607f6a142edb3f` contains217 stage-owned implementation/evidence files. All commit hooks pass. Final focused documentation hooks pass (`closeout_precommit.log`); whitespace checks pass outside preserved patch context lines. Fresh `/proc` inspection finds no vLLM API/EngineCore process or TT device holder. The plugin remains clean at `5d1785b76d17a8989c9495f6939abea6cef62f09`; no plugin/vLLM edit or push occurred. This is an **incomplete acceptance checkpoint**, not stage completion. The remaining blockers are the unchanged required floors after the documented AutoFix failures.


## Owner acceptance and final review — 2026-11-18

Vladan explicitly accepts the two recorded stage10 speed-floor misses and directs completion with no further speed work. The exact runner-delivered message and measured scope are in [owner_acceptance.json](owner_acceptance.json) and [owner_decision.md](owner_decision.md). Primary128/128/1 remains31.820072t/s/user versus46.229309; canonical32users×6000out, maximum74input, remains2.36t/s/user in its first1000-token bucket versus4.764924. Raw measurements and prior failed review/AutoFix records are unchanged. Both limitations files record explicit owner acceptance, not an inferred time-box or HF-same waiver.

Keep checkpoint5313d9b211e767489e88847a84607f6a142edb3f's validated implementation/configuration and capacity fix. No source, precision, context, sampling, trace, plugin or vLLM change is made. The temporary P4 prototype remains unadopted. All required hardware runs already finished; no speed work or hardware rerun is performed. Round3 is a fresh independent diff-only review of the owner disposition and documentation.

The installed checker is read-only and lacks an explicit-owner-decision branch. It recognizes only same-HF behavior or a two-hour fix attempt; P4 remains0.5786h with original_model not_run. Preserve raw exit2 and failed measurements; do not falsify fields to force a machine pass. The owner's instruction revises the acceptance contract for precisely these two speed rows.


## Final completion — 2026-11-18

Independent [review round3](stage_review_round3.md) returns **pass-with-carried**, with no Required Work under the revised owner contract. RW1 stays fixed; RW2/RW3 are closed as acceptance blockers by the explicit owner decision, not by a new performance result. Stage10 is complete with accepted limitations and the unchanged validated configuration. No additional speed work or hardware run occurred.

The CPU-only stage checker ran once after the owner decision using the same installed `10-optimized-vllm.check.sh`, lane environment and Python virtualenv. `owner_acceptance_stage_check.json` records exact command, environment scope and exit2; the log has exactly one critical row, the owner-accepted P4 floor. No passing raw checker result is claimed. `owner_acceptance_consistency.json` verifies matching owner/limitation rows, exact primary TPOT-derived rate, unchanged raw measurements/review2 and preserved original_model/fix_hours. The installed checker remains unchanged. A runner must apply the explicit owner decision separately from its old mechanical speed-floor result.

Fresh read-only cleanup (`owner_acceptance_cleanup.json`) shows no serving processes/device holders. Documentation hooks pass (`owner_acceptance_precommit_final.log`); first run only added the review brief's terminal newline. No runtime source, plugin, vLLM, selected precision, sampling, context or trace change; no push.

### Carried to next stage

- Accepted speed limitations:128input/128output/1user31.820072t/s/user versus46.229309;32users×6000outputs,max74input,first1000-tokenbucket2.36 versus4.764924. These are owner-accepted known limitations, not further stage10 speed work.
- Mixed-traffic ITL:512/2048/4096 budgets reach25.6824/92.4546/186.7118s against1s. No budget qualifies;2048 stays the retained diagnostic configuration.16384 background output tokens are caps, not completed-answer claims.
- Eval planning:15.08h forecast versus14.4h allowance;GPQA12000-token extrapolation at32users exceeds the measured263168-token KV pool (386368logical tokens at prompt74). Use the actual release plan and measured answer lengths; allow for cache/preemption costs.
- Quality evidence is bounded: shared qualitative checks remain separate; P4 forced-ignore-EOS tails and candidate content-only tails do not establish completed-answer accuracy. Candidate reasoning text was not retained. No new quality waiver is introduced.
- Unadopted combined P4 patch and untried geometry, transport, embedding, head-to-sampler, projection/state batching and optional batched-prefill ideas remain conditional handoff opportunities. Full candidate accuracy/sampling/6000-token and selected-path acceptance are required before any future authorized promotion. No further optimization in this stage.
- Automation/evidence: installed checker lacks explicit-owner acceptance and still exits2; preserve this alongside the owner record/review. P4 summaries omit full SSE timestamps/reasoning; large combine payloads and normal logs remain persistent local artifacts linked by summaries. No unsupported numerical/content claim follows.

Local closeout commit and SHA follow below.

### Completion checkpoint

`tt-metal`, branch `main`, commit `2eced3971cd64dce58d9765c81839bf4a4032e2f` records owner acceptance, updated limitations, independent round3 **pass-with-carried**, final status and carried notes. All commit hooks pass. Validated runtime remains checkpoint `5313d9b211e767489e88847a84607f6a142edb3f`; the closeout changes documentation only. Plugin stays clean at `5d1785b76d17a8989c9495f6939abea6cef62f09`, installed vLLM0.31.0 unchanged. No push performed. The following bookkeeping commit records this SHA; it adds no runtime change.

## Runner failure reproduction and scoped refutation — 2026-11-18

The remediation request names `10-10-optimized-vllm.resume-2.check-1.log` and permits either a passing rerun or a concrete refutation in this work log. I reproduced the exact installed check from the checkout using `python_env`, `MODEL_DIR=models/autoports/qwen_qwen3_6_35b_a3b`, `HF_MODEL=Qwen/Qwen3.6-35B-A3B`, and the installed package's environment setup. Command: `bash /home/ttuser/qwen36-35b-a3b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/prompts/model_bringup_multigoal/10-optimized-vllm.check.sh`. The fresh run at 01:07:38–01:07:42 UTC returned **exit 2**, with exactly one critical row: `long_answer_load/functional_floor`, 2.36 tokens/s/user versus 4.76. This line exactly matches the reported runner failure. [Reproduction metadata](runner_failure_reproduction.json) and [complete output](runner_failure_reproduction.log) preserve the command and result.

The numerical shortfall is valid and is not refuted. The conclusion that it introduces new required stage-10 work is refuted by the original owner instruction at `/home/ttuser/qwen36-35b-a3b/multigoal_logs/10-10-optimized-vllm.resume-2.message.txt`: “ACCEPT the two missed stage-10 speed floors as recorded known limitations” and “No further speed work in this stage.” The original file exactly matches `owner_acceptance.json`; this is verified from the source message, not inferred from the completion prose. The current remediation allows refutation and does not explicitly revoke that scoped decision. All other original requirements remain applicable.

The CPU artifact audit in [runner_failure_refutation.json](runner_failure_refutation.json) verifies 32 error-free requests, each with 6000 output tokens (192000 total), maximum prompt length 74, and the canonical failed first-bucket rate. It recomputes primary decode as 31.820072142305893 tokens/s/user from raw TPOT. The raw primary benchmark, canonical P4 result and request records, and batch-32 floor calculation are byte-identical to validated checkpoint `5313d9b211e767489e88847a84607f6a142edb3f`. Tracked changes from that checkpoint to reproduction HEAD `3ba67399d1c97bd6880c5a266c2026fdef201a18` affect documentation only. Both limitations files agree on this exact owner-accepted row.

The cause of the repeated acceptance failure is a checker/contract mismatch: installed `scripts/check_eval_readiness.py::waiver_reason` recognizes `original_model == "same"` or `fix_hours >= 2`, but has no explicit-owner-decision branch. Calling it with the actual P4 row and limitations returns `None`. The truthful fields remain `original_model: not_run` and `fix_hours: 0.5786`; the raw gate remains failed. Changing those fields or the measurements would manufacture a pass. The installed package stays read-only. The existing independent round-3 review already inspected this same discrepancy and returned `pass-with-carried`; this documentation-only reproduction introduces no new runtime path and does not start a fourth review round.

Disposition: use the requested **concrete-evidence refutation** route for this runner report. Retain the validated configuration, original measured shortfalls, explicit owner decision, and existing review. No performance repair or exit-0 result is claimed. Automated advancement still requires the runner to recognize the scoped owner disposition; rerunning the unchanged numerical checker cannot do that. No server, hardware job, reset, runtime edit, plugin edit, or vLLM edit was performed. Existing carried latency, forecast, and evidence limits remain in force.

### Refutation checkpoint

Local `tt-metal` branch `main` commit `df7c7eb4` records the reproduced failure, scoped refutation audit, and documentation. All focused pre-commit hooks and commit hooks pass; `git diff --check` passes. Hook output is retained at `/home/ttuser/qwen36-35b-a3b/multigoal_logs/stage10-refutation-precommit.log`. The follow-up bookkeeping commit includes this SHA and explicitly adds the ignored reproduction log so the recorded failure is reviewable with the checkpoint. No push was performed.

## Completion audit after runner remediation — 2026-11-18

[completion_audit.json](completion_audit.json) maps every one of the 21 original goal requirements to inspected current artifacts. The audit independently compares actual before/after server argv, adapter identity, all40 layers, selected precision, context262144/capacity32 and normalized benchmark metrics; parses the full JUnit (71 passes,71 skips,0 failures); checks the23 mixed-lifecycle conditions, opposite-scheduling controls, three-budget timing evidence, eval cleanup, and existing independent round3 verdict. The pinned plugin is clean at5d1785b76d17a8989c9495f6939abea6cef62f09. Runtime files remain unchanged since validated checkpoint5313d9b2; documentation-only remediation reuses unaffected hardware evidence and does not claim new device validation.

The fresh read-only process scan finds no visible serving processes or device holders;397 processes deny access, so it is not a host-wide absence proof. Existing run-owned shutdown records independently cover the campaign owners. Historical cumulative-contract and command-document status text now points to the dated current disposition. The raw exit2 and real speed deficits remain preserved; completion uses the objective's concrete-evidence refutation alternative and the exact prior owner acceptance. No original correctness, sampling, context, trace or cleanup gate is waived. No required work remains under that contract.

Completion audit checkpoint: `tt-metal`, branch `main`, `b271c642dbba12afd15dea1fd77572668ea98bdf`. Focused and commit hooks pass; whitespace check passes. No push. The following bookkeeping commit records this SHA only.


## Requested waiver-schema reconciliation — 2026-11-18

The owner again accepts the two speed misses and requests the exact `check_eval_readiness.py` waiver shape plus an advisory gate result. Updated the canonical P4 `items` row in `doc/eval_readiness/limitations.json` and its optimized-vLLM mirror: probe `long_answer_load`, check `functional_floor`, nonempty owner-acceptance/failed-AutoFix reason, measured load result and this work-log evidence paths, original_model `not_run`, and honest fix_hours `0.5786`. Existing owner acceptance and raw numerical failure are preserved.

[Repair-time audit](waiver_repair_time_audit.json) records2083seconds from18:00:00 to18:34:43UTC, or0.578611hours rounded0.5786. Even the broader documented diagnosis milestone17:40 to restoration18:34:43 is only0.911944hours, not two hours. Earlier single-user work, acceptance runs, subsequent waiting and waiver bookkeeping are not relabeled as P4 repair; parallel agent time is not summed into elapsed wall time.

Re-ran installed `10-optimized-vllm.check.sh` with lane environment, Python virtualenv, MODEL_DIR and HF_MODEL as recorded in [execution](waiver_schema_stage_check.json). It again exits **2**, with one critical row: the accepted P4 floor. [Full output](waiver_schema_stage_check.log) is retained. The requested automatic advisory outcome is **not achieved**. The schema is valid; the numeric condition `fix_hours >= 2` is false, and the checker has no explicit-owner-decision branch. The owner-accepted stage disposition and round3 pass-with-carried remain valid; automated runner advancement is still unresolved. A metadata-only edit cannot truthfully satisfy this condition. The runner/checker must apply the explicit owner decision rather than requiring invented repair time. No installed checker edit, runtime edit, speed work, device job or fourth review occurred.

Waiver-schema checkpoint: `tt-metal`, branch `main`, `fb40e21104ca468d197f73516696b8d11f216735`. Focused and commit hooks pass; no push. Automatic gate remains critical for the accepted speed row; no advisory result is claimed.


## Concrete checker repair proposal — 2026-11-18

The previous turn made progress by recording the exact waiver schema, honest repair time, and a new exit2 result. The same remaining blocker is the installed checker's lack of explicit owner acceptance. Prepared [an unapplied checker patch](checker_waiver_proposal/owner_speed_waiver.patch), restricted to the long-answer functional performance floor and requiring exact owner-decision/source evidence. [Validation](checker_waiver_proposal/validation.json) passes21host checks: candidate eval advisory1 against unchanged real artifacts, installed eval critical2, and negative cases preserving unrelated correctness, non-waivable and unauthorized rows. No speed work or installed package mutation. This is proposal validation, not an installed full-gate pass.

The installed model-bringup skill's Startup instruction says “Keep the installed package read-only.” The current request authorizes limitation metadata and gate rerun, not an exception to that package-write boundary. A concrete tested change is ready for owner authorization. No fourth stage review is required for this unapplied proposal; if adopted, its actual full-gate result must be recorded.

Proposal checkpoint: `tt-metal`, `main`, `949f2772a852c15d93ee85953500d607784bbccf`. All commit hooks passed. Proposal remains unapplied; installed checker and runtime unchanged. No push.
