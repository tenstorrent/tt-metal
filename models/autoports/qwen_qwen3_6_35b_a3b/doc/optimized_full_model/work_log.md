# Optimized full model work log

Stage 7 started 2026-11-16 from ece17d22, clean worktree. Target Qwen/Qwen3.6-35B-A3B revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7, 1x4 Blackhole chips 0–3. Stage wall cap defaults to four hours. No vLLM integration, no broad precision sweep, no push.

Preserve accepted replicated BF16 inter-layer residuals (the actual selected decoder contract), BFP8 weights, HiFi2 ordinary and LoFi active-expert math, BF16 KV/CCL, FP32 recurrence. Preserve existing rejected-candidate ledger. Stage 5 linear performance limitation and stage 6 owner-accepted HF-reproduced resume PCC limitation remain recorded in their original artifacts.

Startup: enabled tt-autodebug 0.1.8 and tt-model-bringup 0.1.24 verified by `codex plugin list`; shell environment from lane-env.sh and packaged environment.py. `fuser /dev/tenstorrent/*` found no holders. `timeout 60 tt-smi -ls --local` found all four p300c chips. No reset.

Baseline harness correction: allocation tracking default changed from 1 to 0, with explicit 1 still supported. Diagnostics must be separate from headline timing; both before/after use the same untracked harness. Historical stage6 tracked performance is not the comparison baseline.

Baseline command: after lane-env.sh, `wait_for --cmd "env TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 1800 python -u -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_performance --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/before.json" --file doc/optimized_full_model/logs/before.log --timeout 1850 --tail 6` (log path relative to model directory in this prose). Startup and both new probe harnesses pass repository hooks.

Fresh untracked baseline PASS: before.json, 430.727s, TTFT2787.782ms, queued token-out25.081181ms (39.8705t/s/u), caller-delivered25.158932ms. Zero resident-loop token/position/page/activity refresh, readback or capture; clean closure. This confirms the calculated stage7 FUNCTIONAL floor miss (21.6313ms ceiling), distinct from the small wrapper gap against24.9935ms prior layer estimate. AutoDebug launcher preflight failed because nonprivileged user namespaces are disabled; explicit AUTODEBUG_SKIP_CHILD_SANDBOX=1 retry uses existing subagent/read-only authorization, preserving diagnosis isolation. Logs: autodebug_launch.log/autodebug_retry.log.

First terminal candidate PASS: head_r3_b8_c16.json. Real reduced0/3 AIME hidden, full-vocab logits bit-equal, traced terminal0.451994→0.367281ms. This is a terminal component win, not an all40 claim. Serial geometry sweep follows with the same BFP8/HiFi2/FP32-accumulation policy; storage padding leaves local N62080 and vocabulary indices unchanged.

Geometry r3/block32 fails static L1 allocation (2080704B beyond1572864B) in the reduced probe; r3/block8 and r2/block8/16 passed. Recovered via bounded tt-smi reset after idle fuser; reset exit0. A follow-on health command was accidentally launched before reset completion was collected, so its passing results are discarded as acceptance evidence. Both processes closed normally; a fresh strictly serial list/mesh-ones smoke is the recovery source (health_serial_confirm.log, mesh_serial_confirm.log). No foreign processes killed or locks removed.

Terminal geometry continuation: r3/block4 wins component0.353635ms versus same-run0.452275ms baseline; r3/block16=0.382754, r2/block8=0.371851, r2/block16=0.417201ms. All passing candidates are bit-exact on the reduced real activation. One-reader/block1 full width fails static L1 at2408960B; an adapted split-head control is being prepared rather than rejecting the family from this first error. Reset-after-r1 uses one serialized bounded shell list/reset/health sequence; no foreign process holders.

Setup audit: BoundedPrefill._layer_key serializes compute configs with str(), which includes Python object addresses. CPU-only construction of two identical WormholeComputeKernelConfig objects confirms distinct keys despite identical six semantic fields. This makes all 40 layers warm every prefill shape independently. A semantic config key is a generator-setup candidate; it must retain distinct numerical policies and pass tracked no-new-program checks before adoption. Baseline boundary run remains unmodified while active.

Baseline boundary driver PASS (574s): before_prefill.json, warm TTFT64=1474.063ms,255=5905.173,256=5350.840,257=5975.767,1025=21825.866,2049=43157.731. All2-step token-feedback checks preserve trace IDs/program cache and advance positions on device. This repeated AIME fixture is capacity/performance evidence, not quality evidence.

02:00 terminal pressure controls PASS: target581696 persistent L1 bytes/bank. r3/block4 baseline0.460104→candidate0.361358ms; r3/block2=0.360486ms (0.24% difference, not a meaningful further win). Adapted one-reader split4 fits and is bit-exact, but2.398756ms versus0.460045 baseline, so rejected. Retain r3/block4/core16/N32. Model terminal integration now uses this geometry for one physical tile of rows, with unchanged FP32norm→BF16 and BFP8/HiFi2/FP32acc. Legacy many-row API retains explicit native geometry; serving bounded prefill and B<=32 use the sharded path. Original head is retained for legacy API, costing an additional~135MB/chip, within inherited8.6GB max-context spare capacity; final smoke/census pending. The isolated probe now explicitly selects interleaved control. Production integration plus semantic prepare-key edit passes repository hooks; CPU key test confirms equivalent config objects collide and changed numerical fields remain distinct.

AutoDebug external launcher completed and report moved into this stage directory. AutoFix repairs are isolated into structural, scan-reader/store, and exact direct recurrent-sum hypotheses. No rejected numerical policy is promoted. Structural component acceptance PASS under TRACKING=1/SKIP_PROGRAM_CACHE=0, nine interfaces×threechanged inputs×fourranks, including R1/3/31/32 BF16/FP32 collectives with nonzero padding. Exact output equality, no tracker exemption; integrated output/state controls follow. Commands/logs retain individual flags and failures.

AutoFix structural integration PASS: structural_permute, structural_padding, structural_combined_l34 (192 steps,seed154) and structural_l3_b3 (129 input,16decode,B3) exactly match selected-mesh prefill, decode and all saved per-rank states with tracking1/skip0. These timings are diagnostic only. The new driver also exposes evolving128-step clean FUNCTIONAL timing for layer0/3; this is a measurement control, not a waiver of full-model floor.

Direct recurrent-sum initial CPU cross-review found uninitialized ADDR_MOD_7; fixed before hardware. Helper is explicitly limited to actual L1 interleaved products, and comparison now requires4ranks/dtype/shape/raw-bit equality. recurrent_sum_acceptance.json PASS264 checks with Watcher10/tracking1, logs preserved under watcher_recurrent_sum. Clean recurrent_sum_timing.json PASS: head-major30.326→11.170us and terminal-row32.134→9.569us including stable-output copy. Both retain exact16-then8 additions and current product operands. Whole-layer consumer qualification pending at this point.

Scan probes: read32/read64/nop first artifacts used32 activation width shards; reviewer identified production output scan uses8×[32,128]. Raw results preserved as exploratory only, corrected *_production_layout.json reruns pass20 fixtures with exact coefficients/products/K512partials/final casts and changed-input traces. Strict mode gates now require trace coverage and tracking1 for acceptance, tracking0/no diagnostics for timing. elide/read32_elide and cache/cache32/cache32_elide all pass20 fixtures with Watcher10; original logs preserved in watcher_scan_*.

Clean scan ranking (actual227, current production consumer/layout): read32 saves input111.770→99.893us, output66.846→53.680us, router35.892→35.094us. Terminal-store elision alone and combined withread32 add no material gain; not selected. Initial cache32 is slower (input112.109,output65.047,router55.734us) and cache32_elide is similar. Reject these as measured; investigate its full-cache zero initialization separately rather than promote a nominal reuse optimization. Current retained candidate is narrow32 reads, direct recurrent sums and structural deletions, pending integrated/full-model gates.

02:20 combined_read32_sum_l0/l34 PASS tracked exact prefill/decode and8 saved per-rank state tensors at128/192steps. Cache initialization experiment: cache32_poison and cache32_noclear PASS20cases with Watcher10/tracking1, including exact products/partials/finals and changed-input trace replay. Consumed column0 values remain initialized; unused padding may be qNaN without changing outputs. Clean no-clear timings: input93.846us/output46.820us versus narrow-only99.893/53.680us, router37.480us versus35.094us (reject cachedrouter), K1024 103.044us versus100.390us (reject cachedK1024). Full2048read no-clear remains slower. Integrated selective cache32 is being tested only on exact K512 input/output shapes.

02:23 product_sum_{dram,l1}_watched PASS528 bitwise checks each, including scaled-Q/plain-K, both output layouts, real/recurrent/cancellation/signed-zero fixtures and changed-input traces. Clean timings show old product+accepted directsum18.69us→12.38us for L1 plain-K and DRAM scaled-Q terminal18.79us→13.63us. Integrated production_cache_product_l0/l34 PASS exact output/allstates/policy at128/192steps. Their tracked~1.133 speedups are diagnostic, not headline.

One intermediate cache integration attempt (combined_cache32_sum_l0) stopped on the probe's source-anchor guard after production source edits changed inspect.getsource during its run. No device failure and no numerical result; the failure is preserved. Subsequent probes use explicit production full_model_optimized=True versus standalone False, avoiding source-string patching for the promoted path. The accepted production helper kernels/descriptors are CPU-checked against the passing isolated variants; model-level gates remain pending.

02:25 production integration frozen: full_model_optimized=True on QwenModel; standalone decoder defaultFalse preserves paired baseline controls. Concrete helper kernels/descriptor ASTs match qualified probes; no production source rewriting. First terminal-only sampler launch caught a TTNN Shape slicing API TypeError in new DramHead construction, fixed by tuple(weight.shape); no kernel failure/reset. sampler_acceptance_v2 and sampler_timing_v2 PASS all3fixtures×3correctgreedy strategies: real-terminal split0.487768ms, power2split0.521668ms, forceargmax2.665826ms. Retain split physicaltop32/semantick1,p0,temp1. Negative-max and cross-rankties allgreedy andlowest-ID.

Clean full-generator-selected decoder controls PASS exact baseline/candidate output/state with diagnostics disabled: production_functional_l0 baseline0.719497745→candidate0.632541260ms; layer3 0.341083533→0.320842714ms, each3×128 changing-input steps. Estimated optimized40-layer stack=22.184664941ms, terminal/sampling excluded (candidate1_layer_bound.json). The isolated linear layer's10% floor still fails (8.1937%), preserved in its raw artifact; full-model acceptance uses complete useful-byte accounting, not this exactness driver's exit status. All40candidate1 performance is running.

Candidate1 all40 clean performance completedPASS execution295s with separate floorFAIL: TTFT2768.434ms, queued22.272769ms44.8979t/s/u, observed22.334663ms, fixedmodel21.791465ms, actual99HFteacher21.788870ms. Programcache8447stable;127model+samplingreplays withzero refresh/readback/capture; cleanclose. Achieved9.71199% useful-byte floor againstrequired10%; remaining0.641471ms, notaccepted. Estimatedstack22.184665ms plus~0.36ms terminal+0.489ms sampling exceeds measuredloop because isolatedblockinglayer estimates include distinctdispatch/measurement overhead; do not present subtraction asdeviceadditive decomposition. No>15% wrapper gap. Further exact fullstate-slab fusion targetsremaining5pointwise launches/materializations.

Post-candidate source review hardened DramHead dispatch to totalpaddedrows32, so arbitrary leading batch dimensions retain nativelegacy fallback. Selectedgeneratorgeometryunchanged. Current teacheraccuracy refresh runswithtracker1/skip0.

Candidate1 teacheraccuracyPASS (tracker1/skip0):100 predictions,top1=98%,top5=100%,top100=100%; candidate1_teacher_accuracy.json. Timing isdiagnosticandseparatefromcleanperformance.

Fullstate-slab fusion: independentCPUreview foundandfixed CB0cross-producer counters andBF16value-unpack sourceformat beforehardware. state_slab_r1_persistent_watched PASS288 exactstate/outputchecks; all4 rounded0/1×persistentDRAM/temporaryL1 variants pass128step Watcher/tracker checks. Clean same-oracle timing roundedpersistent66.064→35.971us, unroundedpersistent66.158→35.337us, roundedtemporary(includescopycommit)67.676→38.909us. This30us/layer componentgain targetsremaining21.38us/layer requiredbyfull-model floor. Integrationl0/l34/B3nonaligned isrunning; nofull-modelgainclaimeduntilmeasured.

Samplerpersistentbroadcast experiment firstwatched invocation stoppedon its strictaxisassertion: sharedsamplingpassescluster_axis=None for1Dmesh, whereasproductionmapsit tophysicalaxis1. No nativecandidatecollectiveexecuted, cleanclose, noreset. Experimentalguardnowvalidatesmesh(1,4), normalizeslastdim, acceptsNone/1 andcallsexplicitaxis1; nextfreshartifactrequired.

02:44 full state-slab integration: layer 0 (128 steps) and layer 34 (192 steps) pass exact output/state comparison. Batch-3 layer 0 stopped in the unchanged baseline: optional `positions` entries are None but decode_batch calls position.get. This is a Python contract defect, before candidate execution; a minimal repair and rerun are required. Persistent sampler broadcast v2 reaches hardware but fails first greedy token equality (all candidate rows return 127615); candidate remains isolated and rejected pending diagnosis. Both failures close the mesh normally, with no reset.

02:45 production slab dispatch is frozen. B3 layer0 length129/16 decode exact output/state comparison passes after the optional-position repair. Clean changing-input layer0 (3×128 steps) passes exactness at 0.602988375 ms versus original 0.720520261 ms; its separate standalone linear floor remains false. All40 token-out floor is the acceptance gate. Kernel/descriptor identity,33 shape/layout guards and8 dispatch checks are recorded in state_slab_production_cpu.json and state_slab_dispatch_cpu.json.

02:51 candidate2 all40 clean performance passes execution AND full-model useful-byte floor: queued21.424391835ms,46.67577t/s/u; caller-observed21.493701142ms; actual99-step HF teacher20.940451970ms; sampler0.487583ms. Floor achieved10.0966% (ceiling21.631299ms),14.58% less queued latency than baseline. Programcache8440 stable,127 replay pairs with zero refresh/readback/capture. TTFT2880.454ms is higher than baseline2787.782ms; final same-harness/boundary refresh must distinguish variation from regression. Updated isolated stack21.297627850ms (candidate2_layer_bound.json), terminal/sampler excluded; separate blocking layer measurements are not an additive device decomposition. After hardware closed, pinned Black23.10.1 formatted mesh_direct_recurrent_sum.py and run_stage7_structural_probe.py only; final source gates follow. The system Black version differs from repository pin, so its suggested unrelated formatting was not applied.

02:52 all32 stage-owned Python files pass repository pre-commit hooks (final_source_hooks.log). Frozen-source serial final correctness runner started in final/: tracked positive control, prefill/teacher accuracy, exact-ID AIME, six shared qualitative cases and degenerate check, B32, Watcher all40 replay, all40 serving commands and sampler acceptance. Performance/profile/context remain separate. Persistent sampler v3 fails a raw eager BF16 gather before any sampler/capture: each rank writes only its own32 columns;96 remote columns remain poison. Diagnostic repro and native source investigation are retained; unsafe experimental buffers are not selected. Current production sampler remains correct split sampling with enabled shared decoder persistent scratch.

03:02 final-source all40 AIME accuracy passes both installed readiness checks with tracking1/skip0: final/prefill_accuracy.json and final/teacher_accuracy.json each top1=98%,top5=100%,top100=100% over100 predictions. Complete source hashes and regime are in companion provenance and final/source_provenance.json. Positive control rejected the deliberately live corruptible allocation before replay. Diagnostic teacher timing is not substituted for the clean token-out or separate clean teacher measurement. Exact161-ID free-running AIME is next.

03:06 exact-ID AIME free-running execution passes: all40,100 greedy tokens,161 saved chat IDs. Fresh paired outputs and verdict are in final/aime_autoregressive/. TT begins a structured walk/coffee-time setup; HF begins direct algebra. Both are coherent English and truncated mid-reasoning at100 tokens; outputs differ from token1. This is a qualitative smoke, not exact greedy equivalence or a completed AIME-answer score. Six-case shared suite is running next.

03:11 shared qualitative suite and installed degeneracy check pass (six canonical prompts,128 greedy tokens, all40, default thinking, presence penalty0). Fresh manual paired-text review is final/qualitative/verdict.json. All six new TT completions exactly match prior completed full-model outputs. Each pair is coherent and relevant but capped during reasoning; no completed-task or statistical quality equivalence claimed. q03 doubled thinking appears in HF too, preserving the existing limitation. B32 fixed-slot check is running.

03:16 final B32 fixed-slot capacity/state probe passes all40 with512MiB trace reservation (final/b32.json). Uniform4-token prompts, reordered continuation, inactive slots1–31 unchanged, final positions slot0=7/others6, programcache8541 withzero growth and no recapture. Detailed state snapshots in this harness are from mesh chip0; all-rank custom-kernel/component checks are separate evidence. Two resident continuations havezero token/position/page refresh; one requested activity update and caller token readbacks are explicit, not claimed host-free delivery. Full all40 B3 mixed/tail/page/sampling Watcher replay follows. Installed stage07 check exits0 (final/logs/stage07_check.log and exit_code).

03:34 final measurements are prepared in run_final_measurements.sh (syntax and repository hooks pass). The queue requires completed final correctness, checks mesh ownership between bounded jobs, stops on the explicit all40 useful-byte floor, and leaves separate queued layer, clean sampler, final performance, short/tail TTFT, reduced native profile and32769-context artifacts. No production source changes during final gates. Queued representative-layer controls use real weights and seeded128-token activations at resident position128; they are an engineering stack budget, not a mathematical bound or natural autoregressive quality test.

03:37 final Watcher10 all40 replay passes in1269s (final/replay_all40.json, preserved final/watcher_replay/). All17 cases cover11 fresh/resumed length shapes, mixed-versus-solo equivalence with changed/unchanged unused pages, inactive middle row, and three per-row sampled variants. Programcache9632 remains stable at every checkpoint; captures stay2 and both traces release. Sampling parameter/seed/output isolation passes, including a changed-slot output difference. Diagnostic cases intentionally read tokens/state; resident replay counters are separately zero for input/position/page refresh and readback. Serving command protocol is next.

Evidence retention: stage-local .gitignore preserves large raw state-comparison JSON, Watcher/native captures, and byte-sensitive generated text/metadata outside the local checkpoint. Raw files remain in this persistent workspace and will be linked from one evidence_manifest.json; concise result/verdict/search/perf tables are checkpointed. No raw output normalization, per-artifact archive or new checksum scheme is used. Installed evaluator JSON also omits a trailing newline and remains byte-preserved.

03:42 final serving_commands_all40 failed a test-only reduced-scope assertion after prefill, unsorted admission, resident async ownership and simultaneous dormant remap passed. It expected3 page-table uploads, but all40 has10 full-attention layers×3 changed slots=30 (observed30). Generator._bind_pages correctly iterates full layers and slots; the all40 opt-in harness retained reduced-model constant3. Replaced it with the actual cache-group count and added per-layer/slot host-table equality; hooks pass. Production is unchanged, so only affected serving gate is rerun; original failed JSON/log/execution_summary are preserved. The aborted queue never reached sampler acceptance. Clean device close and idle fuser observed; bounded reset/health/mesh smoke follow the failed multi-chip-run recovery rule before the rerun.

03:44 serving-assert recovery: idle fuser returned1; tt-smi -ls --local, tt-smi -r, and post-reset list each exited0, allfour Blackhole devices visible. One bounded1×4 FABRIC_1D mesh plus all-rank add/readback smoke exited0 (serving_assert_mesh_smoke.log). No process killed, no lock cleanup, no hardware-failure claim. Fresh affected test is final/serving_commands_all40_v2.json. measurement_source_provenance.json records exactly one changed source since final queue: test-only expected-table-count correction; all inference sources match prior final accuracy/Watcher/B32/qualitative gates.

03:49 serving v2 passes changed30/unchanged0 page-table assertions, then fails TypeError because the added verification loop named its item state and shadowed the existing state() snapshot function. Renamed that new local to cache_slot; hooks pass. This is a parent-introduced test-only error, not model behavior. Original v2 result/log remain. V3 will rerun the full affected command test after serialized recovery; inference sources and all earlier unaffected gates remain unchanged.

03:52 v3 preflight: second idle fuser returned1; serialized tt-smi list/reset/list and all-rank mesh add smoke all exited0, with allfour devices visible. Logs use serving_shadow_* names. No process killed or hardware fault inferred. Bounded serving v3 then sampler-acceptance sequence is running; it checks idle ownership between jobs and stops on any nonzero exit. The full-model measurement script uses correctness_acceptance.json and the updated source freeze; historical execution_summary.json remains failed, never overwritten.

03:56 serving v3 and sampler acceptance both exit0. Serving all40 passes9 command checkpoints, with expected page updates30 then0 across10 full-attention layers, dormant remap state equality across4chips for the selected recurrent layer, resident async ownership, RNG preservation/reset and explicit host-logits-without-sampling. Sampler acceptance passes9 fixture×strategy rows. correctness_acceptance.json reconciles8 successful original jobs with the affected serving rerun and final sampler; prior failed execution_summary stays unchanged. Final measurements now run serially from run_final_measurements.sh, with source freeze before/after each job.

03:57 queued layer controls pass96 all-rank output/state comparisons per kind and stable program counts. Real weights, seeded128 prefill, resident fixed-position128 decode, BF16 L1 width-sharded retained carrier, context1024,5×200 nonblocking loops: linear median0.586715590ms, full0.306684235ms; weighted30/10 stack20.668310051ms. This is an engineering lower-bound estimate, not a strict bound or natural-token decomposition (final/layer_stack_budget.json). Final clean sampler timing passes all9 cases: real-terminal split0.487649ms, power2split0.521800ms, argmax2.669194ms; selected split remains fastest correct measured greedy strategy. Final all40 clean benchmark is running.

04:02 final clean all40 performance passes execution and useful-byte floor: TTFT2729.973400012ms; queued21.416868244ms/46.692167529t/s/u; observed21.461292819ms; fixedmodel20.934684803ms; actual99HFteacher20.933991111ms. Achieved10.1001% useful-byte floor; queued latency14.61% lower than before. Programcache8440 stable;127 queued model/sampling pairs, no per-token refresh/readback/capture. Small TTFT improvement2.07% is within observed candidate variation, not a robust speedup claim. Candidate2 headline reproduces within0.04%. Short/tail same-context TTFT follows.

### 04:09–04:20 UTC — boundary acceptance and evidence consolidation

Final context4096 boundary counterpart passed logical64/255/256/257/1025/2049.
TTFT ms:1473.618185,5957.422808,5534.284149,6106.160434,22067.904267,
43703.507399. Each same-length warmed run includes first sampled token and
then checks two resident feedback steps, exact final position, stable cache9590
and stable trace IDs. These small differences versus baseline do not establish
a robust prefill speedup; largest observed boundary regression is about3.43%.
The reduced0/3 native capture is still progressing through profiler drains;
context32769/65536 follows it in the serialized measurement driver. No second
process opens the mesh. Runtime/checklist documents now point to completed
all40 replay, serving v3, correct sampler and clean performance artifacts.

### 04:58 UTC — validated native full-path attribution

The unchanged serialized collector finished exit0 after2942s. Capture, native
conversion, three CSV/text tt-perf-report pairs and combined summary all pass.
`final/profile_reduced/component_summary.json` validates29440native executions,
16replays/component/rank with complete signatures on all4ranks. It includes
real layers0/3, embedding, terminal norm/LM head and canonical split sampling.
The native runtime head is BFP8/HiFi2/block4; detailed kernel/advice audits are
being finalized on CPU. No all40 profiler run was substituted for clean timing.

Same-run reduced token-out diagnostic: host fenced2.015464375ms; on rank2,
model operation-work sum1.266784250ms and sampler0.436510563ms. Sampler share of
that rank's operation work is25.6274%, below model work even in the reduced
stack. The0.312169563ms host-minus-work difference includes dispatch, operation
gaps/fencing and instrumentation; this is not a measured critical-path gap or
an all40 latency decomposition. Full40 headline remains21.416868244ms queued.
Raw capture,2.35GB ops CSV and full native report tables stay in place. Compact
per-rank op-group CSVs (all under114KB) and88.9KB component summary are selected
for local checkpoint evidence; oversized full tables and automatic PNGs remain
workspace artifacts linked by the single manifest.

The queued32769-token/context65536 capacity smoke started immediately after
profile acceptance, with no overlapping device owner. Native and prefill audits
run on CPU while this non-headline capacity check completes.

## 05:08 UTC — native and prefill audits reconciled

CPU audits completed after the native collector exited0. `final_profile_audit.md`
and `final_profile_attribution.csv` preserve actual operand/fidelity/kernel/CCL
rows and every advice family. No force-argmax, generic vocabulary TopK or full
vocabulary gather executes. The cached embedding AG metadata height is explicitly
qualified: cached64 differs from captured producer padded32; no dynamic payload
bytes are inferred from that reused height. Isolated FP32-router L1/subblock
geometry remains untried, not falsely rejected by sparse-GU controls.

`prefill_performance_analysis.md`, `before_after_prefill.csv` and
`prefill_role_chunk_summary.csv` distinguish logical lengths, physical chunks,
setup and timed work. Logical128 uses one full-attention SDPA chunk; linear
prefill uses two64 chunks. Generic prefill TopK is expert routing, not vocabulary
sampling. Metadata does not resolve every inline custom prefill program, so
shape/source inferences are labeled. Boundary timing differences are single
samples and do not establish a robust speedup or causal regression.

The long context job still owns the mesh; no concurrent device work was started.

## 05:15 UTC — long-context and serialized final queue complete

`final/prefill_context32769.json` passes all40 at context65536, logical32769.
One long prefill after physical-variant preparation takes691479.281598ms through
first-token delivery; this is a capacity smoke, not a repeated warmed benchmark
or quality claim. Two resident model/sampling replay pairs advance to32771 with
zero refresh/readback/recapture/release and stable9590 programs/traces0,1.
Prepared and final DRAM allocation are21,024,343,552bytes/device, free
12,954,371,584; L1 allocation591,936bytes/bank. Memory endpoints match, not a
sampled runtime peak. `context_memory_audit.md/.json` reconciles extra head
storage and historical maximum context. `doc/context_contract.json` appends
stage7 validation while preserving every earlier capability/history/limitation.

The serial driver exits0 after FINAL_MEASUREMENTS_DONE. Its before/after-job
checks preserve all309 frozen implementation/test hashes. Installed stage07
check exit0 is retained in final/logs. Native and all correctness/performance
hardware gates are complete; no hardware process remains from this queue.

## Final pre-review source and artifact checks

`pre-commit run --files <explicit stage-owned paths>` passes all applicable hooks
for260 selected source/docs/structured-result/compact-CSV files; full output is
`logs/final_all_selected_hooks.log`. The309-file measurement source freeze
matches both before and after hooks. The prior documentation-only165-file hook
run also passed. Local Markdown targets resolve (157 checked before the final
context links; final link check follows). Complete raw output text and large
native captures are deliberately excluded from normalizing hooks and remain
at manifest-listed persistent paths. All12 selected compact CSVs are under500KiB
and have no trailing whitespace; no raw report was altered to evade the hook.

The owner read all six saved HF/TT shared-suite pairs and the AIME pair again.
They support the bounded verdicts: relevant reasoning at the cap, differing
continuations, no completed-answer score; q03 repeats 'thinking' on both HF/TT.
The evidence does not establish broad task accuracy or exact HF generation.

## 05:28 UTC — final native accounting and router-advice closure

`native_accounting_derivation.md/.json` verifies all16 native replay signatures
per rank, reconstructs source-backed operation ranges (entry1–9, linear10–114,
full115–189, terminal190–199; sampler1–31), and reconciles every grouped total.
Maximum same-rank all40 extrapolated operation work is20.631607125ms (rank0),
versus clean21.416868244ms; difference0.785261119ms/3.666554% of wall. This is
not measured host overhead or a critical-path span. Same-run reduced useful-byte
ideal/native-work/fenced-wall is0.3500175625/1.7032948125/2.0154643753ms,
using179,208,992useful bytes/rank at512GB/s. No padding/rereads inflate bytes.

Remaining FP32-router layout/subblock advice received isolated measured controls
in `router_advice_comparison.md/.csv`. Real layer3 weights and explicitly
synthetic inputs; unchanged dtype/fidelity. All three candidates pass192 exact
checks in both clean and separate tracking1/skip0/Watcher10 runs. L1 copy and
L1+subblock2 lose; DRAM/subblock2 shows no clear gain. Keep current production.
Initial tracker rejection was a probe-only live Python x alias after capture;
release out,x fixes it without exemptions. Preserve original failure; six v2
runs supersede initial timings. Each device job was serialized, bounded180/240s,
wait_for exited0 for v2; every final Watcher log is saved. Source/tests frozen
for the complete-model gates remain unchanged. No cumulative rerun is needed
for an unselected probe with no production changes.

Final updated44-file hook pass is in logs/final_pre_review_hooks.log. All309
frozen source hashes still match;169local Markdown targets exist. The final
context-contract comparison confirms all pre-existing fields are byte-value
unchanged after removing the appended optimized_full_model_validation field.

## Stop Rule3 — explicit stage-cap wrap-up (2026-11-16T05:34:07.267199+00:00)

The user states the stage time/token cap is reached and instructs retaining the
best version, running only remaining required gates once, recording handoff
notes and completing. Optimization stops here. All hardware accuracy, runtime,
context, performance and native-profile gates are complete and pass; remaining
work is independent stage review, any required finding repair, and local commits.
The last two kept gains11.1973% and3.8090% do not satisfy branch2. No branch1
exhaustiveness claim is needed. Optional router controls did not change the
selected implementation, and all309 frozen source hashes still match.

### Not tried (stop rule)

Expected gain is unknown for each item below; none is declared slower or
impossible. Preserve selected numerical/residual policies when investigating.

- Embedding replication or alternative batched entry topology.
- LM-head row-parallel decomposition, norm/head fusion and wider output-core,
  subblock or split cross-products at the selected BFP8/HiFi2 policy.
- Wider FP32-router geometry/producer topology beyond the completed explicit
  L1-copy, four-core N2/subblock2 and combined controls.
- Sparse expert block64/grid/subblock cross-products and broader B32 grouped
  expert combine/per-role geometry.
- Shared gate/up packed-versus-separate producer/consumer comparison.
- Whole-stack public batched residual interface and additional prefill2D/chunk
  geometry; individual unresolved inline prefill roles could use finer native
  metadata in a future capture.
- Explicit sampler persistent-output broadcast: narrow native one-tile packet
  repair or width64 padding/compaction. The current candidate has an exact
  partial-packet failure repro; these adaptations remain untried, not enabled.

A broad weight/activation/KV/CCL/fidelity Pareto frontier belongs to the later
`datatype-sweep` stage and was deliberately not performed here. The inherited
rejection ledger and advertised context remain binding. No vLLM work or push.

## Independent stage review and closure

Fresh xhigh stage-review round1 returned **clean-pass**, with no Required Work
([report](stage_review_round1.md)). The reviewer independently reconciled source
hashes, accuracy, qualitative pairs, native accounting, context, runtime counters
and retained policies. Non-blocking limitations remain disclosed in the report.
No additional hardware gate or optimization experiment was run after the cap.
The selected implementation and evidence are ready for the local checkpoint;
its SHA will be recorded below. Nothing is pushed.

Checkpoint packaging: the first normal commit hook formatted generated scan
probe C++ snapshots. Restored their exact pre-hook bytes from the index and
retained them as raw workspace evidence, listed in evidence_manifest.json.
They are excluded from the checkpoint like other raw captures; production
source and measured artifacts are unchanged. CSV CRLF line endings are
preserved; the CRLF-aware whitespace check passes. No hooks are bypassed.

Stage complete: **clean-pass**, optimize Stop Rule3 (explicit user cap).
Implementation/evidence checkpoint: `08977a6f21c68e383ead5f4675f980eb45cfbed8` on `main` in `tt-metal`.
Normal repository commit hooks pass (`logs/local_commit_retry.log`); all309
frozen source hashes still match after the commit. No device run was repeated.
This documentation-only closure commit records the implementation SHA; its own
SHA is supplied by `git log -1` and the final response. No changes were pushed.
