# Full model stage 6 work log

Started 2026-11-14 from local checkpoint 121ab5df. Worktree clean at start.
Target Qwen/Qwen3.6-35B-A3B revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7;
Blackhole p300c chips 0–3, 1x4 mesh. No vLLM adapter or server work authorized.
Enabled installed skills: tt-model-bringup 0.1.24 and tt-autodebug 0.1.8.
Shell environment: source /home/ttuser/qwen36-35b-a3b/lane-env.sh.

Initial fuser showed no device holders. `timeout 60 tt-smi -ls --local` exited 0,
listing all four devices. No reset performed.

Preserve MultichipDecoder defaults, including replicated BF16 residual interface,
BF16 KV/CCL, BFP8 ordinary/expert weights, HiFi2 ordinary/shared projections and
LoFi active experts; shared CollectiveWorkspace. Stage 5 owner-accepted linear
latency limitation remains a baseline fact, not a full-model acceptance waiver.
Lower bound from stage 5: 30*0.719504474 + 10*0.340836807 = 24.99350229 ms/token.

Implementation and acceptance are in progress. No full-model accuracy, timing,
context capacity, trace safety, or serving support pass is claimed yet.

## Initial wrapper probe

`timeout 600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_probe --output .../reduced_probe.json`
uses real layers 0 and 3, actual embedding/norm/vocab head and a 256-token cache.
First run failed because embedding requires UINT32 indices, while attention positions
are INT32. Device-side typecast fixes only the lookup boundary. Repeat passed in
23 seconds: prefill [1,4,248320], decode [1,1,248320], finite logits and position
advanced 4→5. This is shape/runtime smoke, not HF accuracy or trace acceptance.
Artifacts: logs/reduced_probe.log (failed), logs/reduced_probe_uint.log, reduced_probe.json.
Per-device resident DRAM after two layers and terminal weights: 1,305,628,672 bytes.

HF AIME24 reference generated on CPU through the packaged runner with pinned
revision helper; 161 prompt tokens, 100 predictions, top-100. Exact cache hit
verified. See hf_reference/reference_summary.json and terminal_contract.md.

## Sampling capture repair and CPU work during bounded wait

Initial `--generate` probe stalled in finally/close. Preserved py-spy stack and
bounded tt-triage output before the original timeout; see AUTOTRIAGE.md. The
source-local missing grammar_on keyword is repaired, capture end is now in
finally, and exceptions print before device close. No early SIGTERM/reset.

Development source audit identified full-vocabulary prefill amplification,
late live logits, slot feedback mapping, external cache binding, warmup coverage,
and missing sampler request handling. Last-token prefill now selects hidden row
before LM head; temporary logits are released before replay. Shared physical KV
pools and shared immutable per-kind RoPE plans replace per-slot/per-layer copies.
Other findings remain work. See development_audit.md, not a stage-review verdict.

CPU shared-suite HF controls completed (six canonical prompts, 128 tokens each).
Artifacts and matched-seed/blind comparison tools are in qualitative/. No TT
qualitative verdict or model limitation waiver is inferred from these alone.

## Recovery after unfinished capture

Original bounded sampler probe ended with exit124 after600s. No manual process
termination. `fuser /dev/tenstorrent/*` then showed no holders. Ran serialized
`timeout 60 tt-smi -ls --local`, `timeout 180 tt-smi -r`, and
`timeout 60 tt-smi -ls --local`; sequence exited0 with all four chips listed.
No second reset or lock removal. A bounded90s 1x4 FABRIC_1D open/add/read/close
smoke exited0 and printed MESH_SMOKE_OK (logs/recovery_mesh_smoke.log).
List/reset/list output is in logs/recovery_{list_before,reset,list_after}.log.
This is infrastructure recovery, not a model performance result.

Retry command uses timeout180, same reduced layers and real terminal graph,
with grammar_on=False and paired captures. Exact command in
logs/reduced_generate_fixed.log via scripts/wait_for; result pending.

Trace allocation positive control: `logs/reduced_positive_rowmajor.log` exits 0 in 25s, with tracking=1 and SKIP_PROGRAM_CACHE=0. The deliberately live row-major buffer raises, then is freed; the normal 4-token reduced split trace succeeds. Earlier control allocated a tiled FP32 source and left two new program-cache buffers, correctly rejected too; the control now isolates a plain BF16 row-major allocation. Evidence: reduced_positive.json.

Full40 prefill gate passes (239s including load): `source lane-env.sh; timeout1200 python tests/run_full_model_readiness.py --mode prefill ...` via wait_for. Exact command/source in prefill_accuracy.json.provenance.json; log logs/prefill_accuracy.log; top1=.98,top5=1.0,top100=1.0 over100 AIMEtokens. All40load reaches L1 residency571456 bytes/core after immutableSiLU sharing and release of unused inherited routers. New reset/fixedslot work was not part of this run; finalacceptance must use finalsource.

2026-11-14 11:58 UTC: reduced fixed-slot decoder probe passed with three slots, inactive recurrent/conv/token/position/seed state preservation and reordered active slots. Artifact: reduced_batch3.json; command and source sidecar, logs/reduced_batch3.log. This predates bounded prefill integration and is not final acceptance.

Bounded prefill transport probe passed on all four chips in 6s, with tracking=1 and SKIP_PROGRAM_CACHE=0. Exact BF16 rows, unaligned counts, zero tails, clamped page windows, changed offsets and retained trace after eager operations: program count remained2. Command: `timeout300 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_prefill_transport_probe`, via wait_for; log logs/prefill_transport_probe.log; artifact prefill_transport_probe/result.json.

Bounded prefill integrated probe passed (288s cold compilation, 2011/3566 JIT cache hits). `reduced_bounded_prefill.json` tokens match original reduced trace exactly: [28433, 150929, 11927, 66275]. Log logs/reduced_bounded_prefill.log includes tracker positive control and successful subsequent sampled replay. Additional paired MoE-tail/arbitrary-length and all-layer checks remain.

Batch32 reduced probe passes after per-row attention DRAM drain ordering repair; see AUTOFIX.md and reduced_batch32_drain.json. Paired prefill equivalence reports10exact MoEtails,8exact complete/continuation cases,1singletonterminal-only discrepancy; cache/residuals all exact. Targeted repair underway, no acceptance claim.

2026-11-14 12:20 UTC: terminal normalization now explicitly consumes interleaved DRAM. Targeted singleton paired check passes bit-exact logits/residual/cache (`prefill_equivalence/singleton_fixed.json`, `logs/prefill_equivalence_singleton.log`, 29s). Remaining paired cases were already exact. This terminal boundary preserves the inter-layer decoder layout.

Serving-state toy device probe passes all four chips after converting TT Shape to tuple before slicing (`serving_state_probe/result.json`, `logs/serving_state_probe_shape.log`, 7s). Actual generator command integration remains a separate gate.

Traced common sampler comparison passes all nine fixture/strategy pairs in 12s. Selected unpadded split greedy0.4883ms versus padded0.5218ms and force-argmax2.6647ms on real terminal logits; all-negative/tie tests also pass. See sampler_comparison.md, sampler_contract_probe.json and logs/sampler_contract_shape.log. Command: tracking=1/SKIP_PROGRAM_CACHE=0 timeout600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_sampler_contract_probe --output .../sampler_contract_probe.json.

2026-11-14 12:38 UTC: mixed decode root cause localized and controlled: global grouped-MoE stochastic rounding couples target arithmetic to other expert counts. SR-off diagnostic makes fixed-shape slots1/2 bit-exact; separate slot0 path switching explains remaining control failure. Implemented opt-in deterministic rounding for batched decode only and one fixed trace per configured capacity; B1 and prefill strategies retained. See AUTOFIX.md. Fused C++/nanobind build and runtime installs passed. Post-install replay acceptance pending. New prefill-logit row staging toy probe PASS6s (`logs/serving_state_prefill_rows.log`). Cache construction/validation, slot release, reset metadata cleanup and prefill whole-batch validation integrated, with final device evidence pending.

2026-11-14 12:44 UTC: installed mixed repair passes targeted replay (`generator_replay_probe/mixed_fixed.json`, log `logs/generator_mixed_installed.log`,36s): fixed-slot solo/mixed exact, changed/unchanged pages, inactive middle,8489 program entries with zero growth. Actual serving command integration passes33s (`serving_commands_probe/result.json`, `logs/serving_commands_actual.log`): sampled prefill, unsorted slots, two async leases, simultaneous dormant remap, page-only changed/unchanged, parameter-only RNG preservation, independent reset, host-logit output, stable traces/program cache. CPU command suite passes. Final AutoDebug diagnosis copied to AUTODEBUG_MIXED.md; no child source edits.

2026-11-14 12:49 UTC: all40 teacher gate failed before inference after233s: packaged readiness standalone-module discovery exposed generator-local relative imports. Converted every relative import in generator.py to its absolute model package path; retained helper-module relative imports are safe because those are package imports. Reduced packaged-discovery probe added before full rerun. No accuracy result, device reset, policy or context change. Source audit also moved pending-output prefill rejection before the mutation/poison boundary and made legacy decode reject unknown low-level fields.

2026-11-14 12:57 UTC: packaged all40 teacher-forcing gate PASS392s including loading/preparation (`teacher_accuracy.json`, provenance sidecar, `logs/teacher_accuracy_discovery_fixed.log`). AIME100 predictions: top1=.98,top5=1,top100=1. Token-out teacher-forcing decode99steps=39.63t/s/u, TTFT3945.55ms for161prompt (accuracy workload, not headline128/128). Model and common sampling traces captured with allocation tracking1/SKIP_PROGRAM_CACHE0; clean closure. Subsequent source changes only add validation to legacy authoritative inputs and serving ownership checks; normal resident compute/token-feedback path unchanged. Reduced packaged factory smoke passed30s before this retry.

2026-11-14 12:58 UTC: external cache lifecycle probe PASS32s (`external_cache_probe/result.json`, `logs/external_cache_final.log`), real0/3 fixed3slots allfourchips. Confirms per-layer external allocation, empty-slot admission, malformedrequest/alias/pageownership rejection before mutation, pending-output rejection without poisoning, slot-only release isolation/idempotency, full reset/readmission exactness, stablecacheaddresses/programs/traces and exactly-once release. This is lifecycle evidence; full40 numerics are separate.

2026-11-14 13:05 UTC: all40 shared qualitative suite completed409s including load/preparation (`qualitative/tt_greedy/tt_greedy_outputs.json`, paired per-prompt readiness artifacts, `logs/qualitative_full.log`). Read all six HF/TT128-token completions: relevant reasoning, no mechanical loops or wrong-language drift, first divergence indices64/85/53/107/19/74. q05 first difference is formatting; q03 repeated opening phrase appears in HF too. Both sides use reasoning cap; no final-answer or statistical distribution quality claim. Descriptive verdict in qualitative/tt_qualitative_verdict.json. No qualitative failure chosen for debugging;10-seed blind control remains mandatory before any future failure diagnosis.

2026-11-14 13:12 UTC: all40 B3 teacher diagnostic PASS401s (`batched_teacher/result.json`, `logs/batched_teacher_final.log`): eachslot prefills1prediction(top1=0/1,top5/top100=1/1), then99forceddecodepredictions(top1=98/99,top5/top100=99/99); combined98/100,100/100,100/100 per slot. This matches the B1 combined accuracy and preserves the gate under deterministic batched rounding. Three disjoint physical page sets, positions260, program entries8491unchanged. Explicit logit readback and HF token/position uploads are diagnostic teacher forcing, not measured token-out or a runtime fallback. Common sampling integration is separately proven by fixed-slot replay/serving controls.

2026-11-14 13:18 UTC: full40 capacity32 capture exceeded the runner reservation: 467,361,792 model-trace bytes versus 200,000,000 reserved (`logs/full_batch32_final.log`). Loading, prefill preparation and eager fixed-batch warmup completed; mesh closed cleanly, no reset. Added mesh-owner recommended_trace_region_size: 200 MB solo, 512 MiB fixed batched. This is a configuration correction, with no batch/context reduction. Final retry uses full_batch32_trace_budget.json.

2026-11-14 13:29 UTC: all40 capacity32 PASS424s with512MiB trace reservation (`full_batch32_trace_budget.json`, source/command sidecar, `logs/full_batch32_trace_budget.log`). All32 active decode outputs,31 inactive slots bit-exact recurrent/conv/token/seed/position state, reversed returned rows, unchanged program cache/traces and zero resident input/page/position refreshes. State readback checks chip0; computation spans all4 chips. The prior reservation failure is resolved without capability reduction.

2026-11-14 13:35 UTC: final-source all40 packaged prefill check PASS381s (`prefill_accuracy_final.json`, provenance sidecar, `logs/prefill_accuracy_final.log`). AIME100: top1=.98,top5=1,top100=1. This supersedes the earlier pre-bounded-prefill accuracy artifact. All40 resumed-prefill checker launched next at context2048 with default thresholds and both negative controls.

Acceptance measurement boundary: final performance uses exactly128 input/128 output tokens (first prediction from prefill plus127 decode steps). Caller-observed token-out includes127 token readbacks without input feedback; queued throughput separately excludes intermediate delivery. The model-only fixed-token replay is not teacher forcing. The reduced profiler retains16 steps/component. Source formatted before measurement.

2026-11-14 13:49 UTC: all40 resumed-prefill gate FAIL793s (`readiness/prefill_resume_check.failed.json`, `logs/resumed_prefill_final.log`). PCC failures:257/start128 min.8586;1025/start260 min.9687;1025/start1024 mean.9870. All top1 checks pass; exact case details and negative controls in artifact. Device closed cleanly. Stage remains incomplete; AutoFix boundary investigation starts before performance/maxcontext.

2026-11-14 13:50 UTC: explicit host_sampling=True packaged-factory compatibility probe PASS30s (`host_sampling_probe.json`, `logs/host_sampling_final.log`). Real0/3 reduced model emits expected4greedy tokens[28433,150929,11927,66275]. This mode intentionally reads logits/samples on host, is explicitly selected and is excluded from device-token throughput. No inference policy fallback.

2026-11-14 13:55 UTC: CPU-only original-HF control completed in25.45s (exit0) before resumed-prefill failure diagnosis. Command after python_env/lane-env setup: `timeout 1800 python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_resume_control/run_control.py`; exact setup/redirection command is in `qualitative/hf_resume_control/README.md`, output in `run.log`. Pinned revision345d467b117d1fd5eda9b879a5988de9ae5f47a7, default BF16 HF policy,8CPUthreads, exact same natural-book tokens/sparse positions, actual prefix cache; no TTNN/device/generation/policy override. All three case-level numeric failures reproduce: L257/S128 HFmean/minPCC=.987027/.496343 vsTT=.993512/.858570; L1025/S260 HF=.991617/.952856 vsTT=.993122/.968665; L1025/S1024 HF=.988465/.988465 vsTT=.987021/.987021. All cases on both sides meet originaltop1>=.98 and finaltop1 equality. HFworst positions135/300/1024; HFtop1 mismatches172/304/none. Full per-position HF values are preserved in result.json; TT artifact stores only aggregate metrics, so exact row-overlap is unknown. Fixed numeric rubric scored opaque permuted records before unblinding. Seeds0–9 explicitly reuse one deterministic observation per case, not10independent samples; all8actual HF forwards consume no Torch RNG. This is a deterministic numerical control, not a sampled-content statistical waiver. Recorded `original_model:same` limitation in `doc/eval_readiness/limitations.json` with these boundaries. Raw `readiness/prefill_resume_check.failed.json` remains FAIL; unchanged thresholds and the user-required passing resume gate remain work. See `qualitative/hf_resume_control/verdict.json` and README for exact artifacts/limitations.

2026-11-14 14:01 UTC: all40 B1 performance PASS413s (`performance/result.json`, `logs/performance_final.log`): warmed128input/128output TTFT2890.065ms; caller-observed127decode steps25.1348ms/token=39.7855t/s/u; queued25.0821ms=39.8690t/s/u. Positions128→255,127model+127sampling,0token/position/page/activityrefreshes/recaptures. Observed delivery127reads; queued0reads. Fixed-token model-only24.6002ms (notteacherforcing); fixed-logit sampler.48793ms (~1.945% queued). Prior layerbound24.9935ms from differentworkload; no sampler-dominance finding. Program/trace identitiesstable,device_onlyguard/alloc trackingpass. Reducedprofilepending.

2026-11-14 14:04 UTC: AutoDebug retained-state audit completed exit0; canonical report `AUTODEBUG_RESUME.md`. Independent source investigations found no supported wrapper/cache/page/position defect; raw all40 resume FAIL remains unwaived. Focused actual-token legacy/bounded parity first hit diagnostic legacy-reference L1 pressure (`prefill_equivalence/resume_legacy_parity.json`, `logs/resume_legacy_parity.log`). A test-only option drains legacy attention outputs to DRAM immediately; production unchanged. Retry PASS47s (`prefill_equivalence/resume_legacy_parity_drain.json`, `logs/resume_legacy_parity_drain.log`): real layers0/3, all4 chips, starts/lengths128/129,260/765,1024/1; logits, layer outputs and retained caches bit-exact. Command: `timeout 1800 python -u -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_prefill_equivalence_probe --section full --tail-lengths "" --prompt-lengths "" --continuations 128:129,260:765,1024:1 --token-fixture models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_resume_control/prompt_tokens.json --drain-legacy-attention --require-exact --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/prefill_equivalence/resume_legacy_parity_drain.json`. This is diagnostic inherited-path parity, not an all40 resumed-prefill pass.

Reduced native profiling launched with `bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_full_model_components.sh models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/profile_reduced_v1` after idle-mesh verification. Layers0/3, real shapes/weights,16 steps/component; watcher/allocation tracker unset during profiler only. No full-stack profiling.

2026-11-14T14:13:09.561789+00:00: local shared-code commit `69a03c857a5366dd575dc9a7a655b7248c15fad8` — `Shared-code fix: make fused MoE stochastic rounding configurable`. Eight C++/binding files only; repository pre-commit checks passed, including clang-format. Existing build/device evidence is recorded above and in AUTOFIX.md. No push.

2026-11-14 14:16 UTC: repository pre-commit formatting/check pass applied black to6Python files and isort to25; autoflake changed nothing. Exact files/results in logs/precommit_check.log; second full check over36stage-owned Python files passed (logs/precommit_final.log). These are formatting/import-order edits; no intended operation graph, dtype, cache, sampling or control-flow changes. Existing acceptance artifacts retain their original byte hashes; do not claim byte equality to formatted files. Reduced profiler had already imported its preformat source. Maximum-context runner will execute formatted source.

2026-11-14 14:20 UTC: reduced profiler v1 failed: first preparation dump overflowed markers and host kernel OOM killed PID3826202 (188465852kB anonymous RSS). Kernel record preserved in performance/profile_reduced_v1/host_oom.log, with capture/postprocess logs and failure.json; no profiler throughput claim. Four-chip bounded arithmetic/open/close smoke PASS (`logs/mesh_smoke_after_profile.log`), no reset or process kill by agent. Profiling-only repair drains after each preparation attention/MoE component before capture and reduces program buffer capacity100000→5000; model/generator unchanged. Pre-commit, py_compile and bash-n passed. Fresh v2 capture launched by `bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_full_model_components.sh models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/profile_reduced_v2`.

AutoFix retained-state outcome recorded in AUTOFIX_RESUME.md: failed to restore the required passing contract; no supported policy-preserving source repair isolated. Exact reduced accepted-decoder parity and original-HF correspondence do not waive the all40 gate. Stage remains incomplete while independent remaining evidence is collected.

2026-11-14 14:26 UTC: profiler v2 also host-OOMed while preparing linear geometries (last progress rows32), despite5000-program buffer and per-component drains. Raw/local evidence and kernel OOM record retained under performance/profile_reduced_v2; no profiler conclusion. Second bounded four-chip arithmetic smoke PASS (`logs/mesh_smoke_after_profile_v2.log`), no reset. CPU-only profiler source investigation started before any further retry.

Single all40 maximum-context acceptance attempt launched after idle-mesh/health verification: `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 14400 python -u -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_context_acceptance > models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/context_acceptance.log 2>&1`, bounded wait_for14430s. Configured context262144, logical prefill262143 followed by one decode at262143. This independent capacity acceptance does not waive the still-failed short resumed-prefill gate. No profiler/watcher, no context reduction or sweep.

2026-11-14 14:30 UTC: maximum-context acceptance failed during short2-token warmup, before long-prefill/decode (both counters0), because SDPA staticCB end920640 overlaps lowestL1 allocation898496. Not a DRAM/capacity limit. Preserved `context_acceptance/result.json`, `logs/context_acceptance.log`. Source investigation `AUTODEBUG_CONTEXT.md` rejects coordinate-only norm relocation under the lockstep allocator and identifies dead K/V heads retained across SDPA.

2026-11-14 14:39 UTC: lifetime-only repair drops final singletonK/V and surviving loopalias after ordered cache updates, before SDPA. No op/dtype/layout/geometry/CCL changes. Reduced0,3 full8192-entrypage-table reproduction with residentL1=571456 fails at912064<920640 (`logs/context_l1_repro_before.log`,31s); stronger repaired residentL1=585024 passes (ballast488704B,33s), trackerpositivecontrolrejectsunsafeallocation,2traces and3model/4samplerreplays emit expectedtokens[28433,150929,11927,66275]. Artifact `context_acceptance/l1_repair_tracked.json`, provenance and `logs/context_l1_repair_tracked.log`; TRACKING=1/SKIP_PROGRAM_CACHE=0. An earlier repaired launch omitted tracker env and failed its positivecontrol after preparation; retained `logs/context_l1_repair.log`, no trackerpassclaimed. The all40 original run had13568B more temporary pressure than the first reducedrepro; strengthened ballast accountsfor that difference.

2026-11-14 14:40 UTC: profiler-only shared repair honors existing disable-push flag in real-timeTracy handler; collector uses MID_RUN_DUMP=1 to flushhostmarkers and rejects instrumentation failures/partialreports. `cmake --build build_Release --target tt_metal --parallel2` and `cmake --install build_Release --component tar` pass (`logs/build_profiler_gate.log`, `logs/install_profiler_gate.log`). Hardwareprofileproof pending. All40 maxcontext retry launched only after shortpressureprobe passed; fresh artifact `context_acceptance/final.json`, log `logs/context_acceptance_lifetime_fixed.log`; timeout14400 viawait_for14500. Original longacceptance counters were0, so this is the first attemptedlongprefill.

2026-11-14 15:27 UTC: maximum-context execution continues; read-only host snapshots show progression through layer15, with stable roughly2.2GiB host RSS. No device interruption, reset or parallel hardware work. `context_acceptance/progress_inspection.md` documents the process-specific address mapping; these snapshots establish host progress only, not acceptance or device completion. Prepared allocator census is25372641280 allocated plus8606073856 free=33978715136 bytes per chip. The supplementary tensor visitor skips callable decoder/helper objects, so its inventory is partial; `context_acceptance/memory_audit.md` records the limitation and full constructor/KV/prepared accounting. Preserve raw inventory and use authoritative allocator census in the final context contract. `contract_update_plan.json` is pending completed acceptance, not an applied change.

CPU-only profiler validation now checks native execution coverage, four-rank/session completeness, and actual cached tensor/kernel metadata. Eleven tests pass;17 real stage5 native programs, including allfour MoE ranks, collectives and movement-only kernels, pass unchanged in0.939s (`performance/native_metadata_compatibility.json`, `profile_pipeline_audit.md`). A profiler-only preparation log now records PID, currentRSS and peakRSS after each existing drain; no added device operations or measured-loop changes. Final capture remains pending.

The unprofiled performance harness now includes `teacher_forcing_logits_only`:99 distinct immutable HF inputs through the existing model trace, with forced host uploads inside timing and sampling/logit readback excluded. It checks99modelreplays/99inputrefreshes, no sampling or otherrefresh/recapture, positions161→260, stabletrace/programidentities and finalpersistentinput. CPU syntax/precommit and an isolated99-step fake contract pass; actual hardware measurement remains pending. This supplements rather than relabels the earlier sampling-inclusive packaged teacher-forcing measurement.

Full-model boundary review applied the multichip skill and local `tech_reports/LLMs/llms.md` section3.3 to the added embedding/norm/head/sampler. `preserved_policy.md` now records exact local logical/padded shapes, source-derived communication payload formulas, precision/configuration, and measured versus unmeasured alternatives without reopening inherited decoder decisions. Stage5 watcher results cover the inherited decoder, but do not establish current wrapper coverage; a separate reduced full-model replay with `TT_METAL_WATCHER=10` and isolated logs is planned. `profile_components_commands.md` records the watcher, unprofiled teacher benchmark and fresh repaired profiler commands. No additional hardware work ran during maximum-context acceptance.

2026-11-14 16:23:55 UTC: single actual maximum-context acceptance PASS, wait_for exit0 after6277s, clean device closure and idle fuser check. `context_acceptance/final.json` records one all40 prefill of262143tokens (5883.448102s) followed by one traced decode at262143 (33.730192ms), nextposition262144. All10full-attentionlayers/all4chips verify31populated/onezero final-page rows before decode, finalrowwritten and precedingrowsunchanged afterward. All240linear-state chip checks and logits finite. Decode has1model+1samplingreplay,0token/position/page/activityrefreshes,0readbacks/recaptures. Allocator prepared/post-prefill/post-decode snapshots agree:25372641280allocated+8606073856free=33978715136B/chip;200MBtrace reservation reported separately. Both traces released, no cleanup errors, contextcontract hash unchanged during run. No context reduction or second long acceptance. Reporting-only visitor repair now traverses callable model/helper owners; the measured raw inventory remains preserved and explicitly partial, with no retroactive correction or rerun.

2026-11-14 16:24:53 UTC: reduced unprofiled performance harness PASS33s (`performance/reduced_teacher_smoke.json`, `logs/performance_reduced_teacher_smoke.log`). New99-step HF logits-only teacher measurement passes at1.563740ms/step with99modelreplays/99forceduploads,0sampling/otherrefresh/readback/recapture,positions161→260,stable8450programentries,finalinput393 matchingHFindex98. This two-layer smoke validates the harness, not all40 throughput or accuracy. Exact command: `env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout1800 python -u -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_performance --reduced --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/reduced_teacher_smoke.json`, throughwait_for1850. Fresh repaired profiler v3 launched only after clean closure; collector performs its own idle guard, with output`performance/profile_reduced_v3_midrun` and driverlog`logs/profile_reduced_v3_midrun_driver.log`.

2026-11-14 16:39 UTC: repaired profiler v3 continues through128 preparation drains (~812s), currentRSS7.75GB, without earlier source-location overflow/OOM. Raw data is large (>100GB) and conversion remains unverified; no performance/profile acceptance claimed yet. CPU review confirms later eager prefills skip universal geometry warmup, but their undrained intervals remain a profiling-only robustness opportunity. Production/trace graph unchanged. Documentation provenance audit now distinguishes historical parity hash matches from subsequent formatting/L1-lifetime/reporting edits (`source_provenance_audit.md`); raw artifacts remain unchanged. Context inventory runner pre-commit checks pass (`logs/precommit_context_inventory.log`).

2026-11-14 16:46:42 UTC: profiler v3 completed workload/replay windows but collector correctly FAILs1299s: device DRAM marker buffers overflowed after preparation/capture before PERF_TTFT. No host OOM/source-location overflow; raw capture and failure.json preserved under performance/profile_reduced_v3_midrun. No throughput/complete-profile acceptance from this capture. Profiling-only AutoFix extends drains over all eager prefill/setup intervals while retaining capture and timeddecode graph; CPU agent owns harness repair. Devices closed normally; next isolated Watcher replay is independent of profiler harness.

2026-11-14 16:53 UTC: profiler-boundary AutoFix implemented in test harness only (`AUTOFIX_PROFILE.md`): eager executor/model/sampling/precompile, seed handoff, warmreplays and before/aftercapture drains; capture query skips reads/synchronization while recording. ExitStack restores wrappers in finally. Unprofiled measurements and traced operation graph unchanged; profiler TTFT explicitly includes drain overhead. CPU fakecontract/syntax/precommit PASS; buffer5000unchanged, device validation pending. Watcher run remains active independently with no reported fault so far.

2026-11-14 17:02:16 UTC: current-source reduced generator Watcher PASS899s including cold compilation (1893/7204 JITcache hits). Artifact generator_replay_probe/lifetime_fixed_watcher.json, watcher/summary.json and logs/generator_replay_watcher.log. WATCHER10, TRACE_ALLOC_TRACKING1/SKIP_PROGRAM_CACHE0, profilerunset. Real0/3/M3 nonaligned/mixed/page/seed/replay assertions pass,90completed Watcherdumps withall4devices attached/observed/detached and no faultlines. Cleanclosure/idle fuser. No all40/maxcontext Watcher claim. Fresh profile v4 will use repaired eager/capture-boundary drains; no inference changes.

## Stage-cap wrap-up

2026-11-14T17:05:45.333882+00:00: user reports stage time/token cap reached. Freeze implementation at current best verified version. Finish already-running profiler v4 once; run remaining all40 prefill, teacher-forcing, performance and installed stage06 checks once, then one independent stage review. No further experimental sweeps/retries. Preserve failed gates; a blocking correctness gate cannot be relabeled pass-with-carried. Commit only stage-owned work locally; never push.

### Untried hand-off ideas

- Resumed-prefill: if work resumes, localize the original all40 length1025/start1024 case layer by layer, preserving actual uninterrupted/retained-state branches and accounting for the inherited BF16 recurrent boundary. No new intervention is justified yet; see AUTOFIX_RESUME.md. This remains blocking, not a carried pass.
- Profiler conversion: if v4 conversion exceeds its bounded limit or coverage checks fail, investigate a streaming native-window conversion retaining complete execution IDs and referenced metadata. Not implemented or validated; preserve all raw evidence.
- Full-model terminal performance: potential embedding/LM-head/collective alternatives recorded in preserved_policy.md remain unmeasured candidates, not claimed improvements or rejected options. Inherited decoder rejection ledger stays closed.
- Tensor inventory: the reporting-only callable-owner traversal fix has not been used to regenerate a full max-context inventory. Do not repeat the maximum-context acceptance for bookkeeping; authoritative allocator snapshots and explicit partial-inventory caveat remain the measured contract.

2026-11-14 17:18 UTC: profiler v4 workload PASS916.515s and clean device closure;0marker-drop/0source-location-overflow warnings, peakRSS8357664KiB. Capture-aware wrappers skipped1model/1sampler drain while recording. Native conversion/coverage/report still pending; no complete-profile PASS yet. Idlefuserverified before launching single remaining all40 prefill accuracy refresh. This accuracy run overlaps CPU-only profiler conversion; its timing is diagnostic only, not headline performance. Hardware jobs remain serialized.

2026-11-14 17:20 UTC: local shared-code commit 958dbadc5c4eb4b6b6ce73f792a0b862ba8f9e35 — Shared-code fix: honor disabled Tracy pushes in realtime profiler. One C++ handler file only,16insertions/1deletion; repository commit checks PASS (`logs/commit_profiler_fix.log`). Build/install and repaired v4 runtime evidence above validate the existing disable flag while retaining native reporting. No push.

2026-11-14 17:25:24 UTC: final-source all40 prefill accuracy PASS417s,98/100top1,100/100top5,100/100top100 (`prefill_accuracy_lifetime_fixed.json`, provenance, logs/prefill_accuracy_lifetime_fixed.log). Cleanclosure/idlefuser. Single final teacher-forcing gate launched next withsame TRACKING1/SKIP0;CPU-only profile conversion stillactive, so these readiness timings are not headline metrics.

2026-11-14 17:27 UTC: installed stage06 gate ran once under cap, exit2 (`logs/stage06_final_gate.log`, exit_code, logs/stage06_serving_problems.txt). DegeneracyPASS and context262144PASS. Serving checker reports86problems: required string evidence/declaration fields absent or structured instead ofstrings, plus4knownresume missingrows. Documentation-only schema reconciliation now maps existing observed evidence and declarationdetails into requiredstring fields, preserving all support claims/failures. No hardware or installedgate retry; final review must inspect corrected manifest, and rawfailedgate remains preserved.

2026-11-14 17:32:43 UTC: final-source all40 teacher-forcing accuracy PASS416s,98/100top1,100/100top5,100/100top100 (`teacher_accuracy_lifetime_fixed.json`, provenance, logs/teacher_accuracy_lifetime_fixed.log). Cleanclosure. Both finalsource accuracy gates passed exactlyonce. Packaged teacher path includescommon traced sampling and overlapsCPU-only reportconversion; headlineperformance remains pendingisolatedrun.

Cap wrap-up gate record: the single installed stage06 gate exited2 (`logs/stage06_final_gate.log`, `.exit_code`). Degeneracy and context subchecks pass. The manifest schema was corrected after that invocation:82 of86 findings were format issues (51 evidence strings,30 declaration strings,one kind), preserving prior status/evidence;four required resume/chunk rows remain missing. `gate_schema_reconciliation.json` retains the raw findings and corrections. No gate rerun and no overall pass claim. CPU JSON parsing and `git diff --check` pass.

2026-11-14T17:53:53.614707+00:00: profiler v4 complete PASS, wait_for exit0 after3040s. Native conversion, six tt-perf-report exports and final coverage/metadata validation complete. `performance/profile_reduced_v4_drains/component_summary.json` records32896 native executions and16 replay sessions/component/rank. Same-rank reduced0/3 token-out kernel sums:model1.471036125ms,sampler0.4548093125ms (23.6161%,not dominant); diagnostic attribution, not all40 or headlinewalltime. `native_program_metadata.json` retains actual operands/kernels/fidelity. Raw captures stay locally ignored. Final isolated all40 performance started aftercollector exit and idlefuser, TRACKING1/SKIP0, output`performance/lifetime_fixed.json`,log`logs/performance_lifetime_fixed.log`,timeout3600/wait_for3650; no further retry.

2026-11-14 18:00:56 UTC: final isolated all40 performance PASS440s,clean device closure and idlefuser. `performance/lifetime_fixed.json`, `logs/performance_lifetime_fixed.log`; all9recordedsourcehashes match. Warmed128input/128output TTFT2919.990325ms; caller-observed127decode25.128267976ms=39.795818834t/s/u, queued25.081887921ms=39.869407085t/s/u. Both127model+127samplingreplays,0token/position/page/activityrefreshes or captures; positions128→255; observed127tokenreadbacks,queued0. Device-resident feedback, nohostargmax/full-logitreadback. True HF logits-onlyteacher99steps at24.600471626ms=40.649627178t/s/u with99forcedinputuploads,0sampling/readback/otherrefresh/capture,positions161→260,finalpersistentinput393. Fixedtokenmodel-only24.600408906ms is a separate diagnostic. Fixedlogitsampler0.487787433ms (~1.9448%queued; separate controls are not additive attribution).8452programentriesstable,TRACKING1/SKIP0,profiler/Watcherunset,2tracesreleased. No further hardware runs or gate repeats. All remaining capped executions finished; independentreview and localcheckpoint next.

Exact final performance command after sourcing lane-env.sh: `"$TT_MODEL_BRINGUP_ROOT/scripts/wait_for" --cmd 'env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 3600 python -u -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_performance --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/lifetime_fixed.json' --file models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/performance_lifetime_fixed.log --timeout 3650 --tail 8`.

Checkpoint packaging: repository hooks limit files to500KiB and normalize whitespace. Preserve raw evidence bytes instead: `checkpoint_evidence/manifest.json` maps32 individually gzipped reports/logs to original paths, sizes and SHA256; gzip round-trips match every original. Originals remain locally ignored. No inference source, measurement or gate changed. Git whitespace checks are applied to committed text; native report whitespace is preserved inside gzip.
The archive also preserves generated completion/metadata files lacking a final newline; no newline was added to original HF/TT output. Restore via `checkpoint_evidence/README.md` when inspecting a fresh checkout.

Checkpoint self-check: all applicable repository pre-commit hooks PASS without edits (`logs/precommit_stage_checkpoint.log`); staged `git diff --check` PASS after exact-byte evidence packaging. All9 final-performance source hashes remain unchanged. No hardware/evaluation rerun. Independent stage review is in progress against settled runtime evidence.

## Final independent review and capped disposition

Fresh independent xhigh round1 reviewed the settled artifacts and actual six HF/TT output pairs after all executions finished. `stage_review.md` returns **more-work-needed**, with one combinedP1: required resumed-prefill acceptance still fails and four serving rows remain missing. Exact finding, evidence, scopes and next step are preserved verbatim in that report. No new correctness problem or runtime requirement was selected. The existing `AUTOFIX_RESUME.md` failed attempt remains the stop basis; latest user cap prohibits further experiments/retries. Do not relabel the failed gate or this verdict as clean-pass/pass-with-carried. This is an **incomplete local checkpoint**, not stage completion or authorization to begin vLLM.

## Carried to next stage

These are nonblocking hand-off notes, not a passing-stage verdict or permission to advance past the required continuation gate:

- Preserve inherited decoder rejection decisions and the owner-approved stage5 linear performance limitation. Terminal distribution/performance candidates remain untried, not rejected.
- Historical source comments in serving helpers still mention pending validation. Later actual reduced probes and final audit updates give their scope; tidy comments only when work resumes, without invalidating the frozen measured source.
- Keep qualitative claims descriptive: six128-token paired outputs show relevant reasoning but do not establish final-answer task accuracy or distributional quality. Preserve the controlled HF numerical sensitivity and exact limitations.
- Native report category labels for nine operation names can be improved later; complete operands/kernel metadata are retained. Do not rerun profiling solely for labels.
- A future detailed tensor inventory may use the corrected visitor. Do not repeat262K acceptance for bookkeeping; prepared allocator totals remain authoritative.

The blocking all40 continuation localization idea remains separately documented in `Untried hand-off ideas`; it is not a carried-pass item. No further hardware runs, installed-gate reruns, model changes or push occurred during closure.

## Local checkpoint SHAs

Repo `/home/ttuser/qwen36-35b-a3b/tt-metal`, branch `main`; no push.

- `69a03c857a5366dd575dc9a7a655b7248c15fad8`: shared MoE stochastic-rounding control.
- `958dbadc5c4eb4b6b6ce73f792a0b862ba8f9e35`: shared profiler disable-push repair.
- `b0cbe4ab9cd932bd1d2bb43d030c21cba9799d25`: `Checkpoint Qwen3.6 full model with validation and open resume gate`. Contains stage-owned model/generator/helpers/tests, capacity/accuracy/performance/qualitative/serving evidence, independent review and lossless compact evidence archives. All applicable commit hooks PASS (`logs/commit_stage_checkpoint.log`); working tree clean immediately after commit.

This following documentation-only commit records the checkpoint identity, closes the bookkeeping item and preserves **incomplete / more-work-needed** acceptance. All planned capped executions and the review have finished; no new model/generator changes or runtime runs. Untried ideas and nonblocking hand-offs remain above. The required resume gate was not waived, the installed gate was not rerun, and no next-stage/vLLM implementation was started.

## Owner decision and stage6 acceptance follow-up (2026-11-16)

Owner message from `/home/ttuser/qwen36-35b-a3b/multigoal_logs/06-06-full-model.resume-1.message.txt` is copied verbatim in `owner_decision.md`. The owner accepts resumed-prefill PCC failure as a known HF-reproduced limitation, directs completion of stage6, and requests no further work on this gate. This supersedes the prior blocked acceptance disposition only. Raw failures, unchanged thresholds, AutoFix's unsuccessful repair conclusion and first review remain preserved.

`doc/eval_readiness/limitations.json` now records owner acceptance with `original_model: same`, the exact decision and HF/TT evidence. `owner_acceptance.json` re-derives seven passing TT top-1/final-token checks, ten expected negative-control failures, unchanged program count, and the three HF-reproduced PCC-failing cases from existing artifacts. All top-1 criteria passing is not every-row equality: length257/start128 agrees64/65. No claim of exact HF/TT worst-row correspondence or ten independent sampled controls is added.

The four implemented continuation/chunk serving rows are now supported with this explicit limitation. `QwenServingCapabilities.serving_capabilities["supports_chunked_prefill"]` changes False→True, moves its pending entry to accepted evidence, and updates descriptive context/resume status. Automatic prefix caching stays False under recurrent_state. No arithmetic, layout, dtype, fidelity, KV, inter-layer residual, trace, sampling, reset or measured runtime path changes. No vLLM files, hardware tests, profiler runs or max-context repeats. Existing runtime evidence is retained; CPU-only stage/metadata checks and a fresh diff-only review follow.

The earlier untried full-depth PCC localization is no longer requested work under this decision. Nonblocking performance/category/inventory notes remain hand-offs; they do not reopen this accepted gate.

Owner-acceptance CPU validation PASS:7top1cases/10controls inspected directly; declaration import uses Python runpy without TTNN; repository pre-commit on `tt/serving_capabilities.py` passes (`logs/owner_acceptance_precommit.log`). Installed stage06 returns0 (`logs/owner_acceptance_stage06.log`, `.exit_code`): degeneracy, full262144context and all86servingrows pass. Command after sourcing lane-env.sh: `MODEL_DIR=models/autoports/qwen_qwen3_6_35b_a3b HF_MODEL=Qwen/Qwen3.6-35B-A3B timeout 300 bash "$TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/06-full-model.check.sh"`. This validates updated acceptance metadata only; no numerical gate or device workload repeated.

## Stage 6 completion under owner decision (2026-11-16)

Fresh independent xhigh diff-only round2 review returns **clean-pass**, with no Required Work (`stage_review_round2.md`). The reviewer independently re-derived all seven top-1 criteria, ten negative controls, unchanged raw HF/TT results, capability declarations and 56 supported/30 unsupported/zero missing serving rows. Prior P1 is closed by the explicit owner acceptance, not by numerical repair. Stage6 is **complete under the owner-revised contract**; raw PCC failures, thresholds and prior review remain intact. No further resumed-prefill work or device execution was performed.

Current README, checklist, serving contract and runtime acceptance metadata now point to this final disposition. Earlier snapshots are preserved as history. Review Other Concerns carry the existing unmeasured terminal performance alternatives and historical-comment cleanup as nonblocking hand-offs. The HF-control and six-pair qualitative limitations remain stated at their measured scope. This completion starts no vLLM work and authorizes no push.

Local owner-acceptance commit and subsequent SHA-log commit follow; the acceptance commit SHA is recorded below. Repository hooks and diff checks apply to the final text/declaration changes; the CPU stage06 PASS recorded above is not repeated.

Owner-acceptance checkpoint: `63c0b7d46b1d03eb49023d216f312f885b72eb16` — `Complete Qwen3.6 full-model stage with owner-accepted resume limitation`, repo `/home/ttuser/qwen36-35b-a3b/tt-metal`, branch `main`. All applicable commit hooks PASS (`logs/owner_acceptance_commit.log`). Contains the accepted limitation, declarations, clean-pass review, CPU stagecheck evidence and completion documents. No push. The following documentation-only SHA-log commit records this checkpoint and final metadata; it changes no runtime path or raw evidence.
