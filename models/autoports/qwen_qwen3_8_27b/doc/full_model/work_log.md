# Full-model work log

Stage 06 in progress. No vLLM changes.

Started from cdcf1091d55e793f68fd46acdbe80372337e1c34; clean worktree at startup. Enabled skills inventory provides tt-autodebug 0.1.8 and tt-model-bringup 0.1.21. Environment is the existing qwen38-functional-build container and tests/optimized_multichip_env.sh.

Hardware: bounded tt-smi -ls --local returned four Blackhole p300c ASICs. No reset performed. Full checkpoint source /data/mvasiljev/weights/Qwen3.8-27B, revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; copying to ignored .cache/qwen38-full for container access. Hard links were unavailable across mounts; ordinary copy selected.

Initial wrapper and reduced real-terminal probe are unvalidated. Sampling selection comparison is in sampling_contract.md. Required mixed-length/slot handling, generator-wide warmup, trace safety, correctness and performance remain work.

AutoDebug launched for finite prefill warmup / recurrent-tail contracts. Default child sandbox could not create namespaces; retried with the documented explicit skip under current unrestricted execution authorization. Child remains inspection-only. Reduced prefill probe command: `docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.probe_full_model --checkpoint .cache/qwen38-probe`; probe checkpoint uses the exact source norm tensor extracted without numeric conversion and completed source shard copies. See probe_prefill_01.log. Startup MPI shared-memory warning and unknown motherboard discovery warning require classification; no runtime acceptance claimed.

Reduced results: prefill02 passes finite [1,33,248320] logits after replacing unsupported row-major in-place multiply reset with zeros_like + copy. trace01 found sampler k/p/temp logical width32 mismatched B1; changed parameter buffers to exact logical batch. trace02 correctly found71 late prefill program buffers under tracking. trace03 prepares the explicit33-token signature and both full-logit/token-out terminal paths before capture; split model/sampler replay then passes with tracking/program-cache included. All three failures are wrapper/preparation bugs, not decoder precision evidence. Exact commands are the probe command above with --trace and log suffix01/02/03. Current general-length preparation remains incomplete; debug preparation is explicitly labeled.

`probe_trace_contract.py` passed with tracking enabled and program-cache allocations included (probe_contract_01.log and reduced_trace_contract.json). Two same-prompt requests retained both trace IDs and produced identical reduced outputs despite a changed page mapping. Position/rope inputs reached40 after8 tokens from33-token prompt; explicit scheduler override reached41, then device replay reached42. Token feedback retained its backing address. The one recorded token/position refresh is the deliberate scheduler override, not autoregressive feedback. General mixed-signature preparation and full-stack semantics remain unproven.

Fresh HF reference command: `docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.generate_hf_reference` (hf_reference_01.log). This loads exact BF16 text tensors into the installed HF Qwen3_5ForCausalLM; the source checkpoint is multimodal but this stage is text-only. Generation records100 greedy cached HF steps and top100 at each position.

First full-stack smoke: same environment, `python -m models.autoports.qwen_qwen3_8_27b.tests.run_full_smoke` (full_smoke_01.log). Tracking enabled; this is a functional smoke, not the4096/128 headline performance measurement.

Full checkpoint copy and fresh AIME HF reference completed. The first full-stack smoke loaded all64layers with native262144 context and produced a coherent128-token continuation; full_smoke.json records8,942,155,264 allocated DRAM bytes/device before trace setup. Its203-token prompt and earlier wrapper version make it functional smoke evidence only, never the headline4096/128 measurement.

AutoDebug completed: AUTODEBUG.md records the finite prefill signature/masking diagnosis. AutoFix remains active. Prepared recurrent-tail code was checked against inherited logical prefill for lengths1,2,31,32,33,127,128,129,255,256,257,4095,4096,4097 (`python -m models.autoports.qwen_qwen3_8_27b.tests.probe_prefill_buckets`, probe_buckets logs and prefill_bucket_equivalence.json). All-device convolution states were exact; recurrent PCC1. Logits PCC1 except4095(.9999272849774414), where padded4096 selects the inherited fused BF8 projection branch. The final full-model accuracy gate must validate that branch rather than dismiss the delta.

Complete129-bucket preparation probe started with `python -m models.autoports.qwen_qwen3_8_27b.tests.probe_preparation` (probe_preparation_01.log). Kernel compilation remained active after15minutes; no hang/reset claim. Its imported source predates subsequent page-window ownership fixes and therefore cannot alone close the final preparation gate.

Source audit found stale host page-map shadow use after external device table mutation, inconsistent all-inactive output types, warmup exception cache restoration, ignored non-greedy parameters in host compatibility, and stale inactive prefill logits. Fixes and expanded mixed-slot tests are in progress. Page-window gathering must read the actual bound device table, with runtime indices prepared before capture. Host compatibility must explicitly reject unsupported sampled distributions rather than silently argmax them.

CPU shared-suite HF control: `python -m models.autoports.qwen_qwen3_8_27b.tests.generate_hf_qualitative` (hf_qualitative_01.log). No packaged shared-suite asset exists in installed plugin; doc/qualitative_prompts.json fixes three chat-template prompts for this and subsequent stages. This is a local shared suite, not a claim that an unavailable upstream asset ran.

Harness prompt fix: installed readiness run_autoregressive stripped the rendered template's trailing newline (202tokens versus203 in the exact fresh reference). Patched the installed runtime to preserve prompt bytes and use strip only for empty-input validation. `readiness_preserve_prompt.patch` is the reproducible two-line dependency patch; comparison/scoring logic is unchanged. The local cached plugin path is ignored by git, so the patch artifact is stage-owned. The reference adapter still asserts exact token identity and100-token length before reuse.

Additional wrapper audit: context/position bounds are explicit; constructor can bind an existing replicated int32 device page table without changing its address. One setup-time map read establishes the host update baseline; it is not token-loop work. Persistent bucket masks can occupy substantial tile-padded storage at B32, so benchmark inventory now records request buffers as well as weights instead of relying on the provisional1MiB metadata estimate.

Terminal candidate: decode hidden rows are packed from[B,1,H] into[1,B,H] before final norm/head, eliminating one padded32-row matmul per user. This is terminal-only and preserves inherited inter-layer layout. `probe_terminal_packing.py --batch32` compares real reduced-stack hidden activations, logits and traced latency against the unmodified terminal shape. Candidate is unvalidated until this probe runs; no speedup claim yet.

Initial complete-domain preparation passed in2051.188s (cold compilation). All16 tested logical lengths1..4349, including4095/4096/4097, retained7243 program-cache entries and both trace IDs; diagnostic259-token all-logits path and repeated33-token output also passed. Exact artifact preparation_contract_initial.json/probe_preparation_01.log. This is evidence for the finite-bucket design before ownership fixes, not final-source acceptance. B8 external-cache/mixed-slot probe now running with those fixes and compact terminal rows (probe_mixed_slots_01.log).

Mixed probe reached nonzero external-cache seeding, successful injected-failure restoration, and normal preparation. Device-authoritative RM int32 gather works in that initial path. During static review, inactive SDPA output rows were confirmed intentionally unwritten. Added a masked-decode-only device where before sampling to zero those logits, preventing undefined inactive values from entering token feedback. Active-only decode remains unchanged. This edit occurred after mixed01 imported source; mixed01 is diagnostic, and final mixed02 must rerun the current source.

Mixed01 passed all-active/mixed/repeated ordering, external cache restoration, masked-decode nonzero inactive preservation, eager-prefill isolation and direct device page-map relocation. It failed only at a test-side torch.equal UInt32-versus-Long promotion on the continuation position assertion. Cast the readback to long for comparison. Mixed02 reruns current source, including inactive-logit clearing, with the same hardware command and log suffix02. No precision/layout fallback or reset was used.

Mixed02 passed on current generator source with tracker1 and program-cache checks included. All four devices preserved inactive nonzero cache/recurrent state exactly; failed and successful preparation restored external state; in-place device table remapping targeted exact physical pages;256+33 continuation versus289 whole prefill produced logits PCC1 and all recorded convolution/recurrent state PCC1. Three trace IDs were reused through all-active/mixed/low-level/device-map/all-inactive ordering. Artifact mixed_slots.json. Next: semantic greedy sampler comparison (probe_sampling_b1_01.log).

B1/B32 semantic greedy comparison passes; normal top-k(k1) is faster than force-argmax, recorded in sampling_contract.md. Terminal packing B32 gives exact logits/PCC1 and reduces complete terminal+canonical-output traced host-wall replay from53,103.86us to1,976.06us (100iterations), terminal_packing_b32.json/probe_terminal_b32_02.log. Initial comparison probe retained a newly allocated second-trace output across the first trace and tracker rejected it; repaired the probe to preallocate both canonical output buffers and warm copy signatures before capture, with no acknowledgments/scopes or tracking exclusions. Production generator already uses that stable-logit-buffer pattern. Retain compact terminal output; no inter-layer policy changed.

Packaging correction: selected skills require installed plugin packages to remain read-only. Restored the installed autoregressive runner byte-for-byte and moved the exact prompt-preservation change into tests/readiness_autoregressive.py, a stage-owned copy with only the two-line whitespace fix. The full validation script invokes that run_autoregressive function; prefill/teacher comparators remain imported unchanged. readiness_runner_provenance.json records the upstream original hash and local copy; readiness_preserve_prompt.patch remains the minimal diff. No scoring/output substitution was introduced.

Final reduced trace contract passes (probe_contract_03.log/reduced_trace_contract.json): sampled k8/p0.9/temp2 requests repeat from the configured seed, then greedy k1 and explicit host compatibility agree; both trace IDs remain unchanged. Token feedback and position proofs remain device-based. The final counters include deliberately requested scheduler/teacher-style and host-compatibility updates, not optimized autoregressive per-token host work.

Final complete-domain preparation passes on current source in48.591s using the now-warm JIT cache:7261 program-cache entries remain fixed across all16 logical lengths and all-logit259-token diagnostic. Artifact preparation_contract.json/probe_preparation_02.log. All-layer readiness command is `python -m models.autoports.qwen_qwen3_8_27b.tests.run_full_validation` in the recorded container environment (full_validation_01.log).

All-layer validation01 passed AIME readiness: prefill99/100top1,100/100top5,100/100top100; teacher decode98/100top1,100/100top5,100/100top100. Full AIME free-running output is coherent. Shared-suite inspection found TT structured output completed the correct JSON then continued past im_end into template/control tokens. This is a generator EOS bug, not model quality or precision failure. Added default stopping on generation_config.json EOS IDs (both248046 and248044 for the full checkpoint); fixed-length benchmarks explicitly use stop_on_eos=False. Teacher callbacks retain their fixed step count. Output-side EOS handling does not rewrite feedback/masks/positions per token. Reduced probe_contract_04 passes forced-stop, fixed-length, teacher, sampled-seed and host-compatibility checks.

Capability audit also identified an avoidable wrapper restriction: inherited decoder permits128-aligned continuation, while the initial wrapper required256. Restored128 alignment with the inherited odd128 K128 SDPA variant, preserving default K256 at256-aligned offsets; warm both variants before capture. No dtype/fidelity/layout policy relaxation. Mixed03 tests both128+33 and256+33 continuation against whole-prefill controls. Full readiness/qualitative is rerun after those fixes.

Mixed03 passes restored128/256 continuation and every ownership/inactive-row check.128+33 versus161 whole prefill has logits PCC0.9999728798866272 and state PCC1 across devices;256+33 remains logits/state PCC1. The128 variant retains the inherited K128 versus default K256 attention chunking, so this is not a new precision-policy change. Full validation02 is followed, only on success, by benchmark_full_model --batch1 --context262144 --lengths4096 --outputs128 --repeats3 --verify-native-context. Both use the normal stage environment with tracking1/program-cache checking0-skip; benchmark measurements explicitly include tracking overhead. Native checks exercise262141-token nonaligned generation, final decode input position262143, and exact262144-token prefill; they are capacity/finite-output checks, not accuracy or headline performance rows.

Full validation02 passes after EOS/128-continuation fixes: same99%prefill top1,98%decode top1,100%top5/top100 both phases. Structured TT output ends at EOS248046/token114 with correct JSON and no trailing template text. Shared outputs read; qualitative_verdict.md records controls, truncation limits and first AIME divergence(index15, omission of “carefully”). Mechanical degeneracy check passes.

Headline target now measured in benchmark_b1.json:4096 input/128 output/B1/concurrency1,3 warmed autoregressive repetitions with normal HF chat-template input and explicit fixed-length output. Median TTFT433.080411ms; median decode35.167404tokens/s/user. Each measured loop has127 model+127 sampler replays,128 token readbacks,0 host copies/token/position/page refreshes/captures. Tracking is enabled and throw_exception_on_fallback=True. Before/after preparation DRAM allocations are8,993,135,104/9,189,253,632 bytes/device. Native-context probes continue after the measured rows; their result is separate.

Native all-layer context checks passed: nonaligned262141+3, exact262144prefill, final decode position262143/next262144, finite logits and unchanged7389programs/traceIDs/allocator. Added CPU-only tests/summarize_full_model.py to reconcile measured inventories with allocator readings and regenerate performance.json/context/agentic evidence. B1 setup residual51,295,244bytes covers unenumerated helpers/scalars/semaphores and bank padding;196,118,528warmup delta is combined program/cache/trace/persistent overhead, not isolated trace bytes.

Final seed setter audit removed modulo/+1 remapping so supplied uint32 seeds remain exact, with bounded range reserving context increments and avoiding manual_seed sentinel. Current probe_contract_05 passes seed42 and high4294966783, repeated sampled outputs, greedy/hostcompat/EOS/page/feedback contracts. Current sampler probes suffix02 add cross-shard vocabulary boundary maxima and both canonical strategies pass. Their B1/B32 normal times451.935/488.003us beat force1621.336/2454.605us. B1 fullmodel evidence predates only this setter repair; default greedy path is unchanged.

Serial hardware pipeline after reduced probes runs benchmark_full_model --batch8 --context32768 --lengths4096 --outputs128 --repeats3, then batch16 same, then batch32 --context33024 --lengths4096 32768. Outputs benchmark_b{8,16,32}.json and logs benchmark_b{8,16,32}_01.log. All use trace tracking1 with program-cache checks included and strict fallback exceptions. B32 pool1,056,768tokens covers32x32768 working set plus generation/read-window headroom.

CPU-only HF control of the repeated-text4096 throughput fixture launched via tests.hf_benchmark_control (benchmark_hf_control.log/json). This checks whether repetitive prompt restatement is fixture-induced; it is not a performance measurement. Independent xhigh stage reviewer full_model_review started read-only while final batch/profiler/docs work remains pending; final rereview required before completion.

Initial independent review found an unused-tail out-of-bounds page gather at odd128 continuation offsets near the end of a bound map. Fixed _fill_page_window to assert required physical pages fit and clamp only unused256-rounded tail indices to the final valid page. Added physical external page-table multiple8-int row validation, matching SDPA's32-byte row alignment. Whole logical prompt lengths remain arbitrary. Mixed probe now includes384+127 versus511, exact bounded fill map, malformed external-map rejection, and actual prefill/decode logits across repeats, duplicate slots and reordered mixed prompts. Hardware validation pending. The active B8 process imported before this edit; its post-preparation source-file hash may describe newer disk content. Treat B8_01 as pre-fix diagnostic and use forthcoming tracker-off final run plus current mixed probe for final source evidence; do not silently claim that live hash is an import snapshot. Subsequent benchmark source hashes are now captured at startup.

Review also requires separate uninstrumented performance and measured memory reserves. Existing tracking-on runs remain trace-safety evidence. Planned tracker0 runs cover B1/8/16/32 and agent32768, with distinct filenames. tests/full_model_memory.py captures allocation/deallocation events (not device profiler) after performance to measure simultaneous extra DRAM aligned bank bytes; before/after trace release isolates trace buffer bytes. Reduced parser probe precedes all-layer use. No all-layer Tracy/device profiling is introduced.

HF4096/128 throughput-fixture control completed: HF reasoning summarizes the repeated input; TT128 spends more tokens literally restating it. This fixture is intentionally repetitive, but the difference merits a longer control rather than prose dismissal. Launched CPU HF512 control (tests.hf_benchmark_control --tokens512 --output .../benchmark_hf_control_512.json, log same stem). Final B1 clean timing run will additionally request --qualitative-control-tokens512 with normal EOS stopping; this separate output never enters headline performance. HF fixed-length control will be reviewed through its first EOS if one occurs, with raw output preserved. Both initial128 completions end during reasoning.

Added all64layer mixed trace ordering to benchmark_full_model after its measured rows: three prompts33/65/1, repeated masked generation, then all-active fixed slots, with exact repeated outputs, stable all trace IDs/capture count and zero steady-state host refreshes. This specifically exercises both captured decode keys at full stack. B8_01 imported the earlier script; a current tracking-on B8 acceptance run is required alongside the final tracker-off performance. Upcoming B16/B32_01 imports include this check. Reduced mixed-slot state/logit checks remain distinct.

Final verification is encoded in tests/final_full_model_evidence.sh (exclusive hardware, fail-fast). It runs current mixed/logit boundary probes, reduced allocation parser validation, then full_validation03 on final source, tracking-on B8 mixed acceptance, tracker-off B1/B32/B8/B16 measurements with post-timing memory audits, and reduced B1/B32 Tracy reports. Full validation03 reuses the exact fresh AIME reference under its recorded metadata and reruns the TT accuracy/qualitative gates after targeted repairs. Reduced profiler contexts now exactly match measured full-model pools: B1 native262144 and B32 configured33024; no all-layer device profiling.

B8_01 completed all129prefill shapes but failed identical-prompt slot token equality before persisting records; the serial pipeline stopped, so B16/B32_01 never started. No reset/hang occurred. Current mixed04 confirmed near-end page fix and external-map checks, then exposed prefill permutation maxabs0.0625. AutoDebug launched via installed launcher with the already-authorized child sandbox skip; report AUTODEBUG_slots.md identifies inconsistent rotary core-group FP32 settings and excludes unsupported generic rounding/CCL excuses.

AutoFix controls: tests.probe_slot_logits --control production and uniform_fp16_rope (diagnostic only) at B8/context512 and B8/context32768/identical-length4096. Production repeats are exact but mixed permutation differs0.0625. At4096, embeddings/layer0 are exact, first differences appear at full-attention layer3 and terminal. Uniform rotary FP16 diagnostic makes all observed stages and prefill/decode permutation exactly equal. Keep optimized packed path and requested FP32 policy in production; do not select this lower-precision diagnostic.

Minimal requested-config real-weight packing repro: tests.probe_prefill_rope --requested-config fails atB8/S160/Q (rope_requested_before.log). Repaired ttnn/.../rotary_embedding_program_factory.cpp so multi-tile group1 uses requested math_fidelity/fp32_dest_acc_en just as group2 does. Incremental build: docker exec qwen38-functional-build ninja -C build_Release -j8 _ttnn.so (rotary_build.log). First after-build test still failed because Python loaded the older installed build/lib/_ttnncpp.so, not the rebuilt build/ttnn copy; rope_requested_after.log is this stale-install failure, not evidence against the repair. Installed with cmake --install build_Release --component ttnn-runtime and --component tt_pybinds (rotary_install.log). /proc/self/maps and installed-library hash now recorded in rotary_build_provenance.json; subsequent full validation/benchmarks record the actually mapped backend hash and the factory source hash.

Requested-config packing retest uses rope_requested_after_02.log/rope_requested_after.json, followed by production slot_logits_repaired_s4096. Final script now uses mixed05, refreshed fullvalidation03, tracker-on B8/B32/B16 acceptance, then clean B1/B32/B8/B16 performance. B1 clean run repeats native-context validation after the kernel correction and adds the longer qualitative fixture control. No stage completion claim yet.

### Repaired all-layer batch8 closure and external preparation review

`final_full_model_evidence.sh` completed `full_validation_03.log` on the installed
rotary correction: prefill and decode top1=99/100, top5=100/100, top100=100/100.
The original all-layer B8 failure now passes (`benchmark_b8_tracking_final.json`):
three4096/128 repetitions, identical outputs in all slots, zero steady-state host
refreshes, and masked/repeated/all-active replay with three unchanged trace IDs.
Median47.666ms/token is close to the inherited stage05 layer-stack estimate47.410ms;
this tracker-on result is separate from the pending clean timing pass.

Read-only review of external preparation identified a separate issue: cloning the
entire external32x32768 pool adds10,358,358,016bytes/device, while warmup also uses
caller page maps whose unused entries may be -1. Paged fill skips that sentinel,
but prefill attention's physical-page lookup does not. This is a memory risk and
source-backed sparse-map hazard, not an observed B32 OOM. AutoFix is implementing
bounded, full-spec scratch state and an independent valid warmup map; external
cache/map identities must be restored before capture without replay or reset.
Hardware remains exclusively assigned to the serialized final evidence pipeline.

Focused source review of bounded external scratch found no remaining implementation
blocker after repairing capture cleanup. A post-record_end exception formerly
caused a second end_trace_capture call to mask the original failure before handles
were cleared. The repair attempts known trace releases independently and preserves
the original exception; the reduced probe injects this exact boundary and retries.
Active-cache replay assertions now cover every sampled layer/field/device, while
inactive sampled entries remain exact. Source compilation and diff checks pass;
no device validation is claimed until `final_external_evidence.sh` runs. The
full-stack external probe uses B32/context33024 and default complete preparation.
See `AUTOFIX_external_preparation.md` for scope and allocator-checkpoint limitations.

Tracked repaired-backend batch acceptance is complete: B8 three4096/128 runs median20.979tps/user, B16 one4096/128 run13.701tps/user, B32 one4096/128 run8.825tps/user and target32768/128 run7.408tps/user (TTFT142284.53ms). Every run has exact identical-slot outputs and zero steady-state host refreshes; all three larger batches pass mixed/repeated/all-active trace reuse. These tracking-on measurements are acceptance evidence; clean performance remains separate. Artifacts: benchmark_b{8,16,32}_tracking_final.json.

### Final temporary-map validation and resumed clean measurements

The first clean B1 attempt stopped before measurement: PyTorch reshape rejected a
TTNN Shape object in the new valid-map helper. Convert it to tuple; no tensor
policy changed. Failure log: benchmark_b1_perf_setup_failure.log. Device teardown
completed normally; no reset or process termination was used.

Targeted current-source checks then pass with allocation tracking enabled and
program-cache allocations included: external_prepare.json (sparse external cache,
pre-capture and post-record_end failures, successful retry, three replays changing
every sampled active layer/field/device and preserving inactive samples);
owned_sparse_{host,device,refreshed,mutated}.json (independent map ownership and
pre-prepare updates); mixed_slots.json/probe_mixed_slots_06.log; and
reduced_trace_contract.json/probe_contract_07.log. Generator SHA256 is
6a24b49785f02bef6dcd10a31f618a550fa53d87576a03a71de7f474eef46c5c.

Resumed without repeating completed tracker-on batches:
`docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh bash models/autoports/qwen_qwen3_8_27b/tests/resume_full_model_evidence.sh`.
It runs clean B1/B32/B8/B16 benchmarks, reduced profiles, target full-stack external
preparation, then full_validation_04. Reduced profile envelopes are explicitly named
reduced_stack_timing.json; no reduced timing enters full-model telemetry.

### Required long-generation quality repair

Clean B1 median4096/128 results are431.509ms TTFT and35.178tokens/s/user; all3output hashes identical and zero steady-state host refreshes. Current native262144 and nonaligned262141+3 checks pass. Allocation-event peak adds226,852,864bytes/device to9,189,253,632baseline; released traces25,526,272bytes/device. However TT512 on the exact4096 fixture loops on the supplied sentence; exact-prompt HF512 exits reasoning and writes substantive prose. Reviewer confirms first divergence56 and last320TTtokens an exact15-token cycle, no EOS in either control. This is required work, not a quality waiver. AutoDebug launcher running inspection-only (autodebug_repetition_launcher.log); current stage and telemetry explicitly incomplete. Existing serialized benchmarks continue while source diagnosis runs; no new TT job overlaps them.

To prioritize the new quality failure, paused only the owned resume script orchestration shell (containerPID1313003, SIGSTOP) while its active B32benchmark child1313114 continues normally. No device process was interrupted and no reset occurred. After that child finishes and closes the mesh, run the discriminating quality probe before resuming further benchmark phases. This pauses the test launcher, not the bringup goal.

### Long-quality discrimination, 2026-10-25 01:18 UTC

The B32 clean benchmark child completed normally before quality probes began; the owned orchestration parent remains stopped. B32 median3 autoregressive results:4096/128 TTFT14760.0405ms,8.821637tps/user; target32768/128 TTFT147101.6917ms,7.418408tps/user. These do not waive the B1 quality failure. ReducedB1 profile completed and compressed successfully (profile_b1.log, tracy/b1/reduced_stack_timing.json); whole reduced-stack envelopes117353.492us prefill and3106.996us decode, explicitly not all64 performance. Raw profiling was large; subsequent harness limits unmeasured decode replays and labels synthetic continuation history.

Production shared-history y56 control and both sampler comparisons rule out sampler choice on this trajectory; explicit host512 and device512 outputs are identical. Realprefix GDN comparisons are exact for layers0/1/2, all4ranks and128chunks, fused/phased/AUTO. See AUTOFIX_repetition.md and the corresponding compact JSON artifacts. Nativeprojection probe required test-only corrections: map `_prefill_impl` identities; allow legitimate `decode_batch=None`; capture actual first returned MM tensor rather than persistent RS scratch. Initial diagnostic artifacts are preserved as prefill_rs_probe_identity_failure, prefill_rs_probe_kwargs_failure and prefill_rs_invalid_scratch_capture. Scratch-MM metrics are invalid and cannot support a numerical claim; finite final-output comparisons remain scoped operator diagnostics. Corrected all4layer probe is running. No decoder policy change/reset occurred.

CPU HF y56 diagnostic launched via optimized_multichip_env.sh (hf_long_quality_decision_retry.log); initial invocation without established environment failed during transformers import and never loaded/executed model. A same-history eager-versus-trace model control is being prepared because host sampling still reused the model trace.

Corrected nativeRS probe exited0/closedmesh01:21:17, eight projections/all4ranks finite. Follow-on matched-MM process was launched while prior host teardown remained pending; held it with SIGSTOP before any device initialization (empty log and /proc fd inspection confirmed no Tenstorrent device descriptors), then SIGCONT only after prior mesh closed and session exited0. No overlapping device use or device-process interruption occurred. Matched-MM uses identical operands/configuration/directBF8; results pending. Full-stack eager/traced control script is ready and remains serialized behind it.

Matched-MM reduced control completed/exited0 at01:23:15: layer0 out/down, all4ranks, native returned MM exactly equals standalone matmul with identical geometry/config/directBF8. `prefill_rs_matched_matmul.json/.log`. This rules out fusion-specific MM differences for these operands, not the entire reduction or final full-model distribution. Launched full64 `tests.probe_long_quality_eager --separate-prefill` through established wrapper; log `long_quality_eager.log`, output same stem JSON. It first compares identical-history traced/eager model dispatch with traced sampler unchanged, then optionally retires traces before diagnostic separate-prefill warmup/control. It is not performance or an authorized production policy fallback.

Intermediate CPU gates (`stage_checks_inprogress.log/.json`): stock autoregressive degeneration checker exit0 on its recognized100-token AIME artifact only; it does not inspect the512-token throughput fixture, so explicit quality failure still blocks. Context checker exit0 at full262144. Agentic-serving checker exit1 advisories only for inherited, accounted norm/gate/up/gateup representations with existing measured retention reasons. No final-review or completion claim follows these intermediate checks. HF exact-history control completed: quote token1=21.0(rank1), token279=20.125(rank2), margin0.875; TT samehistory reverses ordering with3.25 margin. See hf_long_quality_decision.json.

Full eager/control completed/exited0 at01:29:31. Exact full y56 logits and outputs rule out trace-only execution mismatch. Diagnostic separate-prefill512 is coherent, reasoning closer338, substantive prose truncated512; same-historyy56 remains279 but earlierdecision55 changes. Matched reduction control then exited0 at01:30:27: original standalone reduction on identical matchedMM is bit-exact with fused postRS, all4ranks/layer0out+down. See AUTOFIX_repetition.md. Root read the complete separate-prefill text; no final fallback adopted.

ReducedB32 profile initially failed before device initialization because direct shell invocation omitted readiness runtime environment. Archived profile_b32_import_failure.log and tracy/b32_import_failure, then relaunched THROUGH optimized_multichip_env.sh. Successful device capture closedmesh01:33:28; host report extraction continues. Bounded harness records synthetic127-token continuation and actualposition4223, no autoregressive-history claim.

After that mesh closed, launched `tests.probe_native_prefill_geometry --variants both out down out16` (native_prefill_geometry.log/.json). It retains native MM/RS fusion, accepted dtypes/fidelity/TP4 and residual policy, changing only K-block grouping per diagnostic variant. Exact model/backend baseline identity is checked; all prior traces retired before each new warmup. No measured variant is automatically adopted.

ReducedB32 profile extraction/compression exited0. Complete windows: prefill489961.311us, decode5544.812us; modeltrace max5066.132us, sampler max575.425us. Scope includes onlyreal layers0/3 plusfull terminal/sampler, with syntheticcontinuation cachehistory4223; never substitute these for all64 or autoregressive performance. Inactive-row trace1 is captured but not replayed by all-active profile, explaining corresponding host-only-row warnings. Compact timing and scope artifacts under tracy/b32; rawCSV compressed after extraction.

Nativegeometry diagnostic exited0/closedmesh01:41:18. OutK4-only/downK4-only breakcopyloop but do not exit reasoning by512; bothK4/outK16 reproduceoriginal512 exactly. No productionpolicyadopted. Independent reviewer readallcompletions andrequests1024normalEOS extension preservingcandidate512prefix plusindependentlongcases. While those pairedclosure scripts are finalized, launchedfull64 targetB32/context33024 externalcache-preparation ownership probe through establishedwrapper (`external_prepare_b32.log/.json`); serializedaftergeometryclosed. That path cannot enter B1native4096geometry and is unaffected bycandidate-only retuning.

Retired old paused resume launcher: confirmed containerPID1313003 stateT and only child1313114 stateZ (benchmark alreadyexited), then sentTERM+CONT to the owned shell only. Its remaining stale automatic phases will not run or overwrite newer evidence. Current targetexternal ownership probe andCPUHFcontrols are independent anduninterrupted; noTTprocess terminated/reset. Remaining cleanB8/B16, finalB1 candidateclosure/validation/perf andfinalreview will be launched explicitly aftercurrentwork.

Full-stack external preparation passed/exited0/closedmesh01:55:41. Artifacts external_prepare_b32.json/.log: all64/B32/context33024/defaultcompletebucketdomain, allocationtracking1/programcacheskip0; sampled livecache/map/request/RNG preserved acrosspre-capturefailure+retry,3original-bufferreplaysupdateactive/preserveinactive+unusedpageswithoutrecapture. ExactcheckpointallocationnumbersinAUTOFIX_external_preparation.md (notpeaks). Reconciled memory/context/agentic evidence viaCPU-only summarize_full_model.py. LaunchedTTcandidateclosure `run_long_quality_cases --tt --geometry out --max-new-tokens1024`, log long_quality_cases_tt_out.log; CPUHFnewcases continue independently.

Read-only remaining-contract audit found documentation/summary corrections, nowapplied: distinguishmodelnative262144/configuredB8-B16pools32768/B32pool33024/largestexecutedprompts/unknownhardwaremaximumperbatch; updatecurrentnativeandexternalproofreferences; stateEOS-finishedhigh-levelrowsstillrunphysicalslots whileexplicitlow-levelinactiveKV/recurrentstateispreserved; repeatdeterminism isnull forone-recordgroups insteadofvacuoustrue; currentprofileevidence replacesfuture-tenseclaimswithoutattributingfabricpacket-warningcost. Externalactivecachecomparisons useinitialsentinelbaseline (notpreviousstep), nowexplicit. Nonechangesproductionmath.

Candidateclosure exited0/closedmesh02:06:04: original1024prefix512exact,reasoningclose887/prosebuttruncated; nearby4095coherent5paras/EOS892; distinctrecords4096strictJSONfacts3/3/EOS47. IndependentreviewerreadandendorsednarrowoutK4forFINALVALIDATION,notstagepass. Adoptedonlyprefillout_k48->4inmultichip_decoder.py; allotherstrategy/dtype/fidelity/layoutpoliciesretained, effectiveBF16partialstorageexplicitlydocumented. Savedpre-repairB1benchmark/readiness/qualitativebaselineunderbefore_geometry. Launched `run_full_validation --long-cases` throughestablishedwrapper, full_validation_04.log, usingONEfreshDEFAULTall64generator forAIME100/sharedsuite/three1024longcases; no geometryoverride. CPUHFnewcontrolscontinue. B1headlinecurrentlylabeledpre-repairbaselinependingreplacement.

PairedHFnearby4095 controlcompleted512tokens: reasoningcloser285, substantiveEnglishprose,truncated/noEOS. Rootreadentireoutput; TTcandidate sameexactinput closes325 andcompletesatEOS892. HFindependentfactual4096 nowrunning. Finaldefaultall64validator stillwarmingcomplete129buckets. Remainingcleanbenchmarklauncher final_full_model_post_quality.sh isprepared/syntaxchecked andwillrunonlyafterCPUreferenceworkends; B1freshnativecapacity+1024qualitycontrol+memoryaudit, thenB8/B16freshprocesses. B32measurementsretainedasscopedunchanged-path evidence.

Final-default validation04 completed02:22:05: prefill/decode top1=99%, top5/top100=100% on100reference positions; three long-case outputs exactly reproduce the reviewed outK4 candidate. Paired HF controls and reduced B1 profiling completed before clean benchmark launch02:24:46. Current reduced B1 whole-window prefill16331.556us/decode3104.421us, sampler558.194us; synthetic continuation history, not full-model performance or controlled comparison to the archived old profile. Kernel policy verified from embedded fused-op matmul ATTRIBUTES, not generic CSV fidelity column.

Final clean B1 completed02:32:43: median3 exact4096/128 TTFT433.812150ms,35.186165tps/user, identical output hashes, no steady-state host refreshes/recaptures. Clean AIME203/100 teacher-forced token-out control121.607590ms/35.507727tps; diagnostic readiness timing is separately scoped. Native262144/nonaligned262141+3/finaldecode262143 pass with7389programs and fixed traces0/1. Prepared DRAM9,189,270,016bytes/device; additional allocation peak226,852,864 gives9,416,122,880total; trace release25,526,272. Long1024 benchmark completion read: finite reasoning then substantive English prose, truncated, consistent with reviewed final-default quality. Serial launcher continues B8 then B16; B32 final branch unchanged by B1-only geometry correction.

Final clean B8/B16 launcher completed02:44:15 with exit0 and mesh closed. B8 median3 TTFT3577.570232ms/20.978294tps; B16 median3 TTFT7316.120286ms/13.701415tps. Both mixed/repeated/all-active validation and allocation reconciliation pass. CPU summary regenerated performance, context/agentic ledgers and decoder-stack comparison. Allclean benchmark packets use fixed4096/128, three repeats, diagnostics disabled; targetB32 additionally uses32768/128.

Final CPU checks in stage_checks.json/.log: degeneration and context pass; agentic checker reports only six advisories for inherited retained norm/projection representations with explicit byte accounting and measured retention reasons, no critical finding. Python compilation, shell syntax and git diff whitespace checks pass. artifact_manifest.json records133 local log/compressed raw-capture sizes and SHA256 hashes; compact reports remain versioned. Packet at supplied external telemetry path contains final4096/128/B1 autoregressive metrics and six accuracy records; device-time/rooflines remain unknown. Final independent review requested against this completed evidence set.

Independent final xhigh reviewer `/root/full_model_final_review` returned clean-pass with no required work at02:48UTC; full verdict in stage_review_final.md. Reviewer independently checked source/runtime provenance, raw completions, all133 manifest hashes, allocation-event arithmetic, medians, and supplied telemetry. Required repairs are closed; stated quality/context/profile-scope limitations remain. No vLLM integration, device reset or push occurred. Local checkpoint commit follows this accepted implementation/evidence.

Staging audit includes the previously root-ignored build_long_quality_cases.py explicitly. Source/Markdown/JSON/shell staged whitespace checks pass. An unrestricted staged whitespace check flags only literal unified-diff blank context lines and verbatim tt-perf-report padding/blank EOF; those diagnostic artifacts are preserved as generated. No production-code whitespace failure.

### Local checkpoint receipt

Stage complete; independent review clean-pass. Repository `/home/mvasiljev/tt-metal`, branch `exp26-qwen38-bringup`, implementation/evidence checkpoint `8d2790658451b296dd63da7d17a670fc42a61400`. All stage-owned implementation, tests, compact evidence and required rotary backend correction are committed. Used the existing prior-stage `Codex <codex@localhost>` commit identity through command-local git options because this checkout has no configured identity; global/repository config unchanged. This receipt is a separate documentation commit; its SHA is included in the external telemetry packet. No other repository was changed and nothing was pushed.

### Runner resume 025039c5 — environment repair

The runner's check-1 and check-1-retry logs both exit127 at `python: command not found`; no model gate executed. Added user-local `/home/mvasiljev/.local/bin/python -> /usr/bin/python3` (already on runner PATH), preserving the installed check script and model implementation. Exact command: `MODEL_DIR=models/autoports/qwen_qwen3_8_27b HF_MODEL=Qwen/Qwen3.8-27B bash .codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/prompts/model_bringup_multigoal/06-full-model.check.sh`. Result exit1: degeneration/context pass and six previously reviewed inherited-weight advisories only; multigoal defines1 as advisory and continues. Log: external run directory `06-06-full-model.resume-025039c5.check.log`. No new hardware run or measurement. Benchmark source hashes match current files after mapping the recorded container `/workspace` prefix to this checkout; prior independent clean-pass remains applicable. New telemetry packet `telemetry/packets/025039c5-ad75-41b4-bafc-f4ab8598fd9a.json` starts from the supplied new template and reuses verified existing evidence.

### Runner advisory refutation — attempt 795b8327 (2026-10-25)

Reproduced the supplied bug report using the unmodified installed checker from
this checkout after the skill environment setup:
`MODEL_DIR=models/autoports/qwen_qwen3_8_27b HF_MODEL=Qwen/Qwen3.8-27B bash .codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/prompts/model_bringup_multigoal/06-full-model.check.sh`.
Actual result **exit 1**, preserved in `resume_795b8327_check.log`: degeneration
and native-context checks pass; six weight-copy advisories, no critical finding.
This is a refutation of treating that advisory exit as evidence of an incomplete
model, not a claim that the stock script exits 0.

The cause is explicit in installed `scripts/check_agentic_serving.py`:
`check_memory` sends every `copies > 1` row to `shortfall`; an existing
`copy_reason` produces level 1, and `main` returns the maximum level. Thus even
fully evidenced retained copies necessarily return 1. The agentic-serving skill
explicitly permits measured retention reasons and calls these advisory. The
original stage goal also requires preserving the optimized decoder strategy.

Concrete controls for all six rows:
- `gate`, `up`, `gateup`: stage05 `prefill_projection_selection.json` records
  real-layer host prefill means for layer0 packed 6197.314us versus separate
  6002.467us, and layer3 packed 6634.490us versus separate 6271.887us. These
  are inherited projection comparisons, not new full-model/device measurements.
  `tt/multichip_decoder.py:140` allocates the separate projections and line142
  the packed representation; `_finish` selects packed decode/one-row and separate
  multi-row prefill. Removing the copies would discard the selected strategy.
- `in_norm`, `post_norm`, `norm_shards`: construction at
  `tt/multichip_decoder.py:135`–138 and distributed norm consumption at line448
  support the recorded full/sharded residual layouts. The inherited
  `optimized_multichip_decoder/residual_contract.md` and final all-layer
  `benchmark_b{1,8,16,32}_perf.json` establish the measured batch-dependent
  contracts. These are representations of the same norm, not accidental loads.
- CPU recomputation from `benchmark_b1_perf.json:weight_inventory` exactly
  equals the ledger sum: **6,522,077,184 bytes/device**. Each row's bytes count
  its named representation once; `copies` is not an extra multiplier. Target
  B32 pool 1,056,768 tokens exceeds 32 x 32,768 = 1,048,576. Allocator and reserve
  reconciliation remains in `memory_capacity_plan.json` and the ledger.

No checker or copy count was changed to suppress the advisories. No runtime
model repair was warranted by this report. One actual evidence defect was fixed:
`summarize_full_model.py` previously carried forward the stale agentic status
`implementation_in_progress` after all clean batch measurements were present.
It now emits `full_model_measurements_reconciled` when those measurements exist
and retains the in-progress status otherwise; this status does not assert an
independent review verdict. Rerunning this CPU-only summarizer changed only that
status in `doc/agentic_serving.json`.

Existing original completion evidence remains applicable: B1 benchmark provenance
SHA256 values for all 12 recorded source files match current files (container
`/workspace` mapped to this checkout). B32 retains the previously reviewed
historical benchmark/multichip source differences for changes to B1-only paths;
no fresh all-batch hash equality is claimed. Prior full independent clean-pass is in
`stage_review_final.md`; implementation checkpoint is
`8d2790658451b296dd63da7d17a670fc42a61400`. No device commands, new benchmarks,
profiling, reset, dependency changes or vLLM work were performed in this attempt.

Required local packet was built from the exact supplied template at
`telemetry/packets/795b8327-1563-4a7d-ba4c-022ce9e3254c.json` under the external
run directory. CPU checks verified all three benchmark rows are autoregressive,
4096 input /128 output/B1/concurrency1, then recomputed medians: TTFT
433.81215003319085ms and 35.186164937945414 tokens/s/user. Six accuracy records
come from the prefill and teacher-forcing JSON, each over 100 token positions
from one AIME24 prompt, not 100 problems. Device time/rooflines remain unknown
with reasons; reduced profiles are not substituted. Prior long-output quality
and batch/context scope limitations remain disclosed.

Independent xhigh review `/root/advisory_review` returned **clean-pass**, no
required work; see `stage_review_795b8327.md`. Reviewer independently reran the
stock checker (exit1/six advisories), verified allocations/consumers, paired
projection measurements, all four 1253-tensor inventories and ledger totals,
source provenance, status repair and packet values. Final CPU compilation and
`git diff --check` pass. The supplied failure is refuted as a completion blocker
with the stock advisory outcome explicitly retained. Local checkpoint follows;
no push. Commit receipt is stored in the required external packet.

Local checkpoint: repository `/home/mvasiljev/tt-metal`, branch
`exp26-qwen38-bringup`, commit `f047d7df040adf1c384349ba1d4414619f80a590`. No push.
