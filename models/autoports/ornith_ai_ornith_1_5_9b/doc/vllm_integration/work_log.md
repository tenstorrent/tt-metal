# Ornith vLLM integration work log

Stage 9 starts from datatype-sweep checkpoint `d1c644f3` on `main`. Status: in progress; no serving, accuracy or performance pass claimed. No push authorized or performed.

Selected policy is `bfp8_all_lofi`; adapter delegates construction to the default `OrnithModel`, so all weight groups, activation/CCL/KV dtypes, compute fidelities and layer exceptions come from the selected policy. Context remains 262144. Four Blackhole chips 0–3, mesh1x4, FABRIC_1D. Initial fuser check found no device owners. No reset performed.

The pinned stack is recorded in `stack_provenance.json`; installation is in `install.log`. Imports resolve to upstream vLLM0.31.0 and the standalone editable plugin. `pip_check.log` records three metadata conflicts and provenance classifies them. Plugin and vLLM source are unmodified.

The initial adapter constructs no standalone attention cache. The plugin calls its per-layer allocator, which creates the cache through the model and injects that exact object into `OrnithGenerator`. Sampling, trace capture, reload commands, recurrent-slot movement and async readback delegate to the generator. Async capability remains disabled pending actual adapter tests.

The first reduced server uses `OrnithServingDiagnostic` (real layers0 and3), runtime `TT_MODEL_CLASS_OVERRIDES`, max_num_seqs1, max_model_len262144, trace_mode decode_only, trace_region_size200000000, FABRIC_1D, sampling all and allocation tracking with program-cache allocations included. Logs: `diagnostic/server.log`, `diagnostic_runner.log`. It is solely an integration diagnostic, never final output or performance evidence. The runner's block-size default64 was overridden with32; subsequent runs use the typed flag directly.

Prefix caching retains the documented recurrent_state exclusion. No generation_config.json exists at the pinned HF revision (Hub404); server defaults and request-level sampler routing still need validation. Final run requires full sampling, P2/P3/P5 accuracy, mixed-shape trace reuse, qualitative review, single-user and burst benchmarks, independent review and local commits.

## Initial interface diagnostics

The first server captured all six traces but logging a MeshTraceId through JSON raised TypeError. Converting trace IDs to strings fixed that instrumentation bug. Diagnostic2 reached startup, then129-token prefill failed on empty multimodal fields passed by the plugin despite language-model-only. The adapter now drops only all-None fields and rejects actual media. Diagnostic3 then reached generator sampling, where plugin TTSamplingParams lacks log_probs_enable_temperature expected by SamplingSlotState. AutoDebug was launched inspection-only through the installed launcher. The child sandbox preflight failed because this container disallows unprivileged namespaces; the explicitly supported AUTODEBUG_SKIP_CHILD_SANDBOX=1 retry uses the existing unrestricted run authorization. No plugin source changed. Servers closed devices; post-failure fuser checks found no holders.

## Adapter and async diagnostics, 2026-11-13

The dataclass/M-RoPE repairs passed five host tests (`interface_final.log`) and reduced diagnostic4 requests. `adapter_decode_probe.json` passed through the adapter at B3, context256, layers0/3 with tracking enabled: two pending submissions, stale host objects unread, changed/unchanged page tables, resident remap, single-slot reset preservation, invalid transitions rejected before mutation. This is reduced correctness evidence only.

Generator instrumentation now records capture IDs, signatures, times and counters without changing sampling. Diagnostic5 failed due to a missing `time` import in instrumentation, fixed before diagnostic6. Diagnostic6 temporarily enabled supports_async_decode=True and exercised actual `--async-scheduling`. Its six mixed requests matched diagnostic4's non-overlapped control exactly for all 70 output IDs; 128 outstanding-step overlaps were logged. The trace baseline has six captures (two model, four sampling), zero releases before failure. Explicit logprobs=0 host requests passed. Returning to device sampling with a different mixed order failed `Continuation must follow this binding's exact prior end`; the mixed-shape gate is FAILED, not a partial pass. `diagnostic6_async/events_summary.json` records actual starts, ends, counters and reload cases. The current adapter capability is False again until the full lifecycle is repaired.

No final qualitative outputs exist; reduced synthetic-token outputs are unsuitable for coherence/language judgments. No headline or CI benchmark metrics were collected. Host logprobs mode is explicit compatibility testing; partial prefill chunks also select host logits through the pinned plugin. The six completed device requests used canonical traced device sampling and resident token/position state. No host argmax or adapter-owned alternative sampler was added.

The source investigations are `seed_continuity_investigation.md` (model-owned missing-history RNG offset) and `resume_contract_investigation.md` (shared prefill history information deficit). The proposed exact shared extension is recorded separately in `plugin_changes.md`; no extension has been implemented. Stage review cannot run before these findings and the remaining gates are resolved. Local SHAs: starting tt-metal d1c644f35289ee46bfa4a23598aa607d5ea96522; plugin base 042a94863ad27d88207bcf71abdf67122c6f4c27; no new stage commits, no push.

### Diagnostic6 command and requests

At this run only, the adapter async capability was True. Environment: activated tt-metal python_env and sourced `/home/ttuser/ornith-9b/lane-env.sh`; TT_MODEL_CLASS_OVERRIDES selected `OrnithServingDiagnostic`; VLLM_TT_PLUGIN_ROOT pointed at `/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin`; VLLM_SERVER_DEV_MODE=1, TT_METAL_TRACE_ALLOC_TRACKING=1, TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0. Serial TP4, chips0–3; no profiler.

```bash
timeout 3600 python -m readiness_check.run_vllm_server --stages serve \
  --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" --mesh-device P300x2 \
  --max-num-seqs 3 --max-model-len 262144 --block-size 32 --sampling-profile smoke \
  --tt-config '{"trace_region_size":200000000,"fabric_config":"FABRIC_1D","trace_mode":"decode_only"}' \
  --additional-server-args '--max-num-batched-tokens 1024 --language-model-only --async-scheduling' \
  --output-dir "$MODEL_DIR/doc/vllm_integration/diagnostic6_async" --server-timeout 1800
```

Targeted client: `timeout 400 python $MODEL_DIR/tests/vllm_request_probe.py --lengths 129,1033,131,130,129,2051 --output-tokens 70 --concurrency 3 --output $MODEL_DIR/doc/vllm_integration/diagnostic6_async/mixed.json`. Then lengths131,129,130 with `--sampling '{"temperature":0,"logprobs":0}'` to host_logprobs.json; then device lengths2051,129,1033,131,130,129 to mixed_repeat.json (failed). This client is nonstreaming; wall completion times are diagnostic only and cannot establish TTFT, TPOT or ITL. Logs: diagnostic6_async/server.log and diagnostic6_async_runner.log.

The server error cleanup invoked plugin shutdown and force-terminated EngineCore526372. Subsequent `fuser /dev/tenstorrent/{0,1,2,3}` and `pgrep -af 'vllm.entrypoints|EngineCore|readiness_check.run_vllm_server'` returned no holders/processes. No reset or new hardware operation followed. Run a bounded trivial-op health check before the next device job.

### Continuation diagnosis and authorization handoff

`async_continuation_investigation.md` identifies the exact mixed-shape failure as the pinned plugin allocator changing a continuing prefill owner's slot without moving state. Its AST host reproduction maps `{A:0,B:1}` and incoming rows `[C,B]` to erroneous slots `[1,2]`; preserving ownership requires `[2,1]`. This is an existing allocator contract defect, requires no new interface, and must be fixed with a complete separate plugin-change entry and regressions when continuing. The model continuation check must remain intact. It is distinct from resumed penalty history and does not invalidate the six exact control comparisons.

The persisted `tests/vllm_resume_contract_probe.py` reproduced four information-gap checks against the pinned SHA; JSON evidence is `resume_contract_probe.json`. `plugin_changes.md` contains the concrete eight-part proposed record for the opt-in original prompt lengths plus shared host-history correction. Specific human authorization remains pending. Current goal is incomplete; no completion or stage pass is claimed. The goal tool remains active because its blocked-status rule requires three consecutive turns with the same blocker; this is the first authorization handoff.

## Allocator repair completed, authorization still pending

Previous turn made concrete progress: repaired the independent existing ownership defect. Plugin local commit `a09b0ef5f600305c738da0482bb76fee0e2e776a` (base042a9486), clean checkout, no push. Full record in plugin_changes.md Entry2. New host regressions failed3/passed16 before; full plugin suite after repair passed2322/skipped1. pytest changed9.0.3 ->8.4.2 to follow plugin requirements; pre-commit passed. Files: slot_regression_before.log, slot_plugin_host_tests.log, slot_plugin_precommit.log.

Post-crash hardware health: timeout90 TP4 FABRIC_1D mesh open, replicated ones+ones, read/verify all4chips, close succeeded; post_crash_mesh_health.log contains MESH_TRIVIAL_OP_PASS. No reset. Diagnostic7 reused diagnostic6 command/environment with output diagnostic7_slots and the repaired plugin. Same mixed.json, host_logprobs.json and mixed_repeat.json workloads all returned200; all12 device results matched diagnostic4 synchronous control exactly. Actual mixed execution includes continuation start768/1024/1792/2048, fresh rows alongside continuation, and decode between chunks. Six captures at setup, no request-time releases; pre-teardown counters model347/sampling278/readbacks352/captures6/releases0. Summary: diagnostic7_slots/events_summary.json. This is two-layer diagnostic correctness only. No TTFT/TPOT/ITL or quality claim.

supports_async_decode=True restored after direct adversarial adapter tests plus fixed served lifecycle. The root full-model adapter still needs final all-layer proof. At /metrics running0/waiting0, sent SIGINT to owned runner536010; session94434 exited0. fuser devices0–3 and pgrep serving processes subsequently empty. No leftover owners.

The original-prompt boundary proposal remains pending through the original authorization handoff and two automatic continuation turns. Those messages repeat the goal but do not authorize the named capability/shared host-history change. Updated installed skill0.1.24 has an identical plugin-change policy to0.1.23. Seed/history correctness and the final sampling/readiness gate still require resolving this proposal; do not implement the new interface from automatic continuation alone. No stage review or full completion claimed.

## Explicit resume: ordinary-request seed repair

Resumed from the existing worktree, plugin commit a09b0ef, and original prompt copy `/home/ttuser/ornith-9b/multigoal_logs/09-09-vllm.prompt.txt`. No restart or dependency migration. Current lane-env supplies installed bringup0.1.24. The original prompt and supplied skills still require specific human authorization for the named original_prompt_lens interface; none was supplied. `resume_contract_probe_current.json` revalidates the information gap at a09b0ef (all four distinguishing checks reproduced).

AutoFix completed the independently implementable generator seed-offset repair. Files: tt/generator.py, tests/vllm_seed_continuity_test.py, tests/vllm_seed_decode_probe.py. Production metadata follows cache/slot ownership, authoritative position reloads and explicit histories; resident tokens/positions remain unread. It handles ordinary host/device prefills and transitions, explicit empty histories/rewinds, dormant slots, remaps and release/reuse. It does not guess original boundaries for resumed generated history. Unspecified seeds retain the existing random-base reset behavior.

Host evidence: seed_host_tests.log (25 pass), seed_host_before.log (original code reproduces5 numeric failures), seed_precommit.log (repository hooks pass). Device commands, fixture corrections and results are in AUTOFIX.md. Final adapter_seed_decode_probe.json status pass, 6 cases; adapter_decode_after_seed.json status pass. Both bounded300s jobs ran serially on TP4 with trace-allocation tracking including program-cache allocations. No performance claim; no all-layer pass. No hardware reset or plugin source change. Device/process owner audit after the runs is empty.

The existing plugin slot fix remains locally committed at a09b0ef5f600305c738da0482bb76fee0e2e776a. Generator/adapter stage changes remain uncommitted until full completion and review. No push. Stage-review remains deferred because required hard gates and full-model evidence are incomplete. This is the first turn of the resumed authorization audit, so goal remains active rather than prematurely re-marked blocked. Concrete dependency is plugin_changes.md Entry1.

Authorization audit, resumed turn2: previous turn made progress (ordinary-request RNG repair and device validation). Current plugin remains clean at a09b0ef; Entry1 still says pending, and the automatic continuation names no interface approval. Existing device/host evidence is preserved without rerunning completed checks. No shared-interface change is authorized. Completion remains unproven; goal stays active until the resumed blocked-audit threshold.

Authorization audit, resumed turn3: previous turn was no implementation progress, revalidating the pending authorization. The same concrete blocker remains through three consecutive resumed turns: plugin_changes.md Entry1, supports_original_prompt_lens and shared host-prefill history correction. Automatic continuations do not name approval. Independent allocator and ordinary-request RNG repairs are complete and tested; the remaining history correctness boundary prevents final acceptance. Marking goal blocked, not complete. Full all-layer sampling, quality/eval, performance, max-context acceptance and independent review remain outstanding.

## Owner-approved history interface implementation

Owner explicitly approved Entry1 in this thread. Implemented in plugin-only local commit `215effdee4e473f8c50af608d9812fafc56fd48b`, on top of `a09b0ef`; working tree clean, no push. Original prompt lengths are snapshotted per submission and passed only to opted-in adapters; shared host-prefill penalties use the authoritative boundary. Before/after regression logs and full suite (2348 passed,1 skipped) are `entry1_*.log`. Model-side canonical prefill now restores separate histories and logical RNG offsets; adapter requires the approved metadata. Focused device and served preemption validation follows.

### Approved interface: focused device and served tests

`timeout 420 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_original_boundary_probe --output $MODEL_DIR/doc/vllm_integration/original_boundary_device.json` passed 14 checks under allocation tracking on TP4 (layers0/3,B3,context256). It checks exact RNG state and all four penalty-history tables, resumed/ordinary parity, crossing chunks, host-to-device transitions, and fresh/replayed-prefix collisions after slot reuse; no recapture. Host22tests and hooks pass (`entry1_model_host.log`, `entry1_test_precommit.log`, `entry1_model_precommit.log`).

Diagnostic8 uses diagnostic7 launch shape with approved plugin215effd, authoritative boundaries, async scheduling, and explicit HF/tokenizer revision489cb97981b8654bcfcf30ce1f94ed1b62e07b53. `vllm_preemption_probe.py` sends upstream development `/pause?mode=keep&clear_cache=false`, `/reset_prefix_cache?reset_running_requests=true`, `/resume` after20streamedtokens. All six requests returned100tokens, reset accepted, original129/end151 captured. Exact resumed trajectories differed: greedydevice andhost firstdivergence62, sampleddevice22. This is under investigation, not a parity pass. External AutoDebug launched on retained evidence. No runtime source changed to suppress it.

Same-server streamed mixed workload passed all nine checks (`diagnostic8_history/mixed_trace.json`): device/host/device, 15requests70outputs, nonaligned129/130/131 and1033/2051chunks, exactgreedyoutputs acrossorders/modes, distinctcontinuationoffsets withconcurrentdecode, sixretainedcapture signatures andzero recapture/releases. This is reduced evidence only. Idle metrics were0/0 before SIGINT runner805699; it exited0.

### Immediate-resume page ownership repair

Read `diagnostic8_history/AUTODEBUG.md` after the isolated launcher completed0. Source chains and minimal host reproduction confirm retained-row appending of replacementblocks. New Entry3 recordsownership, alternatives, existing mechanisms, effects andauthorization before runtimeedit. Localplugincommit `2a1504554aa7babb1d426dfbca8c7db8c0a795ef` replacesretainedrows via existingadd_row andmarksdecodelayoutchanged. Before2failed2passed; after96focusedtests pass; full2354passed1skipped; precommitpass. No push. Diagnostic9 repeatsrealpreemption afterrepair.

`replay_numerics_served_context.json` repeats canonical adaptercomparison atservedcontext262144,B3,onlyslot0active,exact129prompt. Its first22sampledoutputs match the serverbaseline exactly. Uninterruptedoutput23=264 andresumedoutput23=51 reproduce the servedfirstdifference, withidentical presamplingseed275437 andallfourpenaltytables. Rawdecode/replaylogits PCC0.9999462981,maxabs0.25; host/deviceprefilllogitsidentical. This isolates thatfirstsampledifference tothecanonicalprefill/decodegraphnumerics independentlyofpluginpageownership. Largerphysicalpool thanserverused bydirecttest isrecorded inbuilder; context-dependentSDPAchunk128 matches. Exactautoregressivereplaytokenparity isnotassumedfromseedalone.

## Full-model acceptance launch, 2026-11-13

Diagnostic9 (local plugin `2a1504554aa7babb1d426dfbca8c7db8c0a795ef`) passed all nine mixed-shape trace checks with 15 requests, 70 outputs each, concurrency 3. Evidence: `diagnostic9_pages/mixed_trace.json`; six prepared traces retained, no new captures or retirements, multiple continuation offsets and concurrent decode. Forced preemption completed all requests but strict uninterrupted/resumed token equality remains false; retained as a failed exploratory check pending numerical-control interpretation, not a serving pass. Diagnostic9 was stopped with SIGINT to owned runner PID841375 after running/waiting metrics both reached0; runner exited0, fuser found no device holders.

Started full 32-layer adapter through the same plugin with `max_num_seqs=32`, `max_model_len=262144`, physical block32, scheduler token budget1024, TP4 P300x2, async scheduling, trace tracking enabled, trace region200000000, FABRIC_1D, decode_only. Exact runner command is in `readiness_vllm/full_tracked/runner.log` and the runner-generated summary. Default chat thinking is explicitly off for bounded ordinary answers; P3 thinking requests override it. Built-in reasoning parser qwen3 and tool parser qwen3_coder are enabled. Operator serving recipe explicitly overrides omitted top_k to20 because the pinned checkpoint has no generation_config sampling defaults; this fits the verified32-candidate device sampler. Explicit unsupported features still use documented host compatibility. No benchmark performance will be attributed to this tracking-enabled run.


## Diagnostic9 read-only preemption audit, 2026-11-13

Saved `diagnostic9_pages/preemption_analysis.json` and `.md`, comparing diagnostic8/9 and the served-context canonical control. Entry3 `2a1504554aa7babb1d426dfbca8c7db8c0a795ef` visibly replaces resumed prefill page rows: greedy/host `[8,7,6,5,4]` becomes `[4,5,6,7,8]`; seeded reverses that ordering. Every replay uses start0/end151/original129/slot0. All baseline sequences are unchanged and all six requests complete100 tokens. Forced host raw scores first change between plugin versions at output index32, absolute decode position160 and first newly needed logical page5; seeded old/new token sequences first change there too. This supports the ownership repair; decode page rows were not recorded.

**All three strict uninterrupted/resumed token comparisons remain failed.** Diagnostic9 greedy device and host first differ at index49 (53 versus498), seeded at index22 (264 versus51). Canonical served-context recomputation independently reproduces only the seeded first difference with identical four penalty tables and seed275437, differing raw logits (PCC0.9999462981, max_abs0.25), and identical host/device prefill logits. The control ends at output23 and does not compare recurrent/convolution/KV state. Greedy index49 has not yet been independently reproduced; no exact equivalence or numerical-tolerance acceptance is claimed. A targeted canonical greedy replay through output50 remains useful. Full32-layer/B32 acceptance is separately underway; this audit ran no device job and edited no runtime code.

Full-model all-layer mixed trace acceptance passed all nine checks: `readiness_vllm/mixed_trace.json`, same EngineCore851944, B32 prepared bucket,15requests×70outputs/concurrency3, logical lengths129/130/131/1033/2051, actual continuation offsets and overlap logged. The exact selected policy is emitted by registration; pool8224blocks fits required context plus32 reservation pages.

The first full sampling attempt failed during collection in `test_tile_size.py`: importing plugin platform before upstream vLLM initialization triggers a circular import (`current_platform`). `PYTEST_ADDOPTS='-p vllm'` initializes the upstream package before collection using pytest's normal preload mechanism; the unchanged tile test then passed. No source, tests, assertions, selected items, or server config changed. Preserved failure: `readiness_vllm/sampling_collection_failure.log`, failure summary alongside it; passing focused import check: `sampling_collection_fix.log`. Full runner rerun with this test-process-only environment is in progress.


### Prepared targeted greedy numerical control

Extended `tests/vllm_replay_numerics_probe.py` with optional `--greedy`, `--follow-outputs 50`, and `--served-reference .../diagnostic9_pages/preemption.json`; defaults remain sampled output23. It generates the uninterrupted branch to the requested total, replays the original129 plus22 outputs, then follows the resumed branch to the same total. Each additional decision checks all four sampler tables against its authoritative history before sampling. At the first unequal token it records both state oracles/signatures, raw-logit metrics/top64 and full raw vectors in `<output>.first_divergence.pt`. Optional served comparisons retain separate baseline/forced equality results. These are diagnostic reads, not corrective runtime inputs; the usual explicit first decode reload after prefill is preserved.

Host-only fake control-flow validation exercised default23 and greedy50, including a synthetic divergence49, exact state checks, served-reference matching and raw artifact output. Compile and Ruff checks passed (`replay_followup_host_validation.log`). No device execution or actual greedy reproduction is claimed; parent will run after the full-model server releases devices. Suggested flags: `--context 262144 --greedy --follow-outputs 50 --served-reference $MODEL_DIR/doc/vllm_integration/diagnostic9_pages/preemption.json --output $MODEL_DIR/doc/vllm_integration/replay_greedy_follow50.json`.

Full shared sampling and qualitative runner exited0: `readiness_vllm/full_sampling_runner.log`, 67passed/75skipped in690.87s, final profile full, B32. All twelve generated qualitative texts read directly; coherent/on-topic, no mechanical repetition, gibberish, language drift or request contamination. Cap-limited answers recorded in `qualitative_observations.json`; HF/TT controls pending. P3 all12conformance cases pass, plus two render checks; only render_c07 advisory fails (legacy reasoning history field normalization under investigation). `doc/eval_readiness/chat_conformance.json` and raw directory are authoritative. Attached sampling runner summary has unused default block_size64; actual server launch uses32 as recorded in full_tracked/server.log and launch summary. This metadata mismatch does not describe execution; future attached commands specify32.


### P3 render_c07 advisory explanation

All12 behavioral cases and all10 critical checks pass. The retained `render_c07` advisory is a deprecated-field discrepancy in upstream `/tokenize`: legacy `reasoning_content` yields73 tokens, while current `reasoning` yields88. The original actual `/chat/completions` request already returned `Ada` with88 prompt tokens; its request validator normalizes the legacy alias, so the source chain and recorded usage support preserved chat history. No model/template change is indicated.

Authorized CPU-only tokenization checks reproduce73/88/88 server tokens for legacy/current/identical-dual fields with thinking off. The pinned HF template expects `reasoning_content`; copying canonical `reasoning` into that template alias gives exact token equality for the current API. Thinking on produces71/86/86, confirming the default thinking-off suffix is not the cause. Operators should use canonical `reasoning`; a direct HF reference should mirror the existing template alias, or a shared fixture may supply both fields identically. No new interface/default, plugin/upstream edit, installed-probe weakening or limitations waiver is needed. The original failed advisory remains unchanged.

Evidence: `doc/eval_readiness/chat_conformance/reasoning_field_analysis.md`, `reasoning_field_tokenization.json` (six exact CPU requests/responses and HF tokens), and `reasoning_field_host_proof.json` (actual installed schema-normalizer AST execution and source hashes). No new chat inference or device job was submitted during this investigation.

HF controls run concurrently on CPU during correctness-only serving (not benchmarking). Greedy worker PID862015 was restricted to CPUs0-3,8-11 after prompt03; sampled worker uses4-7,12-15 from launch, torch8threads, OMP_WAIT_POLICY=PASSIVE/GOMP_SPINCOUNT=0. Commands are in control invocation manifests; affinity log: `readiness_vllm/qualitative_hf_affinity.log`. Same pinned BF16 HF model, templates, policies and token limits; scheduling changes only. Performance probes/benchmarks will run after CPU generation finishes.

### Continued acceptance, 2026-11-13 18:57 UTC

The existing full-model tracked B32 server (EngineCore851944) and P5 accuracy
process890952 were retained across context continuation. No competing device job
was started. HF greedy and sampled controls continue on their disjoint CPU
sets; these overlap correctness traffic only, never headline timing.

The single maximum-context acceptance request is queued after P5 exits and
writes its result. It uses the existing server, not a new allocation:

```bash
"$TT_MODEL_BRINGUP_ROOT/scripts/wait_for" --pid 890952 \
  --file "$MODEL_DIR/doc/eval_readiness/long_context_accuracy.log" \
  --timeout 5400 --tail 18
# Require the P5 result to exist and no earlier capacity artifact before launch.
timeout 7500 python "$MODEL_DIR/tests/vllm_context_acceptance.py" \
  --output "$MODEL_DIR/readiness_vllm/context_acceptance.json" \
  > "$MODEL_DIR/readiness_vllm/context_acceptance.log" 2>&1
```

This sends262143 explicit prompt tokens plus one output, matching the262144
context contract. It is capacity/non-alignment coverage, not a quality verdict.
Its status remains pending until the actual result is written.

The final measurement plan separates tracked correctness/route auditing from
untracked timings. Before shutting down the tracked server, run the shared
B32 benchmark/prefill-burst workloads once into a separate tracked-burst
output directory and count its per-step device/host markers plus OOV errors.
These are acceptance/route artifacts, not the primary performance result.
Repeat the identical greedy workloads with tracking disabled and CPU controls
finished for final timing. This preserves explicit sampling/burst fallback
auditing without per-token log writes in the measured performance path.

2026-11-13 19:07 UTC: matched HF greedy control exited0; all six outputs were
read in full. They are coherent and on topic. The longer explanations/story
and Fibonacci response reach the256-token budget, including an incomplete
third-law sentence and final heading, as also observed in serving. French
translation is correct and code is valid. No mechanical repetition, gibberish,
unwanted language drift or contamination was seen. This is recorded in
`readiness_vllm/qualitative_observations.json`; the comparative verdict remains
pending sampled HF and same-code TT controls. Once the greedy process released
its cores, the existing sampled HF process900282 was widened to CPUs0-15;
`qualitative_hf_affinity.log` records the change. No performance metric uses
these overlapping CPU-control periods.

Sampling skip audit: of75 skips,71 belong to optional speculative-decoding
modules (no speculative model is configured), one is unsupported all-vocabulary
logprobs, and three are `test_chunked_prefill.py` requiring an explicit
`--tt-chunked-prefill-budget` that the installed full-profile runner omits.
The full-model mixed trace probe already exercises chunking, but these three
shared tests will additionally run against the same tracked server with its
actual1024 budget. Use the documented chat mode (`--tt-recall-reasoning-effort
low`) so this instruct model is evaluated with its chat template; the server's
explicit default thinking-off still controls its template. No test or fixture
will be modified. Preserve this targeted result separately from the original
67passed/75skipped full-run log.

### P5 accuracy passes, 2026-11-13 19:18 UTC

The original `run_long_context --part accuracy` job exited0 and wrote
`doc/eval_readiness/long_context_accuracy.json` plus
`long_context_accuracy/cases.jsonl`. All four checks pass: recall at8192,
32768 and131072 target tokens, all three depths(.1/.5/.9), plus on-device
sampling declaration. Actual prompt lengths8166–8167,32734,130898–130899;
served context262144, temperature0, thinking off. All nine returned the exact
needle; raw answer tails and finish reasons were read. No context clamping,
request failure, quality waiver, HF recall exception or sampler change was
needed. Intermediate chunks return logits without sampling; final greedy
sampling uses the device path. The queued single max-context capacity request
began after this probe completed on the same EngineCore851944.

Full sampling/qualitative route counts are now in
`readiness_vllm/sampling_runtime_audit.json`. Timestamp-bounded full-suite
traffic contains1174 device and1239 explicit compatibility host decode calls;
the suite deliberately requests logprobs/unsupported sampling processors.
The following twelve qualitative requests contain2133 device decode calls and
zero host decode calls, with12 initial input reloads and69 page-only refreshes.
All12 prefill completions use device sampling. There are no OOV assertion lines
through the audit timestamp. The JSON records exact window boundaries and the
raw server log; no inference clients overlapped these windows. This independently
confirms the operator's top-k20 default keeps served sampled text on device.

2026-11-13 19:23 UTC: sampled HF control also exited0. All six sampled outputs
were read in full; they are coherent, on-topic, free of mechanical repetition,
gibberish, unwanted language drift and request contamination. Translation and
Fibonacci algorithms are correct; long explanations/stories/code examples
truncate at the same256-token budget. `qualitative_observations.json` now records
both six-output HF comparisons. Only the same-code/same-chip TT control remains
for the final qualitative verdict. Both CPU generation jobs have finished, so
subsequent performance measurements will not overlap them.

### Queued remaining tracked-server acceptance

After the maximum-context JSON records pass, the same server will run:

```bash
export PYTEST_ADDOPTS='-p vllm'
timeout 1200 python -m pytest \
  /home/ttuser/ornith-9b/dependencies/vllm-tt-plugin/tests/tt/test_chunked_prefill.py \
  -v --tt-server-url=http://localhost:8000 --tt-model-name="$HF_MODEL" \
  --tt-max-num-seqs=32 --tt-chunked-prefill-budget=1024 \
  --tt-recall-reasoning-effort=low \
  > "$MODEL_DIR/readiness_vllm/chunked_prefill_tests.log" 2>&1
timeout 900 python "$MODEL_DIR/tests/vllm_mixed_trace_probe.py" \
  --server-log "$MODEL_DIR/readiness_vllm/full_tracked/server.log" \
  --length-offset 8 --output "$MODEL_DIR/readiness_vllm/lifecycle_trace.json" \
  > "$MODEL_DIR/readiness_vllm/lifecycle_trace.log" 2>&1
timeout 600 python "$MODEL_DIR/tests/vllm_logit_determinism_probe.py" http \
  --output "$MODEL_DIR/readiness_vllm/full_tracked/logit_http.json" \
  > "$MODEL_DIR/readiness_vllm/full_tracked/logit_http.log" 2>&1
```

A separate waiter starts the shared `benchmark,prefill_burst` runner only after
HTTP logit status is pass. It attaches with actualB32/context262144/block32,
meshP300x2, sampling-profile full, trace_region_size200000000/FABRIC_1D/
decode_only/sample_on_device_mode all, and metric percentiles50,99. Artifacts
stay under `readiness_vllm/tracked_burst/`; the raw runner command is included
in its result JSON. `audit_start.json` records the exact source-log byte boundary
and timestamp before the workloads; `runtime_audit.json` recounts decode modes,
reloads, trace captures and OOV errors after they finish. These jobs are queued,
not passes, until their outputs exist. They are serialized and bounded; no
competing TT process is launched.

## 2026-11-13 19:37 UTC — prefill source audit during capacity acceptance

The running EngineCore851944 continues its one maximum-context request. A CPU-only
audit found that the adapter does not pass the already-known original prompt
horizon to the canonical prefill runtime. At scheduler chunks1024 and prompt262143,
the actual planner predicts48 full-prefix rebuilds and6,111,232 replayed prefix
tokens; this is a source-level work estimate, not a measured timing claim. The
plugin also suppresses intermediate sampling but the generator currently reads
back its discarded logits. These are model-owned gaps: no new plugin API is needed.
A minimal correction is being implemented: pass known horizons while all rows are
still within their original prompts, and explicitly discard intermediate device
logits without host readback. Mixed final rows retain real host compatibility
logits; replay beyond original prompt retains the unknown-horizon logic.

The running server has the earlier code loaded; its exact source is preserved in
`tracked_loaded_source.json`, `tracked_loaded_generator.patch` and
`tracked_loaded_adapter.py.source`, relative to the recorded tt-metal base. Its
capacity/mixed/sampling evidence will remain labeled accordingly. The corrected
path requires new tracked mixed/chunked/sampling validation before untracked final
measurements. The maximum-context acceptance request will not be repeated merely
for this refinement. No runtime source is hot-reloaded into the running engine.

`run_standalone_acceptance.sh` serializes the pending greedy replay attribution,
B32 external-cache logit oracle and same-code TT qualitative controls. It refuses
to start while any device is held; all device jobs have bounded timeouts. Its
completion messages deliberately require separate result interpretation.

The minimal prefill correction is now implemented. `discard_logits` is an
explicit per-row canonical prefill option; validation rejects malformed rows,
inactive discards, device sampling and all-logits combinations before mutations.
The adapter marks only `0 < end < original_prompt_len` for discard and forwards
known request horizons only when every row remains within its original prompt.
No terminal row is guessed from the original boundary during generated-token
replay. CPU evidence:42 tests pass in
`readiness_vllm/intermediate_prefill_host_tests.log`; targeted pre-commit passes in
`intermediate_prefill_precommit.log`. Hardware acceptance remains pending.

The standalone B32 logit oracle is deferred until the corrected tracked HTTP
artifact exists, avoiding a duplicate obsolete control. The standalone chain now
contains greedy replay attribution and both TT qualitative collections.

A CPU-only execution of the actual pinned host sampler for P2 reasoning
(temperature1, top-k40, top-p.95, seed42; vocab248320) measured B32 medians
152.978/146.766/152.216ms at1/2/4 Torch threads, and means153.4/153.1/160.0ms.
Tokens and RNG states match across thread settings. Three warmups plus ten
measured calls use official hardware-blocking stubs; no device transfer, model
execution or scheduler work is included. This does not measure P2 serving or
establish a waiver or completed fix time box. It rejects an assumed large gain
from simply increasing host sampling threads. Exact reproducer, results and
limits: `p2_host_sampler_cpu_probe.md` and its adjacent JSON/log, plus
`tests/vllm_host_sampler_cpu_probe.py`. No plugin/runtime setting was changed.

The unknown-to-known correction is implemented: compatibility and boundary
checks now run for every continuation unless missing-history behavior was
explicitly requested.44 CPU tests pass. Restoring the old condition makes both
new actual-runtime validation regressions fail with the expected profile/tail
exceptions, recorded in `intermediate_prefill_transition_negative_control.log`.
Independent follow-up inspection confirms the original finding is closed.
`prefill_transition_validated.json` releases the corrected-server launch guard;
it explicitly records hardware validation as pending. No plugin source changed.

Maximum-context acceptance passed on original full_tracked EngineCore851944: HTTP200,262143 input tokens+1 output=262144 total, elapsed2248.311s. Artifact: `readiness_vllm/context_acceptance.json`; request formula and exact response are preserved. This is the single Stage9 capacity request, and remains labeled with the earlier loaded prefill source. No context reduction or truncation was used. The subsequent model-owned refinement removes repeated prefix work and discarded-logit readback; new tracked/P5 validation is separate.

## Chunked recall output-budget diagnosis

The first targeted three-test chunked suite used its default24-token cap and
`--tt-recall-reasoning-effort=low`; all three failed the completed-answer
assertion (`finish_reason=length`). The explicit low reasoning setting activates
reasoning despite the server default thinking-off recipe. A direct identical
solo request with max_tokens512 completed in89 tokens,72 reasoning tokens, and
returned the exact passphrase cobalt-heron-42 with finish_reason=stop. The full
request/response is `readiness_vllm/chunked_recall_budget_probe.json`. This was an
output-budget failure, not an observed wrong recall or quality verdict.

Preserved the first raw log as `chunked_prefill_tests_24_token_cap.log`; rerunning
the same three tests with the plugin's documented
`--tt-recall-reasoning-token-budget=512` (total cap536), retaining every original
recall and finish-reason assertion. Corrected tracked script uses the same budget.
No plugin/test/template source was changed.

## 2026-11-13 20:02 UTC — old tracked acceptance completed

The targeted canonical chunked-prefill suite passed all3 tests in30.82s with the supported512-token reasoning budget (total536 output cap); the original24-token budget failures remain preserved. Lifecycle offset8 passed all9 checks, with running/waiting requests and KV-cache usage zero before and after. HTTP logit repeat/reorder comparisons are exact (`readiness_vllm/full_tracked/logit_http.json`).

Tracked burst route audit passed:261 device decode calls,0 host decode calls,11 device prefill calls,0 new captures,0 OOV errors. Raw benchmark artifacts live in `readiness_vllm/tracked_burst/`; these tracking-enabled older-source measurements are auxiliary, not final headline performance. The JSON wait helper missed a changed status at the beginning of an atomically replaced result. After verifying actual status pass, only its waiting child was stopped and the saved remainder was run as `restart_tracked_burst.sh`. No serving process was interrupted for that queue repair.

The old server was then confirmed idle twice and its owned runner851433 terminated normally; EngineCore851944 exited and device ownership was empty. Shutdown evidence is `full_tracked/shutdown_idle.json`, `holders_after_shutdown.log` and `runner.log`. The serialized standalone controls started next, followed by the prepared corrected tracked acceptance.

## 2026-11-13 — matching standalone text controls reviewed

All12 TT control outputs were read in full, six greedy and six sampled, alongside the already-read12 HF controls and12 earlier served outputs. All are coherent/on-topic, with no mechanical repetition, gibberish, unwanted language drift or request contamination. Prompt01 is a sensible ML haiku,02 distinguishes labeled/unlabeled data,03 follows the inventor story,04 explains thermodynamics,05 translates correctly into French,06 supplies valid Fibonacci code. TT greedy01/02/05 exactly match earlier served text; bitwise cross-geometry identity is not a quality gate (standaloneB1/context2048 vs servedB32/context262144). Both controls use the same256-token budget; long responses end mid-paragraph/code when capped. Sampling controlseed0 versus unseeded serving is a qualitative comparison only. `readiness_vllm/qualitative_observations.json` records per-control artifacts and limits. Corrected tracked serving outputs still need their own review.

The reduced greedy replay control reproduces both served branches exactly through50 outputs and the first divergence at index49 (53 vs498). Its prior49 tokens and sampler state match, including seed275464. Raw logits at that decision have PCC0.9999743544,maxabs0.0625. This closes serving-only reproduction uncertainty but does not establish recurrent/KV-state correctness; an optional aligned consumed-prefix151 state comparison is being prepared independently. No strict replay-equality pass is claimed.

## 2026-11-13 20:26 UTC — corrected tracked acceptance and sampler repair

Corrected full-model EngineCore1060978 passed the full canonical sampling profile (67 passed,75 skipped), mixed_trace and lifecycle_trace (all9 checks each), and HTTP logit repeat/reorder equality. Actual chunk starts include0/640/1024/2048, six discarded intermediate rows per mixed workload,347 decode calls and no captures beyond the six prepared signatures or retirements. P3 passes every behavioral case and all10 critical checks, retaining the known render_c07 advisory. P5 accuracy is still running; no result is asserted yet. Artifacts remain isolated under `readiness_vllm/corrected_tracked/` with the exact loaded model source.

All12 corrected served qualitative outputs were read in full and judged coherent/on-topic, without mechanical repetition, gibberish, language drift or contamination. The sampled haiku is on-topic but not strict5/7/5. Matching HF/TT controls support the quality verdict, not sampled RNG equality. `corrected_tracked/qualitative_observations.json` and `qualitative_runtime_audit.json` record the verdict and2096 device/0 host decode calls over the collection window. Root artifacts are not yet promoted.

The opt-in64-candidate primitive repair is prepared separately in `/home/ttuser/ornith-9b/tt-metal-sampling64`, with host compilation/linking, sanitizer checks, falsifying mutants and offline Blackhole kernels validated. Default32 assembly matches the original after path-directive normalization. No device validation or model capability widening has occurred. The current server and queued original32 measurements still use the untouched32-candidate native build. Exact isolated handoff and commands: `sampling64_validation/AUTOFIX.md` in that checkout. After corrected acceptance, the serialized queue runs the B32 standalone logit oracle, reduced prefill horizon test, aligned151-token model-state comparison, and original32 untracked P2/benchmarks. Each result needs inspection; measured/complete markers are not blanket correctness verdicts.

The benchmark driver accepts an artifact label and preserves P2 failures while collecting independent benchmark evidence, returning the P2 failure afterward. Its environment sidecar now uses MEASURE_CONTROL consistently; shell syntax passes. No hardware process was interrupted and no reset was performed.

## 2026-11-13 20:43 UTC — corrected P5 accuracy passed

All nine installed P5 accuracy cases pass on the corrected full-model tracked server: actual prompt lengths8166/8167,32734,130898/130899 at depths0.1/0.5/0.9. Read every raw answer; each is exactly its distinct needle passphrase, finish_reason=stop, with no error or contamination. Raw cases and compact result are under `readiness_vllm/corrected_tracked/eval_model/doc/eval_readiness/long_context_accuracy/` and adjacent JSON; explicit review is `corrected_tracked/p5_text_review.json`. This uses the original32-candidate canonical sampler. The tracked benchmark route audit and normal shutdown run next, before standalone device controls.

## 2026-11-13 20:50 UTC — exact logits and prefill edge cases

Corrected tracked route audit passed:261 device decode calls,0 host decode calls,238 steady steps with neither input nor page-table reload,11 input reloads,12 page refreshes,11 prefill calls,0 new captures and no OOV errors. Normal idle shutdown completed, with the device-owner guard empty before standalone work.

The B32/context262144/8224-block standalone full-model logit oracle matched corrected HTTP output exactly for all9 prompt/order comparison groups and3 output positions each: every selected logprob and top20 candidate logprob agrees, maximum absolute candidate error0. Repeated standalone raw logits are exact. `corrected_tracked/logit_verdict.json` records interpretation of the raw `logit_tt.json`, whose original status remains measured.

The reduced B3/context4096 horizon device probe passed all4 cases. Unknown-to-known incompatible profiles and missing-tail continuations rebuild correctly. Mixed final/intermediate calls read exactly1 real final-logit row; all-intermediate continuations read0. Device seeds, four history tables, and host sampling metadata remain unchanged. Evidence: `readiness_vllm/prefill_horizon_device.json/.log`; `prefill_transition_validated.json` now records hardware pass.

The aligned151-token model-state diagnostic initially failed allocation tracking: diagnostic SliceDeviceOperation program-cache buffers were first created after capture. This is a probe preparation defect, not a model-state comparison result. The probe author is preparing those readback keys before capture, without exemptions or disabling tracking. Failure artifacts are preserved. Two earlier queue guards correctly refused the brief process-destructor interval after completion markers; each next job started only after released devices were confirmed. Only idle wait child1096884 was terminated to make benchmark launch depend on root review, not an early marker. No device job was terminated and no reset was needed.

### 2026-11-13 20:55 UTC — original32 baseline and aligned-state evidence

`readiness_vllm/sampling_capacity32.json` passed on the untouched32 native/common/model path (session44337, normal device close). Its canonical TP4 sampling trace maintained473 program keys before/after capture and probes; greedy unique/ties outputs are saved for the future64 comparison. Auxiliary tracked sampling-only mean latency was0.571ms k1 and0.574ms k20, physical32 rows/3active users/64steps; these are not serving performance.

The repaired diagnostic-only snapshot preparation ran before capture:847→875 program keys, then stable875. Aligned151 replay probe session31632 exited0 with normal device close; `replay_model_state151.json` records identical original prefix, logical ends/page mapping, exact host-versus-device rebuilt model state and preserved inactive slots. Incremental decode versus rebuilt prefill still differs numerically in active conv/recurrent/KV state; strict forced-preemption token parity remains failed and is not relabeled as passed. Independent numerical interpretation is pending.

Launched `bash doc/vllm_integration/run_final_measurements.sh before_sampling64` (repository-relative model prefix implied) as session12782, log `before_sampling64_measurements.log`, tracking disabled. It records original32 P2 plus separate B32/CI/prefill and headline B1 baselines before any native deployment. Queued isolated native validation session85142 waits for its completed-measurements marker. No simultaneous device jobs or heavy CPU builds; no reset.

### 2026-11-13 21:00 UTC — original32 untracked measurements complete

The baseline driver completed both normal server shutdowns and all benchmark collections; final exit2 preserves P2 reasoning ratio failure. Exact full-profile P2: reasoning top_k40/seed42 accepted on explicit host compatibility, 32-user median TPOT122.5ms versus plain16.9ms (7.27x); 1-user41.1 versus17.0ms (2.41x). Agentic effective server defaulttop_k20 measured1.00x even though the installed predictor reports host from omitted fields; actual route proof is separate. Model-card profile skipped because generation_config.json was not readable.

`before_sampling64_measurement_execution/metrics.json` indexes untouched32 baselines with raw paths and launch sources. Primary B1 capacity,128input/128output/1request/concurrency1: TTFTp50/p99=232.763ms (one request), meanTPOT8.674625ms, decode115.278753t/s/u, ITLp50=8.673390ms/p99=8.887775ms, outputthroughput95.892280tok/s. B32 CI100/100/32(default unlimited concurrency): TTFTp50=2613.318591ms/p99=3107.078203ms, meanTPOT26.753224ms, ITLp50=11.132538ms/p99=636.101115ms, aggregate528.018212tok/s. B32serialprefill128/32/32/concurrency32: TTFTp50=2241.985684ms/p99=3223.534519ms, meanTPOT64.669688ms, aggregate207.726511tok/s. These remain original32 comparisons, not final64 headline artifacts.

Queued native validation started after device release: untouched original32 fixture recording, isolated patched default32 exact comparison, then explicit64 suite. Main source/native files remain unchanged until all pass.

### 2026-11-13 21:03 UTC — isolated native trace test preparation failure

Original32 baseline recording passed; isolated default32 comparison passed exactly. Seventeen eager tests passed, including stable ties, tail distribution, mixed k and parameter reload. At21:00:55 the trace fixture failed `Cannot load new binaries during trace capture`; its warm call omitted output_tensor, and program-cache fixture enablement also needs checking. Teardown then encountered the still-open capture (`Event Synchronization is not supported during trace capture`) and remained live. Native source is not deployed.

The1200-second bounded suite process1197444 remains owned by timeout1197442. It is not terminated manually. `timeout180 tools/tt-triage.py --llm-output ...` completed exit0; reports are under `sampling64_trace_failure/`, including warnings about triage halt/resume while teardown is active. CPU AutoFix is preserving the original fixture/log and repairing exact-key warmup/capture cleanup, without suppressing allocation checks. After bounded exit, verify hardware health before the focused rerun.

The completed measurement driver has now been updated for its next invocation to require integrated64 and snapshot native sources plus successful rebuild provenance; earlier measurement artifacts remain original32. `SAMPLING64_REBUILD_PROVENANCE` must identify the successful main rebuild when final tracked/untracked runs start.

### 2026-11-13 21:11 UTC — preliminary stage gate and trace-fixture repair

The preliminary installed09-vllm.check.sh exited2 (`pre_final_gate.log`). Its sole critical is the known baseline P2 reasoning32-user ratio7.27x. Output degeneracy, full-context contract, serving contract and P5accuracy pass; prefill-burst canonical promotion is pending (run artifact exists), render_c07 and diagnostic512-context log references remain advisory. This is not a final gate claim.

`sampling64_validation/TRACE_CAPTURE_FIX.md` identifies exact output-tensor signature mismatch, not an absent historical fixture: cache is enabled by default in this checkout. Corrected test explicitly enables it, warms the supplied-output signature, checks stable program count, ends capture on dispatch exceptions and releases trace on replay failures. CPU AST mock/negative-control checks and targeted precommit pass. Native patch sections remain byte-identical. New wrapper `run_trace_fix_device_tests.sh` writes fresh record/compare/suite evidence under `trace_capture_fixed/`; it must run against untouched main32 before integration. No hardware rerun yet; failed suite still awaits its outer1200s timeout.

Saved CPU tensor diagnostics `replay_model_state151.model_state.pt` (24.652MB) and `replay_greedy_follow50.first_divergence.pt` (1.989MB) are local raw evidence excluded by the repository's existing*.pt ignore rule. They remain intact at the named paths for review; source and compact JSON/log interpretations are stage-owned commit candidates.

### 2026-11-13 21:17 UTC — final evidence tooling prepared

`run_sampling64_adapter_checks.sh` serializes capacity64, adversarial adapter decode, seed and original-boundary tests with tracking enabled and device-owner guards. It is not executed before integration. `snapshot_native_sampling.py` records actual loaded-main library paths/stat identities and copies native source plus successful rebuild provenance; no binary copies. The pending tracked and untracked final drivers invoke it and require `SAMPLING64_REBUILD_PROVENANCE`.

`promote_final_evidence.py` is prepared and syntax-checked, not executed. It validates final64 source/run identity and explicit qualitative/logit reviews, then preserves canonical originals and copies the separate B1 headline, B32 CI/prefill, tracked sampling/qualitative/mixed and P3/P5 evidence with a provenance manifest. It compares source/P2 bytes directly without adding artifact hashes. The legitimate absent-generation-config accepted_model_card skip is permitted; accepted_lm_eval_reasoning must pass.

Recovery check session98089 waits for timeout owner1197442 to exit, refuses any remaining device holder, then runs a90-second bounded TP4 mesh open/add/read/close smoke using unchanged main libraries. No reset or next device job is scheduled before that result.

### 2026-11-13 21:21 UTC — timeout recovery without reset

Original failed native suite session85142 exited124 at its1200s outer deadline. No manual process termination. Queued recovery session98089 then passed TP4 open/add/readback/close under90s; its process exited0 and all device nodes were unowned. `sampling64_trace_failure/post_timeout_mesh_smoke.log` records normal cluster close at21:20:43. No reset was required.

Fresh native validation session3176 starts untouched-main32 recording, isolated default32 comparison and corrected22-case suite via `run_trace_fix_device_tests.sh`, with outer300s per command, tracking1/program-cache checks0skip. Files go to `sampling64_validation/trace_capture_fixed/`; earlier failure and exact compatibility evidence remain unchanged.

### 2026-11-13 21:24 UTC — native64 validated, integrated and rebuilt

Corrected native chain session3176 exited0. Fresh original32 baseline recording and patched default32 comparison passed exactly; explicit64 suite passed22tests/1deselected in4.01s, including warmed trace replay and unchanged cache counts. Normal device teardown completed21:21:20. Native tests emit nanobind interpreter-exit reference warnings (also present in baseline logs); no device holder remains.

Applied the native, common and model integration patches to main. Updated the two negative top-k fixtures from64 to65, keeping rejection semantics. Six actual-common AST host tests and targeted main precommit pass. Main rebuild session23458 succeeded: three unity compiles (9.38/9.83/4.37s), two serialized links (0.31/52.89s), then main-only import provenance pass. Exact output is `sampling64_repair/ornith_integration/main_rebuild_20261113T212253Z/provenance.json`; normal main libraries are deployed. No private serving library path.

Repair evidence was copied into `sampling64_repair/` with one evidence manifest, excluding binaries/objects/kernelcache/assembly. Integrated adapter chain session17014 now checks TP4 capacity64 against original32 greedy references, followed by adversarial adapter decode, seed continuity and original prompt boundary. Tracking remains enabled; final full-model serving evidence remains pending.

### 2026-11-13 21:29 UTC — integrated64 adapter checks pass; final serving started

Capacity64 passes TP4 k40/k64 tail draws, eager/trace/resume equality, seed/history checks and unchanged greedy references against32 (`readiness_vllm/sampling_capacity64.json`). Adversarial adapter decode and seed continuity pass. The original-boundary test initially passed generator-only request_lens through the adapter, duplicating its derived keyword; removed that test argument, with actual-helper CPU regression and negative control. Eleven interface tests and targeted precommit pass. No production change was needed. The focused rerun exits0 and closes the mesh normally; all four final results are indexed in `sampling64_adapter_checks/acceptance.json`. Initial failure is preserved.

Started `run_sampling64_tracked_acceptance.sh` with the successful main rebuild provenance and integrated64 libraries. Full-model tracking-on sampling/qualitative, mixed/lifecycle, P2 device route, P3 and P5 accuracy evidence goes to `readiness_vllm/sampling64_tracked/`. Context262144 and maximum-context acceptance are retained; no repeat capacity sweep. Final timing remains a separate tracking-off run.

### 2026-11-13 21:38 UTC — final64 full sampling profile passes

The current32-layer tracked server passed the installed full sampling profile:67 passed,75 skipped in375.05s. Exact log: `readiness_vllm/sampling64_tracked/sampling_tests.log`. Optional host-logprobs compatibility is explicitly exercised by this suite; it is separate from device-only timing. Current collection proceeds to qualitative and same-server mixed/lifecycle checks.

CPU audit of native/common/model capacity64 found no new blocker in the validated path (`sampling64_repair/CPU_AUDIT.md`). It retains preexisting limits for more-than-capacity equal local maxima and raw primitive callers supplying invalid device-k; model admission rejects unsupported stochastic k and the common layer normalizes its inputs. This does not claim arbitrary-tie parity. Canonical candidate greedy remains the earlier measured strategy, not a newly introduced fallback.

Queued `run_final64_post_acceptance.sh` waits for tracked flow.exit0, then serializes fresh same-code TT qualitative controls, matching B32 standalone logit oracle, and tracking-off final64 measurements. It refuses held devices; the queue creates no acceptance verdict. All new texts and oracle comparisons still require review.

### 2026-11-13 21:43 UTC — coverage-only mixed-shape retry

The first final64 mixed workload completed15 requests×70tokens with exact greedy agreement across device/logprobs controls and return order, six prepared signatures and zero recaptures/releases. It failed only concurrent_decode_during_chunks: simultaneous HTTP submission did not guarantee active decode between continuation chunks on that schedule. This is incomplete coverage, not a passing mixed check. The idle server shut down normally; all original files/execution/launch evidence remain in `sampling64_tracked/first_attempt/`.

The probe now starts one short request, waits for its first streamed token, then admits both long requests while it remains active. It records actual request timestamps and adds an admission-order check; the original device-event concurrency condition is unchanged. CPU scheduling negative control and precommit pass. `resume_sampling64_tracked_acceptance.sh` restarts the same production/native/plugin/config and reruns the affected mixed check before continuing remaining probes. The completed fullsampling/qualitative collection is explicitly reused (`completed_sampling_reuse.json`), not relabeled as results of the new EngineCore or a single combined runner. No production edit occurred.

Read all12 final64 served qualitative outputs. All are coherent/on-topic without repetition/gibberish/language drift/contamination; lengthy prose/code reaches256-token cap. Per-prompt observations are saved with pending fresh TT comparison status in `sampling64_tracked/qualitative_observations.json`.

### 2026-11-13 21:45 UTC — mixed/lifecycle and P2 device routing pass

Retry EngineCore1293243 passes all10 mixed/lifecycle checks with deliberate streamed-short admission before long prompts. Mixed:15 requests×70outputs/concurrency3, starts0/896/1024/2048,355decode calls, six prepared captures and zero new captures/retirements. Lifecycle changes lengths by8 on the same server and passes; all three installed chunked-prefill tests pass29.60s. `page_growth_analysis.json` links automatic logical page crossings, actual page-only reloads/steady steps and exact distinct-request token agreement with non-overlapped logprobs controls; it explicitly does not claim per-request physical-block telemetry.

Tracked installed P2 lm_eval_reasoning users1/32 passes,17.1/17.5ms per token respectively, each1.00x its plain control. `p2_route_audit.json` observes644device decode calls,0host,0newcaptures,0OOV errors. These are tracking-on route results; final full-profile P2/timing remains separate. HTTP logit repeat/reorder passes; the queued standalone comparison remains pending.

### 2026-11-13 — owner stage-cap wrap-up instruction

Owner requested: freeze the best version, run only remaining required gates once, record untried ideas, and close with non-blocking carried items allowed. Implementation is frozen. Current required sequence is already serialized: in-flight P5 accuracy and tracked route audit; fresh same-code TT qualitative controls and standalone logit oracle; tracking-off full P2 plus primary/CI/prefill benchmarks; final installed stage gate and independent stage review. No optional optimization or new experiment is started. Completion requires actual results; pending checks are not counted as passes.

### Carried to next stage

Under the owner stage-cap instruction, untried ideas for the later optimization stage: enable dense batched prefill only after batched-versus-solo proof; investigate canonical prefill/decode numerical sensitivity behind strict forced-preemption trajectory differences; improve arbitrary-many-local-maxima tie selection without regressing measured greedy latency; add per-request physical-block telemetry if a future allocator diagnosis needs it. Existing functional page-growth/control evidence does not claim such telemetry. No performance gain or correctness repair is claimed for these ideas.

### 2026-11-13 22:13 UTC — final64 P5 accuracy passes

All9 installed P5accuracy cases pass at requested8K/32K/128K and depths.1/.5/.9 (actual8166/8167,32734,130898/130899 tokens). Read every raw answer: all exactly match their distinct expected passphrases, finish_reason=stop, no errors or contamination. `sampling64_tracked/p5_text_review.json` records the observations and raw JSONL path. Temperature0 and thinkingoff; final-prefill events confirm device sampling. The long cases remain on the same retry EngineCore1293243 and retain six captures/no retirements. No max-context acceptance rerun or reset occurred.

### 2026-11-13 22:17 UTC — tracked acceptance and standalone controls complete

Tracked flow exits0; idle EngineCore1293243 closes normally, no reset. Benchmark/prefill route audit:261device/0host decode,238steady steps,11inputreloads/12page-refreshes,0newcaptures/OOV. Fresh TT greedy/sampled controls each complete all six256-token prompts on the same production/native source and TP4 chips (B1/context2048 controls versus B32/context262144 serving). All12 texts were read in full and judged coherent/on-topic, without repetition/gibberish/language drift/contamination. HF metadata equality passes; retained HF controls are explicitly reused. Final qualitative verdict is `sampling64_tracked/qualitative_observations.json`.

The matching full32/B32/context262144 standalone logit oracle closes normally and passes all27positions across9 comparison groups, with exact selected tokens/top20 ordering/logprobs and maximumerror0. Twelve HTTP repeat/reorder groups also pass exactly. `logit_verdict.json` records the limited optional full-logit diagnostic scope. Tracking-off final64 measurement driver has started; source remains frozen.

### 2026-11-13 22:22 UTC — final64 measurements and evidence promotion complete

Frozen-source tracking-off measurement flow exits0; B32 and B1 servers shut down normally. Full32/TP4/context262144/block32, greedy explicit settings, random prompts, ignore_eos, no profiling. Primary B1 uses128 input/128 output/1 request/concurrency1; secondary CI B32 uses100/100/32/concurrency unset; serial-prefill B32 uses128/32/32/concurrency32. All requested outputs completed. Full-precision source: `final64_measurement_execution/metrics.json`; latency triples below are mean/P50/P99 in milliseconds (rounded to six decimals).

| Workload | TTFT mean/P50/P99 | TPOT mean/P50/P99 | ITL mean/P50/P99 | Output tokens/s | Decode tokens/s/user |
| --- | --- | --- | --- | ---: | ---: |
| Primary B1 | 236.249337 / 236.249337 / 236.249337 | 8.700078 / 8.700078 / 8.700078 | 8.700078 / 8.705536 / 8.926978 | 95.410489 | 114.941493 |
| CI B32 | 2289.410984 / 2733.208891 / 3253.989109 | 27.122795 / 24.475838 / 35.305579 | 27.122795 / 11.168118 / 619.431904 | 519.415016 | 36.869356 |
| Serial-prefill B32 | 2232.048485 / 2315.043664 / 3360.730757 | 69.003276 / 66.111424 / 92.647742 | 69.003276 / 11.146836 / 824.356172 | 197.497607 | 14.492066 |

Final full-profile P2 passes applicable acceptance and TPOT checks: users1/32 reasoning1.000/0.998, non-reasoning1.000/1.009, agentic1.000/0.996 (threshold1.15). Model-card profile retains the documented missing-generation_config.json skip. Prior original32 failure remains historical.

`readiness_vllm/final_evidence_promotion.json` records successful exact artifact promotion at22:22:06UTC and archive `readiness_vllm/before_final64_promotion`. Sampling/qualitative retain first EngineCore1265789 origin through the explicit reuse manifest; retry EngineCore1293243 supplies remaining tracked evidence. B32 and B1 performance have separate actual launch summaries. Earlier-source maximum-context acceptance and prefix exclusion are preserved. No combined-run fiction or strict forced-preemption parity pass is inferred. Final gate and independent stage review remain pending; no stage-complete verdict is claimed.

### 2026-11-13 22:22 UTC — installed gate advisory and clean device release

`final_stage_gate.log` / `final_stage_gate.exit` record exit1, advisory, with no critical failures/errors. P2 reports13 checks pass/skip; P5 reports4 summary checks pass. Advisories retain P3render_c07 legacy-alias tokenization, historical512-token host diagnostic logs, and no serial_reference for the serial-prefill burst. Batched prefill is disabled and no batched/serial ratio is claimed. `final_process_cleanup.json` passes: all four device nodes have no process owners, no vLLM/EngineCore processes remain, reset_used=false. This supersedes the preceding pending-gate note; independent stage review remains pending.

### 2026-11-13 22:27 UTC — served-default authorization clarification

Independent review identified that the tested `--override-generation-config '{"top_k":20}'` recipe was recorded above as an operator recipe, but no separate explicit owner approval was quoted. This record does not infer that approval from Entry1. The exact proposed decision is to retain the tested default top_k20 for requests omitting top_k; explicit request values remain honored. The pinned checkpoint has no generation_config.json; this default keeps ordinary sampled requests within the validated device sampler capacity64. The final measurements and sampling artifacts already describe this concrete recipe.

Status: **owner decision requested, pending** through the asynchronous question in this thread. The stage-review skill treats a served-default change as an operator decision. No new device run or implementation change is proposed; elapsed time is not approval. Independent review continues, and stage completion remains withheld until its requirements and this decision are resolved.

### 2026-11-13 22:29 UTC — local implementation checkpoint

Committed reviewed stage implementation/tests locally on tt-metal main as `aa3d320cd6f0f1b64c0ddd9a5034c15b0f8b68bf` (parent `d1c644f35289ee46bfa4a23598aa607d5ea96522`). No push. The first normal commit hook formatted only `tests/vllm_replay_numerics_probe.py`; AST equality against the staged original was verified, then the normal full pre-commit suite passed on retry (`stage_code_commit.log`, `stage_code_commit_retry.log`). Production source is unchanged from measured snapshots. This is a preservation checkpoint, not a stage-pass claim while the explicit serving-default decision is pending.

The evidence checkpoint preserves raw JSON/logs/source snapshots byte-for-byte. Only for that artifact commit, formatting/size hooks (`trailing-whitespace,end-of-file-fixer,black,autoflake,isort,clang-format,check-large-files`) are skipped to avoid rewriting measured text/source hashes and to retain requested multi-megabyte raw token/trace artifacts. Production/test source already passed normal hooks in the implementation commit; no global hook configuration is changed. Ignored binary model-state `.pt` files remain at their documented local paths and are not added, nor are weights/build outputs/caches. Raw logs are explicitly added; no logs are compressed.

### 2026-11-13 22:30 UTC — independent review completed

`stage_review_round1.md` returns `more-work-needed` solely for explicit operator acceptance of the tested omitted-top_k20 default. It independently found no concrete current implementation defect, incorrect metric or missing mandatory runtime gate. It verified all26 exact promotion pairs, source identity, raw benchmark metrics, sampling/trace/P5 evidence and actual served/HF/TT texts. Strict forced-preemption parity remains a controlled residual risk, not a demonstrated adapter defect. No further hardware run is requested. After owner acceptance, obtain a fresh diff-only follow-up review of the authorization/status delta; do not rerun completed hardware gates. The owner question remains pending; completion is not claimed.

Evidence-only hook exception: the first evidence commit was rejected by `check-redefined-helpers` on unchanged `FACE_WIDTH`/`FACE_HEIGHT` in native source snapshots under the model documentation/evidence paths. The actual native implementation path passed its normal hook in the code commit. Also skip this hook for the artifact-only retry to preserve the exact loaded source bytes; no source is rewritten and no global hook policy changes. Preserve the rejection and retry logs.

### 2026-11-13 22:32 UTC — local evidence checkpoint and hand-off

Evidence/docs/raw logs committed locally on tt-metal main as `e4746774a229575bd4bcd08c6be7cdba9692cdf6`, after implementation commit `aa3d320cd6f0f1b64c0ddd9a5034c15b0f8b68bf`. Evidence retry passes with the explicitly scoped immutable-artifact hook exceptions above (`stage_evidence_commit_retry.log`). Plugin checkout remains clean at `2a1504554aa7babb1d426dfbca8c7db8c0a795ef`, preceded by owner-approved Entry1 `215effdee4e473f8c50af608d9812fafc56fd48b` and ownership repair `a09b0ef5f600305c738da0482bb76fee0e2e776a`. No repository was pushed.

Only remaining action: record owner response to the pending omitted-top_k20 default question, then fresh diff-only independent review of authorization/status. Accepted code, runtime gates, measurements and cleanup are already complete; do not restart hardware, benchmarks or the full review. Stage is not marked complete while that decision is pending.

### 2026-11-14 — owner accepts the tested serving default

The owner delivered this explicit decision through the multigoal runner:

> Owner decision: the tested default top_k=20 for requests that omit top_k is approved (it matches the model card recommendation). Record this approval beside the serving command, finish the follow-up review and complete the stage.

**Status: approved.** This supersedes the historical pending/blocked decision
entries above, including the sole Round1 Required Work item. Exact approved
argument: `--override-generation-config '{"top_k":20}'`; applies only when the
request omits top_k. Explicit request values remain honored. The exact tested
B32 server command is recorded beside this approval in README.md and as argv
in `serving_default_approval.json`, copied from the actual final64 B32 launch
summary. No source, recipe, or measurement changed. No new runtime experiment
is needed; the follow-up is an independent authorization/status review only.
Existing implementation/evidence checkpoint SHAs remain authoritative; final
checkpoint-log commit is `cc9b199e26521aac5729f7f446e9d7b5088da8e5` on main.
The invoked AutoFix skill requires no new repair because the independently
identified remaining finding was authorization, not a source/runtime defect.

### 2026-11-14 — independent follow-up clean-pass and stage closure

Fresh xhigh round2 review `stage_review_round2.md` returns **clean-pass** with
no Required Work. The exact owner approval and actual serving argv agree;
no implementation, recipe, runtime gate or measurement changed. Previously
controlled concerns and untried ideas remain under `Carried to next stage`.
The sole authorization finding is closed; prior blocked/pending entries are
historical. No further hardware run or whole-stage review was performed.

Closure audit revalidated all26 promotion source/destination pairs byte-for-byte,
the four measured model source snapshots, and the accepted gate exit1 advisory
result. `closure_process_cleanup.json` records a fresh host-only fuser/ps audit:
all four devices unowned, no serving processes, no open/reset. Plugin worktree
remains clean at its recorded local commit. Runtime acceptance and the independent
review now satisfy the stage contract. Final documentation checkpoint follows;
no repository is pushed. The runner may execute its stage check and continue.

Final approval/review/closure checkpoint on tt-metal `main`: `f1a5db0735af7fdded229f51bf3f79358cc33e07`. Normal repository pre-commit hooks passed, with no skips required for these documentation changes. Plugin checkpoint remains `2a1504554aa7babb1d426dfbca8c7db8c0a795ef` on its recorded local branch. No push. This appended SHA record is committed separately as bookkeeping.
