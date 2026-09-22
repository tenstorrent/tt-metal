# Stage 10 optimized vLLM work log

Qwen/Qwen3.8-27B, HF revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
Start from completed stage9 and selected head_bfp4_lofi datatype policy.
Context262144, TP4 Blackhole P300x2 ring1x4, payload8192, trace region134217728,
block32, sample_on_device_mode all; no capability reduction. Installed plugins
are enabled and startup uses ../run-env.sh. No profiler collection is permitted.

Pre-existing operator-owned PIPELINE_BLOCKERS.md and root AUTODEBUG.md are excluded.
Initial source hashes and repo SHAs: before/source_manifest.json. Stage9 artifacts
are copied under before/ before the runner overwrites shared readiness paths.
Four p300c devices visible in bounded tt-smi list; no stale serving processes.

## Baseline and optimization target

Stage9 primary native S128/G128/N1,max sequences 1: TTFT85.8247ms, TPOT24.20998ms,
decode 41.3053tokens/s/user. These are historical controls pending fresh baseline.
Comparable standalone token-out S128/G128/B1:41.0955tokens/s, TTFT58.4179ms.
Decode already uses nonblocking split traces and device token/position/seed advance.
Serving prefill dispatch and eager first-token sampling are the identified gaps.

Fresh baseline command (native environment; no host compatibility/test layers):
`QWEN_VLLM_MAX_SEQS=1 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages serve,benchmark --sampling-profile full --no-benchmark-ci-serving --additional-benchmark-args='--num-warmups 1'`
Log: before_primary_runner.log. Before/after workload and TT config must match.

## Operation topology audit

| Boundary | Existing sequence | Decision / constraint |
| --- | --- | --- |
| Decode model | Persistent token/position/RoPE/page inputs;64 selected layers; final norm; vocab-sharded head | Preserve stage8 selected math, layouts and collectives; no measured decode deficit |
| Decode sampling | Local tile-shaped TopK32; gather candidates; semantic greedy k1,p0,temp1; device token feedback | Preserve canonical split trace; no force-argmax/full-logit fallback |
| Serving prefill | Host upload; eager chunked model; concat/pad; eager canonical sampling | Investigate reuse of existing full-model coordinated prefill trace with safe persistent outputs |
| Scheduler refresh | Compare page contents every step; copy only changes; reset token/position at authoritative state changes | Preserve allocator-growth correctness and pending-token drain contract |
| Host boundary | One replica32 UINT32 tokens deferred read; event; host formatting | Preserve plugin-required boundary and existing owned pending snapshots |

Independent AutoFix diagnosis of prefill trace lifecycle requested; no implementation
change until the hypothesis and discriminating check are established.

## Fresh baseline results

Primary native S128/G128/N1,max sequences 1,concurrency1,one warmup: completed1/1;
TTFT P50 / P99  81.373372ms; TPOT mean / P99  24.200183ms; ITL P50 / P99
24.193656/24.872318ms; aggregate40.568990tokens/s; decode 41.322002tokens/s/user.
Raw/normalized/logs: before/primary_vllm_result.json, primary_vllm_benchmark.json,
primary_vllm_benchmark.log, primary_server.log. Runner exits0; guard cleanup complete.

CI native S100/G100/N32,max sequences 32,unbounded burst,zero warmup: completed32/32;
TTFT P50 / P99  3287.203267/3288.339370ms; TPOT mean / P99
201.800399/221.782478ms; ITL P50 / P99  186.791386/647.941967ms; aggregate
138.090364tokens/s; burst-affected TPOT-derived4.955392tokens/s/user.
Command: `QWEN_VLLM_MAX_SEQS=32 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages serve,benchmark --sampling-profile full`.
Before CI is preceded by the runner's default B32 single-request S128/G128 check,
which is separate cold capacity evidence. Preserve that ordering for after CI.
Raw artifacts: before/batch32_vllm_ci_serving_{result,benchmark}.json and server.log.
Runner exits0, guard cleanup complete. No profiler, watcher, tracking, host
compatibility or layer-subset environment is present in these baselines.

Before CPU serving-contract suite:39 tests pass. Exact command/log:
before_host_tests.log; unittest modules test_vllm_adapter_host, test_vllm_host_async,
test_vllm_prefill_host, test_vllm_seed_continuity_host, test_prefill_sampling_trace_host.

AutoFix source diagnosis: AUTODEBUG_prefill.md. Independent investigation confirms
eager first-token dispatch and the need to keep persistent output ownership
separate from public independently owned logits. Implementation hypothesis uses
a generator-owned bounded serving helper and a coordinated fourth sampling trace;
standalone3trace behavior is preserved. Device A/B remains required.

## Reduced trace verification

New tests/check_vllm_prefill_tracing.py compares actual adapter requests to a
generator-owned eager prefill + identical common device sampler control. No host
argmax oracle is used. Exact per-shard prefill logits,4token streams, changed
prompt logits, physical page remaps, device positions/RoPE/seeds and stable4trace
identities pass at B1 lengths31/32/33/127/128/129/4095/4096/4097, and B4 single-active
slot0 at31/33/4097. Reduced layers0/3, full context 262144, actual external
cache shape/pages/dtypes and production trace_region_size134217728.

Commands: source ../run-env.sh; prefix `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.check_vllm_prefill_tracing`, then:
- `--lengths 31,33 --output .../doc/optimized_vllm/prefill_reduced_short.json`
- `--lengths 32,127,128,129,4095,4096,4097 --output .../doc/optimized_vllm/prefill_reduced_boundaries.json`
- `--batch 4 --lengths 31,33,4097 --output .../doc/optimized_vllm/prefill_reduced_b4.json`
Each exits0 and closes the mesh. Corresponding .log files retained. These are
correctness probes, with diagnostic host logits reads; not serving metrics.

Warm eligible requests explicitly forbid eager model.prefill, _sampling_step and
_capture calls. Each uses1prefill+1prefill-sampling trace replay and3decode pairs;
page copies0 for unchanged mapping. Input tokens upload once per new prompt,
decode token/position/RoPE once per authoritative request boundary; no steady
refreshes. New prompt/page cases match a fresh eager control exactly.

AutoFix report: AUTOFIX_prefill.md.23 focused host tests pass; larger inherited
host suite and all 64 serving checks still required.

Primary candidate uses the exact before command with one warmup; log
after_primary_runner.log. No instrumented environment is carried into serving.

## First full-model candidate

Same baseline command, native full 64-layer S128/G128/N1,max sequences 1,one warmup, exits0.
Candidate TTFT63.250856ms vs81.373372ms before (22.27% lower); TPOT24.202883ms
vs24.200183ms; decode 41.317391tokens/s/user vs41.322002; ITL P50 / P99
24.182097/24.563808ms; aggregate40.799059tokens/s. Raw JSON/server logs and source
hashes: candidate_primary/. This remains candidate evidence pending final default.
Runtime counters:254model+254sampling decode replays,1cold prefill+1prefill replay,
1cold prefill sample+1prefill-sampling replay,1four-trace capture,2token/position/RoPE
refreshes,10changed-page copies,256minimal token reads; no full-logit reads. The
measured request follows exactly one same-shaped warmup, proving traced prefill
and traced first-token sampling at the measured boundary. Decode unchanged.

Follow-up required: warmed long/multirow/nonzero-slot serving prefill must also
reuse sampling traces. AutoFix extends only generator-owned fallback preparation,
staging packed logits into one persistent input, dropping every temporary before
replay. Public independently owned logits/sampling APIs keep their prior behavior.
No new sampler or host fallback is introduced.

## Final generator correctness

Final fallback uses one cache-bound padded logits input and releases every public
output/packing temporary before sampling replay. It retains eager chunked model
prefill for long/multirow/nonzero-slot requests, with warmed first-token sampling
traced. One-time warmup/recapture remains necessary for new program signatures.

`fallback_reduced.json/.log` exits0 with B4 S31/33/4097 and two rows31/45 in
slots1/3. All warm samplers are guarded against eager calls and recapture; exact
logits/tokens, changed page mapping and zero unchanged-page copies pass.

`watcher_prefill.json/.log` repeats B4 S33/4097 plus rows31/45 on real layers0/3:
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_FABRIC_OPT_LEVEL=O3
TT_METAL_CACHE=/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache-full-model-watcher
TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m
models.autoports.qwen_qwen3_8_27b.tests.check_vllm_prefill_tracing --batch 4 --multi
--lengths 33,4097 --output models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/watcher_prefill.json`.
All four chips checked; no ETH disablement or profiler. Exit0, mesh closes cleanly.

Full 64-layer selected weights, B1/context 262144, production128MiB trace region:
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1
python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.check_vllm_prefill_tracing
--full --lengths 31,33,128,129,4095,4096,4097
--output models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/prefill_full.json`.
All seven exact eager-versus-traced logit/token comparisons pass, including
changed prompt tokens and reversed physical page tables. Repeated requests keep
trace/buffer identities and copy no unchanged page table. Warm4097 sampling is
traced with no recapture; its model prefill intentionally uses existing chunks.
Exit0 and clean mesh close; corresponding `.log` retained.

`TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m
models.autoports.qwen_qwen3_8_27b.tests.check_vllm_adapter --output
models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/adapter_final.json` exits0.
The68-step reduced B4 test compares synchronous/async output, page growth,
changed tokens/current positions, request remapping, pending output ownership
and inactive state. Its200MB correctness trace reservation is separate from
the128MiB production proof above. No profiler; mesh closes cleanly.

Final host command: `source ../run-env.sh; PYTHONPATH="$VLLM_ROOT:$PYTHONPATH"
python_env/bin/python -m unittest models.autoports.qwen_qwen3_8_27b.tests.test_vllm_adapter_host
models.autoports.qwen_qwen3_8_27b.tests.test_vllm_host_async
models.autoports.qwen_qwen3_8_27b.tests.test_vllm_prefill_host
models.autoports.qwen_qwen3_8_27b.tests.test_vllm_seed_continuity_host
models.autoports.qwen_qwen3_8_27b.tests.test_prefill_sampling_trace_host
models.autoports.qwen_qwen3_8_27b.tests.test_serving_prefill_trace_host`.
All 59 pass (`final_host_tests.log`). First run caught two missing-method errors
in the seed test fake after the API change; adding the matching fake entry point
resolved them without changing seed assertions.

## Final primary reproduction

Final default native command is identical to the fresh baseline, including one
same-shaped warmup. `final_primary_runner.log` exits0 and guard cleanup completes.
`after/primary_vllm_{result,benchmark}.json`, benchmark.log and server.log preserve
the final raw evidence; `after/source_hashes.json` records the final runtime.
S128/G128/N1,max sequences 1,concurrency1: TTFT P50 / P99 62.954747ms, TPOT mean / P99
24.205248ms, ITL P50 / P99 24.186334/24.595602ms, aggregate40.799661tokens/s,
TPOT-derived41.313354tokens/s/user; all 128 output tokens delivered. This is the
final result, reproducing the earlier candidate. TTFT falls 22.6347%; decode
changes−0.0209%, effectively unchanged for this single-request measurement.
P99 TTFT/TPOT equal the sole measured request, not population-tail estimates.

Native server reports TTScheduler async_scheduling=True, sample_on_device_mode
all, context 262144 and head_bfp4_lofi. Counters across warmup+measurement show
254 model/254 sampling decode replays, one cold prefill/sample and one replay
of each, two token/position/RoPE refreshes, four seed refreshes, ten changed-page
copies and256 minimal token reads. No full-logits readback or host-compatibility
marker appears. Sampling is eager only for the unmeasured first-use warmup.

## Warning classification

Native primary server warnings about an uninstalled vLLM version module, absent
CUDA/Triton driver, VLLM_ROOT, disabled Inductor and the custom scheduler reflect
this source checkout's TT plugin environment. The plugin activates and serves
successfully; exact sibling SHA is recorded instead of an unavailable package
version. The motherboard discovery uses bus IDs as tray IDs; four-chip topology
is independently checked. Scheduler max-batched-tokens is raised to262144, not
a context reduction. Generation-config defaults are printed, while benchmark
commands explicitly set temperature0.0 and native greedy uses k1.

L1 semaphore fragmentation is an allocation advisory, with no failed allocation
in final runs. The generic untracked live-trace allocation warning is investigated
with trace allocation tracking in full 64-layer boundary and reduced Watcher probes.
Those checks require temporary destruction before replay and verify exact
outputs/stable identities. The warning itself is not proof of correctness.
No stale-input, allocator-tracker violation, watcher assertion or device error
was observed in those completed probes. No resets or profiler collection occurred.

## Final CI serving burst

`final_ci_runner.log` repeats the exact baseline B32 command, including the
preceding cold B32 S128/G128/N1 request and zero explicit burst warmups. Exit0
and guard cleanup complete. `after/batch32_vllm_ci_serving_{result,benchmark}.json`
and `.log`, plus `after/batch32_server.log`, preserve results. Benchmark command
arrays and normalized workload configs match before exactly.

CI S100/G100/N32,max sequences 32,unbounded burst:32/32 complete,3200 output tokens;
TTFT P50 / P99 3372.122240/3373.813229ms; TPOT mean / P99
200.827626/221.324565ms; ITL P50 / P99 186.892141/609.087726ms;
aggregate138.169955tokens/s. TTFT median is2.58% higher, aggregate0.058% higher;
this cold capacity run does not establish a burst speedup. Its burst-affected
TPOT-derived4.979395tokens/s/user is secondary, never the headline decode rate.

Counters across the cold B32 single request and burst:226 model/226 sampling
decode replays, two model-prefill cold calls, three first-use sampling calls,
one fallback persistent-input allocation/copy, two sampling-trace captures and
eleven page changes. No per-token eager decode sampling or host logits reads.
The matched cold CI workload necessarily includes first-use prefill/sampling
warmup; its counters do not claim the warmed primary's prefill replay pattern.

Both before and after B32 server logs report nanobind object-reference warnings
at interpreter shutdown, after mesh close. These reproduce in the unchanged
baseline and do not leave a server/EngineCore process. Final process audit is
recorded after all checks; this stage does not claim to repair nanobind teardown.

Reviewer found a stale datatype confirmation link. The exact
`doc/datatype_sweep/selected_confirmation.json` was restored from local stage8
object3f48cac393a, as stage9 had restored historical quality controls.
`restored_evidence.json` records full source SHA and byte hash. This is inherited
evidence, not a new run. It records actual runtime layer tensors and compute
policy; `propagation_check.json` alone is only a summary. Final native policy
logs and unchanged model/decoder source tie the selected configuration to serving.

## Native qualitative gate

Held server: `QWEN_VLLM_MAX_SEQS=1 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages serve`
(log native_quality_server_runner.log). Native flags unset; full 64-layer/context 262144.
Attached shared suite: `QWEN_VLLM_MAX_SEQS=1 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages qualitative --server-url http://localhost:8000 --sampling-profile full`
(log qualitative_runner.log), exit0. Then
`python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.check_vllm_qualitative_extended --output models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/qualitative_extended.json`
(log qualitative_extended.log), exit0. All 8 extensions reachEOS.

All 12 shared and8 extended outputs read completely. Six shared greedy texts and
all 8 extended seeded/greedy texts and token IDs exactly match the Stage9 controls
in before/. Fresh unseeded shared samples differ coherently. Prompt source/template
hashes and all rendered prompt IDs rechecked; logical lengths60/67/75/61/66/62.
The common degeneracy checker passes all 20 completions. Explicit packaged command:
`python_env/bin/python -m readiness_check.check_degenerate_output models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/vllm_qualitative_outputs.json --scope vllm --json models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/qualitative_degeneracy.json`.
Restricted host execution tests inspected Fibonacci functions at−1/0/1/2/8/10;
all pass. `qualitative_control_comparison.json` preserves exact comparisons and
checker/code results. `qualitative_review.md` records all outputs and the exact
prior sampled-haiku5/7/4 control; this model-quality limitation is unchanged.
SIGINT sent only to the owned guard, exit130 as expected for held serving; mesh
closes and cleanup completes. Full server log copied to native_quality_server.log.

The independent reviewer verified the restored325099-byte confirmation against
its original git blob: all 64 runtime layers record256 BFP4 uploaded tensors and
320 LoFi/FP32-accumulation compute configs, matching the selected policy. The
stale precision citation finding is resolved. The reviewer also independently
read all 20 native qualitative outputs and checked exact prior controls. Final
review remains pending native B32 controls, full sampling and process cleanup.

## Full-model native concurrency controls

Held correctness server: `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 QWEN_VLLM_HOST_COMPATIBILITY=1 QWEN_VLLM_MAX_SEQS=32 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages serve`.
Production128MiB trace reservation, all 64 layers/context 262144, no profiler.
Explicit compatibility1 permits only requested logprob controls to use the
existing synchronous host sampler; supported requests remain native. This is
not a performance run and is separate from native-only metric/quality servers.

`python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.check_vllm_concurrent_control --output models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/concurrent_control.json`
passes all 12 exact token/text comparisons: native concurrent, synchronous
logprob control and reordered native. Four distinct prompts have logical
lengths67/69/66/69, generate100 tokens and cross positions96/128/160.
Read all four native streams; they are request-sensitive coherent100-token
prefixes. They are overlap/state controls, not completed-answer quality evidence.
Exact comparison proves all synchronous/reordered streams match.

`python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.check_vllm_seed_continuity --output models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/seed_continuity.json`
passes6/6 exact seeded native continuations. Top-k5, temperature0.7, top-p0.9,
seeds42/43, S67/69 with G100 versus companion G33/47; requests are tested alone,
with reordered admission and after companion departure. Streams are coherent
request-specific prefixes, not complete-answer quality judgments.

`python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.vllm_lifecycle --output models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/lifecycle.json`
passes all six sequential HTTP requests S31/33/127/129/31/33, G70 each.
The cold first-use trace cost is intentionally outside the warmed primary
optimization claim: the matched B32 cold S128/G128/N1 control has
TTFT 521.109ms before versus567.871ms after, with the additional prefill/sample
capture/staging path. This is a recorded cold-start tradeoff, not evidence of
a warmed decode regression or an isolated timing attribution.

`benchmark_audit.json` checks exact before/after launch configuration equality,
actual async scheduling/device sampling, zero native host markers/full-logit
reads, paired decode replay counters, and byte-identical model, multichip decoder,
optimized decoder, precision loader and selected config versus the initial SHA.

The same lifecycle command with `--concurrent --output models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm/lifecycle_concurrent.json` also passes6/6,420tokens.
Both lifecycle artifacts report running0/waiting0/KV-cache usage0 after requests.
The full 64-layer tracked server completed every concurrency/seed/lifecycle request
without a trace allocation violation. SIGINT only to owned guard gives expected
exit130, clean mesh close and cleanup complete. Final log archived as
native32_control_server.log; runner provenance native32_control_server_runner.log.

Full73 shared sampling suite launch (separate explicit compatibility server):
`QWEN_VLLM_HOST_COMPATIBILITY=all QWEN_VLLM_MAX_SEQS=32 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages serve,sampling --sampling-profile full`.
Log full_sampling_runner.log. This intentionally uses the completed integration's
existing all-host compatibility mode for broad penalties/logprobs/top-k tests;
it is not the native performance or native stochastic-sampling evidence.
The native benchmarks, quality and B32 controls above exercise the optimized
device path. Profiler, Watcher and allocation tracking are unset for this suite.

Independent B32 review rederived all 12 raw exact concurrency comparisons, six
seeded continuation prefixes and12 lifecycle responses. Native32 counters show
1123 model versus1024 sampling replays:99 synchronous control decode steps;
103 full-logit reads equal those99 decode plus four explicit control prefills.
These are confined to the labeled logprob controls, not the native metric path.
Both lifecycle idle gauges are zero; API RSS increases131072B sequentially and
is unchanged across the concurrent batch. This is a bounded lifecycle check,
not a general leak-free allocator claim. No tracker failure, clean mesh close
and owned-guard cleanup were independently verified.

Repository artifact policy (`models/autoports/qwen_qwen3_8_27b/.gitignore`) keeps
runtime JSON/log output local and commits source, tests and compact Markdown
reviews. Stage checkpoints follow that policy; raw evidence and its hashes stay
in this workspace. The already-tracked context_contract.json remains tracked.

## Rejected options and scope decisions

| Option | Evidence and disposition |
| --- | --- |
| Capture sampling over independently owned public logits in the adapter | Rejected by the AutoDebug ownership analysis: caller-retained tensors can outlive trace scratch allocation. AutoFix tests weak-reference destruction at replay; selected generator helper stages into persistent storage and destroys temporaries. |
| Stop after optimizing only short single-row prefill sampling | Initial narrow candidate improved primary TTFT, but left warmed long/multirow sampling eager. Extended fallback staging/sampling traces, then proved S4097 and mixed-slot exact results under tracking and Watcher. |
| Replace canonical split sampling with force-argmax or a generic greedy sampler | No new serving strategy accepted. Completed full-model sampler controls in ../optimized_full_model/full_path_checklist.md show split faster than force-argmax on its reduced real-layer workload; the final serving path reuses split and matches full-model decode speed. This is inherited component evidence, not a new serving microbenchmark. |
| Keep a trace for every prompt length | Not selected: one bounded retained shape follows the existing full-model ownership/capacity contract. Changed shapes may pay cold capture cost; no unbounded persistent allocation is introduced. |
| Lower context or require aligned prompts | Forbidden by the goal and unnecessary with existing capacity headroom. Full64 exact nonaligned tails and unchanged context contract validate the selected path. |
| Retune decoder geometry/dtype/CCL to explain serving overhead | No decode deficit remains in the measured primary regime; byte-identical selected math already reproduces full-model speed. Stage10 removes serving prefill dispatch and preserves completed precision/topology evidence. No new geometry family is claimed rejected by this experiment. |
| Use CI burst as headline decode or collect a serving profiler | Rejected by the goal's measurement contract. CI remains secondary capacity; no serving device-time/roofline claim is made. |

Cold first-use work and shape-dependent recapture remain explicit tradeoffs.
The selected final default reproduces the warmed candidate improvement; it is
not presented as the fastest possible implementation for every prompt mix.

## Final sampling and cleanup gates

Full sampling suite exits0:73 passed, no failed/skipped/xfail cases,962.82seconds.
Artifacts full_sampling_tests.log, full_sampling_runner.log and full_server.log.
Three pytest warnings are the source checkout's missing vllm._version and SWIG
deprecations. The server closes its mesh, then the owning guard completes cleanup.
Full compatibility is explicitly all-host; native supported sampling, quality,
async overlap and benchmarks were verified separately above.

final_process_audit.json scans all nine recorded launch identities and live
vLLM/EngineCore/runner process names: no leftovers. After every serving job had
closed, `timeout 60 /home/mvasiljevic/tt-metal/python_env/bin/tt-smi -ls --local`
exits0 and lists all four Blackhole p300c devices (final_device_health.log).
No reset was required. Source hashes remain identical to final primary timing.
The sibling vllm checkout remains clean at5dfd818f4f0f5444533331d85f4711c43f8f2f2b.

All experiment gates are complete. Final independent review and local checkpoint
commits are the remaining closure steps; no push is authorized or performed.

## Independent stage review and checkpoint

Fresh xhigh reviewer `/root/serving_review` returns **clean-pass**, no required
work (`stage_review.md`). It independently rederived all73 sampling passes,
89 immutable artifact hashes/sizes, final runtime and unchanged-math hashes,
matched benchmark launch configs, generated outputs, native stateful controls,
all nine closed launch identities and a fresh empty process scan. Interim
missing/pending findings were fixed or completed and rereviewed.

Final pre-commit passes (precommit_final.log); its first invocation removed
trailing whitespace in this work log, then the rerun passed. All runtime hashes
remain equal to the final primary measurement, and git diff --check is clean.
Local checkpoint below isolates stage-owned files. Operator PIPELINE_BLOCKERS.md
and root AUTODEBUG.md remain untouched and excluded. Raw runtime evidence stays
local as required by the autoport ignore policy. No push is performed.

| Repository | Branch | Local checkpoint |
| --- | --- | --- |
| tt-metal | `mvasiljevic/qwen38-full-bringup` | `c901b7d879ca124559e557d05dc2b659ae4a0ed9` — Stage10 implementation/tests/reports; all commit hooks pass |
| vllm | `mvasiljevic/qwen38-full-bringup` | `5dfd818f4f0f5444533331d85f4711c43f8f2f2b` — unchanged completed integration; no new commit needed |

The checkpoint contains15 stage-owned source/test/report files. Unrelated
operator files remain dirty and unstaged. `local_commits.json` records this
checkpoint and the following documentation-only provenance commit; no push.
