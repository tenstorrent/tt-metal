# Stage 9 vLLM integration work log

Target Qwen/Qwen3.8-27B, pinned HF revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
Started from completed datatype_sweep stage with selected head_bfp4_lofi policy.
Context contract is 262144, no reduction. Four Blackhole p300c chips, TP4 Ring1x4.
Enabled installed model-bringup and autodebug plugins verified in Codex config;
package startup environment loaded via ../run-env.sh. Installed packages remain read-only.
Initial vLLM HEAD 03fa3af2e15b5f8dc07cbaa67d92f979aa00be11, clean worktree.
The pre-existing PIPELINE_BLOCKERS.md operator change is not stage-owned.

Status: work in progress. No final serving, sampling, quality, async, benchmark,
review pass or checkpoint commit is claimed.

## Environment and reduced integration

Bounded /home/mvasiljevic/tt-metal/python_env/bin/tt-smi -ls --local exit0:
all four p300c devices visible. Current-checkout Ring1x4 mesh open/close exit0:
startup_mesh.log. No reset needed. No profiler or watcher used in serving.

Shared runner command is tests/run_vllm_stage.sh, using installed readiness_check.
Reduced loop: QWEN_VLLM_TEST_LAYERS=0,3 bash tests/run_vllm_stage.sh --stages serve
--sampling-profile smoke (paths relative to model directory shown for readability;
actual commands invoke the script from repository root).
This executes one real layer of each kind and the unchanged full embedding/norm/head/sampler.
max_num_seqs=4, max_model_len=262144, block_size=32, P300x2,
trace_region_size=134217728, FABRIC_1D_RING, trace_mode=decode_only,
sample_on_device_mode=all. Async capability remains false pending probes.

Initial launch failures occurred before hardware open: malformed shlex quoting
of --hf-overrides, then unprefixed custom architecture. Raw logs:
reduced_launch_quoting_failure.log and reduced_architecture_failure.log.
Corrected command specifies TTQwen38ForCausalLM directly. Registration is in
vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models.

## Cache and sampling contracts under test

Adapter delegates prefill/decode to QwenGenerator, uses its canonical sampler
and persistent tt_out_tok, and binds the vLLM allocated ModelCache by identity.
Paged pool has max_model_len total tokens; this supports one long or multiple
short requests. Constant-size recurrent/conv state covers every serving slot.
Canonical generator helpers perform device-only recurrent resets/remaps.
No attention pages are cleared on a new request. Host fallback requires explicit
QWEN_VLLM_HOST_COMPATIBILITY=1; normal token-out path does not read full logits.

The page-growth AutoDebug investigation found current vLLM always passes its
updated host page table; adapter forwards it independently of reset_batch.
Canonical _refresh_table compares values and skips unchanged device writes.
Host tokens/positions are omitted on steady device-sampled decode.
Focused host regression and allocator-driven runtime controls remain required.

2026-09-14 00:24 UTC: reduced server reached API readiness. Final registration
uses explicit TTQwen38ForCausalLM architecture override; its inspection-only GPU
protocol methods reject execution, while TT plugin uses low-level APIs. A trial
through original multimodal architecture failed processor-factory inspection and
was removed (reduced_multimodal_failure.log). No image capability is advertised.

First S31/G3 request executed prefill, then failed at plugin submit_prefill:
expected (output, rope_deltas). HF uses M-RoPE metadata, so text-only adapter now
returns per-request zero offsets. Evidence: reduced_rope_contract_failure.log.
Initial request collector lost JSON on missing usage; collector now preserves
error responses before its completion assertion.

Next launch failed before model construction at llrt return-to-base firmware /
Ethernet heartbeat. Preserved startup_heartbeat_failure.log and triage/ output.
Runner/API exited; stale EngineCore634395 received SIGTERM and became defunct;
all KMD device-owner lists then empty. Bounded list/reset/list recovery underway.
This is infrastructure recovery, not model correctness evidence.
Recovery list/reset/list all exited0; second reset not required. Initial reused
mesh_smoke helper was 1x1 only (recovery_mesh.log); exact Ring1x4 smoke then
passed independently (recovery_mesh_tp4.log). No locks manually cleared.
Triage wrapper returned0 but dependent checks failed; triage capture is limited,
not a clean device verdict. The successful mesh smoke is the recovery evidence.
S31/G3 retry exposed the scalar RoPE offset type: plugin calls .item(), so
list[int] is insufficient. AUTODEBUG/AUTOFIX_prefill_contract investigation and
host regression are in progress. Captured reduced_rope_type_failure.log and
reduced_request31.json. Runner/API/EngineCore exited without a visible live owner.
Transient foreign PID0 device owners were left untouched; existing host supervisor
logs show collector cleanup at 00:30:22; owner lists were empty at 00:30:57.
A bounded reset precedes retry after the failed device process.

Sampling contract audit: native canonical sampler supports k<=32. Shared tests
include k100, penalties and host-only parameters. These must use an explicit
optional compatibility boundary; silently clamping k100 or ignoring penalties
is not accepted. Canonical greedy/default sampled performance remains native.

2026-09-14 00:34 UTC: focused prefill contract repair changed only the returned
RoPE offsets from list[int] to CPU int64 tensor [N]. Actual runner host tests
reproduced the exact .item() failure before the change and passed afterward
(5 tests, including one/two prompt device output, host logits, chunk and fixed
slot/page mapping, and cache identity). Black check passed. See
AUTODEBUG_prefill_contract.md, AUTOFIX_prefill_contract.md, and
prefill_host_before.log / prefill_host_after.log. Original S31/G3 server runtime
verification remains with the parent; this host pass makes no device claim.

Reduced adapter device control passed (adapter_device.json/.log): batch4,
context262144, 8192 physical pages, prompts31/45/27/53, two active and two
nonzero inactive requests, 68 steps each reset control and stale-host steady path.
Five real page-growth events, drained reorder at step36, exact active token-stream
parity, preserved inactive states, retained async host output snapshots. Steady
steps have zero token/position/RoPE/seed refreshes. Not a real vLLM allocator test.
Capability supports_async_decode enabled only after that proof.

Reduced live async server then passed S31/G3 and lifecycle S31/33/127/129/31/33,
G70 per request, followed by concurrent distinct S31/45/53/67/G70. Artifacts:
reduced_request31.json, reduced_lifecycle.json, reduced_concurrent.json and logs.
Server ran async_scheduling=True, sample_on_device_mode=all; raw server log
archived as reduced_async_server.log after SIGINT to runner639939; runner exit0,
KMD owners empty. These are reduced correctness controls, not model quality/perf.
Live non-overlapped control now uses --no-async-scheduling plus trace allocation
tracking to investigate allocator warnings without profiler/watcher collection.

Shared vLLM worker now accepts optional fabric_max_packet_payload_size_bytes.
8192 matches the selected full-model fabric configuration; default behavior for
other models is preserved. Reduced successful server logged 8192 at mesh open.
Worker already allocates max_num_seqs extra blocks (8196 at max_num_seqs4), so
the null/reserved block does not reduce the advertised262144 capacity.

2026-09-14 00:47 UTC: source-only repeated-startup AutoTriage/AutoFix report found
a verified worker lifecycle gap: framework shutdown inherited a no-op; mesh
close existed only in __del__, whose partial-init AttributeError path could skip
it. Explicit TTWorker.shutdown now drains existing async steps, invokes optional
model.close while mesh is alive, and always attempts mesh close; destructor is
an idempotent best-effort fallback. Adjacent vLLM host regression reproduced
5 failures/1 pass before repair and passed all6 after, without TTNN imports.
See AUTOTRIAGE_repeated_startup.md, AUTOFIX_repeated_startup.md and
worker_shutdown_host_before.log / worker_shutdown_host_after.log. No hardware
verification by this investigator. Native atexit already attempts RISC cleanup;
normal successful runner exit did not use SIGKILL. Telemetry cleanup at00:37:24
falls between UMD close and failed reopen, so heartbeat causality is unresolved.
Parent owns adapter close forwarding and explicit serve/stop/reopen verification.

Nonoverlap serving reached ready but exposed a distinct read boundary:
model_runner passes raw device token output directly to process_decode_output_host
when async_read=False. The adapter previously expected already-read host tokens.
Raw failure: nonoverlap_read_failure.log; output JSON records incomplete requests.
Fix uses the same one-replica 32-token read when storage is still device-side;
already-read async results perform host formatting only. Updated direct device
control will reproduce that exact synchronous plugin boundary before parity check.

The prior direct probe's reset control pre-read outputs and therefore did not
cover this plugin boundary. Its async feedback/page-growth evidence remains valid,
but it is not used to claim the failed nonoverlap server passed.

2026-09-14 00:54 UTC: final direct adapter probe adapter_device_final.json passed raw-device synchronous control vs stale-host async path, exact streams, changed page maps, inactive state and owned snapshots under allocation tracker. Actual nonoverlap server concurrent S31/45/53/67 G70 also passed, exact text equality to async reduced_concurrent.json; reduced_concurrent_control_final.json preserves responses. Repeated S31/33/127/129/31/33 G70 passed (reduced_warm_final.json). Reduced smoke then 2pass/1skip/1failure: unseen packed sample_prefill concat/tilize program buffers remained alive across an existing decode trace. preserved packed_sampling_trace_failure.log/reduced_sampling_before.log. Repair warms new packed-logit signatures after releasing old traces, retaining repeated-shape traces. Explicit shutdown logged both closing/closed markers even after this failure; exact Ring1x4/8192 reopen succeeded WITHOUT reset (explicit_shutdown_reopen.log). Initial probe invocation used invalid FabricRouterConfig constructor kwargs and failed before opening; corrected attribute assignment succeeded.

2026-09-14 01:03 UTC: optional host async TypeError fixed after fresh AUTODEBUG_host_async.md and focused real-plugin submit/finalize red/green tests (six pass). Native minimal token transfer unchanged. Targeted min_p now passes async (min_p_host_fix.log). Final reduced shared smoke passes ALL FOUR tests, including previously skipped all-vocabulary chat logprobs after adding --max-logprobs -1. reduced_sampling_final_runner.log and archived reduced_sampling_final.log are the runner evidence. Native S31/45/53/67 G70 concurrent requests after host compatibility pass and exactly match original native streams (reduced_native_after_compat.json). Allocation tracking remains enabled; packed sampling trace guard no longer trips this sequence. Host suite22pass (adapter_host_final.log); initial collection lacked sibling vLLM PYTHONPATH and was corrected, no dependency install.

2026-09-14 01:06 UTC: normal reduced server SIGINT shutdown logged both explicit close markers at01:04:39 and exited0. Foreign telemetry owners then reappeared as PID0 in container owner files during full32 handoff. SIGINT sent to new runner652239 during initialization had no effect (serve-only signal handler is installed after readiness); model load continued. Supervisor repeatedly terminated host collector669520, then new server674913/collector674930 at01:06:20; owner files now contain only our EngineCore652449, with no foreign0 entries. Full32 mesh startup itself succeeded without reset. Continued loading final model after checking ownership. Preserved normal-close log reduced_final_server.log.

2026-09-14 01:20 UTC: startup-cancellation diagnosis corrects the earlier
"SIGINT had no effect" inference. full32_server_runner.log proves SIGINT raised
KeyboardInterrupt during readiness polling; API-only cleanup timed out and
killed API652301, leaving EngineCore652449 loading through layer63. Coordinator
observed PPID1/device ownership and separately sent TERM. The stage wrapper now
execs vllm_process_guard.py, which assigns a unique inherited launch marker,
handles EXIT/INT/TERM before readiness, allows runner cleanup, then signals only
marked owned survivors with TERM and a bounded wait. It reports remaining PIDs
without force-killing device processes. Installed readiness package is unchanged.
Seven host tests passed, including actual packaged-cleanup orphan reproduction,
ordinary/cancelled cleanup, unrelated sentinel preservation, zombie reaping,
PID reuse protection, and TERM-ignoring survivor reporting. No hardware used by
investigator. See AUTODEBUG_startup_cancel.md and startup_cancel_host.log.
AUTOFIX_repeated_startup.md now includes normal and exceptional explicit worker
close plus successful reopen without reset; prior heartbeat causality is still
unresolved. Signal the printed VLLM_STAGE_GUARD PID on subsequent launches.

2026-09-14 01:09 UTC correction to the01:06 handoff note: SIGINT DID stop runner652239/API652301; it left startup EngineCore652449 orphaned, which completed all64 layers but never served an API. Thus that load is NOT successful final serving. Ownership audit found PPID1; parent sent SIGTERM to this exact orphan at01:08:34. It exited (zombie awaitingPID1reap), deviceowners empty. full32_cancelled_startup.log preserves evidence. New AutoFix investigation targets cancellation cleanup in the stage launchwrapper; installed runner package remains read-only. Retrying full32 from emptyowner state with direct same-configuration sharedrunner command (full32_live_runner.log). No reset needed so far.

Reduced full-vocab normalized-logit probe passed all four sequential/shuffled-batch comparisons with maxabs difference0 (reduced_logit_determinism.json). Diagnostic script initial errors were tokenizer length248077 versus configured vocabulary248320 and new transformers BatchEncoding default; fixed by config.get_text_config().vocab_size and rendered-text encoding, then reran successfully. This is reduced control only; full-model equivalent and standalone comparison remain required.

2026-09-14 07:42 UTC: resumed SAME stage after external Codex capacity failure at01:28:02 (root AUTODEBUG.md is operator-owned infrastructure diagnosis). Initial continuation inspected stale tails before process audit; those messages are not evidence of ongoing test progress. Preserved partial52/73 sampling log as full_sampling_interrupted.log (no complete-suite pass), server as full_server_interrupted.log, and runner as full_checks_interrupted.log. No live server/deviceowner remained. Bounded tt-smi list and exactRing1x4/8192 mesh reopen passed (resume_device_list.log, resume_mesh_smoke.log), no reset required. Hostoperator reserved telemetry by pausing kubelet; this agent does not modify that reservation. Operator also sanitized priorstagehistory; currentHEAD125d29dda46, preserved ourstagecode. RootAUTODEBUG.md and PIPELINE_BLOCKERS.md edits are unrelated and will not be staged.

Restarted full64 server maxseq32 using tested processguard, same selected policy/context/cache/async settings, explicit optionalhostcompatibility1, and normal nanobind trace replay WITHOUT allocation-tracker instrumentation. Earlier reducedtrackerfullsmoke and partialfulltrackedcases remain trace-safety evidence. Finalfullsamplingprofile must complete anew; finalperformance will use uninstrumented replay. Command script run_vllm_stage.sh --stages serve --sampling-profile full with QWEN_VLLM_MAX_SEQS=32; resumed_full32_runner.log. PrecommitPython passed; vLLMruff formatted two payload-config statements (whitespaceonly), then allvLLMhooks passed in precommit_vllm_final.log.

### 2026-09-14 07:54–07:58 UTC: final-profile findings

The resumed full-profile suite is live (guard726608, EngineCore726802; attached
check guard727483). At52% it reported mixed-parameter reproducibility failure;
logprob checks and subsequent seed controls passed so far. This is not a full
profile pass. A fresh AutoFix/AutoDebug investigator is tracing backend and
seed-state behavior; final traceback is pending suite completion.

Independent review found the reduced active-token parity oracle was weak:
all active streams collapse to token220. Its structural table/position, inactive
state, and async snapshot ownership checks remain useful, but do not by themselves
prove request-sensitive page growth. Added a targeted full-model concurrent chat
comparison against logprob-forced synchronous greedy controls; execution pending.

The operator's sanitized checkpoint omitted four earlier raw JSON controls. Restored
local ignored copies directly from existing historical commit3f48cac393a via
`git show <sha>:<path>`: datatype-sweep `tt_qualitative.json`,
`tt_qualitative_extended.json`, `tt_qualitative_story_2048.json`, and
`selected_token_out.json`. They are historical evidence, not new reruns and not
staged for commit. Rendered serving prompt metadata is now in
`readiness_vllm/qualitative_prompt_format.json` (shared prompt lengths60/67/75/61/66/62).

### 2026-09-14 08:03–08:09 UTC: reproducibility diagnosis and native correction

Resumed full sampling completed **72 passed, 1 failed in940.84s**. The only failure
was `test_mixed_params_batch`: legacy `Letter: `, temperature0.01, seed42,
returned`201` versus`197`. Preserved exact original log as
`doc/vllm_integration/full_sampling_mixed_failure.log`. Runner exited1 and skipped
qualitative/benchmarks; attached guard cleanup completed without stopping serving.
A direct unchanged-node rerun passed22.47s (`readiness_vllm/mixed_targeted_original.log`).
Diagnostic same-body requests with explicit `logprobs=0` all used the existing
host backend and passed22.31s (`mixed_host_control.log/.jsonl`; diagnostic hook
`readiness_vllm/host_control_plugin.py`, not part of final canonical pytest suite).
These results establish intermittent reproduction, not a complete gate pass.

`AUTODEBUG_mixed_sampling.md` corrects its initial inference: absent top_k inherits
HFk20 and therefore can use native sampling. A separate native seed rewind was
proven by actual adapter source with a fake device seed tensor: batch reset,
row remap, and companion-parameter changes rewound continuing requests. Fixed
explicit seeds to normalized-request-seed plus absolute output position only on
authoritative refresh; parameter-only updates preserve advancing device seeds.
Generator `set_batch_sampling_params(seed=None)` updates policy without copying
seeds. Steady decode still performs no seed/token/position refresh. Fixed mapping
reserves the whole262144 context below signed-int32 overflow/manual_seed sentinel.
See `AUTOFIX_native_sampling.md`, seed continuity before/after logs.

Added explicit `QWEN_VLLM_HOST_COMPATIBILITY=all` for the canonical full compatibility
suite: all requests use the plugin's existing host sampler so changing neighbors
cannot change RNG backend. Value1 retains selective unsupported-request fallback;
unset is native performance mode. This does not claim cross-backend RNG equivalence.
Eleven CPU seed/mode tests pass; final native runtime controls remain required.

Before restarting, full64 native async versus logprob-forced synchronous greedy
control passed all12 streams for four distinct chat prompts, S67/69/66/69,G100.
Each crosses positions96/128/160. Outputs were read: Moon phases, ordered bread
preparation, bicycle gears, and botanist/blue-flower story remain distinct and on
request. `full_concurrent_control_before_seed_fix.json/.log` records exact output
matches, request metadata and cache counters. This remedies the reduced active
stream sensitivity gap for the loaded pre-fix source; rerun final code separately.
Full-vocabulary normalized prefill logprobs also reproduce exactly across repeated
and reordered concurrent requests (all four max differences0) in
`full_logit_before_seed_fix.json/.log`. Vectors are copied to persistent ignored
`readiness_vllm/full_logprob_vectors_before_seed_fix.npz`.

Sent SIGINT only to owned guard726608 after clients completed. Guard reports
cleanup complete; EngineCore/API/runner disappeared and all four /proc device
owner lists are empty. Archived pre-fix source/run manifest and server log under
`doc/vllm_integration/full32_before_seed_fix_*`. Nanobind reference-leak shutdown
warnings remain observable; no process/device owner remains. No reset or foreign
process termination was needed.

### 2026-09-14 08:10–08:20 UTC: final native controls and benchmark CLI repair

Final native-capable B32 server started without reset: guard736236/API736303/
EngineCore736451, `QWEN_VLLM_HOST_COMPATIBILITY=1`, no layer subset or allocation
tracker. Exact command/environment/source hashes are in
`readiness_vllm/full32_run_config.json`; runner log is
`doc/vllm_integration/final_native32_runner.log`. All39 host adapter tests pass
(16 subtests), including seed/mode, cache, async and prefill boundaries.

`native_seed_continuity.json/.log` passes all6 exact-token-ID comparisons:
A(S67,G100), B(S69,G47) alone; concurrent B(G33)+A(G100), then A(G100)+B(G47).
All use native k5,T0.7,p0.9,seeds42/43. Saved `native_seed_server.log` has zero
explicit-host markers before later logprob diagnostics. HTTP ordering does not
force physical row placement; CPU real-adapter tests separately force remaps.
Final `full_concurrent_control.json/.log` passes both full text and token IDs for
all12 streams (four distinct prompts,S67/69/66/69,G100), native async versus
logprob-forced synchronous greedy and reordered native repeats. The serving
`kv_cache_usage_perc` is0.0 before/after. Final full-vocabulary prefill logit
probe `full_logit_determinism.json/.log` again has all four max differences0.
Standalone comparison remains pending. Raw vectors stay in ignored persistent
`readiness_vllm/full_logprob_vectors.npz`.

The first benchmark client failed before any HTTP generation request because
`importlib.metadata.version("vllm")` requires wheel metadata absent from this
source-only environment. Saved `benchmark_metadata_failure.log` and runner log.
Attempted normal `VLLM_TARGET_DEVICE=empty python setup.py egg_info` (metadata
only, no compilation), but installed build helper `setuptools_scm` is absent.
No dependencies were installed. A five-line CLI fix catches only
`PackageNotFoundError` and uses the package's existing `vllm.__version__` fallback;
installed-distribution version behavior is preserved. `python -m
vllm.entrypoints.cli.main --version` now exits0 with`dev` (`vllm_cli_version.log`).
Shared benchmark retry runs against the same healthy B32 server. This CLI parsing
change adds no model/decode work and will be included in the sibling vLLM commit.

### 2026-09-14 08:21–08:31 UTC: CI, final trace probe and standalone comparison

Shared native B32 benchmark retry exits0; no server restart was needed for the
CLI metadata correction. CI S100/G100/N32, max sequences32, unbounded burst,
T0, ignore_eos, zero warmups:32 completed/3200 output tokens; TTFT P50/P99
6450.5041/6451.9435ms; TPOT P50/mean/P99 200.6061/201.6713/224.1289ms;
ITL P50/P99 186.7862/638.7489ms; aggregate121.61558tokens/s;
TPOT-derived4.9585635tokens/s/user. Raw/config/log are
`readiness_vllm/vllm_ci_serving_{result,benchmark}.json` and
`vllm_ci_serving_benchmark.log`. vLLM chunked prefill is disabled in the logged
engine config, and S100 is below the internal4096 chunk limit; this measured
burst includes admission/first-use costs and fixedB32 decode, not chunked prefill.

Archived cold B32 single-user S128/G128/N1 control as
`vllm_batch32_single_user_{result,benchmark}.json` and `_benchmark.log`:
TTFT623.1237ms; mean TPOT224.0304ms; ITL P50/P99212.2996/213.2588ms;
output4.402364tokens/s; TPOT-derived4.463679tokens/s/user. This is not the headline.
The installed CLI defaults both ready-check timeout and warmup count to0, so its
message 'initial single prompt test' did not warm the shape. Final B1 primary
will record one unmeasured same-shape warmup via `--num-warmups 1` and still measure
exactly one128/128 request. No performance instrumentation is enabled.

Sent INT only to native guard736236 after clients exited. Cleanup completed,
API/EngineCore disappeared, and /proc owner lists were empty. Final native logs
and source/config manifest are archived as `final_native32_*`.

After shutdown, with compatibility unset and allocation tracking enabled:
`python -m models.autoports.qwen_qwen3_8_27b.tests.check_vllm_adapter --output
models/autoports/qwen_qwen3_8_27b/doc/vllm_integration/adapter_device_after_seed_fix.json`
passes68 steps on layers0/3, then closes devices. Steady counters are68model and
sampling replays,68reads,2token/position/RoPE/seed refreshes,6changed-page copies;
inactive states stay equal and14distinct snapshots remain independently owned.
This is structural evidence supplemented by the diverse full-model control.

With tracker unset, `check_vllm_logit_determinism.py --standalone-control
readiness_vllm/full_logit_determinism.json --vectors
readiness_vllm/full_logprob_vectors.npz --output
readiness_vllm/standalone_logit_control.json` (full model-root paths in executed
command) passes: both selected all64-layer prompt vectors match serving exactly
in explicitly swapped slots0/1 with standalone cache capacity128. All four
maximum logprob differences are0. Mesh closes and owners are empty.

Started final canonical compatibility profile:
`QWEN_VLLM_MAX_SEQS=32 QWEN_VLLM_HOST_COMPATIBILITY=all bash
models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages serve,sampling
--sampling-profile full`. Guard740207/API740274/EngineCore740381;
`readiness_vllm/full_host_profile_run_config.json` records exact commands/environment/
hashes, and `full_host_profile_runner.log` records the shared launch/check/shutdown
flow. Sampling is still running at this entry; do not treat it as a pass yet.

### Final full sampling gate, 2026-09-14 08:46 UTC

`QWEN_VLLM_MAX_SEQS=32 QWEN_VLLM_HOST_COMPATIBILITY=all bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages serve,sampling --sampling-profile full` exited 0: **73 passed, 3 warnings in 973.88s**. This explicit optional host compatibility profile covers the canonical shared suite; native quality/performance and native seed/page-growth controls are separate. Archived `full_host_profile_server.log`, `full_host_profile_sampling.log`, and `full_host_profile_run_config.json` under this doc directory. Guard reported cleanup complete; guard/runner/API/EngineCore/pytest PIDs are gone and all four device-owner directories are empty. No reset was needed.

### Final native B1 quality and performance, 2026-09-14 08:47–08:54 UTC

Launched `QWEN_VLLM_MAX_SEQS=1 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages serve --sampling-profile full` with host compatibility, test layer subset and allocation tracking unset. Guard742972, API743039, Engine743187; launch `ce18f5fe603e43079b0fe5b7ced67866`. All64 layers use the selected policy, max-model-len262144, block32, max-num-seqs1, TP4 ring, payload8192, trace region134217728, trace_mode decode_only, sample_on_device_mode all; prefix caching disabled. `readiness_vllm/primary_run_config.json` contains exact expanded commands, source hashes, environment and workload. Archived doc `final_native1_server.log`, `final_native1_run_config.json`, `final_native1_runner.log`.

Attached shared runner `--stages qualitative --server-url http://localhost:8000 --sampling-profile full`: exit0; all6 chat prompts in greedy and sampled mode saved to `readiness_vllm/vllm_qualitative_outputs.json`, console `final_qualitative_runner.log`. Read all12 texts. Then `python models/autoports/qwen_qwen3_8_27b/tests/check_vllm_qualitative_extended.py --output models/autoports/qwen_qwen3_8_27b/readiness_vllm/qualitative_extended.json`: exit0; all8 extensions stop, greedy/sample token counts for IDs0/2/3/5 are476/139,1174/392,268/246,400/331. Read all8 full texts. Four extended greedy prefixes match shared outputs exactly. Matched prompt metadata to HF/selected-TT stage8 controls; prior initial TT text exact only for IDs3/4, wording differs for0/1/2/5. No general token-equality claim. All3 inspected Fibonacci functions pass restricted host checks, including negative/zero/small cases. `qualitative_review.md` records each verdict: coherent/on-topic, no mechanical looping, gibberish, wrong-language drift or request contamination. Sampled haiku5/7/4 is retained as a meter limitation, comparable priorselectedTT5/7/6; greedyserving5/7/5. Raw reasoning is template/control-consistent. Extended sampledseed42 controls are new requests, not continuations of unseeded shared samples. Exact comparisons/code results: `readiness_vllm/qualitative_control_comparison.json`.

Attached shared runner `--stages benchmark --server-url http://localhost:8000 --sampling-profile full --no-benchmark-ci-serving --additional-benchmark-args='--num-warmups 1'`: exit0. Native greedy S128/G128/N1,maxseq1,concurrency1,ignoreEOS; one unmeasured same-shape warmup. Completed1/1 and128/128outputtokens. TTFT P50/P9985.8247019351ms; TPOT P50/mean/P9924.2099810069ms; ITL P50/P9924.1904864088/25.3316571563ms; aggregate output40.4966017257tokens/s; `1000/meanTPOT`=41.3052781708tokens/s/user. Raw `readiness_vllm/vllm_result.json`, normalized `vllm_benchmark.json`, console `vllm_benchmark.log`, runner `primary_benchmark_runner.log`. TTFT/TPOT P99 are single-request observations. Secondary finalnativeB32 CI S100/G100/N32 remains in `vllm_ci_serving_benchmark.json`: TTFT P50/P996450.5041/6451.9435ms, TPOT P50/mean/P99200.6061/201.6713/224.1289ms, ITL P50/P99186.7862/638.7489ms, output121.61558tokens/s, TPOTderived4.95856tokens/s/user. CI has unbounded burst admission, no warmup, and chunkedprefillFalse; not headline decode speed.

Prior teacher-forcing B1/S203/G10040.878tokens/s,76.607msTTFT is only a decoder/generator latency lower-bound reference. Canonical B1/S128/G128 split-token reference41.0955tokens/s,58.4179msTTFT (no per-token reads), or41.0750tokens/s,58.9856msTTFT (deferred complete delivery), is a closer shape but a different harness/output boundary. Native serving decode is within about0.6%; no speedup claim. Audited steady path has no host fallback, full logits, blocking replay, unchanged-page copies or host token feedback. No avoidable serving-specific decode overhead remains identified.

### Final gates and cleanup, 2026-09-14 08:54–08:56 UTC

Native B1 server log has zero explicit-host markers, all64 layer-load records, and post-check cache occupancy0 (`native1_final_metrics.txt`). SIGINT sent only to owning guard742972. Held server exits130 intentionally; guard cleanup complete, worker logs mesh closing/closed, all four guard/runner/API/EngineCore PIDs gone. `readiness_vllm/final_process_audit.json` verifies empty owner files and no remaining vLLM API/EngineCore processes, including after final mesh smoke.

`timeout 60 /home/mvasiljevic/tt-metal/python_env/bin/tt-smi -ls --local` exits0 with all four p300c devices (`final_device_list.log`). Under sourced run environment, `timeout 60 python` constructs FabricRouterConfig with max_packet_payload_size_bytes8192, sets FABRIC_1D_RING, opens MeshShape(1,4) with trace_region_size0, then closes it: exit0, MESH_SMOKE_OK (`final_mesh_smoke.log`). No reset, watcher or profiler was needed. Nanobind teardown reference-leak warnings remain visible; actual process/device cleanup is verified. Generic active-trace allocation warning is explicitly scoped in README against transient allocation ownership, the repaired persistent packed-prefill buffer defect, tracker-enabled representative tests and partial tracked all-layer evidence; the final73 suite is not claimed an exhaustive all-shape allocator audit.

With sourced environment, `MODEL_DIR=models/autoports/qwen_qwen3_8_27b HF_MODEL=Qwen/Qwen3.8-27B bash "$TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/09-vllm.check.sh"` exits0. This runs the required vLLM degeneracy check (no findings) and context-contract gate (target=supported262144); `stage_check.log`. Both repo `git diff --check` pass. Python/doc changes require no C++ build; relevant host/device/shared-serving checks and precommit results are retained. Final raw artifact hashes: `readiness_vllm/final_artifact_manifest.json`.

Final stage-owned source/docs `pre-commit run --files ...` exits0 (`precommit_closure.log`). Measured generator/adapter/plugin/CLI source SHA256 values still match `primary_run_config.json`; no implementation changed after final hardware runs.

### Independent review and local checkpoint

The fresh xhigh `$stage-review` subagent returns **clean-pass**, no required work (`stage_review.md`). It independently read all serving outputs, compared controls and token streams, recomputed metrics, checked source/artifact hashes, and reviewed cache/trace/sampling/shutdown changes. Findings were fixed or resolved with controlled evidence and rereviewed; limits are retained in its anomaly ledger. Local stage-owned commits follow this review; no push is authorized or performed.
