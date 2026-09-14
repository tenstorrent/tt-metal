# Optimized full model work log

Stage 7, Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
Start: 7c4a053822f, clean worktree. Target four Blackhole p300c devices, 1x4 ring.
Enabled installed tt-autodebug verified in Codex config; packaged environment resolved.
Device listing: existing /home/mvasiljevic/tt-metal/python_env/bin/tt-smi -ls --local,
exit 0, four devices. No new dependencies installed.
Historical runtime evidence restored locally from backup/qwen38-stage6-with-artifacts;
restored_artifacts.json.gz lists exact paths. Restored files are excluded from commits.

## Frozen decoder policy

BFP4/LoFi projections, FP32 accumulation/recurrent state; BF16 activations,
residual, norms, CCL; BFP8 paged KV. Stage 5 default and rejection ledger retained.
Residual replicated across TP ranks, L1 width sharded on 40 cores within each
chip for B1; inherited B2..32 layout. This is the selected prior contract, not
an introduced replicated alternative. Full context remains 262144.

## Operation topology audit and candidate plan

| Boundary | Current path | Stage 7 experiment |
| --- | --- | --- |
| Embedding/RoPE | Device lookup, one hidden gather at entry, persistent RoPE indices | Preserve ownership; inspect terminal share and persistent gather |
| 64 layers | Selected TP4 defaults, shared CCL resources, no boundary restore | Preserve decoder policy and compare layer lower bound |
| Final norm | 40-core sharded local RMSNorm | Preserve; sweep head working input grid coherently |
| LM head | Eight vocabulary-local DRAM chunks, BFP8/HiFi2, block10, 2 readers | Geometry/chunk and reader candidates with real head weights |
| Logits/sampler | Local62080 logits, top32 then candidate gathers, greedy k1/p0/T1 | Power-of-two local TopK padding; correct split-greedy vs force-argmax |
| Token delivery | Two nonblocking traces, immediate read per output | Queued no-readback benchmark; defer output delivery while keeping device feedback |
| Cache/request state | Explicit fixed slots/tables/positions, internal4096 chunks | Preserve; rerun changed boundaries, mixed prompts and inactive rows |

No vLLM implementation or broad datatype sweep belongs to this stage.

## Startup recovery

2026-09-12: TP4 ring mesh open failed before model code, exit134:
Device0 active Ethernet core29-25 heartbeat did not change, timed out returning
to base firmware. Firmware19.8.0, KMD2.8.0, all four devices discovered.
Command: PYTHONPATH=.:$PYTHONPATH timeout60 python_env/bin/python, setting
FABRIC_1D_RING then open_mesh_device(MeshShape(1,4), trace_region_size=0).
No watcher or profiler used; bounded reset started after process exit.

Both bounded resets completed exit0; first subsequent list failed Query mappings,
second list showed four healthy devices (recovery_list2.log). Post-reset mesh smoke
exits1 before model code: expected NOC0x1000000000000000, got0x1000000040000000.
The UMD safety check is retained. fuser shows no local owners; each device driver
pids file reports four PID0 entries, suggesting owners outside this PID namespace.
Operator restoration requested; no process killed and no locks removed.

Prepared benchmark_full_model.py separates immediate delivery from queued replay,
with persistent-buffer identity, final-token parity and zero refresh/readback checks.
Syntax compilation passes; no device benchmark result exists yet.
Historical baseline provenance and source equivalence are in historical_baseline.json.
No model implementation change has been made.

AutoFix reports completed: AUTODEBUG_startup.md and AUTOFIX_startup.md.
KMD source confirms PID0 here denotes unrepresentable opener PIDs, not killable
PID0 and not necessarily four distinct processes. Host operator action remains
required. AUTODEBUG_delivery.md preserves the verified host-read boundary and
proposed request-sized device history experiment; exact UINT32 TP4 behavior is
unverified and implementation was not changed speculatively.

Validation performed: python_env/bin/python -m py_compile on benchmark_full_model.py;
python_env/bin/python -m black on that file; bash -n on the stage launcher.
No C++ changes; build not required. No clean-pass review or commit occurred,
because required hardware validation and optimization measurements are outstanding.

## Continuation 1: ownership revalidated

Previous goal turn made progress: recovered discovery, isolated the retained
sysmem mapping and prepared the baseline harness. This continuation read current
KMD pids records again: four PID0 entries/device remain; container fuser finds
no visible owner. This is the second consecutive turn with the same external
hardware blocker. No reset or mesh open repeated while ownership is unresolved.

Prepared tests/probe_token_history.py from AUTODEBUG_delivery.md: UINT32 row-major
TP4 replicated token/history/cursor, traced indexed_fill/copy/plus_one, capacities
1/31/32/33/127/257, two different initial token sets, exact all-device history,
token and cursor comparisons. All persistent buffers are allocated before capture.
This is a data-movement experiment, not model-accuracy evidence. No hardware run
was attempted. Python syntax and black --target-version py312 pass. The experiment
launcher now hashes both new benchmark/probe sources. Generator source is unchanged.

Next after owner recovery: mesh smoke, before_reduced/before_full baseline,
then the token-history probe before adopting deferred output collection.

## Continuation 2: blocked audit threshold reached

Previous turn made concrete progress by preparing the exact UINT32 trace probe;
it did not establish a device result. Current ownership_revalidation.json records
four unresolved opener entries/device, all rendered PID0 in this namespace, and
no container-visible fuser owner. No operator recovery has been confirmed.
The same hardware access condition has persisted for three consecutive goal
turns. AutoFix recovery and source diagnosis are complete; the remaining action
requires the host operator. Marking the goal blocked, not complete. Resume stage7
after host-side owner cleanup or reboot and a passing exact ring TP4 mesh smoke.
No optimization result, clean-pass review, or stage checkpoint commit is claimed.

## 2026-09-13 resume: ownership recurred after verified host recovery

Resumed the same stage, retaining all prepared code and evidence. Current HEAD
is b70863460380bdf57cc0a9c0d5769fe654ac161d, the operator's recovery documentation
commit. PIPELINE_BLOCKERS.md records host telemetry collector cleanup, empty
owner lists, a passing bounded device list and exact ring TP4 MESH_SMOKE_OK.
Its uncommitted follow-up records a respawned collector, supervisor SIGSTOP and
collector termination; that operator-authored edit is preserved.

At 20:26 UTC the driver again reports two PID0 opener records on each chip.
`ownership_revalidation_resumed.json` preserves this read-only check. Both
ordinary and `sudo -n fuser -v /dev/tenstorrent/0 /dev/tenstorrent/1
/dev/tenstorrent/2 /dev/tenstorrent/3` found no visible owner. The source-only
AutoFix investigator rechecked this recurrence: prior cleanup validates the
ownership diagnosis, but the current records do not identify the new owner.
No accessible host process namespace, runtime control socket, or SSH control
connection was found. No hardware command, reset, lock deletion or process kill
was attempted while these foreign owners remained unresolved. Requested host
operator identification/cleanup of current owners, without assuming historical
PIDs can safely be reused.

This is the first audit of the resumed recurrence, distinct from the earlier
three-turn blocked audit. Stage remains incomplete. Prepared benchmark/probe
were inspected; plus_one supports the probe's exact UINT32 row-major token and
cursor ranks by source. No hardware result or optimization gain is inferred.

## 2026-09-13 recovery verified; baseline started

Previous turn yielded evidence of the ownership recurrence and preserved the
operator's intervening recovery. At 20:28 UTC, resumed audit 2 found all four
owner lists empty (`ownership_resumed_audit2.json`). Bounded device listing and
exact current-checkout FABRIC_1D_RING MeshShape(1,4) open/close passed exit0,
MESH_SMOKE_OK (`resumed_list.*`, `resumed_mesh.*`). No reset needed. The blocker
is cleared; do not carry the earlier blocked status into the current stage.

Started the preserved `before_reduced` benchmark with the stage launcher before
implementation changes. Source-only delivery integration design review runs in
parallel; coordinator retains exclusive ownership of all device commands.

## Before measurements

`before_reduced` and `before_full` launcher runs passed exit0. Full64 S128/G128
warmed TTFT86.312ms, immediate-delivery decode39.433t/s/user; queued decode
39.536t/s/user includes final synchronization but excludes final-token check.
Queued counters:127 model and127 sampling replays, no input refresh/readback
or capture. Repeated immediate sequences and queued final token agree.
AIME S203/G100 teacher-forced sampling-inclusive delivery39.300t/s/user,
TTFT85.763ms. No accuracy gate is inferred from the latency-only benchmark.

Started `padded_split_reduced` with common sampler's local power-of-two padding,
physical top32 and semantic greedy k1/p0/T1. Added all-token equality against
unpadded control within the same loaded model, outside the timing windows.

## Terminal candidates and history contract

`padded_split_reduced` passed all-token equality to the unpadded control; queued
433.405t/s vs baseline441.105t/s. Padding is not selected on this current
Blackhole route. `argmax_reduced` passed repeat/final-token checks, queued
240.305t/s; keep split semantic greedy. Current common sampler source asks the
authoritative C++ large-indices route before relaxing stable TopK, so power-of-two
padding alone does not imply a route improvement. Final profile must confirm it.

`token_history_probe` passed exit0 for capacities1/31/32/33/127/128/129/257 on
all four devices: every UINT32 lane, cursor-only shorter reuse with preserved
tail, and nonzero-cursor warm/capture restoration. Persistent addresses stayed
unchanged. This verifies data movement only, not model accuracy. Agent preparing
a minimal deferred-generator patch; coordinator running head-only geometry
experiments against unchanged model/generator implementation in the meantime.

## LM-head geometry and intervening initialization stall

Head-only traced baseline8192/block10/readers2=999.725us.4096/block20/readers2
=1014.728us;16384/block5/readers2=949.427us. All pass active local greedy
and logit PCC against the selected head on the same recorded real-weight
layer0/3 activation. Head activations BF16, weightsBFP8, HiFi2/FP32 unchanged.
The first head harness failed tokenization before inference; fixed by rendering
the template then explicitly extracting input_ids, matching the existing runner.

Readers1 with inherited two-reader tail padding fails PCC.7906657, including
an active-row-only rerun. Source AutoDebug identifies physical bank width20
tiles versus the single-reader factory's19-tile stride. Cache hashing includes
reader count. Reader4 is explicitly unsupported (valid1..3). Reader3 requires
exact reader-dependent padding: the first attempt fails64tiles%3; LCM padding
fails the tail54 !=3*17. Probe now rounds each bank to readers*32 columns.
`AUTODEBUG_head_geometry.md` records the source contract and controls.

The first compatible readers1 run stalled during unchanged model initialization
after LOAD_LAYER3, before MODEL_LOADED and before candidate geometry ran.
Live triage captured under `triage_head_startup/` shows device1 prefetch waiting
for a NoC read inside RELAY_LINEAR and its dispatch waiting for payload. Other
chips await host work; ETH checks pass. GDB attach failed ptraceENOTTY. This is
not a reader-candidate result. Preserved evidence then terminated only owned
PID540646 (launcher exit143); all device owner lists emptied. Bounded reset
passed exit0 (`head_startup_reset.*`). Discovery/mesh recovery logs are
`head_startup_list.*` and `head_startup_mesh.*`. Added timed Python stack dumps
to probe initialization only, disabled before measurement, for a recurrence.

Applied the reviewed deferred-history generator patch after the exact TP4 probe
passed. Sampling appends all32 token lanes to persistent UINT32 history, advances
a device cursor, and defers the final read. First-token TTFT and immediate
callback/compatibility behavior remain. Recording-mode changes invalidate trace
pairs; warm capture restores the history cursor. Added low-level fixed-slot
history/overflow/recapture tests and complete immediate/deferred benchmark
comparison. Syntax/Black pass; generator hardware validation remains pending.

## Selected terminal and deferred generator validation

Exact reader padding fixes the head reader1 PCC failure (>=.99999988), but
reader1 costs1150.478us. Compatible reader3/16384/block10 passes and costs
1174.401us. Reader2/16384/block10 hits L1: static CB end1393664 versus allocation
start1291904,101760-byte overlap. Selected reader2/16384/block5=949.427us
vs8192/block10=999.725us. Updated only head chunk/block constants; dtype,
fidelity, norm/core/residual/CCL policy unchanged.

`deferred_reduced` passes all-token immediate/deferred/repeated equality for
S1/G1,S33/G2,S128/G128,S129/G129,S33/G257,S31/G128, sampled/greedy switches,
callbacks andG0. Its independent warmed decode guard rejects host conversions,
uploads, synchronization and blocking trace replay. Final delivery read timing
and event order pass. Selected reduced S128/G128 immediate444.21t/s versus
deferred447.42t/s; these component results do not replace full-model timings.

`watcher_contract_b3` failed before model loading on instrumented Ethernet
program29104B >26624B. Reused stage6's verified NOINLINE+fabricO3 settings
without disabling Ethernet/assert checks. After reset/list,
`watcher_contract_b3_fit` passes exit0 with TT_METAL_WATCHER=10 and trace
allocation tracking. Includes mixed31/33 prompts, inactive rows, physical page
permutation, changed-only page tables, all32 token-history lanes, history-mode
recapture at nonzero cursor, overflow rejection before replay, and callback
compatibility. See environment/source hash files; no C++ change or build needed.

Full-model prefill remains eager across64 layers. Started reduced
`before_prefill_trace` profile with selected head/history to quantify host
dispatch contribution before adding a bounded generic prefill-trace candidate.
This is full-path optimization work; no decoder policy or capability reduction.

## Intermediate full-model measurement and prefill profile

`after_terminal_full` completed exit0: all64-layer immediate39.545t/s,
deferred39.589t/s (final history read included), queued39.598t/s (final token
check outside timing), teacherforcing39.430t/s. Deferred TTFT83.013ms;
all output tokens equal immediate mode. This is before the prefill trace change.

Reduced `before_prefill_trace` profile completed exit0. Per-device/phase advice
tables and CSVs are under `tracy/before_prefill_trace`. Large raw `.logs` and
`reports` were moved to the TT_METAL_CACHE `optimized_full_model_raw_profiles`
directory; `raw_archive_manifest.json` preserves exact paths, sizes and hashes.
Portable per-device input CSV gzip copies and rendered advice tables remain in
the stage directory. This avoids committing2GB of duplicated raw Tracy events.

`terminal_contract_full_b32` passed exit0 on all64 layers: slots31/0 with logical
lengths31/33, exact page-remapping logits, inactive recurrence/convolution,
all-lane history and nonzero-cursor recapture, overflow guard, explicit host
compatibility control.31+2 versus33 continuation PCC=.99921447 (>=.999).

`argmax_control_terminal` reran the rejected argmax candidate with current
selected head and deferred delivery. All128 outputs equal split semantic greedy.
Deferred242.044t/s on reduced layers, far below selected split447t/s; no
force-argmax default or full-vocabulary all-gather was selected.

## Prefill tracing candidate

Applied `/tmp/qwen38-prefill-trace.patch`: persistent tokens/absolute positions
and padded-logit anchor, model/sample captures first, prefill capture last;
all three traces ordered on CQ0. One cached <=4096-token prompt graph bounds
trace storage; longer prompts retain the full-context chunk path. Public
prefill output ownership/mixedslots/continuation remain eager and independent.
`prefill_trace_smoke` S33/G8 passed repeat equality and allocation tracking.
Tracking adds heavy host overhead (2.52t/s reduced), so those timings are not
performance evidence. A mistakenly tracked benchmark `prefill_trace_reduced`
was interrupted with SIGINT during head upload and closed cleanly (exit1,
KeyboardInterrupt); no candidate measurement was produced. Rerun untracked as
`prefill_trace_reduced_perf` for performance; tracking remains in correctness
probes and watcher. No tracker bypass was added to implementation.

`prefill_trace_reduced_perf` passes all128 output tokens against explicit eager
prefill and immediate/deferred controls. Warmed deferred TTFT4.203ms,
447.125t/s; trace counters show exactly one prefill replay per request and no
steady-state host refresh/readback. `prefill_deferred_reduced` passes the prior
full delivery/guard matrix with cached prefill enabled.

Tested perf-report packet advice through the supported FabricRouterConfig
max_packet_payload_size_bytes override before mesh open. `fabric8192_reduced`
passes and measures450.484t/s versus447.125 default4352, with TTFT4.086ms.
This small reduced gain is pending full-model validation; no default changed.
Payload8192 is aligned and below Blackhole max15232; ring/links/dtypes/layout
are unchanged. Source contract details are in the full-path checklist.

## Full-model prefill and fabric selection

`prefill_trace_full` passes all128 tokens against eager-prefill/immediate controls.
Full deferred TTFT59.687ms,39.583t/s; baseline before_full TTFT86.312ms,
39.433t/s immediate. `fabric8192_full` passes the same controls and measures
TTFT59.725ms,40.381t/s deferred; immediate40.314, queued40.392 and traced
sampling-inclusive teacherforcing40.108t/s. Selected8192 payload (about2.0%
higher full decode than4352). Added `configure_fabric()` in tt/generator.py,
called before mesh open by full-model runners. Benchmark CLI retains explicit
4352/8192 control. No decoder dtype/layout/topology/link policy changed.

A bounded local head LoFi control (`head_16384_block5_lofi`) fails the existing
real-activation PCC>=.999 gate: one shard.99892634, local greedy equal. This
is moderate numerical loss under changed multiply fidelity, not a catastrophic
layout failure. HiFi2 remains selected. A report-only rerun will retain all
four shards and latency without selecting the failed candidate. No broad
full-model dtype frontier is being run.

`head_16384_block5_lofi_report` captures the rejected local fidelity candidate:
942.765us versus selected HiFi2 949.427us; PCC by rank .99971789/.99960959/
.99936688/.99892634, all local greedy equal. Rank3 misses.999, so the
report explicitly says accepted=false. The small component speed difference
is not selected and does not reopen a full-model datatype Pareto search.

`prefill_contract_reduced` passes on selected8192 fabric, with 45 guarded
captures and no program-cache miss during capture, device-only prefill guard,
all-token eager equivalence at31/32/33/4095/4096/4097, persistent all-device
addresses and exact nonblocking replay order, changed prompts/sampling modes,
public output survival across later traced generation, changed/restored page
mapping with exact physical K/V writes, G1/G0. The long4097 request is valid
and uses the intentional chunked prefill path. Final64-layer checks follow.

## Final accuracy, qualitative and capacity gates

`readiness_final` completes exit0 on selected default8192 fabric. Standard
AIME24 chat-template prefill/decode top1=.99/.98 and both top5=top100=1.0,
unchanged from Stage6. Fresh autoregressive artifacts are under autoregressive/.
Shared six-prompt256-token and four-prompt1024-token extensions completed.
Prompts0/1/3/4/5 reach EOS with coherent haiku/explanation/laws/translation/code;
the generated Fibonacci function passes n=-1/0/1/2/8. Both TT and HF story
controls reach the1024-token budget; a separate longer TT story completion
check is pending, explicitly not a same-budget HF comparison. The standard
degeneracy metrics are clean (`qualitative_metrics.json`).

All64 layers pass S262143/G2 with finalposition262144 in79.065s and
S262144/G1 in78.231s. No context/alignment reduction. Updated nested Stage7
context contract includes4MiB prefill persistent reserve, selected8192 payload,
final model/generator hashes and current full-capacity evidence.

Runner check: MODEL_DIR=models/autoports/qwen_qwen3_8_27b with Python venv PATH,
packaged prompts/model_bringup_multigoal/07-optimized-full-model.check.sh ->
runner_check.log, exit0. This does not replace the independent stage review.

`contract_final_full_b32` passes the entire mixed-slot contract on selected
8192 fabric; continuation PCC=.99921447. `prefill_contract_full` passes
all64 layers at1/31/32/33/4095/4096/4097,50 guarded captures, exact eager
oracles, changed prompts/modes, persistent identities, retained outputs and
physical page-table remapping. These cover the optimized public non-aligned
path in addition to the maximum-context runs.

`watcher_prefill_final` passes exit0 on all4 chips, selected8192 fabric,
WATCHER10 + NOINLINE1 + FABRIC_OPT_LEVEL=O3 + TRACE_ALLOC_TRACKING1, cache
root tt-metal-cache-full-model-watcher. It runs S33 plus changed-input/mode,
public-output, physical-page, G1/G0 and lifetime tests. No ETH disable flag or
assert suppression. Watcher is stopped/mesh closed before `profile_final`.

`profile_final` captured the selected prefill/fabric path with the earlier
reduced runner's G2 geometry (cache160, history1). Device collection and CSV
export pass. That is valid component evidence but understates the production
history allocation and differs in page-table geometry. Updated profile-only
setup to reserve S+127 cache entries and127 history rows, matching S128/G128
benchmark allocation; final matched profile will be `profile_final_buffers`.
Only representative real layers0/3 are profiled. This change does not alter
model/generator behavior. `after_full_final --full --qualitative-story` measures
the selected default before an extra2048-token story completion probe.

`after_full_final` selected default completes exit0: S128/G128 deferred
TTFT59.295ms,40.385t/s; immediate TTFT58.852ms,40.328t/s; queued40.394t/s;
S203/G100 sampling-inclusive teacherforcing TTFT77.233ms,40.238t/s.
All128 output tokens match eager-prefill and immediate controls. Full no-host
steady counters and one final history read pass.

The extra story probe returns coherent EOS-complete text at1700 tokens under
a2048 budget. Its first1024 tokens exactly equal the earlier1024-token TT
story, so this was a budget cutoff rather than a replay/feedback stall. HF's
same-prompt1024 control also truncates; the longer TT result is completion
coverage, not an identical-budget HF comparison. Final qualitative metrics
show all six prompts complete, no degeneration, and executable Fibonacci
examples pass. Reproducer: python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/check_qualitative_artifacts.py
--doc models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model; outputs qualitative_metrics.json/.log, exit0.

## Final matched profile and host verification

`profile_final_buffers` exits0 with cache256/page-table8/history127, matching
S128/G128 allocations. All4-device prefill/model/sample/token-out CSVs and
`tt-perf-report` advice tables were rendered using `tests/full_model_profile_tables.py`.
Portable compressed device op CSVs remain in each profile directory. Raw
multi-GB Tracy streams and duplicate uncompressed CSVs are archived under
`/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_full_model_raw_profiles/`;
each local `raw_archive_manifest.json` records paths, sizes and hashes.

Final accounting command (repository root):

```bash
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/summarize_full_model_perf.py \
  --profile-dir models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model/tracy/profile_final_buffers \
  --benchmark-json models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model/after_full_final.json \
  --output models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model/perf_summary.json
```

Runtime source/library hashes match final benchmark. The48/16 kernel-only
stack estimate22.315–22.446ms plus surrounding kernels yields23.756–23.884ms,
within3.7–4.2% of measured24.762ms/token. The gap-inclusive estimate overshoots
and is not host overhead. Read-only mandatory-byte lower bound7.671ms is not
an attainable end-to-end target. Native sampler~.320ms is1.3% of full decode.
Prefill gap falls to.731–.735ms from3.518–3.568ms on representative layers.
Full64 remains unprofiled per the optimization skill.

`precommit_code.log`: all authored stage Python/shell hooks passed, exit0.
Python-only changes require no C++ build. Context checker with
`--stage optimized-full-model --require-contract --strict-caps` and the packaged
`07-optimized-full-model.check.sh` both exit0 (`context_check.log`, `runner_check.log`).
The qualitative checker exits0 with all six completions and functional code
checks passing. Final independent stage review and local commit records follow.

## Independent review metadata correction

The reviewer found the2048-budget qualitative artifact inherited HF's
`executed_steps=1024`. Corrected harness metadata to distinguish
`hf_control_executed_steps` from TT steps. Every row's existing
`perf.decode_tokens+1` independently proves the TT executed budget; generated
tokens, text and timing are unchanged. `qualitative_metadata_correction.json`
records before/after artifact hashes. Historical hardware source manifests
remain intact; this is a host-only provenance fix, with refreshed qualitative
checker hashes and gates.

Final host verification: `precommit_final.log/.exit_status` passes all authored
stage Python/shell/docs hooks. `compileall`, launcher `bash -n`, authored-source
`git diff --check`, and local link checks pass. The portable perf summarizer's
11 host regression/negative cases plus3 format/syntax checks pass on its final
source hash (`perf_summary_host_validation.json`, embedded reproducible driver).
The final allocation comparison is derived from raw SDPA/page/history shapes:
cache256/page-table[1,8]/history127 matches the benchmark; the intermediate
cache160/history1 profile remains explicitly different. `final_validation_index.json`
records accepted artifacts/hashes and final runtime sources. No runtime source
or hardware measurement changed during these final host metadata refinements.

## Independent stage review and local checkpoint

Fresh xhigh agent `/root/stage7_independent_review`, no conversation fork,
reviewed the original goal and selected skills against raw code/artifacts.
`stage_review.md` returns **clean-pass**, no required technical work. The
reviewer independently reproduced all4-device accounting, verified runtime
hashes and trace/cache contracts, and inspected actual TT/HF text and generated
code. The qualitative step-count metadata finding was fixed and rechecked
before this verdict. No runtime source changed after review.

Stage-owned code, compact reports, CSVs/gz/provenance and restored context
contract are checkpointed in `tt-metal`, branch
`mvasiljevic/qwen38-full-bringup`. The operator-owned `PIPELINE_BLOCKERS.md`
change is excluded and preserved. No vLLM repo was touched; no push is performed.
The next provenance-only commit records the implementation checkpoint SHA.

The first checkpoint attempt was rejected by whitespace hooks because raw
logs/report text contain original trailing spaces and EOF formatting. Restored
all hook-mutated evidence byte-for-byte from the reviewed index; no source
change occurred. Authored code/docs already pass these hooks. The artifact
checkpoint uses `SKIP=trailing-whitespace,end-of-file-fixer` for these two
formatting hooks only, retaining raw evidence hashes and running every other
commit hook. This is not a skipped correctness or hardware gate.

Implementation/evidence checkpoint: `36e086487d5b6d3e4ab3815a7657bc52a28ba76e` in repository
`/home/mvasiljevic/qwen38-full-rerun/tt-metal`, branch
`mvasiljevic/qwen38-full-bringup`. Commit command:
`SKIP=trailing-whitespace,end-of-file-fixer git commit -m 'Optimize Qwen3.8 full-model TP4 generation and tracing'`
exits0; all other commit hooks pass. Exact hook/commit output is preserved at
`/tmp/qwen38-stage7-checkpoint.log` and copied to the stage runner log directory.
The following docs-only provenance commit changes only this log; its SHA is
recorded in the external stage checkpoint log and final handoff. No push.
