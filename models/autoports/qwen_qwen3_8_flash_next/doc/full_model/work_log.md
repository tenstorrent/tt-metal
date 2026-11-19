# Full-model stage work log

Stage 6 is in progress, not accepted. Start: 2026-11-13 22:02 UTC.
Repository `/home/ttuser/qwen38-flash/tt-metal`, branch `main`, starting HEAD
`2589dcde` (accepted optimized multichip decoder `d103c86fc35f59f9d10f1e73daef969f92ffe258`).
No vLLM implementation, installation or serving work is authorized in this stage.
No push. Pre-existing untracked fused-decoder artifacts are unrelated and preserved.

Skills: installed tt-model-bringup 0.1.24 full-model, tt-device-usage,
tt-enable-tracing, qualitative-check, multichip, optimize and stage-review;
tt-autodebug 0.1.8 enabled in the active Codex config. Environment is
`source python_env/bin/activate; source /home/ttuser/qwen38-flash/lane-env.sh`.
The installed plugin remains read-only. Hardware jobs are serialized, bounded,
and use the complete 1x4 Blackhole mesh, devices 0–3.
Initial `fuser /dev/tenstorrent/*` found no holders. `timeout 60 tt-smi -ls --local`
passed (`initial_device_list.log`). No reset so far.

## Inherited contract

Use `MultichipDecoder` unchanged: TP4 attention, EP4 routed experts, selected
DRAM projection policies, BFP4/LoFi routed experts, BF16 activation/CCL and QSA
paged KV (page32), FP32 recurrent state, packed group10 scale8 PLE table.
Each rank owns one complete BF16 2560-wide hyperconnection stream across every
layer. Prior rejection ledger and context planning remain authoritative;
no full-model capacity or quality claim follows from decoder evidence alone.

## Implementation and first probes

`tt/model.py` loads one real layer at a time, shares the expert/rope workspace,
shards token embedding hidden width and LM-head vocabulary, and implements the
HF terminal hyperconnection mixer. HF has no separate final RMSNorm; the mixer
performs group norm and learned mixing. Reduced probe uses real layers 0/1/3.
`tt/generator.py` is under active construction and has no acceptance evidence.

First bounded reduced probe (`timeout 900 python -m
models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model --output
models/autoports/qwen_qwen3_8_flash_next/doc/full_model/reduced_smoke_v1.json`)
loaded all three layers and terminal weights, then failed in the test's memory
view call (`MeshDevice.get_device` does not exist). Changed to the mesh memory
view API already used by the accepted decoder probes. No kernel failure or
correctness claim. Exact stdout/stderr is `reduced_smoke_v1.log`; v2 reruns the
same probe after that harness fix. Format hooks run before device commands.

The HF audit found 70 dangling checkpoint links. Exact-revision
`hf_hub_download` repaired them from shared CAS data without a bulk download;
all 131 safetensors shards are now readable. The HF reference worker records
its safe mmap loader and prompt-format evidence under `hf_reference/`.

The CPU serving audit pinned plugin `042a94863ad27d88207bcf71abdf67122c6f4c27`
and recommended upstream vLLM 0.31.0. It read docs/source/device tests only.
`serving_contract.json` currently has missing rows: these are work, not waivers.
`sampling_contract_audit.md` compares both common implementations. Select common
SamplingGenerator operations; its broad capture allocation exemption must be
replaced by a compliant generator-owned capture before acceptance. Explicit
host-logits compatibility is required for plugin requests such as logprobs on
this four-chip mesh. KV-only prefix caching has the documented `recurrent_state`
exclusion; retained-state resumed prefill still requires the readiness gate.

## Resume 2026-11-14 01:01 UTC

Resumed the same stage/artifacts after the interrupted turn. The v2 process and
HF worker have exited; no device holders remain. v2 had no output JSON or
post-upload progress before its 900-second timeout. Its dispatched upload stall
and host/device evidence are classified in `AUTOTRIAGE.md`: source cause remains
unproven, not a model-memory/precision result. No process was manually killed.

Recovery1: bounded list/reset/list all exit0; all four p300c devices visible.
`recovery1_{list_before,reset,list_after}.log` preserve output. No locks cleared.
A 90-second bounded 1x4 ring mesh open/add/readback/close passes, with
`MESH_SMOKE_OK` in `recovery1_mesh_smoke.log`. Restored accepted
`TT_VISIBLE_DEVICES=0,1,2,3` and
`TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto`
after sourcing lane-env. This restores configuration and recovers dispatch; it
is not proof that autodiscovery caused the earlier stall. v3 reduced probe is
bounded at300 seconds and adds flushed setup/layer-upload markers.

HF AIME reference passes:203 prompt tokens,100 generated tokens, exact pinned
revision/chat template/cache. See `hf_reference/result.json`, prompt, tokens,
completion and exact command/log. Completion is coherent English math reasoning
within the100-token prefix; no TT quality verdict yet. A separate exact HF
three-layer fixture includes prefill and3 decode inputs, terminal logits and
per-layer hidden states (`hf_reduced_reference.json`; external .pt path therein).

## Reduced wrapper and first split trace, 2026-11-14

`reduced_smoke_v3.json` passes after recovery: prefill top1/top5/top100 all1.0 (3 positions), PCC0.996059,47.47s total. Accepted mesh descriptor retained. `generator_v1.json`/log: tracker enabled with program-cache checks,4 greedy outputs,3 model and4 sampling replays,2 captures,0 host token/position/page-table refreshes,65.01s including setup. This is smoke evidence only; repeat and reduced teacher reference checks follow. Inactive sampler padding changed from infinite to finite logits; last prefill slice lifetime released before replay. Scheduler reload/inactive-row handling is under implementation and has not yet passed its mixed-row gate. No full-model completion claim.

`generator_v2.json`: repeated reduced requests deterministic; teacher-forced predictions56680,201617,38030,25834 match HF top1/top5/top1001.0. `generator_prefill_trace_v1.json` repeats this with traced logical prefill chunks and includes all program-cache allocations in tracking. New per-slot persistent prefill outputs keep transient tensors out of retained trace lifetimes.

`full_load_v1.json`: all48 layers plus262144-token cache loaded in157.72s and produced finite short-prefill logits. DRAM view:8banks,3,962,144,064 allocated B/bank,310,197,312 free B/bank (31,697,152,512 allocated and2,481,578,496 free B/rank). This is construction/short-execution evidence, NOT the once-at-acceptance maximum-context run. CPU ledger is `capacity_plan.json`/`.md`; additional generator buffers/migration scratch introduced subsequently need final measured accounting.

`full_accuracy_v1` runs the unmodified installed prefill/teacher scorers on one resident all-layer generator (test driver replaces only factory to reuse weights), context262144 and trace-allocation tracking. HF qualitative control runs concurrently on CPU; device jobs remain serial. Its performance fields are diagnostic and will not replace the isolated128-in/128-out measurement.

## Full accuracy v1 and serving-state work

`full_accuracy_v1.json/log` records the unchanged readiness scorers with all48
layers, context allocation262144, and allocation/program-cache tracking enabled.
Prefill100 and traced teacher100 both pass: top1=.95, top5=1, top100=1.
The diagnostic teacher timing (TTFT12058ms,6.1959t/s/u) includes tracking and
concurrent HF CPU controls; it is not the final isolated performance figure.
Resumed-prefill L257 split32 and256 pass near1; split68 fails pcc_min=.9512
against .98 (mean .9957, top1agreement1). The gate remains failed, without
threshold changes. `tests/probe_prefill_resume.py` isolates this on real0/1/3
layers; AutoDebug inspection is running before AutoFix.

`device_seed_device_v1.json/log` passes on all4 chips in10s with trace and
program-cache allocation tracking. It checks exact common SplitMix64 hashing,
negative/full64-bit seeds, carry, inactive counters, slot migration, reset
reproducibility and stable storage. Sampled-token integration remains pending.
The helper only advances seed state; common SamplingGenerator still draws tokens.

The subsequent600s bounded `serving_probe_v1` stalls in packed embedding upload.
`serving_upload_triage/{tt-triage.txt,summary.txt,host_stacks.txt}` preserves
live evidence. Host `fetch_queue_reserve_back` blocks while all device
`cq_prefetch` consumers poll empty slots, matching the prior upload stall.
No outstanding device work was manually terminated. Descriptor was explicitly
restored in this run, disproving descriptor omission as the sole cause.

CPU audit found explicit active-slot input reload overwrote dormant token
feedback. Generator now saves the persistent tokens and device-merges only
active replacements; a targeted serving probe will verify it. Prefill now
accepts explicit sampling_params, merges each row into its fixed slot, and
uses the same traced sampler for first-token output. These runtime changes
are not yet accepted.

Recovery2: serving upload job exited124 at601s. Fresh post-timeout mesh
open/add/read/close passed. Bounded list/reset/list then completed; all4 chips
visible, second mesh smoke passed. No manual process termination or lock
clearing. Exact logs: recovery2_{before_smoke,list_before,reset,list_after,mesh_smoke}.log.
Reduced resume baseline now progresses after weight loading; its capacity384
is a diagnostic difference from acceptance262144. Probe now accepts --capacity
and defaults262144 for subsequent source-matched controls.

AutoDebug completed (`resume_AUTODEBUG.md`, launcher `resume_autodebug.log`).
Launcher initially failed its sandbox preflight (user namespaces disabled),
then its child could not find code-mode-host. Retried using the already-installed
paired codex/code-mode-host from codex-venv, with explicit authorized no-child-
sandbox execution. Only the two failed CPU investigator processes were stopped.
The report finds no proven offset/padding/causal omission; it proposes absolute
chunk-boundary alignment as a discriminating repair, retaining precision and
readiness floors. AutoFix is using reduced per-position evidence before full
revalidation.

Resumed-stage continuation: reduced absolute-grid candidate finished192s,
`reduced_resume_aligned.json/log`: split32/256 pass; split68 has PCCmin0.9966
and top1agreement0.9457, still below0.98. Alignment alone is insufficient.
`tests/diagnose_resume_chunk_split.py` is prepared to compare identical state64
and residual input under32 versus4+28, with per-layer and terminal taps. No
full-stack rerun until the reduced producer diagnostic is understood.

Seeded-sampling integration v1 (`seeded_sampling_device_v1.json/log`) fails
first capture on generic-op binary loading. The seed-only probe had an eager
draw, masking this setup difference. DeviceSeedState.prepare now dispatches
initially inactive reset/draw once (idempotent) in addition to host compilation;
request state remains unchanged. Generator closes/releases failed captures
without replaying partial streams. Twelve seed/slot CPU tests pass; device
retry pending. `seeded_sampling_triage/` preserves live failure evidence; the
600s timeout owns termination. No manual signal or reset while it owns devices.
See `AUTOFIX.md` for the open repair loop and unproven claims.

CPU capacity refresh includes seed state, saved tokens, per-slot prefill logits
and migration scratch. Arithmetic B1/262144 leaves235,069,952 bytes beyond the
separate2GiB runtime allowance. Shared-pool formulas are now recorded separately;
they are planning bounds, not measured capability reductions. The advertised
context remains262144. Shared-pool integration and setup-only deferred K/V are
in progress to eliminate unnecessary constructor allocation overlap.

Seed integration retry `seeded_sampling_device_v2.json/log` PASSES in9s: 42 checks, 6 captures and 41 replays with allocation/program-cache tracking. Reset, same-seed/different-seed streams, reordered rows, migration, inactive pause/resume, sampled first-token prefill and non-sampling prefill counter semantics pass. Before retry, timeout124 was followed by successful all4-rank mesh/add/read/close smoke; no reset or manual kill. Resume producer diagnostic is now running under timeout600.

## MoE localization and shared-pool execution, 2026-11-14 02:12 UTC

`resume_chunk_split_v1.json/log` completes the reduced eager causal diagnostic:
identical layer inputs/state at64,32 versus4+28 tokens. Local GDN recurrence and
QSA state/output are exact; MoE output is the first producer that differs for
all rows. Layer1 pre-MoE inputs/router IDs/weights are entirely exact. Fixed
terminal residual yields exact logits, while propagated logits reproduce a
flip at88. This is diagnostic evidence, not a resumed-prefill pass. V2 adds
shared-expert, owned expert-lane and routed-reduction taps.

`shared_kv_pool_device_v1.json/log` passes6 tracked ownership/remap/reset checks
with global pool IDs, but does not exercise model paged writes. The real
layers0/1/3 B2C128 deferred-pool test `shared_pool_generator_v1.log` executes
model writes and passes mixed prefill equality, then fails exact mixed decode
versus solo at line127. Device close completes normally, exit1 at176s, no
reset/kill. The test is being instrumented to distinguish grouped MoE geometry
from cache-state errors. Stage remains incomplete; all thresholds unchanged.

## Rounding discriminator and remaining mixer drift, 02:25 UTC

`resume_chunk_split_reseed_v3.json/log` proves per-expert/M-block stochastic
reseed fixes the first4 token rows but not relocated suffix rows. Rejected
as an insufficient repair; no dtype/fidelity changes. `moe_reseed_v3.patch`
records the exact tested candidate.

`resume_chunk_split_deterministic_v4.json/log` uses native deterministic FPU
rounding, retaining BFP4/BFP8, LoFi, BF16 DEST and the grouped expert strategy.
Isolated same-input expert lanes, weighted routed output, shared output and
complete MoE output are exact for all0/1/3. Local residual mixer in layer0
still differs first at token82 (chunk offset18), before MoE. Propagated terminal
32-position top1 agreement1, minimum PCC .998517. This localizes a second
issue rather than accepting the stage.

`reduced_resume_deterministic_v4.json/log` (139s) passes the existing reduced
HF four-token teacher fixture at top1/top5/top100 all1, with tracking enabled.
Installed resumed-prefill gate still fails split68 top1 .96124<.98; PCCmin
.987509 now passes. Splits32/256 and negative controls retained. No threshold
changes and no all48 rerun yet. More detailed mixer taps are next; the native
rounding candidate remains provisional pending complete quality validation.

## 02:42 UTC — native RMS producer isolated and repaired provisionally

The real-input RMS-only diagnostic now completes in10s with all stages exact
across32 versus4+28 (`rms_rows_v3.json/log`). v1/v2 failures and source explanation
are in `AUTOFIX.md`. Root changed Blackhole SFPU row reduction addition ordering;
no precision/fidelity or cache-policy change. Native MoE rounding choice is now
an explicit per-op argument (defaultTrue, QwenFalse) pending build/acceptance.
`native_build_v1.log` is the incremental ttnn build. No gate is waived; reduced
resumed-prefill and all48 HF accuracy still pending these candidates.

## 02:46 UTC — reduced resumed-prefill gate passes

`reduced_resume_rms_v5.json/log`:229s total,208.8ssetup (concurrent HF CPU controls),
real0/1/3 capacity384; installed32/68/256 resume cases pass with top1=1 and
PCCmean/min1.0000. Negative controls reject; exact reduced HF four-step smoke
100/100/100 top1/5/100. Allocation tracker includes program cache. No further
reduced sweep needed; native row invariance regression then all48 accuracy.

## 02:56 UTC — fresh all48 HF accuracy passes

`full_accuracy_v2.json/log` uses scoped RNE and repaired native row reduction,
full262144 capacity and allocation tracking. AIME24chat100 prefill and traced
teacher forcing both pass97/100/100% top1/5/100 (earlier revision95/100/100).
Resident resumed-prefill check still running. Setup535.6s includes changed JIT
binaries and concurrent HF CPU controls; diagnostic6.347decode t/s/u is not
final clean performance. No reduced-precision/cache/fidelity fallback was used.

## 02:59 UTC — full48 resumed prefill passes

`full_accuracy_v2.json/log` exits0 after775s and closes all4 devices normally.
All original resume points pass, PCCmean/min1.0000 and top1agreement1; both
missing-history controls reject. Numerical repair accepted for correctness;
performance still pending. Full262144 allocated DRAM perbank: loaded3,974,114,624,
prepared4,011,411,264, final4,147,144,512 of4,272,341,376 (8banks/rank).
Final free125,196,864/bank, largest121,603,392/bank. This is short-workload
allocation evidence, NOT the single max-context execution gate. Next bounded
job: pool lifecycle `shared_pool_generator_v4.json/log`, tracker enabled.


## Tracked serving v4 — lifetime repair confirmed; ledger reconciliation

`serving_probe_v3.json/log` failed before sampled-prefill replay with one live
unsafe allocation. `prefill_forward` retained last-logit slice tensors in
`device_rows` after copying them into persistent `sample_logits`. The narrow
repair releases the dict before `_sample`; no tracker suppression, host
fallback or precision change. Four AST-extracted CPU ownership tests pass,
including a negative control removing the release. Diagnosis and buffer-ID
attribution limits are in `sampled_prefill_lifetime_audit.md`.

`serving_probe_v4.json/log` passes16/16 checks: reduced real0/1/3, B2/context128,
allocation tracking enabled including program-cache allocations. JSON wall191.47s,
preparation181.69s; external process about199s.20 captures,88 prefill/54 model/
59 sampling replays,10905 program entries before/after. Resident token feedback
has0 token/position/page refreshes. Changed page-table real writes, immutable
reverse-order output snapshots, seeded active/dormant migration, explicit
three-token host sampling and subsequent device route all pass. Host mode matches
an independent CPU sampler, never advances device RNG; restored device mode
traps full-logit readback and performs0 host token refreshes. The tracker rejects
its deliberate live-allocation control and accepts clean replay. Unload releases
all20 traces once; caller mesh close succeeds. These instrumented times are not
performance evidence. Reproduction command, after restoring accepted lane/mesh:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
timeout 3600 python_env/bin/python -m \
models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_serving \
--output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/serving_probe_v4.json
```

Poolv5 independently proves uninterrupted versus migrated B2 exact at forced
42/43/44, plus exact private/shared outputs and logical state through release,
continuation, reuse and unchanged tables. Overall pool acceptance still fails
cross-geometry43 (PCC0.9796497,top1/top5 agreement0.5) and44 survivor
(PCC0.9942773). Tests retain those failures. `capture_moe_decode_geometry.py`
prepares real upstream/reduction fixtures; no hardware run by the audit author.

Serving ledger now36 supported /21 declared unsupported /6 missing. Remaining:
measured capacity/max-context, all48 cache spec and shared allocator geometry,
later-token cross-batch decode accuracy and request isolation, longer simultaneous
ragged prefix dependence. Host-only feature rows cite a proven full-vocabulary
writable isolated logits handoff plus CPU semantics, not plugin/server execution.
Full48 B32, watcher, qualitative, final split-sampling performance/profile,
review and local commit remain stage work.

Parent reconciled `doc/context_contract.json` with full_accuracy_v2: loaded
31,792,916,992 bytes/rank; final33,177,156,096; free1,001,574,912;
BF16 paged K/V3,221,225,472 bytes/rank. This is full262144 cache allocation,
not maximum-context execution. Native reduction v1/v2 synthetic invariance
probes pass (~8s each); n1/native differences remain diagnostic. Independent
AutoDebug launcher failures (namespace/old CLI code-mode-host) were handled
without device access; paired CLI source diagnosis logs to
`batch_reduction_autodebug_paired.log`. No completion verdict is claimed.


## Watcher corroboration and pinned chunked-prefill gap scope

`serving_watcher_v1.json/log` passes the same16 checks on real0/1/3,
B2/context128: JSON281.706s, external process about299s,10905 program entries
unchanged and clean caller mesh close. Parent reports no ERROR/ASSERT/CORRUPT
hits in the dedicated watcher log. Raw48MB evidence remains in
`doc/full_model/serving_watcher/generated/`; no profiler was combined with watcher.

The remaining chunked-prefill feature maps specifically to pinned
`tests/tt/test_chunked_prefill.py`: solo and long needle tests plus four
simultaneous requests sharing ragged scheduler remainders. Full48 natural-text
resume already passes, and reduced short mixed resumes pass. The additional
prepared `tests/probe_ragged_prefix_dependency.py` tests longer simultaneous
prefix-dependent resumes directly, without a server, using the pinned prompt
constructor and HF tokenizer chat template. Its numerical comparison uses the
installed readiness thresholds unchanged and identical B2 forced-token history.
It does not score semantic recall; a later semantic verdict needs matched HF
controls under qualitative-check before debugging. No new content-quality gate
or weakened numerical threshold is introduced. Test is prepared, not run.


## AutoDebug completion, real reduction localization and candidate source review

Paired AutoDebug launcher session46361 completed with exit0 (parent process
result); `batch_reduction_autodebug_paired.log` records the completed report.
`batch_reduction_AUTODEBUG.md` includes exact ISA CPU replay matching all
61,440 saved native BF16 outputs across three synthetic cases, including the
one-owned-expert case. This explains those saved finite fixtures without
establishing a native indexing defect; the previous phase-RNE approximations
were incomplete goldens. No device access was used by the independent review.

The subsequent `moe_decode_capture_v1.json` real reduced capture completed in
149.946s. At layer0 for both forced42 and43, normalized MLP inputs, paired
IDs/weights, owned combined values and shared-expert outputs are exact across
paired B1 and B2. The first recorded difference is routed reduction:42 has
8,901/20,480 differing elements (max0.000244140625),43 has9,196/20,480
(max0.00048828125). Later layers already receive different inputs. This
localizes the real-model boundary missing from the earlier AutoDebug report;
it does not establish a repaired gate. Nonowned combined storage is excluded
from the exactness claim. Capture and12 exported fixture hashes are in JSON.

`batch_reduction_candidate_review.md` records independent source review of
`tests/probe_batch_reduction_counterfactual.py`. No blocking source defect
found. BF16 mode preserves HiFi4/BF16 DEST and public precision, EP4 grouped
experts, prefill, single-row decode, BF16 CCL, residual and cache policy while
changing the batched reduction schedule. Source preparation covers all logical
decode widths; actual trace/layout support, accuracy and performance remain
hardware work. FP32 mode widens batched accumulation and remains diagnostic
only. Both modes force acceptance_eligible=False/passed=False. No production
policy selection, accuracy threshold relaxation or stage acceptance follows
from this review. Existing shared-pool cross-geometry failures remain work.

## BF16 scheduling counterfactual and terminal profile setup

Serialized commands used the accepted lane environment and descriptor. The
isolated component `profile_moe_reduction_candidate --output
doc/full_model/moe_reduction_candidate_bf16_v1.json` completed in8s with tracker
off; the corresponding `--tracked-correctness` run completed in6s with tracking
and program-cache tracking enabled. All duplicate/permutation/repeat and20
trace outputs pass. Native/candidate speedup is3.83–3.86 atB2 and0.62–0.75 atB32,
including mask, layout conversions and persistent-output copy. This is not a
uniform performance win and no production method was selected.

`probe_batch_reduction_counterfactual --accumulation bf16 --output
doc/full_model/batch_reduction_counterfactual_bf16_v1.json` fails the preserved
token43 cross-width gate after50s (wholePCC0.99012,minrow0.97881,top1.5,
top5/top1001); all same-geometry ownership controls remain exact. The equivalent
`--accumulation fp32` diagnostic completes46s with all cross-width outputs
exact, but is permanently disqualified from acceptance because it changes
batched precision. Both commands used tracker1 and a1800s timeout. Detailed
reasoning and retained failure scope are appended to `AUTOFIX.md`.

`terminal_ops_v1.json/log` fails during force-argmax setup because the CCL
adapter lacks `get_and_cycle_barrier_semaphore_handle`; no target profile
signposts execute. Mesh teardown is clean. Tracy export additionally fails on
unquoted multiline kernel-source CSV fields. Raw capture/export files remain
in `terminal_tracy/.logs`; independent source repairs are being prepared for
the adapter and lossless exporter. Warmup durations are not terminal profiling
or accepted performance evidence. All66 HF controls are now complete and
validated; paired TT qualitative evidence remains work.


## All48 simultaneous ragged prefix continuation passes

`ragged_prefix_dependency_v1.json/log` passes with all48 layers, context2048,
B2 and allocation/program-cache tracking enabled. External run is about294s;
JSON wall time289.096s. HF-tokenizer chat prompts generated from the pinned
plugin prompt builder have lengths300/668 and absolute starts283/68. Each
resumed row matches its uninterrupted same-B2 control for24 predictions:
top1agreement1.0, minimum PCC0.9999864/0.9999663, final top1 equal. Greedy
tokens are sampled on device; comparison uses the identical forced history.
Program entries10908 remain unchanged;64 prefill,46 model and48 sampling
replays complete with clean mesh teardown. Readbacks and forced token uploads
are diagnostic controls, not a clean token-out performance measurement.

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
timeout 1800 python_env/bin/python -m \
models.autoports.qwen_qwen3_8_flash_next.tests.probe_ragged_prefix_dependency \
--output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/ragged_prefix_dependency_v1.json
```

This supports numerical retained-prefix chunked prefill, including mixed
nonaligned continuation. Semantic needle recall is explicitly unassessed;
matched HF controls and blind comparison precede any semantic investigation.
No complete plugin/server test or KV-only prefix reuse is claimed. Serving
ledger is now37 supported /21 declared unsupported /5 missing. Remaining
rows are capacity/max-context, all48 cache spec and shared allocation geometry,
decode_forward and request_isolation cross-geometry accuracy. Existing later
B1/B2 failures and stage acceptance requirements are unchanged.

## Lossless Tracy export repair and retained-capture verification

The secondary `terminal_ops_v1.log` export failure is repaired in the Tracy
submodule CSV exporter: quote comma/quote/CR/LF fields and remove manual
zone_text quoting; correct special-parent column index9 to8 in the nine-field
schema. Message backtick protocol stays unchanged. Exact source, binary and
capture hashes plus re-export argv are in `terminal_csv_reexport_validation.json`;
source diagnosis is `terminal_csv_export_diagnosis.md`. CPU commands:

```bash
python_env/bin/python tests/ttnn/tracy/test_csvexport_quoting.py
cmake --build build_Release --target tracy-csvexport --parallel 2
```

Both pass; only exporter object/executable rebuilt after CMake regeneration.
Actual C++ quoting round-trips seven records through comma/semicolon/tab and
Python CSV/pandas. New `.logs/tracy_ops_times.quoted.csv` retains all436,897
records with exactly9 fields, including130,648 multiline source fields.
Re-applying the old serialization to decoded fields reconstructs the original
malformed file byte-for-byte (SHA256
`7bbdcdcb13c8ef09ab7538f20cba72170092064b92eec53fba943b95228bac23`).
Message re-export is byte-identical. Installed importer correlates all131,108
TT host ops with no skipped-op warnings using its existing pandas C fallback
for valid multiline CSV. All original artifacts remain unchanged. No devices
were accessed by the repair author. The retained failed setup capture has zero
signposts; it cannot provide terminal performance acceptance. Successful
recollection and the final signposted tt-perf-report remain work.


### Complete reduced terminal profile v3 (2026-11-14 UTC)

The bounded eager-only profile passed and closed cleanly. Minimal preparation
warmed only the exact B1 decode and two common sampler bodies; clean timing still
uses complete generator preparation. Existing Tracy flags
`--no-web-server --op-support-count 8192` avoided the failed broad capture's
source-location limit and device buffer drops. Global postprocessing succeeded
without filtering or assertion changes. Strict independent CPU validation proves
all 23,048 host/device/final report operations match exactly, six signposts,
positive preserved durations, and model/split/force counts 1,284/65/49 per rank.

Advice-enabled tt-perf-report 1.5.0 artifacts and exact commands are in
`terminal_perf_v3/README.md`; `coverage.json` and `summary.json` preserve numerical
results. Merged kernel sums are model9.4403 ms, split0.46754 ms, force2.93959 ms.
These are eager reduced attribution, not traced full-model throughput. Existing
split wins the two common sampler comparison and is4.72% of reduced
model-plus-sampling kernel time; both return expected token243975. Force
all-gather1.4429 ms plus argmax1.1537 ms explains its cost. Clean traced
`terminal_timing_v1.json` separately records split10.3061 ms versus
force12.5737 ms token-out and identical14-token feedback streams.

Trace advice is already fulfilled; fidelity changes are rejected by the
preservation contract. Decoder layout/program-config advice and the809.006 µs
LM-head DRAM-sharded recommendation require source triage; no untried advice is
claimed resolved. No serving capability row changed. Complete raw v3 capture
was retained outside Git at
`/home/ttuser/qwen38-flash/reference/full_model_raw/terminal_tracy_v3`, with
repository symlink preserving paths and per-file SHA/byte manifest in
`raw_artifacts.json`. Prior malformed/failed captures remain untouched.

## Prefill signature OOM and reduced causal control

CPU/source audit identifies unbounded retained `(count,slot)` prefill traces:
B32 prefixes203..234 need63 old full-stack command buffers versus measured
4,169,844,224 bytes/rank free after preparation. Failed buffer143,278,080 bytes;
net growth before failure4,001,431,552 bytes/rank. This does not establish a
physical B32/context reduction. Trace eviction was rejected under the selected
tracing lifetime contract. Original source is archived at
`tests/diagnostic_sources/prefill_trace_v1/generator.py`, SHA256
`8da9d84e637f2e7db1bc210cc1f050a7afd6efdfd95095cbf94d46c1ee4a6233`.

Parent-run `prefill_signatures_original_v1.json` passes mixed all-logits equality
and start68→257 resume, but start31→265 fails the unchanged98% top1 threshold:
221/234=94.4444%, PCCmean.9997294/min.9967314. Strong PCC does not waive top1.
Current canonical/dyadic reduced candidate also exposes the resume31 failure.
`prefill_native_single_counterfactual_v1.json` then makes both resumes exact
under a prefill-only native-BF16 n1 reduction override (6 warmup/capture override
calls,9 unchanged outside-prefill calls,0 decode overrides). Its
`diagnostic_checks_passed:true` is explicitly NOT acceptance: `passed:false`,
`acceptance_eligible:false`; this changes accepted n1 accumulation and remains
unselected. Detailed scope/source diagnosis: `prefill_trace_source_diagnosis.md`.

`tests/probe_prefill_trace_signatures.py` retains failed comparisons and snapshots,
collects independent diagnostics before the final unchanged aggregate gate, and
supports isolated original/candidate and two diagnostic-only prefill controls.
The BF16 batched-matmul mode reuses the existing vectorized helper for n>1 while
keeping n1 FP32 and decode unchanged. `--all-layers` rejects arithmetic controls
and timing-only mode and labels its result full48 correctness, never stage
completion. Snapshot v2 embeds layer indices and rejects cross-depth/unlabelled
baselines before device setup. Optional matched-reset submission-order check
compares actual device tokens, physical-row logits, full request state and
model/sampling trace reuse. No new device or HF run was launched by the CPU
audit author; syntax, CLI rejection and AST guard checks passed.

Parent-run `prefill_bf16_batched_counterfactual_v1.json` is rejected: resume68→257
is exact, but resume31→265 top1 is225/234=96.153846%, below the unchanged98%
requirement (PCC mean0.9997680474482344/min0.9964045202905336; max absolute
delta0.4677734375). It records30 scoped prefill overrides,9 unchanged
outside-prefill calls,0 decode overrides, `diagnostic_checks_passed:false`,
`passed:false`, and `acceptance_eligible:false`. Neither arithmetic control is
selected for production. Parent launched the separate unmodified-policy full48
shared-KV/decode-order confirmation as `prefill_signatures_full48_v1`; its result
is pending. The reduced native-single diagnostic pass remains causal evidence
only, and original/candidate failures remain intact.

Full48 `prefill_signatures_full48_v1.json` is a preserved harness setup failure,
not a model result: one reduced pool spec was still used for12 attention layers.
Fixed by deriving the count from actual cache states, enumerating independent
pool tensor indices, and validating all slot/scratch per-layer aliases. Extracted
the shared layout arithmetic into `tests/shared_pool_layout.py`; existing batch
probe validation is unchanged. Page-count constants in prefill/decode helpers
now derive from configured capacity. CPU command
`python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_pool_layout_cpu`
passes3 tests (reduced/full48 enumeration, missing spec/alias rejection and B1
capacity rounding); syntax checks pass. Parent may rerun a fresh full48 v2.


## Terminal row assembly, LM head and planning revision (2026-11-14)

Device-owner results, source integration and CPU arithmetic only; no full48
acceptance or max-context execution claim is made by this entry. Full48 v2 was
still running when these planning updates were written.

`terminal_rows_v1.json` first showed the tiled row writer removes the original
32-row concat/untilize overhead at small batches, but regresses atB32. The
expanded `terminal_rows_v2.json` added8/16/24/28 and the native in-place fill
candidate. All12 exact/poison case selections passed for all three strategies.
The separate `terminal_rows_tracked_v1.json` passed with allocation/program-cache
tracking enabled:30 traces and39 programs, all buffers/variants warmed before
capture, and final replay after reads. Fill assembly versus concat was
0.047810/0.537668ms atB1,0.067042/0.537574ms atB2,0.490270/0.537737ms atB24,
0.567541/0.537589ms atB28 and0.644717/0.537739ms atB32. Select fill plus the
existing tiled row writer through24 slots, original concat above24. The
alternative immutable-template copy was nearly tied with fill but needs another
3,981,312 allocated bytes/rank; native fill adds none. NaN/Inf poisoning before
every correctness replay was fully overwritten, including inactive -100 rows.
The actual common split sampler selected equal global-max token IDs across all
four ranks. Production integration is restricted to `_decode_body` assembly;
model arithmetic and sampler math are unchanged.

The older row launch wrapper was not preserved in the compacted command history.
The following are exact reproducible CLI commands for the recorded options,
not a claim to reconstruct that historical wrapper byte-for-byte. Use the
accepted lane environment, watcher absent and all profilers/perf counters off:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto timeout --signal=INT --kill-after=60 240 python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_terminal_row_assembly --fill-candidate --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_rows_v2.json
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto timeout --signal=INT --kill-after=60 240 python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_terminal_row_assembly --fill-candidate --tracked-correctness --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_rows_tracked_v1.json
```

`lm_head_dram_v1.json` compares real checkpoint head weights and replicated,
seeded normalized BF16 terminal inputs against the original TT head. Q1/Q32
common greedy IDs and top1/5/100 membership are100% for both block1 and block2.
Block1 PCCmin is0.9999999977/0.9999999951; max absolute error is
0.0078125/0.015625. This is TT-versus-TT synthetic-input component evidence,
not HF accuracy. Projection including reshard/output restore/slice/padding/copy
improved Q1 from0.868638 to0.718163ms andQ32 from0.847203 to0.693952ms.
Block2 is a near tie (Q1 0.716828ms,Q32 0.693952ms); select block1 for simpler
K blocking and smaller L1 demand. The original TT head and comparison head
coexist only in the isolated benchmark.

Device-owner LM-head launch environment was activation of `python_env`, source
`/home/ttuser/qwen38-flash/lane-env.sh`, then restoration of visible devices
`0,1,2,3` and the descriptor above, plus the usual pinned `QWEN_CHECKPOINT` and
`QWEN_PLE_CACHE`. Watcher/profilers were unset and skip-program-cache was0.
The command wrapper was `$TT_MODEL_BRINGUP_ROOT/scripts/wait_for` with
`--timeout680 --tail12`; the actual command payloads/options were:

```bash
# Clean: TT_METAL_TRACE_ALLOC_TRACKING=0, TT_METAL_TRACE_ALLOC_TRACEBACKS=0
$TT_MODEL_BRINGUP_ROOT/scripts/wait_for --cmd 'timeout --signal=INT --kill-after=60 600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_lm_head_dram --blocks 1,2 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/lm_head_dram_v1.json' --file models/autoports/qwen_qwen3_8_flash_next/doc/full_model/lm_head_dram_v1.log --timeout 680 --tail 12
# Tracked v1: TT_METAL_TRACE_ALLOC_TRACKING=1, missing CLI mode flag; intentionally preserved failure
$TT_MODEL_BRINGUP_ROOT/scripts/wait_for --cmd 'timeout --signal=INT --kill-after=60 600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_lm_head_dram --blocks 1 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/lm_head_dram_tracked_v1.json' --file models/autoports/qwen_qwen3_8_flash_next/doc/full_model/lm_head_dram_tracked_v1.log --timeout 680 --tail 12
# Tracked v2: same environment, explicit tracked mode
$TT_MODEL_BRINGUP_ROOT/scripts/wait_for --cmd 'timeout --signal=INT --kill-after=60 600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_lm_head_dram --blocks 1 --tracked-correctness --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/lm_head_dram_tracked_v2.json' --file models/autoports/qwen_qwen3_8_flash_next/doc/full_model/lm_head_dram_tracked_v2.log --timeout 680 --tail 12
```

Tracked v1 stopped in preflight before device access because tracking was on
without `--tracked-correctness`; it produced only the retained `.log`, no result
JSON. Tracked v2 passed,47 stable programs and final replay. No threshold or
tracker exemption was introduced to turn the setup error into a pass.

Production `model.py` directly prepares one DRAM-bank-sharded head from HF
partitions, with a shared decode/rows representation and no native TT duplicate.
It preserves BF16 weights/activation/output, HiFi4, accurate math, FP32 DEST and
packer accumulation off. Constructor/logits CPU tests cover tied and untied
heads and verify no native head upload; both pass. Detailed component selection
and evidence links are in `terminal_strategy.md`.

Updated `capacity_arithmetic.py`, regenerated `capacity_plan.json`, and changed
`cache_budget.py` to charge the bank-padding delta exactly once: head bytes per
rank317,849,600→318,504,960, delta655,360; known static planning total
28,289,880,064→28,290,535,424. All planned profile headrooms/global-pool budgets
are recomputed. Historical32 prefill-input bucket allowance is retained
conservatively rather than taking unmeasured runtime credit for the new six
buckets. `doc/context_contract.json` preserves every measured allocator value
from `full_accuracy_v2.json` and labels it historical interleaved-head evidence;
new layout figures are explicitly planning-only. Advertised262144 context,
KV dtype, cache policy and runtime reserve are unchanged. Fresh integrated
memory evidence and the one max-context acceptance run remain required.

CPU commands:

```bash
python3 models/autoports/qwen_qwen3_8_flash_next/doc/full_model/capacity_arithmetic.py
python_env/bin/python -m pytest models/autoports/qwen_qwen3_8_flash_next/tests/test_cache_budget.py -q
python_env/bin/python -m unittest models.autoports.qwen_qwen3_8_flash_next.tests.test_lm_head_setup_cpu models.autoports.qwen_qwen3_8_flash_next.tests.test_trace_signatures_cpu models.autoports.qwen_qwen3_8_flash_next.tests.test_decode_validation_cpu models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_validation_cpu -q
```

The budget tests independently derive per-bank head padding and prove a formerly
just-fitting admission is rejected after the655,360-byte charge, rather than
silently advertising fewer context tokens. The source-editing agent performed
no hardware launches.

Validation result:16 budget tests passed;23 constructor/trace/input-orchestration CPU tests passed. Historical loaded/prepared/final memory dictionaries exactly equal `full_accuracy_v2.json`; supported context remains262144. Python compilation and `git diff --check` pass. Pytest imported repository TTNN bindings and emitted shutdown nanobind reference-leak diagnostics; it opened no device and all16 CPU tests passed.

CPU-only acceptance/text preflight after six canonical prefill buckets and the
new direct-DRAM head passes (`text_preflight_v2.json`): exact installed AIME128
cache and all66 HF qualitative caches still match; pinned chat-template prompts
and EOS match. No HF model or TT device was opened. The acceptance driver and
qualitative runner use public interfaces and make no old bucket/head-layout
assumption. Existing artifact-normalization/blind-provenance tests pass7/7;
canonical prefill CPU tests pass10/10. New read-only cache checker is
`tests/preflight_full_model_text.py`; command queue is `acceptance_commands.md`,
with tracked accuracy/resume, clean performance, AIME autoregression, TT66 and
last once-only max context separated. No acceptance/device command was executed.


Current-revision reduced serving and all48 prefill follow-up (device-owner runs):
`serving_probe_v5.json/log` passes all16 checks on real layers0/1/3, B2/context128,
with the integrated DRAM head, row-fill terminal assembly and six canonical
prefill traces. Preparation33.10s, JSON wall38.80s, external about41s; these
tracked times are not clean performance metrics. Program entries remain2179
(before/after), versus10905 in the older v4 revision.138 prefill,54 model and59
sampling replays use12 captures with0 request-time releases; teardown releases
all12 once. Resident feedback has0 token/position/page-table refreshes. Changed
page writes, inactive rows, async immutable snapshots, explicit host mode and
return to on-device sampling pass. The deliberate postcapture live allocation
is rejected, then clean replay passes; caller mesh close succeeds.

Device-owner launch after the established activated lane, accepted1x4 descriptor
and visible devices, checkpoint and PLE cache restoration; tracking enabled,
skip-program-cache0, watcher and profilers absent:

```bash
$TT_MODEL_BRINGUP_ROOT/scripts/wait_for --cmd 'timeout --signal=INT --kill-after=60 1200 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_serving --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/serving_probe_v5.json' --file models/autoports/qwen_qwen3_8_flash_next/doc/full_model/serving_probe_v5.log --timeout 1280 --tail 12
```

`prefill_signatures_full48_v2.json/log` is a real unresolved correctness failure,
not a passing full-model refresh: shared-pool B2/context384, current head and
canonical prefill, resume31→265 top1=0.9529914529914529,
PCCmean=0.9894826780031827, PCCmin=0.6825643852129475; resume68→257 exact.
Mixed/reversed all-logits outputs are exact; all six prefill trace IDs reuse,
no request-time release,2181 programs remain stable. The separate matched-reset
decode order check has exact logits/tokens/feedback and trace reuse, but its
state-hash failure comes from the test's scalar seed initializing only slot0;
explicit seed rows fix the harness and require a fresh run. This does not
explain or dismiss the independent resume31 failure. The earlier
`full_accuracy_v2.json` remains evidence for its recorded revision only.

Read-only checkpoint audit for the proposed odd-tail repair: an odd exclusive
end E commits the last singleton at even E−1. QSA updates compressed keys only
when absolute position mod4 equals3; its native paged-update −1 index sentinel
skips the write. Thus a strictly odd-tail checkpoint needs GDN conv/recurrent,
PLE conv/token history, QSA raw history/device position and host position, but
no compressed-key or KV snapshot. Replay overwrites the old last KV row before
gather; completed-group and tail masks are causal. Keep current scheduler page
ownership, migrate checkpoint metadata with the request, invalidate after
reset/decode and verify the saved tail identity. This is source reasoning,
not a device-validated repair or acceptance claim. No hardware was launched by
the source-audit agent, and no serving-ledger missing row was removed.

Full48 v2 confirms resume31 numerical failure (top1=223/234, PCC mean.9894826780,
min.6825643852); resume68 remains exact. Decode-order seed-only state mismatch
is a distinct harness defect, fixed with explicit seed=[0,0], slots=(0,1),
and7 passing actual-setter CPU tests. Original artifact preserved; source
sidecar added with parent attestation and future model/decoder source hashes.
Reduced-only `--prefill-tail-checkpoint-diagnostic` is prepared, not executed:
compact per-request snapshot before final odd singleton, same-policy paired
replay on immediate continuation, slice old row; no KV/index/ownership copy.
Missing-checkpoint negative control remains required to fail its original
resume gate. Three CPU orchestration tests pass. Full48 estimated additional
snapshot bytes/rank/slot34,478,592; device allocator measurement and B32/context
capacity assessment pending. Detailed source proof and limits are in
`prefill_trace_source_diagnosis.md`. No model/generator changes or device run.


Resident token-feedback selector component evidence: device-owner
`token_feedback_tracked_v1.json/log` passes14 changed-mask/repeated cases on all
four ranks,38 stable programs and one candidate trace reused across seven active
sets. Exact UINT32 tests include unsigned high bits and0xffffffff, preserved
saved/membership buffers and final replay. `token_feedback_clean_v1.json/log`
measures the original32-slice/concat restore at0.068061–0.068081ms versus the
single dataflow selection at0.004318–0.004415ms (about15.5× faster). These are
isolated restore timings, not full-model sampling or decode performance. Native
`where` requires tiled inputs; the helper preserves existing row-major UINT32
buffers without casts, host token feedback, sampler math or persistent allocation.
Two CPU tests compile the exact selection loop and pass unsigned edge/random
cases, in-place aliasing and changing membership. Production integration and
real-generator tracked rerun are separate gates.

Compact prefill checkpoint planning is now charged per configured request slot,
including eight-bank padding:36*(163840+786432)+(163840+512)+12*(16384+512)
=34,576,896 bytes/rank/slot. The initial34,478,592 estimate omitted98,304 bytes
because each QSA raw-history tensor's4 useful BF16 tiles reserve8 banks. The
reduced0/1/3 B2 total4,163,584 matches diagnostic allocation evidence. No extra
checkpoint migration scratch is charged: request-owned checkpoint buffers and
metadata permute by references outside capture, retaining all allocations.
`capacity_arithmetic.py`, `capacity_plan.json`, `cache_budget.py` and context
planning provenance include this amount without modifying historical allocator
measurements, selected dtype/fidelity or advertised262144 context. Full48
integrated memory and once-only max-context acceptance remain pending.

CPU command `python_env/bin/python -m pytest models/autoports/qwen_qwen3_8_flash_next/tests/test_cache_budget.py -q`
passes17 tests, including independent bank-padding derivation and B1/B8/B32
checkpoint ownership at context128/262144. Repository conftest imports TTNN and
prints the existing shutdown binding-leak diagnostics; no device was opened.

Token-feedback integration after checkpoint-generator handoff is now in source:
constructor binds `DeviceTokenFeedback` to existing decode/saved tokens and the
seed-state active mask; `prepare()` warms it before any capture. Restore updates
membership only when changed, then enqueues exact in-place selection. Sampling
trace keys now retain semantic flags only, independent of active subset; the
same persistent membership mask drives seed advancement and token preservation.
Eager terminal profiler preparation and synthetic seeded-sampling constructor
metadata were updated. Five actual-method/kernel CPU tests pass, including
changed membership reusing one trace, same membership performing no upload,
inactive tokens surviving, and retained parameter configurations reusing their
trace. Forty checkpoint/input/lifetime/profiler CPU regression tests pass;
compilation and diff-whitespace checks pass. Real-generator tracked validation
after both checkpoint and feedback integrations remains required.

Parent-run `prefill_tail_checkpoint_diagnostic_v1.json` passes all diagnostics:
both resume68/31 bitwise exact, deliberate missing checkpoint fails original
resume gate, decode-order all checks pass with two explicit seeds, six trace
IDs retained/no late compile. Measured reduced B2 snapshots4,163,584 bytes/rank.
Correct full48 banked charge is34,576,896 bytes/rank/slot (raw-history four-tile
payload reserves eight tiles); prior34,478,592 estimate is superseded.

Integrated compact checkpoints in `tt/prefill_checkpoint.py` and generator
prefill/lifecycle only. Restore occurs after full validation/current table
binding, snapshots migrate with requests without extra scratch, and active
decode/reset invalidate only affected requests. Public slicing excludes the
replayed old row. Generator then handed to upload_triage for its separately
proven token-feedback integration. CPU command covering
`test_prefill_checkpoint_cpu`, `test_prefill_validation_cpu`,
`test_decode_validation_cpu`, `test_generator_pool_cpu` passes38 tests; production
modules compile. Includes shared-pool COW/new-table preservation and invalid
ownership rejection before mutation. No TT run by the CPU/integration author;
current-production reduced/full48 numerical acceptance remains pending.

2026-11-14 integration follow-up: `prefill_checkpoint_production_v1.json/log`
passes tracked reduced real0/1/3 production-generator checks (43s external).
Both starts68/31 match uninterrupted prefill bitwise, mixed/reversed outputs and
all decode-submission-order state/token/trace checks pass; six prefill traces
retained with no request-time releases or late program compilation. This includes
both production checkpoint and bounded sampling/token-feedback changes. Source
hashes are in the artifact. Full48 confirmation launched separately.

`factored_decode_slots_tracked_v3.json/log` passes23 exactness checks in83.285s
(86s external): arbitrary/nonprefix slots, changing membership without reset,
full logits, all private/logical KV state and common device-sampled tokens.
222 diagnostic captures occupy224,927,744 bytes/rank. It is not a production
selection or full48 fit proof. Priorv2 stopped on a Torch int64-to-int32 advanced
index assignment in the test harness; v3 explicitly constructs int32 inputs.
The exact-width diagnostic remains frozen while a separate duplicate-valid-row
MoE bucket candidate is developed. Native singleton FP32/grouped BF16 reduction
policies are unchanged; no dummy attention/cache rows are executed.

Acceptance driver now records all tt/*.py source SHA256s and its own hash,
and rejects reuse of an existing top-level report path before opening devices.
This does not replace the existing max-context once-only marker.

Final terminal profiler CPU audit: the selected B1 terminal helper is compatible
with current decode factoring, the direct-DRAM head, bounded token feedback and
single prefill workspace. No live per-slot `prefill_outputs` references remain
outside archived sources. Historical B>1 direct-capture probes use the explicit
whole_decode_control loader maintained by the serving owner. Fixed the terminal
scope test's missing token_feedback.prepare fixture and obsolete 32-bucket
comment; constrained eager profile to B1/non-factored scope, added fresh-output
rejection, six expected signposts, and source hashes for all tt model files,
common samplers and the helper. Four CPU scope tests and py_compile pass.
Prepared `doc/full_model/terminal_final_commands.md` with exact serialized
reduced Tracy capture and strict coverage/advice-enabled tt-perf-report commands
using the currently installed CLI. All old failed raw captures remain intact.
No profiler/device work was launched; final attribution evidence is pending.

Shared prefill output workspace: replaced B persistent physical32-logit outputs
with one independent canonical workspace. Final request rows stream into their
fixed sampler-canvas slots; inactive rows are cleared once per submission and
temporary sliced rows are released before subsequent trace replay. All six
slice/write shapes are warmed in prepare. Host output slicing and compact
checkpoint lifecycle are unchanged. Archived pre-edit generator SHA256
f7c458a3fae43f481cca514acb9699efbe3e8991d25ac811f34823ebded16675 in
tests/diagnostic_sources/prefill_workspace_before/generator.py, with provenance
doc/full_model/prefill_workspace_source_baseline.json. Expected persistent
saving at B32 is123,420,672 bytes/rank (117.703125 MiB), B1 unchanged; measured
allocator/performance evidence is pending.

Prepared `doc/full_model/qualitative/judge_handoff.md` with a fresh xhigh,
fork-none judge message restricted to the blind packet and frozen rubric. It
specifies the hash-bound ratings envelope, all 120 sampled examples, frozen
per-prompt strict >20-percentage-point rule over ten matched seeds, owner-only
label joining, and separate AIME/greedy direct-reading obligations. No judge
was launched and no generated text was assessed by this source-aware subagent.

Latest CPU text preflight passed at
`doc/full_model/text_preflight_workspace_v1.json`: all 66 HF qualitative cache
entries and the installed AIME 128-token greedy entry match exactly, with pinned chat
template/tokenizer and EOS IDs [248046,248044]. No HF generation or device call.
Acceptance artifact and sampling-parameter CPU tests passed 14/14. Source audit
of generator SHA256
6b38b367a6fca1b9e212561a7c6e4883964b047883b6faaad7885a726249dc84
confirms both text drivers use public generate/prefill APIs, independent of the
removed per-slot output list. Prepared prompt lengths 60/67/75/61/66/62 reach the
six canonical buckets through public padding/chunk handling; B1 sampled seeds
0–9 route through the existing explicit sampler setter. The streamed logits
canvas feeds the common sampler, and logical EOS slicing remains in the text
driver. This is source/cache compatibility evidence, not a hardware text pass.

CPU validation command (44 passed): python_env/bin/python -m unittest
models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_checkpoint_cpu
models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_validation_cpu
models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_sampling_lifetime_cpu
models.autoports.qwen_qwen3_8_flash_next.tests.test_decode_validation_cpu
models.autoports.qwen_qwen3_8_flash_next.tests.test_generator_pool_cpu -q.
py_compile passed for generator.py, probe_seeded_sampling.py and the changed
CPU tests. No device or HF work was run by this subagent. Reduced and full48
tracked correctness must be rerun on the new source; the preceding full48 job
loaded the archived source. Existing failed artifacts and gates are preserved.

`prefill_checkpoint_full48_v1.json/log` passes the production all48 checkpoint
confirmation (269.265s internal/275s external, tracked). Starts31→265 and68→257
are bitwise exact. Mixed/reversed all logits, matched-seed decode tokens/logits/
private state/feedback, trace reuse and no late compilation pass. This resolves
the exact failing start31 case in `prefill_signatures_full48_v2.json` without
changing native singleton FP32/grouped BF16 reduction. Captured generator hash
predates the subsequently authored shared prefill-output workspace; that change
has separate validation pending. Device teardown was clean, no reset.


Shared-prefill-output capacity planning update (CPU only): `cache_budget.py`
and `capacity_arithmetic.py` now charge exactly one independent3,981,312-byte
BF16 physical32 prefill workspace per rank at every configured batch size,
separate from the sampler canvas. Regenerated `capacity_plan.json` moves that
output from the per-slot ledger into a shared-workspace ledger. Per-request
checkpoints remain charged separately. B32 source saving is31×3,981,312 =
123,420,672 bytes/rank; B1 is unchanged. `doc/context_contract.json` records
source hash6b38b367a6fca1b9e212561a7c6e4883964b047883b6faaad7885a726249dc84
and source-only provenance; all six historical allocator fields are verified
unchanged and supported context remains262144. This is not an isolated device
allocation-delta measurement.

```bash
python3 models/autoports/qwen_qwen3_8_flash_next/doc/full_model/capacity_arithmetic.py
python_env/bin/python -m pytest models/autoports/qwen_qwen3_8_flash_next/tests/test_cache_budget.py -q
```

All18 CPU budget tests pass, including independent tile/bank rounding, one
output across all batch profiles, exact B32 saving and unchanged B1 charge.
Diff checks pass. Repository conftest imports TTNN and emits existing nanobind
shutdown-leak diagnostics; no device was opened by the planning author.
`memory_forecast_current.md` now reflects the new ledger and keeps source-only
credit separate from measurements. `runtime_fallback_audit.md` records the
latest reduced servingv6 pass and all48 checkpointv1 pass, with the latter
explicitly scoped before the workspace edit. Factored decode, full48 workspace
and fresh HF/capacity acceptance remain work; no stage promotion is claimed.

Tracked `serving_probe_v6.json/log` passes16/16 in43.905s (47s external),
including shared prefill workspace, compact checkpoints and bounded device token
feedback.2177 programs remain constant,10 native traces retained. Source hashes
are captured in the artifact; this precedes production decode factoring.

`factored_decode_buckets_tracked_v1.json/log` passes44 exact checks in110.793s
(114s external), real layers0/1/2/3 and actual ordinary-GDN storage.185 native
captures use230,957,056 bytes/rank. Full48 role projection2,194,677,760 bytes/rank
is a forecast, not full48 allocation proof. Duplicate valid rows preserve the
native n1 FP32/n>1 BF16 policies; inactive attention/cache rows are never run.
`factored_decode_buckets_clean_v1.json/log` also passes44 checks (143s external).
Twelve resident-feedback timing cases show0.26–1.39% overhead versus unbounded
whole-model subset traces; B32 is300.2237→300.9989ms. The configured-B1 whole-model
path remains unchanged. This bounded strategy is selected for integration at
configured batch>1; reduced/full48 production validation still remains.

`acceptance_text_v1` has completed its128-token AIME autoregressive phase;
the66-request matched qualitative phase is still running. Installed automated
degeneracy check returns0. Root directly read both saved HF/TT completions and
recorded hash-bound `qualitative/autoregressive_review.json`: coherent English,
no mechanical repetition/wrong-language drift, same walking-time problem setup.
The first token divergence is generated index6 (absolute208 for this202-token
rendered autoregressive prompt), a benign wording change. Both128-token outputs
end in partial reasoning without EOS; final mathematical correctness is not
assessed at this cap. The main100-token readiness reference separately has203
prompt tokens. No source/model mismatch between the paired autoregressive sides
is implied by these distinct, separately cached workloads.

Acceptance harness refresh: `acceptance_harness_cpu_v1.json/log` records20 CPU tests and7 subtests passing. Coverage is bounded trace ownership/metadata plans, capture memory evidence, fresh acceptance artifacts and reduced terminal profiling scope; no device is opened. Existing nanobind shutdown leak diagnostics remain in the raw log. Hardware validation is separate.

Retained historical artifacts over500KiB outside Git under `/home/ttuser/qwen38-flash/reference/full_model_raw`, preserving every repository artifact path as a symlink and verifying byte count/SHA256 in `raw_artifacts.json`. No active log moved, no data discarded, and small numerical reports remain ordinary files. This keeps raw tensor/profiler/verbose-log payloads outside the stage commit.

Pinned Black formatting preparation (CPU only): selected 74 new Python files
using git ls-files --others --exclude-standard under model tt/, tests/ and
doc/full_model, excluding every diagnostic_sources path and the tracked
multichip_decoder.py. Black 23.10.1 reformatted 43 files; all 74 before/after
ast.dump(ast.parse(source), include_attributes=False) values are identical.
Frozen Python snapshots and tracked decoder hashes remain unchanged.
formatting_manifest_v1.json records exact tool argv, versions, every old/new
SHA256, AST equality, protected hashes and check-only findings. Commands used
/home/ttuser/.cache/pre-commit/repow99qs9xl/py_env-python3/bin/black --workers 4
<manifest file list>, then the same --check invocation. Pinned isort 5.13.2
--check-only --diff reports 58 import-order findings; autoflake 2.3.1
--check-diff --remove-all-unused-imports --exclude=__init__.py --jobs 4 finds
one unused import (importlib.util in tests/preflight_full_model_text.py).
Neither import tool modified files. Exact logs are formatting_black_v1.log,
formatting_black_check_v1.log, formatting_isort_check_v1.log and
formatting_autoflake_check_v1.log. No devices, model execution or semantic
changes; no pending Python writes at formatter handoff.

Repository-pinned clang-format19.1.4 diff check on the stage-owned Blackhole row-reduction header and native moe_fused_swiglu paths reports no changes. `git diff --check` passes. No native source or executable changed in this check.

Authorized import-lint preparation: pinned isort 5.13.2 applied only to the
same 74 untracked stage Python files in formatting_manifest_v1.json. Removed
only confirmed unused stdlib importlib.util from preflight_full_model_text.py.
Pinned Black 23.10.1 then ran; Black/isort/autoflake 2.3.1 checks all pass.
58 files changed; all non-import ASTs remain identical and py_compile passes
for all 74. Every frozen diagnostic Python and accepted tracked decoder hash
remains unchanged. formatting_imports_manifest_v1.json records exact commands,
old/new hashes, scope and results. Four requested actual CPU suites passed
22 tests (factored_decode, batch_hf_evidence, acceptance_artifacts,
terminal_profile_scope); formatting_imports_cpu_v1.log preserves output.
text_preflight_imports_v1.json passes all 66 exact qualitative HF cache hits
and the installed 128-token AIME greedy cache with pinned tokenizer/EOS.
No HF generation, device run or production arithmetic change. Source handoff
is complete; new measurements should record the new import-formatted hashes.

`acceptance_text_v1` finishes all48 AIME128 and all66 TT qualitative requests in1634.45s, then exits1 after clean device teardown because the final marker receives `phase` both positionally and by keyword. Raw JSON records completed phases, while `acceptance_text_v1_process_status.json` records the actual process failure. Repaired only final reporting keyword to `completed_phase`;8 CPU tests pass, including executing the actual final marker call. Saved completions are retained, not regenerated. A fresh blind judge receives only the frozen120-example packet/rubric; owner separately read all6 greedy pairs (`qualitative/greedy_review.json`).

Current production factored decode passes `serving_probe_v7.json/log`:16 checks,49.394s internal/52s external,2180 stable programs,22 retained captures,132 prefill/54 model/59 sampling replays,zero request-time releases and clean caller mesh close. Real0/1/3 B2/C128 scope includes feedback, inactive state, changed/unchanged pages, host compatibility and tracker controls; full48 B2/B32 still require their own evidence.

Completed qualitative content verdict: qualitative/verdict.json records
within_matched_control_margin for all six prompts, each HF 0/10 and TT 0/10
failures over seeds 0–9. Fresh /root/qualitative_blind_judge used fork_turns=none
and inherited model/reasoning defaults (no explicit xhigh override); it rated
120 label-hidden examples. Owner froze blind_ratings_frozen.json before joining
labels with blind_compare.py --ratings qualitative/blind_ratings.json.
Canonical packet SHA256 dee53f0991c18b41ca73bf85e903354e94df03e51c08762246828748e3721abf;
rubric d0a84d0339b55d908f5062ff9b76aba78ab2a4321213be6004faa78c6bcf5084;
ratings e7a8a57f428115634a9cc4bb5946202934633c46d40512e7302245187b793d60.
The freeze record also binds exact ratings-file bytes. Strict per-prompt rule
remains TT failures minus HF failures >2/10; no thresholds changed.

Original HF controls were reused by exact model/revision/tokenizer/template/
prompt/settings/library-version cache matches, verified by
text_preflight_imports_v1.json: 66 shared controls and installed AIME 128-token greedy.
Recipe: checkpoint default thinking chat template, temperature 1.0, top_k 20,
top_p 0.95, repetition_penalty 1.0, presence_penalty 0, seeds 0–9, 128 new tokens;
greedy uses temperature 0/top_k 1/top_p 1. Pinned generation_config EOS IDs
[248046,248044] terminate logical outputs including EOS; raw TT tokens remain.
Both backends have 15/60 sampled and 1/6 greedy EOS stops, 45/60 sampled and 5/6
greedy cap stops. Neither reference regeneration nor approximate cache reuse.

Owner directly read all 12 greedy outputs and both AIME outputs; exact reviews
are qualitative/greedy_review.json and qualitative/autoregressive_review.json.
AIME diverges at generated index 6 in wording but retains coherent walking-time
reasoning; no mechanical loop or wrong-language drift. Shared greedy snippets
include “learning with a teacher” (q02), “Bonjour, comment allez-vous aujourd’hui ?”
(q05), and “def fibonacci(n):” (q06). Blind seed 0 examples: q01 HF “patterns
emerge from the noise” and TT “Data flows inward” remain coherent haiku
planning; q06 unfinished code is cap-limited. No final-answer factual/math/code
completeness claim. Added full_model_shared_thinking_128_token_cap with
original_model:same to doc/eval_readiness/limitations.json, preserving all
existing numerical entries and thresholds; it is a scoped output-window
limitation, not a waiver for accuracy/cache/serving failures.

Runtime source scope is all 48 B1, generator 6b38b367... before B>1 factoring and
formatting; B1 semantic path is retained. acceptance_text_v1 completed AIME 128-token
and TT 66 with on-device sampling and zero token refreshes, then exited 1 only
after teardown in the duplicate-phase final status marker. Companion
acceptance_text_v1_process_status.json records that actual exit;
acceptance_reporter_cpu_v1.log records 8 passing reporter-fix tests. Saved
outputs were reviewed without rerunning generation. No process-pass, broader
full-model acceptance, performance or maximum-context claim is made.

`prefill_factored_full48_v1.json/log` passes current integrated all48 B2/C384 shared-cache signatures in272.656s (278s external). Starts68→257 and31→265 are bitwise exact; mixed/reversed all logits, matched-seed decode logits/tokens/all request state, feedback and trace reuse pass.2182 programs remain stable;202 captures are retained with0 request releases. This includes the shared prefill workspace and bounded factored production code (generatorSHA f539343185c824cbc73b820c241b03d9d953fc73e0808fa421be0e21db4abc63), closing their all48 B2 integration gap. It does not substitute for full48 B32 HF accuracy or maximum context.


## HF geometry control and diagnostic scorer correction

The exact original-HF CPU control v2 completed with exit0,64 real reduced-layer forwards,
9.62975s and peak RSS5084500KiB. Command (offline CPU only):
`timeout --signal=INT --kill-after=30 1200 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_shared_pool_hf_control --threads 4 --repeats 2 --snapshot /home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_pool_hf_control_v2.json --tensors /home/ttuser/qwen38-flash/reference/shared_pool_hf_control_v2.pt`.
The matching log is `shared_pool_hf_control_v2.log`. The original HF survivor
(B2 through forced43, then B1 forced44 after native cache reordering) differs
from natural B1 at44: PCC0.9951953,raw top1agreement0,top5/top100agreement1.
Repeated identical histories remain exact. Independent interpretation distinguishes
original HF batch-history sensitivity from the TT forced43 excess; no TT numerical
acceptance gate is waived by this control.

Fixed one diagnostic scoring defect in `decode_geometry_comparison`: compare
actual.argmax with reference.argmax, rather than actual.topk(100)[0] with
reference.argmax. `topk` tie order is not argmax's lowest-index tie contract.
Original scorer source is retained in `sources/shared_pool_geometry_scorer_v1.py.txt`;
all historical JSON/tensor results remain unchanged. PCC,top5/top100 and all
thresholds are unchanged. `python_env/bin/python -m unittest
models.autoports.qwen_qwen3_8_flash_next.tests.test_shared_pool_diagnostics_cpu`
passes4 tests (`geometry_scorer_cpu_v2.log`). The added BF16 equal-maxima case
has argmax3 but topk-first201 under the old scorer; it now passes exact equality,
while changing the maximum still fails. This repair cannot explain the original
TT forced43 PCC/top5 miss and does not change production execution.


During `batch32_hf_factored_v1`, first B32 prefill required about275s after
setup, longer than the reduced probes. Parent captured read-only live triage
with `timeout --signal=INT --kill-after=10 120 python tools/tt-triage.py
--llm-output --run=dump_callstacks --run=dump_running_operations --run=check_eth_status
--llm-output-path models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch32_hf_factored_v1_triage/tt-triage.txt
--triage-summary-path models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch32_hf_factored_v1_triage/summary.txt`.
Capture exited0 and the target independently reached DECODE step0 at533.63s;
no termination/reset or hung-device diagnosis. This is tracked diagnostic
correctness, never a clean performance measurement. Retained capture bytes
are2,194,677,760/rank across1857 traces, exactly the pre-run projection.


## Current all48 B32 factored integration: accuracy passes, state gate fails

`batch32_hf_factored_v1.json/log` exits1 after1305.641s internal/1311s external,
clean device close. Both baseline and migrated repeats contain2704 predictions:
prefill32 top1=30/32 (93.75%),top5/top100=32/32; decode2672 top1=2615/2672
(97.8668%),top5/top100=2672/2672. Every per-request accuracy gate passes.
Matching-history logits and device tokens are bitwise exact across all positions,
including reverse-slot migration and shrinking active widths. Dormant state and
all-inactive no-op pass. Four additional resident sampled steps pass with zero
token/position/page-table refreshes, stable IDs and coherent device QSA positions.
However all32 final request-state hashes differ across the repeats; the overall
artifact remains FAILED. No acceptance promotion is made.

Measured model capture:1857 native IDs,2,194,677,760 bytes/rank (exact projection).
All1864 captures include6 prefill+1 sampling, zero request-time releases;
2179 program entries remain stable.204 native-ID stability checks,207 program
checks,137 unchanged-membership replays,136 cached-plan identity reuses;
65 request membership uploads across both runs and resident transitions.
Final allocated/free/largest contiguous per bank are4222408640/49932736/46266496;
eight banks per rank. Actual shared pool has512 blocks per QSA pair, not the
plugin-expanded545-block setting. Allocation is not maximum-context proof.

Source inspection identifies a likely harness sampling-state mismatch: its
scalar seed0 describes only lane0 under the common sampler's deliberately
non-broadcast seed contract. Other requests initialize unseeded state, and the
migration reload again places the seeded lane at physical0. Full state equality
therefore requires an explicit per-request seed vector. Independent CPU
confirmation and a targeted reduced state repro are assigned; production seed
semantics and historical artifacts stay unchanged. Final all48 rerun will
follow the targeted check.

Started the independent serial watcher/B1 accuracy-resume/clean-perf queue with
`bash models/autoports/qwen_qwen3_8_flash_next/doc/full_model/run_acceptance_queue_v1.sh
> models/autoports/qwen_qwen3_8_flash_next/doc/full_model/acceptance_queue_v1.log 2>&1`.
The script preserves the prior fixed-path resume JSON, refuses existing output
paths, stops on failure, and honors acceptance_queue_v1.stop between jobs.
Current queue source records exact environment, bounded commands and logs.
No production edit is planned for the B32 test-seed issue; if diagnosis finds
one necessary, stop the queue between jobs before changing its measured code.


Current production watcher corroboration `serving_watcher_v2.json/log` PASSES
all16 reduced0/1/3 B2/C128 checks with tracking/program-cache checks enabled.
72.266s total,47.138s prepared,2180 program entries unchanged,22 retained
captures and0 request-time releases; caller mesh closes cleanly. Dedicated
`serving_watcher_v2_raw/generated/watcher/watcher.log` has0 ERROR/ASSERT/CORRUPT
matches. Watcher and profiler were separated. This validates current factored
and indexed-row/token-feedback kernels under watcher; it is not a performance
number or a substitute for the failed B32 test-state comparison. Queue advanced
to refreshed all48 B1 accuracy/resume after preserving historical resume JSON
as `prefill_resume_before_final_v1.json`.

Current production accuracy/resume evidence verified from
acceptance_accuracy_resume_final_v1.json (passed=true, all48, B1, capacity262144,
serial TT_METAL_TRACE_ALLOC_TRACKING=1). Prefill top1/top5/top100=97/100/100
over100 reference predictions; traced teacher forcing=96/100/100 over100.
Both exceed the unchanged top5>=98%/top100=100% user bar. Every recorded tt
source hash matches the inspected files; generator SHA256
f539343185c824cbc73b820c241b03d9d953fc73e0808fa421be0e21db4abc63.
Accuracy artifact SHA256
0e71968dc38b99a6187d32068f363dafbf10f3c33e10a77575f41a96053aa57f.

Current readiness/prefill_resume_check.json status=pass, natural prompt257,
block32/window128, unchanged PCCmean>=.99/PCCmin>=.98/top1>=.98. Starts32 and68
compare129 positions each, start256 compares1; all top1agreement1.0 and final
token equality. Global minimum reported PCC=.9999656772030441. Both no_history
and fresh_suffix controls fail as expected at32/68 (four negative controls
total); none runs at256 because control_min_window=64 excludes its one-position
suffix. The artifact label prefix_cache_hit at256 means retained full request
state here, not reconstruction from shared KV alone. Resume artifact SHA256
6eeb5c57ec36dc2954dcec99f36e9a82725a9f1f2b440bfd2db7f4107311fb71.
Reported PCCmeans slightly above1 are retained finite-precision diagnostic
values, not a claim of stronger-than-exact correlation or bitwise equality.

Accuracy run wall360.8146s and teacher-forcing TTFT11606.3275ms,
decode6.716107t/s/u (99 tokens/14.740682s), e2e3.795460t/s/u include allocation
tracking and explicit reference-token feedback. They are not clean token-out
performance. acceptance_perf_final_v1 was still incomplete at this audit; no
performance result is inferred from its setup-only report. Full48 B32 HF,
maximum-context execution and final review/commit remain independent gates.
No devices or HF generation were run by this documentation subagent.

Clean perf completed during the documentation pass: acceptance_perf_final_v1.json
passed=true, serial all48 B1, allocated capacity262144, prompt128/generate128.
Tracker0/tracebacks0, watcher/profiler off; all recorded tt source hashes match
current files. TTFT7038.67206396535ms includes public reset/setup, complete
prefill, first on-device sample and caller read.127 steady token-out decodes
average138.75426899451088ms =7.206985466080038t/s/u. Separate fixed-resident-token
model-only trace averages138.1526096085981ms =7.238372136676336t/s/u; it advances
positions but omits sampling/readback and is explicitly not teacher forcing.
Steady counters:127 model and127 sampling replays,127 caller reads; zero
token/position/page-table refreshes, captures, releases or resets. All12 QSA
layers on all4 ranks report position255 in each timing path. No clean-speed
claim is taken from tracked teacher-forcing6.7161t/s/u.
Artifact SHA256 eefb34d7685410328e880c766ab9628bbf06b758ae80c30ca4dc5d5779f70373.
Model-only budget comparison uses the inherited110.3006ms layer-stack estimate,
not a hard lower bound; its28.4537ms token-out overhead remains analysis work.
Completed clean performance does not establish maximum context, B32 HF,
remaining profiler/advice analysis, stage review or commit acceptance.


## 2026-11-14 continued acceptance and scoped numerical disposition

The serialized `run_acceptance_queue_v1.sh` completed all three jobs with exit0:
current watcher_v2 (75s external), accuracy/resume_final_v1 (367s external), and
clean perf_final_v1 (249s external). Raw logs retain clean device teardown.
The previous reporter-only exception is absent from both final acceptance jobs.

Started `run_batch_followups_queue_v1.sh`, exact commands/environment and bounds
in that script, driver log `batch_followups_queue_v1.log`. Reduced explicit-seed
B32 state test passed (137.131738s internal/exit0); current actual common-sampler
geometry/HF test passed (35.556325s/exit0). Full48 B32v2 is still running and is
not accepted by these reduced results. `batch_geometry_current_evidence.md`
records exact metric denominators, seed/state scope and actual survivor44 token.

The owner disposition is `batch_geometry_disposition.md`, with a scoped entry
in `doc/eval_readiness/limitations.json`. Current forced43 TT and matched original
HF both fail frozen PCC limits, while current TT top1/top5 agreement and direct
HF top5/top100 all pass. Frozen failures remain FAILED; the inherited precision
policy remains unchanged. Same-history exact state/position/migration and all
full-model HF bars remain requirements. Independent stage review must assess
this controlled limitation; this is not stage acceptance.

`final_capacity_reconciliation.md` distinguishes measured loaded/prepared/final
snapshots from calculated plugin-expanded pools. B32 extra33 pages cost
12,976,128B/rank; B1 extra2 pages cost786,432B/rank. Existing reserves cover the
observed workloads plus these calculated additions. Expanded pools are not
falsely described as executed, and no context reduction is made. Current
`runtime_fallback_audit.md` was refreshed without changing its historical snapshot.

The model-root README now exposes current clean TTFT/token-out numbers and
separates the tracked accuracy teacher-forcing and model-only timing boundaries.
`performance_boundary_audit.md` establishes stage5 capacity128 versus current
capacity262144 and source-exact QSA width scaling. It does not assign the28ms
gap entirely to QSA. Final reduced profiling now takes `--capacity262144` via a
minimal validated CLI option; five CPU tests pass in
`terminal_capacity_scope_cpu_v1.log`. No production code changed for this audit.
`run_terminal_final_v1.sh` and `terminal_final_commands.md` contain the exact
capture/coverage/tt-perf-report recipe, not yet executed.

Pinned Black/isort/autoflake checks over74 new stage Python files and
`git diff --check` passed (`final_*_check.log`, file list
`final_python_files.txt`); profiler-only follow-up formatting passes separately.
The standard autoregressive checker exited0 with "No degenerate output detected"
(`autoregressive_degeneracy_final_v1.log`). Final combined stage gate still waits
for completed serving/context evidence. `run_max_context_final_v1.sh` prepares
the once-only262143-token public prefill plus2 outputs with a43200s bound;
no max-context attempt has run yet. No local parent-stage commit or push.


Full48 B32v2 completed PASS,1296.826094s internal, clean process exit0:
`batch32_hf_factored_v2.json/log`, `batch_followups_queue_v1.log`. Baseline and
reverse-slot migration at step32 each score2645/2704 top1 and2704/2704 top5/top100.
Per-request and per-phase HF gates pass. Same-history logits/tokens/final state
are exact; final/dormant field mismatch dictionaries are empty. All-inactive
no-op and four resident-feedback transitions pass. This closes the scalar-seed
probe defect after its reduced proof, without production changes or rewriting
B32v1. Actual final per-bank allocated4222408640/free49932736/largest46266496,
identical to v1's measured allocation. Started final reduced C262144 profile via
`bash doc/full_model/run_terminal_final_v1.sh` (full path from checkout), driver
`terminal_final_driver_v1.log`; profiler and watcher remain separate.

## 2026-11-14 final profile and once-only maximum context

The final reduced profile completed with clean exit0 via
`bash models/autoports/qwen_qwen3_8_flash_next/doc/full_model/run_terminal_final_v1.sh`.
`terminal_final_ops_v1.json`, `terminal_final_perf_v1/coverage.json`, all three
advice-enabled report CSV/text pairs and `terminal_final_analysis.md/.json`
record real layers0/1/3, B1, capacity262144, two warmups and one measured eager
iteration. Coverage independently reproduces21488 host/device/report ops and
six signposts, without omissions. The current 39 source hashes agree.
Merged kernel sums are11.256808ms model,0.429263ms split greedy and2.886563ms
force-argmax; both select true greedy token243975. Split is3.673% of the
model-plus-split sum and force-argmax is6.724x slower. No sampler dominance or
new terminal rewrite is indicated. Eager kernel sums are not traced throughput.

Named terminal costs are embedding0.036766ms, final mixer0.372330ms,
head/layout0.668237ms and native row fill/write0.045470ms. The QSA indexer score
graph is2.387150ms at the full65536 compressed-key width, versus historical
C1280.025220ms; its native selector stays about0.010177ms. This supports the
capacity-dependent scorer diagnosis without assigning an exact additive all48
cost to a reduced eager profile. `performance_boundary_audit.md` retains the
historical layer-stack estimate110.300609ms and explicitly scopes its C128
geometry against current C262144. BF16/HiFi4/FP32DEST/block1 head policy is
visible in the final report; no head/sampler advice remains. Twenty inherited
decoder advice rows do not override the stage5 selected/rejected policy.

Two profiler limitations are classified: the optional UI WASM-copy warning
does not affect raw capture or strict CSV coverage; some BinaryNg cached shape
annotations are inconsistent, so no FLOP/bandwidth claim uses those shapes.
Adjacent cast/reduction metadata and frozen source order identify the scorer.
The raw final capture remains in place at `terminal_final_tracy_v1/` (about
9.225GB), excluded from git by the stage-local `.gitignore`; it has not been
compressed, moved or rehashed. `raw_artifacts.json` records the exact persistent
path and reproduction/report paths. Compact reports and analysis are committed
separately when stage review passes.

Started the once-only maximum-context acceptance at06:41:45UTC using
`bash models/autoports/qwen_qwen3_8_flash_next/doc/full_model/run_max_context_final_v1.sh`.
It uses all48 layers and a262143-token non-aligned public prompt plus2 sampled
tokens, consuming262144 positions, under a43200s SIGINT timeout. The exclusive
`max_context_attempt.json` latch prevents accidental reruns. Status is RUNNING;
heartbeat counts denote host-submitted prefill replays, not completed device
work. No concurrent hardware command or reset is launched. Final serving/context
closure, combined stage check, independent review and parent commits await the
actual result; the advertised context remains262144 throughout.

## Stage-cap wrap-up instruction

The user issued the stage time/token-cap wrap-up instruction while the
once-only maximum-context acceptance was already running. Keep the current
validated implementation; no further exploratory sweep or optional device
experiment is planned. Remaining required work is that same context run,
final context/serving reconciliation, the stage check script once, independent
stage review and local stage-owned commits. Do not interpret the cap as a
passing result for an unfinished correctness or capability gate.

### Untried ideas for later stages

- Exact all48 component decomposition remains unmeasured. Current reduced
  eager attribution and clean full-model timing have different boundaries;
  no additional full48 profiler run is proposed.
- Larger live-history QSA scorer/selector optimization could be evaluated by
  a later optimization stage while preserving the selected precision and
  context contract. This stage retains the accepted optimized decoder.
- Literal plugin-expanded8194/545-block pool executions have not been run.
  Current arithmetic is explicitly calculated from measured private/shared
  snapshots; future serving integration must validate its actual allocation.
- BinaryNg cached shape annotations merit a separate profiler metadata
  investigation. Current coverage and adjacent tensor shapes support the
  timing attribution, but anomalous fields cannot support FLOP/bandwidth claims.

These are hand-off notes, not waivers for required acceptance or a
`pass-with-carried` stage-review verdict.

## Owner-authorized final closure after maximum-context PASS

Owner resumed once to finish stage6 and explicitly prohibited new tuning.
The original once-only run completed; it was not restarted. Verified
`max_context_attempt.json` matches the nested record in
`acceptance_max_context_final_v1.json` exactly. All48 layers, serial TP4/EP4,
262143 logical prompt tokens plus2 generated tokens consume262144 positions.
The trace counters show8196 prefill replays,1 model decode replay,2 sampling
replays,0 token refreshes,1 request-boundary position refresh,0 page refreshes,
2 caller readbacks,8 first captures and0 releases during generation. All12 QSA
layers on all4 ranks end at262144; final active-row logits are finite on every
rank. Sampled IDs are1292 and7285. This is capacity/execution evidence only.

The prefill/decode check took14895.095443s; driver wall time15073.876830s
includes178.742352s setup. Tracking was enabled, program-cache allocations
included, watcher/profiler disabled. No clean-performance claim is taken from
this long tracked acceptance. The existing clean128/128 TTFT and token-out
metrics remain authoritative. All36 recorded TT source hashes match current
source. `max_context_closure.json` records this CPU evidence verification.

The log contains `MAX_CONTEXT_DONE` pass, `ACCEPTANCE_DRIVER_DONE` passed=true
after generator/mesh close, and completed UMD teardown at10:53:04UTC. No test
process remains. The interrupted outer wait driver has only its START marker,
so its numeric process exit code is unavailable and is not fabricated.

After final finiteness helpers, allocated/free/largest-contiguous DRAM bytes
per bank are4099126528/173214848/165250368. Before those helpers, memory_after
is4099118336/173223040/165258560. Eight banks/rank make final free1385718784B;
subtracting the calculated786432B worker-pool expansion leaves1384932352B.
These remain snapshots and calculated translations, not transient high-water
measurements or a literal8194-block run. Supported context stays262144.

Remaining work is documentation reconciliation, the final stage check,
fresh independent review and local stage-owned commits. No additional tuning,
device measurement, capacity sweep or vLLM implementation is planned.

Final required check executed once after reconciliation:
`source python_env/bin/activate; source /home/ttuser/qwen38-flash/lane-env.sh;`
`TT_VISIBLE_DEVICES=0,1,2,3 bash "$TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/06-full-model.check.sh"`.
Exit0, `stage_check_final_v1.log`: no degenerate output, context target/supported
262144/262144, serving63 rows with42 supported/21 not_supported/0 missing.
`git diff --check` also passes. No Python/native implementation changes were
made after the recorded pinned formatter checks and final device measurements.

## Final independent review

Fresh xhigh round1 reviewer returned **clean-pass** in
`stage_review_round1.md`, with no Required Work and no additional hardware run
or tuning required. It independently checked current source hashes, actual
HF/TT outputs and saved logits, B32 request/state maps, profiler sums and the
maximum-context completion provenance. The active-width numerical sensitivity
is accepted as a scoped matched-HF known limitation, not a human waiver or a
retroactive pass for historical failures. All scripted/runtime gates are
complete. Final work is isolated local checkpoints and SHA recording; never push.

### Carried to next stage

No blocking review finding is carried. The existing untried-ideas list remains
a hand-off: literal serving-worker expanded pools, wider prompt coverage and
long-context semantic/performance evaluation, exact all48 component attribution,
and BinaryNg profiler metadata investigation. These are not claims of current
coverage, tuning performed in this closure, or required stage6 gates. Native
row-reduction regression is scoped rather than exhaustive over every special
floating-point pattern. Future serving integration must retain the recorded
cache, seed, active-width and trace-ownership limits.

Checkpoint isolation includes707 stage-owned paths, including intentionally
force-added raw logs, profiler CSVs and saved tensor fixtures. It excludes all
unrelated `doc/fused_decoder` artifacts, Python caches and the9.225GB raw final
Tracy capture retained at its recorded persistent path. The authored
code/documentation staged whitespace check passes. An unrestricted staged
whitespace check flags unchanged generated log/table spaces and CSV CRLF;
those raw evidence bytes are deliberately preserved rather than reformatted.
This does not change the final formatter/test outcomes or reviewed source.

The first checkpoint commit was stopped by repository hooks, recorded in
`checkpoint_hook_attempt_v1.log`. All171 hook-modified working files were
restored exactly from the reviewed index. No hook mutation was accepted.
Scoped checkpoint exclusions preserve raw evidence and historical source:
`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `yamllint`,
`black`, `isort`, `check-large-files`, `prefer-expect-error`. Generated inspector
YAML contains unquoted kernel text; raw logs/tables retain formatting, reference
cache JSON retains original bytes, and evidence fixtures exceed500KB. Formatter
changes targeted archived snapshots and inherited decoder formatting; final
new production Python already passed the recorded pinned checks. The fixture
preference finding concerns four CPU cache-budget ValueError assertions, not
a runtime failure. These repository-hook exceptions are disclosed and are not
represented as passing hooks. Remaining native/API/import hooks run normally.
No runtime or required acceptance gate was skipped or invalidated.

## Stage6 complete: local checkpoints

- Repository `/home/ttuser/qwen38-flash/tt-metal`, branch `main`:
  `8d53247c2b8b552c1f04efc7336bc5ec9db74349` — reviewed implementation and708 isolated stage-owned paths.
- Repository `tt_metal/third_party/tracy`, detached HEAD:
  `390ef12730f570dfec54b6407f3735d03213d1c0` — CSV export repair.
- Neither repository was pushed. Plugin and tt-transformers checkouts remain
  unchanged; unrelated fused-decoder artifacts remain untracked and untouched.

The successful parent checkpoint retained all measured source/evidence bytes;
no working-tree diff remained afterward. The final metadata-only commit records
these SHAs and the complete status. Independent review is clean-pass, no
Required Work remains, context stays262144, and all required gates are complete.
No new tuning, maximum-context rerun, or vLLM work was performed during closure.
