# Stage 04 work log

Model google/gemma-4-26B-A4B-it, revision4d7ae4984b7db7de8f8457170b3f1a419ee76d52.
Starting clean checkout2e3a1779d3, optimized implementation checkpointa9259624f2.
Only multichip decoder, stage tests/docs and context contract changed. No push.

## Startup and topology

Enabled tt-model-bringup/tt-autodebug plugin inventory verified in selected
Codex home's config. Packaged scripts/environment.py returned valid exports.
`timeout 60 tt-smi -ls --local`: exit0, four Blackhole P300c ASICs.
`set_fabric_config(FABRIC_1D); open_mesh_device(MeshShape(1,4)); close_mesh_device`:
exit0, MESH_SMOKE_OK. Unknown B850M-C motherboard warning falls back to bus IDs;
auto-discovery reported matching physical/logical degree histograms and all4
opened. No failure is inferred from that warning.
Read llms.md3.3, common Attention1D/MLP1D/RMSNorm1D, GPT-OSS CCL and sparse-expert
contracts, and the optimized decoder and evidence. Initial plan: mesh_plan.md.

## Initial experiments

Commands run as `HF_HUB_OFFLINE=1 timeout 240 python -m
models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder` plus flags
below. Reports/logs are named by output stem in this directory.

- probe_multichip_collective: BF16/FP32 reduce-scatter [1,1,32,2816] exact4.
- paired_initial: setup/runtime API mismatch, shared projection is callable;
  minimal fix calls it. Reset all devices after closed failed run:
  `timeout 180 tt-smi -r` exit0 (reset_initial.log), bounded list exit0
  (list_after_reset.log), next run opens full fabric successfully. No process
  killed and no locks cleared.
- paired_second: default layer0,length65,steps2; prefill.9996149855;
  decode.9998534455/.9999265616. Device-only guard passes.
- full_trace: --layer5 --trace --steps8 (length65). All replicas exact;
  deterministic repeated traces with refreshed positions; minimum PCC.9986253045.
- full_sharded_trace: same plus --sharded-residual. End-to-end sharded
  residual consumed by distributed norms/residuals and gathered only for input
  projections and comparison boundary. Minimum PCC.9984544464. Traced host
  latency about1.03ms versus.91ms replicated in these short controls; not a
  headline or final optimization claim.

The required4096/128 workload and final optimization/validation are in progress.
No clean-pass review or stage completion commit exists yet. Native profiler,
maximum context, cache ownership, batch, stack and watcher gates remain open.

## Geometry and topology experiments

`run_multichip_decoder --layer 0 --length 4096 --steps 128 --trace`
produced sliding_headline_v0/v1/v2.json. V1 uses 12 gate/up and 88 down
cores with prefill K-block 11 instead of 1. V2 uses explicit QKV and WO
projection configs. Sliding v2 minimum PCC .999855352, warmed host medians
224246 us prefill / 864.36 us decode, versus paired TP1 221219 / 824.99 us.
Full attention v1 (`--layer 5`) minimum PCC .99778229, TP4 host medians
191667 / 911.81 us, TP1 186599 / 876.54 us. Host times are not device metrics.

Native v0 profile collected with `python -m tracy -r -p -v
--op-support-count 100000 --no-op-info-cache
--disable-device-data-dump-to-files --disable-device-data-push-to-tracy
-o <stage>/profile_v0 -n tp4 -m <tests>.run_multichip_decoder
--tp 4 --layer 0 --length 4096 --trace --steps 1 --profile`.
Final CSV and tt-perf-report tables are under profile_v0; this one-decode
profile is diagnostic, not the required target measurement. AUTOTRIAGE.md
refutes the apparent shutdown hang: the run exited successfully without
intervention.

AutoDebug proposed EP4 to avoid narrow TP expert projections. The component
experiment owns 32 complete experts per rank and uses runtime sparsity counts;
fixed nnz=8 would be invalid for partitions with 0..8 selected experts.
Component correctness/trace/zero partitions passed, but the all-check watcher
run asserted in fabric Ethernet teardown after EP_PROBE_PASS. This is not
watcher-clean evidence. AUTOFIX_ep.md records recovery and scoped experiments;
AUTOTRIAGE_watcher.md is being prepared independently. No C++ changes.

## Attempt stop after scoped AutoFix failure

Paired EP runs used `--expert-parallel --length 4096 --steps 128 --trace
--check-cache`, layer 0 and 5, output sliding_ep_v2.json/full_ep_v2.json.
Both exited 0; per-layer host timing/speedup/efficiency and accuracy are in
candidate_summary.json. No final path is accepted.

CCL-only all-check Watcher control exited 134 after both reductions passed;
normal two-ERISC and supported single-ERISC modes both failed teardown.
Commands and distinctions are recorded in AUTOFIX_watcher.md. After each
failed process had exited, `timeout 180 tt-smi -r` and bounded list exited 0.
Final recovery files: ccl_single_erisc_reset.log, final_device_list.log,
final_mesh_smoke.log (FABRIC_1D1x4 open/close exit 0). No processes were killed
or locks removed in these controls. AutoFix found no supported scoped remedy;
C++ fabric repair requires a separately authorized scope.

An untested fused-tail edit was reverted. Current runtime is only formatting
different from the measured v2 EP snapshot; source_provenance.json confirms
AST equivalence except top-level import ordering. All 8 stage Python files pass py_compile and pre-commit
(black/autoflake/isort and other applicable hooks); precommit.log is retained.
No build needed. Prepared stack/batch/fused probes are explicitly unrun.
Telemetry JSON was produced locally from the exact supplied template.
Stage remains incomplete; no clean-pass or completion commit is claimed.

Final staged pre-commit run passed after the whitespace hooks normalized two
generated text reports. Preserved precommit_staged.log.gz records the successful
rerun. The local checkpoint is an incomplete/blocked attempt, not a stage-pass
commit. No push is authorized or performed.

## Local checkpoint

Repository tt-metal, branch gemma-4-26b-a4b-it:
`41a8c80072c3ce2f0480cd4bbfbefeccfedc8ddc` — incomplete/blocked attempt
checkpoint, not stage completion. Commit hooks passed. Identity was supplied
per command as Codex <codex@openai.com>, matching prior stage commits; no
global/repository identity configuration changed. Nothing pushed.

Independent fresh xhigh stage-review returned **more-work-needed**
(stage_review.md): fabric Watcher gate, final capability/context/batch/stack
validation, topology/geometry/default selection, and target native profiling.
The review verified numerical summaries, diagnostic windows, memory arithmetic
and provenance. No additional arithmetic defect was demonstrated. These gates
remain work after the separately scoped fabric repair; AutoFix's failed scoped
workaround is the stopping condition for this attempt, not the review verdict.

## Authorized fabric teardown repair

The separately authorized infrastructure investigation confirmed that all
post-failure board resets and four-device `tt-smi -ls --local` checks had
completed with exit 0. A fresh list also showed all four p300c devices before
the repair. After a deliberate short pause, the router teardown was changed to
fully drain NoC work and clear the current ERISC's packet tags before the final
two-ERISC rendezvous and termination publication.

All applicable pre-commit hooks pass. The previously failing model-free
Watcher control now passes BF16 and FP32 reduce-scatter and exits 0. The
original EP Watcher probe passes all 16 comparisons, prints `EP_PROBE_PASS`,
and exits 0 with normal driver shutdown; minimum local PCC is 0.9994366683.
Full analysis, upstream comparison, rejected bypasses, commands, and remaining
gates are in `FABRIC_TEARDOWN_INCIDENT.md`. Stage work resumes from this point;
these focused results do not claim stage completion.

## Resumed attempt bff5abf7-6db4-4696-8fc0-a904af9d0396

Resume starts at9a529836fc91b1117a48f6d63ef30455a69cd42d, whose separately
authorized fabric cleanup passes the model-free and EP all-check Watcher
controls. Existing model results remain candidate evidence. Fused CCL probe
agent owns hardware exclusively while the coordinator prepares a fused-tail
A/B: baseline's sharded RMSNorm and fused residual/scalar tail replace the
current repeated FP32 normalization path. This is opt-in, unvalidated until
paired real-weight runs; no default or precision-policy change yet.

Fused-tail sliding A/B (`run_multichip_decoder --fused-tail --layer 0
--length 4096 --steps 128 --trace --check-cache`) exits 0; minimum output PCC
.9998675537, local cache PCC .9999999987. Host medians TP4 prefill218609us /
decode816.58us, paired TP1 221298us /825.18us. Older unchanged TP-tail v2
decode was864.36us. This supports the fused-tail candidate; full attention
is being checked separately. Host measurements are not device telemetry.

Full-attention fused-tail target run also exits0: minimum outputPCC .9979092160,
cachePCC .9999715022; host medians TP4 185068us prefill/811.09us decode,
pairedTP1 186445/875.85us. Fused-tail candidate retained opt-in.

Next isolated candidate combines EP-prefill and indexed TP-decode behind
`--hybrid-experts`, using the previously audited dual-layout memory bound
29,085,106,176 bytes/device. The runtime dispatch uses the inherited expert
contract: one logical token selects indexed TP, multiple padded prefill rows
select EP. Both paths keep active routing. This candidate is unvalidated,
not yet default. Full-stack capacity remains calculated, not measured.

Hybrid expert and shared-policy follow-up: `sliding_hybrid_tail.json` and
`full_hybrid_tail.json` pass4096/128; EP-prefill reduces warmed prefill host
latency to about95ms/81ms while preserving indexed TP-decode. The initial
stack failed full-attention decode PCC (.994799/.993103). AutoDebug traced
a precision-policy mismatch in the shared MLP; isolated BFP8-sliding/BFP4-full
LoFi decode policy restores stack PCC .999832/.999773 with identical prefill
results. See AUTODEBUG_stack.md, AUTOFIX_stack.md, stack_shared_policy.json,
and paired sliding_shared_policy.json/full_shared_policy.json. These are
candidate tests, not final-default acceptance. Device-only guards and repeated
trace equality remain enabled. Batch32 sliding hybrid passes prefix/slot
preservation and refreshed request/page ownership; it predates shared-policy
selection and will be rerun on the final path.

Long-prefill capacity audit found native concat batching and non-aligned
untilize could retain five full-size activation buffers. Model-local bounded
concat retains tile-aligned chunks, merges groups of32, releases groups before
final logical slice. This restores the three-buffer bound without changing
the public logical-length contract. Paired1025 sliding and32769 full runs
pass output/cache PCC. `sliding_max_capacity.json` also passes262143+1,
minimum outputPCC .9999608401, cachePCC .9999999995, with21,474,836,480 bytes
of anonymous other-resident DRAM per device. This tests capacity with all-layer
weight/KV expectations reserved; it is not a full-model computation or allocator
fragmentation proof. Full-attention maximum-context run is in progress.

Fused collective experiments: Linear fused MMRS stalled; triage was captured
before terminating only the probe, followed by reset/list/mesh recovery. Ring
adaptation works and tuned MMRS loses to same-contract separate producer and
distributed-norm consumer. Tuned AGMM wins its component control; a whole-layer
Ring/sharded A/B remains required. See AUTOFIX_fused_ccl.md and fused_ccl_plan.md.
No unsupported architectural rejection is inferred from the Linear stall.

Full maximum-capacity execution: original paired `full_max_capacity.log`
completedTP1 but its later host progress became unclear. Captured
full_max_triage.txt and full_max_triage_second.txt before terminating only
PID19790 (exit143). Bounded reset/list/fabric mesh-smoke all exit0.
AutoFix initial investigation proved117.3s of initial per-offset SDPA
compilation but did not prove the later delay's root cause. The unchanged
TP4 diagnostic retry completes both256-chunk prefills, deterministic trace,
replica equality and normal process exit. The unchanged original paired
rerun then exits0: `full_max_capacity_verified.json`, minoutputPCC .9999247236,
mincachePCC .9999710212,21,072,183,296 bytes other-resident reservation perdevice.
See AUTOTRIAGE_max_context.md and AUTOFIX_max_context.md. No runtime/fabric
fix was made for this episode, and no persistent kernel hang is claimed.

Both layer kinds now exercise262143+1 at the HF262144 limit. The candidate
hybrid+shared weight bound is11,000,048,640B/device, conservative peak
29,220,105,216B before reservation rounding. Rounded bounds including2GiB
reserve are29,261,726,208B sliding /29,267,128,832B full. Context contract
records the exact scope: anonymous resident reservation, not full-model
compute or fragmentation proof. Candidate selection remains open.

The inherited nonzero-prefix path also retained one physical32-row tile for
one logical token until final concatenation:262143 outputs would consume
47,244,460,032 bytes per replicated rank before other tensors. The model-local
continuation override preserves exactly the per-token decode/cache updates,
collapses32 logical rows into tiled blocks, then merges32-way tiers. Last-row
references supply inert output padding only; no padded cache update occurs.
`sliding_long_prefix.json` and `full_long_prefix.json` pass B1prefix31 plus
continuation1025 (total1056), final trace, prefix preservation and fallback
guards. Sliding prefill/decodePCC .9999549786/.9999517512; full
.9999147818/.9997500998. B1 reordering is an identity control; meaningful
ownership reassignment remains the heterogeneous batch32 check. These tests
validate the output assembly boundary, not a maximum-length prefix latency
claim. Source audit: prefill_continuation_capacity_audit.md.

Fused AGMM integration is now measured with compatible hidden-sharded
residuals and Ring fabric. Both attention kinds pass4096/128 output/cache PCC,
exactreplay and runtimeguards. The1D+L1adaptation matches the baseline's
projection geometry and placement: sliding separate936.42us/fused943.43us;
full937.34/933.93us host medians. It does not beat the replicated candidate's
~818/812us overall layer. 2D and1D/DRAM intermediates are preserved as controls;
no family is rejected on a first API mismatch or restored-replicated-only
measurement. See AUTOFIX_fused_ccl.md and its source snapshots.

Shared decode geometry1 (11x4 GU/down) and grouped shared/routed reduction
passed both4096/128 runs. Grouping preserves the independent branch norms
and exactly matches the ungrouped candidate PCC vectors. Host decode drops
from769.43/761.41us to750.40/741.93us sliding/full. Shared DRAM-bank sharding
also passes, but783.62/774.58us loses togeometry1; it is not the default.
See shared_dram_grouped_results.md and the runtime snapshots.

Decode projection fidelity experiments keep prefill unchanged. Sliding QKV
and WO LoFi combined pass real-weight128-step PCC/cache and repeat checks
at748.79us, minimum outputPCC .998953932; full WO LoFi passes at741.31us.
Attention CCL BF16 then passes both kinds at735.84/728.62us. Full BFP8
passes at723.98us, while sliding BFP8 fails exact repeated trace equality.
The failed process closed normally; reset/list/mesh smoke all pass. AutoFix
is investigating this boundary with source diagnosis and isolated frozen-
payload/runtime-cast probes. The gate is not relaxed and the failure does
not establish general BFP8 CCL unsupportability. Refreshed EP-only decode
is slower under otherwise matching policies:808.40us sliding BF16 and
798.71us full BFP8. All these intervals are host medians, not device metrics.
See attention_ccl_dtype_results.md for commands, PCC, hashes and recovery.

Attention QKV and WO DRAM-bank candidate patches are prepared separately,
with CPU geometry checks and full-stack extra-weight accounting. They remain
unapplied pending serialized hardware experiments. Final defaults, native
profiles and final correctness/Watcher coverage remain open.

All four attention DRAM-bank candidates passed the paired4096/128 gates but
were slower than matched controls. QKV sliding/full host medians736.70/730.12us;
WO743.37/731.68us; controls735.84/723.98us. FullQKV PCC .995221 remained
above threshold but lost precision and latency. No candidate required API
adaptation. See attention_dram_results.md; optional helpers remain diagnostic.

BF8 sliding replay investigation reproduced the same failure in a second
unchanged run, then localized the original-class output-only diagnostic to
step20/position4116:1114 finite values changed on each replica, maximum0.0625.
Four native cast/precast BF16/BF8 probes pass128 seeds with3 duplicate replays.
The instrumented layer retaining all intermediate handles passes all128 steps,
so allocation lifetime/read timing sensitivity remains unresolved; it is not
a fix or evidence accepting the candidate. See AUTODEBUG_bfp8_ccl.md and
AUTOFIX_bfp8_ccl.md. Recovery completed before each subsequent hardware owner.

Applied selected_policy.patch after attention DRAM experiments. Runtime
8b59370c and runner06f0a057 now share explicit constructor precision policy
with stack/contracts and consistent BooleanOptionalAction defaults. Default
hybrid/fused-tail/optimized-shared/grouped-reduce are enabled withgeometry1,
LoFi decode QKV/WO and BF16 attention CCL. Full-onlyBF8 remains an override
until expanded contract checks pass. Stack trial started; final acceptance
is still pending.

Full-only BFP8 collective override fails selected-policy two-layer stack
replica equality during traced decode (`stack_selected_full_bfp8.log`).
Process closed normally; serialized reset/list/fabric-smoke all exited0,
four ASICs visible (`stack_policy_{reset,list,smoke}.log`). No processes killed.
Identical BF16both control passes (`stack_selected_bf16.json`): minimum
layer0decodePCC .9999121782, layer5decode .9996771634, fullprefill .9989911988;
exact stacked replicas/replays and runtime guard pass. Fresh source diagnosis
for the stack failure is ongoing alongside the sliding BFP8 investigation.

Selected BF16 sliding batch32 contract passes, lengths32–63, prefix31 plus
continuations1–32, randomized disjoint pages and in-place reversed request
ownership. Minimum prefill/decodePCC .9998000599/.9976372595. Prefix/other-slot
cache preservation, exact replicas/replays, refreshed ownership and fallback
guard pass. Artifact batch32_sliding_selected.json, runtime8b59370c.

Selected BF16 full batch32 passes the same ownership/cache/replay gates;
minimum prefill/decodePCC 0.9997261667/0.9994612386.
Artifact batch32_full_selected.json, runtime8b59370c.

Native accounting source audit found RS OUTPUT_0 represents allocated DRAM
scratch, not its reduced logical output. Multichip-only summarizer now excludes
that allocation from logical operand traffic and reports it separately; original
whole-layer windows and useful FLOPs are unchanged. Actual-v0 regression removes
5,767,168 estimated bytes. This is an accounting correction, not a performance
measurement or runtime improvement; see native_accounting_audit.md.

Correction to the BF8 boundary experiment interpretation: the diagnostic's
expanded post-attention norm bypassed OptimizedDecoder.normalize's decode
width-sharded RMSNorm implementation. Its passing runs are unsharded-norm
controls, not exact-operation retention controls. Symbolic forward-order
checks did not inspect the overridden helper and were insufficient. The
diagnostic is being corrected to retain the actual sharded normalization
path; no runtime fix or cause has been inferred from those passes.

Corrected v3 boundary capture reproduces failure atstep6/position4102 with
actual sharded post-attention normalization. First difference is routed_local
on rank1 only (2697 values,max .01953125); all upstream attention/CCL/norm,
router scores/IDs/routes and expert_input are exact between replays. Shared
branch stays exact. Routed BF16 reduction propagates the changing local value.
This localizes the problem to indexed TP experts for these real inputs, not
the attention collective itself. Fresh AutoDebug_indexed_expert investigation
and frozen-input expert-stage probes are required before declaring the
determinism issue resolved. Artifact bfp8_boundary_v3_all.json records the
full boundary comparisons; its tensors remain local and outside telemetry.

Native selected-sliding capture failure: profile_selected_sliding.log records
PERF_PREFILL/PERF_DECODE complete and TP_DONE4, followed by child SIGABRT
during close. No cpp_device_perf_report.csv or raw device log is produced;
the later Python report error is secondary. No native measurement is accepted
from this run. Original command uses ordinary Tracy-r-p,250000op support,
uncached op metadata, disabled raw-file/device-Tracy dumps, default C++ marker
validation and4096/128. Root preserved host capture/logs, stopped only its
auto-started GUI serverPID76960, then list/reset/list/FABRIC_1Dsmoke all exit0.
FourASICs visible. No target process was killed; it had already aborted.
Apport core pattern exists but no local core or /var/crash artifact was found.
Profiler AutoFix investigation/retry remains required; old fused-stage clock
rollover diagnosis is a hypothesis only without matching endpoint evidence.

### Native profile recovery and current expert localization

The selected sliding100k native capture completed normally and passed all
four-device prefill/128-session marker checks. `AUTOFIX_profile_selected_abort.md`
records the uint32 profiler allocation overflow and bounded100k workaround.
Human tables and CSVs were generated in separate tt-perf-report invocations,
compressed with SHA256 provenance. Whole-layer accounting retains profiling
gaps; prefill338681.719us and decode800.823us are current-candidate device
measurements, not final accepted default performance. `optimization_advice_audit.md`
records the actionable current rows. A sharded decode RoPE candidate was prepared
as an unapplied patch from source contracts; hardware measurements remain pending.

Expert AutoFix: gate K88 and40ms host delay do not remove original-class replay
failure. Retaining gate/up raw passes128, but retaining hidden alone reproduces
at position4096: rank1 first32/192 columns change across8expert slots with identical
logical and physical expert operands. Frozen logical uploads erase nonfinite
padding and do not clear this same-context failure. Physical-only padding changes
are being recorded separately from logical output failures. Hardware remains
serialized under the dedicated diagnostic agent.

### Expert sender ownership experiment

The exact physical-input standalone expert control passes128 repeats; original
whole-layer K88 and40ms host-delay controls still fail. Watcher initially exceeds
ACTIVE_ETH config capacity (28464>26624 bytes), then NOINLINE=1 preserves full
NoC checks and passes128 steps/normal close with8 completed dumps. This is an
instrumentation-sensitive control, not a release-mode fix.

N2/K44 gate/up on6x1 workers (down unchanged) fails in the original output-only
path at step7. The corresponding slice/address diagnostic fails at step0 on
rank2: gate changes254 finite elements across the first64columns, with per32tile
counts126,128,0,0,0,0; up remains exact. N1 failures affected first32columns.
The changing columns therefore follow sparse sender worker0 ownership across
geometry changes. `bfp8_boundary_n2_slices_addresses.json` preserves221 scalar
allocation records without adding retained tensor handles. Further source audit
and a same-dtype input-memory control are in progress. No geometry or lifetime
change is accepted as a fix.

The stack test now labels failed reads by TP, phase, layer, step and replay,
records nonfinite and replica-difference counts in a compact `.failure.json`,
and reports changed counts by layer for replay mismatch. Forward/trace programs
are unchanged. Python compilation and Black check pass; next stack execution
will exercise these diagnostics if the separate full-BF8 failure recurs.


### Resume: integrated router placement, full-attention failure

Continued existing runtime070613dd without restarting the stage. The ordinary
paired `full_router1_ccl_bfp8.log` invocation (`--layer5 --length4096 --steps128
--trace --check-cache --attention-ccl-dtype bfloat8_b`) fails at decoder-runner
line315, `read(y)`, with `Replicas differ`. The old assertion cannot distinguish
nonfinite values from finite rank divergence. TP1 completed; TP4 closed normally.
No full-attention result or latency is accepted from this failed run. Sliding
ordinary integrated placement remains a passing128-step candidate only.

Recovery: bounded `timeout180 tt-smi -r`, `timeout60 tt-smi -ls --local`, then
FABRIC_1D MeshShape(1,4) open/close all exit0; logs `router1_full_reset.log`,
`router1_full_list.log`, `router1_full_smoke.log` (`MESH_SMOKE_OK`). No stale job
was killed, no locks cleared, no second reset required. Exclusive hardware
handed to the existing AutoFix investigator to localize full-attention failure
and compare BF16/BF8 under the integrated placement. No low-level fix claimed.

CPU work prepared `combined_geometry_runner.patch` and its candidate/provenance,
combining opt-in sparse and dense geometry controls with incompatible-backend
validation. Runtime and ordinary runner were not modified by this preparation.

CPU all-reduce audit identified another diagnostic contrast: the generic helper force-frees its input before AG allocation, whereas the retaining diagnostic invokes RS/AG directly. `allreduce_lifetime_control_plan.md` records a one-boundary control and storage arithmetic; investigator notified. This is an unverified lifetime hypothesis, not a claimed CCL bug.

Correction/refutation: the preceding forced-deallocation hypothesis inspected the wrong MeshConfig. The model imports `models.demos.gemma4.config`, whose active unpadded allreduce does not force-free either tensor. The investigator caught the import mismatch before any control ran. `allreduce_lifetime_control_plan.md` now records the refutation; no code changed. TP4-only original-class diagnostic also passes128, so paired-runner versus diagnostic state/lifetime differences need isolation.

Context contract memory fields now describe the actual hybrid/shared defaults:11,000,048,640 weight bytes,24,790,920,192 resident+reserve bytes,29,220,105,216 conservative long-prefill peak bytes per device. The earlier TP-only values remain explicitly named as the reservation base; the runner arithmetic is unchanged. Default-selection flags reflect current code, while current-runtime capacity acceptance remains pending. Advertised262144 is unchanged. Arithmetic verified CPU-only.

Fresh AutoDebug source audit completed via independent CodexCLI(gpt-5.5/xhigh), since collaboration rejected another fresh agent at thread limit. Existing CLI workspace sandbox lacked bwrap; explicit danger-full-access matched the enclosing environment and allowed source tools. No hardware/import/implementation edits were authorized or performed. Report/prompt/log hashes are in `full_replica_fresh_provenance.json`; report `AUTODEBUG_full_replica_fresh.md`. Subsequent cleanup failure is separately attributed in a coordinator addendum. This is source diagnosis, not stage review.

### Integrated core(10,9) placement

AutoFix core1 allocation roundtrip fails step13 despite four new addresses;
core1091024 duplicate controls pass. Investigator recovery reset12/list12/smoke12
all exit0 and hardware returned to coordinator. Applied the one-coordinate
placement patch: runtimef57234ba. Ordinary runnerb592fa08 full-attention4096/128
BF8 trace/cache passes minimum PCC.9994477563/cache.9999715022, exact replicas
and duplicate replay. Sliding counterpart running. Kernel-state mechanism
remains unproven; no native C++ changes made.

Both ordinary integratedBF8 headline runs pass: slidingminPCC.9987453344,
full.9994477563; cachechecks/exactreplicas/duplicate replay pass. Mixed33-token
stack passesdirecthandoff, independentcaches, jointtrace, exactreplay:
stack_router109_bfp8.json (minPCC.9972870388). FreshmatchedBF16controls decode
hostmedians737.44us sliding/729.38us full versusBF8 737.56/724.29us.
No BF8 speedbenefit established for sliding; fullbenefit~5us.

Precision-locked sparseN2K44/N2K88/downN2K6 trials allpass paired4096/128:
sliding758.72/762.15/739.49us versusbaseline737.56us;
full756.21/758.50/725.28us versus724.29us. Reject wider sparse blocks as slower
under current indexed active8 LoFi(BFP8sliding/BFP4fullgate, BFP4down) policy.
Densegeometrysweep nowrunning, with one role changed percommand.

### Optional sharded RoPE AutoFix

Initial FP32 sharded candidate fails key-cache PCC.958–.965 while Vcache stays
>.999996, replicas/replays exact. Normal process close; bounded reset/list/smoke
all exit0 (sharded_rope_{reset,list,smoke}.log), then exclusivehardware handed to
AutoFix investigator. Source/component controls identify unsafe destination
width and mixedoperand-format behavior; no C++ edit. Freshindependent source
CLI report/provenance is AUTODEBUG_sharded_rope_fresh.md/rope_fresh_provenance.json.
HomogeneousBF16 width256 nativecalls with pairedhalfchunks atD512 pass component
PCC>=.9999942 forheads1/2/4 and exactreplicas. Realpairedmodel tests pending.

Ordinary runner now supports --duplicate-replays (default1 unchanged) for
final1024comparisons, and writes cache-PCC failure metadata before raising.
Forward/capture/math/timing window unchanged; Black/pycompile pass.

Adapted RoPE realpaired4096/128: sliding671.712µs vs731.481µs host decode,
minimumPCC.99786138/cache.99999666; full740.513µs vs724.293µs,
minimumPCC.99970713/cache.99997177. Both exactreplica/replay checks pass.
Keep sliding candidate; reject full adaptation forlatency. Hardware returned
normally after both successes. Optional path remains off bydefault until
combined policy selection. B32 sliding candidate withheterogeneous32..63
lengths/prefix31/trace/pageownership passes, minprefill.99969358,
mindecode.99763156: batch32_sliding_rope_candidate.json.

ProjectionK sweep: all12paired4096/128cases pass; projection_k_results.json.
SlidingWO K32 reduces hostdecode to658.022µs with unchangedPCC; K16 is673.592µs.
SlidingQKV44/88 and router44/88 do not materiallybeat671.712µs control.
FullQKV44/88,WO16/32/64,router88 do notbeat724.293µs baseline.
GroupedMoE BF8 payload trials nowrunning against compatible current candidates.

Packed-vs-split gate/up controls allpass; splitexperts are slower
(702.06µs sliding vs658.02,762.60µs full vs724.29), and splitshared also loses
(662.63/729.90µs). Tests preserve each role's dtype/LoFi policy and indexedtop8
execution. Keep packed projections. Matched groupedMoE BF8 trials pass:
sliding655.427µs/minPCC.9978812 versus658.022µs BF16payload;
full727.589µs/minPCC.9994411 versus724.293µs. Keep BF8 as sliding-only candidate.

DirectHF-to-BFP4 sliding indexed gate/up (no BFP8 intermediate requantization)
passes4096/128 withminimumPCC.9977556/cachechecks/exactreplicas/replays,
hostdecode653.513µs. BFP4/LoFi sparsegeometryrecheck also passes:
N2K44 675.64µs,N2K88 677.97µs,downN2K6 655.62µs. Keep original sparsegeometry.
Sharedgate/sharedDownBFP4 and expertinputBFP8 are being measured separately
before freezing finalpolicy.

## Final default packing and acceptance

Integrated device-BFP4 packing (1683b8a4) failed sliding output PCC despite exact replay/cache checks. Raw-checkpoint-only control restores all129 candidate PCC values; CPU AutoDebug confirms equivalent padding/order and different conversion paths. Runtime a12a913c now constructs the sliding gate directly from checkpoint. Default sliding/full4096/128 stress passes minimum PCC0.9976112404200184/0.9994477563467595 and1024 duplicate replays per TP, plus cache/all-rank checks. Mixed33-token layer0→5 joint trace also passes. See AUTOFIX_final_policy_packing.md and *_final_policy* artifacts. Final matched-precision geometry/DRAM/sharded-layout controls are running before broader acceptance.

## Final maximum-context and layout gates

Current a12a913c passes batch32 heterogeneous lengths32..63 and31+1025-token continuations for both kinds, including page/slot preservation and refreshed ownership. All four final capacity commands pass: sliding262143+1 minimumPCC0.9995173073041244 and262144+0 minimum0.9999230200763243; full results are recorded in full_final_max_262143.json/full_final_max_262144.json. Anonymous other-resident reservations are19,797,114,880bytes/sliding and19,260,243,968bytes/full per device, using the revised27,451,657,216byte final peak budget. Both maximum-context shapes preserve the advertised262144-token contract. Watcher checks now running separately before native profiling.

Both final4096/128 Watcher runs pass output/cache checks and normal process shutdown (exit0) with TT_METAL_WATCHER=10 and TT_METAL_WATCHER_NOINLINE=1; all four devices detach cleanly. See final_watcher_summary.json and preserved watcher logs. All final runtime acceptance gates pass. Final100k native profiles are next, separate from Watcher.

## Final native target profiles

Both100k captures exit0 and preserve complete native metadata for4096/128 on all4ASICs. Whole-layer device windows (including all operations/gaps) are325830.533us prefill/705.420us decode sliding and309683.707us/766.584us full. Rooflines use useful active work and four-device theoretical peaks; see profile_final_{sliding,full}/whole_layer.json, human tables/CSVs/provenance, capture_integrity.json and final_perf_findings.md. Native rows confirm selected BFP4 expert/shared gate, active8 execution and LoFi decode projection policy. Local telemetry packet bff5abf7-6db4-4696-8fc0-a904af9d0396.json updated from these measurements. Independent stage review is running.

Independent review returned clean-pass with a generic table-filter warning. Root verified the same-name end filter left the prefill table open into decode, corrected it to explicit PERF_PREFILL_END/PERF_DECODE_END, and regenerated supporting tables/CSV/provenance offline. Every corrected merged table count matches the native phase count; accepted whole-layer JSON/windows/integrity remain byte-identical and telemetry is unchanged. Historical tables are archived in prior_signpost_filter/. Stale multichip context wording also corrected. A reviewer follow-up will check the report-only correction.

## Stage completion and local checkpoint

Independent fresh xhigh review session01a0e0da-7b03-7b90-af8f-146d6d500594 exits0 and returnsclean-pass with no required work. It rechecked the explicit END-signpost correction and final row counts before closure; stage_review_final.md is authoritative and stage_review.md mirrors it. The initial failed review is retained as stage_review_initial.md.

Repo /workspace/tt-metal, branch gemma-4-26b-a4b-it, stage checkpoint **ab2892308b8960b8652546bbaf070ad8b120996d**. The reviewer created this already-user-authorized local commit after passing, despite its narrower report-only assignment; root audited scope and final sources. All859 touched paths belong to this model stage. Runtime remains a12a913c; one diagnostic-probe Black parenthesis edit is AST-identical and recorded in source_provenance.json. Commit hooks pass. No push, full-model or vLLM work. Earlier checkpoints41a8c80072c3ce2f0480cd4bbfbefeccfedc8ddc/b42e1f3f3d812cffa75fbebcf317e951a1e152ba and externally authorized fabric repair9a529836fc91b1117a48f6d63ef30455a69cd42d remain ancestors. This final log/status/provenance update is a separate documentation checkpoint.

The documentation-checkpoint SHA is recorded in local telemetry packet bff5abf7-6db4-4696-8fc0-a904af9d0396.json notes after commit, avoiding a self-referential commit hash.
