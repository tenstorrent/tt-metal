# AutoFix — initial TP4 source repairs

Status: initial repairs and cold-cache semaphore fix are device-verified.
The early pending-authorization and proposed-check sections below are a
chronological record; later sections and final evidence supersede them.

## Failing contract and diagnosis

`AUTODEBUG.md` independently traces the default hidden-sharded norm to
`LayerNormPreAllGatherDeviceOperation::compute_output_specs`, which writes
shape[3]. The model's residual is rank3. This would reject the first forward
for both layer kinds. The earlier setup failures are separately explained by
the None physical-device list and wrong exchange-core helper argument; both
were corrected. No PCC failure or device hang was observed.

## Hypotheses and interventions

1. Correct the earliest rank mismatch locally: expand residual and stats to
   rank4 only around distributed pre/post norm, returning rank3 before the
   collective helper. Applied. Host fake primitive enforces the real rank4
   contract for seven batch/sequence shapes. Changing global CCL rank handling
   or abandoning hidden sharding is unnecessary.
2. Preserve optimized decode arithmetic: the initial native chunk recurrence
   was a different floating-point operation graph. Copied only its local12
   reader/wrapper/partition into multichip_decoder.py, importing the baseline's
   unchanged compute/writer/config. Changed all head-dependent dimensions,
   gate tile batch stride, Q/K/V tiled strides, RM page bytes and packed byte
   offsets together. Kept the FP32 live readout/BF16 persistence distinction,
   separate affine rounding and FP32 SILU gate. Native multi-token prefill is
   retained. This prevents an unmeasured numerical-path substitution; no PCC
   or latency gain is claimed yet.
3. Require mesh shape1x4, validate residual-layout choice, and fix the runner's
   baseline SHA provenance to hash the actual optimized implementation.

No optimized baseline or shared kernel source was modified. Only the new
multichip decoder, its tests and stage docs changed.

## Validation

- `python models/autoports/qwen_qwen3_6_27b/tests/validate_multichip_host.py`
  passes630 partition cases, local GQA ownership/gate-address bounds at
  B1/2/3/16/32, and seven rank adapters. Artifact: host_validation.json.
- `python models/autoports/qwen_qwen3_6_27b/tests/audit_multichip_runtime.py`
  passes29 reachable functions with zero direct Python host-conversion calls.
  Artifact: runtime_audit.json. Imported compute/writer/config inspected in
  AUTODEBUG.md; native operations and CCL still require runtime evidence.
- Python compileall passes. Pre-commit hooks recorded in hooks.log.
- No device run follows the setup failure: multichip requires reset after a
  failed multichip run; device-usage requires operator authorization for reset.
  An asynchronous authorization question remains pending. No reset performed.

## Remaining uncertainty / next discriminating checks

After authorized reset/list/mesh-smoke, first test local rank adaptation with
real weights B1/S33, then both kinds and B2 to exercise the gate batch stride.
Compare sharded and replicated residuals, recurrent state/readout, awkward tails,
current-position and page-table changes, traced allocation safety and watcher.
Measure performance before accepting any topology/geometry choice. The initial
baseline_l0.json predates corrected provenance; regenerate that reference
sidecar with final tests. Full context, stress, profiling and review remain open.

## Device verification after owner authorization

Owner authorized reset on2026-11-13. Reset/list/mesh smoke all exit0.
`tp4_l0_v3.json` verifies rank adapter and local12 recurrence with real weights,
B1/S33 prefill+decode; all output/contribution/state PCC exceed0.99997.
The earlier pending authorization text is historical; device work has resumed.

## B32 cold-run non-finite prefill investigation

correctness_v1_sharded_l0_b32_s33 fails initial prefill with NaN, while
cache/decode pass. diagnose_b32 (intermediate reads) and uninstrumented
b32_repro pass after reset. watcher_v1_sharded_l0_b32_s33 reproduces NaN
during its first JIT compilation; watcher has no reported memory violation.
AutoDebug source inspection found shared CCLManager semaphore cores8x8 versus
11x10 actual default subdevice. all_gather default_workers chooses4workers
perdirection above256KiB; choose_worker_cores traverses all default cores,
including coordinates outside8x8. GlobalSemaphore resets only allocated cores.
A model-local CCLManager subclass now allocates/initializes global semaphores
on the full11x10 grid. This repairs a concrete ownership mismatch; causal
verification by watcher/repeated cold runs is pending, not yet claimed.

Full-grid semaphore repair passes watcher_b32_full_semaphores and
watcher_cold_b32_v2. The latter uses a fresh TT_METAL_CACHE path, reports
0/476 JIT hits, passes initial finite outputs, PCC, cache checks, replay
bit equality, changed-input control and allocation-tracker positive control.
Copied watcher_cold_b32_v2.watcher.log preserves watcher evidence.
Both full-attention B2/B32 also pass correctness_v3 with the repaired grid.

## AGminimal repeated-launch admission

The persistent-scratch/dedicated-semaphore hypothesis alone was insufficient:
synchronized replay passes, unfenced50-replay sequence stalls after a successful
first replay. Live evidence is in triage_agminimal_async; bounded timeout exits124.
Reset/list/mesh smoke pass (reset_agminimal_async.log). A tiny ordinary Linear
AG with persistent output supplies the entry barrier missing from the fused
native Linear path. Its writer waits all required peers before resetting its
barrier counter (minimal_default_writer.cpp:258–310). Initial Python call used
the wrong overload keyword; repaired to persistent_output_tensor and reset
(reset_agminimal_fence_api.log). With the device barrier included inside trace,
`topology_agminimal_device_fence_v2` passes eager, first replay and50 queued
replays: PCC0.999999999972,0.404619ms including barrier, versus0.476434ms
separate8x4 control. This is a verified model-local workaround for this probe,
not a shared-kernel root-cause claim. Must compare against current8x8 best
separate matmul and integrated layer before selection.


## Final topology resolution

`topology_agminimal_best` compared the repaired fused candidate with the then
fastest separate8x8 path:0.431357ms versus0.339615ms. The final packed-SwiGLU
contract is measured again in `final_packed_agminimal`:0.753429ms fused versus
0.496683ms separate, PCC0.99999267 through50 queued replays. Therefore final
production uses ordinary AG and native SwiGLU, while retaining fused MMRS at
the row-parallel prefill boundaries where it wins. Admission-stall investigation
is resolved for the test candidate and is not hidden in production.

## Native MMRS watcher ownership defect (repair verified)

Fresh source-only report: `debug_narrow_watcher/AUTODEBUG.md`. Dedicated
NCRISC/NoC0 reader explicitly issues W reads on NoC1 while dedicated
BRISC/NoC1 writer compares the core-wide response counter against its own
private count. The writer issues no reads; tagged reader barriers cover the
steps consumed by compute. This establishes an accounting contract defect,
not measured proof of every hardware counter at failure. An ordinary writer
read barrier can hang on the same mismatched counts. No watcher suppression,
blind counter reset, or shared C++ modification is used.

Discriminating device checks: original B32/S33 fails twice; existing separate
matmul/RS passes; 128-row native pieces pass and record NoC0 thin-wide plans
for both row projections. Candidates preserve output width1280 and logical
row order. Broader geometry/performance comparison is pending to select the
smallest acceptable production adaptation. Native plan depends on live L1;
chunk size alone is not a universal guarantee. Any retained native route
requires resource-pressure coverage. The64-owner test's old prefill was B1/S65
for both decode batches; the renewed test will record and exercise actual
B32/S33 prefill as well.

Selected repair in production SHA256
`63d2e53af8490dd63ec081ea202611e21dfb3ddca3e033e851e89837d0debb88`:
separate row matmul/RS through512 flattened rows; native128-row pieces above
512, concatenated/cropped to logical rows. Both kinds pass all six
`final_safe_watcher_sharded_l{0,3}_` cases with suffixes
`b32_s33`, `b3_s513`, and `b31_s33`. The separate
`final_safe_resources_b32` test passes actual B32/S33 prefill for all64 owners
with watcher, then exact changed-input decode replay. B1 resource checks also
pass. This supplies the required pressure contrast in addition to native-plan
NoC0 evidence from `mmrs_chunk128_watcher_v2.json`. No shared C++ edits or
watcher suppression. Native NoC1 ownership remains an upstream defect avoided
by this target-specific implementation; arbitrary meshes/configurations and
concurrent mesh queues are not claimed. Remaining final numerical/performance
stage gates are tracked in `work_log.md` and `gate_summary.json`.

## B31/S31 per-user accuracy counterexample (controlled)

Fresh `debug_pcc31/AUTODEBUG.md` independently verifies an attention-fidelity-
dependent failure on seeded synthetic inputs. User25's prefill rows lie in a
full128-row MMRS piece, before the96-row tail. More decisively, the row
projections do not feed this layer's recurrent state or independently uploaded
decode input, and disabling MMRS preserves every reported decode/state PCC.
The HiFi4 intervention changes packed projections, native prefill convolution
and output projection; it does not isolate a single faulty primitive. No
rank/layout bug was established, and low-level arithmetic is not declared
proved correct from this control alone.

The production LoFi policy remains selected from recorded target-model input
evidence and the existing real stack controls, following the optimization
precision rule. B31/S31 recorded inputs pass all original0.995 per-user,
contribution and state/cache gates, watcher and exact trace checks on both
kinds. HiFi4 is an available explicit policy control if a caller requires the
synthetic fixture to pass. No input-dependent fidelity fallback or threshold
reduction was added. Full-model quality remains a future-stage requirement.

Only harness/docs changed: initial failed PCC reports are now saved before
raising; watcher31x31 explicitly selects recorded fixtures; all synthetic
failures/HiFi4/separate controls remain archived and indexed. Report runner
hashes and artifact rename hashes preserve provenance. Production source
stays082c82fa. `pcc31_control_summary.json` is included in the final gate
aggregate so this accuracy limitation cannot disappear behind aggregate PCC.

## Profiler marker collection recovery

Contract: complete normal Tracy/device-profiler collection with intact marker
pairing and all four ranks. `debug_profiler_marker/failed.log` aborts during
final pairing after successful numerical/trace checks. AutoDebug independently
found the legacy Blackhole sampler reading the live high clock word before
low; the failing END timestamp has low word10. A rollover can backdate it by
2^32 cycles. Parent verified the source ordering and normal parser assertion;
the actual rollover is inferred because raw neighboring markers are unavailable.

Rejected hypotheses/workarounds: raising the existing5000-program DRAM buffer
count or adding drains does not repair clock coherence; MID_RUN_DUMP bypasses
the pairing check, so it is not accepted. No assertions are suppressed and no
shared profiler/model implementation changes are made.

The collection-only experiment reruns the same source/settings in a fresh
process under new `delivery_profile_retry_*` paths. The formerly failing
B32/S33 GDN collection now completes with normal final pairing, no dropped-
marker diagnostic, intact device CSV and four-rank ops data. All new profiles
must pass the aggregate gate's raw CSV hash, rank, dtype/fidelity and diagnostic
checks. This controls the transient collection failure; it does not fix the
shared clock sampler or guarantee future collection never encounters it.

Changed files: profile artifact selection in the host aggregator/renderer;
reproduction script explicitly unsets MID_RUN_DUMP. Production SHA remains
082c82fa. The owner-authorized idle reset and mesh smoke are recorded in
`debug_profiler_marker/reset.log`; no foreign process was terminated.
