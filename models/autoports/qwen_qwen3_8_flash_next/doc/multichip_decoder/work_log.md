# Multichip decoder work log

Stage4 started 2026-11-13, base `fb462731`, branch `main`. Scope is model-local
multichip decoder, tests and docs. Prior-stage untracked files are preserved.
No full-model or vLLM implementation, no push. Enabled tt-autodebug0.1.8 and
tt-model-bringup0.1.24 confirmed by `codex plugin list`; installed environment
exports validated. Existing multigoal session environment retained.

## Hardware admission and planning

- `fuser /dev/tenstorrent/*`: no holders before admission.
- `timeout 60 tt-smi -ls --local`: four p300c chips; `hardware_initial.log`.
- `timeout 60 tt-smi -s --snapshot_no_tty`: `hardware_snapshot.json`.
- `timeout 60 build/tools/umd/topology -f models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/cluster_initial.yaml`: connected ring with two links/edge.
- Bounded120s TTNN smoke with `TT_VISIBLE_DEVICES=0,1,2,3`, p300_x2 descriptor,
  `FABRIC_1D_RING`, MeshShape(1,4), BF16 width-sharded input and native all_gather:
  `mesh_smoke_initial.log`, exit0, `MESH_SMOKE_OK`. Mesh closed. Grid11x10, DRAM8x1.
- No resets so far. tt-smi ETH_LIVE_STATUS=0 contradicted a naive link-down
  inference; UMD topology and a successful actual collective resolve it.
- `mesh_plan.md` records design before final path coding. Full-stack memory is
  an active gate; baseline BFP8 cannot fit with PLE even at zero context.

## Status

Implementation and acceptance gates are pending. No stage completion claim.

## First component integration

`baseline1_provenance.json`: optimized single-chip real layer0 logical1 plus3 eager decode steps completed. `mesh1_provenance.json`: construction, GR and local GDN reached output projection, then host TypeError because the baseline linear helper recognizes precise-projection subclasses rather than arbitrary callables. Made the gathered exact projection implement that protocol. No hang; mesh teardown completed. Reset1 follows the multichip failed-run rule before retry.

Reset1 listed all4 chips afterward and mesh open/close passed (`mesh_after_reset1.log`). Initial reset shell did not retain its standalone exit status, so no reset-success inference beyond healthy list/smoke. `mesh2` reached grouped MoE after GR/GDN. Fused SwiGLU validation rejects FP32 destination because it needs8 DEST tiles; changed only its compute config to the required BF16 accumulator, matching the reference module. This precision difference requires real-weight validation. Reset2 is recorded with an explicit provenance exit status.

## Source audits and non-aligned reduction repair

`mesh3` closed gracefully after the native post-combine reduction rejected one
logical token: its hardware tilizer requires32 logical rows. No device hang.
`source_audit.md` and `qsa_ple_source_audit.md` found the same reduction contract,
the QSA selected-width typo1080 versus2080, and public list-batch/length checks.
The implementation now pads only reduction inputs with initialized zeros and
slices back to logical n. Routing/counts/state never consume dummy rows.
Shared `MeshWorkspace` removes one83MB expert-output allocation per layer;
long prefill uses the existing bounded chunk writer. Batched decode routes
all request MLP rows in a single grouped call, with independent attention states.
These changes still require target-mesh validation.

AutoDebug launcher attempts: sandbox preflight lacked user namespaces; existing
unrestricted execution authorization allowed the documented explicit skip.
The old default CLI then lacked a matching code-mode host. A temporary symlink
was removed after detecting an IPC version mismatch. The final launch selects
the matching0.163.0 CLI directory via PATH and uses the installed launcher.
`autodebug_launch_current.log.gz` records the actual source investigation. The
failed child had no hardware access and was terminated before relaunch.

Fresh real-checkpoint optimized references at logical33, reversed128-capacity
page table, three decode inputs and16 allocation-tracked trace replays pass
for layers0,1,3. Initial reports are `baseline_l*_n33.json`; enhanced reports
`baseline_l*_n33_state.json` also persist baseline state tensors for mesh parity.
Reference tensors are outside git under `/home/ttuser/qwen38-flash/multichip_l*_n33.pt`.
All use runtime Torch/conversion fallback guards. Exact commands/source hashes
are in adjacent provenance JSONs. Reset3 is the final reset under the stage's
three-reset limit; it follows the failed-multichip-run rule, not a detected hang.

Reset3 exited0 in44.67s; all four devices listed and real native all-gather
returned exact values (`hardware_after_reset3.log`, `mesh_after_reset3.log`).
`mesh4_l0_n33` passed: prefill whole-output PCC0.9997434; three decode PCCs
0.9999071/0.9998727/0.9998010. Local convolution state is exact and recurrent
state PCC>=0.99999988 versus properly sharded single-chip state. Sixteen
allocation-tracked replays refresh three inputs and restore prefix state;
every output and mutable state exactly matches the corresponding eager call.
Warmed prefill22.01ms, decode2.609ms. These are one representative kind at
physical128 allocation, not final all-kind/context/performance acceptance.

Capacity follow-up caught PLE's value geometry: key projects10240 stream-specific
features, but value projects one shared2560-vector. Replicate the full value
projection and shard only key/norm/convolution. The implementation now does so.
`capacity_audit.md` includes immutable-sharing recommendations, construction peaks,
long input/output buffers and explicitly provisional runtime reserves.

## PLE setup repair and hardware-budget hold

`mesh_l1_n33_provenance.json` records a clean-teardown host ValueError from
StripedGather: local striped width must be divisible32. The initial per-rank
code/scale widths both ended16 columns short. `_repack_ple_table` now appends
16 zero columns independently to each rank, not to the global concatenation.
Load-time row ownership and all valid codes/scales remain unchanged.

The full CPU repack check passed (`repack_cpu.json`, `repack_cpu.log`): six
synthetic table/encoding cases,8448 real row IDs including both table ends,
168960 exact code words and33792 exact scale words, and zero padding checks.
No device was opened. Complete v2 memmaps are cached outside git in the
existing packed-table cache directory. `qsa_ple_followup_audit.md` independently
confirms setup geometry, replicated PLE value and local cache ownership.

The selected multichip skill mandates reset after a failed mesh run; the
selected device skill caps resets at3/stage. All3 have been used, despite all
failures being host validation with clean device teardown. An asynchronous
question requests exactly one additional reset exception. No further device
run or reset has occurred while this authorization is pending.

Meanwhile direct local-head QSA allocation removed the temporary duplicate
cache; MeshWorkspace shares RoPE and immutable index arrays. The proven
maximum320 active experts in a32-token top10 group bounds dispatch capacity
at10240 rows, reducing scratch from16192. The context contract now records
31203443520 bytes/rank for planned all-layer resident/static/B1 max-context
state, before runtime reserves/input-output. These latest changes are source-
reviewed but require renewed device evidence. Context remains262144.

The harness now contains all32 tail cases, grouped batch/list API trace checks,
cross-kind stack boundaries, full-stack DRAM reservation, and signposted
warmed eager profiling. They are prepared, not claimed passing.

Repository pre-commit hooks normalized whitespace in hardware-list logs, mesh1.log and the initial topology YAML. `log_normalization.json` preserves the original captured hash and identifies the normalized artifact hash for the affected recorded run. Passing correctness/trace reports were not rewritten.

## CPU probe admission while reset exception is pending

Moved reference loading and validation before weight loading and mesh opening.
References now record snapshot/layer/length/capacity metadata and must contain
all four output/state snapshots plus the optional requested batch/tail/stack
evidence. Historical reference tensors must be regenerated; their recorded
measurements remain historical evidence. Invalid capacities, insufficient
decode space, unsupported layers, incompatible component retention and missing
batch trace tracking are rejected before hardware access. Failed baseline
validation no longer overwrites a reference file.

CPU command: `python -m unittest models.autoports.qwen_qwen3_8_flash_next.tests.test_multichip_probe_admission -v`.
All six tests passed; no device was opened. The recorded repetition is in
`probe_admission_cpu_provenance.json` and `probe_admission_cpu.log`.
No additional reset or device run occurred. The unanswered reset exception
still prevents the required hardware acceptance work.

## Local checkpoint and blocked audit

Preserve the candidate, tests, context plan and captured evidence in a local
checkpoint commit. This is not stage acceptance: PLE mesh parity, final-source
refreshes, long-context validation, profiling, watcher and stage-review remain
outstanding. No full-model or vLLM work was started and nothing was pushed.

The same reset-budget blocker has persisted across three consecutive goal
turns. CPU repack, source audits and probe admission work made progress during
that hold; those checks are now complete. `fuser` reports no holder for devices
0..3. There is no live device job to wait for and no evidence of a wedged chip;
the hold follows the selected skill's mandatory post-failure reset rule and
its three-reset cap. The pending exception has no user answer. Further device
validation requires that exception or an operator-performed reset. AutoFix has
not been declared failed, and hardware-dependent repairs are not claimed
verified. Resume the existing stage and evidence after the hold is resolved.

Candidate checkpoint: `c6ef9daaf663f62f1115d3d337c757a127f022cb`.
All repository commit hooks passed. This checkpoint contains only stage-owned
source, tests and docs; unrelated pre-existing fused-decoder artifacts remain
outside the commit. This entry is a subsequent documentation commit, not an
acceptance commit.

## Owner-authorized hardware resume

The owner explicitly authorized `tt-smi -r` when devices are idle or after
terminating this run's own hung process, resolving the reset-budget hold.
`fuser` reported no holders on devices0..3. Reset4 is recorded in
`reset4_provenance.json`/`reset4.log`; no process was terminated. Subsequent
resets, if necessary, follow this explicit authorization and remain serialized,
bounded and logged. The existing stage resumes; no prior acceptance claim changes.

## Resumed AutoFix: multirow MoE corruption

Reset4 completed exit0; all devices listed and native collective smoke passed.
Fresh `baseline_l1_contract` passed tails1..32, B32 tracked trace and PLE->GDN
stack. `mesh_l1_contract` completed but failed prefill33, tails2..32 and B32;
single-token decode and mutable-state comparisons passed. Reset5 followed
clean teardown. `chunk_writer` passed exact33/65/33 rank-distinct writes, also
proving mesh open/close after reset5.

`mesh_l1_components` isolated normal PLE/attention/MoE inputs but huge finite
MoE outputs. The native reduction agrees with CPU weighted reduction of the
same selected combined values; the corruption already exists before reduction.
Reset6 completed, list and mesh smoke passed. `mesh_l1_experts` then showed
all selected routed inputs and raw expert down outputs have normal magnitudes
(input<=7.1, output<=0.125 in this diagnostic), including the failed n2 prefill.
The next check compares combined slots directly with raw expert rows using
dispatch metadata. Reset7 follows the cleanly terminated diagnostic failure.
No arithmetic change has been made based on an unproven hypothesis. The owner
authorization covers these idle-device resets. No process was terminated.

## Native combine repair verified

`mesh_l1_combine` isolated corruption to the first active expert's TILE-combine
slot; raw down output and19 other slots were exact. Reset8 completed after clean
teardown, and mesh open/close passed. Explicit ROW_MAJOR input to combine passed
`mesh_l1_rowmajor_combine` and the complete `mesh_l1_contract_rm` contract.
The latter reports prefill27.253ms, traced decode3.982ms and B32 decode95.451ms;
these are intermediate candidate timings. All tails1..32, state comparisons,
tracked trace replays and stack1->0 passed. The native internal defect remains
localized, not claimed fully diagnosed. The next candidate slices down/metadata
to32*min(512,10*n) before conversion, reducing B1 conversion extent32-fold.

## Final short-contract refresh and communication candidate

Bounded ROW_MAJOR conversion passed `mesh_l1_contract_bounded`: prefill26.985ms,
decode3.744ms (6.0% below full-workspace conversion), B32 decode95.503ms.
BF16 transport followed by local FP32 widening passed the same complete gate in
`mesh_l1_contract_bf16ccl`, with unchanged PCC and decode3.752ms; this is within
measurement noise, while reducing those collective payloads by2x. FP32 inputs
remain FP32. The rank-ordered sum and values are unchanged.

`baseline_l0_contract`, `mesh_l0_contract`, `baseline_l3_contract`, and
`mesh_l3_contract` all passed final-source tails1..32, B32, stacked interfaces,
state comparisons and16 tracked trace replays. QSA B32 minimumPCC0.997164.
The PLE report before final black formatting is semantically identical.

Long-context baseline layer1 completed32769 logical rows, then the new harness
failed a host-only UInt32/Long equality comparison in its history assertion.
`baseline_l1_long32769` preserves that failed attempt. Normalizing only the
read-back history dtype fixes this test bug; `_v2` reruns are in progress.
No decoder change, reset or process termination followed this host failure.

## Long context and residual topology evidence

`baseline_l{1,3}_long32769_v2` and `mesh_l{1,3}_long32769_v2` all passed.
Mesh used one public prefill call over32769 logical rows and three final decodes.
Baseline streamed32-row public chunks, with identical saved inputs and global
PLE token order. Every mutable state is compared; QSA positions/reversed pages
and PLE token history are exact. Minimum output PCC: PLE0.999843, QSA0.999172.
Both mesh runs allocated at least31,203,443,520B/device before IO (86 and115
filler buffers), validating execution under the full-stack reservation. This
is not full-model instantiation. `memory_capacity_plan.json` accounts for bank
rounding and leaves14,702,912B inside that reservation beyond its enumerated
banked ledger plus future embedding/head assumptions. B1 context262144 stays
advertised; full-stack maximum-context IO must stream bounded chunks.

`residual_contract.json` passes all four policies at logical1,2,31,32. It uses
real WO/GR weights and a common native WO producer, then collective, residual
write, and complete next GR read. Hidden640-wide residual shards remain sharded
through addition and distributed normalization; this is not immediateRS->AG.
Hidden residual was0.699x/0.902x/0.967x/0.956x the FP32-gather control speed at
those lengths. NativeFP32AR was1.045x/0.990x/1.047x/1.041x. This is a topology
probe, not the shipping exactGDN projection or whole-layer performance claim.
A signposted whole-layer profile is running before the integratedAR candidate.

## Profiler capture recovery

`profile_mesh_l0.json` reports decoder correctness success, but its original
Tracy postprocessing failed on unescaped multiline generic-kernel source in the
all-zone CSV. Re-exporting TT_ operation zones from the same binary preserved
all original messages and the raw export; the ordinary host/device join then
correctly refused missing device operation IDs. The console explicitly reports
full profiler DRAM buffers and dropped markers. The raw device CSV and C++
report agree:8548 device/op IDs;10872 host IDs have no device data. All2964
prefill-window and1120 decode-window records are missing. No timing from this
capture is accepted. Failed exports and raw capture remain intact.

`process_multichip_profile.py` reuses the previously tested operation-only
export, then performs the standard per-op device join (no synthesized timings).
The probe now drains ReadDeviceProfiler after setup/each initial call/each warmup
and between CLOSED signpost windows. Source inspection confirms NORMAL drain
finishes all CQs, resets device control/cursors, and appends raw/C++ CSV rows.
The v2 capture is a new bounded run; drain time is outside measured windows.
No reset was needed after clean device closure and host-only export failures.

## Integrated collective candidate and valid profiling

`profile_mesh_l0_v2` completed with zero profiler-buffer overflow and a complete
ordinary host/device join. Advice-enabled tables and CSV are under
`tracy/mesh_l0_v2`. The summed worst-rank per-op device totals are21.575ms
prefill and2.379ms decode; these are operator attribution, not critical-path or
host-wall latency. Decode hotspots include BinaryNg410us, Reduce309us,
Generic243us, post-combine reduce232us, GR injection168us, CCL135us and
Typecast129us. Separate traced layer timing is2.591ms.

`mesh_l0_native_ar` exhausted32768B/core L1_SMALL on mixed logical lengths.
A source audit identified workload-owned multicast semaphore cache growth.
`mesh_l0_persistent_ar` used two persistent barriers, three RS and two AG
semaphores across all serial CQ0 sums. It completed every tail and16 exact
B1/B32 trace replays without storage exhaustion. However two diverse B32
requests had PCC0.990708/0.990973 against the optimized baseline, below the
probe's per-request0.995 gate. Decode2.548ms and B3256.851ms improve only about
2%; this candidate is not selected. Ordered gather/FP32 sum remains the
shipping contract. This failed optimization does not invalidate the passing
default-path evidence.

The first single-chip profile overflowed within individual33-token prefill
calls despite per-forward drains:53/1212 measured prefill op IDs were missing;
decode215/215 was complete. No timing from that incomplete join is accepted.
The existing bounded_profiler helper now drains every64 operation-counter
increments for profiler-only runs. Their drain IDs are recorded and captures
are labeled segmented per-op analysis. Drain gaps cannot be used as layer
latency; separate warmed trace runs supply latency. Segmented baseline0/1/3
and mesh1/3 captures are queued serially.

## Segmented profiles and reset9

All three `profile_baseline_l{0,1,3}_segmented` captures completed exit0 and
ordinary host/device joins; advice-enabled tables/CSV are in matching `tracy/`
folders. `profile_mesh_l1_segmented` also completed. QSA mesh capture aborted
inside final profiler marker-stack validation: an NCRISC end marker lacked its
start, with no buffer-full/drop warning. This is not accepted timing evidence.
The process had exited, `fuser` showed no device holders, and owner-authorized
`reset9` completed exit0. `reset9_health.log` records the bounded local list and
successful1x4 ring open/close. No process was terminated. AutoFix source audit
and a separate profiler retry will close this evidence gap.

The default MoE reduction was extracted unchanged into `_reduce` so a test-only
ownership-masked FP32-destination batched-matmul candidate can be compared at
logical token count. It masks nonowned values before multiplication, including
stale NaNs, and keeps gate-selected expert execution. Its accumulation differs
from native sequential BF16 L1 accumulation; no numerical equivalence is assumed.

## Kept single-token MoE reduction

`mesh_l0_matmul_reduce` first failed a host-only TTNN Shape slicing assumption;
converting the shape to tuple fixed the diagnostic. `mesh_l0_matmul_reduce_v2`
passed tails1..32, whole-output/state PCC, B32, stack0->3 and tracked trace replay.
Its decode2.3995ms improves2.5909ms by7.4%, while prefill21.5905ms and B3258.2760ms
are effectively flat. Native reduction remains selected for n>1 because its
larger-group path gains nothing from this change. `mesh_l1_matmul_b1` and
`mesh_l3_matmul_b1` passed the same full contract, decode3.5747ms/4.8888ms.

The default now uses ownership-masked BF16 matmul with HiFi4 FP32 accumulation
for logical n1 only. Nonowned values are selected to literal zero before
arithmetic, avoiding stale-NaN contamination. Routing and fused expert execution
still use gate-selected experts. Both ownership bounds live once in shared
MeshWorkspace:8192B payload,65536B eight-bank reservation, charged to the
existing allowance. Workspace payload is54079488B; unassigned reservation
allowance is14637376B. Advertised context and31203443520B total reservation are
unchanged. Source pre-commit passed after formatting, recorded in
`precommit_final_source.log`. Final default short, long-reservation and watcher
runs are queued serially; candidate numbers are not yet final acceptance data.

## Final default results

`final_mesh_l0`, `final_mesh_l1`, and `final_mesh_l3` all passed on the formatted
production default, including every logical tail1..32, diverse B32, same-layout
cross-kind stacks, runtime fallback guards and16 tracked B1/B32 replays.
`performance_summary.json` derives exact speedups/efficiencies from these and
`baseline_l{0,1,3}_contract`. B1 decode is2.398565/3.583480/4.881483ms;
B32 step latency58.176109/95.290123/131.574371ms. Whole-output PCC is the
acceptance metric; the probe also retains conservative per-request diagnostics.

`final_mesh_l1_long32769` and `final_mesh_l3_long32769` passed the changed default
with the unchanged full-stack reservation. Minimum whole-output PCC is
0.9998434904 and0.9991748793. Public mesh prefill was one32769-row call, then
three decodes; all state/cache contracts passed. Actual reserved bytes remain
31203453440/31203453952 per device. The new shared ownership constants fit the
existing reservation. `max_absolute` in probe JSON is observed magnitude, not
error; PCC/exactness fields carry comparisons. Watcher runs are separate from
both profiling and the latency figures.

## Performance hand-off candidates

The selected path is the fastest correct integrated configuration measured in
this stage; no claim of global optimality is made. Unmeasured work remains:
packed-only indexed B1 experts with a compatible grouped/prefill storage plan;
fused MM-RS or AG-MM under the measured residual contract; GR injection K
parallelism or compact dot products (ordinary M/N tiling has only one output
tile); precision-locked DRAM-reader geometry and lower-fidelity sweeps for
remaining native projections; QSA indexer head partitioning; and fabric packet
size tuning. Exact shapes/API constraints are in indexed_decode_audit.md and
performance_analysis.md. These are performance hand-off ideas, not rejected
families or changes to the logical-length/context contracts. The measured
hidden-residual candidate and native AR are separate, explicitly evaluated
alternatives. No optimization stop-rule claim is inferred from these notes.

## Watcher gate complete

All three `watcher_mesh_l{0,1,3}` full-contract runs passed with
TT_METAL_WATCHER=10, no watcher exclusions and no profiler environment.
Watcher logs attach/detach all four devices and contain48/33/23 completed
sampling dumps respectively, with no error-like lines. Full correctness and
tracked B1/B32 trace output/state checks also pass. Exact logs and source
provenance are indexed in `watcher_summary.json`. Watcher timings are not
used in performance comparisons.

### Final profiler acceptance

All three `profile_final_mesh_l{0,1,3}` captures passed with the shipping
implementation, full profiler nesting validation and strict host/device
joins. The failed earlier QSA capture remains excluded. Baseline segmented
captures also passed for all three layer kinds. Final tables are under
`tracy/final_mesh_l{0,1,3}`; operator analysis is in
`performance_analysis.md` and uninterrupted latency in
`performance_summary.json`. No device job remains active. Stage review and
local acceptance commits are pending.

### Final source and host-check self-check

`final_cpu_checks.log` / `final_cpu_checks_provenance.json`: six admission
tests and twelve subtests passed. `precommit_final_source_recheck.log`:
all applicable source hooks passed without changing the measured source.
All eleven final short, long, watcher and profile provenance records match
`tt/multichip_decoder.py` SHA256
`af4699739ad869466bbc825909daf868e8a2f01ed64e0af149602fe6128349c1`.

### Final operator analysis

`final_operator_summary.json` and `performance_analysis.md` now cover all
three final layer kinds. QSA ordered index projection dominates at about
60.195ms for the measured32-token prefill window and2.100ms for decode;
all-gather totals are0.425ms /0.129ms in those windows. These are operator
attributions, not summed critical paths. PLE's one-core striped code lookup
is133.887us in decode. The n1 reduction matmul is3.548–3.593us across88
reported cores, and no native post-combine reduction remains in decode.
Unmeasured indexer partitioning, lookup parallelization and fused/indexed
families are explicitly listed as future experiments, without claiming an
optimization stop rule or global optimality.

Final documentation precommit normalized trailing whitespace/final newlines in generated human tables. Provenance records this presentation-only transformation; CSV numeric content is unchanged. Large reports remain line-split with indexed parts.

Final source and documentation hooks passed together: `precommit_final_stage.log`. Independent review is the only remaining acceptance gate before local commits.

## Independent stage review, round1

`stage_review_round1.md`: more-work-needed, one P2 evidence repair. Existing
latency runs used allocation tracking and runtime audit. The unchanged
harness has `--clean-timing`; run paired baseline/mesh layers0,1,3 with
logical33/capacity128, B1/B32,32 replays, tracking0 and diagnostics/profiler/
watcher disabled. New `clean_timing_{baseline,mesh}_l{0,1,3}` outputs and
`multichip_l{0,1,3}_clean_timing.pt` references preserve full-contract evidence.
No implementation change or repeated long/watcher/profile gate is needed.
The inherited functional-decoder allocation positive control is now indexed
in `evidence_manifest.json`, following the reviewer's source audit.

## Carried to next stage

- Preserve shared MeshWorkspace/static tables, bounded max-context IO and
  documented future embedding/head/final-mixer placement. Measure the actual
  full-stack trace/cache envelope; the reservation test is not an instantiated
  full model, and B32/capacity128 is distinct from B1 maximum context.
- Performance candidates in `performance_analysis.md` remain unmeasured:
  QSA indexer partition, packed-only indexed experts, fused projection/CCL,
  GR injection K partition, PLE lookup parallelism and precision-locked geometry.
- The harness still enforces a stricter per-request B32 threshold in addition
  to storing whole-output PCC. Final whole outputs pass the user's0.995 gate.
  When next changing the harness, gate the aggregate and retain request PCC as
  diagnostics. Historical native AR lacks an aggregate and is not described
  as a proved failure of the user's gate.
- Future generator/serving work must validate coexisting traces, changing
  scheduler/page mappings and advancing token feedback. This stage validates
  each decoder B1/B32 signature separately with restored state.

### Round1 required work resolved: clean timing

All six `clean_timing_{baseline,mesh}_l{0,1,3}` runs passed with the same
implementation SHA `af4699739ad869466bbc825909daf868e8a2f01ed64e0af149602fe6128349c1`.
Each uses `--clean-timing --trace --batch 32 --repeats 32 --length 33
--capacity 128`, `TT_METAL_TRACE_ALLOC_TRACKING=0`, no watcher/profiler, and
no runtime audit wrapper. The32 B1 and32 B32 replay outputs and states are
exact in every run. Median timing discards two warmup samples; restoration,
transfers and output checks remain outside timers. Prefill has five warmed
public calls. Exact commands/environment are adjacent provenance. Original
tracked all-contract evidence remains the correctness/fallback acceptance set.
No production or test-source changes were needed.

## Stage acceptance

Fresh independent `stage_review_round2.md` returned **clean-pass**, closing
round1's clean-timing requirement. It verified six clean runs, twelve decode
medians, nine ratio/efficiency pairs,384 exact replay records, unchanged
source maps and the inherited tracker positive-control index. No required
work remains. Prior correctness/long/watcher/profiler evidence was preserved.
Source, documentation and new-evidence hooks passed. Stage scope remains only
multichip decoder/tests/docs; full-model and vLLM work did not begin.
Local acceptance checkpoint SHAs are recorded below; never push.

### Local acceptance checkpoint

- Repository: `/home/ttuser/qwen38-flash/tt-metal`; branch: `main`.
- Accepted implementation, tests, docs and evidence: `52ec22af5916ad7f824d06fd8d60d6c70167f1dc`
  (`Accept Qwen3.8 Flash Next four-chip decoder`). All commit hooks passed.
- Earlier stage-owned checkpoints: `c6ef9daaf663f62f1115d3d337c757a127f022cb`
  and `cdb5e8f161972ca9d2958e81d809c29f686b817a`.
- This SHA record is a following metadata-only local commit. The complete
  acceptance/metadata SHA pair is also logged at
  `/home/ttuser/qwen38-flash/multigoal_logs/04-04-multichip-decoder.commits.json`.
- No push was performed. Unrelated pre-existing fused-decoder artifacts remain
  outside these commits. Devices were released after the completed runs.
