# AutoFix status: partial repair verified; stage incomplete

## Failing contract and diagnosis

`AUTODEBUG.md` establishes that `post_combine_reduce` requires32 logical
rows. The original grouped MoE passed n1..31 directly. Padding only
contributions leaves a weights shape rejection; leaving IDs unpadded causes
out-of-bounds index reads. Routing/counts must continue to use exactly n.

The minimal repair pads initialized contribution, score and UINT16 ID rows
only at this native reduction boundary, then slices the BF16 output to n.
No dummy token reaches routing, recurrence, hashing, or paged-cache updates.
The host contract and native paths in the report were checked against this
checkout. Native L1-estimator undercount is a separate source concern;
no allocation or kernel failure here was attributed to it.

## Validation

`mesh4_l0_n33.json` and `mesh_l3_n33.json` pass real-weight whole-output PCC,
local state parity, nonaligned32+1 prefill, three decode inputs, runtime
fallback guards and16 allocation-tracked exact trace replays. These are
actual device results for the recorded source hashes. They do not establish
all32 tails, B32, PLE, long-context, watcher, or final-stage acceptance.

A later PLE run (`mesh_l1_n33.log`) failed the host striped-table width
validator before lookup. The local code and scale tables each required16
zero columns. The setup helper now pads each rank to32 columns; a bounded
memmap repack removes full-table temporary copies. `repack_cpu.json` verifies
six synthetic cases and202752 real checkpoint words byte-for-byte, including
boundary rows and zero padding. `qsa_ple_followup_audit.md` found no additional
blocking defect in this repair, the corrected PLE value projection, shared
immutable tables, or direct local-head cache allocation.

## Resumed device repair

The owner explicitly authorized idle-device resets. Resets4..8 completed cleanly;
no hung process was terminated. The full PLE contract exposed huge finite values
for n>1. Component diagnostics proved routed inputs and raw expert down outputs
were normal, and post-combine reduction agreed with CPU weighted reduction.
`mesh_l1_combine.json` showed exactly one corrupt combined slot: rank0, expert21,
first region offset0, destination token1/slot1. The other19 selected slots matched
raw expert output exactly. This isolates native combine's TILE input path; a
precise C++ initialization defect has not been established.

Explicit device-side TILE-to-ROW_MAJOR conversion before native combine repaired
all20 selected slots and every subsequent decode in
`mesh_l1_rowmajor_combine.json`. `mesh_l1_contract_rm.json` then passed logical33
prefill, three decodes, every tail1..32, diverse B32, exact allocation-tracked
trace replay, state parity, runtime fallback guards and PLE->GDN stacking.
Minimum B32 output PCC0.998862 exceeds0.995. No routing or expert arithmetic was
changed. A static logical-count bound for conversion is being measured next.

The stage remains incomplete until other layer contracts, long context,
profiling, watcher and independent stage review pass on the selected source.

## Native all-reduce candidate: allocation repaired, numerical gate failed

`mesh_l0_native_ar.log` records the synchronous `ttnn.all_reduce` candidate
exhausting the32768-byte L1_SMALL reservation during the logical-tail sweep.
Initial output checks and trace replay had completed before this allocation
failure. The stack identifies `AllGatherMulticastFactory` semaphore creation.
The factory retains two workload-owned semaphores per cached program variant;
logical-shape cache misses increase retained allocations. This is not evidence
that every cached sum allocates fresh semaphores, nor a reason to enlarge the
reservation and hide the growth.

The test-only candidate now calls `ttnn.experimental.all_reduce_async` with
one persistent set of seven semaphores: two barriers, three reduce-scatter,
and two all-gather. All are allocated before warmup/capture and retained
through trace release, shared only by serial CQ0 operations. Width2560 gives
an aligned640-column scatter on the four-chip mesh for every logical tail.
The async path is a separately tested collective implementation, not merely
an allocation-only change to the synchronous multicast kernel.

`mesh_l0_persistent_ar.json`, its `.log`, and
`mesh_l0_persistent_ar_provenance.json` show the allocation repair completing
the mixed-tail sweep and16 exact output/state trace replays for both B1 and
B32. However, B32 requests at zero-based indices1 and13 have whole-output
PCC0.9907076955 and0.9909727573 against the single-chip baseline, below0.995.
Deterministic replay of those outputs does not repair their numerical error.
The report correctly records `passed: false`.

Warmed candidate decode was2.547587633ms at B1 and56.850929260ms for the
serial B32 path. The owner's comparison found only about2% latency gain,
which did not justify adopting the less accurate candidate. These were per-request diagnostics; the historical candidate did not record a whole-B32 aggregate, so this is not evidence that it failed the user’s whole-output gate. The ordered gather then
FP32 reduction remains selected. These failures reject an optimization
candidate; they do not change the selected decoder's correctness evidence.

## Profiler capture repair: segmented analysis completed

The first mesh capture overflowed profiler DRAM buffers and lost every
device-op record in its measured prefill/decode windows. Reading after each
whole forward repaired the inter-forward accumulation issue, but the
single-chip baseline still overflows within one33-token prefill. The exact
counts and source audit are in `profiler_audit.md`:6942 host IDs versus6676
matching raw/C++ IDs, with266 missing; measured prefill lost53 of1212 IDs,
while all215 measured decode IDs were present.

`probe_multichip_decoder.py` now reuses the test-only `bounded_profiler`
context from `profile_fused_ops.py`. It drains at guarded host operation
boundaries based on operation-counter progress, in profiler mode only.
Reports identify this as segmented per-operation timing and retain drain
operation IDs. Separate warmed trace runs remain the latency evidence;
profiling drains introduce host gaps and cannot be included in an ordinary
whole-forward latency comparison.

Final accepted captures are `profile_baseline_l{0,1,3}_segmented` and
`profile_final_mesh_l{0,1,3}`. All six exited zero with ordinary nesting
validation and strict host/device joins enabled. The final mesh captures
contain the selected B1 reduction. Their tables and provenance are under
`tracy/baseline_l{0,1,3}_segmented` and `tracy/final_mesh_l{0,1,3}`.
No overflow warnings appear in these logs. The failed intermediate QSA
capture is retained and excluded; the ordinary retry passed without
mid-run-dump validation bypass. See `profiler_audit.md` and
`performance_analysis.md`. Independent stage review remains pending.

## Acceptance closure

All selected-path runtime gates passed. Independent round2 returned `clean-pass` after separate tracking/diagnostics-disabled timing runs closed round1. See `stage_review_round2.md`, `performance_summary.json` and `work_log.md`; earlier pending notes above are historical.
