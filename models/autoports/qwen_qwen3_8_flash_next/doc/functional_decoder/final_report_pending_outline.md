# Final report rewrite outline — pending evidence

Source-only reconciliation after bounded chunk I/O, 2026-11-12. This document is
not a stage-review verdict or acceptance report. Runtime, tests, README and
work_log are owned separately. Planned paths below must be reconciled with
completed commands, actual schemas, source hashes and independent review.

## Reporting contract

Retain PCC >= 0.995. Report literal HF whole-output and residual-subtracted
contribution PCC separately, including aggregates and minimum logical rows.
An explicitly classified fresh contribution discrepancy is not a strict literal
pass. Every literal whole-output gate and every literal resumed-prefill/decode
contribution gate remains strict. A controlled top-level `passed` never removes
`literal_hf_strict_passed=false` or replaces independent stage review.

Separate these context claims:

1. Complete public 10240-wide decoder prefill, including every output and MoE.
2. Reduced long prefix/state-path coverage, naming every omitted operation.
3. Complete public traced final decode after an actual TT-executed prefix.

HF advertises262144; the last valid decode input position is262143. Reduced
2560-wide attention or PLE/hash/GDN state coverage cannot populate column1.
The functional skill permits a reduced long layer-level harness, but not an
undisclosed full-MoE claim. Prefix export/reuse must bind runtime/source hashes,
position, page permutation, actual device origin and its all-output accuracy.

## Bounded chunk-I/O repair and capacity

Native input slicing formerly retained DRAM binaries for each input length and
chunk offset. The native control recorded338 programs and5,537,792 retained
bytes. The former offset0-twice capacity proxy omitted intermediate offsets.
Its bounds and allocator-order-dependent failures remain historical diagnostics.

Current runtime uses fixed-descriptor BF16/UINT32 readers with runtime offset
and count, a runtime-count writer, and executed setup of all three descriptors.
BF16 padding is zeroed; all valid token words are copied exactly. The public loop
releases each completed chunk after its last queued writer. The proxy now
matches that lifetime. Default state/hash preparation covers physical counts
1..32; custom subsets are explicit caller restrictions. Read/copy tests must
prove exactness, no retained cache/DRAM growth, first-unseen-plan capture and
live-trace interleaving. Setup-only compilation is insufficient to load binaries.

No public PLE capacity is accepted yet. Stronger all-count proxy v2 candidate2528
and the separately scoped public-chunk proxy candidate3712/failure3713 are
search observations, not complete-public bounds. Earlier5792/5793, prepared5664,
aligned7648 and tail7647 results used older source/preparation profiles. Do not
combine them into a continuous supported interval or relabel them as final.

Close capacity with all32 math/hash and fixed-I/O preparation, matching full
input/output/decode-input residency, fresh actual-public candidate and adjacent
failure, decode warm/capture/replay and repeated prefill. A finite terminal32
residue proxy sweep plus maximum-resident-I/O all-count workspace checks can
cover tail footprints after constant I/O descriptors are proved; complete real
HF accuracy remains mandatory at the accepted maximum and a long nonaligned
neighbor. Record exact allocation failures, free/largest blocks, resources and
lifecycle. A stronger sweep failure alone is not a public-prefill failure.

The full BF16 table is102400491520 bytes; packed table30720147456 bytes. The
advertised full10240-wide input alone is about5.37GB, exceeding residual memory
with that table and full decoder resident. This proves some resident-prefill
reduction is unavoidable, not its final largest feasible limit. Context JSON
must retain null/unaccepted capacity until the actual bound is established.

## Fresh-contribution controls requiring review

- Layer0 batch request16: the unchanged HF layer with only attention GR-down
  FP64 accumulation reproduces the known BF16 midpoint/backend route change.
  Preserve literal failures and independent literal rerun evidence; final batch
  control is distinct from strict continuation/decode.
- Layer1 batch request2 row0: actual packed-cache lookup values in unchanged HF
  reproduce the fresh contribution discrepancy. CPU experts remain checkpoint
  BF16; device BFP8 experts are separately recorded, not called identical.
- Layer1 public unseen fresh lengths65/97/129: strict v4/v5 failures remain
  preserved. Packed-only control explains129 row1 but not65 row2. The consistent
  actual-packed-lookup plus all six accurate GR down/up/GDN QKV/out CPU projection
  boundaries reproduces both failures. FP64 is a mathematical backend control,
  not bitwise emulation of the runtime FP32 addition tree. All other HF modules,
  router, experts and state semantics remain unchanged; no TT operands/states
  enter the CPU control.

`ple_public_backend_matched.json` records all six repeated TT outputs against
that single policy: minimum matched whole/contribution PCCs are
65=.99885267/.99872315,97=.99845189/.99797136,129=.99818861/.99787492.
Literal contribution failures remain65 row2=.99493819 and129 row1=.99475437;
control-versus-literal reproduces the same rows. Independent literal reruns are
bit-exact, actual cache words/scales are checked, input hashes match and TT repeats
are exact. These are diagnostic facts, not an accepted stage classification.

Final public PLE v6 explicitly enables `--ple-backend-control`: all literal whole
rows/aggregates and all matched whole/contribution rows/aggregates must pass. Each
carried fresh row/aggregate must fail control-versus-literal at the same location.
Fresh prefixes31/33, resumed8/7, changed decode, negatives and state isolation
remain strict. The final binding requires false literal-strict disclosure and
records policy, source/config/checkpoint/cache identities and exact carried rows.

## Final evidence paths and closure work

These are binding plans, not invented successful results. Retain failed and
superseded records; validate actual completion before changing any status.

| Gate | Planned evidence and required closure |
|---|---|
| Short public boundaries | `linear_final_contract_v4.json`, `ple_final_contract_v4.json`, `qsa_final_contract_v4.json`; verify final sources, all literal gates and watcher provenance. |
| Public continuation/live unseen lengths | `public_continuation_layer{0,1,3}_v6.json`; layer0/3 strict literal, layer1 explicit fresh-only control above. Verify changed trace inputs, repeats, isolation, negative controls and QSA remap. |
| Batch | `linear_final_batch32_v4.json`, `ple_final_batch4_controlled_v4.json`, `qsa_final_batch32_v4.json`; preserve respective control disclosures and physical failed-batch evidence. |
| Full-context final public decode | `gdn_final_full_context.json`, `ple_final_full_context.json`, `qsa_final_full_context.json`; device-executed262143 prefix and traced262143 final position, literal decode/contribution, repeated state/output. Preserve reduced-prefix scope. |
| Full QSA aligned/nonaligned attention boundary | `qsa_final_device_full_boundary.json`; all262144 outputs, device checkpoint, restored262143 branch with31-row tail and final traced decode, paged state and all audits. |
| PLE complete public prefill and hard bound | `layer1_final_full_public_prefill.json`, `final_capacity_summary.json`; capacity still pending. Summary helper/schema and exact accepted measurements must be supplied before its currently empty gate checks can close. |
| Chunk components | `chunk_input_final_v5.json`, `chunk_output_final_v4.json`; validate final source/provenance and retained exactness/cache-growth/trace/canary evidence, not just filenames. |
| Synthetic CI | `synthetic_final_decoders_v4.json` and XML; full real PLE evidence complements component-only synthetic PLE. |
| Watcher | `final_watcher_summary.json` with hashes/archives, per-kind final commands and no fatal/invalid/overflow/sanitizer evidence. Keep separate from profiler runs. |
| Review/commit | Actual independent review rounds/verdict and fixed findings; local stage-owned commit SHAs in work_log, never push. |

Long helper schemas have been inspected. Bindings now explicitly distinguish
full public final decode from reduced prefix, require final positions, all-row
prefix checks, repeated state/output and runtime audit evidence. Existing
long filenames remain unchanged. Pending runs cannot inherit old numerical
passes from differently hashed sources.

## Final performance protocol

All six planned directories are
`tracy/layer{0,1,3}_{prefill,decode}_device_trace_v2/` with `profile.json` and
`report/`. Capture provenance is
`provenance/layer{L}_{mode}_device_trace_v2.json`.

The actual capture helper is `profile_trace_decoder.py --device-only`, using
`profile_trace_protocol.py`; the offline helper is `report_trace_performance.py
--device-protocol`. The new schema has `passed`, exact eager-versus-warm/measured
output and every state, runtime/allocation audits, two ordered replay calls and
`metadata_authority=harness_trace_protocol_v1`. Old `/finite`, `weights=real`,
eager/streaming filenames and streaming merge requirements do not match it.

Each mode needs these real offline outputs:

- `report/{mode}_perf_report.txt` and `.csv.xz` from actual tt-perf-report.
- `report/raw_device.csv.xz`, `report/whole_trace_ops.csv.xz`.
- `report/latency_summary.json` and `report/provenance.json`, completed offline
  conversion command/log provenance and exact capture/profile/native-CSV hashes.

A capture `passed` establishes equality, not timing acceptance. Strict offline
validation must match both replay ordinals and every independently inventoried
worker's raw endpoints, select the final warmed replay, and agree with the
native parser. Timing is TRACE-KERNEL first-to-last device elapsed, including
internal idle gaps. `device_kernel_sum_ms` and `op_gap_sum_ms` remain null;
per-op timings and roofline/utilization are unavailable. Host monotonic markers
establish ordering only; they are not substituted for device cycles or relabeled
as exported Tracy events. Prefill is a warmed whole-prefill trace replay; decode
is warmed traced decode. Report both methodology and measured shapes.

Layer1 v2 capture/offline artifacts are now present, but final source/provenance
and stage reconciliation remain required. The renderer header says Worker
cores120 because native parser `max_compute_cores` supplies the physical budget;
the independently validated participating logical grid is110 and the aggregate
row correctly reports110. No roofline/utilization claim is made. Native elapsed
cycles and latency are unaffected. Preserve closed reports/helpers and disclose
this metadata distinction in the final README rather than silently rewriting
measured evidence.

## Final reconciliation

Update README and context_contract only from completed scoped evidence. Replace
historical main tables with final-source facts, moving older passes/failures to
repair history. Include fresh-control literal failures and independent review
status, actual public prefill limits, advertised reduced state/decode coverage,
batch/storage policies and exact artifacts.

Refresh runtime and test/helper source identities after the final freeze. The
binding manifest's historical source snapshot is not a current attestation.
Its generic evaluator does not enforce `additional_required_artifacts` or prose
review items: offline report provenance, archives, table semantics and long
prefix export identity need explicit final reconciliation/independent review.
Do not mark these complete from file existence or a controlled `passed` flag.
Only the later independent stage-review may return clean-pass or the authorized
pass-with-carried after three rounds. This outline grants neither verdict.
