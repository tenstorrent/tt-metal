# Stage Review

Verdict: clean-pass

Independent review of stage03, optimized decoder for `google/gemma-4-26B-A4B-it`,
after the stage owner declared final evidence ready. This verdict covers the live
stage-owned tree on branch `gemma-4-26b-a4b-it`, based on
`b36116ff5c2560ed47148df57a784c857f864d73`, with optimized runtime SHA256
`5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`.
It does not relabel earlier measurements as executions of this source.

## Required Work

None. The correctness, trace-lifetime, cache-boundary, optimization-evidence and
documentation findings raised during this review have been resolved and checked
again. The local checkpoint commit and recording its SHA are the stage owner's
required post-review steps under the stage-review workflow.

## Other Concerns

- The accepted evidence is for representative sliding layer0 and full layer5 on
  one Blackhole ASIC, with real checkpoint weights and recorded HF layer inputs.
  It is not full-model generation or a qualitative text-quality result.
- The final policy is the fastest correct evaluated family supported by these
  controls. Small changes within timing spread are not ranked, and no absolute
  optimum is established.
- Some historical reports, logs, source snapshots and compact CSVs require
  reconstruction from their deterministic gzip companions. All 249 archives
  were independently decompressed and checked against original bytes, sizes and
  hashes after the final refresh. Raw large Tracy payloads and tensor fixtures
  remain local with reproduction commands.

## Hard-Check Gaps

No unmet required check remains. The following limits are explicit and do not
contradict the stage contract:

- Maximum-context prefill executes the whole TT layer, but HF accuracy is checked
  on291 selected query rows, including chunk boundaries and the final tail. It
  is not an all262144-row HF comparison.
- Current v8 evidence comprises12 primary commands plus five boundary commands.
  Unchanged sliding broad contracts are inherited from v5 and tracked lifecycle
  evidence from v6 through exact source comparisons. This is not claimed as a
  fresh18-command v8 suite. Full paths affected by HiFi2, M2 and L1 placement were
  rerun, including maximum context, prefix, B32, trace reuse and stress.
- B32 uses distinct short-context inputs, pages and positions. B1 maximum context
  does not establish32 simultaneous maximum contexts.
- Decode DRAM traffic and useful prefill FLOPs are explicitly defined estimates.
  They are not memory-controller counters or measured FPU utilization. The
  declared theoretical peak uses120 theoretical cores; the measured device has
  110 available workers. The distinction is disclosed.
- The native profile checks selected outputs; the separate correctness suite
  checks every one of128 headline and512 stress decode positions. A profile
  exit code alone was not accepted as correctness evidence.

## Anomaly Ledger

### A01: Live-trace allocations survived later prefill calls

**Observed anomaly:** Earlier reuse runs emitted a warning about allocations
surviving while a trace was live. Full allocation tracking then failed both
layer kinds: two sliding tail-clone buffers and53 full program-cache buffers.

**Evidence:** `trace_alloc_v5_layer{0,5}.log`,
`AUTODEBUG_trace_lifetime.md`, `AUTOFIX_review_contracts.md`,
`trace_alloc_v6_commands.json`, `validated_v8_request_reuse_layer5.json/log`.

**Affected path:** Prefill between replays of a retained decode trace.

**Control or comparison:** Original full-tracker failures versus both repaired
v6 runs and the current v8 full run, without tracker exclusions.

**Likely subsystem:** Private sliding-tail lifetime and program-binary allocation
from signatures first encountered after capture. Tracking proved surviving
allocations; actual address overlap was not claimed.

**Investigation performed:** Read allocator/program-cache and model ownership
code; checked exact recorded buffers and source deltas. The runtime drops final
unused tail clones. Reuse preinitializes its exact signature catalog and forbids
cache misses while the trace is live. Current full program count remains379
through all nine requests; inherited sliding remains370.

**Resolution:** fixed. An unseen signature requires initialization before capture
or release/rebuild of the trace. Arbitrary valid logical lengths remain supported.

### A02: A full-attention tail could exceed logical page-table capacity

**Observed anomaly:** S1025 with cache extent1152 could select K256 and round its
read end to1280. Padded page metadata could conceal this from an ordinary value
check or Watcher run.

**Evidence:** `AUTOFIX_review_contracts.md`, `source_delta_v6.json`,
`tight_cache_v6_commands.json`, `tight_cache_v8_layer5_1025.json`,
`tests/probe_optimized_cache_capacity.py`.

**Affected path:** Full paged prefill at a tightly allocated cache boundary.

**Control or comparison:** Source-derived read bounds, eight v6 tight-capacity
cases exercising K128/K256, and the current1025/cache1152 Watcher case.

**Likely subsystem:** SDPA chunk rounding versus the public logical page table.

**Investigation performed:** Read the SDPA factory/reader and logical-table
indexing. Checked the guard independently of numerical pass flags. The selected
fallback is Q64/K128 only when the wider rounded read exceeds capacity. Current
native-call evidence records read-end1152 at capacity1152. Near-tail cases and
prefill-only1152 exercise both programs; public capability is unchanged.

**Resolution:** fixed. Earlier numerical success is not used to dismiss the risk.

### A03: Full request-window accuracy failed despite clean cache controls

**Observed anomaly:** A real2049-token window produced decode PCC0.9948247936.

**Evidence:** `AUTOFIX_reuse.md`, `reuse_control_router_center.json`, current
prefix/reuse and512-step reports.

**Affected path:** Router score conversion and top8 selection.

**Control or comparison:** Fresh-request and isolated policy controls; centering
FP32 scores before BF16 conversion repairs the same window to about0.99818.

**Likely subsystem:** Router numerical sensitivity, rather than demonstrated
cross-request cache corruption.

**Investigation performed:** Reviewed the ablations, centering implementation,
preserved learned route scaling and subsequent current contracts. No unmeasured
router-ID flip is asserted.

**Resolution:** fixed; centered routing is selected and the real window passes.

### A04: Full maximum-context attention failed under LoFi

**Observed anomaly:** Earlier full long-context runs failed aggregate and sampled
query-row gates. A first larger-chunk attempt also exceeded L1 on the first
BF16-K/V chunk.

**Evidence:** `AUTODEBUG_long_prefill.md`, `long_prefill_config_results.md`,
`validated_v8_long_262144_layer5.json`,
`validated_v8_long_262143_layer5.json`.

**Affected path:** Full prefill attention precision and chunk-specific SDPA setup.

**Control or comparison:** Expert-precision controls did not repair the failure;
paged HiFi2 Q64/K256 did. The first nonpaged chunk retains legal128/128 geometry.

**Likely subsystem:** Attention numerical accumulation and first-chunk CB capacity.

**Investigation performed:** Read the phase-specific branch and failed/successful
controls. Independently checked291 unique current rows for each long length,
minimum0.996029643684647, plus final tails and boundary decode.

**Resolution:** fixed, with no context reduction or relaxed threshold.

### A05: Aggregate PCC hid sliding long-context row failures

**Observed anomaly:** Four real sampled sliding rows failed even though aggregate
PCC passed.

**Evidence:** `AUTOFIX_sliding_prefill.md`, `AUTODEBUG_sliding_prefill.md`,
`validated_v5_long_262144_layer0.json`,
`validated_v5_long_262143_layer0.json` and the source-delta chain.

**Affected path:** Sliding prefill expert gate/up precision and long-test acceptance.

**Control or comparison:** BFP8 gate/up repairs all four rows; down-only, QKV,
output and attention changes do not. The fused real-input control passes.

**Likely subsystem:** Expert gate/up numerical sensitivity.

**Investigation performed:** Inspected isolated actual-input reports and the
strengthened per-row gate. Verified unchanged current sliding arithmetic and
the inherited291-row minima0.9951303778312409.

**Resolution:** fixed. BFP8 gate/up shares the existing decode weight allocation;
BFP4 down remains. Historical aggregate-only checks are not rowwise passes.

### A06: Faster or lower-precision candidates failed real-input accuracy

**Observed anomaly:** Attention BFP4, sliding native head normalization, sliding
router HiFi2, full minimal-QKV K8 and sliding minimal-QKV HiFi2 each had real
acceptance failures despite some shorter checks passing.

**Evidence:** `precision_evidence.md`, `native_headnorm_control.md`,
`router_direct.md`, `minimal_selection.md`,
`minimal_hifi2_acceptance_v6/long_layer0.json`, `minimal_pairs_v6_summary.md`.

**Affected path:** Attention/expert-adjacent projection precision, head norms,
routing and minimal prefill arithmetic.

**Control or comparison:** The selected alternatives pass the same real cases.
Notably native head norms fail positions1257/1459; full K8 fails position4211
at0.994989; sliding minimal HiFi2 fails maximum-context row32 at
0.9949930133358956. Full HiFi2 passes and is adopted.

**Likely subsystem:** Precision-sensitive layer arithmetic.

**Investigation performed:** Checked thresholds and individual rows, source
policy consumption, matched geometry and later cumulative gates. Synthetic
PCC is not used to veto a real-input winner.

**Resolution:** controlled. Failing candidates are rejected; correct faster full
HiFi2 is retained. No numerical failure is waived.

### A07: First L1/API errors could have caused premature family rejection

**Observed anomaly:** Selected full QKV M4 with L1 placement overlaps static CBs;
reader1 K11 exceeds L1; an initial reader wrapper fails on Shape slicing.

**Evidence:** `qkv_l1_results.md/json`, `prefill_qkv_l1_v7_command.json`,
`dram_reader_results.md`, `reader_layer_results.md` and their command journals.

**Affected path:** Minimal QKV placement, DRAM reader programs and control harness.

**Control or comparison:** QKV M2 and N4 adaptations both execute and pass;
matched M2 placement and producer controls identify the selected faster path.
Legal reader1 K1 and readers2/3 execute; all22 layer integrations pass.

**Likely subsystem:** Static-CB/dynamic-L1 capacity and a test-only shape API error.

**Investigation performed:** Independently checked CB formulas, legal adaptation
configs, paired samples and actual source bindings. QKV CB payload falls from
1,196,032 to737,280 bytes per compute core at M2. The copied-L1 producer variant
is further improved by the existing normalization multiply writing L1 directly.

**Resolution:** fixed/controlled. The QKV family is adopted after adaptation;
reader alternatives are rejected by completed whole-layer measurements, not
their initial errors.

### A08: QKV output placement was initially described incorrectly

**Observed anomaly:** Some placement descriptions assumed DRAM QKV output even
though the v8 native rows showed L1 output.

**Evidence:** Current full `ops.csv`, `source_delta_v8.md/json`,
`final_memory_accounting.md`, `qkv_l1_results.md`, `final_policy.json`,
`../context_contract.json`; minimal-matmul device operation line297.

**Affected path:** Full prefill normalized input, minimal QKV and tied-K/V slice.

**Control or comparison:** Raw v7 and v8 operation rows around all four QKV calls.

**Likely subsystem:** The op defaults output memory to input memory when its
output-memory argument is absent.

**Investigation performed:** Independently traced the exact chain: weighted norm
output goes L1, QKV output and tied-K/V slice inherit L1, concat returns DRAM,
then head creation onward keeps the previous placement. The placement benefit
is therefore coupled input/output placement. Payloads are recorded separately,
without inventing a summed allocator peak.

**Resolution:** fixed in current policy, source proof, advice, memory/context
documentation and README. Measurements and source bytes were not rewritten.
Current Watcher, maximum context and tracked reuse already exercise this path.

### A09: Apparent small timing wins and host/device arithmetic gaps

**Observed anomaly:** Some reader/router candidates have small negative medians
within spread; sliding fixed-position host time is below the mean varying-
position device window.

**Evidence:** `reader_layer_results.md`, `prefill_router_results.md/json`,
current `timing_reconciliation.json` for both kinds.

**Affected path:** Performance selection and interpretation.

**Control or comparison:** Alternating paired samples, stable warmed program
counts, all128 device windows, and same-run host signposts.

**Likely subsystem:** Measurement variation and different execution regimes;
no specific hardware cause is inferred.

**Investigation performed:** Recomputed all four32-pair router results and their
ordered spreads. No resolved gain supports changing router defaults. Recomputed
same-run host means896.349664063/914.732039063us and retained the sliding
fixed-position minus device difference of−1.444611574us without clamping.

**Resolution:** controlled. Host, device, profiler and candidate scopes remain
separate; tiny differences are not claimed as improvements.

### A10: Profiler/report limitations could misstate the executed policy

**Observed anomaly:** The report does not parse MinimalMatmulConfig and does not
infer active experts from indexed sparse operands. Prefill expert unions vary.

**Evidence:** `final_perf_advice.md/json`, `final_roofline_audit.md/json`,
native projection audits, current report commands and raw rows.

**Affected path:** Program advice, dtype/config verification and traffic estimates.

**Control or comparison:** Explicit raw minimal `config=` attributes, UINT16
eight-index operands, compact E8 outputs and resident E128 weights.

**Likely subsystem:** Report parsing/accounting coverage, not missing runtime config.

**Investigation performed:** Checked actual full M2/K16/HiFi2 L1 and sliding
M4/K8/HiFi4 DRAM rows, output projection and every decode matmul family across128
replays. Decode active8 is supported by metadata; prefill never fabricates8.
Independent fidelity/grid/placement recommendations were still tried.

**Resolution:** controlled with explicit native proof and scoped estimates.

### A11: Existing grouped-FFN/remap operators are not a drop-in replacement

**Observed anomaly:** Prefill union-expert computation remains a large share of
device time; a generic claim that existing routed FFN solves it would be unsupported.

**Evidence:** `prefill_routed_ffn_contract_audit.md`, current native sparse rows.

**Affected path:** Prefill token dispatch, GELU expert computation and combine.

**Control or comparison:** Actual existing op activation, dtype and remap contracts.

**Likely subsystem:** Missing compatible GELU grouped-dispatch/combine interface.

**Investigation performed:** Read the relevant existing implementations. The
named FFN hardcodes SiLU or exposes non-GELU activations; remap ops manipulate
routing metadata rather than gathering activations/counts needed for this FFN.
The restriction is not inferred from model name or expert dimensions.

**Resolution:** controlled by exact existing-op contract evidence. No promised
drop-in optimization is deferred to a later stage.

### A12: Harness success codes and metadata failures were not reliable alone

**Observed anomaly:** A Tracy invocation returned zero after argument parsing
failed; a long-result postprocessor treated a list as a dict; a profile campaign
stopped before hardware because its validation manifest was absent.

**Evidence:** `operator_audit_failed_cli_commands.json`,
`minimal_hifi2_acceptance_v6/metadata_failure.json`,
`minimal_hifi2_acceptance_v6/full_long_resume_command.json`,
`v6_final_campaign.json`, completed operator/profile journals.

**Affected path:** Evidence collection and campaign orchestration.

**Control or comparison:** Named-case Tracy wrapper with required runner/native
outputs; preserved complete numerical report; separately journaled continuation;
standalone profiles after the correct validation manifest existed.

**Likely subsystem:** Shell argument boundaries and host metadata code.

**Investigation performed:** Checked exact commands, returns and required output
existence. Failed attempts are retained and excluded from passing measurements.

**Resolution:** fixed/controlled. No device failure or completed test is invented
from the misleading wrapper result.

### A13: Formatting and ignore rules threatened evidence provenance

**Observed anomaly:** Formatting altered a reader probe during a campaign;
historical executed helpers later needed formatting; root ignore rules hid logs
and CSVs, while some reports exceeded the repository size limit.

**Evidence:** `dram_readers_v5/provenance_incident.*`,
`minimal_prefill_probe_format_equivalence.json`,
`prefill_pair_helper_format_proof.json`, exact source snapshots,
`package_evidence.py`, `evidence_archives.json`.

**Affected path:** Source attribution and reproducible committed evidence.

**Control or comparison:** Per-phase reader hashes and reconstructed execution
plan; byte-exact old helper snapshots; full-AST comparisons; gzip originals.

**Likely subsystem:** Formatting/packaging workflow, not accelerator arithmetic.

**Investigation performed:** Independently verified both helper AST equivalences,
including old pair-helper hash13f71dfb and formatted hashb682c87a. Rechecked all 249
final archives, including 58 CSVs and 104 logs. Historical measurement bytes remain
unchanged. Required staged pre-commit passes1546 files without mutation; explicit
pinned Black passes85/85 Python files.

**Resolution:** fixed/controlled. No required formatting exception remains and
no hook was bypassed. Raw tensors/large Tracy dumps are not staged.

### A14: Stale current-document labels and benign environment warnings

**Observed anomaly:** During integration, primary advice/checklist files still
described older source versions or pending completed work. Logs also contain
motherboard/subset-MMIO notices and Python SWIG deprecation warnings.

**Evidence:** Final README/checklist/advice/context files and current correctness,
Watcher, profile and pytest logs.

**Affected path:** Stage closure claims and log interpretation.

**Control or comparison:** Frozen v8 artifacts and source hashes versus explicitly
archived v4/v5/v7 records; successful separate Watcher and device executions.

**Likely subsystem:** Documentation refresh and host discovery/binding metadata.

**Investigation performed:** Reread the final primary documents after refresh and
checked current links/hash bindings. Environment notices do not report failed
model operations; all required outputs and device checks complete. Trace
corruption warnings were separately investigated under A01, not grouped here.

**Resolution:** fixed/controlled. No stale profile is presented as v8 evidence.

## Scope Inspected

**Goal/skill paths:**

- Original stage contract:
  `/workspace/tt-metal/bringup/artifacts/multigoal-runs/20260925T171711Z/03-03-optimized-decoder.prompt.txt`.
- Installed `stage-review`, `optimize`, `tt-device-usage` and model-bringup startup
  requirements under
  `/home/mvasiljevic/.codex-personal-gemma4/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.14/skills/`.
- Repository authoring instructions and the actual formatting-hook coverage.

**Artifact paths:**

- README, work log, all38 checklist entries, final policy, context and memory
  accounting, current advice/roofline audits and their historical snapshots.
- Broad v5 validation, v6 trace/capacity repairs, exact v6/v7/v8 source chains,
  current v8 validation, all current primary/boundary outputs, stress comparisons,
  Watcher/environment logs, fixture provenance and command journals.
- Precision/geometry/topology controls,28 legal isolated reader cases and two
  exact capacity failures,22 reader layer controls, minimal projection and
  producer-placement comparisons, four32-pair prefill-router controls.
- Both raw v8 native CSVs, report CSVs/tables, complete replay windows,
  reconciliation, archive manifest and staged pre-commit/Black records.

**Code paths:**

- Entire optimized decoder and relevant inherited fused/functional attention,
  routing, normalization, cache and public-call behavior.
- Contract runners, batch/prefix/reuse/long/stress tests, capacity guard,
  candidate probes, profiling and accounting scripts.
- Relevant TTNN SDPA reader/factory, minimal matmul memory/CB rules, indexed
  sparse topology, trace allocation/program-cache behavior and existing
  grouped-FFN/remap contracts. No C++ changes are part of this stage.

**Commands run:**

- Read-only `cat`, `sed`, `grep`, `find`, Git status/diff/hash inspection and small
  Python-standard-library artifact scripts. `rg` was unavailable.
- Independent AST/source comparisons; original/archive/manifest SHA256 checks;
  row-level PCC, unique-position and cache-count checks; paired-sample arithmetic;
  CSV signpost/session grouping, firmware-span reconstruction and native
  dtype/config/memory inspection.
- `git diff --check` and final
  `git diff --cached --check -- . ':!*.patch'` pass. Historical patch context is
  intentionally preserved and excluded by the repository whitespace hook.
- The reviewer inspected the owner's hardware and pre-commit evidence; the
  reviewer did not run hardware, import TTNN, start servers, reset devices or
  rerun long tests. This report is the reviewer's only written artifact.

The final staged1546 files were all inside the model autoport directory, with no
raw tensor or raw Tracy files. The runtime hash remained unchanged. Independently
checked37 path/hash bindings in the current validation manifest matched.

Raw timing reconstruction exactly reproduced:

| Layer kind | Prefill device us | Decode device us, mean128 | Prefill speedup vs fused | Decode speedup vs fused |
| --- | ---: | ---: | ---: | ---: |
| Sliding | 221367.854917 | 881.410845 | 12.7087x | 5.8390x |
| Full | 186562.835556 | 899.335532 | 15.1437x | 6.2119x |

The measured prefill windows contain2169/2109 native operations; decode has128
sessions of123/126 operations. Both kinds pass all128 headline checks and512
stress checks, with exact repeat equality. No failed/shared HF rows are excluded.

## Residual Risk

- Numerical acceptance covers representative kinds and the recorded input set,
  not every layer/input. Some selected sliding sampled rows are close to.995;
  the threshold is enforced without rounding or exclusions.
- Trace-safe reuse depends on the documented preinitialized signature catalog.
  A caller must manage trace lifetime when introducing an unseen signature.
- Cache capacity is validated through source bounds, selected nearby cases and
  maximum-context execution, not every possible page arrangement/length.
- Performance is measured on one ASIC and finite warmed samples. It should not
  be extrapolated to full-model serving, other hardware or unmeasured batch/context
  combinations.

These limits are disclosed and do not leave required stage work unresolved.
