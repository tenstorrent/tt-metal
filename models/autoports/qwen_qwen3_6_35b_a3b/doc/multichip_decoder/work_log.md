# Multichip decoder work log

Stage4, Qwen/Qwen3.6-35B-A3B. Baseline commit2901d8a5. Worktree initially clean.
Scope: multichip_decoder.py, its tests and docs. No full-model/vLLM work or push.

Current status: all 27 acceptance gates pass; independent round-2 review is
`clean-pass`. Local checkpoint details appear below. Entries retain the execution chronology.

- Read enabled installed v0.1.24 multichip, tt-device-usage, model-bringup,
  optimize, tracing and review skills; tt-autodebug0.1.8 enabled.
- Read LLM report3.3, optimized decoder evidence and relevant reusable CCL/MoE.
- Source python_env/bin/activate then ../lane-env.sh for jobs.
- `fuser /dev/tenstorrent/*`: no owners. `timeout60 tt-smi -ls --local`:
  four Blackhole p300c chips0–3.
- `timeout120 python` with FABRIC_1D and MeshShape(1,4): open/close PASS,
  log `logs/mesh_smoke.log`;11x10 worker grid. No resets.
- Chosen design recorded before final implementation in `mesh_plan.md`.
- Independent CPU subagents: linear head-sharding audit/candidate and paired
  single/multichip validation runner. No concurrent device work delegated.

Initial status: implementation in progress; no multichip correctness/performance gate
has passed yet. Baseline evidence is not substituted for multichip evidence.

First full-layer smoke reached EP expert compute but failed host validation: fused
SwiGLU requires fp32_dest_acc_en=False (eight DEST tiles), unlike indexed sparse
experts. Added separate grouped compute config; indexed policy unchanged. Exact
log logs/smoke_l3_s32.log. Reset1 follows failed multichip run; no process killed.

Reset1 list and mesh/trivial-add smoke passed. Second smoke v2 executed full
attention+EP prefill and repeated determinism, then sparse decode validation
required UINT16 IDs: FP32 topk produces UINT32. Added explicit device typecast.
Reset2 (all unowned chips) requested under multichip failed-run reset rule.
No hang or unhealthy-device symptom occurred in either host validation failure.

Reset2 list and trivial mesh smoke PASS. Third smoke v3 reached indexed-expert
reduction, whose implicit reshape lost the baseline explicit padded shape.
AutoDebug launcher preflight failed because unprivileged namespaces are disabled;
retried with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under existing unrestricted-execution
and subagent authorization. Child is inspection-only. Logs autodebug_launch.log
and autodebug_retry.log. Reset3 requested after failed mesh job; three-reset
stage allowance is now consumed. No process was killed, no device hang seen.

Capacity arithmetic: tests/multichip_capacity.py reads pinned config (40layers,
10full/30linear). Per-chip planned bytes27828420608, headroom6531317760 under
32GiB assumption. context_contract.json now includes a clearly pending multichip
plan, without changing historical validations or262144 supported context.
Shared serial EP output workspace should avoid per-layer dispatch-buffer
replication in the future stack; no full-model implementation is started.

Supplemental source finding during AutoDebug: FastReduceNCDeviceOperation::
compute_output_specs uses padded_shape unless a preallocated logical output is
provided. Thus singleton MoE result, singleton ordered projection, and the new
eight-head recurrent reduction all need their intended logical shape restored.

Fourth full-attention smoke `smoke_l3_s32_v4.json` PASS: prefillPCC
0.999930024, decodePCCmin0.999936756, K0.999988075/V0.999989574. Paired
permuted pages and duplicated KV ranks agree; changed-input/position replay
is deterministic. TT_METAL_TRACE_ALLOC_TRACKING=1 and program-cache tracking0
(skip flag0) were enabled. Three timed replays are only screening: decode
speedup0.83999, efficiency0.21; prefill speedup1.0642. Performance remains work.
Logical/padded shape fixes restored singleton routed result, singleton ordered
projection result, and eight-head recurrent sum. Linear ordered-norm control
preserves baseline sensitive arithmetic while distributed remains selectable.
Shared CollectiveWorkspace supports serial CQ0 reuse across decoder layers;
EP buffers use power-of-two capacity buckets with logical tokens unpadded.
First linear smoke launched with paired baseline and trace tracking.

Linear smoke `smoke_l0_s32.json` passed: prefill PCC0.999988819,
minimum decode PCC0.999995943, conv/recurrent approximately1. Screening
speedups1.568 prefill/1.274 decode; final timing remains pending.

B32 S33 full-layer test failed host validation in post_combine_reduce,
which requires a multiple of32 rows. `_moe` now internally pads the row axis
and slices to the logical row count. Attention positions and caches retain
logical length33. Validation is pending in the retry.

Reset-policy correction: a preceding summary incorrectly asserted a total
three-reset cap. Re-reading the installed tt-device-usage/SKILL.md gives
"There is no fixed limit on resets; log each in the stage work log."
Only three consecutive resets that fail to recover hardware require escalation.
All previous health checks passed. The unnecessary approval question was
withdrawn; reset4 follows the failed host-validation job, with no owners found
by fuser and no process killed. Exact outputs: logs/reset_4.log.

Reset4 and mesh add/open/close smoke PASS. Nonaligned B32 full retry
`batch_l3_b32_s33_v2.json` passed; prefill0.999935500, decode min0.999904415.
All eager stream steps exactly matched trace replay, determinism and live-trace
prefill replay passed with allocation tracking enabled. The internal MoE pad
fix is now device-verified. No attention/cache length padding was introduced.

Added explicit projection candidates: interleaved control versus baseline
DRAM-sharded projection builder applied to local mesh shapes, and ordered
(full gather then baseline norm) versus distributed full-layer norm. These
are construction choices with no runtime weight conversions. Batched linear
correctness and DRAM/ordered full candidate are queued serially.

B32 S33 linear PASS (`batch_l0_b32_s33.json`): prefill0.999990290,
minimum decode0.999986738, screen decode speedup1.8343.
First DRAM/ordered candidate failed because inherited norm returned L1 shards
that propagated through dispatch to an op requiring DRAM interleaved. Explicit
MoE input conversion fixes the contract; reset5 plus mesh smoke PASS. No hang,
no process termination. Ordered and distributed DRAM retries both PASS,
respectively minimum decode PCC0.999983945/0.999936756 and warmed mesh
latency0.4815/0.4872ms (instrumented screening,20 repeats).

Trace environment correction: earlier shell commands set the extra unused
TT_METAL_TRACE_ALLOC_TRACKING_SKIP_PROGRAM_CACHE=0. The real flag is
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE and was unset, whose source default
is false. Thus program-cache tracking was enabled; future commands explicitly
set the correctly named flag to0. This does not invalidate those tracking runs.

Fused O projection MM-RS first attempt hit a compute-config binding mismatch:
new API takes ComputeConfigDescriptor, not WormholeComputeKernelConfig. Adapted
at construction, reset6/smoke PASS, and fused retry `collective_l3_fused_ordered`
PASS: prefill0.999975127, minimum decode0.999984842,20-replay screening0.47054ms.
The reduced512 output feeds residual add and next norm directly, no immediate
all-gather introduced. Internal32-row pure-projection padding does not touch KV.

AG-MM legacy API rejected Linear topology in the native program factory after
prefill passed. Investigating the strided minimal AG-MM implementation as the
compatible optimization family, rather than dismissing fusion at this first
API mismatch. Reset7 requested per failed-mesh rule. No hang/no process killed.

Mixed sharded stack PASS: min whole-output PCC0.999714846 across prefix,
continuation and decode; changed page tables and heterogeneous current positions
under one trace pass. Artifact mixed_stack.json. Resident census included.

Topology sweep completed: direct AR wins both layer kinds in tracked screening.
The adapted Linear minimal AG-MM ran correctly at32 and64 internal rows but
was slower; no unsupported-family dismissal remains. Selected defaults now use
replicated residual, local norm, DRAM decode projections, direct BF16/FP32 AR.
Shared workspace owns the1.5MiB direct-AR scratch to avoid multiplying per-core
L1 allocation by40 layers. Selected mixed-stack, natural B32 and context jobs
run serially via tests/run_multichip_acceptance.sh; STOP file checked between
jobs. Existing sharded evidence remains an alternative, not selected evidence.

Reset8/smoke PASS; selected mixed stack retry PASS after normalizing prefill
concat layouts. It shares direct-AR and EP scratch across layer0 and layer3.
The acceptance driver continues to natural B32 and long-context evidence.
Formatting/lint prior to acceptance: logs/precommit.log, all applicable hooks
passed. New validation driver tests/run_multichip_validation.sh defines clean
B1/B32 timing, boundary lengths1/31/65/257,128-step decode stress, and separate
watcher B1/B32 runs. These queued gates are not yet claimed passed.

Reset9 and mesh health smoke PASS after the B32 sharded concat fan-in failure.
No owners were present before reset; no process was killed. A two-level concat
with at most 16 inputs preserves the selected L1 residual layout. The resumed
acceptance command was `bash tests/run_multichip_acceptance.sh natural` (paths
relative to this autoport; launched from repository root using the full path).
`natural_l0_b32.json` passed with prefill PCC 0.999966521 and minimum decode
PCC 0.999902546; `natural_l3_b32.json` passed with prefill PCC 0.999969649 and
minimum decode PCC 0.999854728. Both use 32 distinct real prompt activations,
eight changing decode steps, trace allocation tracking and repeated streams.
`context_l3_32k.json` also passed at logical length 32769 with a 65536-token
cache. Linear 32K and the single changed-KV-layout maximum check remain running.

Linear 32K also passed. CPU audit then found a maximum-linear-length concat
limit: 4096 chunks require at least 8195 unique native runtime words, beyond
the usable 4094-word kernel limit. Added ordered bounded concat (DRAM fan-in
64, L1 fan-in 16), preserving logical order and all cache updates. The initial
full maximum run is allowed to finish; context gates will be refreshed against
the final source, reusing identical saved optimized baselines. No baseline
source is changed. Added all-rank residual equality at test boundaries and
strict final-source/default-policy checks to the acceptance summarizer. Shared
physical KV-pool coverage is being added to the mixed-stack test.

Profiler collection now supports B1 and prompt-diverse B32 in separate roots.
The B32 root is `perf_b32/`; each capture still records 16 profiled replays and
reports whole-batch durations. These are pending device evidence.

Initial full maximum context PASS: length 262142 plus two decode steps,
prefill PCC 0.999976139, minimum decode PCC 0.999976423, K/V PCC above
0.9999999996 and ownership checks pass. First mesh prefill took about eight
minutes including new offset-specific slice-kernel compilation. Read-only
`tt-triage --run=dump_running_operations` found no stalled operation; active
compiler descendants confirmed progress. No hang, timeout, kill or reset.
After the final device job printed PASS and closed, the live-edited acceptance
shell reported an EOF parse error. This was a driver edit during execution,
not a device failure; `bash -n` passes on the finalized script. Future driver
edits are made only while it is inactive. Final-source context refresh remains
required after bounded concat; saved optimized controls avoid rerunning them.

B1 profiler captures PASS for both kinds. Raw CSV critical-rank kernel-duration
sums per decode replay are 0.882343 ms (linear) and 0.365605 ms (full); these
exclude gaps and are not clean elapsed latency. `tt-perf-report` merges ranks:
maximum duration for noncollectives, mean for collectives. Its displayed device
is a representative, not a per-rank workload. The summarizer uses raw rows and
records this distinction; see `profile_findings_b1.md`.

Historical first B32 profiler attempt failed the tooling limit "Too many source locations"
(32K) with uncached op descriptions; the workload later printed "Killed" and
no complete device CSV was produced. This is incomplete instrumentation, not
a passing correctness result. Preserved at `failures/profiler_b32_no_cache/`.
No owner remained. Reset10 list/reset/list and trivial-add mesh smoke PASS;
the agent did not kill a process. Collector B32 now enables metadata caching
by native program hash; the retry remains pending. B1 uncached evidence stays
unchanged. Clean local dense projection geometry screening runs first via
`python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_geometry`.

## Selected geometry and final-source acceptance

Dense projection screening (`geometry/search.json`) measured reader counts, K blocks
and input/output core counts at unchanged dtype/fidelity. Best output-core variant
was only 0.16% faster; the existing geometry remains selected. Sparse screening
(`sparse_geometry/search.json`) measured DRAM/L1 inputs, GU K blocks 16/32/64,
smaller GU/down core grids and wider output subblocks. DRAM GU block 64 achieved
0.3926215 ms versus 0.4057965 ms control (3.25%); it is the selected default.
Down geometry stays 64 cores, K block 4. These are whole-layer matched screening
measurements, not final single-chip speedup claims.

Local-only expert dispatch uses the native -1 remote-expert table and zeros
remote counts before aligned offsets. B32 capacity falls from 8192 to 2240.
Matched full-layer decode improves 6.846780 to 6.443363 ms; linear improves
22.209062 to 21.729884 ms. Full prefill improves 46.465132 to 43.017564 ms;
linear prefill increases 1321.335477 to 1331.782327 ms (0.79%). Selected for
reduced movement and decode improvement; no claim that every phase improves.
Exact evidence: `local_dispatch_l{0,3}_{false,true}.json`.

Final-source acceptance passed shared physical KV pool versus independent caches,
guard/future-cell preservation, all-rank output equality, changed page tables,
nonaligned resumed prefill, and poisoned expert workspace with three empty ranks.
Mixed stack, natural B32 both kinds, 32769-token context both kinds, and full
262142-token prefill plus two decode steps all passed. Commands:
`bash tests/run_multichip_acceptance.sh all` from this autoport (repository script
uses repository-relative paths, so execute its full path from repo root).
The fresh `run_multichip_validation.sh clean|boundary|stress|watcher` sequence
now runs serially. Earlier B1 profiles moved intact to
`profiling_global_dispatch/`; selected-source B1 and B32 profiles remain required.

Fresh clean paired timing completed for both kinds at B1/B32. See
`latency_comparison.csv` for all prefill/decode medians, speedups and efficiencies.
Linear decode: 1.348760→1.015573 ms B1, 42.490509→21.726680 ms B32.
Full decode: 0.432803→0.393623 ms B1, 13.495971→6.453329 ms B32.
Single-chip controls executed freshly; no reused baseline, profiler, watcher or
allocation tracking in these timings. All whole-output/state PCC and trace
contracts passed. Source 97792b176b8fdc2f41d0b1c9c6ab7e98f08dd7bb94d87fac08b008c3e27fcaa3.

CPU audit corrected the shared-pool gate's coverage checks and requires both B1
and B32 raw-rank profiler reconstructions. Final context/capacity arithmetic
was regenerated with all three passing context artifacts and shared-stack
constructor census. Independent source audit confirmed inherited sparse policy
keys are reporting-only; actual mesh geometry uses `mesh_sparse_geometry` and
explicit source defaults. Historical JSON is retained unchanged; manifest
explains the distinction and final raw profiler rows verify actual precision.

Interim gate check: 19/26 pass. Only unfinished stress, watcher and final
profiler evidence is missing. All eight nonaligned boundary cases pass.

All four watcher device jobs returned PASS with mesh trace controls. The CPU
gate rejected B1's reused early baseline because its older summary lacks the
all-step eager/trace flag. This is an evidence-contract gap, not a failed mesh
run; no reset is required. After profiler completion, rerun B1 watcher with
fresh controls (driver now omits baseline-load for B1). B32 reused controls
already contain the required flags. Final profiler queue runs B32 then B1,
serial layers 0 and 3, with the collector and CPU summary commands recorded in
artifact provenance.

Historical cached B32 profiler retry also exceeded 32K source locations, workload printed
Killed, and postprocessing lacked the device CSV. No passing measurement is
claimed. Capture preserved under `failures/profiler_b32_cached/`. No owner or
leftover workload remained; reset11 list/reset/list and trivial-add four-rank
smoke passed (`logs/reset_11{,_smoke}.log`). No process was killed by the agent.
AutoFix uses an isolated source audit to narrow the fixture; see AUTODEBUG and
AUTOFIX. Root serial queue now creates paired state seeds, reruns B1 watcher
with fresh controls, then collects selected B1 profiles.

Narrow B32 profiler implementation and CPU schema checks are ready. It validates
all256 state uploads, one representative prefill, step0 eager/trace equality,
two changing-input replay steps against accepted mesh tensors and optimized
outputs, then16 profiled replays. Source identities and exact seed paths are
retained; individual seed artifacts are not hashed, per tt-device-usage.
Full correctness flags are inherited through the explicitly linked eight-step
paired seed acceptance, never fabricated in the narrow capture. The profiler
summary labels prefill as one representative request and decode as a whole
B32 step. Fresh B1 watcher replacements passed. Selected B1 linear profiler
capture/reports passed; full layer collection follows serially.

## Narrow B32 capture and CPU report conversion

The narrow linear B32 device fixture passed and closed the mesh. Exact device
results and provenance are `perf_b32/tracy/layer_0/correctness.json` and
`correctness.json.provenance.json` in that directory; `correctness.pt` retains
the checked outputs. Its prefill phase measures **one natural 32-token prompt**,
not 32 prefills. Decode measures one complete B32 step initialized from all
32 accepted post-prefill states in `seed_l0_b32.pt`, linked to
`seed_l0_b32.json`. The fixture checks state uploads, step 0 eager/trace equality,
two changing-input replay steps and deterministic repeated streams before the
16 profiling replays. This device PASS does not imply completed profiler reports.

The subsequent, now historical, CPU report-conversion attempt exited 137 after
the device had closed. Its raw host timing CSV is approximately 35 GiB and the
legacy device CSV approximately 44 GiB; pandas memory reached approximately
249 GiB. The PID was already gone when inspected; no agent killed the process.
This was a CPU conversion OOM, so no device reset was required. The complete raw
capture remains in `perf_b32/tracy/layer_0/raw/.logs/`; `capture.log` in its parent
records the successful capture and the native pandas import reached afterward.

The model-local CPU repair is `tests/postprocess_multichip_profile.py`. Its
current version exports `TT_` host roots directly from the saved trace, then
streams native-relevant host rows and device sub-device metadata needed for
lookup into a separate `postprocess/.logs/` directory. Optional CompileProgram
and HWCommandQueue_write_buffer host child timings are excluded and recorded
explicitly in `host_times_compaction.json`; device timing and op metadata are
unchanged.
It preserves all raw inputs and uses the original authoritative
`raw/.logs/cpp_device_perf_report.csv` for device timing. Native report conversion
is unchanged. Command, from the repository root:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.postprocess_multichip_profile \
  --artifact models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/perf_b32/tracy/layer_0
```

CPU processing is running; progress is recorded in
`perf_b32/tracy/layer_0/postprocess.log`. Planned compaction census artifacts are
`host_times_compaction.json` and `device_metadata_compaction.json` in that
directory. Report conversion and B32 four-rank accounting are not yet accepted;
the narrow full-attention B32 capture also remains pending.

The first streaming attempt, now historical, rejected a malformed unfiltered
host CSV after scanning over 300 million rows. `Program_` source-file fields
contained unescaped commas/newlines; the strict row-width assertion caught the
problem. Preserve `perf_b32/tracy/layer_0/postprocess_malformed_host/` and
`postprocess_malformed_host.log` in that directory. The raw export was not edited.
The revised path uses the native exporter options `-u -c -f TT_` on the saved
trace to produce `perf_b32/tracy/layer_0/raw/.logs/tracy_ops_root_times.csv`.
It then applies the native-relevant row predicate, with the optional host-child
timing omission above. All authoritative C++ device timings and op metadata are
retained. CPU processing is running in session 78084, and an independent CPU
audit is checking omitted-child semantics. No new hardware job or reset occurred;
report acceptance remains pending.

## B32 profiler tooling resolved

Both narrow B32 device runs and native CPU CSV conversions now PASS. The
linear conversion command recorded above succeeded; the full-attention capture
and conversion used:

```bash
bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_multichip_profile.sh 3 \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/perf_b32/tracy/layer_3
```

Both `perf_b32/tracy/layer_{0,3}/` directories contain `ops.csv`, `ops_source.txt`,
`host_times_compaction.json`, `device_metadata_compaction.json` and
`postprocess.log`. Native root order is preserved: 44,876 linear and 19,596
full-attention host rows retained. Legacy device scans covered 306,761,184 and
51,894,480 rows respectively, with no sub-device metadata and passing native
lookup equivalence checks. Original C++ device timings and op metadata are intact.
Independent CPU audit confirmed the omitted nonroot host-child timing columns
are optional and unused by device timing and `tt-perf-report`.

The tooling failures are resolved. Human-readable tables are rendering; final
profiler summaries and acceptance gates remain pending. Prefill still denotes
one natural 32-token prompt; decode denotes a complete B32 step initialized from
32 accepted states. The broad `collect_multichip_perf.sh` now rejects B32 with
exit 2 and directs callers to the narrow collector; CPU shell syntax and
rejection-path checks passed. No reset or decoder change was needed for the
CPU conversion repair.

## Final acceptance before independent review

All 26 gates passed after rebuilding B1/B32 profiler summaries and checking
them independently against raw four-rank CSVs and all sixteen replay signatures.
Commands: `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_multichip_perf`
(B1), the same with `--batch 32 --doc models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/perf_b32`,
then `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_multichip_validation`.
Result: `MULTICHIP_GATES PASS: 26/26`, recorded in `gate_summary.json`.
`tests/summarize_multichip_moe_share.py` also passed for all four captures.
The CPU analysis corrected SDPA's category collision with the substring `pad`;
it changed compute/movement accounting only. The exact B32 output-split lowering
is now validated by the MoE attribution script and excluded from its numerator.

Final selected source identity remains
`97792b176b8fdc2f41d0b1c9c6ab7e98f08dd7bb94d87fac08b008c3e27fcaa3`.
B32 linear/full kernel-work sums are 19.730907/5.802994 ms, with MoE shares
4.67%/14.45%; these are not clean elapsed times. Final speedups and efficiency
remain those in README and latency_comparison.csv. Reports analyze actual
CCL, DRAM placement, compute, data movement and dtype/fidelity.

Compact CSV line endings were normalized to LF with parsed-field equality
verified; no measurements changed. All devices are closed and unowned.
Formatting and scope checks precede the fresh independent stage review.

## Review round 1 follow-up

Independent [round 1](stage_review_round1.md) returned `more-work-needed` for
one missing required artifact: the trace-allocation positive rejection control.
Existing numerical, timing and profiler measurements remain applicable. Added
`tests/run_multichip_trace_control.py` and a mandatory acceptance gate checking
its observed exception, exact injected buffer ID, four output replicas, recovery
replay and source/environment provenance. Only the exact capture output is
acknowledged; input/output remain alive through trace release. The injected
post-capture allocation is never acknowledged and is explicitly deallocated
before recovery replay. Corrected the two stale constructor census numbers in
`mesh_plan.md`; the capacity arithmetic was already based on the selected values.

The first control launch incorrectly set `TT_METAL_WATCHER=0`. Runtime source
`tt_metal/llrt/rtoptions.cpp` enables watcher whenever this variable is present,
so this means zero-interval polling, not disabled. It stalled in mesh open;
`py-spy` confirmed `open_mesh_device`, and live tt-triage evidence was captured
before the bounded 180-second timeout ended the process (exit124). Evidence is
under `failures/trace_control_watcher_zero/`. No decoder code ran. After confirming
no device owner, reset12 and device listing were logged in `logs/reset_12.log`.
The corrected launcher unsets watcher and the test now rejects a present watcher
variable before importing TTNN. This is an infrastructure/launcher correction;
no numerical or latency result is drawn from the failed attempt.

### Performance hand-off notes

Round 1 identifies concrete future optimization candidates: batched attention
instead of per-request B32 attention, approximately200 us output splitting,
B32 shared/grouped geometry, and adapting the baseline's tuned prefill programs
to local mesh projection shapes. These are unmeasured candidates, not rejected
alternatives or correctness gaps. No full-model or vLLM work was started.


Reset12 restored all four devices, and `logs/reset_12_smoke.log` records
`MESH_SMOKE_OK` after exact negation on all replicas. Corrected control command:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout 180 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_trace_control \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/multichip_decoder/trace_alloc_tracking/positive_control.json
```

Result: PASS, exit0. `trace_alloc_tracking/positive_control.log` records the
expected RuntimeError for buffer30, and exact recovery outputs on four devices
after deallocation. Output23 is the sole acknowledged allocation. Decoder source
remains unchanged. Updated full CPU evidence check:
`python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_multichip_validation`,
log `logs/final_gate_review2.log`, result `gate_summary.json`.


## Final acceptance and review

`logs/final_gate_review2.log` reports `MULTICHIP_GATES PASS: 27/27`.
[Independent round 2](stage_review_round2.md) returned `clean-pass`, with no
Required Work. The positive-control and stale census findings are closed.
Final staged precommit checks and both working-tree/index whitespace checks pass.
All stage changes are under this model's decoder/tests/docs; no full-model or
vLLM work, external repository edits, or push occurred.

The failed control's recovery changed both device state and the watcher setting;
the evidence therefore does not isolate zero-interval polling as the sole cause
of the original open stall. The watcher setting was demonstrably incorrect,
and the corrected control plus recovery passed. This qualification supersedes
causal wording in the initial follow-up entry.

Optional checker hardening for later stages: generic `off()` accepts watcher
`"0"`, but actual runtime semantics require it unset. All accepted clean timing
and profiler sidecars were checked and have watcher null; the new control gate
already enforces unset. The earlier performance hand-off notes remain useful
future candidates, with no open required stage-4 gate.

## Local checkpoint receipt

- Repository: `/home/ttuser/qwen36-35b-a3b/tt-metal`.
- Branch: `main`.
- Implementation, tests, evidence and passing review commit:
  `679bb405aed1c267f9bb66ab3a72d199b0664ba7` (`Add validated TP4/EP4 Qwen3.6 multichip decoder`).
- Only stage-owned model files were committed; worktree was clean afterward.
- This receipt is a separate documentation commit. Both commit SHAs are recorded
  in local `logs/commit_receipts.log` after that commit, avoiding a self-referential
  commit hash. No push was performed. No other repository was modified.
