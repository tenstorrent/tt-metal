# Stage Review

Verdict: clean-pass

Independent round 1 review of stage 3, optimized-decoder, for
`ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. Reviewed the live, uncommitted
implementation on `main`, starting HEAD `814ba2ac`. The production file hashes to
`b678803e78b74ede66d61e7609c6ca9147b2977dbba702f8431eb621ef997c4e`.

## Required Work

None. The inspected evidence supports the stage contract without a correctness,
accuracy, context, trace-safety, or final-performance waiver.

## Findings Supporting the Verdict

- The delivered pytest invokes `validate_optimized` and checks the implementation
  name, real-weight mode, diagnostic exclusion and unchanged 0.995 PCC threshold.
  The runner instantiates `OptimizedDecoder` directly; the functional module
  imported by its shared weight/config helper is not an execution fallback.
  `device_only()` rejects torch dispatch and host tensor transfers inside forward.
  Both final profile regions contain zero host operations.
- Independently checked the source identity and all positive PCC fields in 24
  final artifacts, including per-row continuation scores omitted from the summary
  audit's score selection. Every score passes 0.995. The minimum is
  0.9978809357 in `linear_trajectory128.json`; final HF correctness minima are
  0.9980282187 linear and 0.9988375902 full attention. Paired comparisons check
  both outputs and state; the 128-step comparison advances distinct tokens and
  positions without restoring state.
- Changed boundary rules have relevant final-source coverage. The convolution
  history branches at lengths 1/2 versus 3 and above are exercised, as are
  31/32/33, 1023/1024/1025 and 2051. Full-attention continuation consumes the
  unaligned lead through decode before returning to page-aligned prefill. Starts
  32, 33, 1024, 1025 and 2050 pass, and their empty-state controls fail the same
  contribution gate. Normalizing each returned chunk to DRAM addresses the
  recorded mixed-layout concatenation failure. The watcher battery exercises
  these same branches and batch32 on the final source.
- Context remains 262144. Both long-context runners execute complete TT prefill
  at 262139, 262143 and 262144, then validate the final traced token at position
  262143. Inspection of the reference construction confirms complete preceding
  HF recurrent history or K/V history, rather than a truncated-window reference.
  Sampled-window reference coverage is disclosed. Page-aligned state capacity is
  distinct from logical prompt length; the public prefill path has no logical
  chunk-alignment restriction.
- The tracked final runner sets `TT_METAL_TRACE_ALLOC_TRACKING=1` and
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`. It prepares the tested prefill
  variants before capture, interleaves them with 36 replays of the same decode
  trace, changes inputs/positions/page maps, checks exact repeat determinism, and
  separately exercises batch32. The positive-control log reports the deliberately
  live buffer as unsafe. Acknowledgments name only returned tensors. Single-row
  output values are read before another eager call or replay can overwrite them;
  batch outputs belong to one trace, with their Python references retained during
  capture. The optional caller prefill trace is consumed and released before
  decode replay resumes. No broad allocation exemption is present.
- The new GDN state alias is consistent with the kernel lifetime: the receiver's
  `read_vslice` reads its complete disjoint head/V slab, performs a read barrier,
  and publishes it to the state circular buffer; the final-state writer waits for
  compute to publish that slab before writing it back. Scratch and recurrent
  backing allocations are separate. State comparison, tracked replay, watcher and
  advancing-token evidence supplement this source inspection.
- Runtime profiler rows confirm BFP8 projection weights, attention HiFi2 and MLP
  LoFi in both phases and both layer kinds. The recurrent op enforces its separate
  HiFi4/FP32/no-approximation contract in `chunk_gdn_compute_config.hpp`.
  Diagnostic BFP4 profiles actually contain BFP4/LoFi MLP kernels. The down-only
  linear candidate's passing 0.995332 score is retained for stage 8, while the
  analogous full-attention candidate scores 0.994838. The explicit stage-3 BFP8
  default requirement, rather than a synthetic-weight veto, determines shipping
  precision here.
- Matched S1024 medians reproduce the claimed changes: linear prefill
  5.932618 to 3.174400 ms and decode 1.207735 to 0.643936 ms; full prefill
  6.543463 to 3.664886 ms and decode 1.130236 to 0.624672 ms. The best accepted
  earlier linear candidate is 0.643618 ms, only 0.000319 ms faster than the
  final default, within overlapping recorded sample ranges; the same decode
  configuration is retained. The report uses the final measured number.
  Short, boundary and S8192 rows, plus batch32, have matched baseline evidence.
- Rejection evidence includes successful adaptations, not just initial errors:
  legal 2D prefill configurations, explicit minimal-MLP configurations and L1
  placement, coherent native residual grids, fused head splitting, receiver-count
  alternatives and tuned separate attention projections. The 1200-row role sweep
  crosses BFP8 and BFP4/LoFi with reader counts, core counts and larger K blocks.
  Whole-layer integration validates the selected role configurations.
- Independently recomputed padded BFP8 projection storage as 232849408 bytes
  linear and 222822400 bytes full attention. With the recorded 512 GB/s chip
  roofline and state/KV accounting, the final S1024 fractions reproduce 71.90%
  and 70.39%; conservative batch32 fractions are 72.59% and 71.88%. Direct
  batch32 ablations give the last two retained decode gains as 0.80% and 1.20%.
  Stop Rule 2 is supported. The work log explicitly retains untried prefill,
  kernel-fusion, dense-batch and stage-8 precision work as hand-offs.

## Other Concerns

- Preserve the existing `Not tried (stop rule)` hand-offs. In particular, the
  role sweep uses K blocks 1/2/4/8/16; it is not an exhaustive search of every
  potentially legal non-power-of-two down-projection block. No global-optimum
  claim is established or needed after the measured stop rule.
- Final prefill reports still label matmuls `SLOW` and say no `program_config`
  was supplied. Source inspection shows explicit `MinimalMatmulConfig` is
  supplied to every selected minimal-matmul call, including fused SwiGLU.
  The missing generic report field is not evidence of an untuned default.
  Measured adapted configurations and the stop-rule hand-offs bound the remaining
  performance work.
- Local checkpoint commits and their SHA entries remain the stage owner's
  post-review closure action. The run plan authorizes local commits only and
  prohibits stage-agent pushes; this review does not infer push state from reflog.

## Hard-Check Gaps

No blocking gap. The compact host audit does not itself check continuation
`min_row_contribution_pcc` or reconstruct all runner control flow. This review
checked those scores, failing negative controls, warmup/replay ordering and
representative source/profiler correspondence directly. No additional device run
is required by this observation.

## Anomaly Ledger

- Observed anomaly: requested rectangular output shards are replaced by a compact
  computed grid.
  Evidence: final logs and `matmul_device_operation.cpp:299`;
  `native_c64_l0.json` and `native_c64_l3.json`.
  Affected path: decode projection/residual boundaries.
  Control or comparison: native-grid candidates pass but measure 0.69631/0.64491
  ms versus the faster selected family; final profiles expose remaining movement.
  Likely subsystem: matmul output-layout inference.
  Investigation performed: coherent native norm/residual/MLP layout adaptation.
  Resolution: controlled.
- Observed anomaly: one-token final chunks originally produced incompatible
  output memory layouts at concatenation.
  Evidence: `linear_final_tail_layout_error.log.gz`, current prefill boundary,
  final correctness and watcher records at 1025 and 2051.
  Affected path: chunked prefill return assembly.
  Control or comparison: final integrated non-aligned and continuation battery.
  Likely subsystem: sharded/interleaved chunk boundary.
  Investigation performed: normalize chunk outputs to DRAM and rerun affected
  boundary paths.
  Resolution: fixed.
- Observed anomaly: watcher runs emit the generic live-trace allocation warning.
  Evidence: `linear_watcher.log.gz` and `full_watcher.log.gz`.
  Affected path: eager prefill between retained decode replays.
  Control or comparison: matching final-source tracked battery includes program
  cache; the positive control proves the tracker is active.
  Likely subsystem: conservative allocator warning for allocations while a trace
  exists, including forward temporaries.
  Investigation performed: exact output acknowledgment and lifetime inspection;
  tracked replay after eager work with unchanged persistent addresses.
  Resolution: controlled.
- Observed anomaly: profiler utilization can exceed 100%, and the GDN gap remains
  despite already traced decode.
  Evidence: final decode reports, raw durations, `perf_summary.json`.
  Affected path: performance interpretation.
  Control or comparison: explicit physical byte accounting, same-signpost
  host/kernel/gap reconciliation and separate unprofiled grouped timing.
  Likely subsystem: report accounting for multi-reader matmul cores and dispatch.
  Investigation performed: geometry/receiver/layout trials and actual trace-runner
  inspection; no host conversion appears in the measured graph.
  Resolution: controlled; no claim that remaining gaps vanished.
- Observed anomaly: startup logs report optional HF fast-path absence and an
  unknown motherboard mapping.
  Evidence: final raw logs and the CPU HF reference path.
  Affected path: reference execution and UMD discovery metadata.
  Control or comparison: guarded device-only TT forward, serial physical chip 0,
  successful watcher/long-context runs and clean device teardown.
  Likely subsystem: optional CPU-reference extensions and tray-ID identification.
  Investigation performed: separated these messages from runtime fallback or
  device-health failures; no such failure is present in the final runs inspected.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: `review_brief.md`; installed `stage-review`, `optimize`,
  `tt-enable-tracing`, `tt-device-usage` skills; optimize evidence checklist;
  model-bringup startup requirements; checkout `TraceCorrectness.md`.
- Artifact paths: this directory's README/work log, final audit/accuracy/performance
  summaries, 24 final source-bound JSONs, final logs and watcher evidence,
  positive control, pytest/packaging logs, context contract, candidate/geometry
  tables, queue manifests/results, selected rejection artifacts, stage-8 hand-off,
  final and BFP4 profiler reports/provenance, archive manifest. Four final raw
  profiler source hashes and six key compressed log hashes were verified.
- Code paths: `tt/optimized_decoder.py`; validation, pairing, context, pytest,
  reporting and orchestration scripts; relevant candidate helpers and shared
  checkpoint loader; GDN receiver/final-state writer and compute contract.
- Commands run: read-only `git status`/`git diff`, `rg`, `cat`, `sed`, `head`,
  `wc`; small Python standard-library artifact parsing, gzip/hash validation,
  numeric reductions and byte-accounting checks. Only this report was written.
  No TTNN import, hardware command, test, server, reset or implementation edit
  was performed by the reviewer.

## Residual Risk

This is decoder-stage acceptance on the recorded single-chip hardware, with
real checkpoint weights and seeded hidden-state probes. It does not establish
full-model language quality, a final precision policy, dense-batch efficiency,
multichip behavior or serving trace preparation. Long-context correctness samples
output windows while processing the complete history. Those boundaries are
accurately disclosed and do not weaken the stage's existing acceptance contract.
