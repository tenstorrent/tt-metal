# Stage Review

Verdict: clean-pass

Independent review of stage 01, functional decoder for
`google/gemma-4-26B-A4B-it`, completed on 2026-09-25 after the stage owner's
explicit final-evidence notice. Reviewed the live, uncommitted worktree on
`gemma-4-26b-a4b-it`, based on `a3a9fb4229a045ad9361b4e39ad854b491346ea9`.
No hardware commands, servers, implementation edits, or additional reviewer
agents were used. This report is the reviewer's only authored artifact.

## Required Work

- None. The performance metadata and traffic-accounting findings raised during
  this review were corrected and independently checked against fresh captures.
  The local stage checkpoint follows this review, as required by the skill.

## Other Concerns

- This passes the functional-decoder contract. It does not establish full-model
  generation, multi-device execution, serving readiness, or an optimized decoder.
- The documented cost of computing every expert during prefill and converting
  the entire cache pool before decode embedding remains. The reported whole-layer
  times include those operations; no unmeasured speedup is claimed.

## Hard-Check Gaps

- No remaining prerequisite gate is missing. Long-context prefill uses the
  permitted reduced oracle: all TT output rows and K/V positions are computed,
  while 291 HF query rows are compared. This limitation is explicit in the
  README, context evidence and telemetry packet.
- The checkpoint SHA cannot precede this clean-pass. The stage owner must record
  the local stage-owned commit afterward; no push is authorized by this review.

The following results were checked in JSON/logs and against the harness source:

| Contract | Evidence checked | Result |
| --- | --- | --- |
| Real HF configuration and layer kinds | `tests/config.json`, installed HF layer source, `tt/*.py`, `weight_stats.json` | Hidden width 2816; sliding/full head geometries, tied full K/V, seven norms, shared MLP and routed top-8 experts agree. Both real-weight loaders are exercised. |
| Public loading and runtime API | `FunctionalDecoder.from_state_dict`, `prefill_forward`, `decode_forward` | Strict layer keys/shapes; documented paged cache, positions, padding, continuation and trace ownership. |
| Real-weight 4096/128 headline | `watcher_sliding_final.json`, `watcher_full_final.json` | Prefill PCC 0.99852634 / 0.99944062; minimum traced decode PCC 0.99904322 / 0.99977388. Each has all 128 distinct positions 4096 through 4223 and bitwise repeated output equality. |
| Maximum and nonaligned context | `long_sliding_{262144,262143}_final.json`, `long_full_{262144,262143}_fixed.json` | All four prefill/decode cases pass the unchanged 0.995 bar. Lowest full-context prefill PCC is 0.99622397. Context 262144 is retained without reduction. |
| Page ownership, batch and boundaries | `batch32_sliding_exact_qkv_integrated.json`, `batch32_full_identity_gather_integrated.json`, `reuse_*_exact_qkv_integrated.json`, `continuation_*.json`, `cache_indices.json` | Batch 32, changed random page maps, one reused decode trace, lengths 31/32/33/1023/1024/1025/2049/2047, partial-page continuation, unchanged prefix/other slot and integer address boundaries pass. |
| Runtime fallback and tracing | `tests/runtime_audit.py`, forward methods and imported helpers; final guarded runs | TTNN operations in the pass; setup and harness conversions are outside. Replay output is compared. Warmed prefill/capture reject program-cache misses. |
| Watcher 10 | Raw `watcher/{sliding,full}/generated/watcher/watcher.log`, final console/JSON files | 13 / 10 completed dumps, clean detach, no error matches; minimum reported free stack 1328 / 1252 bytes. Both complete headline workloads pass. |
| Synthetic regression and formatting | `synthetic_pytest_final.log`, `pytest_final_tmp/`, `precommit_final.log` | Two real-shape statistical-weight tests pass after the final page-table change; all applicable pre-commit hooks pass. Python/docs-only change requires no C++ build. |
| Reporting | `README.md`, `PERFORMANCE.md`, `COMMANDS.md`, `work_log.md`, `../context_contract.json`, supplied telemetry packet | Commands, accepted versus historical evidence, limitations, values and template IDs are consistent. |

Complete device windows were independently recomputed from the fresh CSVs,
using first firmware start through last firmware end, including internal gaps:

| Layer kind | Prefill device us | Mean traced decode us | Prefill native ops | Decode records | Estimated decode bytes | Useful prefill FLOPs % | Estimated decode DRAM % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding_attention | 4998536.8156 | 9811.0668 | 2501 | 128 x 560 | 1211068608 | 0.1064270 | 24.1091863 |
| full_attention | 5009442.5444 | 10894.3714 | 2473 | 128 x 502 | 1508803108 | 0.1462574 | 27.0495741 |

Every measured native row records a cache hit in the fresh uncached-metadata
exports. Each `decode_example_ops.csv` exactly equals its first replay group.
Both kinds have actual rendered prefill/decode tt-perf-report tables, all-replay
CSV/text, five successful report commands, existing source captures, and matching
CSV SHA256 provenance. Inter-replay input refreshes are excluded from individual
device windows, as documented. The denominators are neither matmul-only time nor
host time.

The useful-work formulas and estimated byte model match the final source.
The theoretical participating-ASIC basis is supported by the official
[P300 specifications](https://docs.tenstorrent.com/aibs/blackhole/p300.html) and
[QuietBox per-ASIC bandwidth](https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html),
plus the installed tt-perf-report 1.3.0 phase-divided Blackhole FLOP model.
This is a common useful-work roofline reference for mixed arithmetic, not a
measurement of FPU occupancy or DRAM controller transactions.

## Anomaly Ledger

- Observed anomaly: cached profiler metadata gave decode operations prefill-sized
  tensors and stale false cache-hit flags.
  Evidence: preserved `tracy/sliding/ops_stale_metadata.csv`;
  `ttnn/api/tools/profiler/op_profiler_serialize.hpp:331` and `:377`;
  `tools/tracy/process_ops_logs.py:477`;
  `binary_ng_program_factory.cpp:409` documents volume-independent hashes.
  Affected path: traffic estimates and per-operation metadata.
  Control or comparison: fresh profiles with `--no-op-info-cache` and the
  runtime cache-miss guard for both kinds.
  Likely subsystem: profiler metadata caching, not decoder computation.
  Investigation performed: inspected serializer/parser and independently
  compared shapes, cache flags, replay counts and source CSV hashes.
  Resolution: fixed for reported evidence; stale traffic estimate withdrawn.

- Observed anomaly: the initial estimator charged whole-pool cache RMW and
  assumed eight sparse experts even for all-expert prefill.
  Evidence: final `tests/summarize_perf.py`; native paged-cache reader/writer
  transfer loops; imported `models/demos/gemma4/tt/experts/prefill.py`.
  Affected path: roofline traffic numerator.
  Control or comparison: independently summed final per-replay operands,
  correcting cache updates to the selected page and sparse weights to actual nnz.
  Likely subsystem: stage reporting code.
  Investigation performed: source/CSV accounting; confirmed full-pool untilize
  operations really occur and remain counted. Prefill DRAM estimate was removed.
  Resolution: fixed; independently reproduced both final decode byte totals.

- Observed anomaly: native full page-table gather corrupted wide tables.
  Evidence: `page_gather_boundary.json`, `long_attention_probe.json`,
  `long_attention_identity_gather.json`, `AUTODEBUG_long_context.md`.
  Affected path: maximum-context full-attention decode.
  Control or comparison: direct table use is exact at the failing widths;
  complete attention control PCC 0.99999976.
  Likely subsystem: native row-major gather above the selector boundary.
  Investigation performed: inspected primitive A/B evidence, current direct-table
  path and both real full-layer maximum/near-maximum reruns.
  Resolution: fixed by a valid identity-gather elimination in the model path.

- Observed anomaly: precision errors changed discrete expert selection;
  FP32 address arithmetic also aliased sufficiently large flattened cache rows.
  Evidence: `AUTOFIX_decode.md`, `HF_PRECISION_CONTROLS.md`,
  `cache_write_integrated.json`, `cache_indices.json`, local precision helpers.
  Affected path: decode attention/router and cache addressing.
  Control or comparison: real-weight integrated headline, batch-32, reuse,
  continuation and long-context results; integer row-address equality.
  Likely subsystem: arithmetic/storage boundaries and address dtype.
  Investigation performed: checked retained SFPU/QKV/norm/rotary/router path,
  explicit BF16 cache writes, uint32 shift/add addresses and unchanged PCC bar.
  Resolution: fixed in the inspected model implementation and tested cases.

- Observed anomaly: profiler close aborted with support count 150000.
  Evidence: `AUTOFIX_profiler.md`, preserved first profile log,
  `tt_metal/impl/profiler/profiler.cpp:2392`.
  Affected path: profiler host-buffer allocation/export.
  Control or comparison: support count 100000 avoids uint32 allocation overflow
  while retaining the complete 128-step workload.
  Likely subsystem: native profiler allocation.
  Investigation performed: source arithmetic and successful final capture/export
  logs and complete device records inspected.
  Resolution: controlled with a command-level workaround; no native fix claimed.

- Observed anomaly: reuse diagnostics warn about post-capture allocations.
  Evidence: reuse console logs, `RUNTIME_AUDIT.md`,
  `tt_metal/impl/allocator/allocator.cpp:119` and allocation tracker source.
  Affected path: harness buffer lifetime between prefill and replay.
  Control or comparison: request temporaries are released; stable decode
  tensors/cache predate capture; nine changing requests and repeated outputs pass.
  Likely subsystem: generic allocator warning for any new allocation with a trace.
  Investigation performed: checked lifetimes, the unused sliding tail and warning
  condition. Final ordinary headline runs supersede diagnostic instrumentation.
  Resolution: controlled for the documented tested lifetime pattern.

## Scope Inspected

- Goal/skill paths: saved `01-01-functional-decoder.prompt.txt`; supplied stage
  contract; `.agents/skills/{stage-review,functional-decoder,tt-device-usage}/SKILL.md`;
  Tenstorrent review core/router, model bringup and trace review skills.
- Artifact paths: autoport `doc/functional_decoder/` files named above;
  `doc/context_contract.json`; local raw profiler/watcher evidence;
  `bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/c7de8060-b30e-40b0-ab7e-a6ae3c7cc666.json`
  and its supplied template.
- Code paths: all five `tt/*.py` files; main correctness, batch, reuse,
  continuation, long-context, synthetic, runtime-audit and perf runners; relevant
  diagnostic runners; imported Gemma4 attention, weight, MoE, expert, shared MLP
  and norm helpers; installed HF Gemma4 decoder/attention; native profiler,
  paged-cache, gather and allocator sources.
- Commands run: read-only `git status/branch/diff`, `find`, `cat`, `sed`, `grep`,
  and small Python JSON/CSV/hash/log analyses; opened two cited official
  hardware specification pages. `rg` was unavailable, so searches used the
  available alternatives. No device tests or build were run by this reviewer.

## Residual Risk

- Long prefill correctness is sampled; batch 32 is validated at short context,
  not every batch/context combination. Real weights are represented by layers
  0 and 5, with synthetic activations rather than full-model activation traces.
- Cache pages must be BF16 with 32 tokens; positions must be valid active lanes.
  Caller ownership and trace lifetimes remain explicit API requirements.
- Traffic is estimated with stated exclusions. Performance is a single-layer,
  single-ASIC device result, not full-model throughput or serving latency.
- Large captures and full tables remain local and ignored, with compact reports
  and provenance selected for the checkpoint. This review covers the live state
  and leaves the post-pass local commit to the stage owner.
