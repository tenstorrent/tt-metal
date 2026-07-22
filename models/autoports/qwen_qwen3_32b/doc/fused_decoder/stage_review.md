# Independent Stage 02 review

## First review

Reviewer: `/root/stage_review_qwen3_32b_fused`

Verdict: `more-work-needed`.

Required findings were:

1. Prefill-length and decode-position view dictionaries retained every DRAM
   slice indefinitely, contradicting context preservation.
2. The direct post-concat slice/reshape candidate was no longer represented by
   the checked-in candidate runner after the final implementation changed.
3. The fused 4,096-token pass had no retained raw runner artifact.

The reviewer otherwise reconciled correctness, the required internal cache
reshard, profiler totals, watcher cleanliness, scope, and the graph-pattern
audit.

## Remediation

- Replaced both dictionaries with single-entry caches and explicit old-tensor
  deallocation.
- Added a device allocator stress over repeated prefill tile classes and all
  decode positions 0-127. Repeated allocations remain exactly flat at
  976,015,360 and 980,276,736 bytes.
- Added `DirectConcatViewDecoder` to the checked-in candidate runner and reran
  final versus direct over 500 trace replays: 81.767446 versus 81.820552 ms,
  with identical 0.998694 PCC.
- Retained the post-fix capacity output in `logs/context_4096.log`.
- Reran the final suite, 11/500 before-after performance gate, 4,096 capacity
  probe, and watcher correctness run.

## Fresh rereview

Reviewer: `/root/stage_rereview_qwen3_32b_fused`

The fresh reviewer independently checked the remediated source, tests, JSON,
raw/reduced profiler CSVs, final performance, context log, suite log, watcher
logs, documentation, context contract, and repository scope.

Required work: none.

Residual risk: known nanobind shutdown leak diagnostics appear after clean
process/device closure; later watcher evidence completes and detaches both
visible devices cleanly.

Verdict: `clean-pass`.
