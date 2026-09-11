# tt-perf-report drops measured kernel durations when operation gaps are negative

Draft; not posted. A fresh row-level reproduction is still needed.

## Observation

During September 10 analysis, `tt-perf-report` 1.3.0 blanked some matmul durations that were present in the input CSV when operation-to-operation gaps were sufficiently negative. This can distort totals and bottleneck rankings for overlapping work. A negative gap alone should not invalidate a separately measured kernel duration.

[PROFILING_REPORT.md](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b3-findings/profiling_reports/2026-09-11/PROFILING_REPORT.md) records the observation. The original `/tmp` capture and tool environment are gone, so the affected rows, exact threshold and source-code cause have not been rechecked. No analyzer source was found in the local paths inspected or downloaded.

## Environment

- Analyzer: `tt-perf-report` 1.3.0.
- Runtime: `akhan/mistral4-prefill-followups` at `b48bf4095de1601786c4cf0e91a647a532c730c5`.
- Hardware and workload: 32-device Blackhole Galaxy, PP4, 36 layers, 5,120-token chunks, traced execution with subdevice overlap and eight profiling requests.
- Input: operation CSVs from Tracy/device profiling. Analysis retained source durations for direct totals where analyzer values were blank.

Expected: retain valid measured kernel durations and report overlap separately. If a measurement is invalid, identify the input and reason for rejecting it.

Subdevice overlap can produce negative gaps, but clock alignment, row ordering and malformed inputs must also be checked. Summed kernel durations are not elapsed request latency.

## Investigation plan, not yet run

1. Capture two overlapping subdevice streams with positive kernel durations. Keep the source CSV and analyzer output.
2. Analyze without device merging, preserving trace mode where needed.
3. Match an affected row by operation, device, trace and session. Compare its source duration, preceding timing, derived gap and output duration.
4. Reduce this to a small accepted CSV fixture. Change only gap or ordering conditions and sweep gaps to find any threshold.
5. Inspect version 1.3.0 to distinguish validation, parsing, ordering and display behavior. Check how missing values affect totals.

## Acceptance

- A retained fixture, tool version and command reproduce the behavior.
- Positive measured durations remain available despite negative gaps, unless a separate validation check justifies rejection and explains it to the user.
- Gap metadata remains separate from duration. Missing, malformed and negative durations remain diagnosable.
- Tests cover non-overlapping rows, overlapping same-device subdevice rows and invalid timing inputs.
- Unmerged totals match direct sums of accepted source durations. Neither is presented as elapsed latency.

No analyzer change or fresh hardware reproduction is included.
