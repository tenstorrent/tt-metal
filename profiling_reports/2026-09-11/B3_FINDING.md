# B3 finding: direct SP8/TP1 MLA measurement

The requested MLA row is implemented and passed locally on an eight-device Blackhole Galaxy column. The test and CI invocation are inherited from the followups branch; this branch adds the findings and evidence.

The measured MLA operation sum is **8.585052 ms**: SDPA **7.195124 ms (83.81%)**, matmuls **0.737823 ms**, other operations **0.652105 ms**. This is one cold, untraced random-weight forward at 51,200 cached tokens plus 5,120 new tokens. It is not elapsed latency, a correctness check, a warmed baseline or LoudBox calibration.

For the separate historical full-layer TP=1 stage-0 capture, Dispatch/Combine account for 35.65% of accumulated work, SDPA 23.44%, all matmuls 17.53% and expert FFN 15.77%. These workloads and metrics differ; do not combine their absolute timings or use the shares as expected throughput gains.

Useful next work is a warmed MLA sweep across KV depths, then correctness and matched performance validation of any SDPA or projection change.

- [B3 draft completion summary](B3_ISSUE_DRAFT.md)
- [Implementation, invocation and validation](B3_IMPLEMENTATION.md)
- [Historical TP=1 ranking and reproduction](B3_TP1_RANKING.md)
- [Direct measurement breakdown](b3-glx-column-PzSANj/DIRECT_BREAKDOWN.md)
- [Validated source hash and row coverage](b3-glx-column-PzSANj/validation.json)

The related profiler issue drafts concern earlier reporting observations. They are not additional B3 test deliverables or confirmed reproductions: [empty trace](ISSUES_PROFILER_EMPTY_TRACE.md), [negative gaps](ISSUE_TT_PERF_NEGATIVE_GAPS.md).
