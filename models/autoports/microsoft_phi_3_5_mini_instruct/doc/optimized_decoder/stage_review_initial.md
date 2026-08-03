# Initial independent stage review

Verdict: **more work needed**.

The fresh xhigh reviewer identified seven evidence or implementation gaps:

1. Functional and optimized performance used different logical positions.
2. Some topology and precision/geometry claims were stale or not backed by
   inspectable runner output.
3. The batch-32 prefill block-width recommendation was not addressed.
4. The head-width-96 RoPE composite remained unadapted.
5. Decode output-subblock advice was not explained.
6. Long-context tests were mostly zero-state and lacked exact limit,
   page-boundary, and multi-user coverage.
7. Watcher used level 1 and no explicit byte roofline was recorded.

All findings were treated as stage blockers. Their resolutions and artifact
mapping are in `work_log.md`; a fresh independent rereview is required before
commit.
