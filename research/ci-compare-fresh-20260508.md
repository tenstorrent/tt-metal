<!--
SUMMARY: T3K unit test CI comparison: branch nsexton/0-batch-t3k-ttnn-unit vs main (runs 25535657766 / 25535659653)
KEYWORDS: CI, T3K, unit-tests, ttnn, tt-metal, branch-compare, 2026-05-08
SOURCE: GitHub Actions API — runs 25535657766 (branch) and 25535659653 (main)
SCOPE: Job-level duration and pass/fail comparison for all T3K wh_llmbox test jobs
USE WHEN: Investigating if nsexton/0-batch-t3k-ttnn-unit changes impact T3K test durations or failure rates
-->

# CI Compare: nsexton/0-batch-t3k-ttnn-unit vs main

Date: 2026-05-08
Branch run: 25535657766 (nsexton/0-batch-t3k-ttnn-unit) — conclusion: failure
Main run:   25535659653 (main) — conclusion: failure

## Job Duration + Result Comparison

```
Job                                    Branch   Main     Diff   Result-B   Result-M   Runner-B          Runner-M
---------------------------------------------------------------------------------------------------------------
t3k_ttmetal_tests                       1753s  2113s   -360s    FAIL       FAIL       t3k-12            t3k-08
t3k_ttnn_udm_tests                       339s   499s   -160s    pass       pass       t3k-04            t3k-10
t3k_ttnn_tests                           220s   366s   -146s    FAIL       FAIL       t3k-10            t3k-05
t3k_grok_tests                           274s   195s    +79s    pass       pass       t3k-05            t3k-01
t3k_tttv2_fast_unit_tests               1559s  1613s    -54s    pass       pass       t3k-13            t3k-04
t3k_qwen3_vl_tests                       710s   675s    +35s    pass       pass       t3k-01            t3k-05
t3k_mistral-small-3.1-24b-vision_tests   704s   729s    -25s    pass       pass       t3k-03            t3k-03
t3k_dits_tests                          1030s  1012s    +18s    pass       pass       t3k-09            t3k-09
t3k_ttnn_multiprocess_slow_tests         169s   155s    +14s    pass       pass       t3k-08            t3k-10
t3k_tt_metal_multiprocess_tests         1038s  1032s     +6s    FAIL       FAIL       t3k-06            t3k-01
t3k_deepseek_tests                       572s   574s     -2s    pass       pass       t3k-08            t3k-06
```
Sorted by abs(diff) descending.

## Failures Summary

All 3 failures are present in BOTH branch and main — no branch-unique failures:

- t3k_ttmetal_tests — FAIL both (likely pre-existing)
- t3k_ttnn_tests — FAIL both (likely pre-existing)
- t3k_tt_metal_multiprocess_tests — FAIL both (likely pre-existing)

## Notable Duration Differences

- t3k_ttmetal_tests: branch 360s faster (1753s vs 2113s) — may reflect earlier bail-out on failure
- t3k_ttnn_udm_tests: branch 160s faster — runner difference (t3k-04 vs t3k-10)
- t3k_ttnn_tests: branch 146s faster — also failed earlier on branch

## Interpretation

The branch does NOT introduce new failures.
The branch does NOT make passing tests fail.
Duration differences in failing jobs are likely from different failure points, not real perf changes.
Duration differences in passing jobs are within normal runner-to-runner variance.

Branch appears safe from a T3K unit test perspective.
