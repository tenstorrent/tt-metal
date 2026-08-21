# Bring-up cost: wall clock and tokens per stage

Extracted 2026-08-21 from `~/_fmf-qwen-logs/stage*/`, using the same method as
`run-cost-analysis/extract_run_costs.py` so these numbers are comparable to the fleet table:
start = first `ts` in the stage jsonl, end = last `ts`, tokens = the **last**
`thread/tokenUsage/updated` event, which carries the cumulative total for that thread.
Reproduce with `tests/extract_stage_times.py`.

## Stages that produced the result

| stage | start (UTC) | end (UTC) | hours | total tokens | output tokens | log |
|---|---|---|---:|---:|---:|---:|
| 04 multichip-decoder | 08-13 13:10:54 | 08-13 14:52:58 | **1.70** | 80,822,054 | 120,300 | 53.6 MB |
| 05 optimized-multichip-decoder | 08-13 14:53:56 | 08-13 16:25:18 | **1.52** | 57,490,464 | 87,220 | 27.3 MB |
| 06 full-model | 08-13 22:19:10 | 08-14 06:34:23 | **8.25** | 1,385,744 | 8,061 | 1.4 MB |
| 07 optimized-full-model | 08-14 06:35:04 | 08-14 08:37:47 | **2.05** | 47,598,289 | 45,859 | 13.7 MB |
| 08 datatype-sweep | 08-14 08:40:07 | 08-14 12:23:46 | **3.73** | 74,244,560 | 59,945 | 28.2 MB |
| 09 vllm | 08-14 21:16:50 | 08-15 04:13:07 | **6.94** | 165,690,691 | 96,375 | 21.9 MB |
| 10 optimized-vllm | 08-15 04:17:18 | 08-15 07:12:39 | **2.92** | 47,758,701 | 50,634 | 10.6 MB |
| 11 tti-release | 08-15 07:13:32 | 08-15 13:31:00 | **6.29** | 149,317,622 | 86,614 | 22.5 MB |
| **total** | | | **33.40** | **624,308,125** | **555,008** | |

## Abandoned threads — 28% of the wall clock, 49% of the tokens

Two stages were restarted on a fresh thread after the original re-derived a wrong conclusion
repeatedly. That work is preserved separately and is **not** in the table above:

| abandoned thread | start | hours | total tokens |
|---|---|---:|---:|
| `_stage6-blocked-thread` (first) | 08-13 16:28:05 | 4.27 | 183,429,778 |
| `_stage6-blocked-thread` (second) | 08-13 22:05:04 | 0.19 | 190,172,782 |
| `_stage9-blocked-thread` | 08-14 12:25:52 | 8.80 | 232,647,353 |
| **total discarded** | | **13.26** | **606,249,913** |

So the real cost of reaching stage 11 was **46.7 thread-hours and 1.23 billion tokens**, of
which **13.3 h (28%) and 606 M tokens (49%) went to threads whose output was thrown away.**

The timeline is continuous once the abandoned threads are placed: stage 05 ends 16:25:18 and the
first abandoned stage-6 thread starts 16:28:05; stage 08 ends 12:23:46 and the abandoned stage-9
thread starts 12:25:52. Total elapsed from stage 04 start to stage 11 end is **48.34 h**, so only
~1.7 h was genuinely idle.

## Two anomalies worth reading, not smoothing over

**Stage 06 spent 8.25 h on 1.39 M tokens** — two orders of magnitude below every other stage,
with a 1.4 MB log against 10–55 MB elsewhere. That is the *replacement* thread: after the
original was abandoned, the fresh one did very little talking (8,061 output tokens) and spent its
time waiting on long device runs. So stage 06's apparent cheapness is an artifact of where the
thinking happened — the 373 M discarded tokens in its two abandoned threads are the real figure.

**Stage 09 is the most expensive single stage at 15.74 h** when its abandoned thread is included
(8.80 + 6.94), against 6.94 h as logged. It is also the stage whose blocked thread burned the
most tokens of any thread in the run, 232 M.

Both are the same failure mode, recorded in the session notes at the time: resuming a poisoned
thread let it re-derive the same wrong conclusion instead of re-examining the evidence, and the
fix was a fresh thread at the same stage index.

## What these numbers do and do not measure

- **Do**: wall clock and token cost of the agent threads driving each stage, on the same basis as
  the fleet cost table.
- **Do not**: pure device time. A stage's hours include agent reasoning, tool calls, compilation,
  and long device runs indistinguishably. Stage 06 is the clearest case — 8.25 h of mostly
  waiting.
- **Do not** cover stages 01–03. This port began from the `advchal-v3/nofuse-noadvise`
  optimized-decoder output, so functional, fused and optimized-decoder work happened in an earlier
  run and is not in this log directory.
- The post-pipeline work in this directory — the eval re-runs, the CI-faithful release run, the
  benchmark sweep, the isolation experiments and the spec tests — is separate again, and is
  roughly 20 h of device time on top, itemised in the individual documents.

## Fleet comparison

`~/run-cost-analysis/run_times_and_cost.md` tabulates 71 stage runs across the other models from
the July corpus. **Qwen3.6-27B is not in it** — that corpus predates this run — so these figures
should be appended there if a cross-model comparison is wanted. The 33.4 h / 624 M figure is for
stages 04–11 only and is not comparable to a full 01–11 run without adding the earlier stages.
