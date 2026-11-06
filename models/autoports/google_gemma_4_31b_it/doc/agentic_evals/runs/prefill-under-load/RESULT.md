# Prefill interference result

All six synthetic requests completed on the full60 B8 server: three2048-token decode anchors and three single-output prefill probes. Each anchor remained live through its probe. These host HTTP observations show substantial decoder delivery stalls during longer prefills; they do not measure device time or establish a performance optimization.

| Prefill prompt | Submit-to-first-token window | Anchor baseline tokens/s | Anchor during-prefill tokens/s | Baseline p95/max gap | During-prefill p95/max gap |
|---|---:|---:|---:|---:|---:|
| 4096 | 1.426s | 25.061 | 23.840 | 45.256/50.656ms | 52.261/81.662ms |
| 32768 | 9.517s | 24.674 | 5.359 | 49.363/51.515ms | 1123.067/1343.860ms |
| 86016 | 35.411s | 21.126 | 2.062 | 55.518/62.039ms | 2095.811/2427.044ms |

Source: result.json, including individual stream-event timestamps and boundary-straddling gaps. Planned concurrency was two, with one anchor decoding and one request prefilling, on a fixed captured B8 bucket. Solo baseline still pays B8 padding. The86016-token probe means84Ki tokens, an approximate observed SWE context scale, not replay of a task's exact prompt.

Long-prefill windows include scheduler wait and HTTP delivery through first token. Streaming events can coalesce tokens. Gap percentiles count events rather than weighting by token count; some gaps begin before the window and end inside it, explicitly retained in the raw result. Baselines lasted only about2seconds and were sequential immediately before each probe. Arm order was fixed, with no randomized controls, repeats or separate controlled isolation experiment. The declining third-arm baseline demonstrates time/regime variation; these observations support a forecast risk, not a statistically isolated causal effect size.

The generic implication is to account for decoder stalls while long requests prefill rather than projecting multi-agent throughput solely from isolated decode rate. Real release impact still depends on task request lengths, generation share, overlap and tool waits. No release hours saved or model improvement is claimed; predicted savings remain zero. No release caps, source or numerical settings changed. This is diagnostic traffic, not scored evaluation or headline4096/128/B1 performance.

Client elapsed278.585768seconds fits the300second client prediction. Coordinator queue start08:24:11.191302UTC through measurement_end08:33:21.429436UTC (final result write) gives550.238134seconds (0.152843926device hours), including startup and intervals before the final measurement but excluding subsequent idle polling. This slightly exceeds the540second overall estimate and fits the0.35hour box. Actual timings should replace the initial estimate for equivalent diagnostic work, not for suite durations.

Coordinator launch evidence confirms the full60 model loaded through layer59, exact autoport overrides and pinned revision. No profiler was used. bringup/artifacts/stage12-prefill-under-load-idle.prom records zero running/waiting requests before coordinator graceful shutdown; the coordinator subsequently confirmed server exit0 and clean closure. A shutdown-only EngineDeadError after deliberate SIGINT with zero requests is not a workload failure. No additional hardware work was performed by this effort.

After execution, measure.py received formatting only with the cached repository Ruff tool; Ruff check and Python compilation pass. This formatting does not change the measured algorithm. Final evidence is written once here; result.json remains the measurement source.
