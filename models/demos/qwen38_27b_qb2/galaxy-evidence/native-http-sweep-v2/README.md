# Native physical-Galaxy HTTP sweep, completed Oct 9 2026 UTC

All nine admitted cells finished at 06:48:56 UTC on the physical 32-chip Galaxy,
using eight independent TP4 vLLM engines. Each cell has one warmup and three
measured bursts with fresh full prompts, fixed 128-token outputs and no prefix
reuse. Three further cells exceeded the configured 1,050,592-token per-replica
KV pool and remain untested. Near-256K means 262,016 input tokens, leaving room
for the output; it does not mean generating 256K output tokens.

| Input | Global concurrency | Median client stream tok/s/user | Aggregate end-to-end output tok/s | p50 / p90 TTFT (s) |
|---|---:|---:|---:|---:|
| 16K | 32 | 17.057 | 226.35 | 10.51 / 10.58 |
| 16K | 64 | 16.786 | 290.41 | 20.40 / 20.56 |
| 16K | 128 | 11.334 | 314.31 | 40.58 / 40.87 |
| 32K | 32 | 16.662 | 141.32 | 21.12 / 21.27 |
| 32K | 64 | 16.083 | 163.27 | 41.85 / 42.08 |
| 32K | 128 | 2.705 | 172.26 | 47.47 / 82.49 |
| 128K | 32 | 3.378 | 33.25 | 84.67 / 113.43 |
| 128K | 64 | 1.348 | 34.65 | 139.74 / 222.84 |
| 262,016 | 32 | 1.105 | 12.75 | 191.36 / 306.23 |

These are fresh-prefill HTTP bursts, not isolated steady-state decode rates.
The stream rate includes interruptions after first token, and the aggregate
rate includes queueing, prefill, decode and transport. Do not compare the
aggregate column directly with an x8 projection of native decode-only tests.
Three bursts are insufficient for a general production tail-latency claim.

Repeated prefill pauses remain a source-supported scheduling lead. For example,
32K/C128 warmup and all three measured bursts each took about 95 seconds, so
the slow rate is not explained by one cold start. The current source disables
scheduler chunked prefill, although internal chunks bound activation memory.
The [queued continuation test](../chunked-prefill-hardware-v1/README.md) is a
correctness prerequisite for changing that policy, with no measured uplift yet.

The surrounding full GPQA still scores 170/198 (85.86%), below the unchanged
177/198 release gate. Completing this sweep is not a reference-eval pass.

[Interactive graph](index.html), [PNG](sweep.png), [PDF](sweep.pdf),
[CSV](sweep.csv), [raw results](sweep.json.gz).
