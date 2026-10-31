<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-XS-2.1 p150x4 latency sweep (2026-09-14)

Result: **PASS, 9/9 requests**. Every row completed with exactly 512 output tokens at concurrency 1.
No request failed or aborted.

This is the first p150x4 sweep in this checkout. The previously cited D4 raw sweep
(`doc/vllm_integration/sweep_vllm.tsv`) was removed on 2026-08-07, which left the four-ASIC numbers
in circulation unbacked. This run replaces them with measured data.

Hardware: four Blackhole ASICs in a 1x4 mesh on one TT-QuietBox 2 (both internal P300c cards). Served
from the published container package `tt-hous/laguna-xs-2.1` at 131,072-token context. The `p150x4`
profile disables prefix caching (`--no-enable-prefix-caching`), so every request is cold by
construction; no cache salt is needed.

| Requested ISL | Actual prompt | OSL | C | TTFT | TPOT | E2EL | Decode tok/s/user | Aggregate output tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 82 | 512 | 1 | 0.129 s | 34.49 ms | 17.754 s | 28.99 | 28.838 |
| 1,024 | 1,066 | 512 | 1 | 1.232 s | 34.73 ms | 18.980 s | 28.79 | 26.975 |
| 2,048 | 1,939 | 512 | 1 | 1.408 s | 34.80 ms | 19.189 s | 28.74 | 26.681 |
| 4,096 | 4,138 | 512 | 1 | 4.975 s | 34.99 ms | 22.853 s | 28.58 | 22.403 |
| 8,192 | 8,234 | 512 | 1 | 10.896 s | 35.33 ms | 28.948 s | 28.31 | 17.686 |
| 16,384 | 16,426 | 512 | 1 | 25.643 s | 36.02 ms | 44.049 s | 27.76 | 11.623 |
| 32,768 | 32,810 | 512 | 1 | 66.650 s | 37.39 ms | 85.757 s | 26.74 | 5.970 |
| 65,536 | 65,578 | 512 | 1 | 194.450 s | 40.17 ms | 214.977 s | 24.89 | 2.382 |
| 130,048 | 130,090 | 512 | 1 | 206.156 s | 45.65 ms | 229.481 s | 21.91 | 2.231 |

## Findings

**Decode scales with chip count.** p150x4 sustains 28.99 tok/s/user at short context against 19.97 on
p150x2, and 21.91 against 18.15 at 130K. Four ASICs buy roughly 45% more decode throughput per user at
short context, narrowing to 21% at full context.

**These numbers reproduce the pre-2026-08-07 D4 figures.** The circulated-but-unbacked values (29.0 at
1K, 27.9 at 16K, 26.9 at 32K, 22.0 at 128K) match this run within measurement noise (28.79, 27.76,
26.74, 21.91). The old sweep was accurate; only its raw data was lost.

**p150x4 still pays the power-of-two prefill padding cliff.** TTFT at 65,536 is 194.450 s, and at
130,048 it is 206.156 s — 6% more time for twice the tokens. The 65K request is padded to
approximately the same total rows as the 130K request. Streaming prefill removed this on p150x2, but
`serve_vllm.sh config` reports `streaming_prefill_status=topology_inactive` for p150x4, so the profile
does not benefit.

**Consequence: p150x4 is slower than p150x2 at 65K.** TTFT 194.450 s against 156.630 s, despite twice
the ASICs. Below 32K and at full 130K context, p150x4 wins on every metric. Enabling streaming prefill
for the D4 topology is the obvious next optimization.

## Method

- `vllm bench serve`, OpenAI chat endpoint, `--dataset-name random`, OSL 512, `--max-concurrency 1`,
  `--request-rate inf`, `--temperature 0`, `--ignore-eos`, `--seed 1234`, one prompt per point.
- `--num-warmups 0` and `--ready-check-timeout-sec 0` disabled benchmark-generated probes and warmups.
- Each point is one sample, not a variance study.
- Served via `tt-model serve tt-hous/laguna-xs-2.1 --profile p150x4`; server reported ready in 11m13s
  with `GPU KV cache size: 131,584 tokens`.
- `Decode tok/s/user = 1000 / mean_tpot_ms`. `E2EL = TTFT + TPOT x (OSL - 1)`; this vLLM build reports
  `mean_e2el_ms` as null for single-prompt runs, so E2EL is derived and cross-checked against
  `1 / request_throughput`.
- Requested ISL and actual server-counted prompt tokens both appear above. Random token IDs are decoded
  to text, wrapped in the chat template, and tokenized again, so the two need not match.

Full-precision values are in [`p150x4_latency_sweep_20260914.tsv`](p150x4_latency_sweep_20260914.tsv).
The p150x2 comparison is [`p150x2_latency_sweep_20260822.md`](p150x2_latency_sweep_20260822.md).
