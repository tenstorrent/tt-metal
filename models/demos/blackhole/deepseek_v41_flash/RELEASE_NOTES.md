# DeepSeek-V4.1-Flash on Blackhole Galaxy (B=4): release notes

Release `a06adeba8f1-91da919e-dd4fbc6`, previous release `0.24.0-0655530-4743e01`. Start-up commands: `GETTING_STARTED.md`.

| | |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-a06adeb-91da919` (image id `68748a55fba8`) |
| Commits | tt-metal `a06adeba8f1` (branch `ssinghal/dsv4p1-rebase-1017-v3d`), tt-inference-server `91da919eb` (branch `ssinghal/dsv4p1-rebase-1017-v3d`), vllm-tt-plugin `dd4fbc6` (branch `ssinghal/dsv4p1-replay-v3`) |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |

## Features (B=4)

| | |
|---|---|
| Users | 4 |
| Max context | 1,048,576 tokens |
| Speculative decoding | up to k = 5 (3 or 5 drafts per round, chosen by measured round time), lossless, at every prompt length |
| Runtime spec (`assets/`) | `runtime_spec_b4_k5_1048576_vision.json` |
| Start-up | 20 to 30 minutes (1,217 to 1,787 s on four boots, depending on the host cache) |

- OpenAI-compatible API (chat, completions, streaming); requests beyond the user count queue.
- On-device sampling: temperature, top-k (clamped to 32), top-p, per-request seed.
- Chunked prefill: token budget 16,384 with `long-prefill-token-threshold` 4,096 and `DSV41_VLLM_MAX_PARTIAL=4`, so four concurrent prompts advance together.
- Prefix caching (APC): on by default (`DSV41_APC=1` and `enable-prefix-caching`); `--no-enable-prefix-caching` turns it off. Images bypass it.
- Paged prefill: on by default (`DSV41_PF_PAGED=read`, bf16 KV pool, which APC needs); `DSV41_PF_PAGED=0` restores the previous dense-table prefill with the fp8 pool.
- Bounded prefill replay: on by default (`DSV41_PF_REPLAY=2560`): of a cold prompt only the last 2,560 tokens run through all 40 layers; the output is bit-identical. `DSV41_PF_REPLAY=0` turns it off.
- Not supported with speculation (HTTP 400): `logprobs`, structured output, `logit_bias`, `bad_words`, `allowed_token_ids`, `min_tokens`, `min_p`.

## Perf (B=4)

Benchmark workflow report (`--workflow benchmarks`), B=4 / 1M catalog default, speculation on. Measured on the image `0.24.0-afb7869-5b6f926`; this release differs from it only in the image limit (16 to 250 images per prompt) in the specs and the catalog. The rows shown are unedited from the report. tok/s/user is 1000 / TPOT (ms).

### vLLM Benchmark for deepseek-ai/DeepSeek-V4.1-Flash on BLACKHOLE_GALAXY

| Concurrency | Num Requests | ISL     | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:--------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 4            | 128     | 1024 | 448.3     | 448.1         | 456.3         | 13.2      | 13904.0   | 9.2              | 73.6              | 82.8             | 0.072          |
| 4           | 16           | 128     | 1024 | 552.7     | 465.5         | 941.0         | 18.8      | 19776.0   | 24.3             | 194.7             | 219.1            | 0.190          |
| 1           | 2            | 8192    | 1024 | 4302.0    | 4302.0        | 4322.4        | 16.1      | 20759.5   | 394.6            | 49.3              | 443.9            | 0.048          |
| 4           | 8            | 8192    | 1024 | 4761.4    | 4566.8        | 5384.2        | 28.0      | 33456.4   | 833.2            | 104.2             | 937.4            | 0.102          |
| 1           | 2            | 10000   | 1024 | 4782.1    | 4782.1        | 4795.1        | 16.4      | 21546.9   | 464.1            | 47.5              | 511.6            | 0.046          |
| 4           | 8            | 10000   | 1024 | 5329.6    | 5360.8        | 5864.7        | 25.0      | 30925.5   | 1203.7           | 123.3             | 1326.9           | 0.120          |
| 1           | 4            | 1024    | 1024 | 1123.9    | 1123.8        | 1128.8        | 14.9      | 16391.0   | 62.5             | 62.5              | 124.9            | 0.061          |
| 4           | 16           | 1024    | 1024 | 1255.9    | 1131.8        | 1645.1        | 22.4      | 24204.8   | 155.7            | 155.7             | 311.4            | 0.152          |
| 1           | 1            | 32768   | 1024 | 13499.7   | 13499.7       | 13499.7       | 13.8      | 27653.1   | 1184.9           | 37.0              | 1221.9           | 0.036          |
| 4           | 8            | 32768   | 1024 | 14755.9   | 13695.5       | 16957.8       | 47.9      | 63796.9   | 1852.8           | 57.9              | 1910.7           | 0.057          |
| 1           | 1            | 131072  | 1024 | 50824.9   | 50824.9       | 50824.9       | 13.1      | 64260.2   | 2039.7           | 15.9              | 2055.6           | 0.016          |
| 4           | 8            | 131072  | 1024 | 54389.1   | 51473.9       | 59398.2       | 94.0      | 150509.3  | 3458.5           | 27.0              | 3485.5           | 0.026          |
| 1           | 1            | 262144  | 1024 | 101970.1  | 101970.1      | 101970.1      | 20.7      | 123132.1  | 2128.9           | 8.3               | 2137.2           | 0.008          |
| 4           | 8            | 262144  | 1024 | 110730.6  | 111646.3      | 118067.4      | 73.0      | 185366.1  | 5599.0           | 21.9              | 5620.9           | 0.021          |
| 1           | 1            | 524288  | 1024 | 205105.1  | 205105.1      | 205105.1      | 17.2      | 222697.8  | 2354.2           | 4.6               | 2358.8           | 0.004          |
| 4           | 8            | 524288  | 1024 | 224200.8  | 218956.8      | 243989.5      | 128.9     | 356060.3  | 5800.4           | 11.3              | 5811.7           | 0.011          |
| 1           | 1            | 1047296 | 1024 | 412319.4  | 412319.4      | 412319.4      | 35.6      | 448701.9  | 2334.0           | 2.3               | 2336.3           | 0.002          |
| 4           | 8            | 1047296 | 1024 | 563523.3  | 516445.3      | 788947.0      | 287.6     | 857720.5  | 4872.6           | 4.8               | 4877.3           | 0.005          |
| 1           | 1            | 128     | 4096 | 447.8     | 447.8         | 447.8         | 11.8      | 48956.9   | 2.6              | 83.7              | 86.3             | 0.020          |
| 4           | 8            | 128     | 4096 | 614.1     | 460.0         | 878.6         | 13.7      | 56569.1   | 7.9              | 253.0             | 260.9            | 0.062          |
| 1           | 1            | 1024    | 4096 | 1135.0    | 1135.0        | 1135.0        | 24.7      | 102341.0  | 10.0             | 40.0              | 50.0             | 0.010          |
| 4           | 8            | 1024    | 4096 | 1385.6    | 1386.3        | 1644.0        | 15.4      | 64347.7   | 61.0             | 244.1             | 305.1            | 0.060          |
| 1           | 1            | 8192    | 4096 | 4287.7    | 4287.7        | 4287.7        | 13.0      | 57494.1   | 142.5            | 71.2              | 213.7            | 0.017          |
| 4           | 8            | 8192    | 4096 | 4618.9    | 4408.1        | 5097.3        | 21.1      | 91058.7   | 318.7            | 159.4             | 478.1            | 0.039          |
| 1           | 1            | 32768   | 4096 | 13483.4   | 13483.4       | 13483.4       | 11.1      | 58979.4   | 555.6            | 69.4              | 625.0            | 0.017          |
| 4           | 8            | 32768   | 4096 | 14753.6   | 13712.0       | 16958.1       | 19.6      | 94846.6   | 1278.7           | 159.8             | 1438.5           | 0.039          |
| 1           | 1            | 131072  | 4096 | 50842.6   | 50842.6       | 50842.6       | 16.1      | 116666.9  | 1123.4           | 35.1              | 1158.6           | 0.009          |
| 4           | 8            | 131072  | 4096 | 55835.6   | 56868.1       | 60293.1       | 29.4      | 176072.0  | 2942.5           | 92.0              | 3034.4           | 0.022          |
| 1           | 1            | 262144  | 4096 | 101349.0  | 101349.0      | 101349.0      | 14.6      | 160970.9  | 1628.5           | 25.4              | 1653.9           | 0.006          |
| 4           | 8            | 262144  | 4096 | 106010.5  | 102941.3      | 111669.5      | 52.4      | 320399.5  | 3207.0           | 50.1              | 3257.1           | 0.012          |
| 1           | 1            | 524288  | 4096 | 204410.7  | 204410.7      | 204410.7      | 19.5      | 284355.2  | 1843.7           | 14.4              | 1858.1           | 0.004          |
| 4           | 8            | 524288  | 4096 | 249969.6  | 218775.0      | 321500.9      | 66.8      | 523507.7  | 3972.1           | 31.0              | 4003.1           | 0.008          |
| 1           | 1            | 1044224 | 4096 | 411024.2  | 411024.2      | 411024.2      | 32.9      | 545626.2  | 1913.7           | 7.5               | 1921.3           | 0.002          |
| 4           | 8            | 1044224 | 4096 | 486801.5  | 434486.0      | 592257.3      | 86.5      | 840957.7  | 4903.8           | 19.2              | 4923.0           | 0.005          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.

At 4 concurrent users the prefill windows of a new long prompt pause the other users' decoding, so concurrency-4 rows have a higher TPOT than concurrency 1, most of all with long prompts and short outputs: ISL 131,072 / OSL 1,024 has a TPOT of 94.0 ms at concurrency 4 against 13.1 ms at concurrency 1, and ISL 1,047,296 / OSL 1,024 has 287.6 ms against 35.6 ms. A longer output spreads the same pause over more tokens (ISL 131,072 / OSL 4,096: 29.4 ms against 16.1 ms). No number in the table is adjusted for it.
