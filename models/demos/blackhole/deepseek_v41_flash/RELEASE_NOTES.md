# DeepSeek-V4.1-Flash on Blackhole Galaxy: release notes

Release `afb78699c8b-5b6f9268-dd4fbc6` (vLLM stack as in the previous release: the Python environment is reused; versions not re-read from this image), previous release `0.24.0-0655530-4743e01`. Start-up commands: `GETTING_STARTED.md`. Test logs, switches and long-form findings: `RELEASE_DETAILS.md`.

| | |
|---|---|
| Image | `dsv41-flash-local:afb78699c8b-5b6f9268-dd4fbc6` (local build, image id `de6952f6f477`; not on ghcr.io yet; offline copy `/mnt/tt-data/ssinghal/dsv4-docker-ctx/img_afb78699c8b.tar`, 5.38 GB) |
| Commits | tt-metal `afb78699c8b` (branch `ssinghal/dsv4p1-rebase-1017-v3`), tt-inference-server `5b6f9268` (branch `ssinghal/dsv4p1-rebase-1017`), vllm-tt-plugin `dd4fbc6` (branch `ssinghal/dsv4p1-replay-v3`) |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |

<!-- sources: image tag and shas: dsv4-logs/image_v3b/TAG, dsv4-logs/image_v3b/MS (tt-metal), PS (plugin); ttis 5b6f9268 = git rev-parse in wt/ttis_rebase; checkpoint: GETTING_STARTED.md (commit 81284183e99) -->

## What is supported

| | B=16 | B=4 |
|---|---|---|
| Users | 16 | 4 |
| Max context | 262,144 tokens | 1,048,576 tokens |
| Image input | up to 16 images per prompt (1,024 tokens each at most) | same |
| Speculative decoding | up to k = 5 (3 or 5 drafts per round, chosen by measured round time), lossless, at every prompt length | same |
| Runtime spec (`assets/`) | `runtime_spec_b16_k5_262144_vision.json` | `runtime_spec_b4_k5_1048576_vision.json` |
| Start-up | not measured on this image | 20 to 30 minutes (1,217 to 1,787 s on four boots, depending on the host cache) |

<!-- sources: spec file names checked to exist in dsv4-docker-ctx/afb78699c8b-5b6f9268-dd4fbc6/tt-metal-models/demos/blackhole/deepseek_v41_flash/assets/ (identical to the repo assets/); users, context, budget from those JSONs-->

Serving features:
- OpenAI-compatible API (chat, completions, streaming); requests beyond the user count queue.
- On-device sampling: temperature, top-k (clamped to 32), top-p, per-request seed.
- Chunked prefill: token budget `max_num_batched_tokens` in the spec (B=4 / 1M specs: 16,384 with `long-prefill-token-threshold` 4,096 and `DSV41_VLLM_MAX_PARTIAL=4`, so four concurrent prompts advance together; B=16 specs: 67,584, or 4,096 in the `_apc` spec; no admission window).
- Prefix caching (APC): on by default (`DSV41_APC=1` and `enable-prefix-caching` in the specs and the catalog); `--no-enable-prefix-caching` turns it off. Images bypass it.
- Paged prefill: on by default (`DSV41_PF_PAGED=read`, bf16 KV pool, which APC needs); `DSV41_PF_PAGED=0` restores the previous dense-table prefill with the fp8 pool.
- Bounded prefill replay: on by default (`DSV41_PF_REPLAY=2560`): of a cold prompt only the last 2,560 tokens run through all 40 layers; the output is bit-identical (see What changed). `DSV41_PF_REPLAY=0` turns it off.
- Not supported with speculation (HTTP 400): `logprobs`, structured output, `logit_bias`, `bad_words`, `allowed_token_ids`, `min_tokens`, `min_p`.

<!-- sources: specs in assets/*.json; RESULTS_phase3.md (replay); previous RELEASE_NOTES.md for the unchanged bullets -->

## Performance change against the previous release

Short prompts, batched, B=4 (`vllm bench serve`, random prompts, 128 output tokens, 4 requests, greedy, 3 repeats after a warm-up). TTFT ms / TPOT ms / output tok/s. The previous-release column is the "this release" column of the previous notes; ISL 4k was not measured; 1,024 and 8,192 bracket it.

| | Previous release | This release | Change against the previous release |
|---|---|---|---|
| B=4, 4 requests, ISL 1,024 | 1590 / 67.5 / 48.0 | 1649 / 27.2 / 94.4 | +4 % / -60 % / +97 % |
| B=4, 4 requests, ISL 8,192 | 9059 / 120.9 / 20.2 | 5342 / 28.2 / 56.2 | -41 % / -77 % / +178 % |
| B=16, 16 requests, ISL 1,024 | 3626 / 162.8 / 82.1 | not measured | |
| B=16, 16 requests, ISL 8,192 | 26208 / 522.7 / 21.8 | not measured | |

ISL 1,024 TTFT is not better than the previous release (+4 %): a short prompt needs several windows. B=16 was not part of the verification of this image.

<!-- sources: this release: image_tests/afb78699c8b-5b6f9268-dd4fbc6/laneC/bench.log (1649 / 27.2 / 94.4 and 5342 / 28.2 / 56.2, lane C ran on host .34); changes computed from the columns -->

<!-- sources: previous release: previous RELEASE_NOTES.md (same harness: OPTIMIZATIONS.md section 0 names dsv4-logs/image_tests/new_text/bench.log for the same 1590 / 67.5 and 9059 / 120.9 reference); for context: production a58ff4b5c2b image 1559 / 30.9 / 85.9 and 6512 / 33.6 / 44.5 (image_tests/merged_image/bench.log) -->

Decode at long context, 4 users on the 1M build, speculation on (demo runs on the merged tree, 40 layers, 1017 checkpoint, round time of the adaptive scheduler with its time table frozen so that runs are comparable). Round ms / tok/s per user:

| Scenario | This release |
|---|---|
| GSM8K prompts, frozen table | 52.30 ms / 73.92 |
| GSM8K prompts, live adaptive k | 53.53 ms / 76.87 |
| 64k needle prompt | 56.13 ms / 44.66 |
| 64k same-prompt needle | 49.66 ms / 41.15 |
| 128k needle prompt | 57.18 ms / 35.93 |

The live GSM8K row is the tree without the replay change (decode is unaffected by it). Every run reproduces plain greedy decoding for all 4 users. (Replay changes prefill only: decode round times of the tree with and without it agree within 0.2 ms.)

<!-- sources: dsv4-logs/levers/merged/MERGED.md: table "v3 + replay" (round ms / tok/s per user of the v3+replay column: gsm8k 52.3 and 73.92; needle64k 56.13 and 44.66; same_needle64k 49.66 and 41.15; needle128k 57.18 and 35.93) and the gsm8k live row (53.53 ms, 76.87 tok/s) -->

Prompt processing, 4 concurrent cold prompts on the 1M build (server, speculation, images and APC on; same host and server session, replay switched live):

| Prompt length (each of 4 requests) | Replay off | Replay on | Change |
|---|---|---|---|
| 8k | 8.41 s | 6.66 s | -21 % |
| 32k | 32.07 s | 19.65 s | -39 % |
| 64k | 64.24 s | 37.94 s | -41 % |
| 128k | 129.60 s | 75.02 s | -42 % |

Batch TTFT (every user has its first token). When the scheduler starts the first request alone, the first batch is slower; this run is the same case on both sides. A single cold request: 8k 3.99 s against 5.44 s, 16k 7.08 against 10.83, 32k 12.96 against 21.23, 64k 25.13 against 42.83 (-27 to -41 %). Needle prompts of 64k / 131k / 600k tokens: 25.8 / 50.7 / 231.1 s against 44.1 / 89.0 s / 430-445 s (production image).

Verification of this image, same host (.45), fresh server each, 4 concurrent cold prompts, batch TTFT: 8k 4.95 s, 32k 14.47 s, 64k 26.89 s, 128k 51.97 s. Longest decode gap while 3 x 32k prompts prefill: 3.95 s. Replay is active in the server (the per-call prefill statistics carry `replays_a_only`).

Needles on this image: 64k / 128k / 600k / 1M tokens 33 / 55 / 240 / 396 s. Prefix cache, cold / hit: 8k 4.01 / 2.65 s, 16k 7.11 / 4.27 s, 32k 12.96 / 7.18 s, 64k 25.11 / 13.43 s; cold and hit text are equal. Multi-turn, turn 2 against cold: 32k 7.16 / 12.94 s, 64k 13.44 / 25.07 s.

<!-- sources: image_tests/afb78699c8b-5b6f9268-dd4fbc6/results.md (batch TTFT, APC, needle, stall rows); dsv4-logs/replay_study/RESULTS_phase3.md (table "Numbers": run 2 row 6.66 / 19.65 / 37.94 / 75.02 vs 8.41 / 32.07 / 64.24 / 129.60 s = -21 / -39 / -41 / -42 %; good-run row; one cold request row; needle row); image_tests/merged_image/results (production 21.8 / 43.4 / 87.5) via memory production image -->

Speculation share of the adaptive scheduler (GSM8K eval, full 8-shot set, same server settings): the scheduler ran 5-draft rounds in 96.3 % of the rounds on this image (acceptance length 4.20).

<!-- sources: this image: results.md "Server facts" (k_hist {3: 359, 5: 9369}, acceptance length 4.20) -->

Model maximum, one request, 128 output tokens (previous release):

| | Previous release | This release |
|---|---|---|
| B=16, 250,028-token prompt | TTFT 283 s, TPOT 31.6 ms | not measured on this image |
| B=4, 1M-class prompt, 1 user, speculation on | TTFT 1694 s, TPOT 36.7 ms (1,000,028-token prompt) | a 1,000,000-token needle request takes 396 s on this image; benchmark workflow numbers: see the TODO below |

Benchmark workflow report (`--workflow benchmarks`), image `dsv41-flash-local:afb78699c8b-5b6f9268-dd4fbc6`, B=4 / 1M catalog default, speculation on, copied unedited from the report:

### vLLM Benchmark for deepseek-ai/DeepSeek-V4.1-Flash on BLACKHOLE_GALAXY

| Concurrency | Num Requests | ISL     | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:--------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 8            | 128     | 128  | 441.6     | 427.9         | 528.2         | 21.9      | 3218.8    | 39.8             | 39.8              | 79.5             | 0.311          |
| 4           | 32           | 128     | 128  | 524.3     | 476.1         | 896.7         | 34.6      | 4914.1    | 101.5            | 101.5             | 203.1            | 0.793          |
| 1           | 4            | 128     | 1024 | 448.3     | 448.1         | 456.3         | 13.2      | 13904.0   | 9.2              | 73.6              | 82.8             | 0.072          |
| 4           | 16           | 128     | 1024 | 552.7     | 465.5         | 941.0         | 18.8      | 19776.0   | 24.3             | 194.7             | 219.1            | 0.190          |
| 1           | 4            | 1024    | 128  | 1142.7    | 1142.8        | 1145.7        | 26.5      | 4512.8    | 226.9            | 28.4              | 255.2            | 0.222          |
| 4           | 16           | 1024    | 128  | 1364.4    | 1187.1        | 2050.1        | 42.5      | 6762.6    | 592.8            | 74.1              | 666.9            | 0.579          |
| 1           | 4            | 2048    | 128  | 1831.3    | 1831.9        | 1845.8        | 28.6      | 5461.3    | 375.0            | 23.4              | 398.4            | 0.183          |
| 4           | 16           | 2048    | 128  | 1981.4    | 1811.5        | 2493.8        | 55.6      | 9039.5    | 896.9            | 56.1              | 953.0            | 0.438          |
| 1           | 4            | 4096    | 128  | 2759.5    | 2759.6        | 2766.2        | 25.6      | 6012.6    | 681.2            | 21.3              | 702.4            | 0.166          |
| 4           | 16           | 4096    | 128  | 3093.3    | 2739.3        | 5022.2        | 73.3      | 12397.7   | 1295.9           | 40.5              | 1336.4           | 0.316          |
| 1           | 2            | 8192    | 128  | 4299.3    | 4299.3        | 4304.7        | 21.3      | 7008.2    | 1168.8           | 18.3              | 1187.0           | 0.143          |
| 4           | 8            | 8192    | 128  | 4760.4    | 4556.6        | 5394.8        | 75.9      | 14400.4   | 2247.7           | 35.1              | 2282.9           | 0.274          |
| 1           | 2            | 8192    | 1024 | 4302.0    | 4302.0        | 4322.4        | 16.1      | 20759.5   | 394.6            | 49.3              | 443.9            | 0.048          |
| 4           | 8            | 8192    | 1024 | 4761.4    | 4566.8        | 5384.2        | 28.0      | 33456.4   | 833.2            | 104.2             | 937.4            | 0.102          |
| 1           | 2            | 10000   | 1024 | 4782.1    | 4782.1        | 4795.1        | 16.4      | 21546.9   | 464.1            | 47.5              | 511.6            | 0.046          |
| 4           | 8            | 10000   | 1024 | 5329.6    | 5360.8        | 5864.7        | 25.0      | 30925.5   | 1203.7           | 123.3             | 1326.9           | 0.120          |
| 1           | 2            | 16384   | 128  | 7326.9    | 7326.9        | 7331.6        | 23.7      | 10335.8   | 1585.0           | 12.4              | 1597.4           | 0.097          |
| 4           | 8            | 16384   | 128  | 9090.1    | 8825.4        | 14117.3       | 104.9     | 22409.9   | 2873.5           | 22.4              | 2895.9           | 0.175          |
| 1           | 1            | 32768   | 128  | 13532.3   | 13532.3       | 13532.3       | 20.9      | 16191.8   | 2023.6           | 7.9               | 2031.5           | 0.062          |
| 4           | 8            | 32768   | 128  | 17054.4   | 17569.5       | 20653.2       | 148.8     | 35951.6   | 3618.3           | 14.1              | 3632.5           | 0.110          |
| 1           | 1            | 65536   | 128  | 26027.4   | 26027.4       | 26027.4       | 20.5      | 28627.7   | 2289.1           | 4.5               | 2293.6           | 0.035          |
| 4           | 8            | 65536   | 128  | 29926.4   | 30175.6       | 35641.5       | 163.6     | 50704.2   | 5119.4           | 10.0              | 5129.4           | 0.078          |
| 1           | 1            | 131072  | 128  | 50870.9   | 50870.9       | 50870.9       | 24.3      | 53956.2   | 2429.1           | 2.4               | 2431.5           | 0.018          |
| 4           | 8            | 131072  | 128  | 57696.3   | 60276.2       | 61376.6       | 251.3     | 89617.6   | 5808.2           | 5.7               | 5813.8           | 0.044          |
| 1           | 1            | 262016  | 128  | 101689.1  | 101689.1      | 101689.1      | 26.5      | 105058.8  | 2493.8           | 1.2               | 2495.0           | 0.009          |
| 4           | 8            | 262016  | 128  | 108432.2  | 110706.9      | 111005.6      | 104.1     | 121654.1  | 8600.4           | 4.2               | 8604.6           | 0.033          |
| 1           | 1            | 524160  | 128  | 204105.9  | 204105.9      | 204105.9      | 26.2      | 207432.7  | 2526.7           | 0.6               | 2527.4           | 0.005          |
| 4           | 8            | 524160  | 128  | 212120.9  | 214037.8      | 214687.7      | 49.1      | 218353.5  | 9585.5           | 2.3               | 9587.9           | 0.018          |
| 1           | 1            | 1048320 | 128  | 413394.2  | 413394.2      | 413394.2      | 42.4      | 418783.9  | 2503.0           | 0.3               | 2503.3           | 0.002          |
| 4           | 8            | 1048320 | 128  | 430189.2  | 432196.3      | 442500.7      | 132.6     | 447034.6  | 9362.6           | 1.1               | 9363.7           | 0.009          |
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

The concurrency-4 rows with short outputs include the stall of the decoding users while another user's long prompt prefills (see Known issues); no number in the table is adjusted for it.

Benchmark tables, ISL x OSL up to 1,048,576 tokens, B=4 / 1M, speculation on: the report above; how to read the columns and run the workflow: `GETTING_STARTED_DETAILED.md`, section Performance.

## What changed

Since the previous release `0.24.0-0655530-4743e01`:

- Checkpoint `DeepSeek-V4.1-Flash-1017` is the default (the weight cache is keyed per checkpoint, see `GETTING_STARTED.md`).
- Prompt processing: layers 21-39 of a cold prompt run only on its last 2,560 tokens (the checkpoint shares one compressed KV between layers 20-39, so earlier chunks never need them). Output is bit-identical to running all layers on every chunk, at no extra DRAM (6.1 MiB per bank, the same as the single chunk trace). TTFT of 4 concurrent cold prompts is 21 to 42 % lower at 8k to 128k tokens; `DSV41_PF_REPLAY=0` turns it off.
- Decode at long context is faster: the indexer scoring of the speculative rounds is one fused kernel (9 to 12 ms per round less at 64k tokens, exact same selected keys), plus smaller gains from the attention front (1.4 to 1.5 ms per round), the drafter (1.4 ms per round), the collectives (about 0.6 ms) and the mHC blocks (k = 5 verify round: -2.5 ms over an 8-layer window).
- The adaptive speculation scheduler re-measures the other k every 128 rounds (it could stay on k = 3 for good after a slow first measurement): 5-draft rounds in 96.3 % of the rounds of the GSM8K eval on this image.
- Long-context exactness: the ids of the selected index entries are sorted, so a plain step and a verify block read the same keys in the same order (this removed a plain-versus-speculative difference at the 64k position).
- Server prefill: the host side of a prefill step (read-backs, uploads of the next window) is queued behind the device work and a step with no final row returns once its windows are enqueued. Time outside the device in a 4 x 64k prompt prefill: 10.7 s -> 1.35 s at budget 2,048 / 512 (batch TTFT 52.74 -> 42.75 s) and 2.4 s -> 1.22 s at 16,384 / 4,096 (the default). `DSV41_PF_RETURN_LAG=-1` restores the wait.
- Prefix caching and paged prefill are on by default, with one chunk trace for any prompt length and per-row chunk starts; the B=4 / 1M specs use budget 16,384 / threshold 4,096 / `DSV41_VLLM_MAX_PARTIAL=4` and `trace_region_size` 1,200,000,000 (two verify runners, k = 3 and k = 5; about 1.09 GB is used, 906 to 912 MB did not boot).

<!-- sources: host-side prefill: OPTIMIZATIONS.md P17 (10.7 -> 1.35 s, 52.74 -> 42.75 s, 2.4 -> 1.22 s; RETURN_LAG=-1 restores the wait); replay and numbers: RESULTS_phase3.md, OPTIMIZATIONS.md 8.6; indexer: OPTIMIZATIONS.md 8.4 and MERGED.md (needle64k -9 to -10 ms per round expected, -11.96 measured combined); qkpost 8.2 (-1.4 / -1.5 ms), drafter 8.1 (-1.4 ms), lean CCL 8.3 (-0.6 ms estimate), mHC: dsv4-logs/levers/mhc/README.md; probe fix: OPTIMIZATIONS.md 8.5; ids sort: tt/indexer.py canonical_ids docstring; trace region and budgets: previous RELEASE_NOTES.md and specs -->

## Known issues

- B=16 at 262,144 tokens scored about 2 points lower on GSM8K than at 131,072 on the previous release (about 93.5-95 % against 95.2-95.4 %); cause not identified. Not re-measured on this image (B=16 is not part of its verification).
- B=16 with budget 67,584: a long prompt stalls the other users' decoding while it prefills (32k prompt: 32.9 s on the previous release; 9.1 s with budget 4,096). At B=4 with the 16,384 / 4,096 scheduler the longest decode gap during 3 concurrent 32k prompts is 3.95 s on this image.
- At 4 concurrent users with speculation on, the prefill windows of a new long prompt pause the other users' decoding. Cells of the benchmark table with concurrency 4 and a short output therefore show a low rate per user: at ISL 65,536 / OSL 128 the TPOT is 163.6 ms (10.0 output tokens/s over the run), against 20.5 ms at concurrency 1. Cells with long outputs are much less affected (ISL 8,192 / OSL 4,096: TPOT 21.1 ms at concurrency 4, 13.0 ms at concurrency 1).
- At B=16 concurrent long prompts are prefilled one after another (16 prompts of 250,000 tokens: 5,035 s in total on the previous release; not re-measured).
- Requests that cannot speculate (frequency / presence / repetition penalty) lose accuracy on the B=4 1M spec (GSM8K about 52 %, against 97 % with speculation); send them to B=16. Not re-measured on this image.
- APC must not be enabled with the 131,072-token specs (they keep `DSV41_APC=0`, `DSV41_PF_PAGED=0`).
- Prompt replay is exact only for `DSV41_PF_REPLAY` of 2,414 tokens or more (19 layers x 127 + 1); smaller values are clipped and not bit-exact (the default is 2,560). The B=16 / 262,144 build was not measured with replay.
- With replay on, a prefix-cache hit on an 8k prompt is slower than a hit with replay switched off in the same server (3.84 s against 2.91 s in the phase-3 study): partial snapshots above the 5,632-token cutoff are not used. A cold request is faster than before as well, so the hit is no longer 0.6 x the cold time at 8k and 16k (2.65 / 4.01 s and 4.27 / 7.11 s); cold and hit text are equal.

<!-- sources: APC rows: image_tests/afb78699c8b-5b6f9268-dd4fbc6/results.md (Server facts); carried items: previous RELEASE_NOTES.md Known issues (carried unchanged where marked not re-measured; 131,072 specs checked in assets/); replay limits and APC hit numbers: RESULTS_phase3.md "Design notes / limits" and the APC hit row -->

Next release: cheaper APC, host-tiered prefix cache on paged prefill (`PAGED_PREFILL_APC_DESIGN.md`), more in `FUTURE_WORK.md`.
