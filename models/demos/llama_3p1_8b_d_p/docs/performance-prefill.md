<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B-Instruct: prefill performance

**32K execution and performance passed on 17 September 2026.** The two slots reached about **1,076 and 1,072 tokens/s/user**. These are warmed eager host-wall measurements. Saved 2K, 4K, 8K and 16K results are retained without new runs.

Each chunk runs embedding, all 32 decoder layers, final RMSNorm and the full vocabulary head. Configuration: SP4/TP8, two independent slots, BF16 weights/activations, BFP8_B cache, 1,024-token chunks, no tracing. Capacity equals the tested prompt length.

## Measured requests

One warmup ran per slot. Then three measured requests ran per slot, alternating slots. Each request completed all its chunks before the next slot began. The users did not run concurrently. Entries below show median (minimum–maximum) across three requests.

| Context | Slot | Prompt wall, ms | Sum of chunk forward wall, ms | Tokens/s/user |
|---|---:|---:|---:|---:|
| 2K saved | 0 | 432.562 (427.171–434.197) | 430.608 (425.203–432.149) | 4,734.58 (4,716.75–4,794.34) |
| 2K saved | 1 | 431.713 (429.613–434.998) | 429.661 (427.595–430.864) | 4,743.89 (4,708.07–4,767.08) |
| 4K books | 0 | 1,058.649 (1,056.521–1,059.395) | 1,054.325 (1,052.627–1,055.159) | 3,869.08 (3,866.36–3,876.87) |
| 4K books | 1 | 1,062.032 (1,057.305–1,066.372) | 1,057.849 (1,053.245–1,061.879) | 3,856.76 (3,841.06–3,874.00) |
| 8K books | 0 | 2,870.726 (2,865.775–2,892.967) | 2,862.449 (2,855.382–2,884.643) | 2,853.63 (2,831.69–2,858.56) |
| 8K books | 1 | 2,882.602 (2,882.293–2,883.411) | 2,874.283 (2,874.074–2,875.482) | 2,841.88 (2,841.08–2,842.18) |
| 16K books | 0 | 8,747.287 (8,734.832–8,778.168) | 8,727.026 (8,718.289–8,761.432) | 1,873.04 (1,866.45–1,875.71) |
| 16K books | 1 | 8,772.629 (8,770.791–8,796.333) | 8,755.833 (8,754.786–8,771.069) | 1,867.63 (1,862.59–1,868.02) |
| 32K books | 0 | 30,440.846 (30,173.171–30,580.015) | 30,408.173 (30,138.385–30,542.304) | 1,076.45 (1,071.55–1,086.00) |
| 32K books | 1 | 30,563.509 (30,309.807–30,729.001) | 30,524.658 (30,269.160–30,690.541) | 1,072.13 (1,066.35–1,081.10) |

2K used the earlier chat fixtures and source snapshot; 4K, 8K, 16K and 32K use book passages and the capacity-enabled model. Their fixed book endpoints differ. The 4K/8K runs used b09u02; the 16K/32K runs used c04u14. This table records different contexts, fixtures and machines. It is not a controlled speedup or regression comparison.

## Per-chunk medians

| Context | Slot | Token range | Forward wall, ms | Upload/sync plus forward, ms |
|---|---:|---|---:|---:|
| 2K saved | 0 | [0,1024) | 222.458 | 223.418 |
| 2K saved | 0 | [1024,2048) | 208.149 | 209.137 |
| 2K saved | 1 | [0,1024) | 221.425 | 222.455 |
| 2K saved | 1 | [1024,2048) | 207.491 | 209.250 |
| 4K books | 0 | [0,1024) | 267.013 | 268.026 |
| 4K books | 0 | [1024,2048) | 266.203 | 267.189 |
| 4K books | 0 | [2048,3072) | 265.966 | 266.925 |
| 4K books | 0 | [3072,4096) | 254.981 | 255.942 |
| 4K books | 1 | [0,1024) | 267.453 | 268.548 |
| 4K books | 1 | [1024,2048) | 266.831 | 267.855 |
| 4K books | 1 | [2048,3072) | 266.330 | 267.365 |
| 4K books | 1 | [3072,4096) | 255.502 | 256.512 |
| 8K books | 0 | [0,1024) | 366.163 | 367.200 |
| 8K books | 0 | [1024,2048) | 357.439 | 358.422 |
| 8K books | 0 | [2048,3072) | 357.183 | 358.163 |
| 8K books | 0 | [3072,4096) | 357.653 | 358.657 |
| 8K books | 0 | [4096,5120) | 359.282 | 360.277 |
| 8K books | 0 | [5120,6144) | 358.683 | 359.722 |
| 8K books | 0 | [6144,7168) | 358.691 | 361.239 |
| 8K books | 0 | [7168,8192) | 347.565 | 348.564 |
| 8K books | 1 | [0,1024) | 361.788 | 362.890 |
| 8K books | 1 | [1024,2048) | 359.403 | 360.339 |
| 8K books | 1 | [2048,3072) | 361.203 | 362.242 |
| 8K books | 1 | [3072,4096) | 359.768 | 360.755 |
| 8K books | 1 | [4096,5120) | 360.258 | 361.289 |
| 8K books | 1 | [5120,6144) | 359.503 | 360.529 |
| 8K books | 1 | [6144,7168) | 360.456 | 361.483 |
| 8K books | 1 | [7168,8192) | 349.203 | 350.171 |
| 16K books | 0 | [0,1024) | 549.545 | 550.647 |
| 16K books | 0 | [1024,2048) | 547.885 | 548.935 |
| 16K books | 0 | [2048,3072) | 547.922 | 548.930 |
| 16K books | 0 | [3072,4096) | 549.554 | 550.622 |
| 16K books | 0 | [4096,5120) | 548.181 | 550.567 |
| 16K books | 0 | [5120,6144) | 547.302 | 548.365 |
| 16K books | 0 | [6144,7168) | 546.697 | 548.141 |
| 16K books | 0 | [7168,8192) | 545.530 | 546.532 |
| 16K books | 0 | [8192,9216) | 544.484 | 545.525 |
| 16K books | 0 | [9216,10240) | 545.450 | 546.493 |
| 16K books | 0 | [10240,11264) | 545.769 | 546.770 |
| 16K books | 0 | [11264,12288) | 543.371 | 544.407 |
| 16K books | 0 | [12288,13312) | 544.771 | 545.801 |
| 16K books | 0 | [13312,14336) | 542.936 | 543.953 |
| 16K books | 0 | [14336,15360) | 542.411 | 543.453 |
| 16K books | 0 | [15360,16384) | 531.713 | 532.751 |
| 16K books | 1 | [0,1024) | 551.657 | 552.900 |
| 16K books | 1 | [1024,2048) | 546.317 | 547.334 |
| 16K books | 1 | [2048,3072) | 550.408 | 551.415 |
| 16K books | 1 | [3072,4096) | 546.170 | 547.163 |
| 16K books | 1 | [4096,5120) | 545.656 | 546.654 |
| 16K books | 1 | [5120,6144) | 545.295 | 546.308 |
| 16K books | 1 | [6144,7168) | 547.519 | 548.492 |
| 16K books | 1 | [7168,8192) | 551.075 | 552.120 |
| 16K books | 1 | [8192,9216) | 550.826 | 551.815 |
| 16K books | 1 | [9216,10240) | 550.209 | 551.194 |
| 16K books | 1 | [10240,11264) | 547.189 | 548.181 |
| 16K books | 1 | [11264,12288) | 547.993 | 549.026 |
| 16K books | 1 | [12288,13312) | 548.095 | 549.076 |
| 16K books | 1 | [13312,14336) | 549.943 | 550.926 |
| 16K books | 1 | [14336,15360) | 546.202 | 547.109 |
| 16K books | 1 | [15360,16384) | 533.452 | 534.463 |
| 32K books | 0 | [0,1024) | 961.248 | 962.320 |
| 32K books | 0 | [1024,2048) | 949.263 | 950.300 |
| 32K books | 0 | [2048,3072) | 950.926 | 951.923 |
| 32K books | 0 | [3072,4096) | 947.279 | 948.335 |
| 32K books | 0 | [4096,5120) | 947.058 | 948.064 |
| 32K books | 0 | [5120,6144) | 946.935 | 948.031 |
| 32K books | 0 | [6144,7168) | 944.260 | 945.248 |
| 32K books | 0 | [7168,8192) | 946.918 | 947.939 |
| 32K books | 0 | [8192,9216) | 944.780 | 945.836 |
| 32K books | 0 | [9216,10240) | 948.998 | 950.065 |
| 32K books | 0 | [10240,11264) | 951.981 | 953.002 |
| 32K books | 0 | [11264,12288) | 950.320 | 951.296 |
| 32K books | 0 | [12288,13312) | 950.196 | 951.218 |
| 32K books | 0 | [13312,14336) | 945.855 | 946.885 |
| 32K books | 0 | [14336,15360) | 947.706 | 948.711 |
| 32K books | 0 | [15360,16384) | 948.990 | 949.998 |
| 32K books | 0 | [16384,17408) | 949.647 | 950.792 |
| 32K books | 0 | [17408,18432) | 946.726 | 947.735 |
| 32K books | 0 | [18432,19456) | 947.174 | 948.146 |
| 32K books | 0 | [19456,20480) | 948.491 | 949.468 |
| 32K books | 0 | [20480,21504) | 951.020 | 952.051 |
| 32K books | 0 | [21504,22528) | 954.587 | 955.545 |
| 32K books | 0 | [22528,23552) | 950.760 | 951.713 |
| 32K books | 0 | [23552,24576) | 947.647 | 948.668 |
| 32K books | 0 | [24576,25600) | 948.317 | 949.358 |
| 32K books | 0 | [25600,26624) | 951.787 | 952.849 |
| 32K books | 0 | [26624,27648) | 945.428 | 946.435 |
| 32K books | 0 | [27648,28672) | 945.644 | 946.621 |
| 32K books | 0 | [28672,29696) | 952.962 | 954.029 |
| 32K books | 0 | [29696,30720) | 947.589 | 948.708 |
| 32K books | 0 | [30720,31744) | 951.874 | 952.952 |
| 32K books | 0 | [31744,32768) | 943.157 | 944.189 |
| 32K books | 1 | [0,1024) | 962.719 | 963.866 |
| 32K books | 1 | [1024,2048) | 950.355 | 951.398 |
| 32K books | 1 | [2048,3072) | 949.584 | 950.572 |
| 32K books | 1 | [3072,4096) | 951.118 | 952.141 |
| 32K books | 1 | [4096,5120) | 957.732 | 958.749 |
| 32K books | 1 | [5120,6144) | 962.207 | 963.174 |
| 32K books | 1 | [6144,7168) | 957.844 | 961.988 |
| 32K books | 1 | [7168,8192) | 956.345 | 957.398 |
| 32K books | 1 | [8192,9216) | 958.464 | 959.722 |
| 32K books | 1 | [9216,10240) | 957.354 | 958.394 |
| 32K books | 1 | [10240,11264) | 960.617 | 962.569 |
| 32K books | 1 | [11264,12288) | 957.682 | 959.165 |
| 32K books | 1 | [12288,13312) | 956.027 | 957.031 |
| 32K books | 1 | [13312,14336) | 952.633 | 954.969 |
| 32K books | 1 | [14336,15360) | 955.693 | 956.681 |
| 32K books | 1 | [15360,16384) | 956.222 | 957.239 |
| 32K books | 1 | [16384,17408) | 961.947 | 963.458 |
| 32K books | 1 | [17408,18432) | 953.185 | 954.172 |
| 32K books | 1 | [18432,19456) | 955.311 | 956.306 |
| 32K books | 1 | [19456,20480) | 949.760 | 950.797 |
| 32K books | 1 | [20480,21504) | 947.845 | 948.876 |
| 32K books | 1 | [21504,22528) | 952.942 | 953.958 |
| 32K books | 1 | [22528,23552) | 959.450 | 963.669 |
| 32K books | 1 | [23552,24576) | 953.514 | 954.641 |
| 32K books | 1 | [24576,25600) | 953.290 | 954.283 |
| 32K books | 1 | [25600,26624) | 947.627 | 948.696 |
| 32K books | 1 | [26624,27648) | 949.268 | 950.297 |
| 32K books | 1 | [27648,28672) | 948.023 | 949.082 |
| 32K books | 1 | [28672,29696) | 946.956 | 948.015 |
| 32K books | 1 | [29696,30720) | 946.670 | 947.737 |
| 32K books | 1 | [30720,31744) | 949.222 | 950.268 |
| 32K books | 1 | [31744,32768) | 938.838 | 939.884 |

The [six measured requests](performance-prefill-4k-requests.csv) and [all 24 measured chunks](performance-prefill-4k-chunks.csv) retain the unrounded seconds. The [six 8K measured requests](performance-prefill-8k-requests.csv) and [all 48 8K measured chunks](performance-prefill-8k-chunks.csv) use the same units. The [six 16K measured requests](performance-prefill-16k-requests.csv) and [all 96 16K measured chunks](performance-prefill-16k-chunks.csv) retain the same unrounded seconds. The [six 32K measured requests](performance-prefill-32k-requests.csv) and [all 192 32K measured chunks](performance-prefill-32k-chunks.csv) retain the same unrounded seconds. Independently calculated medians need not add exactly.

## What the timer includes

- **Prompt wall:** before first upload through completion of the last synchronized model forward. It includes upload, dispatch and synchronization.
- **Chunk forward wall:** completed upload synchronization through completed model-call synchronization.
- **Throughput:** input tokens divided by each sequential user's prompt wall.
- **Outside the measured prompt window:** fixture/tokenizer loading, model/cache loading, output readback, finite/repeat checks, hashes, token ranking/decoding, cleanup and report writes.
- Warmups use the same prompt timer but are excluded from measured medians. Cold warmup is not an isolated compiler timing.

| 4K startup or check | Wall time |
|---|---:|
| Fixture/tokenizer load | 1.346 s |
| Model/cache load and synchronization | 68.191 s |
| Slot 0 warmup | 123.178 s |
| Slot 1 warmup | 1.056 s |
| Post-timer readback/check/ranking per request | 2.408–2.481 s |

The first 4K warmup grew the program cache from 10 to 134 entries. All measured requests stayed at 134. Outputs remained allocated until each prompt timer ended. All 1,024 chip/chunk output checks across the eight requests were finite and matched their slot/chunk warmup exactly.

## Book continuation observations

The Instruct checkpoint received **raw book continuation**, not a chat template. Each fixed 4,096-token prefix includes one BOS. Endpoints and expected book tokens were selected before execution. Only the next-token logits were read; no generated token was appended and no decode loop ran.

| Book | Prompt ending | Predicted token | Actual book token | Actual token rank | Top five tokens |
|---|---|---|---|---:|---|
| Pride and Prejudice | …had been fortunate enough to be never without partners, | and | which | 2 | and, which, when, till, from |
| Great Expectations | …my slice. I felt that I must have something | to | in | 2 | to, in, more, a, else |

The predicted **and** and **to** are plausible continuations. The books continue with **which** and **in**, both ranked second. All repeats produced the same observations. This is a readable sanity check, not a golden accuracy test or proof of free-running generation. The [compact evidence JSON](performance-prefill-evidence.json) retains token IDs, probabilities, hashes and all eight observations.

## 8K startup and book observations

| 8K startup or check | Wall time |
|---|---:|
| Fixture/tokenizer load | 2.040 s |
| Model/cache load and synchronization | 64.620 s |
| Slot 0 warmup | 146.347 s |
| Slot 1 warmup | 2.853 s |
| Post-timer readback/check/ranking per request | 4.720–4.875 s |

The first 8K warmup grew the program cache from 10 to 170 entries. All six measured requests stayed at 170. All 2,048 chip/chunk checks across eight requests were finite and exactly matched their slot/chunk warmup. The complete pytest case took 333.53 seconds; that includes setup and cold work and is not warmed prefill latency.

The 8,192-token raw-book inputs include one BOS and use the same Instruct checkpoint. Endpoints were preselected, without a chat template or generated continuation.

| Book | Prompt ending | Predicted token | Actual book token | Actual token rank | Top five tokens |
|---|---|---|---|---:|---|
| Pride and Prejudice | …a woman is partial to a man, and does | not | not | 1 | `not`, `all`, `_`, `him`, `help` |
| Great Expectations | …half expected to see him drop down before my | eyes | face | 2 | `eyes`, `face`, `very`, `CRLF`, `feet` |

The first prediction, **not**, matches the next book word. For “before my”, **eyes** is plausible; the actual word **face** ranks second. All repeats gave the same observations. This remains a next-token sanity check, not a golden numerical accuracy or free-running generation test. The compact evidence JSON retains all eight observations at each book context.

## 16K startup and book observations

| 16K startup or check | Wall time |
|---|---:|
| Fixture/tokenizer load | 3.157 s |
| Model/cache load and synchronization | 206.802 s |
| Slot 0 warmup | 250.951 s |
| Slot 1 warmup | 8.719 s |
| Post-timer readback/check/ranking per request | 9.090–9.490 s |

The first 16K warmup grew the program cache from 10 to 244 entries. All six measured requests stayed at 244. All 4,096 chip/chunk checks across eight requests were finite and exactly matched their slot/chunk warmup. The complete pytest case took 667.629 seconds; that includes setup and cold work and is not warmed prefill latency.

Each fixed 16,384-token raw-book prefix includes one BOS and uses the same Instruct checkpoint. No chat template or decode loop ran.

| Book | Prompt ending | Predicted token | Actual book token | Actual token rank | Top five tokens |
|---|---|---|---|---:|---|
| Pride and Prejudice | …the two elegant ladies who waited on his sisters. | She | In | 11 | `She`, `The`, `These`, `They`, `Her` |
| Great Expectations | …go, if Joe would. Joe said he was agreeable, | and | and | 1 | `and`, `if`, `but`, `so`, `“` |

**She** is a plausible start of a new sentence, but it does not match the book's **In**, which ranks 11th. **and** matches the next token in Great Expectations and ranks first. All repeats gave the same observations. These two endpoints are a next-token sanity check, not a golden accuracy result or a free-running generation test.

## 32K startup and book observations

| 32K startup or check | Wall time |
|---|---:|
| Fixture/tokenizer load | 1.587 s |
| Model/cache load and synchronization | 80.524 s |
| Slot 0 warmup | 346.565 s |
| Slot 1 warmup | 30.046 s |
| Post-timer readback/check/ranking per request | 18.773–19.369 s |

The first 32K warmup grew the program cache from 10 to 388 entries. All six measured requests stayed at 388. All 8,192 chip/chunk checks across eight requests were finite and exactly matched their slot/chunk warmup. The complete pytest case took 850.057 seconds; that includes setup and cold work and is not warmed prefill latency.

Each fixed 32,768-token raw-book prefix includes one BOS and uses the same Instruct checkpoint. No chat template or decode loop ran.

| Book | Prompt ending | Predicted token | Actual book token | Actual token rank | Top five tokens |
|---|---|---|---|---:|---|
| Pride and Prejudice | …go. We are not on friendly terms, and it | is | always | 71 | `is`, `would`, `will`, `does`, `has` |
| Great Expectations | …high overhead, as if she were going out into | the | the | 1 | `the`, `another`, `a`, `heaven`, `some` |

After “and it”, the model predicts **is**. The actual book token is **always**, ranked **71st**, so this endpoint does not match the book. **the** matches the next token in Great Expectations and ranks first. All repeats gave the same observations. These two endpoints are a next-token sanity check, not a golden accuracy result or a free-running generation test.

## Coverage and limits

| Length | Status |
|---|---|
| 2K | Detailed full-model/KV validation published; saved performance retained |
| 4K | Full 32-layer book execution, finite/repeated outputs and eager timing passed |
| 8K | Full 32-layer book execution, finite/repeated outputs and eager timing passed |
| 16K | Full 32-layer book execution, finite/repeated outputs and eager timing passed |
| 32K | Full 32-layer book execution, finite/repeated outputs and eager timing checks completed |
| 64K / 128K | Pending live results; fixed book inputs prepared |

No golden KV comparison was run in the 4K, 8K, 16K or 32K benchmarks. The earlier incomplete larger-context golden matrix remains historical and is not called a complete pass. Future lengths use one combined execution/performance/book run after resource review. Native migration has separate tests. Decoder and SC4 work are outside the current scope.

Attention currently gathers and reorders the entire configured cache for every layer/chunk before selecting the logical prefix. A short prompt with a larger allocation may have different latency. Do not infer 128K performance from these results. These numbers do not measure concurrent serving, decode, device kernel time, network TTFT or general numerical accuracy.

## Exact source and reproduction evidence

4K ran on **bh-glx-120-b09u02**, job **107973**, source HEAD **cee680d3fd208f287c6e1427dfce87076f6f5746**. Actual/dispatch/verified exits are 0. One exact JUnit case passed, with no skips/errors; all **14,448** pinned files stayed unchanged. The 32-device mesh closed at **2026-09-17 07:14:33.316 UTC**.

2K retains its original performance-only/provisional status, source HEAD **734a6c29463be5e33a681354a2b039855f094bda** plus pinned Task8 overlay in tt-metal-perf-b09u02. Its original report SHA256 is **27642d239ea1d5b0fecde5d5046eee863dbfb0358ef40d3d8ec6300b941a2f94**. Later 2K correctness acceptance is documented separately in [validation-2k.md](validation-2k.md).

The site-specific files below are on shared Exabox storage. They bind the exact test, interpreter, parameters and native/source identity. Old one-use attempts must not be replayed directly.

- [4K root verification](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-launch-003/attempts/attempt-002-bfp8-4k/root-verification.json)
- [Exact 4K pytest command](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-launch-003/attempts/attempt-002-bfp8-4k/pytest-command.json)
- [Frozen 4K test](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-preparation-003/test_long_context_performance.py)
- [Frozen launcher and guarded commands](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-launch-003/README.md)
- [4K source manifest](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-launch-003/attempts/attempt-002-bfp8-4k/source-hashes.json)
- [Saved 2K exact command](/data/divanovic/llama31-8b-disagg/evidence/task-8-full-prefill/performance-preparation-002/attempts/attempt-002-bfp8-baseline/pytest-command.json)
- [Saved 2K source manifest](/data/divanovic/llama31-8b-disagg/evidence/task-8-full-prefill/performance-preparation-002/attempts/attempt-002-bfp8-baseline/source-hashes.json)

| 4K artifact | SHA256 |
|---|---|
| Report | 3be58eaae6011aaa96c2eafb2006cba20c123db3ab0ca5700378f9fdced6f1cf |
| Controller | 3abcbca0052590cf37e5eb7f083ebbf3309f47247da2b67278aaa7261eac0bab |
| Test | 9512cb59644134d74a07659938f02b44e5deb619d0e767b213ff755a17249bb8 |
| Full source/native pin manifest | 529b3ec00436750a9565db2a1fb81f83b14560918d8ed4680a091f76ab8849ad |

### 8K source and reproduction

8K ran on **bh-glx-120-b09u02**, job **107973**, source HEAD **3e3f90e1d0bc5d978dd721d66f87e2601ff730fb**. Actual/dispatch/verified exits are 0. One exact JUnit case passed without errors/skips; all **14,522** pinned files stayed unchanged. The 32-device mesh closed at **2026-09-17 08:06:08.801 UTC**.

- [8K controller verification](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-006/8k-b09-001/attempts/attempt-002-bfp8-8k/verification.json)
- [Exact 8K pytest command](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-006/8k-b09-001/attempts/attempt-002-bfp8-8k/pytest-command.json)
- [Frozen reusable test](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-preparation-006/test_long_context_performance.py)
- [Reusable launcher and guarded commands](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-launch-006/README.md)
- [8K source manifest](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-006/8k-b09-001/attempts/attempt-002-bfp8-8k/source-hashes.json)

| 8K artifact | SHA256 |
|---|---|
| Report | 014be819d4f653e28ec69b479f655e66100658e068fff000d923409c76f5306e |
| Controller | 146a638cd143b2d524e05b4c6920d0da41c4e78400244ff0049a28a1f420b09e |
| Test | cf518f6617e09a7317a7d4ab0745591bcabb049d81741a7389573647604b31c0 |
| Full source/native pin manifest | 33738df25eab9b4c4be0dd4918f18b32e420b39b16ecc3a6a0df688895e7afd8 |

### 16K source and reproduction

16K ran on **bh-glx-120-c04u14**, job **108887**, source HEAD **7cb03649b4ec4db7492c49ec91c7fe02f490a87b**. Actual/dispatch/verified exits are 0. One exact JUnit case passed without errors/skips; all **14,603** pinned files stayed unchanged. The 32-device mesh closed at **2026-09-17 09:18:41.533 UTC**. Root verification independently recomputed the timing and coverage checks.

- [16K root verification](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-007/16k-c04-001/attempts/attempt-002-bfp8-16k/root-verification.json)
- [16K controller verification](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-007/16k-c04-001/attempts/attempt-002-bfp8-16k/verification.json)
- [Exact 16K pytest command](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-007/16k-c04-001/attempts/attempt-002-bfp8-16k/pytest-command.json)
- [Frozen reusable test](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-preparation-007/test_long_context_performance.py)
- [Reusable launcher and guarded commands](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-launch-007/README.md)
- [16K source manifest](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-007/16k-c04-001/attempts/attempt-002-bfp8-16k/source-hashes.json)

| 16K artifact | SHA256 |
|---|---|
| Report | 686c346519cdb56075d2c7fa01fb6c834097017af1ce1e76579815856dfdc37e |
| Root verification | dd750f2cd1c5f6b96006615397d6bbc392314a4a44113df61e6919971b9148bc |
| Controller | 182916fd5ea41e9a6567bccdb01c957e8aae6ea964c963012a0f3cf3c82cbeab |
| Test | cf518f6617e09a7317a7d4ab0745591bcabb049d81741a7389573647604b31c0 |
| Full source/native pin manifest | 561f4d11597a66adca9bb4437133391526dedaf6acbbb691187ae51215b4521d |

### 32K source and reproduction

32K ran on **bh-glx-120-c04u14**, job **108887**, source HEAD **16ad28fe3ac9ef1f3a413e73dcb0fdc9d2952833**. Actual/dispatch/verified exits are 0. One exact JUnit case passed without errors/skips; all **14,628** pinned files stayed unchanged. The 32-device mesh closed at **2026-09-17 10:09:51.420 UTC**. Root verification independently recomputed the timing and checked coverage, source identity and both book observations. This accepts the execution/performance scope only.

- [32K root verification](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-007/32k-c04-001/root-verification.json)
- [32K independent agent verification](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-007/32k-c04-001/agent-result-summary.json)
- [32K controller verification](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-007/32k-c04-001/attempts/attempt-002-bfp8-32k/verification.json)
- [Exact 32K pytest command](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-007/32k-c04-001/attempts/attempt-002-bfp8-32k/pytest-command.json)
- [Frozen reusable test](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-preparation-007/test_long_context_performance.py)
- [Reusable launcher and guarded commands](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-launch-007/README.md)
- [32K source manifest](/data/divanovic/llama31-8b-disagg/evidence/task-9-long-context/performance-runs-007/32k-c04-001/attempts/attempt-002-bfp8-32k/source-hashes.json)

| 32K artifact | SHA256 |
|---|---|
| Report | d7673210ffd46ea5d702545b1104bc5cec8fa2f642d4a8565a66680eaefb3566 |
| Root verification | 269d655aba7e714d1ab40bc26eaa24f1d96330ac56e038bfe98996f77a2d1d82 |
| Agent verification | 65f7daaa78c62df85ebd68f03a308a2a817397f51e081379ec37d2d7336eb011 |
| Controller | 182916fd5ea41e9a6567bccdb01c957e8aae6ea964c963012a0f3cf3c82cbeab |
| Test | cf518f6617e09a7317a7d4ab0745591bcabb049d81741a7389573647604b31c0 |
| Full source/native pin manifest | ae958102a1d21e69d5f08329ee61ab170a2090b4b3feb3cf79fb198769de7ca7 |

Do not repeat saved measurements for documentation, tests or other small changes that leave execution unchanged. Reconsider measurement after material changes to precision, communication, chunking, model work or kernels.
