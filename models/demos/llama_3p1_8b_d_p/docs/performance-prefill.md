<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B-Instruct: prefill performance

**4K execution and performance passed on 17 September 2026.** The two slots reached about **3,869 and 3,857 tokens/s/user**. These are warmed eager host-wall measurements on one Blackhole Galaxy. Saved 2K results are retained without a new run.

Each chunk runs embedding, all 32 decoder layers, final RMSNorm and the full vocabulary head. Configuration: SP4/TP8, two independent slots, BF16 weights/activations, BFP8_B cache, 1,024-token chunks, no tracing. Capacity equals the tested prompt length.

## Measured requests

One warmup ran per slot. Then three measured requests ran per slot, alternating slots. Each request completed all its chunks before the next slot began. The users did not run concurrently. Entries below show median (minimum–maximum) across three requests.

| Context | Slot | Prompt wall, ms | Sum of chunk forward wall, ms | Tokens/s/user |
|---|---:|---:|---:|---:|
| 2K saved | 0 | 432.562 (427.171–434.197) | 430.608 (425.203–432.149) | 4,734.58 (4,716.75–4,794.34) |
| 2K saved | 1 | 431.713 (429.613–434.998) | 429.661 (427.595–430.864) | 4,743.89 (4,708.07–4,767.08) |
| 4K books | 0 | 1,058.649 (1,056.521–1,059.395) | 1,054.325 (1,052.627–1,055.159) | 3,869.08 (3,866.36–3,876.87) |
| 4K books | 1 | 1,062.032 (1,057.305–1,066.372) | 1,057.849 (1,053.245–1,061.879) | 3,856.76 (3,841.06–3,874.00) |

2K used the earlier chat fixtures and source snapshot; 4K uses book passages and the capacity-enabled revision. This table records both workloads. It is not a controlled speedup or regression comparison.

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

The [six measured requests](performance-prefill-4k-requests.csv) and [all 24 measured chunks](performance-prefill-4k-chunks.csv) retain the unrounded seconds. Independently calculated medians need not add exactly.

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

## Coverage and limits

| Length | Status |
|---|---|
| 2K | Detailed full-model/KV validation published; saved performance retained |
| 4K | Full 32-layer book execution, finite/repeated outputs and eager timing passed |
| 8K / 16K / 32K / 64K / 128K | Pending live results; fixed book inputs prepared |

No golden KV comparison was run in this 4K benchmark. The earlier incomplete larger-context golden matrix remains historical and is not called a complete pass. Future lengths use one combined execution/performance/book run after resource review. Native migration and SC4 decode have separate gates.

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

Do not repeat saved measurements for documentation, tests or other small changes that leave execution unchanged. Reconsider measurement after material changes to precision, communication, chunking, model work or kernels.
