# 16 users x 1M context under vLLM (KV pair split), 2026-10-27

MiMo-V2.6-Flash-MOPD served by vLLM on one Blackhole Galaxy with every request at a full 1M-token prompt: 4 lanes
(DP) x TP8, 4 requests per lane, 16 in all. This is what the global KV pair split (README, "Global KV pair split";
#62679) was for: without it a lane holds 2.05 requests of 1M, with it 4.01.

## Setup

- Code: `mimo-v2.6/dev` at the #62679 merge (the pair split, dev's grouped / K split chunked MLA prefill),
  plugin `mimo-v2.6-flash/vllm-plugin-dev` at ee2b54f.
- Server: `vllm serve` as in README ("Launch"), with `--data-parallel-size 4 --max-num-seqs 4 --max-model-len 1048576
  --block-size 64 --enable-chunked-prefill --max-num-batched-tokens 2048 --enable-prefix-caching`. Additional config:
  `trace_mode decode_only`, `sample_on_device_mode decode_only`, a 1 GB trace region, and `prefill_gather_ms` 200 /
  `prefill_gather_quiet_ms` 25.
- Env:
  - `MIMO_KV_PAIR_SPLIT` is unset (default 1).
  - `MIMO_KV_GB_PER_CHIP` is unset, so the pool takes the pair split's 6.7 GB default.
  - `MIMO_EXPERT_QUANT=gptq`.
- Server start takes 8 minutes. That includes the prefill warmup and a first-start kernel compile after a fresh
  build.
- Client: `tests/perf/vllm_long_prompts.py`.
  - Each prompt is exactly 1,048,000 random token ids, sent as a token list, so the server counts 1,048,000 prompt
    tokens. Every prompt is distinct, so there are no prefix-cache hits.
  - Each request generates 128 greedy tokens with `ignore_eos` and streams them back.
- Two runs on the same server:
  - one prompt alone (`--n 1 --concurrency 1`);
  - then 16 at once (`--n 16 --concurrency 16`).

## Results

All 16 requests completed in 3428 s (57.1 minutes), with no error and no device out-of-memory.

**KV pool.** At the default 6.7 GB vLLM reports 4,202,656 tokens per lane, which is a maximum concurrency of 4.01 for
1,048,576-token requests (34,216 blocks per lane). All four 1M requests of a lane stay resident together. At the old
6.0 default the pool held 3.59 requests, and at 6.6 it held 3.95; the adapter now warns when the cap holds fewer than
`--max-num-seqs` requests of `--max-model-len`.

**Time to first token.** A lane prefills its four requests one after another, in about 855 s each. The four lanes
keep step:

| Request on its lane | TTFT (the 4 lanes) |
|---|---|
| 1st | 853-859 s (14.3 min) |
| 2nd | 1713-1716 s (28.6 min) |
| 3rd | 2569-2573 s (42.9 min) |
| 4th | 3423-3426 s (57.1 min) |

One 1M prompt alone (the other three lanes idle) has a TTFT of 355.5 s.

**Prefill speed.** Averages are over the whole prompt, from position 0 to 1M:

| Case | Time per 1M prompt | Prefill throughput |
|---|---|---|
| One prompt, one lane busy (vLLM) | 356 s | 2.95K tok/s |
| 16 prompts, all 4 lanes busy (vLLM) | 855 s | 1.22K tok/s per lane, 4.9K tok/s for the Galaxy |
| Standalone model, one row, last 4K chunk at 1M (`long_1m.py --kv-pair-split`) | 1.57 s per 4K chunk | 2.6K tok/s at the 1M end |

**Decode.**
- One request alone at 1M decodes at 12.3 ms per token (81 tok/s). That is the short-context c=1 TPOT (9.3 ms) plus
  reading the user's 1.6 GB of global KV per chip (3.6 ms at 448 GB/s).
- While a lane is still prefilling its later requests, a request that already has its first token gets one decode
  step per scheduler step, about 2.76 s per token: its 128 tokens took about 351 s. The lane's prefill cadence stays
  at 855-860 s per prompt.
- c=16 decode with all 16 users at a full 1M context was not measured here. The requests asked for 128 tokens only,
  so the last four decoded with the other twelve done. The bandwidth model gives about 27-33 ms per token there,
  i.e. 4 users x 1.6 GB of global KV per chip each step.

## Standalone comparison

Results from `tests/perf/long_1m.py --mode capacity --users 4` (one row runs; all 4 rows are built), from
`doc/tp8_dp4_plan.md`, "1M prefill on all 4 rows". The DRAM column is for 4 users x 1M per row; the last chunk is the
4K chunk ending at 1M.

| Global KV layout | DRAM per chip (34.18 GB) | Last 4K chunk at 1M |
|---|---|---|
| Pair split, `--kv-pair-split` (vLLM's default) | 30.72 GB used, 3.46 GB free | 1.57 s |
| Users split (one chip of the pair owns each user) | 30.71 GB used, 3.47 GB free | 2.78 s |
| Both chips keep the KV, `--replicate-global-kv` (head split) | out of memory at 4 users; fits 1 user per row | 1.46 s |

The pair split fits 4 x 1M per row like the users split, and its later chunks run within 7% of the head split's. Each
chip runs all 16 heads over its half of the keys (G 2 / S 15 from 64K) and the pair merges by LSE.

## What limits prefill

- **The host, when the four lanes prefill together.**
  - Throughput:
    - one lane alone prefills at 2.95K tok/s;
    - four lanes together reach 4.9K tok/s, only 1.7x one lane.
  - Per prompt, each lane takes 2.4x longer than alone. The device work per chunk does not change with the number of
    busy lanes, so the time goes to host dispatch.
  - The standalone model shows the same effect with one host thread for 4 rows: 476 s per prompt against 277 s for one
    row alone (tp8_dp4_plan.md, the head split).
- **vLLM's 2048-token prefill chunks.**
  - They double the number of chunks per prompt against the standalone model's 4096, and with it the per-chunk host
    overhead.
  - Every chunk also re-reads the context before it.
  - One prompt takes 356 s in vLLM, against about 300 s scaled for the standalone model at 4096-token chunks.
- **Per-lane FCFS scheduling.** A lane's four requests do not overlap. The first gets its first token at 14 minutes,
  the last at 57.

## Next

- Raise `--max-num-batched-tokens` to 4096: half the chunks, and so half the per-chunk host cost of a 1M prompt.
- Cut host dispatch per chunk:
  - a traced prefill chunk, or one host thread per lane, as #62562 lists for the standalone model (traced later
    chunks are now in the adapter, #62972, and on by default: README, `MIMO_PREFILL_CHUNK_TRACE`);
  - with four lanes busy, this is the bound.
- Measure c=16 decode at full 1M: 16 x 1M with long outputs (e.g. 2000 tokens), so that all 16 decode together once
  the last prefill ends.
- The prefill merge's LSE exchange is a row all_gather (`PairSplit.merge`). It costs about 1.2 ms per global layer per
  chunk, about 1% of a 1M prompt. A pair-only exchange would save that.
