# DeepSeek-V4.1-Flash on Blackhole Galaxy: release notes (rc4c)

Release candidate `rc4c` (vLLM 0.31.0). Start-up commands: `GETTING_STARTED.md`. Test logs and long-form findings: `RELEASE_DETAILS.md`. Optimization-by-optimization history and switches: `OPTIMIZATIONS.md`.

| | |
|---|---|
| Image (local build) | `dsv41-flash-local:rc4c-785f0df9b28-2406dcb1-d7b6901` |
| Registry name / tag | placeholder: `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:<tag>` (to be filled in when the image is pushed; nothing is pushed yet) |
| Commits | tt-metal `785f0df9b28` (branch `ssinghal/dsv4p1-rc3`), tt-inference-server `2406dcb15` (`ssinghal/dsv4p1-rc3-ttis`), vllm-tt-plugin `d7b6901` (`ssinghal/dsv4p1-rc3-plugin`) |
| Offline copy | `/mnt/tt-data/ssinghal/dsv4-docker-ctx/img_rc4c_785f0df9b28.tar` (`docker load -i`) |

This candidate is configured for **B=8 and B=16**. B=4 is not a deliverable of rc4c (the catalog still contains B=4 entries, they were not validated for this candidate).

## What is supported

| | B=8 (catalog default) | B=16 |
|---|---|---|
| Users | 8 (4 mesh rows x 2) | 16 |
| Max context | 1,048,576 tokens | 262,144 tokens |
| Select with | no `--impl` (default entry) | `--impl deepseek-v41-flash` |
| Image input | up to 250 images per prompt (`limit-mm-per-prompt`) | same |
| Speculative decoding | adaptive k = 3 / 5 (`DSV41_VLLM_SPEC_KS=3,5`), draft model owned by the plugin | k = 5 |
| Chunked prefill budget | `max_num_batched_tokens` 32,768, `long_prefill_token_threshold` 4,096 | 67,584 |
| Runtime spec file | none needed: the baked catalog (`model_spec.json` in the image) is used | none needed |
| Start-up (weight and kernel caches warm, from `docker run` to `/health` 200) | 1,274-1,635 s (21-27 min) | 803-934 s (13-16 min) |

Start-up times: `dsv4-logs/rc4c_val/srv/*/info` (`SERVER_READY ... boot N s`): B=8 holds a1 1,635 s, a2 1,274 s, a3 1,304 s, r8 1,317 s, s8 1,274 s; B=16 holds f16 813 s, f16b 803 s, r16 934 s. The first start with empty caches is much longer (not measured for rc4c).

## What is in rc4c

Serving behaviour that is on by default in the catalog:
- **Structured outputs on the speculative path (M1 mode).** `tool_choice` required / named, `strict: true`, `response_format` `json_schema` / `json_object`, and guided regex / choice / grammar are served; a structured row decodes without drafts under an on-device grammar mask while the other rows keep speculating. Streaming requests with an unsupported combination (logprobs, min_tokens, bad_words) get HTTP 400 before the first event; stream headers go out after the first output. Switches (set in the catalog): `DSV41_STRUCTURED_SPEC=1`, `VLLM_TT_STRUCTURED_SPEC=1`, `DSV41_INTRACE_SAMPLE=1`, `DSV41_STRUCTURED_M2=0`, `VLLM_TT_STRUCTURED_M2=0`.
- **Image-cut fix** (`DSV41_PIECE_LAG=1`): a prefill piece that ends off a chunk boundary or inside an image is cut back, and the next piece resumes at the carried end instead of recomputing from position 0 (device A/B by the branch owner; this rc4c validation did not repeat it).
- **Prefix caching on a 512-token grid, block-paged on device at B=8** (`DSV41_APC_GRID=512`, `DSV41_APC_PAGED=1` for the B=8 entries). A repeated or shared prefix is restored instead of recomputed (numbers below). Hits are bit-identical to a cold prefill in the owner's device test. B=16 uses the host-tier prefix cache on the same grid.
- **Prefix cache and images**: image content hashes are part of the cache key (`DSV41_APC_IMAGES=1`); vLLM's own prefix-cache reads are skipped for requests with images (plugin). **`cache_salt` is honoured**: a request with another salt never reuses a cached prefix.
- **Cross-context vision fix**: a stale image mask in one prefill context could leak into the next text-only window (text chats answered with grounding text in a mixed image + text run); fixed on the code path.
- `usage.prompt_tokens_details.cached_tokens` is returned (`enable-prompt-tokens-details` in the catalog).
- Earlier B=8 work included in this image: fp8 KV serving with APC (`DSV41_FP8_SERVING`, paged fp8 prefill), mHC PRE16 / X2LITE, column-TP shared expert, fused MoE tail, post-SDPA glue, fused2 router, key-slab sharded indexer (`DSV41_IDX_KEY_SHARD`), host-side round work (`DSV41_HOSTOPT_*`), drafter sets A and B, row-aware / Up=2 prefill, `DSV41_UNI_TOPO=linear` (fix of an intermittent combine deadlock), MoE core pin (`MOE_COMPUTE_PIN_EAST_X=7`).

Available but off (not part of this release): APC delta snapshots, decode-output caching, cross-row paged hits, the short-prompt prefill lever, structured M2 (masked verify with drafts). They are default off in code and not in the catalog; see `OPTIMIZATIONS.md`.

## Performance

All numbers are rc4c, from `dsv4-logs/rc4c_val/`. Method: tt-inference-server `--workflow benchmarks` (`run.py --workflow benchmarks --tools vllm`, client-only against the running server), the workflow's own random-prompt sweep with one unique `cache_salt` per sweep point (`llm_module/drivers/vllm.py` of the catalog branch), so the rows are cold-prefix measurements (no prefix-cache hit). The grid itself ran from `wt/ttis_bench_grid`, which I did not diff against the catalog branch. Cells are `TTFT s / tok/s per user`. Every cell is a single run (n = 1 at c1, one wave at 262,144), so differences of several percent between neighbouring cells are noise. Sources: grid `grid/final_cmp_g8.txt` (rc4c columns), standard points `table_b8_rc4c.txt`, B=16 `rel_b16_rc4c.txt`; the tables are generated by `mk_perf_tables_rc4c.py` into `perf_tables_rc4c.md`.

## B=8 (1M build)

### B=8 workflow grid (TTFT s / tok/s per user)

| ISL | OSL | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| 1024 | 1024 | 1.59/59 | 1.68/54 | 1.92/47 | 1.99/36 |
| 1024 | 4096 | 1.56/88 | 1.66/71 | 1.78/71 | 2.22/59 |
| 4096 | 1024 | 3.86/66 | 3.98/47 | 4.31/33 | 4.81/25 |
| 4096 | 4096 | 3.86/87 | 3.98/63 | 4.32/65 | 5.07/55 |
| 8192 | 1024 | 6.18/77 | 6.33/48 | 6.78/33 | 8.21/25 |
| 8192 | 4096 | 6.17/93 | 6.34/68 | 6.76/55 | 7.79/45 |
| 16384 | 1024 | 10.83/74 | 11.14/39 | 11.72/24 | 14.94/17 |
| 16384 | 4096 | 10.73/83 | 11.02/64 | 11.68/45 | 13.18/39 |
| 32768 | 1024 | 20.01/68 | 20.63/29 | 21.54/16 | 28.70/11 |
| 32768 | 4096 | 20.03/100 | 20.42/61 | 21.65/36 | 27.22/30 |
| 65536 | 1024 | 38.79/69 | 39.83/21 | 42.20/9 | 49.78/7 |
| 65536 | 4096 | 38.63/95 | 39.44/50 | 42.88/29 | 47.07/20 |
| 131072 | 1024 | 76.38/95 | 81.11/13 | 87.11/6 | 101.23/4 |
| 131072 | 4096 | 76.40/95 | 79.72/29 | 85.81/18 | 102.65/14 |
| 262144 | 1024 | 151.57/83 | 171.01/36 | 183.07/47 | 206.09/51 |
| 262144 | 4096 | 151.81/99 | 171.20/75 | 183.02/75 | 204.94/76 |

### B=8 workflow standard points (TTFT s / tok/s per user)

| ISL | OSL | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| 128 | 128 | 0.93/39 | 0.64/39 | 0.71/29 | 0.81/19 |
| 128 | 1024 | 0.59/76 | 0.70/61 | 0.79/59 | 0.91/51 |
| 1024 | 128 | 1.55/45 | 1.72/32 | 1.92/20 | 2.00/11 |
| 2048 | 128 | 2.51/48 | 2.68/27 | 3.00/18 | 3.49/8 |
| 4096 | 128 | 3.85/47 | 4.05/21 | 4.81/11 | 5.32/5 |
| 8192 | 128 | 6.17/46 | 6.50/20 | 7.84/11 | 8.19/5 |
| 8192 | 1024 | 6.17/69 | 6.50/53 | 7.02/37 | 7.77/24 |
| 10000 | 1024 | 7.05/74 | 7.46/49 | 7.96/37 | 9.68/23 |
| 16384 | 128 | 10.79/49 | 11.42/42 | 13.36/7 | 15.22/4 |
| 32768 | 128 | 19.99/47 | 20.78/41 | 25.25/30 | 29.19/33 |
| 65536 | 128 | 38.74/46 | 39.78/41 | 45.69/30 | 52.27/31 |
| 131072 | 128 | 76.28/40 | 90.88/10 | 99.12/16 | - |


**Beyond 262,144 tokens.** The workflow grid was not run beyond ISL 262,144 for rc4c (the 524,288 / 786,432 cells are not reported). The 1M context is covered by the needle test only: 1,039,901 prompt tokens found, 596 s (`res_a2/needle1m.log`).

## B=16 (262,144 build)

**The B=16 workflow grid (ISL x OSL x concurrency) was not run for rc4c.** The numbers that exist for rc4c B=16 are the release-table method (workflow, GSM8K prompts, ISL 1,024 / OSL 256, 3 repeats), the shared-prefix check, the start-up time and the checks in the accuracy section:

### B=16 (262,144), release-table method, ISL 1,024 / OSL 256 (TTFT s / tok/s per user, mean of 3 repeats)

| Sampling | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|
| greedy | 0.75/35.7 | 0.63/32.5 | 0.68/28.4 | 0.96/23.9 | 1.44/18.4 |
| sampled | 0.77/33.3 | 0.66/31.5 | 0.72/28.0 | 1.04/24.3 | 1.55/18.2 |

Shared 32,006-token prompt on B=16: first request 12.64 s, then 2.49 s with 29,696 tokens cached (`out/f16/apc_grid_shared.log`; the log label says `b16rc3`, the server was the rc4c image per `srv/f16/info`). Start-up 803-934 s with warm caches (`srv/f16/info`, `f16b`, `r16`).

## Shared-prefix rows (rc4c B=8; prefix-cache workloads, not the cold grid)

These use a shared prompt on purpose, so they hit the prefix cache (`res_a1/`): 8,192-token prompt repeated: TTFT 5.67 s cold -> 0.67 s (7,680 tokens cached, `apc_a.log`); 32,006-token shared prompt: TTFT 2.65 s with 29,696 tokens cached (`wl_shared.log`); multi-turn: at 32,000 tokens turn 2 TTFT 0.79 s against 19.42 s for a cold prompt of the same size, at 64,000 tokens 1.34 s against 37.6 s (`multiturn.log`); 8 agents with growing contexts: mean TTFT of turns after the first 9.12 s, 72 % of prompt tokens cached (`wl_agents.log`). A request with another `cache_salt` shows `cached 0` (`apc_a.log`). There is no ISL x OSL x concurrency grid with a shared salt.

## Accuracy and checks (rc4c, B=8, 1M build, catalog default)

| Check | Result | Source |
|---|---|---|
| GSM8K (lm-eval, flexible-extract) | 96.21 % (stderr 0.53) | `res_a2/gsm.log` |
| GPQA diamond (zero-shot CoT, flexible-extract) | 78.79 % (stderr 2.91) | `res_a2/gpqa.log` |
| Needle in a haystack | found at 65,557 / 131,085 / 262,142 / 599,956 / 1,039,901 prompt tokens (1M run: 596 s) | `res_a2/needle_*.json`, `needle1m.log` |
| Tool-calling checks (`toolchecks`) | 35 PASS, 0 FAIL | `res_a1/toolchecks.log` |
| Structured-output matrix (json_schema / regex / choice / tools, plain and stream, greedy and sampled) | 28/28 on two reruns (`res_a3/matrix_rerun1.log`, `matrix_rerun2.log`) and 28/28 mixed with background load (`res_a1/matrix_mixed.log`); **the first run was 27/28** (one `json_schema/plain/sampled` cell failed, `res_a1/matrix.log`) | files named |
| API checks | 23/23 | `res_a1/api.log` |
| Pre-stream validation | HTTP 400 for unsupported streaming combinations; supported ones HTTP 200 | `res_a1/prestream.log` |
| Images | 250 images in one prompt (48,779 tokens, 49.4 s) found 250/250; 100 images 100/100; 33 repeated image prompts all correct (no image prefix hit at these sizes: `cached 0`) | `res_a1/many250.log`, `apcimg_repeat.log` |
| Image + text soak (leak test) | 1,464 requests with the short-prompt lever on / 1,485 with it off, 0 bad, 0 errors | `res_a3/leak.log` |
| `cached_tokens` accounting | `CACHED_TOKENS_OK` (11,264 of 11,626 on a repeat) | `res_a1/cached.log` |

## Known limitations

- The determinism check FAILS (`res_a1/det.log`, `res_a2/det.log`): identical requests give different texts (5 sequential identical requests gave 2 distinct texts, 8 concurrent identical ones 2, and mixed-batch repeats diverged after 107-245 characters). Cause not investigated.
- One structured-matrix cell failed once (27/28, above) and passed on both reruns; the cause was not investigated.
- Image prompts: no prefix-cache hit was observed in the validation (small prompts below the cached block size); vLLM's own prefix reads are skipped for requests with images.
- A long prompt stalls the other users' decoding while it prefills (token budget); concurrent very long prompts prefill one after another.
- Streaming requests with `logprobs`, `min_tokens` or `bad_words` are refused with HTTP 400 (pre-stream validation); other unsupported combinations were not checked in the rc4c validation.
- Cache disk sizes were not measured for rc4c (earlier images: checkpoint 476 GB, weight cache about 610 GB, kernel cache about 43 GB).
