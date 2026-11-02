# DeepSeek-V4.1-Flash on Blackhole Galaxy: release notes (rc4c)

Release candidate `rc4c` (vLLM 0.31.0). Start-up commands: `GETTING_STARTED.md`. Test logs and long-form findings: `RELEASE_DETAILS.md`. Optimization-by-optimization history and switches: `OPTIMIZATIONS.md`.

| | |
|---|---|
| Image (local build) | `dsv41-flash-local:rc4c-785f0df9b28-2406dcb1-d7b6901` |
| Registry name / tag | placeholder: `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:<tag>` (to be filled in when the image is pushed; nothing is pushed yet) |
| Commits | tt-metal `785f0df9b28` (branch `ssinghal/dsv4p1-rc3`), tt-inference-server `2406dcb15` (`ssinghal/dsv4p1-rc3-ttis`), vllm-tt-plugin `d7b6901` (`ssinghal/dsv4p1-rc3-plugin`) |
| Offline copy | `/mnt/tt-data/ssinghal/dsv4-docker-ctx/img_rc4c_785f0df9b28.tar` (`docker load -i`) |

The previous documents of this directory described the `0.24.0-0655530-4743e01` release (B=16 / B=4). This candidate is configured for **B=8 and B=16**. B=4 is not a deliverable of rc4c (the catalog still contains B=4 entries, they were not validated for this candidate).

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
- **Image-cut fix** (`DSV41_PIECE_LAG=1`): a prefill piece that ends off a chunk boundary or inside an image is cut back, and the next piece resumes at the carried end instead of recomputing from position 0 (device A/B by the branch owner: wasted recompute 32.6 % -> 2.0 %; this rc4c validation did not repeat the A/B).
- **Prefix caching on a 512-token grid, block-paged on device at B=8** (`DSV41_APC_GRID=512`, `DSV41_APC_PAGED=1` for the B=8 entries). A repeated or shared prefix is restored instead of recomputed (numbers below). Hits are bit-identical to a cold prefill in the owner's device test. B=16 uses the host-tier prefix cache on the same grid.
- **Prefix cache and images**: image content hashes are part of the cache key (`DSV41_APC_IMAGES=1`); vLLM's own prefix-cache reads are skipped for requests with images (plugin). **`cache_salt` is honoured**: a request with another salt never reuses a cached prefix.
- **Cross-context vision fix**: a stale image mask in one prefill context could leak into the next text-only window (text chats answered with grounding text in a mixed image + text run); fixed on the code path.
- `usage.prompt_tokens_details.cached_tokens` is returned (`enable-prompt-tokens-details` in the catalog).
- Earlier B=8 work that was already in the previous candidates: fp8 KV serving with APC (`DSV41_FP8_SERVING`, paged fp8 prefill), mHC PRE16 / X2LITE, column-TP shared expert, fused MoE tail, post-SDPA glue, fused2 router, key-slab sharded indexer (`DSV41_IDX_KEY_SHARD`), host-side round work (`DSV41_HOSTOPT_*`), drafter sets A and B, row-aware / Up=2 prefill, `DSV41_UNI_TOPO=linear` (fix of an intermittent combine deadlock), MoE core pin (`MOE_COMPUTE_PIN_EAST_X=7`).

Available but off (not part of this release): APC delta snapshots, decode-output caching, cross-row paged hits, the short-prompt prefill lever, structured M2 (masked verify with drafts). They are default off in code and not in the catalog; see `OPTIMIZATIONS.md`.

## Performance

Method: tt-inference-server `--workflow benchmarks` (`run.py --workflow benchmarks --tools vllm`, client-only against the running server), the workflow's own random-prompt sweep with one **unique `cache_salt` per sweep point** (`llm_module/vllm_bench_cache_off.py`), so every row below is a **cold-prefix** measurement (no prefix-cache hit; the unique salt is in `llm_module/drivers/vllm.py` of the catalog branch; the grid itself ran from `wt/ttis_bench_grid`, which I did not diff against it). Cells are `TTFT s / tok/s per user`. The comparison column is the **rc3 image** (`bf46d461048-213b63e9-25ca837`, the previous candidate, measured with the same client); there is no workflow measurement of the `0.24.0` release in these files. Single runs: the rc4c cells are n = 1 at c1 and one wave at 262,144, so per-cell swings of up to about +-10 % are noise. Sources: `dsv4-logs/rc4c_val/grid/final_cmp_g8.txt` (grid), `dsv4-logs/rc4c_val/table_b8_rc4c.txt` and `dsv4-logs/rc3_cmp/table_b8_rc3.txt` (standard points), `dsv4-logs/rc3_cmp/table_b16_full_rc3.txt` (B=16 rc3), tables generated by `dsv4-logs/rc4c_val/mk_perf_tables.py` into `dsv4-logs/rc4c_val/perf_tables.md`.

Summary, B=8 (1M build): over the 64 grid cells TTFT median -3.3 % and tok/s/user median +2.9 % against rc3 (`final_cmp_g8.txt`); over the 48 standard points TTFT median -3.8 %, tok/s/user median +0.0 % (`cmp_b8_std_rc4c.txt`). 24 grid cells moved by more than 10 % in tok/s/user in either direction, mostly c1 single runs (listed in `final_cmp_g8.txt`); the validation owner's reading is no regression pattern (`par/FINAL_SUMMARY.txt`). Outlier of note: ISL 128 / OSL 128 c1 TTFT 0.61 -> 0.93 s (first cell of the run; warm-up / noise per the owner, not re-run).

### B=8 workflow grid, rc3 -> rc4c (TTFT s / tok/s per user)

| ISL | OSL | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| 1024 | 1024 | 1.61/68 -> 1.59/59 | 1.70/53 -> 1.68/54 | 1.88/45 -> 1.92/47 | 2.11/35 -> 1.99/36 |
| 1024 | 4096 | 1.60/97 -> 1.56/88 | 1.72/78 -> 1.66/71 | 1.93/65 -> 1.78/71 | 2.37/54 -> 2.22/59 |
| 4096 | 1024 | 3.99/72 -> 3.86/66 | 4.12/49 -> 3.98/47 | 4.38/35 -> 4.31/33 | 4.83/24 -> 4.81/25 |
| 4096 | 4096 | 3.99/57 -> 3.86/87 | 4.09/57 -> 3.98/63 | 4.42/59 -> 4.32/65 | 5.36/50 -> 5.07/55 |
| 8192 | 1024 | 6.40/75 -> 6.18/77 | 6.57/44 -> 6.33/48 | 7.41/30 -> 6.78/33 | 9.00/25 -> 8.21/25 |
| 8192 | 4096 | 6.41/79 -> 6.17/93 | 6.55/75 -> 6.34/68 | 7.02/60 -> 6.76/55 | 8.49/46 -> 7.79/45 |
| 16384 | 1024 | 11.01/71 -> 10.83/74 | 11.34/38 -> 11.14/39 | 12.05/23 -> 11.72/24 | 14.27/16 -> 14.94/17 |
| 16384 | 4096 | 11.21/83 -> 10.73/83 | 11.38/72 -> 11.02/64 | 12.12/48 -> 11.68/45 | 16.00/39 -> 13.18/39 |
| 32768 | 1024 | 20.83/87 -> 20.01/68 | 21.11/30 -> 20.63/29 | 22.27/16 -> 21.54/16 | 28.71/10 -> 28.70/11 |
| 32768 | 4096 | 20.82/90 -> 20.03/100 | 21.12/52 -> 20.42/61 | 22.23/42 -> 21.65/36 | 31.61/33 -> 27.22/30 |
| 65536 | 1024 | 39.55/57 -> 38.79/69 | 40.58/21 -> 39.83/21 | 44.86/10 -> 42.20/9 | 49.85/6 -> 49.78/7 |
| 65536 | 4096 | 40.24/87 -> 38.63/95 | 40.81/44 -> 39.44/50 | 43.38/28 -> 42.88/29 | 48.84/19 -> 47.07/20 |
| 131072 | 1024 | 77.87/71 -> 76.38/95 | 82.73/13 -> 81.11/13 | 89.47/6 -> 87.11/6 | 105.97/4 -> 101.23/4 |
| 131072 | 4096 | 79.54/98 -> 76.40/95 | 81.71/31 -> 79.72/29 | 90.48/20 -> 85.81/18 | 100.81/12 -> 102.65/14 |
| 262144 | 1024 | 155.37/71 -> 151.57/83 | 176.50/39 -> 171.01/36 | 191.04/37 -> 183.07/47 | 210.17/44 -> 206.09/51 |
| 262144 | 4096 | 158.62/73 -> 151.81/99 | 179.28/75 -> 171.20/75 | 191.77/80 -> 183.02/75 | 207.16/70 -> 204.94/76 |

### B=8 workflow standard points, rc3 -> rc4c (TTFT s / tok/s per user)

| ISL | OSL | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| 128 | 128 | 0.6/52 -> 0.9/39 | 0.7/39 -> 0.6/39 | 0.7/30 -> 0.7/29 | 0.9/20 -> 0.8/19 |
| 128 | 1024 | 0.6/72 -> 0.6/76 | 0.7/71 -> 0.7/61 | 0.7/54 -> 0.8/59 | 0.9/47 -> 0.9/51 |
| 1024 | 128 | 1.6/45 -> 1.6/45 | 1.9/35 -> 1.7/32 | 1.9/19 -> 1.9/20 | 2.2/11 -> 2.0/11 |
| 2048 | 128 | 2.6/49 -> 2.5/48 | 2.8/26 -> 2.7/27 | 3.1/15 -> 3.0/18 | 3.7/8 -> 3.5/8 |
| 4096 | 128 | 4.0/49 -> 3.9/47 | 4.3/25 -> 4.0/21 | 4.4/10 -> 4.8/11 | 5.8/6 -> 5.3/5 |
| 8192 | 128 | 6.4/52 -> 6.2/46 | 6.8/20 -> 6.5/20 | 7.3/10 -> 7.8/11 | 9.3/5 -> 8.2/5 |
| 8192 | 1024 | 6.4/79 -> 6.2/69 | 6.7/47 -> 6.5/53 | 7.4/35 -> 7.0/37 | 8.5/23 -> 7.8/24 |
| 10000 | 1024 | 7.3/74 -> 7.1/74 | 7.8/55 -> 7.5/49 | 8.4/34 -> 8.0/37 | 9.5/22 -> 9.7/23 |
| 16384 | 128 | 11.2/46 -> 10.8/49 | 11.6/15 -> 11.4/42 | 12.6/9 -> 13.4/7 | 17.3/4 -> 15.2/4 |
| 32768 | 128 | 20.8/55 -> 20.0/47 | 21.8/36 -> 20.8/41 | 24.2/38 -> 25.2/30 | 32.0/32 -> 29.2/33 |
| 65536 | 128 | 40.3/48 -> 38.7/46 | 41.6/44 -> 39.8/41 | 48.4/29 -> 45.7/30 | 56.8/32 -> 52.3/31 |
| 131072 | 128 | 79.5/47 -> 76.3/40 | 91.4/9 -> 90.9/10 | 103.2/14 -> 99.1/16 | - |

### B=16 workflow grid, rc3 only (TTFT s / tok/s per user; no rc4c grid exists)

| ISL | OSL | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|---|
| 1024 | 1024 | 1.1/27 | 1.2/28 | 1.2/29 | 1.7/22 | 2.4/21 |
| 1024 | 4096 | 1.1/39 | 1.2/35 | 1.2/39 | 1.7/40 | - |
| 4096 | 1024 | 2.7/35 | 2.8/30 | 2.8/26 | 3.6/20 | 5.2/16 |
| 4096 | 4096 | 2.7/34 | 2.7/36 | 3.2/38 | 3.7/31 | 7.6/32 |
| 8192 | 1024 | 4.2/22 | 4.3/30 | 4.3/24 | - | - |
| 8192 | 4096 | 4.3/39 | 4.3/35 | 4.5/38 | 8.4/32 | 10.0/29 |
| 16384 | 1024 | 7.2/18 | 7.3/27 | 7.3/20 | 8.7/14 | 19.0/10 |
| 16384 | 4096 | 7.2/13 | 7.3/39 | 7.9/32 | 8.7/29 | 17.6/24 |
| 32768 | 1024 | 13.4/30 | 13.4/23 | 14.7/15 | 18.9/10 | 39.8/6 |
| 32768 | 4096 | 13.3/44 | 13.4/37 | 14.3/29 | 21.9/24 | 38.9/17 |
| 65536 | 1024 | 25.4/49 | - | - | - | - |
| 65536 | 4096 | 25.5/53 | 27.0/35 | 30.0/25 | 46.4/15 | 110/11 |
| 131072 | 1024 | 49.9/36 | 51.4/13 | 58.7/6 | 126/4 | 290/2 |
| 131072 | 4096 | 49.7/50 | 51.3/26 | 58.7/16 | 95.2/11 | 217/6 |

**B=8 beyond 262,144 tokens.** rc4c was not run beyond ISL 262,144 in the workflow grid (the 524,288 / 786,432 cells are not reported). The 1M context is covered by the needle test (1,039,901 tokens found, 596 s; `res_a2/needle1m.log`). For reference rc3 measured 1,047,296 x 1,024 at c1 / c8: TTFT 625.6 s / 782.2 s, 55 / 19 tok/s per user (`rc3_cmp/table_b8_full_rc3.txt`); no rc4c counterpart exists.

**B=16 (262,144 build).** The only workflow grid in the validation files is rc3's (table above, ISL up to 131,072; `-` = cell not run). **No rc4c B=16 workflow grid was run**, so there is no rc3 -> rc4c comparison at those points. What exists for rc4c B=16 is the release-table method (workflow, GSM8K prompts, ISL / OSL 1,024 x 256, 3 repeats, greedy; `rel_b16_rc4c.txt` against `rel_b16_rc3.txt`): no change within noise, e.g. c1 TPOT 27.8 -> 28.0 ms, TTFT 0.75 -> 0.75 s; c8 TPOT 41.6 -> 41.9 ms, TTFT 0.94 -> 0.96 s; c16 TPOT 53.8 -> 54.3 ms, TTFT 1.43 -> 1.44 s, 252.7 -> 251.0 output tok/s.

### Shared-prefix rows (prefix-cache workloads, not the cold grid)

These use a shared prompt on purpose, so they hit the prefix cache. rc4c, B=8 (`res_a1/`): 8,192-token prompt repeated: TTFT 5.67 s cold -> 0.67 s (7,680 tokens cached, `apc_a.log`); 32,006-token shared prompt: TTFT 2.65 s with 29,696 tokens cached (`wl_shared.log`); multi-turn: at 32,000 tokens turn 2 TTFT 0.79 s against 19.42 s for a cold prompt of the same size, at 64,000 tokens 1.34 s against 37.6 s (`multiturn.log`); 8 agents with growing contexts: mean TTFT of turns after the first 9.12 s, 72 % of prompt tokens cached (`wl_agents.log`). rc4c, B=16 (`out/f16/apc_grid_shared.log`, the log label says `b16rc3`, the server was the rc4c image per `srv/f16/info`): 32,006-token shared prompt, first request 12.64 s, then 2.49 s with 29,696 tokens cached. There is no ISL x OSL x concurrency grid with a shared salt in the validation files.

**Caveat for prefix-cache comparisons with rc3 (stated by the validation owner; not derived from these files):** rc3's host snapshot tier ignored `cache_salt`, so rc3's prefix-cache numbers were optimistic when salts differ; rc4c isolates salts (`apc_a.log`: a request with another salt shows `cached 0`). The cold grid above uses a unique salt per point for both images, so it is not affected.

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
- Requests that cannot speculate (frequency / presence / repetition penalty) are slower and less accurate than speculative requests (measured on the previous release, not re-measured).
- Not supported with speculation (HTTP 400): `logprobs` with a structured request, `min_tokens`, `bad_words` in streaming (pre-stream validation); other combinations of the previous release (`logit_bias`, `allowed_token_ids`, `min_p`) were not re-checked in rc4c.
- Disk sizes (checkpoint 476 GB, weight cache about 610 GB, kernel cache about 43 GB) are the previous release's figures; the rc4c caches were not re-measured.
