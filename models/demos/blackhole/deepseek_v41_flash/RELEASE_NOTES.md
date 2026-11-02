# DeepSeek-V4.1-Flash on Blackhole Galaxy: release notes (B=8)

Release `0.24.0-785f0df-2406dcb` (internal name rc4c, vLLM 0.31.0). Start-up commands: `GETTING_STARTED.md`. Test logs and long-form findings: `RELEASE_DETAILS.md`. Optimization-by-optimization history and switches: `OPTIMIZATIONS.md`.

| | |
|---|---|
| Image (target name, **not pushed yet**) | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-785f0df-2406dcb` |
| Local image | `dsv41-flash-local:rc4c-785f0df9b28-2406dcb1-a83b4c6` (id `ae6ac9b615f5`, built on .34; also tagged locally with the ghcr name above). The ghcr tag is unchanged, but its content now includes the plugin tool-calling fix (plugin `a83b4c6` instead of `d7b6901`). |
| Commits | tt-metal `785f0df9b28` (branch `ssinghal/dsv4p1-rc3`; the release notes live in later docs-only commits of that branch), tt-inference-server `2406dcb15` (`ssinghal/dsv4p1-rc3-ttis`), vllm-tt-plugin `a83b4c6` (tool-calling fix, on top of `d7b6901`) (`ssinghal/dsv4p1-rc3-plugin`) |
| Offline copy | none (the image with the fix exists as local tags on host .34 only; `docker save` is needed to move it to other hosts) |

This release is **B=8 only**. B=16 and B=4 are not part of this release (the catalog still contains their entries; they were not validated for it).

## What is supported

| | B=8 (catalog default) |
|---|---|
| Users | 8 (4 mesh rows x 2) |
| Max context | 1,048,576 tokens |
| Select with | `--model deepseek-ai/DeepSeek-V4.1-Flash --tt-device blackhole_galaxy` (no `--impl`) |
| Image input | up to 250 images per prompt (`limit-mm-per-prompt`) |
| Speculative decoding | adaptive k = 3 / 5 (`DSV41_VLLM_SPEC_KS=3,5`), draft model owned by the plugin |
| Chunked prefill budget | `max_num_batched_tokens` 32,768, `long_prefill_token_threshold` 4,096 |
| Runtime spec file | none needed: the baked catalog (`model_spec.json` in the image) is used |
| Start-up (weight and kernel caches warm, from `docker run` to `/health` 200) | 1,274-1,635 s (21-27 min) |

Start-up times: `dsv4-logs/rc4c_val/srv/*/info` (`SERVER_READY ... boot N s`): B=8 holds a1 1,635 s, a2 1,274 s, a3 1,304 s, r8 1,317 s, s8 1,274 s. The first start with empty caches is much longer (not measured).

## What is in rc4c

Serving behaviour that is on by default in the catalog:
- **Structured outputs on the speculative path (M1 mode).** `tool_choice` required / named, `strict: true`, `response_format` `json_schema` / `json_object`, and guided regex / choice / grammar are served; a structured row decodes without drafts under an on-device grammar mask while the other rows keep speculating. Streaming requests with an unsupported combination (logprobs, min_tokens, bad_words) get HTTP 400 before the first event; stream headers go out after the first output. Switches (set in the catalog): `DSV41_STRUCTURED_SPEC=1`, `VLLM_TT_STRUCTURED_SPEC=1`, `DSV41_INTRACE_SAMPLE=1`, `DSV41_STRUCTURED_M2=0`, `VLLM_TT_STRUCTURED_M2=0`.
- **Image-cut fix** (`DSV41_PIECE_LAG=1`): a prefill piece that ends off a chunk boundary or inside an image is cut back, and the next piece resumes at the carried end instead of recomputing from position 0 (device A/B by the branch owner; this rc4c validation did not repeat it).
- **Prefix caching on a 512-token grid, block-paged on device at B=8** (`DSV41_APC_GRID=512`, `DSV41_APC_PAGED=1` for the B=8 entries). A repeated or shared prefix is restored instead of recomputed (numbers below). Hits are bit-identical to a cold prefill in the owner's device test.
- **Prefix cache and images**: image content hashes are part of the cache key (`DSV41_APC_IMAGES=1`); vLLM's own prefix-cache reads are skipped for requests with images (plugin). **`cache_salt` is honoured**: a request with another salt never reuses a cached prefix.
- **Cross-context vision fix**: a stale image mask in one prefill context could leak into the next text-only window (text chats answered with grounding text in a mixed image + text run); fixed on the code path.
- **Tool-calling fix (plugin `a83b4c6`).** Invalid tool results are refused with HTTP 400, streaming and non-streaming: an unknown `tool_call_id`, a duplicate id, or a missing / empty id on a `role: tool` message. JSON scalars (int / float / bool / null) for tool-schema properties declared without a `type` are returned as JSON scalars, not strings. Unchanged: parallel results in reverse order pair correctly, forced named / required `tool_choice` work with structured outputs, `tool_choice: none`, raw user `tool_result` content blocks are rejected with HTTP 400, and DSML-shaped text inside a tool result does not create spurious calls. Remaining limits: partial results are accepted (fewer results than calls), typed schemas are unchanged, and DSML text inside string values is not escaped. The fix can be disabled with `VLLM_TT_TOOL_RESULT_VALIDATE=0` (id validation) and `VLLM_TT_UNTYPED_SCALARS=0` (scalar typing). The HTTP 400 for an empty `tools` array is upstream vLLM behaviour and is unchanged. The fix is plugin-frontend only and does not change performance (no table was re-measured). CPU tests: 39 new tests pass (`tests/test_tool_result_validation.py`); the plugin test directory (without the device tests) gives 2225 passed and 4 failed, the 4 `test_num_available_blocks` failures also occur without the fix.
- `usage.prompt_tokens_details.cached_tokens` is returned (`enable-prompt-tokens-details` in the catalog).
- Earlier B=8 work included in this image: fp8 KV serving with APC (`DSV41_FP8_SERVING`, paged fp8 prefill), mHC PRE16 / X2LITE, column-TP shared expert, fused MoE tail, post-SDPA glue, fused2 router, key-slab sharded indexer (`DSV41_IDX_KEY_SHARD`), host-side round work (`DSV41_HOSTOPT_*`), drafter sets A and B, row-aware / Up=2 prefill, `DSV41_UNI_TOPO=linear` (fix of an intermittent combine deadlock), MoE core pin (`MOE_COMPUTE_PIN_EAST_X=7`).

Available but off (not part of this release): APC delta snapshots, decode-output caching, cross-row paged hits, the short-prompt prefill lever, structured M2 (masked verify with drafts). They are default off in code and not in the catalog; see `OPTIMIZATIONS.md`.

## Performance

All numbers are rc4c, from `dsv4-logs/rc4c_val/`. Method: tt-inference-server `--workflow benchmarks` (`run.py --workflow benchmarks --tools vllm`, client-only against the running server), the workflow's own random-prompt sweep with one unique `cache_salt` per sweep point (`llm_module/drivers/vllm.py` of the catalog branch), so the rows are cold-prefix measurements (no prefix-cache hit). The grid itself ran from `wt/ttis_bench_grid`, which I did not diff against the catalog branch. Cells are `TTFT s / tok/s per user`. Every cell is a single run (n = 1 at c1, one wave at 262,144), so differences of several percent between neighbouring cells are noise. Sources: grid `grid/final_cmp_g8.txt` (rc4c columns); the tables are generated by `mk_perf_tables_rc4c.py` into `perf_tables_rc4c.md`.

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

**Beyond 262,144 tokens.** The workflow grid was not run beyond ISL 262,144 for rc4c (the 524,288 / 786,432 cells are not reported). The 1M context is covered by the needle test only: 1,039,901 prompt tokens found, 596 s (`res_a2/needle1m.log`).

## Shared-prefix rows (rc4c B=8; prefix-cache workloads, not the cold grid)

These use a shared prompt on purpose, so they hit the prefix cache (`res_a1/`): 8,192-token prompt repeated: TTFT 5.67 s cold -> 0.67 s (7,680 tokens cached, `apc_a.log`); 32,006-token shared prompt: TTFT 2.65 s with 29,696 tokens cached (`wl_shared.log`); multi-turn: at 32,000 tokens turn 2 TTFT 0.79 s against 19.42 s for a cold prompt of the same size, at 64,000 tokens 1.34 s against 37.6 s (`multiturn.log`); 8 agents with growing contexts: mean TTFT of turns after the first 9.12 s, 72 % of prompt tokens cached (`wl_agents.log`). A request with another `cache_salt` shows `cached 0` (`apc_a.log`). There is no ISL x OSL x concurrency grid with a shared salt.

## Accuracy and checks (rc4c, B=8, 1M build, catalog default)

All results in this section were measured on the rc4c image with plugin `d7b6901`, before the tool-calling fix; the fix (plugin `a83b4c6`) touches only the plugin's tool-call handling (`entrypoints.py`, `tool_arg_typing.py`, `tool_result_validation.py`). Results for the tool-calling fix: see the last row of the table.

| Check | Result | Source |
|---|---|---|
| GSM8K (lm-eval, flexible-extract) | 96.21 % (stderr 0.53) | `res_a2/gsm.log` |
| GPQA diamond (zero-shot CoT, flexible-extract) | 78.79 % (stderr 2.91) | `res_a2/gpqa.log` |
| Needle in a haystack | found at 65,557 / 131,085 / 262,142 / 599,956 / 1,039,901 prompt tokens (1M run: 596 s) | `res_a2/needle_*.json`, `needle1m.log` |
| Tool-calling checks and fix tests (plugin `a83b4c6`) | tool suite 35 PASS, 0 FAIL; structured matrix 28/28; extra tool-id and typing checks 33 PASS, 0 FAIL (stream and non-stream); GSM8K sanity, 50 questions, flexible-extract 1.00 (50/50), strict-match 0.76. These ran on the rc4c `d7b6901` image with the patched plugin `src/` mounted read-only; the `a83b4c6` image holds the same three files (sha256 equal), so no separate boot test was run; the GSM8K sanity run was on the fixed plugin (the GSM8K / GPQA / needle results above stay as measured on the `d7b6901` image) | `dsv4-logs/toolfix/RESULTS.md` (`res_t1/toolchecks.log`, `matrix.log`, `res_t2/extra.log`, `gsm.log`) |
| Structured-output matrix (json_schema / regex / choice / tools, plain and stream, greedy and sampled) | 28/28 on two reruns (`res_a3/matrix_rerun1.log`, `matrix_rerun2.log`) and 28/28 mixed with background load (`res_a1/matrix_mixed.log`); **the first run was 27/28** (one `json_schema/plain/sampled` cell failed, `res_a1/matrix.log`) | files named |
| API checks | 23/23 | `res_a1/api.log` |
| Pre-stream validation | HTTP 400 for unsupported streaming combinations; supported ones HTTP 200 | `res_a1/prestream.log` |
| Images | 250 images in one prompt (48,779 tokens, 49.4 s) found 250/250; 100 images 100/100; 33 repeated image prompts all correct (no image prefix hit at these sizes: `cached 0`) | `res_a1/many250.log`, `apcimg_repeat.log` |
| Image + text soak (leak test) | 1,464 requests with the short-prompt lever on / 1,485 with it off, 0 bad, 0 errors | `res_a3/leak.log` |
| `cached_tokens` accounting | `CACHED_TOKENS_OK` (11,264 of 11,626 on a repeat) | `res_a1/cached.log` |

## Known limitations

- Tool calling: partial tool results (fewer results than calls) are accepted, typed tool schemas are unchanged, and DSML-shaped text inside string values is not escaped.

- Greedy output is not bit-reproducible across batch composition, or when adaptive speculation changes the verify width k (bf16 accumulation differs). The differences occur at near-tie tokens and the text stays coherent; earlier releases behave the same way (`dsv4-logs/det_rc4c/RESULTS.md`).
- For sequential repeatability set `DSV41_VLLM_SPEC_ADAPT=0` (fixed k = 5) or `DSV41_VLLM_SPEC_FORCED_K` in the catalog entry (mount a modified `model_spec.json`; the throughput cost was not measured). Concurrent and mixed batches can still differ at near-tie tokens.
- One structured-matrix cell failed once (27/28, above) and passed on both reruns; the cause was not investigated.
- Image prompts: no prefix-cache hit was observed in the validation (small prompts below the cached block size); vLLM's own prefix reads are skipped for requests with images.
- A long prompt stalls the other users' decoding while it prefills (token budget); concurrent very long prompts prefill one after another.
- Streaming requests with `logprobs`, `min_tokens` or `bad_words` are refused with HTTP 400 (pre-stream validation); other unsupported combinations were not checked in the rc4c validation.
- Cache disk sizes were not measured for rc4c (earlier images: checkpoint 476 GB, weight cache about 610 GB, kernel cache about 43 GB).
