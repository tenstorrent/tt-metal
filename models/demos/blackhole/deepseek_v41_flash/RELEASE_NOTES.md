# DeepSeek-V4.1-Flash on Blackhole Galaxy: release notes

## This release: image `0.24.0-799a223-ffd1a57` (vLLM 0.31.0)

| | |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-799a223-ffd1a57` |
| tt-metal | `799a223987c18909ced4bab280e4bccf673a73b0` (branch `ssinghal/dsv4p1`) |
| tt-inference-server | `ffd1a57ceece25c02b668bbcc4682dba52dbd9da` (branch `ssinghal/dsv4p1`) |
| vllm-tt-plugin | `8cb999ebf8579e0733e378d852bb2139f7cac064` (branch `ssinghal/dsv4p1`) |
| Stack | vLLM 0.31.0, torch 2.11.0+cpu, transformers 5.12.1 |
| Previous image | `0.24.0-3dce07e-233b250` (vLLM 0.31.0, 16 users, 131,072 tokens, up to 4 images per prompt) |
| Build | an overlay on the previous image's build: the compiled tt-metal (C++), the Python environment and the vLLM install are reused unchanged (no file outside `models/` differs in tt-metal between the two commits); `models/`, the tt-inference-server tree, the plugin package, `model_spec.json` and `BUILD_INFO` are replaced from the new commits. `BUILD_INFO` in the image records the three shas and the note. |

## What this release adds

### 1. Long context with more users per galaxy

The budget that fits one galaxy is about **1M tokens of context per mesh row** (4 mesh rows): users x context per row.

| Serving configuration | Users | Context | Speculative decoding | Image input | Runtime spec |
|---|---|---|---|---|---|
| Default (unchanged) | 16 | 131,072 | k = 5 | optional | `assets/runtime_spec_b16_k5_131072.json` / `_vision.json` |
| **Long context, 16 users** | 16 | **262,144** | k = 5 | yes | `assets/runtime_spec_b16_k5_262144_vision.json` |
| **Long context, 4 users** | 4 | **1,048,576** | k = 5 | **no** (does not fit) | `assets/runtime_spec_b4_k5_1048576.json` |
| **Long context with images, 4 users** | 4 | **524,288** | k = 5 | yes | `assets/runtime_spec_b4_k5_524288_vision.json` |

Also measured on the same code (plain decode, not shipped as specs): 8 users at 524,288 tokens, and 8 users at 1,048,576 tokens does **not** fit (out of DRAM at warm-up).

What made it fit (all in this release):
- RoPE tables compacted from 7,168 to 768 B per position per chip (bit-identical), column-sharded from 262,144 positions.
- Dense prefill masks only where the dense path runs; chunked visibility.
- The prefill indexer's key FIFO, scores and visibility mask sharded over the 8 mesh columns (from 131,072 index entries, `DSV41_PIDX_SHARD`); the kv-table FIFO shifted in place in row pieces (`DSV41_PF_FIFO_CHUNK`).
- The decode indexer ranks each user's score row on its own when the concatenated rows would not fit a core's L1.
- Server warm-up reworked for long contexts: arange tables shared across layers, the index-key hand-off in 64K-entry pieces written in place, one eager step before each trace capture (prefill, decode, spec) so no program is compiled inside a capture.
- Still replicated: the prefill kv table (every mesh column holds the single KV head; sharding it is a separate piece of work).

### 2. Speculative decoding at long context

Spec decode (k = 5, lossless) now runs at 262,144 tokens (16 users) and 1,048,576 tokens (4 users) instead of stopping at 131,584. At 1M and 4 users: 3.31 accepted drafts per round, 41.2 tok/s per user, 1.81x plain decode on the demo (short GSM8K prompts). `DSV41_SPEC_MAX_CTX` is the final word on the context cap (policy and vLLM guard); the runtime free-DRAM check when the runners are built still applies. The verify step scores the key slab in pieces (`DSV41_SPEC_SCORE_PIECE`) and uses the matmul indexer backend above 131,584 tokens (forced at build; the fused backend is 75.3 vs 43.8 ms/token at 1M).

### 3. Image input

- Up to **16 images per prompt** (was 4). The model card sets no limit; this is the server's `--limit-mm-per-prompt {"image": 16}`. A request with more is refused with an OpenAI-style HTTP 400 error:
  `{"error": {"message": "At most 16 image(s) may be provided in one prompt. (parameter=image)", "type": "BadRequestError", "param": "image", "code": 400}}` and the server keeps serving.
- An image is at most `max_image_tokens` = 1,024 tokens (31 x 31 token grid, about 1302 x 1302 pixels, 8,649 ViT patches); larger images are resized, keeping the aspect ratio.
- Works together with speculative decoding and with the long-context configurations above that list image input: verified at 16 users / 262,144 tokens and 4 users / 524,288 tokens.
- **Does not fit at 4 users and about 1M tokens**: with the vision tower loaded (about 22 MB per DRAM bank) the prefill trace warm-up needs one contiguous 21 MB block per bank for a 160 MiB buffer and the largest block left is 17-19 MB (79-108 MB per bank free in total). Tried: 1,048,576 tokens at chunk 512 and 256, and 1,000,448 tokens. Use the 4-user text spec at 1M tokens or the 524,288-token vision spec.

### 4. Fixes

- **Plain decode no longer depends on the other users of its mesh row.** The fused rows-layout RoPE kernel rotated every row by the angle of the first row's position, so in a ragged batch each user's compressed latent was written rotated by its first row-mate's position (logit shifts up to 3.5 on 4 layers and 9.2 on 40 when a row-mate moved by one token). The per-row path is the default now (`DSV41_ROPE_ROWS_FUSED=1` restores the old kernel, about 4.6 ms/token faster in plain decode at 16 users). Speculative decoding already used per-row RoPE and is unchanged.
- **The engine no longer dies when more requests than slots are queued** (vLLM 0.31 admits waiting requests against `max_num_active_reqs`; the scheduler's prefill-only pass only lowered `max_num_running_reqs`). Before, the 5th concurrent request of a 4-slot server, or the 17th of a 16-slot server, killed the engine on an assertion.
- Speculative verify no longer pads the score rows of a ragged batch to 32 rows per user (768 MiB at 24 rows x 262,144 entries ran the 1M build out of DRAM).

### 5. vLLM stack

vLLM 0.31.0 with the tt plugin ported to it: request-validation refusals (logprobs / structured outputs under speculation, `prompt_embeds`, `prompt_logprobs`, block-output limits) answer with an OpenAI-style HTTP 400 instead of the bare empty 500 that 0.31 returns for a `ValueError`; the scheduler follows 0.31's admission limits (fix above). Installed with the plugin's `docs/install-vllm-tt.sh`.

## Performance

Method: GSM8K 8-shot prompts (about 700 tokens), 256 output tokens, `vllm bench serve` through tt-inference-server's `benchmarks` workflow run from the image, speculative decoding k = 5, one persistent server per configuration. TPOT in ms, TTFT in s, means; "tok/s" is total output throughput (greedy). Full tables are in `GETTING_STARTED.md`.

| Configuration | Concurrency | Greedy TPOT | Greedy TTFT | Sampled TPOT | tok/s |
|---|---|---|---|---|---|
| 16 users, 131,072 tokens (default) | 1 / 2 / 4 / 8 / 16 | 28.5 / 32.2 / 38.0 / 49.1 / 70.4 | 1.03 / 1.33 / 1.38 / 1.74 / 2.40 | 28.0 / 31.8 / 37.5 / 49.6 / 68.0 | 30.8 / 53.2 / 85.5 / 125.4 / 196.6 |
| 16 users, 262,144 tokens, images on | 1 / 2 / 4 / 8 / 16 | 26.4 / 34.8 / 36.1 / 53.4 / 71.0 | 1.05 / 1.26 / 1.44 / 1.74 / 2.56 | 29.3 / 35.2 / 39.0 / 52.2 / 72.5 | 32.8 / 50.0 / 94.4 / 120.7 / 189.4 |
| 4 users, 1,048,576 tokens | 1 / 2 / 4 | 29.8 / 35.2 / 45.5 | 1.49 / 1.74 / 1.86 | 30.9 / 39.6 / 44.8 | 28.1 / 47.3 / 74.5 |
| 4 users, 524,288 tokens, images on | 1 / 2 / 4 | 31.8 / 33.9 / 40.4 | 1.26 / 1.61 / 1.79 | 31.3 / 33.3 / 43.5 | 27.3 / 48.2 / 81.2 |

Against what the previous image `0.24.0-3dce07e-233b250` shipped (16 users, 131,072 tokens: greedy TPOT 27.7 / 33.7 / 43.7 / 52.5 / 80.1 ms, TTFT 1.24 / 1.51 / 1.54 / 1.96 / 2.75 s, 168.3 tok/s at 16 requests): the default configuration of this release is the same or faster at every concurrency (TPOT 0.8 ms higher at one request, 9.7 ms lower at 16; TTFT lower throughout; throughput up to 17 % higher at 16 requests), and the 262,144-token server has the same speed within the run-to-run noise (3-4 ms). The previous image's numbers were measured on other hosts, so part of the difference can be the hosts; an A/B run of the old image on the same host was not done. The two fixes of this release do not change speculative decoding speed (the 262,144-token server measured 29.95 / 33.0 / 41.4 / 49.8 / 73.9 ms greedy on the same configuration before them).
The 1M-token and 524,288-token servers are slower than the 16-user ones at the same concurrency (every decode step and every prefill chunk scores the index entries of the whole context; TTFT +0.3-0.4 s). A 100,020-token prompt prefills in 115-117 s at 16 users / 262,144 tokens, 132-134 s at 4 users / 524,288, and 158-159 s at 4 users / 1,048,576 tokens.

## Accuracy

GSM8K 8-shot (`dsv41_gsm8k_cot`, all 1319 questions, tt-inference-server `evals` workflow from the image, temperature 1.0, flexible-extract, standard error about 0.6):

| Server | Score |
|---|---|
| 16 users, 131,072 tokens | 95.22 % |
| 16 users, 131,072 tokens, image input on | 95.38 % |
| 4 users, 1,048,576 tokens | 94.92 % |
| 16 users, 262,144 tokens, image input on | **93.56 % and 93.48 %** (two runs on the same server) |
| Previous image, 16 users, 131,072 tokens | 95.45 % |

The 16-user 262,144-token server scores about 2 points lower than the other configurations (about 3 standard errors, reproduced on a second run). It passes the workflow's gate (within 5 % of the model card's 93.0). Checks made to find the cause: vision on or off makes no difference (131,072 tokens: 95.38 % with, 95.22 % without; 262,144 tokens: 93.5 % with, 93.18 % without and with the column-sharded RoPE tables switched off), the new code is not the cause at the default context (95.22 % at 131,072 tokens on this image), and 163,840 tokens scored 94.54 %. **The cause is not identified yet** (a plain-decode run without speculation at 262,144 tokens was still running when these notes were written). The 262,144-token server passes every functional test and has the speed of the default configuration.

## Testing done on this image

Each configuration was served once and every test ran against that one server (no restarts between tests). Scripts: `tools/serving_tests/` (README there).

| Test | 16 users / 262,144 | 4 users / 1,048,576 | 4 users / 524,288 (images) |
|---|---|---|---|
| Sanity (sampled, streaming, 8 concurrent questions) | pass | pass | pass |
| 20,020-token prompt, code found | 4817, 23.5 s | 4817, 32.3 s | 4817, 27.1 s |
| 100,020-token needle at the end / at the start | 4817 / 4817 (115.7 / 117.1 s) | 4817 / 4817 (157.7 / 159.2 s) | 4817 / 4817 (132.4 / 134.0 s) |
| Image checks (1 / 2 images, 4 image + 4 text at once, streaming) | pass | not applicable | pass |
| 16 images in one prompt, one maximum-size image, 17 images -> HTTP 400 | pass | not applicable | pass |
| 16-concurrent image stress, 48 random images (number / colours) | 16/16, 48/48, 48/48 | not applicable | 16/16, 48/48, 48/48 |
| Overload: 4 x slots requests at once, two waves | 64/64 on 16 slots, both waves | 16/16 on 4 slots, both waves | 16/16 on 4 slots, both waves |
| Serving API (23 checks) | 23/23 | 23/23 | 23/23 |
| Benchmarks (greedy, sampled) | done | done | done |
| GSM8K 8-shot | 93.56 / 93.48 | 94.92 | not run |

The default 131,072-token configuration (16 users, no images) was re-measured too: sanity pass, the 20,020-token prompt (4817), the 100,020-token needle at the end and at the start (4817 / 4817, prefill 97.3 / 99.5 s), benchmarks, and GSM8K (95.22 %; 95.38 % with image input on).
Model-level device test of the RoPE fix (`tests/test_decode_batch_sensitivity.py`, 4 layers): the logits of a user are bit-identical (24 of 24 steps) whatever the other users of its mesh row do; the 40-layer variant shows run-to-run noise in the serving environment of the test wrapper (MoE fp32 accumulation / bfp8 flags) and was not conclusive there.

## Known issues

- **16 users at 262,144 tokens score about 2 points lower on GSM8K** (93.5 % against 95.2-95.4 % at 131,072 tokens), cause not identified (see "Accuracy"). Use the default 131,072-token configuration where that accuracy matters.
- **Image input does not fit at 4 users and about 1M tokens** (see "Image input"); it works up to 524,288 tokens at 4 users and 262,144 tokens at 16 users.
- **The plain-decode RoPE fix is verified bit-exact on a 4-layer model** (24 of 24 steps on 7 batch compositions, `tests/test_decode_batch_sensitivity.py`); the same test on the full 40-layer model in this tree still shows differences between runs of the identical batch composition (max logit difference 2.4-3.3, deterministic), so independence from the other users of a mesh row is not shown at 40 layers on this tree. Speculative decoding (all shipped configurations) does not use the plain-decode path.
- Full GSM8K was not run on the 4-user 524,288-token configuration.

## Tests still to run

- **Decode speed at deep context.** The benchmarks above use prompts of about 700 tokens. TPOT with 32k, 100k and 200k tokens already in the context (262,144-token and 1M servers) has not been measured. The needle tests check retrieval, not speed.
- **A prompt close to the context limit.** The longest prompt run is 100,020 tokens. A prompt of about 1,000,000 tokens (4 users, 1,048,576 context) and one of about 250,000 tokens (16 users, 262,144 context) have not been run. The 1M prefill is estimated at about 12 minutes.
- **Request controls under speculation.** `min_p`, `logit_bias`, `bad_words`, `allowed_token_ids`, `min_tokens`, `logprobs` and structured output have not all been tested on a live server. The platform code says `logprobs`, structured output, `logit_bias`, `bad_words`, `allowed_token_ids` and `min_tokens` are refused with HTTP 400 and `min_p` and the penalties are served without speculation. The earlier guide said `min_p` is refused. The tables of this document follow the live tests where they exist.

## Not in this release

- Prefix caching (APC) and chunked-prefill alignment: implemented on branch `ssinghal/dsv4p1-apc` (TTFT 5.2 -> 2.1 s on a shared-prefix burst), not merged.
- fp8 prefill kv table, sharding the prefill kv table over the mesh columns.
- 8 users at 1,048,576 tokens (does not fit).
- `logprobs`, structured output, `logit_bias`, `bad_words`, `allowed_token_ids` and `min_tokens` with speculative decoding (refused with HTTP 400, because the model-owned device sampler of the speculative round does not apply them; answering without them would silently change the request).

## Upgrade notes

- New runtime specs for the two long-context configurations (above); the default 131,072-token specs are unchanged. Use `SPEC=...` in step 3 of `GETTING_STARTED.md`.
- Startup of the long-context servers takes about 25-40 minutes (weight cache and kernel cache populated).
- Environment variables added or changed: `DSV41_SPEC_MAX_CTX` (final word on the spec context cap), `DSV41_VLLM_SPEC_GUARD` (0 disables the vLLM-side guard), `DSV41_ROPE_ROWS_FUSED`, `DSV41_KEY_EXPORT_PIECE`, `DSV41_SPEC_SCORE_PIECE`, `DSV41_DEC_PRECAPTURE_WARM`, `DSV41_SPEC_PRECAPTURE_WARM`, `DSV41_PIDX_SHARD`, `DSV41_PF_FIFO_CHUNK`, `DSV41_PF_PRECAPTURE_WARM`.
