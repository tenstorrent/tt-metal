# DeepSeek-V4.1-Flash on Blackhole Galaxy: release notes

Release `0.24.0-0655530-4743e01` (vLLM 0.31.0). Test logs, bisect tables and long-form findings: `RELEASE_DETAILS.md`. Setup: `GETTING_STARTED.md`.

| | |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-0655530-4743e01` |
| tt-metal | `0655530d603` (branch `ssinghal/dsv4p1`) |
| tt-inference-server | `4743e018` (branch `ssinghal/dsv4p1`) |
| vllm-tt-plugin | `ff71d29` (branch `ssinghal/dsv4p1`) |
| Previous release | `0.24.0-c9cfb00-ffd1a57` |

## Highlights

- **16 users at 262,144 tokens and 4 users at 1,048,576 tokens**, speculative decoding k = 5, image input (up to 16 images per prompt).
- **Speculative decoding now runs at any prompt depth.** It used to stop at 32,768 tokens: single-request TPOT at 1,000,028 tokens fell from 98 to 37 ms, at 250,028 tokens from 74 to 31.5 ms.
- **1M server no longer crashes on long prompts with APC on** (the prefix-cache snapshot needed one 512 MiB DRAM buffer; it now reads 16 MiB pieces).
- **Prefix caching (APC) and chunked prefill are available as opt-in specs** (`*_apc.json`). The default specs run with APC off and a token budget of 67,584, the previous release's settings.
- **Shared tt_transformers sampler is the default** (plain decode and speculative verify). `top_k` above 32 is clamped to 32.
- **Catalog defaults match the specs:** a launch without a runtime spec also gets APC off and budget 67,584.
- **Batched serving is back to the previous release's speed** at 16 users and faster at 4 users (tables below).

## Configurations

| Configuration | Users | Context | Spec decode | Images | Runtime spec (`assets/`) |
|---|---|---|---|---|---|
| **Default** | 16 | 262,144 | k = 5 | yes | `runtime_spec_b16_k5_262144_vision.json` |
| **Long context** | 4 | 1,048,576 | k = 5 | optional | `runtime_spec_b4_k5_1048576.json` / `_vision.json` |
| Smaller context | 16 | 131,072 | k = 5 | optional | `runtime_spec_b16_k5_131072.json` / `_vision.json` |
| APC + chunked prefill (opt-in, budget 4,096) | 16 | 262,144 | k = 5 | yes | `runtime_spec_b16_k5_262144_vision_apc.json` |
| APC + chunked prefill (opt-in, budget 4,096) | 4 | 1,048,576 | k = 5 | yes | `runtime_spec_b4_k5_1048576_vision_apc.json` / `_apc.json` |

4 users at 524,288 tokens (`runtime_spec_b4_k5_524288_vision.json`) stays in the assets, not recommended (see Known issues). 8 users at 1,048,576 tokens does not fit in DRAM.

## What is new since `0.24.0-c9cfb00-ffd1a57`

Both long-context configurations, image input, speculative decoding k = 5 and opt-in APC / chunked prefill already shipped in `c9cfb00`. New in this release:

| Change | Detail |
|---|---|
| Speculation above 32,768 tokens | `DSV41_VLLM_SPEC_ISL_MAX` default 32768 -> unlimited. TPOT 98 -> 37.4 ms (1,000,028 tokens, 4 users), 74 -> 31.5 ms (250,028 tokens, 16 users) |
| Small prefill context at max context >= 524,288 | Second prefill trace at a 16,384-token context for short prompts. Short-prompt TTFT at 4 users / 1M: 0.73 / 0.68 / 0.74 s at 1 / 2 / 4 requests (was 0.90 / 0.83 / 0.87 s). Off: `DSV41_PF_SMALL_BUCKET=0` |
| Shared sampler default | `DSV41_SHARED_SAMPLER` and `DSV41_SHARED_SAMPLER_SPEC` default 1 (0 = previous sampler). Sampler time 1.2-1.9 ms per step against 1.5-3.6 ms |
| Defaults: APC off, budget 67,584 | Default specs and the catalog. Opt-in `*_apc.json` keep the old defaults (APC on, budget 4,096) |
| APC snapshot at >= 524,288 tokens | Fixed: no more device out-of-memory on the 1M server (`SNAP_PIECE_MIB` = 16) |
| Index-key hand-off offset | Fixed for the column-sharded prefill key FIFO (offset was negative for a 100,033-token prompt; decode read zero keys for early entries) |
| Long-prompt measurements | 1,000,028-token prompt completes at 4 users; 250,028-token prompt at 16 users (below) |

## Performance

Previous release `c9cfb00` against this release with the default specs. `vllm bench serve`, random prompts, OSL 128, 3 repeats with a different seed per repeat, images on, k = 5; 16 users measured on host .45, 4 users on host .46. TTFT ms / TPOT ms / output tok/s.

| Config | Row | Previous release | This release | Change |
|---|---|---|---|---|
| 16 users / 262,144 | 16 x ISL 1,024 | 3588 / 164.6 / 81.0 | 3626 / 162.8 / 82.1 | +1.1 % / -1.1 % / +1.3 % |
| 16 users / 262,144 | 16 x ISL 8,192 | 24904 / 499.9 / 22.9 | 26208 / 522.7 / 21.8 | +5.2 % / +4.6 % / -5.0 % |
| 16 users / 262,144 | 1 x ISL 8,192 | 8547 / 47.4 / 8.8 | 8353 / 44.1 / 9.2 | -2.3 % / -7.1 % / +4.5 % |
| 4 users / 1,048,576 | 4 x ISL 1,024 | 2752 / 90.2 / 34.9 | 1590 / 67.5 / 48.0 | -42 % / -25 % / +37 % |
| 4 users / 1,048,576 | 4 x ISL 8,192 | 16853 / 183.4 / 12.6 | 9059 / 120.9 / 20.2 | -46 % / -34 % / +61 % |
| 4 users / 1,048,576 | 1 x ISL 8,192 | 12138 / 45.2 / 7.2 | 6474 / 45.2 / 10.5 | -47 % / +0.1 % / +46 % |

- 16 x ISL 8,192 is 5 % off the previous release; that is inside the previous release's own repeat spread (TTFT 21 %) and the host-to-host difference (up to 7 %).
- Short prompts (GSM8K-style, about 700 tokens in, 256 out; measured on an earlier image), TPOT ms at concurrency 1 / 2 / 4 (/ 8 / 16): 16 users / 262,144: 26.4 / 34.8 / 36.1 / 53.4 / 71.0; 4 users / 1,048,576: 29.8 / 35.2 / 45.5.

Long prompts, single request, 128 output tokens, speculation on:

| Config | Prompt | TTFT | TPOT |
|---|---|---|---|
| 4 users / 1,048,576 | 1,000,028 tokens | 1694 s | 36.7 ms |
| 4 users / 1,048,576 | 599,955 tokens | 1002 s | n/a |
| 16 users / 262,144 | 250,028 tokens | 283 s | 31.6 ms |
| 16 users / 262,144 | 100,020 tokens | 116 s | n/a |

Opt-in APC + chunked prefill (`*_apc.json`): a 100,000-token shared prefix answers in 42 s on a cache hit against 112 s cold (16 users); the worst decode stall while a 32k prompt prefills is 9.1 s (budget 4,096) against 32.9 s (budget 67,584, the default). The cost: batched serving at 16 users is up to 2x slower (16 x ISL 8,192 TTFT 49.8 s against 24.9 s), APC adds about 180 ms per 1k-token prefill call.

## Accuracy

GSM8K 8-shot, `dsv41_gsm8k_cot`, tt-inference-server `evals` workflow, 1319 questions (standard error about 0.6). These workflow scores were measured on earlier images of the same model code, none on this exact image:

| Server | Score |
|---|---|
| 16 users, 262,144 tokens, images on (default spec, previous release `c9cfb00`) | 94.09 % |
| 16 users, 262,144 tokens, images on (APC on, previous release `c9cfb00`) | 95.60 % |
| 16 users, 131,072 tokens | 95.22 % |
| 4 users, 1,048,576 tokens | 94.92 % |

This release, sampled 256-question checks: 94.9 / 96.5 / 93.8 % on the 16-user default spec; 96.9-97.7 % on the 4-user 1M specs and 96.5 / 97.3 / 97.7 % on the 16-user `_apc` spec (candidate images with the same code). A full workflow run on this exact image was not made. 100,020-token needles at the start and at the end answer correctly on both configurations.

## Known issues

- **16 users at 262,144 tokens score about 2 points lower on GSM8K** than at 131,072 tokens (about 93.5-95 % against 95.2-95.4 %). Cause not identified.
- **APC is off by default.** With APC on and budget 4,096, batched serving at 16 users is 14-100 % slower than the previous release (APC: about 180 ms of snapshot copy per 1k-token prefill; budget 4,096: prompts prefilled in small pieces). Turning both off gives the previous speed back but means no prefix caching and, for a long prompt, a decode stall of other users (32.9 s for a 32k prompt).
- **Concurrent long prompts are prefilled one after another** and other users' decode steps wait between pieces: 16 simultaneous 250,000-token prompts took 5,035 s (TPOT about 10 s); 4 simultaneous 1,000,000-token prompts 6,751 s.
- **Decode is slower at deep context:** 31.5 ms (250,000 tokens) and 37.4 ms (1,000,000 tokens) per token against about 23 ms at 16k.
- **Plain decode (requests that cannot speculate: frequency / presence / repetition penalty) loses accuracy on the long-context 4-user specs:** GSM8K 51.6-52.7 % at 1,048,576 tokens and 69 % at 524,288, against 97 % with speculation. Keep such requests to the 16-user 262,144-token spec. Cause not identified.
- **APC must not be enabled with the 131,072-token specs** or with `DSV41_PIDX_SHARD=0` (restores answer wrongly). APC is skipped for prompts that contain images.
- **Refused with HTTP 400 under speculation:** `logprobs`, structured output, `logit_bias`, `bad_words`, `allowed_token_ids`, `min_tokens`, `min_p`.
- **Requests beyond the slots queue** and are served in order; more than 16 images per prompt is refused with HTTP 400.

## Next release

Cheaper APC (asynchronous snapshots: branch `ssinghal/dsv4p1-apc-fast`), packing several prompts into one prefill call, and a host-tiered prefix cache on paged prefill (`PAGED_PREFILL_APC_DESIGN.md`, branch `ssinghal/dsv4p1-paged-apc-design`). Full list: `FUTURE_WORK.md`.

## Upgrade notes

- Pick the runtime spec with `SPEC=...` in step 3 of `GETTING_STARTED.md`; a launch without a spec uses the catalog, which now matches the default specs.
- Startup of the long-context servers takes about 25-40 minutes on a cold cache (weights and kernels).
- New or changed environment variables: `DSV41_VLLM_SPEC_ISL_MAX` (now unlimited), `DSV41_SHARED_SAMPLER` / `DSV41_SHARED_SAMPLER_SPEC` (default 1), `DSV41_PF_SMALL_BUCKET` and `DSV41_PF_SMALL_MIN_CTX`, `DSV41_APC` (1 = prefix caching, needs `enable-prefix-caching`), `DSV41_IDX_SWITCH` (262144), `DSV41_SPEC_MAX_CTX`.
