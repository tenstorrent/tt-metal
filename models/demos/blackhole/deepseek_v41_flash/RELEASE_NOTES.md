# DeepSeek-V4.1-Flash on Blackhole Galaxy: release notes

Release `0.24.0-0655530-4743e01` (vLLM 0.31.0), previous release `0.24.0-c9cfb00-ffd1a57`. Start-up commands: `GETTING_STARTED.md`. Test logs, bisect tables and long-form findings: `RELEASE_DETAILS.md`.

| | |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-0655530-4743e01` |
| Commits (branch `ssinghal/dsv4p1`) | tt-metal `0655530d603`, tt-inference-server `4743e018`, vllm-tt-plugin `ff71d29` |

## What is supported

| | B=16 | B=4 |
|---|---|---|
| Users | 16 | 4 |
| Max context | 262,144 tokens | 1,048,576 tokens |
| Image input | up to 16 images per prompt (1,024 tokens each at most) | same |
| Speculative decoding | k = 5, lossless, at every prompt length | same |
| Runtime spec (`assets/`) | `runtime_spec_b16_k5_262144_vision.json` | `runtime_spec_b4_k5_1048576_vision.json` |
| Start-up | about 25 minutes | about 31 minutes |

Serving features:
- OpenAI-compatible API (chat, completions, streaming); requests beyond the user count queue.
- On-device sampling: temperature, top-k (clamped to 32), top-p, per-request seed.
- Chunked prefill: token budget `max_num_batched_tokens` in the spec (67,584 in the default specs, 4,096 in the `*_apc.json` specs; the budget with APC on is to be decided after the device measurements).
- Prefix caching (APC): on by default (`DSV41_APC=1` and `enable-prefix-caching` in the specs and the catalog); `--no-enable-prefix-caching` turns it off. Images bypass it.
- Paged prefill: on by default (`DSV41_PF_PAGED=read`); read mode uses a bf16 KV pool instead of the fp8 pool, because APC on the paged prefill needs it (about +46 MiB per bank of DRAM). `DSV41_PF_PAGED_FP8=1` keeps the fp8 pool but turns APC off (one log line). `DSV41_PF_PAGED=0` restores the previous dense-table prefill with the fp8 pool.
- Not supported with speculation (HTTP 400): `logprobs`, structured output, `logit_bias`, `bad_words`, `allowed_token_ids`, `min_tokens`, `min_p`.

## Performance change against the previous release

Short prompts, batched (`vllm bench serve`, random prompts, 128 output tokens, 3 repeats). TTFT ms / TPOT ms / output tok/s. ISL 4k was not measured; 1,024 and 8,192 bracket it.

| | Previous release | This release | Change |
|---|---|---|---|
| B=16, 16 requests, ISL 1,024 | 3588 / 164.6 / 81.0 | 3626 / 162.8 / 82.1 | +1 % / -1 % / +1 % |
| B=16, 16 requests, ISL 8,192 | 24904 / 499.9 / 22.9 | 26208 / 522.7 / 21.8 | +5 % / +5 % / -5 % |
| B=4, 4 requests, ISL 1,024 | 2752 / 90.2 / 34.9 | 1590 / 67.5 / 48.0 | -42 % / -25 % / +37 % |
| B=4, 4 requests, ISL 8,192 | 16853 / 183.4 / 12.6 | 9059 / 120.9 / 20.2 | -46 % / -34 % / +61 % |

Model maximum, one request, 128 output tokens (previous release: speculation stopped at 32,768 tokens):

| | Previous release | This release |
|---|---|---|
| B=16, 250,028-token prompt | TTFT 311 s, TPOT 74 ms | TTFT 283 s, TPOT 31.6 ms |
| B=4, 1,000,028-token prompt | TTFT 1676 s, TPOT 98 ms | TTFT 1694 s, TPOT 36.7 ms |

The previous-release numbers at the model maximum come from an earlier image with the same prefill and the same 32,768-token speculation limit.

## What changed

- Speculative decoding now runs at any prompt depth (TPOT at 1M: 98 -> 37 ms).
- The 4-user 1M build no longer crashes on prompts of 524,288 tokens or more when APC is on.
- A prefill context of 16,384 tokens for short prompts on the 1M build (faster short-prompt TTFT).
- Shared tt_transformers sampler by default (`DSV41_SHARED_SAMPLER=0` restores the previous one).
- Index-key hand-off fixed for the column-sharded prefill key FIFO.
- APC and paged prefill on by default in the runtime specs and the catalog. The B=4 / 1M specs raise `trace_region_size` to 780,000,000 (paged read with the fp8 pool needed at least about 766 MB and 650 MB does not boot; the minimum with the bf16 pool, which adds about 46 MiB per bank, is being re-measured and may be higher). The B=16 / 262,144 trace region is unchanged and not yet measured with paged prefill. Throughput with APC on at budget 67,584 and at 4,096: to be measured (budget 4,096 made batched serving up to 2x slower in the previous measurement).

## Known issues

- B=16 at 262,144 tokens scores about 2 points lower on GSM8K than at 131,072 (about 93.5-95 % against 95.2-95.4 %); cause not identified.
- With the default token budget a long prompt stalls the other users' decoding while it prefills (32k prompt: 32.9 s; 9.1 s with budget 4,096).
- Concurrent long prompts are prefilled one after another (16 prompts of 250,000 tokens: 5,035 s in total).
- Requests that cannot speculate (frequency / presence / repetition penalty) lose accuracy on the B=4 1M spec (GSM8K about 52 %, against 97 % with speculation); send them to B=16.
- APC must not be enabled with the 131,072-token specs (they keep `DSV41_APC=0`, `DSV41_PF_PAGED=0`).

Next release: cheaper APC, host-tiered prefix cache on paged prefill (`PAGED_PREFILL_APC_DESIGN.md`), more in `FUTURE_WORK.md`.
