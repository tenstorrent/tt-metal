<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-S-2.1 on p150x4 (TT-QuietBox 2): bring-up qualification

Date: 2026-10-01
Target: `p150x4`, physical device IDs `0,1,2,3` (both internal P300c cards of one TT-QuietBox 2),
vLLM 0.24.0, `vllm-tt-plugin` c127c17d80d6.

## What runs

`poolside/Laguna-S-2.1` (117.6B parameters, 8.45B active per token, 48 layers: 12 full-attention +
36 sliding-window, 256 routed experts top-10 + 1 shared) serves through `serve_vllm.sh` with no
overrides:

```text
model=poolside/Laguna-S-2.1  profile=p150x4  max_model_len=131072  max_num_seqs=1
hybrid_kv=1 (four groups, twelve aliased K/V tensor pairs)  streaming_prefill=1  prefix_cache=0
```

It uses the same decoder code as Laguna-XS-2.1. Every size comes from the HF config; the checkpoint is
selected by `TT_LAGUNA_MODEL` (default S; `tt/model_spec.py`). S needs all four chips: its weights
take 17.7 GB of the 21.15 GB usable DRAM per chip at four-way tensor/expert parallelism.

## Accuracy

Reference: `tests/reference_outputs/readiness_aime24_chat_s.refpt`, produced by
`tests/gen_streamed_reference.py` because the 235 GB bf16 model does not fit in host RAM. It runs the
HF `LagunaDecoderLayer` one layer at a time in fp32 (transformers 5.15.0) over the AIME24 chat prompt
(235 tokens) plus a 100-token continuation. A bf16 run of the same tool picks the same top-1 token at
all 100 positions, so the reference has no top-1 near-ties at the bf16 level.

| Check (full 48 layers, p150x4, 131,072-token context, memory margin enforced) | top-1 | top-5 | top-100 |
|---|---:|---:|---:|
| Prefill (`full_model_checks.py prefill_autoreg`) | 0.97 | 1.00 | 1.00 |
| Teacher-forced traced decode (`full_model_checks.py teacher`) | 0.98 | 1.00 | 1.00 |

A 100-token free-running greedy generation is coherent through the last token.

Decoder layers (`tests/test_multichip_decoder.py`, layers 0/1/4 = full+dense, sliding+MoE,
full+MoE; PCC >= 0.995 against HF): 65 passed, 2 skipped (one D1-only and one D2-only case) on p150x4.
On one chip the 72-head sliding layers are rejected up front (decode SDPA needs a power-of-two
query-tile count; 72 heads pad to 3 tiles) and S does not fit there anyway.

### Router

Laguna picks 10 of 256 experts from `sigmoid(logits) + bias`; HF does this in fp32 and `ttnn.topk`
takes only bf16. The default `precise` router computes the logits at HiFi4 with fp32 accumulation and
output, keeps sigmoid and the bias add in fp32, and runs the bf16 top-k on scores shifted by the fp32
midpoint of the 10th and 11th scores. On the real router inputs of all 47 MoE layers
(`tests/test_router_precision.py`):

| Router | Mean expert agreement with HF fp32 | Worst layer |
|---|---:|---:|
| previous bf16 router | 0.9755 | 0.7824 (layer 1) |
| `precise` (default) | 0.9985 | 0.9931 (layer 1) |

## Memory (per chip, `[laguna memory]` from the serving log)

| Stage | Hybrid KV, 131,072 context (default) | Uniform KV, 32,768 context (rollback) |
|---|---|---|
| weights | 17,786 MiB used, 15.9% free | 17,738 MiB used, 16.1% free |
| KV pool allocated | 18,767 MiB, 11.3% free | 18,569 MiB, 12.2% free |
| after decode trace | 18,942 MiB, **10.44% free**, 228 MiB contiguous | 18,738 MiB, 11.41% free, 261 MiB contiguous |

Both clear the serving floor (10% free, 128 MiB contiguous per bank). Uniform KV costs 25.5 KiB per
token per chip for S's 48 layers, so 131,072 tokens (3.4 GB) does not fit: that boot ran out of memory
while allocating the pool. Hybrid KV lets the 36 sliding-window layers share block slots with the 12
full-attention layers (pool = 12 layer-equivalents), so 131,072 tokens costs about 980 MiB. The hybrid
margin is thin (10.44% vs the 10% floor of 21,151 MiB: about 93 MiB above it); a larger context needs a smaller precision footprint or
fewer warm buffers, not just a flag.

## Long context

Chunk-major streaming prefill was enabled on four chips for S (vLLM's chunked prefill is what makes
hybrid KV work). Qualified with `tests/test_streaming_prefill_hardware.py` on p150x4:

| Gate | Result |
|---|---|
| one layer, 8,192 + 64 streamed vs 8,256 monolithic | tail PCC 1.0, K/V PCC 1.0 on every chip, prefix K/V untouched |
| 48 layers + LM head, same split | hidden and logits PCC 1.0, argmax equal |
| 16,400 tokens: 3 x 8,192-row chunks vs one 32,768-row bucket | logits PCC 1.0, top-10 10/10, 1.34x faster (33.2 s vs 44.5 s) |

Passkey retrieval through the hybrid server (key in the first sentence, question at the end,
`enable_thinking=false`): 7,713, 30,753 and 96,033 prompt tokens all returned the correct key.

## Serving latency (hybrid KV default; cold, OSL 512, concurrency 1)

Method as in `p150x4_latency_sweep_20260914.md` (`vllm bench serve`, random dataset, greedy,
`--ignore-eos`, seed 1234, one prompt per point). Every request completed with 512 output tokens.

| Requested ISL | Prompt tokens | TTFT | TPOT | Decode tok/s/user | E2EL |
|---:|---:|---:|---:|---:|---:|
| 128 | 82 | 0.260 s | 54.06 ms | 18.50 | 27.9 s |
| 1,024 | 1,066 | 2.400 s | 54.36 ms | 18.40 | 30.2 s |
| 2,048 | 1,939 | 2.800 s | 54.41 ms | 18.38 | 30.6 s |
| 4,096 | 4,138 | 9.450 s | 54.51 ms | 18.34 | 37.3 s |
| 8,192 | 8,234 | 19.867 s | 54.68 ms | 18.29 | 47.8 s |
| 16,384 | 16,426 | 33.512 s | 54.98 ms | 18.19 | 61.6 s |
| 32,768 | 32,810 | 64.762 s | 55.60 ms | 17.98 | 93.2 s |
| 65,536 | 65,578 | 142.108 s | 56.83 ms | 17.60 | 171.2 s |
| 130,048 | 130,090 | 327.949 s | 59.29 ms | 16.86 | 358.2 s |

Raw values: `laguna_s_p150x4_hybrid_kv_latency_sweep_20261001.tsv`. The uniform-KV rollback (8
sequences, monolithic prefill at the time of measurement) decodes at 66-69 ms
(`laguna_s_p150x4_uniform_kv_latency_sweep_20261001.tsv`); its decode batch is padded to 8.

## Serving behaviour checked

- Chat and tool calls: reasoning is returned separately (`--default-chat-template-kwargs
  {"enable_thinking": true}` makes the `poolside_v1` reasoning parser match the template's default);
  a `get_weather` tool call returned `{"city": "Paris", "unit": "c"}` with `finish_reason=tool_calls`.
- vllm-tt-plugin `tests/tt` against S (uniform KV, 8 sequences, thinking off so short `max_tokens`
  produce content): **70 passed, 2 failed, 1 skipped**. Both failures are properties of S's vocabulary,
  not of the adapter: `allowed_token_ids=[1,2,3]` are control tokens including EOS (id 2), and
  `"Hello` is a single BPE token that `bad_words=["Hello"]` cannot block.
- The tokenizer warning about an "incorrect regex pattern" is a false positive (transformers flags any
  local checkpoint whose config lacks `transformers_version`). Setting `fix_mistral_regex=True` would
  replace S's newline pre-tokenizer and break tokenization; nothing sets it.

## Defects found and fixed during this bring-up

1. Router precision (above).
2. Standalone generator KV length was rounded to 32-token blocks while decode SDPA reads 64-token
   chunks: a 336-position request ended mid-chunk, decode at position 320 read past the page table
   (2.9e35 in layer 0), and every later token was 0. Now rounded to 128.
3. A per-layer `snapshot_download` metadata call hung a full model load on a stalled TLS handshake;
   checkpoint resolution is now local-first and once per process.
4. Device sampling received T instead of 1/T, so temperature 2.0 sampled like 0.5.
5. Defensive: the adapter now treats the plugin's `-1` "no seed" sentinel as no seed. The installed plugin
   (c127c17) already converts `-1` to `None` before calling the model (`model_runner.py:1814`,
   `async_decode.py:517`), so this guard does not change behaviour with that plugin.
6. Repetition/presence/frequency penalties were silently ignored; penalized decode steps now sample
   on host with vLLM's penalty semantics (`tt/host_sampling.py`).
7. A logprobs request on a hybrid-KV server killed the engine; hybrid host-sampling decode is now
   supported.
8. Decode MoE intermediates are kept in L1 only up to the largest qualified footprint.
9. A served config that disagrees with `TT_LAGUNA_MODEL` now fails at startup with the fix named.

## Not covered

- More than one concurrent sequence with hybrid KV (the hybrid allocator is qualified at one).
- Context beyond 131,072 (S declares 1,048,576; see the memory note above).
- Prefix caching on p150x4, DFlash and speculative decode for S.
- Penalized requests take a host round trip per token (correct but slower than device sampling).
