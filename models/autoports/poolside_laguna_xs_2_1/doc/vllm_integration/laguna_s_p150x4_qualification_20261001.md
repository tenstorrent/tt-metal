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

## Addendum 2026-10-02: context, prefix caching, speculative decoding, concurrency

Everything above was measured on 2026-10-01 with the 131,072-token default and a 1.5e9 trace region.
Changes since then, each measured on this p150x4 machine:

**Trace region and context** (275d9b2c16b). `trace_region_size` is reserved in every DRAM bank
(`tt_metal/impl/allocator/allocator.cpp:39`), so 1.5e9 held 12 GB of each chip's 32 GB. S's decode trace
measures 27,238,400 B (TT_FATAL with a 10 MB region). S now uses 300 MB per bank: usable DRAM 21,151 ->
30,307 MiB per chip. The hybrid-KV default context is S's declared 1,048,576: pool 16,796 blocks, 16.42%
free after the decode trace. Passkey retrieval through that server: 251,553 tokens correct (1,010 s),
503,073 tokens correct (4,106 s). Uniform KV now fits 131,072 (29.95% free; was capped at 32,768).

**Prefix caching on p150x4** (16f2c4c17db; uniform KV, one sequence, experimental acknowledgement).
`tests/prefix_cache_qualification.py` off/on on fresh 131,072-token servers: PASS on every verdict
(exact tokens for cold / full-hit / partial-hit cases up to 129,984 tokens, canonical 8,192-token cached
counts, metrics, health). Full-hit TTFT: 32K 52.4 -> 15.0 s (3.5x), 65K 125.4 -> 20.1 s (6.2x); TPOT
unchanged.

**N-gram speculative decoding** (`TT_LAGUNA_SPEC_DECODE=1`, uniform KV, greedy). Fixed a duplicated
token per answer (3e63196d03a). Outputs are deterministic; 2/5 test prompts equal normal decode, 3/5
diverge after 365-643 characters at near-ties (batched verify numerics). Time vs normal decode:
-27% .. +4%. Sampled requests on a spec server do not deadlock.

**DFlash with poolside/Laguna-S-2.1-DFlash** (445f2d11e43, ece43b4859a; uniform KV, one sequence,
greedy). Hardware gates on chips 0-3: one draft layer PCC 0.99999785 / 0.99994230; full six-layer
round all PCC bars met, one bf16 near-tie row (b81445f439c); full 48-layer target + draft + verify:
verify logits PCC 1.0, argmax equal. Serving: fixed an engine-killing position check (ece43b4859a);
outputs deterministic, 3/5 equal normal decode, 2/5 diverge at near-ties. Warm ms/token vs normal:
60.4/54.6, 57.6/57.5, 103.2/54.5, 57.2/54.3, 41.1/54.4. Functional, but a net loss except on highly
predictable text because verify runs eagerly (~277 ms per 16-row round).

**DFlash, traced verify (2026-10-02 afternoon; 5ce2f42f312 and its two parents).** Three changes:
(1) with `TT_LAGUNA_DFLASH=1` the TT scheduler reserves 16 look-ahead KV slots (`laguna_vllm_ext.dflash_lookahead`;
stock vLLM does `num_spec_tokens + 1` for its own DFlash). Before, at positions 49..63 of every 64-token block a
16-row verify would have written past the allocated block, so DFlash fell back to one eager target row per token
(141 ms each; 395 of 707 rounds in one run). (2) The 16-row target verify is captured once at warmup and replayed:
254 -> 104 ms per round (`TT_LAGUNA_DFLASH_TRACE=0` keeps the eager verify). (3) Every eager piece that runs beside
the resident trace (the draft at each tile-padded context size 32..544 rows, the 1..16-row aux combine, the 1-row
verify) is compiled before capture, and the rolling target context lives in one buffer allocated before capture
and stores combined (fc + norm) states. Without (3) the second draft round hung in its logits read and the board
needed `tt-smi -r`.

Measured with `vllm` chat completions, greedy, thinking off, 512 tokens, second (warm) run of each prompt;
decode tokens/s per user = (completion tokens - 1) / (last token time - first token time):

| prompt | normal decode | DFlash eager, no look-ahead | DFlash traced + look-ahead |
|---|---:|---:|---:|
| Python binary search | 18.5 | 20.2 | 63.1 |
| C++ ring buffer | 18.5 | 16.2 | 42.7 |
| SQL schema | 18.5 | 17.1 | 42.5 |
| `def fibonacci` completion | 18.6 | 16.1 | 51.3 |
| repeat a 7-item list 3x | 19.0 | 22.0 | 54.9 |
| first 60 primes | 18.6 | 21.4 | 60.3 |
| explain a hash map | 18.5 | 10.7 | 29.2 |
| lighthouse short story | 18.5 | 6.3 | 13.9 |

Traced and eager verify (same look-ahead) produce identical completions on all 8 prompts. Per round: 4.37
committed tokens on average, draft median 24.5 ms, verify median 104 ms. DFlash text differs from normal decode on
7/8 prompts at near-ties (log-probability gaps 0.13-1.0 at the first differing token); a 16-row batched decode
against HF fp32 on the optimizer reference (100 positions) gives argmax equal to 1-row decode at 99/99 positions,
top-1 98% vs 99%, mean logits correlation 0.9697 vs 0.9695, with no degradation by row index, so the difference
is numerics of the batched pass, not a verify bug.

**Concurrency.** Uniform KV, 8 sequences, 131,072 context: 8 concurrent greedy chat requests give
outputs identical to the same requests sent one at a time; 642 tokens in 9.9 s (64.6 tok/s aggregate)
vs 43.2 s sequentially.

**Re-checks after these changes.** vllm-tt-plugin `tests/tt` (uniform KV, 8 sequences, thinking off):
69 passed, 3 failed, 1 skipped. Two failures come from test assumptions about the vocabulary, checked with the
Laguna tokenizer: `test_bad_words` blocks "Hello" (token 6352) and " Hello" (16331), and vLLM correctly blocks
exactly those sequences, but the model wrote `"Hello` (one token, 91512), which the test's punctuation-stripping
text check then flags; `test_allowed_token_ids` allows ids 1-3, which in this vocabulary are the control tokens
`〈|CODE_START|〉`, `〈|EOS|〉`, `〈|CODE_END|〉` that decode to an empty string. The third (`test_topk[15]`, all 8
sampled first tokens "\n") passed in 3/3 reruns. Laguna-XS-2.1 on p150x2 with
the same code: prefill top-1/5/100 0.95/1.00/1.00, teacher-forced 0.94/1.00/1.00 (established 0.95).
The TTNN warning "Allocating device buffers is unsafe due to the existence of an active trace" printed
once at the first request of every server is expected: eager prefill allocates temporaries after
trace capture and frees them within the call; it was present in the 2026-10-01 servers too.

## Not covered

- More than one concurrent sequence with hybrid KV (the hybrid allocator is qualified at one; use
  uniform KV for concurrency, up to 131,072 tokens).
- Passkey retrieval beyond 503,073 tokens (the 1,048,576 configuration boots with 16.4% free).
- Prefix caching together with hybrid KV; DFlash or n-gram speculation together with hybrid KV.
- DFlash on low-acceptance text: on free prose (the story prompt) DFlash commits under 2 tokens per 129 ms round
  and stays slower than normal decode (13.9 vs 18.5 tokens/s). DFlash and n-gram speculation are one sequence,
  greedy, uniform KV only.
- More than 8 concurrent sequences on p150x4 (the launcher limit), including vLLM's 32-sequence nightly shape.
- Penalized requests take a host round trip per token (correct but slower than device sampling), and the
  first token of a penalized request (sampled in prefill) is not penalized.
