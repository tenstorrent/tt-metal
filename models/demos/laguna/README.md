<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-S-2.1 on a TT-QuietBox 2

[`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1) (117.6B parameters, 8.45B active, 1M-token
context) as an OpenAI-compatible vLLM server on four Blackhole ASICs (a TT-QuietBox 2 or four P150 cards).

## Status (2026-10-10)

Through the vLLM server, batch 1, versus the first version of this branch (2026-10-06), same measurement
(2026-10-09):

| | First version | Now |
|---|---:|---:|
| Decode | 18 tok/s | 66-71 tok/s (3.7-3.9x) |
| TTFT, 128-token prompt | 0.23 s | 87 ms |
| TTFT, 8K-token prompt | 19.9 s | 0.89 s |
| Batch-32 decode, all 32 users | 11 tok/s/user | 21-23 tok/s/user |

Against the targets (50% of speed of light, measured without vLLM, tables below):

- Met: batch-1 TTFT at 128 tokens (61 ms vs 66 ms) and batch-32 decode at every length (25-32 vs 18-20 tok/s/user).
- Unhittable on this machine: each needs less time than work that cannot be removed already takes.
  - Batch-1 decode without DFlash (83-88 vs 152-158 tok/s): needs 6.3-6.6 ms per token; weight reads at the best
    measured 357 GB/s (4.5 ms), 96 all-reduces (1.3 ms), attention (1.1 ms) and LM head (0.4 ms) already take 7.3 ms.
  - Batch-1 TTFT from 1K tokens up (140 vs 66 ms at 1K): each chip reads all 64 local experts' weights every layer
    (~42 ms), plus prefill collectives (~20 ms) and attention (~7 ms).
  - Batch-32 TTFT (11.6 vs 1.60 s at 4K): prefill collectives alone take ~58 ms per 4K tokens, so for 32 prompts
    they exceed the target at every length (0.46 / 0.93 / 1.86 / 3.7 s vs 0.39 / 0.78 / 1.60 / 3.31 s at 1K-8K). At
    128 tokens the 32 prompts are one 4K-token prefill: 58 ms of collectives plus 31 ms of expert matmuls vs 66 ms.
- DFlash speculative decoding reaches the batch-1 decode target at every input length over a 2048-token answer
  (188 / 200 / 194 / 202 / 160 vs 158 / 157 / 156 / 155 / 152 tok/s at 128 / 1K / 2K / 4K / 8K; over 1024 tokens at
  1K-4K, 3% short at 128 and 8K).
  Over only the first 256 tokens the draft accepts 1.15-1.5 tokens per round on real text: even at the cost of a
  1-row check every round (2.5 + 12.2 ms) that caps 1K / 2K / 8K at 152 / 153 / 146 tok/s, under target.
  About 7x faster than on 2026-10-09 (25 -> 171 tok/s on AIME24 and 21-25 -> 107-132 tok/s on real text over the
  first 256 tokens).

## Results

### Accuracy

Laguna's next-token predictions vs the original model in fp32, over the AIME24 prompt (235 tokens, or cut to 128)
plus a fixed 100-token answer; batch 32 is the worst of 32 identical users. "top-1, traced" is the same top-1 check
through the traced decode path the server runs (token picked on device). `tests/test_accuracy.py`, 2026-10-10.

| Prompt (AIME24) | Batch | top-1 | top-5 | top-100 | top-1, traced | PCC |
|---|---:|---:|---:|---:|---:|---:|
| 235 tokens | 1 | 0.98 | 1.00 | 1.00 | 0.99 | 0.97 |
| 235 tokens | 32 | 0.98 | 1.00 | 1.00 | 0.99 | 0.97 |
| 128 tokens | 1 | 0.96 | 1.00 | 1.00 | 0.97 | 0.97 |
| 128 tokens | 32 | 0.96 | 1.00 | 1.00 | 0.98 | 0.97 |
| Bar | | >= 0.90 | >= 0.98 | 1.00 | >= 0.90 | >= 0.95 |

### Performance

Target = 50% of speed of light (SoL), the roofline limit from [All About Transformer
Inference](https://jax-ml.github.io/scaling-book/inference/) computed by `demo/roofline.py`. Measured with
`demo/perf_direct.py`: no vLLM, device time plus reading the token back, 2026-10-10. Batch 32: 32 prompts arrive
together; TTFT is the last user's; total decode throughput is 32x the per-user speed.

Decode, tok/s per user (measured / target):

| Input tokens | Batch 1 | Batch 32 |
|---:|---:|---:|
| 128 | 88.2 / 158 | 31.7 / 20 |
| 1,024 | 85.1 / 157 | 30.1 / 20 |
| 2,048 | 84.8 / 156 | 28.9 / 19 |
| 4,096 | 83.8 / 155 | 26.7 / 19 |
| 8,192 | 83.2 / 152 | 25.1 / 18 |

Time to first token (measured / target):

| Input tokens | Batch 1 | Batch 32 |
|---:|---:|---:|
| 128 | 60.6 ms / 66 ms | 0.42 s / 0.066 s |
| 1,024 | 140 ms / 66 ms | 2.89 s / 0.39 s |
| 2,048 | 238 ms / 66 ms | 5.82 s / 0.78 s |
| 4,096 | 386 ms / 66 ms | 11.6 s / 1.60 s |
| 8,192 | 714 ms / 103 ms | 23.4 s / 3.31 s |

DFlash speculative decoding, batch 1 (decode tok/s over the first 256 / 1024 / 2048 generated tokens; normal decode is
83-88 tok/s at these lengths; the draft accepts more later in an answer). A round
drafts 15 tokens, then checks up to the first 5 in Laguna in one step (a 3-row check when 2 or fewer are checked) and keeps the matching ones plus one of
Laguna's own. It checks fewer when the draft is unsure (once the product of its top-1 probabilities drops below 0.2),
which reads fewer expert weights. 18-21 ms per round (draft 2.5 ms, check 16.0-18.4 ms) vs 11.3-12.0 ms per token
for normal decode, so DFlash wins when more than ~0.9 drafts per round are accepted:

| Prompt | Input tokens | 256 tokens | 1024 tokens | 2048 tokens | Target |
|---|---:|---:|---:|---:|---:|
| AIME24 | 235 | 171.4 (2.5) | 194.9 (3.1) | 178.1 (2.9) | |
| Real text | 128 | 131.5 (1.5) | 154.2 (2.1) | 188.4 (2.9) | 158 |
| Real text | 1,024 | 112.9 (1.2) | 164.4 (2.4) | 199.8 (3.3) | 157 |
| Real text | 2,048 | 115.1 (1.3) | 156.8 (2.2) | 194.1 (3.2) | 156 |
| Real text | 4,096 | 125.9 (1.5) | 169.1 (2.6) | 202.4 (3.4) | 155 |
| Real text | 8,192 | 107.4 (1.2) | 147.2 (2.1) | 159.6 (2.4) | 152 |

tok/s (drafts accepted per round); target = the batch-1 decode target.

Real text = a "summarize this document" request over a technical report. With the AIME24 answer fed in, the device
draft accepts 2.6-2.7 drafts per round, as the draft model does on CPU (2.64): the acceptance rate is the draft
model's own. Single runs vary by about +-10%, since the output (and so the acceptance) shifts with near-tie tokens
(AIME24 over 256 tokens: 144-174 tok/s at 2.1-2.6 drafts per round across the 2026-10-10 runs).

## Quick start

Set up once:

```bash
git clone --branch jerrywangTT/laguna-s --recurse-submodules https://github.com/tenstorrent/tt-metal.git
cd tt-metal
export MODEL_DIR="$PWD/models/demos/laguna"
sudo ./install_dependencies.sh            # only if this host has never built tt-metal
"$MODEL_DIR/setup_vllm.sh"                 # builds tt-metal (1-3 h fresh) and the vLLM environment (30-45 min)
"$MODEL_DIR/.venv/bin/hf" auth login       # after accepting the model's terms on Hugging Face
"$MODEL_DIR/.venv/bin/hf" download poolside/Laguna-S-2.1   # 235 GB; otherwise downloaded at first start
```

Start, check and stop the server (port 8000):

```bash
"$MODEL_DIR/serve_vllm.sh"            # ready when ~/laguna-logs/latest.log prints "Application startup complete"
curl -fsS http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "poolside/Laguna-S-2.1", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}]}'
"$MODEL_DIR/serve_vllm.sh" stop       # also resets every Tenstorrent ASIC in the system (tt-smi -r)
```

A start takes ~3 min (the first also converts the weights, ~20 min). For answers without thinking, send
`"chat_template_kwargs": {"enable_thinking": false}`.

## Reproduce the results

From the repository root, with no server running.

### Accuracy test

```bash
REPO=$PWD MODEL_DIR=$PWD/models/demos/laguna
# once: fp32 reference scores from the original model (CPU only, ~5 min, ~30 GB of memory)
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python -m models.demos.laguna.tests.gen_streamed_reference --dtype fp32 \
  --output generated/laguna_reference/readiness_aime24_chat_s.refpt \
  --save-logits generated/laguna_reference/Laguna-S-2.1-aime24-logits.pt
# the tests (~6 min): batch 1, then batch 32 (worst user); each prints top-1, top-5, top-100, traced top-1 and PCC
# and fails if any is below its bar
cd /tmp && env -u TT_METAL_HOME PYTHONPATH=$REPO \
  $MODEL_DIR/.venv/bin/python -m pytest -s $MODEL_DIR/tests/test_accuracy.py
```

For the 128-token prompt, make its reference once and point the tests at it:

```bash
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python -m models.demos.laguna.tests.gen_streamed_reference --dtype fp32 \
  --prompt-len 128 --output generated/laguna_reference/readiness_aime24_chat_s_p128.refpt \
  --save-logits generated/laguna_reference/Laguna-S-2.1-aime24-p128-logits.pt
cd /tmp && env -u TT_METAL_HOME PYTHONPATH=$REPO \
  LAGUNA_REFERENCE_TOKENS=$REPO/generated/laguna_reference/readiness_aime24_chat_s_p128.refpt \
  LAGUNA_REFERENCE_LOGITS=$REPO/generated/laguna_reference/Laguna-S-2.1-aime24-p128-logits.pt \
  $MODEL_DIR/.venv/bin/python -m pytest -s $MODEL_DIR/tests/test_accuracy.py
```

### Perf test

```bash
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python $MODEL_DIR/demo/perf_direct.py   # performance tables, ~10 min
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python $MODEL_DIR/demo/perf_direct.py --modes dflash --dflash-prompt aime,text   # DFlash table (256 tokens; --dflash-tokens 1024 / 2048 for the other columns)
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python $MODEL_DIR/demo/roofline.py      # SoL and targets, no device
python $MODEL_DIR/demo/perf_demo.py                                           # end to end through vLLM
```

## Serving options

Set in front of `serve_vllm.sh`; experimental ones also need `LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`.

| Feature | Variables | Notes |
|---|---|---|
| DFlash speculative decoding | `TT_LAGUNA_DFLASH=1` | experimental; draft [`Laguna-S-2.1-DFlash`](https://huggingface.co/poolside/Laguna-S-2.1-DFlash); 1 greedy request |
| N-gram speculative decoding | `TT_LAGUNA_SPEC_DECODE=1` | experimental; 1 greedy request; not measured on this version |
| Prefix caching | `TT_LAGUNA_PREFIX_CACHE=1` | experimental; 1 request |
| Concurrent requests | `LAGUNA_MAX_NUM_SEQS=N` | N up to 32; shared context pool (below) |

Context per request (one ~1.3M-token pool shared by all requests):

| Concurrent requests | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Context per request, if all equal | 1,048,576 (model limit) | 653K | 326K | 162K | 80K | 39K |

Only the 12 full-attention layers keep every past token; the 36 sliding-window layers keep 512.

## Weight formats

The bf16 checkpoint (235 GB) does not fit in ~127 GB of device memory; the first start converts it:

| Weights | Share of parameters | Stored as | Size |
|---|---:|---|---:|
| Routed experts | 97% | `bfloat4_b` (4-bit; 16 values share one 8-bit exponent) | 63.9 GB |
| Attention, shared expert, layer 0, LM head | 3% | `bfloat8_b` (8-bit) | 3.9 GB |
| Router, embedding, norms | <1% | bf16 | 0.7 GB |
| KV cache | - | `bfloat8_b` | - |
