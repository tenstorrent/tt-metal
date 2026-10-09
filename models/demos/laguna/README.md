<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-S-2.1 on a TT-QuietBox 2

[`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1) (117.6B parameters, 8.45B active per token,
1,048,576-token context) served as an OpenAI-compatible vLLM server on four Blackhole ASICs: a TT-QuietBox 2 or four
P150 cards.

- [Results](#results): accuracy (batch 1 and 32), performance vs speed of light, batch-32 performance
- [Quick start](#quick-start): set up, start, check, stop the server
- [Reproduce the results](#reproduce-the-results): accuracy test, perf demo
- [Serving options](#serving-options): DFlash, prefix caching, concurrent requests and their context lengths
- [Weight formats](#weight-formats): how the model fits on the QuietBox 2

## Results

### Accuracy

Compared with the original model run in fp32 on the CPU, over an AIME24 prompt plus a fixed 100-token answer; at
each of the 100 positions both predict the next token. The 128-token prompt is the same question cut to 128 tokens.
At batch 32, 32 users decode together, each with the same prompt and answer; the batch-32 columns show the worst user
(all 32 users measured the same). Measured on 2026-10-08 with `tests/test_accuracy.py`.

| Measure | 235-token prompt, batch 1 | 235-token prompt, batch 32 | 128-token prompt, batch 1 | 128-token prompt, batch 32 | Bar |
|---|---:|---:|---:|---:|---:|
| top-1: Laguna's top token is the reference's | 0.99 | 0.99 | 0.97 | 0.97 | >= 0.90 |
| top-5: the reference's token is in Laguna's top 5 | 1.00 | 1.00 | 1.00 | 1.00 | >= 0.98 |
| top-100 | 1.00 | 1.00 | 1.00 | 1.00 | = 1.00 |
| top-1 of the traced decode the server uses | 0.98 | 1.00 | 0.96 | 0.97 | >= 0.90 |
| PCC of all 100,352 next-token scores, mean over the 100 positions | 0.97 | 0.97 | 0.97 | 0.96 | >= 0.95 |
| PCC, lowest single position | 0.74 | 0.74 | 0.81 | 0.79 | - |

The experts are stored in 4-bit, so the scores carry rounding error while the chosen tokens agree.

### Performance (batch 1)

Speed of light (SoL) is the roofline limit from
[All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/): decode reads the 6.5 GB of
weights a token uses plus the KV cache at 2.05 TB/s; time to first token (TTFT) is the larger of reading the weights
and doing the FLOPs at 2.88 PFLOP/s (`demo/roofline.py` prints every number). The target is 50% of SoL, the usual
mark for MoE models (80% for dense): half the decode speed, twice the TTFT. Measured on 2026-10-06 with the perf demo,
normal decode (the model and server code are unchanged since).

| Input tokens | Decode SoL (tok/s/user) | Decode target | Decode measured | TTFT SoL | TTFT target | TTFT measured |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 316 | 158 | 18.3 | 33 ms | 66 ms | 0.23 s |
| 1,024 | 313 | 157 | 18.1 | 33 ms | 66 ms | 2.39 s |
| 2,048 | 312 | 156 | 18.0 | 33 ms | 66 ms | 2.80 s |
| 4,096 | 310 | 155 | 18.0 | 33 ms | 66 ms | 9.45 s |
| 8,192 | 305 | 152 | 18.0 | 52 ms | 103 ms | 19.9 s |

With DFlash speculative decoding (same run):

| Input tokens | 128 | 1,024 | 2,048 | 4,096 | 8,192 |
|---|---:|---:|---:|---:|---:|
| Decode tok/s/user | 28.0 | 40.5 | 25.3 | 45.8 | 31.1 |
| Decode speedup over normal decode | 1.5x | 2.2x | 1.4x | 2.5x | 1.7x |
| TTFT | 0.37 s | 2.53 s | 2.94 s | 9.59 s | 20.0 s |
| TTFT change from normal decode | +0.14 s | +0.14 s | +0.14 s | +0.14 s | +0.17 s |

DFlash only speeds up decode; the prompt is still processed by Laguna itself, so TTFT stays about the same (0.14-0.17 s
slower). DFlash's decode speedup depends on how much of the draft model's guess Laguna accepts, so it varies from
prompt to prompt.

### Performance (batch 32)

32 users send their prompts at the same moment (`LAGUNA_MAX_NUM_SEQS=32`, normal decode, 512 output tokens each);
measured on 2026-10-08 with `perf_demo.py --batch 32`. Decode SoL is one step for all 32 users: the weights the 32
tokens' experts need (about 50 GB) plus 32 users' KV cache at 2.05 TB/s. TTFT SoL prefills all 32 prompts in one pass,
so every user's first token arrives at the same time. Targets are again 50% of SoL.

The server instead prefills the prompts one after another: a user's TTFT includes waiting for the prefills ahead of
it, and users still decoding slow down while later users prefill. "All 32 decoding" is the fastest user's speed, i.e.
a decode step once every prompt is prefilled (about 89 ms).

| Input tokens | Decode SoL (tok/s/user) | Decode target | Decode measured, all 32 decoding | Decode measured, mean | TTFT SoL | TTFT target | TTFT measured, mean | TTFT measured, first / last user |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 41 | 20 | 11.3 | 11.3 | 33 ms | 66 ms | 11.2 s | 11.2 / 11.2 s |
| 1,024 | 39 | 20 | 11.3 | 7.4 | 193 ms | 386 ms | 75.4 s | 19.5 / 99.4 s |
| 2,048 | 39 | 19 | 11.5 | 4.9 | 392 ms | 783 ms | 124 s | 19.6 / 184 s |
| 4,096 | 37 | 19 | 11.2 | 2.7 | 799 ms | 1.60 s | 218 s | 24.5 / 360 s |
| 8,192 | 35 | 18 | 11.0 | 1.6 | 1.65 s | 3.31 s | 360 s | 51.3 / 640 s |

Throughput across all 32 users:

| Input tokens | Decode SoL (tok/s) | Decode target | Measured, all 32 decoding | Measured over the whole run (prefills included) |
|---:|---:|---:|---:|---:|
| 128 | 1,300 | 650 | 362 | 291 |
| 1,024 | 1,257 | 628 | 362 | 113 |
| 2,048 | 1,237 | 618 | 368 | 72 |
| 4,096 | 1,198 | 599 | 358 | 40 |
| 8,192 | 1,128 | 564 | 352 | 24 |

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

A start takes about 3 minutes; the first one also converts the weights (about 20 minutes, cached under
`~/.cache/ttnn/laguna_s_2_1`). The model thinks before it answers; send
`"chat_template_kwargs": {"enable_thinking": false}` for plain answers. `serve_vllm.sh config` prints the resolved
settings without opening the cards.

## Reproduce the results

Run these with no server running, from the repository root.

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

### Perf demo

```bash
python models/demos/laguna/demo/perf_demo.py              # batch 1, normal and DFlash, ~20 min
python models/demos/laguna/demo/perf_demo.py --batch 32   # 32 concurrent users, normal decode, ~40 min
```

For normal decode and then DFlash, it starts the server, prints the answers to two real prompts, measures TTFT and
decode speed for random-token prompts of 128, 1K, 2K, 4K and 8K tokens (batch 1, 512 output tokens), stops the
server and prints the table. Results go to `generated/laguna_perf_demo/<UTC time>/`. Options: `--modes normal|dflash`,
`--input-lens 128,16384` (a 128K prefill takes about 6 minutes), `--prompts N` (average N prompts per length),
`--output-tokens N`, `--batch N` (N concurrent users; adds total output tok/s).

### Speed of light

```bash
models/demos/laguna/.venv/bin/python models/demos/laguna/demo/roofline.py
```

Prints the full calculation (weights per token, KV cache per token, decode and prefill limits for batch 1, 8 and 32)
from the checkpoint's tensor shapes, ending with the SoL and 50% targets of the batch-1 and batch-32 tables above. No
device needed.

## Serving options

Put the variables in front of `serve_vllm.sh`. Experimental features also need `LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`.

| Feature | Variables | Notes |
|---|---|---|
| DFlash speculative decoding (experimental) | `TT_LAGUNA_DFLASH=1` | Draft model [`poolside/Laguna-S-2.1-DFlash`](https://huggingface.co/poolside/Laguna-S-2.1-DFlash); 1 request at a time; only greedy requests are sped up |
| N-gram speculative decoding (experimental) | `TT_LAGUNA_SPEC_DECODE=1` | Guesses by repeating earlier text, no draft model; 1 request at a time; little speedup |
| Prefix caching (experimental) | `TT_LAGUNA_PREFIX_CACHE=1` | Reuses a repeated prompt prefix; 1 request at a time |
| Concurrent requests | `LAGUNA_MAX_NUM_SEQS=N`, N up to 32 | All requests share one context pool (below) |

Context per request with N concurrent requests (the pool holds about 1.3M tokens, sized from the memory left after
the weights with a 10% margin):

| Concurrent requests | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Context per request, if all equal | 1,048,576 (model limit) | 653K | 326K | 162K | 80K | 39K |

Only Laguna's 12 full-attention layers store every past token; the other 36 keep a 512-token sliding window (hybrid
KV cache). `TT_LAGUNA_HYBRID_KV=0` stores every layer's history instead: 131,072 tokens, up to 8 requests.

## Weight formats

The checkpoint is bf16 (235 GB), which does not fit the QuietBox 2's ~127 GB of usable device memory. The first server
start converts the weights once:

| Weights | Share of parameters | Stored as | Size |
|---|---:|---|---:|
| Routed experts | 97% | `bfloat4_b` (4-bit; 16 values share one 8-bit exponent) | 63.9 GB |
| Attention, shared expert, layer 0, LM head | 3% | `bfloat8_b` (8-bit) | 3.9 GB |
| Router, embedding, norms | <1% | bf16 | 0.7 GB |
| KV cache | - | `bfloat8_b` | - |

The bring-up's precision sweep (`doc/datatype_sweep/selected_precision_config.json`, run on Laguna-XS) kept the lowest
precision that passes the accuracy bars; 4-bit attention or LM head did not.
