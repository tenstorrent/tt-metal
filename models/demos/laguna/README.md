<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-S-2.1 on a TT-QuietBox 2

[`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1) (117.6B parameters, 8.45B active per token,
1,048,576-token context) served as an OpenAI-compatible vLLM server on four Blackhole ASICs: a TT-QuietBox 2 or four
P150 cards.

- [Results](#results): accuracy (batch 1 and 32), performance vs speed of light (batch 1 and 32)
- [Quick start](#quick-start): set up, start, check, stop the server
- [Reproduce the results](#reproduce-the-results): accuracy test, perf test, perf demo through the server
- [Serving options](#serving-options): DFlash, prefix caching, concurrent requests and their context lengths
- [Weight formats](#weight-formats): how the model fits on the QuietBox 2

## Results

### Accuracy

Compared with the original model run in fp32 on the CPU, over an AIME24 prompt plus a fixed 100-token answer; at
each of the 100 positions both predict the next token. The 128-token prompt is the same question cut to 128 tokens.
At batch 32, 32 users decode together, each with the same prompt and answer; the batch-32 columns show the worst user
(all 32 users measured the same). Measured on 2026-10-09 with `tests/test_accuracy.py`.

| Measure | 235-token prompt (AIME24), batch 1 | 235-token prompt (AIME24), batch 32 | 128-token prompt (AIME24), batch 1 | 128-token prompt (AIME24), batch 32 | Bar |
|---|---:|---:|---:|---:|---:|
| top-1: Laguna's top token is the reference's | 0.98 | 0.99 | 0.96 | 0.95 | >= 0.90 |
| top-5: the reference's token is in Laguna's top 5 | 1.00 | 1.00 | 1.00 | 1.00 | >= 0.98 |
| top-100 | 1.00 | 1.00 | 1.00 | 1.00 | = 1.00 |
| top-1 of the traced decode the server uses | 0.98 | 0.99 | 0.95 | 0.96 | >= 0.90 |
| PCC of all 100,352 next-token scores, mean over the 100 positions | 0.97 | 0.97 | 0.97 | 0.97 | >= 0.95 |
| PCC, lowest single position | 0.75 | 0.75 | 0.82 | 0.82 | - |

The experts are stored in 4-bit, so the scores carry rounding error while the chosen tokens agree.

### Performance (batch 1)

Speed of light (SoL) is the roofline limit from
[All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/): decode reads the 6.5 GB of
weights a token uses plus the KV cache at 2.05 TB/s; time to first token (TTFT) is the larger of reading the weights
and doing the FLOPs at 2.88 PFLOP/s (`demo/roofline.py` prints every number). The target is 50% of SoL, the usual
mark for MoE models (80% for dense): half the decode speed, twice the TTFT.

SoL counts only device work, so the measured numbers come from `demo/perf_direct.py`, which runs the model with the
server's settings in one process, without vLLM (no HTTP, chat template or scheduler): TTFT is the prefill of a prompt
of exactly the stated length, the LM head on its last token, on-device greedy sampling and reading the token back
(replayed from a trace up to 2,048 tokens, as the server does for short prompts); decode is one step of the captured
decode trace at a context of the input length. Measured on 2026-10-09, normal decode.

| Input tokens | Decode SoL (tok/s/user) | Decode target | Decode measured | TTFT SoL | TTFT target | TTFT measured |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 316 | 158 | 75.9 | 33 ms | 66 ms | 60.5 ms |
| 1,024 | 313 | 157 | 73.6 | 33 ms | 66 ms | 140 ms |
| 2,048 | 312 | 156 | 73.2 | 33 ms | 66 ms | 238 ms |
| 4,096 | 310 | 155 | 72.2 | 33 ms | 66 ms | 386 ms |
| 8,192 | 305 | 152 | 72.3 | 52 ms | 103 ms | 716 ms |

With DFlash speculative decoding, measured through the vLLM server (`perf_demo.py --modes dflash`; `perf_direct.py`
has no DFlash mode yet):

| Input tokens | 128 | 1,024 | 2,048 | 4,096 | 8,192 |
|---|---:|---:|---:|---:|---:|
| Decode tok/s/user | 22.1 | 53.0 | 26.6 | 42.9 | 29.2 |
| TTFT | 289 ms | 296 ms | 377 ms | 583 ms | 1,056 ms |

DFlash is currently slower than normal decode at every length. Each DFlash round, the draft model proposes 15
tokens and Laguna checks all 16 rows (the last accepted token plus the drafts) in one forward pass through its
multi-token prefill path; the normal-decode speedups are one-token kernels that this pass does not use, so a round
(about 22 ms draft + 85 ms check at 128 tokens) costs more than the tokens it accepts would take to decode one at a
time. Use normal decode (the default) until DFlash is optimized.

### Performance (batch 32)

32 prompts arrive together and are prefilled in the server's groups: up to 4,096 tokens several prompts are packed
into one prefill of at most 8,192 rows (32 prompts of 128 tokens in one pass, 8 of 1K, 4 of 2K, 2 of 4K), and 8K
prompts one at a time. TTFT SoL prefills all 32 prompts in one pass, so every user's first token arrives at the same
time; the measured TTFT is the last user's (the moment all 32 have a token) and the mean over the 32 users, who wait
for the groups ahead of them. Decode SoL is one step for all 32 users: the weights the 32 tokens' experts need (about
50 GB) plus 32 users' KV cache at 2.05 TB/s; the measured decode is one step of the batch-32 decode trace with 32
different tokens at a context of the input length. Measured on 2026-10-09 with `demo/perf_direct.py` (no vLLM);
targets are again 50% of SoL.

| Input tokens | Decode SoL (tok/s/user) | Decode target | Decode measured | TTFT SoL | TTFT target | TTFT measured, last user | TTFT measured, mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 41 | 20 | 28.0 | 33 ms | 66 ms | 0.42 s | 0.42 s |
| 1,024 | 39 | 20 | 26.4 | 193 ms | 386 ms | 2.89 s | 1.81 s |
| 2,048 | 39 | 19 | 25.8 | 392 ms | 783 ms | 5.82 s | 3.27 s |
| 4,096 | 37 | 19 | 23.5 | 799 ms | 1.60 s | 11.6 s | 6.19 s |
| 8,192 | 35 | 18 | 22.4 | 1.65 s | 3.31 s | 23.4 s | 12.1 s |

Decode throughput across all 32 users:

| Input tokens | Decode SoL (tok/s) | Decode target | Decode measured |
|---:|---:|---:|---:|
| 128 | 1,300 | 650 | 896 |
| 1,024 | 1,257 | 628 | 845 |
| 2,048 | 1,237 | 618 | 826 |
| 4,096 | 1,198 | 599 | 752 |
| 8,192 | 1,128 | 564 | 717 |

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

### Perf test

The performance tables above (no server may be running; ~10 min):

```bash
python models/demos/laguna/demo/perf_direct.py                          # batch 1 and 32, 128 .. 8K tokens
python models/demos/laguna/demo/perf_direct.py --batch 1 --input-lens 128,1024
```

It prints one line per batch size and input length (TTFT, decode tok/s/user, the SoL) and then the tables with the
targets. `--output FILE` saves the results as JSON.

### Perf demo (through the server)

End to end through vLLM, including DFlash:

```bash
python models/demos/laguna/demo/perf_demo.py              # batch 1, normal and DFlash, ~20 min
python models/demos/laguna/demo/perf_demo.py --batch 32   # 32 concurrent users, normal decode, ~40 min
```

For normal decode and then DFlash, it starts the server, prints the answers to two real prompts, measures TTFT and
decode speed for random-token prompts of 128, 1K, 2K, 4K and 8K tokens (batch 1, 512 output tokens), stops the
server and prints the table. The prompts are re-tokenized through the chat template, so their real lengths differ
from the requested ones (82, 1,066, 1,939, 4,138 and 8,234 tokens); the server path adds about 20-60 ms to TTFT
(170 ms at 8K, where the 8,234-token prompt is longer than the largest single prefill) and costs 6-8% of decode speed. Results go to `generated/laguna_perf_demo/<UTC time>/`. Options:
`--modes normal|dflash`, `--input-lens 128,16384` (a 128K prefill takes about 6 minutes), `--prompts N` (average N
prompts per length), `--output-tokens N`, `--batch N` (N concurrent users; adds total output tok/s).

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
