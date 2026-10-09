<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-S-2.1 on a TT-QuietBox 2

[`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1) (117.6B parameters, 8.45B active per token,
1,048,576-token context) served as an OpenAI-compatible vLLM server on four Blackhole ASICs: a TT-QuietBox 2 or four
P150 cards.

- [Results](#results): accuracy (batch 1 and 32), performance vs speed of light (batch 1 and 32, through vLLM and
  without it)
- [Quick start](#quick-start): set up, start, check, stop the server
- [Reproduce the results](#reproduce-the-results): accuracy test, perf demo
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
mark for MoE models (80% for dense): half the decode speed, twice the TTFT. Measured on 2026-10-09 with the perf demo
(through the vLLM server), normal decode.

| Input tokens | Decode SoL (tok/s/user) | Decode target | Decode measured | TTFT SoL | TTFT target | TTFT measured |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 316 | 158 | 71.3 | 33 ms | 66 ms | 87 ms |
| 1,024 | 313 | 157 | 67.5 | 33 ms | 66 ms | 186 ms |
| 2,048 | 312 | 156 | 67.3 | 33 ms | 66 ms | 259 ms |
| 4,096 | 310 | 155 | 67.0 | 33 ms | 66 ms | 445 ms |
| 8,192 | 305 | 152 | 66.3 | 52 ms | 103 ms | 888 ms |

The perf demo's random prompts are re-tokenized through the chat template, so the real input lengths are 82, 1,066,
1,939, 4,138 and 8,234 tokens, and the server pads each to its prefill bucket (1,066 runs as 1,152). The same model
code measured without the server (exact lengths, device time plus the token readback) is in
[Performance without vLLM](#performance-without-vllm).

With DFlash speculative decoding (same run):

| Input tokens | 128 | 1,024 | 2,048 | 4,096 | 8,192 |
|---|---:|---:|---:|---:|---:|
| Decode tok/s/user | 22.1 | 53.0 | 26.6 | 42.9 | 29.2 |
| Decode speed relative to normal decode | 0.31x | 0.79x | 0.40x | 0.64x | 0.44x |
| TTFT | 289 ms | 296 ms | 377 ms | 583 ms | 1,056 ms |
| TTFT change from normal decode | +202 ms | +111 ms | +118 ms | +138 ms | +168 ms |

DFlash is now slower than normal decode at every length: normal decode got about 4x faster, while each DFlash round
(a draft block plus Laguna's verify pass over it) did not, so a round costs more than the tokens it accepts would take
to decode one at a time. Its speed still depends on how much of the draft Laguna accepts, so it varies from prompt to
prompt. Use normal decode (the default).

### Performance (batch 32)

32 users send their prompts at the same moment (`LAGUNA_MAX_NUM_SEQS=32`, normal decode, 512 output tokens each);
measured on 2026-10-09 with `perf_demo.py --batch 32`. Decode SoL is one step for all 32 users: the weights the 32
tokens' experts need (about 50 GB) plus 32 users' KV cache at 2.05 TB/s. TTFT SoL prefills all 32 prompts in one pass,
so every user's first token arrives at the same time. Targets are again 50% of SoL.

The server prefills the prompts in groups: up to 4,096 tokens it packs several prompts into one prefill of at most
8,192 rows (32 prompts of 128 tokens in one pass, 8 of 1K, 4 of 2K, 2 of 4K), and 8K prompts one at a time. A user's
TTFT includes waiting for the groups ahead of it, and users already decoding slow down while later groups prefill.
"All 32 decoding" is the fastest user's speed, i.e. a decode step once every prompt is prefilled (about 44-48 ms).

| Input tokens | Decode SoL (tok/s/user) | Decode target | Decode measured, all 32 decoding | Decode measured, mean | TTFT SoL | TTFT target | TTFT measured, mean | TTFT measured, first / last user |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 41 | 20 | 22.9 | 22.8 | 33 ms | 66 ms | 2.04 s | 0.75 / 2.08 s |
| 1,024 | 39 | 20 | 22.7 | 21.7 | 193 ms | 386 ms | 4.24 s | 1.37 / 5.29 s |
| 2,048 | 39 | 19 | 22.5 | 20.1 | 392 ms | 783 ms | 6.05 s | 1.28 / 8.76 s |
| 4,096 | 37 | 19 | 21.6 | 17.3 | 799 ms | 1.60 s | 9.27 s | 1.29 / 15.1 s |
| 8,192 | 35 | 18 | 20.8 | 14.1 | 1.65 s | 3.31 s | 15.9 s | 2.54 / 27.6 s |

Throughput across all 32 users:

| Input tokens | Decode SoL (tok/s) | Decode target | Measured, all 32 decoding | Measured over the whole run (prefills included) |
|---:|---:|---:|---:|---:|
| 128 | 1,300 | 650 | 733 | 671 |
| 1,024 | 1,257 | 628 | 726 | 589 |
| 2,048 | 1,237 | 618 | 720 | 520 |
| 4,096 | 1,198 | 599 | 691 | 423 |
| 8,192 | 1,128 | 564 | 666 | 314 |

### Performance without vLLM

The speed of light counts only device work, so `demo/perf_direct.py` measures the same model code (the serving
settings of `serve_vllm.sh`) in one process with no server: no HTTP, chat template or scheduler, and prompts of exactly
the stated length. TTFT is the prefill of the prompt, the LM head on its last token, on-device greedy sampling and
reading the token back: traced (device time) up to 2,048 tokens, where the server also replays short prefills from a
trace, and dispatched op by op above. Decode is one step of the captured decode trace at a context of the input
length. At batch 32 the prompts are prefilled in the server's groups (see above) and TTFT is the last user's, the
moment all 32 have a token, which the one-pass SoL bounds. Measured on 2026-10-09.

| Input tokens | Batch 1 decode (tok/s/user) | Target | Batch 1 TTFT | Target | Batch 32 decode (tok/s/user) | Target | Batch 32 TTFT, last user | Target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 75.9 | 158 | 60.5 ms | 66 ms | 28.0 | 20 | 0.42 s | 66 ms |
| 1,024 | 73.6 | 157 | 140 ms | 66 ms | 26.4 | 20 | 2.89 s | 386 ms |
| 2,048 | 73.2 | 156 | 238 ms | 66 ms | 25.8 | 19 | 5.82 s | 783 ms |
| 4,096 | 72.2 | 155 | 386 ms | 66 ms | 23.5 | 19 | 11.6 s | 1.60 s |
| 8,192 | 72.3 | 152 | 716 ms | 103 ms | 22.4 | 18 | 23.4 s | 3.31 s |

Batch-1 TTFT at 128 tokens and batch-32 decode meet their targets. Through vLLM, batch-1 decode is 6-8% lower (the
server reads each token back and schedules the next step on the host) and TTFT is 21-59 ms higher up to 4K tokens
(request handling, and real prompt lengths padded up to prefill buckets) and 172 ms higher at 8K, where the perf
demo's prompt is 8,234 tokens, longer than the largest single prefill (8,192).

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

Without the server (no server may be running; ~10 min):

```bash
python models/demos/laguna/demo/perf_direct.py                          # batch 1 and 32, 128 .. 8K tokens
python models/demos/laguna/demo/perf_direct.py --batch 1 --input-lens 128,1024
```

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
