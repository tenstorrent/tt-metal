<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-S-2.1 on a TT-QuietBox 2

[`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1) (117.6B parameters, 8.45B active, 1M-token
context) as an OpenAI-compatible vLLM server on four Blackhole ASICs (a TT-QuietBox 2 or four P150 cards).

## Status (2026-10-09)

Through the vLLM server, batch 1, versus the first version of this branch (2026-10-06), same measurement:

| | First version | Now |
|---|---:|---:|
| Decode | 18 tok/s | 66-71 tok/s (3.7-3.9x) |
| TTFT, 128-token prompt | 0.23 s | 87 ms |
| TTFT, 8K-token prompt | 19.9 s | 0.89 s |
| Batch-32 decode, all 32 users | 11 tok/s/user | 21-23 tok/s/user |

Against the targets (50% of speed of light, measured without vLLM, tables below):

- Met: batch-1 TTFT at 128 tokens (60 ms vs 66 ms) and batch-32 decode at every length (22-28 vs 18-20 tok/s/user).
- Not met: batch-1 decode (72-76 vs 152-158 tok/s), batch-1 TTFT from 1K tokens up, batch-32 TTFT.
- DFlash speculative decoding is not optimized yet and is currently slower than normal decode; it is the next work
  item.

## Results

### Accuracy

Next-token predictions vs the original model in fp32, over an AIME24 prompt plus a fixed 100-token answer (the
128-token prompt is the same question cut short). Batch 32 shows the worst of 32 identical users.
`tests/test_accuracy.py`, 2026-10-09.

| Measure | 235-token (AIME24), batch 1 | 235-token, batch 32 | 128-token (AIME24), batch 1 | 128-token, batch 32 | Bar |
|---|---:|---:|---:|---:|---:|
| top-1 | 0.98 | 0.99 | 0.96 | 0.95 | >= 0.90 |
| top-5 | 1.00 | 1.00 | 1.00 | 1.00 | >= 0.98 |
| top-100 | 1.00 | 1.00 | 1.00 | 1.00 | = 1.00 |
| top-1, traced decode | 0.98 | 0.99 | 0.95 | 0.96 | >= 0.90 |
| PCC of next-token scores, mean | 0.97 | 0.97 | 0.97 | 0.97 | >= 0.95 |

### Performance

Speed of light (SoL) is the roofline limit ([All About Transformer
Inference](https://jax-ml.github.io/scaling-book/inference/), `demo/roofline.py`): decode reads the weights a token
needs plus the KV cache at 2.05 TB/s; TTFT is the larger of reading the weights and doing the FLOPs at 2.88 PFLOP/s.
Target = 50% of SoL. Measured with `demo/perf_direct.py` (no vLLM: device time plus reading the token back), 2026-10-09.

Batch 1:

| Input tokens | Decode SoL (tok/s) | Decode target | Decode measured | TTFT SoL | TTFT target | TTFT measured |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 316 | 158 | 75.9 | 33 ms | 66 ms | 60.5 ms |
| 1,024 | 313 | 157 | 73.6 | 33 ms | 66 ms | 140 ms |
| 2,048 | 312 | 156 | 73.2 | 33 ms | 66 ms | 238 ms |
| 4,096 | 310 | 155 | 72.2 | 33 ms | 66 ms | 386 ms |
| 8,192 | 305 | 152 | 72.3 | 52 ms | 103 ms | 716 ms |

Batch 32 (32 prompts arrive together; measured TTFT is the last user's, the mean in parentheses):

| Input tokens | Decode SoL (tok/s/user) | Decode target | Decode measured | Decode measured, all users (tok/s) | TTFT SoL | TTFT target | TTFT measured |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 41 | 20 | 28.0 | 896 | 33 ms | 66 ms | 0.42 s (0.42 s) |
| 1,024 | 39 | 20 | 26.4 | 845 | 193 ms | 386 ms | 2.89 s (1.81 s) |
| 2,048 | 39 | 19 | 25.8 | 826 | 392 ms | 783 ms | 5.82 s (3.27 s) |
| 4,096 | 37 | 19 | 23.5 | 752 | 799 ms | 1.60 s | 11.6 s (6.19 s) |
| 8,192 | 35 | 18 | 22.4 | 717 | 1.65 s | 3.31 s | 23.4 s (12.1 s) |

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
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python $MODEL_DIR/demo/roofline.py      # SoL and targets, no device
python $MODEL_DIR/demo/perf_demo.py                                           # end to end through vLLM
```

## Serving options

Set in front of `serve_vllm.sh`; experimental ones also need `LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`.

| Feature | Variables | Notes |
|---|---|---|
| DFlash speculative decoding (experimental) | `TT_LAGUNA_DFLASH=1` | Draft model [`poolside/Laguna-S-2.1-DFlash`](https://huggingface.co/poolside/Laguna-S-2.1-DFlash); 1 greedy request at a time; currently slower than normal decode |
| N-gram speculative decoding (experimental) | `TT_LAGUNA_SPEC_DECODE=1` | Guesses by repeating earlier text, no draft model; 1 greedy request at a time; not measured on this version |
| Prefix caching (experimental) | `TT_LAGUNA_PREFIX_CACHE=1` | Reuses a repeated prompt prefix; 1 request at a time |
| Concurrent requests | `LAGUNA_MAX_NUM_SEQS=N`, N up to 32 | All requests share one context pool (below) |

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
