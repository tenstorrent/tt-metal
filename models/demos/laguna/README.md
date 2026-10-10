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

Laguna's next-token predictions vs the original model in fp32, over the AIME24 prompt (235 tokens, or cut to 128)
plus a fixed 100-token answer; batch 32 is the worst of 32 identical users. "top-1, traced" is the same top-1 check
through the traced decode path the server runs (token picked on device). `tests/test_accuracy.py`, 2026-10-09.

| Prompt (AIME24) | Batch | top-1 | top-5 | top-100 | top-1, traced | PCC |
|---|---:|---:|---:|---:|---:|---:|
| 235 tokens | 1 | 0.98 | 1.00 | 1.00 | 0.98 | 0.97 |
| 235 tokens | 32 | 0.99 | 1.00 | 1.00 | 0.99 | 0.97 |
| 128 tokens | 1 | 0.96 | 1.00 | 1.00 | 0.95 | 0.97 |
| 128 tokens | 32 | 0.95 | 1.00 | 1.00 | 0.96 | 0.97 |
| Bar | | >= 0.90 | >= 0.98 | 1.00 | >= 0.90 | >= 0.95 |

### Performance

Target = 50% of speed of light (SoL), the roofline limit from [All About Transformer
Inference](https://jax-ml.github.io/scaling-book/inference/) computed by `demo/roofline.py`. Measured with
`demo/perf_direct.py`: no vLLM, device time plus reading the token back, 2026-10-09 (DFlash 2026-10-10). Batch 32: 32 prompts arrive
together; TTFT is the last user's; total decode throughput is 32x the per-user speed.

Decode, tok/s per user (measured / target):

| Input tokens | Batch 1 | Batch 32 |
|---:|---:|---:|
| 128 | 75.9 / 158 | 28.0 / 20 |
| 1,024 | 73.6 / 157 | 26.4 / 20 |
| 2,048 | 73.2 / 156 | 25.8 / 19 |
| 4,096 | 72.2 / 155 | 23.5 / 19 |
| 8,192 | 72.3 / 152 | 22.4 / 18 |

Time to first token (measured / target):

| Input tokens | Batch 1 | Batch 32 |
|---:|---:|---:|
| 128 | 60.5 ms / 66 ms | 0.42 s / 0.066 s |
| 1,024 | 140 ms / 66 ms | 2.89 s / 0.39 s |
| 2,048 | 238 ms / 66 ms | 5.82 s / 0.78 s |
| 4,096 | 386 ms / 66 ms | 11.6 s / 1.60 s |
| 8,192 | 716 ms / 103 ms | 23.4 s / 3.31 s |

DFlash speculative decoding, batch 1, real-text prompts (a "summarize this document" request; the draft model's
guesses decide the speed):

| Input tokens | Decode, tok/s | Draft tokens accepted per round |
|---:|---:|---:|
| 128 | 20.5 | 1.2 |
| 1,024 | 22.4 | 1.5 |
| 2,048 | 25.2 | 1.8 |
| 4,096 | 25.1 | 1.8 |
| 8,192 | 21.9 | 1.5 |

DFlash is slower than normal decode for now: a round (draft 15 tokens, then check 16 in Laguna) takes ~105 ms and
keeps ~2.5 tokens, while normal decode takes ~13 ms per token.

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
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python $MODEL_DIR/demo/perf_direct.py --modes dflash   # DFlash, ~4 min
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
